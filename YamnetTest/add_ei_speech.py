#!/usr/bin/env python3
"""
add_ei_speech.py

Uploads additional speech clips from Google Speech Commands v0.02 to an
Edge Impulse project until the speech class duration matches the average
of the other classes.

Skips any clip whose stem already appears in upload_log.txt to avoid
duplicates.

Usage:
    python add_ei_speech.py --api-key ei_xxx --project-id 993132
    python add_ei_speech.py --api-key ei_xxx --project-id 993132 --dry-run
"""

import argparse
import json
import random
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE       = Path(__file__).parent
DL         = BASE / "downloads"
LOG_FILE   = BASE / "upload_log.txt"
EI_STUDIO  = "https://studio.edgeimpulse.com/v1/api"
EI_INGEST  = "https://ingestion.edgeimpulse.com"
PAGE_SIZE  = 1000
CLIP_MS    = 1_000   # Speech Commands clips are ~1 second

SPEECH_CMD_CANDIDATES = [
    DL / "speech_commands" / "v0.02",
    DL / "speech_commands",
]
SKIP_DIRS = {"_background_noise_", "_silence_"}

_log_lock   = threading.Lock()
_print_lock = threading.Lock()


# ── Edge Impulse helpers ───────────────────────────────────────────────────────

def ei_get(path: str, api_key: str) -> dict:
    req = urllib.request.Request(f"{EI_STUDIO}{path}")
    req.add_header("x-api-key", api_key)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def fetch_durations(project_id: str, api_key: str) -> dict[str, int]:
    """Return {label: totalLengthMs} for all labels in the project."""
    durations: dict[str, int] = {}
    for category in ("training", "testing"):
        offset = 0
        while True:
            data = ei_get(
                f"/{project_id}/raw-data?category={category}&limit={PAGE_SIZE}&offset={offset}",
                api_key,
            )
            batch = data.get("samples", [])
            if not batch:
                break
            for s in batch:
                lbl = s.get("label", "unknown")
                durations[lbl] = durations.get(lbl, 0) + s.get("totalLengthMs", 0)
            if len(batch) < PAGE_SIZE:
                break
            offset += PAGE_SIZE
    return durations


def fmt(ms) -> str:
    h, rem = divmod(int(ms) // 1000, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


# ── Upload helpers ─────────────────────────────────────────────────────────────

def make_multipart(filename: str, data: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="data"; filename="{filename}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def upload_one(path: Path, api_key: str, project_id: str,
               split: str) -> tuple[Path, bool, str]:
    category = "training" if split == "training" else "testing"
    url = f"{EI_INGEST}/api/{category}/files"
    body, ct_header = make_multipart(path.name, path.read_bytes())

    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("x-api-key",    api_key)
    req.add_header("x-label",      "speech")
    req.add_header("x-project-id", project_id)
    req.add_header("Content-Type", ct_header)

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status in (200, 201):
                    return path, True, ""
                return path, False, f"HTTP {resp.status}"
        except urllib.error.HTTPError as e:
            return path, False, f"HTTP {e.code}"
        except Exception as e:
            if attempt < 2:
                time.sleep(1.0)
            else:
                return path, False, str(e)
    return path, False, "max retries"


def append_log(entry: str) -> None:
    with _log_lock:
        with LOG_FILE.open("a") as f:
            f.write(entry + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key",    required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--workers",    type=int, default=12)
    parser.add_argument("--dry-run",    action="store_true")
    args = parser.parse_args()

    # Find Speech Commands root
    sc_root = next((p for p in SPEECH_CMD_CANDIDATES
                    if p.is_dir() and any(c.is_dir() for c in p.iterdir()
                                          if c.name not in SKIP_DIRS)), None)
    if sc_root is None:
        print("ERROR: Speech Commands dataset not found in downloads/")
        return
    print(f"Speech Commands root: {sc_root}")

    # Load already-uploaded stems from log (anything with 'speech' in path)
    uploaded_stems: set[str] = set()
    if LOG_FILE.exists():
        for line in LOG_FILE.read_text().splitlines():
            if "speech" in line.lower():
                uploaded_stems.add(Path(line).stem.lower())
    print(f"Already-uploaded speech stems in log: {len(uploaded_stems)}")

    # Collect candidate WAV files not yet uploaded
    candidates: list[Path] = []
    for word_dir in sorted(sc_root.iterdir()):
        if not word_dir.is_dir() or word_dir.name in SKIP_DIRS:
            continue
        for wav in word_dir.glob("*.wav"):
            if wav.stat().st_size < 16_384:
                continue
            # log stems look like "speech.{original_stem}" after build_dataset copies them
            if wav.stem.lower() not in uploaded_stems and \
               f"speech.{wav.stem}".lower() not in uploaded_stems:
                candidates.append(wav)

    print(f"New speech clips available locally: {len(candidates):,}")
    if not candidates:
        print("Nothing new to upload.")
        return

    # Fetch current EI durations
    print("\nFetching current EI dataset durations…")
    durations = fetch_durations(args.project_id, args.api_key)
    print("Current durations:")
    for lbl in sorted(durations):
        print(f"  {lbl:10s}  {fmt(durations[lbl])}")

    speech_ms = durations.get("speech", 0)
    # Exclude music from target — it's being trimmed separately and skews the avg
    other_labels = [l for l in durations if l not in ("speech", "music")]
    if not other_labels:
        print("No other classes found — can't compute target.")
        return
    target_ms = sum(durations[l] for l in other_labels) // len(other_labels)

    print(f"\nTarget (avg of {', '.join(other_labels)}): {fmt(target_ms)}")
    print(f"Speech current:                            {fmt(speech_ms)}")

    if speech_ms >= target_ms:
        print("\nSpeech is already at or above target — nothing to do.")
        return

    needed_ms    = target_ms - speech_ms
    needed_clips = max(1, int(needed_ms // CLIP_MS))
    print(f"Need ~{needed_clips:,} more clips ({fmt(needed_ms)})\n")

    random.shuffle(candidates)
    to_upload = candidates[:needed_clips]

    if args.dry_run:
        print(f"[dry-run] Would upload {len(to_upload)} clips.")
        for p in to_upload[:10]:
            print(f"  {p}")
        if len(to_upload) > 10:
            print(f"  … and {len(to_upload)-10} more")
        print("\nRe-run without --dry-run to upload.")
        return

    # Assign deterministic train/test split (80/20 by filename hash)
    import hashlib
    def split_for(p: Path) -> str:
        return "training" if int(hashlib.md5(p.name.encode()).hexdigest(), 16) % 10 < 8 else "testing"

    print(f"Uploading {len(to_upload):,} clips with {args.workers} workers…")
    start = time.time()
    ok = failed = done = 0
    total = len(to_upload)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(upload_one, p, args.api_key, args.project_id, split_for(p)): p
            for p in to_upload
        }
        for fut in as_completed(futures):
            path, success, err = fut.result()
            done += 1
            elapsed = time.time() - start
            rate    = done / elapsed
            eta     = (total - done) / rate if rate > 0 else 0

            if success:
                ok += 1
                append_log(str(path.relative_to(BASE)))
            else:
                failed += 1

            with _print_lock:
                print(
                    f"\r  {done}/{total}  ok={ok}  fail={failed}  "
                    f"{rate:.1f}/s  ETA {int(eta//60)}m{int(eta%60):02d}s  ",
                    end="", flush=True,
                )

    elapsed = time.time() - start
    print(f"\n\nDone in {int(elapsed//60)}m {int(elapsed%60)}s")
    print(f"  Uploaded: {ok}   Failed: {failed}")
    print(f"  Speech duration now ≈ {fmt(speech_ms + ok * CLIP_MS)}")
    if failed:
        print("Re-run to retry failed files.")


if __name__ == "__main__":
    main()
