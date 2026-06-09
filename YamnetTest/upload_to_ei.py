#!/usr/bin/env python3
"""
upload_to_ei.py

Uploads ei_dataset/ to an Edge Impulse project via the Ingestion REST API.
Uses a thread pool for concurrent uploads. Skips files already in upload_log.txt.

Usage:
    python upload_to_ei.py --api-key ei_xxx --project-id 993132
    python upload_to_ei.py --api-key ei_xxx --project-id 993132 --workers 8
"""

import argparse
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE     = Path(__file__).parent
DATASET  = BASE / "ei_dataset"
LOG_FILE = BASE / "upload_log.txt"
EI_BASE  = "https://ingestion.edgeimpulse.com"
CLASSES  = {"danger", "alert", "speech", "music", "vehicle"}
SPLITS   = {"training", "testing"}
CONTENT_TYPES = {".wav": "audio/wav", ".mp3": "audio/mpeg"}

_log_lock    = threading.Lock()
_print_lock  = threading.Lock()
_counter_lock = threading.Lock()
_ok = _failed = 0


def load_log() -> set[str]:
    if not LOG_FILE.exists():
        return set()
    return set(LOG_FILE.read_text().splitlines())


def append_log(entry: str) -> None:
    with _log_lock:
        with LOG_FILE.open("a") as f:
            f.write(entry + "\n")


def make_multipart(filename: str, data: bytes, content_type: str) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="data"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def upload_one(path: Path, label: str, category: str,
               api_key: str, project_id: str) -> tuple[str, bool, str]:
    """Upload a single file. Returns (rel_path, success, error_msg)."""
    rel = str(path.relative_to(BASE))
    url = f"{EI_BASE}/api/{category}/files"
    ct  = CONTENT_TYPES.get(path.suffix.lower(), "audio/wav")

    data = path.read_bytes()
    body, ct_header = make_multipart(path.name, data, ct)

    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("x-api-key", api_key)
    req.add_header("x-label", label)
    req.add_header("x-project-id", project_id)
    req.add_header("Content-Type", ct_header)

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status in (200, 201):
                    return rel, True, ""
                return rel, False, f"HTTP {resp.status}"
        except urllib.error.HTTPError as e:
            return rel, False, f"HTTP {e.code}"
        except Exception as e:
            if attempt < 2:
                time.sleep(1.0)
            else:
                return rel, False, str(e)
    return rel, False, "max retries"


def collect_files() -> list[tuple[Path, str, str]]:
    files = []
    for split in SPLITS:
        category = "training" if split == "training" else "testing"
        for cls in sorted(CLASSES):
            d = DATASET / split / cls
            if not d.is_dir():
                continue
            for f in sorted(d.iterdir()):
                if f.is_file() and f.suffix.lower() in CONTENT_TYPES:
                    files.append((f, cls, category))
    return files


def main() -> None:
    global _ok, _failed
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key",    required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--workers",    type=int, default=16,
                        help="Concurrent upload threads (default 16)")
    args = parser.parse_args()

    files       = collect_files()
    uploaded    = load_log()
    pending     = [(p, lbl, cat) for p, lbl, cat in files
                   if str(p.relative_to(BASE)) not in uploaded]

    print(f"Dataset: {len(files)} files  |  already uploaded: {len(files)-len(pending)}  |  to upload: {len(pending)}")
    if not pending:
        print("Nothing to do.")
        return

    start    = time.time()
    done     = 0
    total    = len(pending)
    failures = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(upload_one, p, lbl, cat, args.api_key, args.project_id): (p, lbl, cat)
            for p, lbl, cat in pending
        }
        for fut in as_completed(futures):
            rel, success, err = fut.result()
            done += 1
            elapsed = time.time() - start
            rate    = done / elapsed
            eta     = (total - done) / rate if rate > 0 else 0

            if success:
                _ok += 1
                append_log(rel)
            else:
                _failed += 1
                failures.append((rel, err))

            with _print_lock:
                print(
                    f"\r  {done}/{total}  "
                    f"ok={_ok}  fail={_failed}  "
                    f"{rate:.1f} files/s  "
                    f"ETA {int(eta//60)}m{int(eta%60):02d}s  ",
                    end="", flush=True,
                )

    elapsed = time.time() - start
    print(f"\n\n{'=' * 55}")
    print(f"Uploaded:  {_ok}")
    print(f"Failed:    {_failed}")
    print(f"Time:      {int(elapsed//60)}m {int(elapsed%60)}s")
    if failures:
        print(f"\nFailed files:")
        for rel, err in failures[:20]:
            print(f"  {err}  {rel}")
        print("Re-run to retry.")


if __name__ == "__main__":
    main()
