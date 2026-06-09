#!/usr/bin/env python3
"""
fetch_fsd50k_danger.py

Downloads danger-labeled clips from Freesound using the FSD50K ground truth CSVs.
Uses the Freesound API preview URLs (HQ 128kbps MP3) — no OAuth2 required.

Usage:
    python fetch_fsd50k_danger.py --token YOUR_API_TOKEN
    python fetch_fsd50k_danger.py --token YOUR_API_TOKEN --limit 500

After running, re-run build_dataset.py to incorporate the new clips.
"""

import argparse
import csv
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE   = Path(__file__).parent
DL     = BASE / "downloads"
GT_DIR = DL / "FSD50K.ground_truth"

# Labels that map to danger, in priority order (most safety-critical first).
# "Alarm" covers fire alarms, smoke detectors, CO alarms — it's the big one.
DANGER_PRIORITY = [
    "Alarm",
    "Fire",
    "Buzz",
    "Explosion",
    "Shatter",
    "Glass",
    "Fireworks",
    "Crack",
    "Crackle",
    "Boom",
    "Gunshot_and_gunfire",
    "Bell",
    "Church_bell",
]
DANGER_SET = set(DANGER_PRIORITY)

# Audio directories mirror where build_dataset.py will look.
OUT_DIRS = {
    "dev":  DL / "FSD50K.dev_audio",
    "eval": DL / "FSD50K.eval_audio",
}


def collect_candidates() -> list[tuple[str, str, str]]:
    """
    Return [(fid, best_label, source), ...] sorted by DANGER_PRIORITY.
    source is 'dev' or 'eval'.
    """
    sources = [
        (GT_DIR / "dev.csv",  "dev"),
        (GT_DIR / "eval.csv", "eval"),
    ]
    candidates: list[tuple[str, str, str]] = []
    for csv_path, source in sources:
        if not csv_path.is_file():
            print(f"[skip] {csv_path} not found")
            continue
        with csv_path.open(encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                fid    = row["fname"].strip()
                labels = {lbl.strip() for lbl in row["labels"].split(",")}
                best   = next((l for l in DANGER_PRIORITY if l in labels), None)
                if best:
                    candidates.append((fid, best, source))

    # Shuffle within each priority bucket for variety, then sort by priority.
    random.shuffle(candidates)
    priority_idx = {l: i for i, l in enumerate(DANGER_PRIORITY)}
    candidates.sort(key=lambda x: priority_idx[x[1]])
    return candidates


def already_downloaded(fid: str, out_dir: Path) -> bool:
    return (out_dir / f"{fid}.mp3").is_file() or (out_dir / f"{fid}.wav").is_file()


def fetch_preview_url(fid: str, token: str) -> str | None:
    url = f"https://freesound.org/apiv2/sounds/{fid}/?token={token}&fields=name,previews,duration"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
        return data.get("previews", {}).get("preview-hq-mp3")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None  # clip was removed from Freesound
        raise


def download_mp3(url: str, dest: Path) -> bool:
    try:
        urllib.request.urlretrieve(url, dest)
        return dest.stat().st_size >= 16_384  # 16 KB minimum
    except Exception:
        dest.unlink(missing_ok=True)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FSD50K danger clips via Freesound API")
    parser.add_argument("--token", required=True, help="Freesound API token (from freesound.org/apiv2/apply)")
    parser.add_argument("--limit", type=int, default=450, help="Max clips to download (default 450)")
    args = parser.parse_args()

    for d in OUT_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    candidates = collect_candidates()
    print(f"Found {len(candidates)} danger-labeled clips in ground truth")
    print(f"Downloading up to {args.limit} (prioritising Alarm → Fire → Buzz → ...)\n")

    downloaded = skipped = failed = 0
    for i, (fid, label, source) in enumerate(candidates):
        if downloaded >= args.limit:
            break

        out_dir = OUT_DIRS[source]
        if already_downloaded(fid, out_dir):
            skipped += 1
            continue

        tag   = f"[{downloaded + 1}/{args.limit}]"
        print(f"  {tag} {fid}  {label}", end="  ", flush=True)

        try:
            preview_url = fetch_preview_url(fid, args.token)
            if not preview_url:
                print("no preview / clip removed")
                failed += 1
                time.sleep(0.3)
                continue

            dest = out_dir / f"{fid}.mp3"
            if download_mp3(preview_url, dest):
                print("✓")
                downloaded += 1
            else:
                print("too small, skipped")
                failed += 1

        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code}")
            failed += 1
        except Exception as e:
            print(f"error: {e}")
            failed += 1

        time.sleep(0.5)  # ≤ 60 API req/min

    print(f"\n{'=' * 50}")
    print(f"Downloaded:      {downloaded}")
    print(f"Already existed: {skipped}")
    print(f"Failed/skipped:  {failed}")
    print(f"\nNow run:  python build_dataset.py")


if __name__ == "__main__":
    main()
