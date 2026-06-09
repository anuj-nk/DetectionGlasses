#!/usr/bin/env python3
"""Prepare AudioSet CSV segments as Edge Impulse-ready WAV folders.

AudioSet's official download gives metadata CSVs, not WAV files. This script:
  1. Reads class_labels_indices.csv to map display names to AudioSet MIDs.
  2. Reads an AudioSet segments CSV, such as balanced_train_segments.csv.
  3. Filters rows containing any requested display name.
  4. Uses yt-dlp + ffmpeg to download each YouTube segment as 16 kHz mono WAV.

Install runtime tools first:
  brew install ffmpeg
  python3 -m pip install yt-dlp

Example:
  python3 scripts/prepare_audioset_wavs.py \
    --segments-csv downloads/audioset/balanced_train_segments.csv \
    --class-map-csv downloads/audioset/class_labels_indices.csv \
    --output-dir prepared_datasets/audioset_speech/training/speech \
    --include "Speech" \
    --include "Conversation" \
    --max-clips 300
"""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments-csv", type=Path, required=True)
    parser.add_argument("--class-map-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--include",
        action="append",
        required=True,
        help="AudioSet display_name to include. Repeat this flag.",
    )
    parser.add_argument("--max-clips", type=int, default=300)
    parser.add_argument("--seed", type=int, default=515)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_mids(class_map_csv: Path, display_names: list[str]) -> set[str]:
    wanted = {name.casefold() for name in display_names}
    mids: set[str] = set()
    found: set[str] = set()
    with class_map_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["display_name"].casefold() in wanted:
                mids.add(row["mid"])
                found.add(row["display_name"].casefold())
    missing = sorted(wanted - found)
    if missing:
        print(f"Warning: display names not found in class map: {missing}")
    if not mids:
        raise SystemExit(f"No matching AudioSet labels found for: {display_names}")
    print("Using AudioSet MIDs:")
    with class_map_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["mid"] in mids:
                print(f"  {row['mid']:12s} {row['display_name']}")
    return mids


def iter_segments(segments_csv: Path):
    with segments_csv.open(newline="") as f:
        reader = csv.reader(line for line in f if not line.startswith("#"))
        for row in reader:
            if len(row) < 4:
                continue
            ytid = row[0].strip()
            start = float(row[1])
            end = float(row[2])
            labels = [label.strip().strip('"') for label in row[3].split(",")]
            yield ytid, start, end, set(labels)


def download_segment(ytid: str, start: float, end: float, dst: Path) -> bool:
    url = f"https://www.youtube.com/watch?v={ytid}"
    duration = max(0.1, end - start)
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Use yt-dlp to hand the best audio URL to ffmpeg. ffmpeg does exact trimming
    # and converts to the Edge Impulse-friendly format in one step.
    try:
        media_url = subprocess.check_output(
            ["yt-dlp", "-f", "bestaudio/best", "-g", url],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()[0]
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start:.3f}",
                "-i",
                media_url,
                "-t",
                f"{duration:.3f}",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-y",
                str(dst),
            ],
            check=True,
        )
        return True
    except Exception as exc:
        print(f"  skip {ytid} {start:.1f}-{end:.1f}: {exc}")
        return False


def main() -> None:
    args = parse_args()
    mids = load_mids(args.class_map_csv, args.include)

    matches = [
        (ytid, start, end)
        for ytid, start, end, labels in iter_segments(args.segments_csv)
        if labels & mids
    ]
    random.Random(args.seed).shuffle(matches)
    matches = matches[: args.max_clips]

    print(f"Matched {len(matches)} candidate clips")
    if args.dry_run:
        for ytid, start, end in matches[:20]:
            print(f"  {ytid} {start:.3f}-{end:.3f}")
        return

    ok = 0
    for index, (ytid, start, end) in enumerate(matches, start=1):
        dst = args.output_dir / f"audioset_{ytid}_{int(start * 1000)}_{int(end * 1000)}.wav"
        if dst.exists():
            ok += 1
            continue
        print(f"[{index}/{len(matches)}] {ytid} {start:.1f}-{end:.1f}")
        if download_segment(ytid, start, end, dst):
            ok += 1

    print(f"Downloaded {ok}/{len(matches)} clips to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
