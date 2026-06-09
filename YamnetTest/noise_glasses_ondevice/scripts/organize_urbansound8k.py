#!/usr/bin/env python3
"""Organize UrbanSound8K into Edge Impulse-style label folders.

Default output:
  prepared_datasets/urbansound8k_edge_impulse/
    training/<label>/*.wav
    testing/<label>/*.wav

UrbanSound8K's original folds are preserved by putting fold 10 into testing and
all other folds into training. Edge Impulse can create its own validation split
from the training upload.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path


DEFAULT_LABEL_MAP = {
    "siren": "alarm_or_siren",
    "dog_bark": "dog_or_animal",
    "street_music": "music",
    "engine_idling": "vehicle",
    "car_horn": "vehicle",
    "air_conditioner": "appliance_or_machine",
    "drilling": "appliance_or_machine",
    "jackhammer": "appliance_or_machine",
    # children_playing mixes speech, playground noise, and background activity.
    # It is useful as a negative class, not as clean speech.
    "children_playing": "unknown_noise",
    # UrbanSound8K has gun_shot, but this project label set has no danger class.
    # Keep it out by default so alarm_or_siren does not learn the wrong sound.
    "gun_shot": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--urbansound-dir",
        type=Path,
        default=Path("../UrbanSound8K"),
        help="Path to the UrbanSound8K directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("prepared_datasets/urbansound8k_edge_impulse"),
        help="Directory to write organized audio into.",
    )
    parser.add_argument(
        "--test-fold",
        type=int,
        default=10,
        help="UrbanSound8K fold number to reserve for testing.",
    )
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=0,
        help="Optional cap per output label per split. 0 means no cap.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of symlinking them.",
    )
    return parser.parse_args()


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def main() -> None:
    args = parse_args()
    metadata_path = args.urbansound_dir / "metadata" / "UrbanSound8K.csv"
    audio_dir = args.urbansound_dir / "audio"

    if not metadata_path.exists():
        raise SystemExit(f"Missing metadata CSV: {metadata_path}")
    if not audio_dir.exists():
        raise SystemExit(f"Missing audio directory: {audio_dir}")

    counts: Counter[tuple[str, str]] = Counter()
    skipped: Counter[str] = Counter()

    with metadata_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src_label = row["class"]
            dst_label = DEFAULT_LABEL_MAP.get(src_label)
            if dst_label is None:
                skipped[src_label] += 1
                continue

            fold = int(row["fold"])
            split = "testing" if fold == args.test_fold else "training"
            key = (split, dst_label)
            if args.max_per_label and counts[key] >= args.max_per_label:
                skipped[f"{src_label} capped"] += 1
                continue

            filename = row["slice_file_name"]
            src = audio_dir / f"fold{fold}" / filename
            if not src.exists():
                skipped[f"{src_label} missing"] += 1
                continue

            dst = args.output_dir / split / dst_label / filename
            link_or_copy(src, dst, args.copy)
            counts[key] += 1

    print(f"Wrote dataset to: {args.output_dir.resolve()}")
    print("\nIncluded:")
    for (split, label), count in sorted(counts.items()):
        print(f"  {split:8s} {label:22s} {count:4d}")

    print("\nSkipped:")
    for label, count in sorted(skipped.items()):
        print(f"  {label:22s} {count:4d}")


if __name__ == "__main__":
    main()
