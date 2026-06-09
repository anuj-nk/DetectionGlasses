#!/usr/bin/env python3
"""Upload a class-folder audio dataset to Edge Impulse with the Python SDK.

Expected layout:
  dataset_root/
    training/<label>/*.wav
    testing/<label>/*.wav

Usage:
  EI_API_KEY=ei_... python3 scripts/upload_edge_impulse_dataset.py \
    --dataset-root prepared_datasets/urbansound8k_edge_impulse_balanced
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("prepared_datasets/urbansound8k_edge_impulse_balanced"),
        help="Root containing training/<label> and testing/<label> folders.",
    )
    parser.add_argument(
        "--category",
        choices=["training", "testing", "both"],
        default="both",
        help="Which split to upload.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        help="Optional label allow-list. Defaults to all labels found.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Edge Impulse SDK upload batch size.",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Allow uploading files that Edge Impulse thinks are duplicates.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without contacting Edge Impulse.",
    )
    return parser.parse_args()


def iter_label_dirs(root: Path, category: str, labels: set[str] | None):
    category_dir = root / category
    if not category_dir.exists():
        return
    for label_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
        if labels and label_dir.name not in labels:
            continue
        wav_count = len(list(label_dir.glob("*.wav")))
        if wav_count:
            yield label_dir.name, label_dir, wav_count


def main() -> None:
    args = parse_args()
    labels = set(args.labels) if args.labels else None
    categories = ["training", "testing"] if args.category == "both" else [args.category]

    if not args.dataset_root.exists():
        raise SystemExit(f"Dataset root does not exist: {args.dataset_root}")

    upload_plan = []
    for category in categories:
        upload_plan.extend(iter_label_dirs(args.dataset_root, category, labels))

    if not upload_plan:
        raise SystemExit("No WAV files found to upload.")

    print("Upload plan:")
    for label, label_dir, wav_count in upload_plan:
        category = label_dir.parent.name
        print(f"  {category:8s} {label:22s} {wav_count:5d} files")

    if args.dry_run:
        return

    api_key = os.environ.get("EI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing EI_API_KEY. Create/copy a project API key from "
            "Edge Impulse Dashboard > Keys, then run with EI_API_KEY=ei_..."
        )

    import edgeimpulse as ei

    ei.API_KEY = api_key

    failures = 0
    successes = 0
    for label, label_dir, wav_count in upload_plan:
        category = label_dir.parent.name
        print(f"\nUploading {wav_count} files: {category}/{label}")
        response = ei.experimental.data.upload_directory(
            directory=str(label_dir),
            category=category,
            label=label,
            metadata={
                "source_dataset": "UrbanSound8K",
                "prepared_by": "noise_glasses_ondevice",
            },
            allow_duplicates=args.allow_duplicates,
            show_progress=True,
            batch_size=args.batch_size,
        )
        successes += len(response.successes)
        failures += len(response.fails)
        print(f"  successes={len(response.successes)} fails={len(response.fails)}")
        for fail in response.fails[:5]:
            print(f"  fail: {fail}")

    print(f"\nDone. Uploaded {successes} samples; {failures} failures.")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
