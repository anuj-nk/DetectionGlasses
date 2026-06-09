#!/usr/bin/env python3
"""Prepare Google Speech Commands as Edge Impulse-ready speech/background data.

Download first:
  curl -L -o ../downloads/speech_commands/speech_commands_v0.02.tar.gz \
    https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
  mkdir -p ../downloads/speech_commands/v0.02
  tar -xzf ../downloads/speech_commands/speech_commands_v0.02.tar.gz \
    -C ../downloads/speech_commands/v0.02

Then run:
  python3 scripts/prepare_speech_commands.py
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


DEFAULT_WORDS = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "backward", "forward", "follow", "learn",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("../downloads/speech_commands/v0.02"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("prepared_datasets/speech_commands_edge_impulse"),
    )
    parser.add_argument("--max-training", type=int, default=1500)
    parser.add_argument("--max-testing", type=int, default=300)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=515)
    parser.add_argument("--copy", action="store_true")
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
    if not args.source_dir.exists():
        raise SystemExit(f"Missing Speech Commands directory: {args.source_dir}")

    rng = random.Random(args.seed)
    speech_files: list[Path] = []
    for word in DEFAULT_WORDS:
        word_dir = args.source_dir / word
        if word_dir.exists():
            speech_files.extend(sorted(word_dir.glob("*.wav")))

    rng.shuffle(speech_files)
    total = min(len(speech_files), args.max_training + args.max_testing)
    speech_files = speech_files[:total]
    test_count = min(args.max_testing, int(total * args.test_ratio))
    testing = speech_files[:test_count]
    training = speech_files[test_count : test_count + args.max_training]

    for src in training:
        dst = args.output_dir / "training" / "speech" / f"{src.parent.name}_{src.name}"
        link_or_copy(src, dst, args.copy)
    for src in testing:
        dst = args.output_dir / "testing" / "speech" / f"{src.parent.name}_{src.name}"
        link_or_copy(src, dst, args.copy)

    background_dir = args.source_dir / "_background_noise_"
    if background_dir.exists():
        for src in sorted(background_dir.glob("*.wav")):
            dst = args.output_dir / "training" / "silence_background" / src.name
            link_or_copy(src, dst, args.copy)

    print(f"Wrote: {args.output_dir.resolve()}")
    print(f"  training/speech: {len(training)}")
    print(f"  testing/speech:  {len(testing)}")
    print(f"  training/silence_background: {len(list((args.output_dir / 'training' / 'silence_background').glob('*.wav'))) if background_dir.exists() else 0}")


if __name__ == "__main__":
    main()
