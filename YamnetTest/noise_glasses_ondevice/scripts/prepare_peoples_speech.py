#!/usr/bin/env python3
"""Export a small People's Speech subset as Edge Impulse-ready speech WAVs.

This intentionally defaults to the tiny `microset` subset so you do not pull a
massive ASR corpus by accident.

Install dependencies:
  python3 -m pip install datasets soundfile

Run:
  python3 scripts/prepare_peoples_speech.py --max-training 200 --max-testing 50
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="microset")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("prepared_datasets/peoples_speech_edge_impulse"),
    )
    parser.add_argument("--max-training", type=int, default=200)
    parser.add_argument("--max-testing", type=int, default=50)
    parser.add_argument("--sample-rate", type=int, default=16000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = args.max_training + args.max_testing

    try:
        import soundfile as sf
        from datasets import Audio, load_dataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency. Install with:\n"
            "  python3 -m pip install datasets soundfile"
        ) from exc

    dataset = load_dataset(
        "MLCommons/peoples_speech",
        args.subset,
        split=f"{args.split}[:{total}]",
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=args.sample_rate))

    training_count = 0
    testing_count = 0

    for index, row in enumerate(dataset):
        category = "testing" if index < args.max_testing else "training"
        if category == "testing":
            testing_count += 1
        else:
            training_count += 1

        audio = row["audio"]
        dst = args.output_dir / category / "speech" / f"peoples_speech_{index:05d}.wav"
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst, audio["array"], args.sample_rate, subtype="PCM_16")

    print(f"Wrote: {args.output_dir.resolve()}")
    print(f"  training/speech: {training_count}")
    print(f"  testing/speech:  {testing_count}")


if __name__ == "__main__":
    main()
