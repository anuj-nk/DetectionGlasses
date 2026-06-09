#!/usr/bin/env python3
"""
build_dataset.py

Assembles an Edge Impulse audio dataset from ESC-50, UrbanSound8K, FSD50K,
GTZAN, Speech Commands, and Common Voice — in that order so the cleanest
recordings fill slots first. Produces ei_dataset/{training,testing}/{class}/
{label}.{stem}.{ext} then zips the whole tree to ei_dataset.zip.

Usage: python build_dataset.py
Re-running is safe: existing destination files are never overwritten.
"""

import csv
import hashlib
import random
import shutil
import tarfile
import zipfile
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE    = Path(__file__).parent
DL      = BASE / "downloads"
OUTPUT  = BASE / "ei_dataset"
ZIP_OUT = BASE / "ei_dataset.zip"

# ── Per-class targets ──────────────────────────────────────────────────────────

TARGETS: dict[str, int] = {
    "danger":  1000,
    "alert":   1200,
    "speech":  1500,
    "music":   1500,
    "vehicle": 1400,
}

# ── Quality thresholds ─────────────────────────────────────────────────────────

MIN_WAV_BYTES = 16_384   # 16 KB  ≈ 0.5 s at 16 kHz 16-bit (covers Speech Commands)
MIN_MP3_BYTES = 16_384   # 16 KB  ≈ 1 s at 128 kbps
SKIP_KEYWORDS = frozenset({"test_tone", "silence", "noise_only", "empty"})

# ── ESC-50 category → class ────────────────────────────────────────────────────

ESC50_MAP: dict[str, str] = {
    "clock_alarm":    "danger",
    "glass_breaking": "danger",
    "fireworks":      "danger",
    "church_bells":   "danger",
    "crackling_fire": "danger",
    "thunderstorm":   "danger",
    "siren":          "alert",
    "car_horn":       "alert",
    "engine":         "vehicle",
    "train":          "vehicle",
    "airplane":       "vehicle",
    "crying_baby":    "speech",
    "laughing":       "speech",
    "clapping":       "speech",
    "sneezing":       "speech",
    "coughing":       "speech",
    "snoring":        "speech",
    "breathing":      "speech",
}

# ── UrbanSound8K classID → class ───────────────────────────────────────────────

US8K_MAP: dict[int, str] = {
    1: "alert",    # car_horn
    5: "vehicle",  # engine_idling
    6: "danger",   # gun_shot
    8: "alert",    # siren
    9: "music",    # street_music
}

# ── FSD50K label → class (labels verified against actual FSD50K ground truth CSVs)

FSD50K_MAP: dict[str, str] = {
    # danger
    "Alarm":                              "danger",
    "Buzz":                               "danger",
    "Explosion":                          "danger",
    "Fire":                               "danger",
    "Fireworks":                          "danger",
    "Glass":                              "danger",
    "Shatter":                            "danger",
    "Gunshot_and_gunfire":                "danger",
    "Bell":                               "danger",
    "Church_bell":                        "danger",
    "Boom":                               "danger",
    "Crack":                              "danger",
    "Crackle":                            "danger",
    # alert
    "Siren":                              "alert",
    "Vehicle_horn_and_car_horn_and_honking": "alert",
    "Screaming":                          "alert",
    "Screech":                            "alert",
    "Shout":                              "alert",
    "Yell":                               "alert",
    # speech
    "Speech":                             "speech",
    "Male_speech_and_man_speaking":       "speech",
    "Female_speech_and_woman_speaking":   "speech",
    "Child_speech_and_kid_speaking":      "speech",
    "Conversation":                       "speech",
    "Crowd":                              "speech",
    "Laughter":                           "speech",
    "Cheering":                           "speech",
    "Applause":                           "speech",
    "Crying_and_sobbing":                 "speech",
    "Chatter":                            "speech",
    "Human_voice":                        "speech",
    # music
    "Music":                              "music",
    "Singing":                            "music",
    "Male_singing":                       "music",
    "Female_singing":                     "music",
    "Guitar":                             "music",
    "Acoustic_guitar":                    "music",
    "Electric_guitar":                    "music",
    "Bass_guitar":                        "music",
    "Piano":                              "music",
    "Drum":                               "music",
    "Drum_kit":                           "music",
    "Musical_instrument":                 "music",
    "Brass_instrument":                   "music",
    "Trumpet":                            "music",
    "Organ":                              "music",
    "Accordion":                          "music",
    "Harmonica":                          "music",
    "Harp":                               "music",
    "Bowed_string_instrument":            "music",
    "Plucked_string_instrument":          "music",
    "Wind_instrument_and_woodwind_instrument": "music",
    "Mallet_percussion":                  "music",
    "Marimba_and_xylophone":              "music",
    "Percussion":                         "music",
    "Cymbal":                             "music",
    "Hi-hat":                             "music",
    "Snare_drum":                         "music",
    "Bass_drum":                          "music",
    "Crash_cymbal":                       "music",
    "Glockenspiel":                       "music",
    "Gong":                               "music",
    "Tabla":                              "music",
    "Tambourine":                         "music",
    # vehicle
    "Car":                                "vehicle",
    "Motor_vehicle_(road)":               "vehicle",
    "Truck":                              "vehicle",
    "Bus":                                "vehicle",
    "Motorcycle":                         "vehicle",
    "Train":                              "vehicle",
    "Rail_transport":                     "vehicle",
    "Aircraft":                           "vehicle",
    "Fixed-wing_aircraft_and_airplane":   "vehicle",
    "Boat_and_Water_vehicle":             "vehicle",
    "Subway_and_metro_and_underground":   "vehicle",
    "Engine":                             "vehicle",
    "Engine_starting":                    "vehicle",
    "Idling":                             "vehicle",
    "Vehicle":                            "vehicle",
    "Race_car_and_auto_racing":           "vehicle",
    "Accelerating_and_revving_and_vroom": "vehicle",
    "Car_passing_by":                     "vehicle",
    "Traffic_noise_and_roadway_noise":    "vehicle",
}

# ── Live counters (train + test totals per class) ──────────────────────────────

counts: dict[str, dict[str, int]] = {
    cls: {"total": 0, "train": 0, "test": 0} for cls in TARGETS
}

# ── Core helpers ───────────────────────────────────────────────────────────────

def get_split(filename: str) -> str:
    """Deterministic 80/20 split — stable across re-runs."""
    digest = int(hashlib.md5(filename.encode()).hexdigest(), 16)
    return "training" if digest % 10 < 8 else "testing"


def quality_ok(path: Path) -> bool:
    name = path.name.lower()
    if any(kw in name for kw in SKIP_KEYWORDS):
        return False
    size = path.stat().st_size
    if path.suffix.lower() == ".mp3":
        return size >= MIN_MP3_BYTES
    return size >= MIN_WAV_BYTES


def full(cls: str) -> bool:
    return counts[cls]["total"] >= TARGETS[cls]


def all_full() -> bool:
    return all(full(cls) for cls in TARGETS)


def try_copy(src: Path, cls: str) -> bool:
    """
    Copy src into the output tree under cls.
    Returns True on success, False if skipped (full / quality / already exists).
    """
    if full(cls) or not quality_ok(src):
        return False
    split = get_split(src.name)
    dest_dir = OUTPUT / split / cls
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{cls}.{src.name}"
    if dest.exists():
        return False
    shutil.copy2(src, dest)
    counts[cls]["total"] += 1
    counts[cls]["train" if split == "training" else "test"] += 1
    return True


def scan_existing() -> None:
    """Count files already present from a previous run so targets are accurate."""
    for split in ("training", "testing"):
        key = "train" if split == "training" else "test"
        for cls in TARGETS:
            d = OUTPUT / split / cls
            if not d.is_dir():
                continue
            for f in d.iterdir():
                if f.is_file():
                    counts[cls]["total"] += 1
                    counts[cls][key] += 1


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


# ── Source: ESC-50 ─────────────────────────────────────────────────────────────

def process_esc50() -> None:
    src = DL / "ESC-50-master"
    if not src.is_dir():
        print(f"  [skip] ESC-50 not found: {src}")
        return
    meta = src / "meta" / "esc50.csv"
    if not meta.is_file():
        print(f"  [skip] ESC-50 metadata missing: {meta}")
        return

    rows = read_csv(meta)
    random.shuffle(rows)

    copied = skipped = 0
    for row in rows:
        if all_full():
            break
        cls = ESC50_MAP.get(row.get("category", ""))
        if cls is None or full(cls):
            continue
        f = src / "audio" / row.get("filename", "")
        if not f.is_file():
            skipped += 1
            continue
        if try_copy(f, cls):
            copied += 1
        else:
            skipped += 1

    print(f"  ESC-50:        {copied:5d} copied  {skipped:5d} skipped")


# ── Source: UrbanSound8K ───────────────────────────────────────────────────────

def process_urbansound8k() -> None:
    # Accept UrbanSound8K at downloads/UrbanSound8K/ or at the project root.
    src = DL / "UrbanSound8K"
    if not src.is_dir():
        src = BASE / "UrbanSound8K"
    tar = DL / "UrbanSound8K.tar.gz"

    if not src.is_dir():
        if tar.is_file():
            print(f"  Extracting {tar.name} (this may take a minute) ...")
            with tarfile.open(tar) as tf:
                tf.extractall(DL)
            print("  Extraction complete.")
            src = DL / "UrbanSound8K"
        else:
            print(f"  [skip] UrbanSound8K not found: {src}")
            return

    meta = src / "metadata" / "UrbanSound8K.csv"
    if not meta.is_file():
        print(f"  [skip] UrbanSound8K metadata missing: {meta}")
        return

    rows = read_csv(meta)
    random.shuffle(rows)

    copied = skipped = 0
    for row in rows:
        if all_full():
            break
        try:
            class_id = int(row.get("classID", -1))
        except (ValueError, TypeError):
            continue
        cls = US8K_MAP.get(class_id)
        if cls is None or full(cls):
            continue
        fold  = row.get("fold", "")
        fname = row.get("slice_file_name", "")
        f = src / "audio" / f"fold{fold}" / fname
        if not f.is_file():
            skipped += 1
            continue
        if try_copy(f, cls):
            copied += 1
        else:
            skipped += 1

    print(f"  UrbanSound8K:  {copied:5d} copied  {skipped:5d} skipped")


# ── Source: FSD50K ─────────────────────────────────────────────────────────────

def fsd50k_classify(labels_str: str) -> str | None:
    """Return class for first matching label in comma-separated label string."""
    for label in (lbl.strip() for lbl in labels_str.split(",")):
        cls = FSD50K_MAP.get(label)
        if cls:
            return cls
    return None


def process_fsd50k() -> None:
    # Ground truth and audio dirs may be flat in downloads/ or nested in FSD50K/.
    def find_dir(name: str) -> Path | None:
        for candidate in (DL / name, DL / "FSD50K" / name):
            if candidate.is_dir():
                return candidate
        return None

    def find_csv(name: str) -> Path | None:
        for candidate in (
            DL / "FSD50K.ground_truth" / name,
            DL / "FSD50K" / "FSD50K.ground_truth" / name,
        ):
            if candidate.is_file():
                return candidate
        return None

    subsets = [
        (find_csv("dev.csv"),  find_dir("FSD50K.dev_audio")),
        (find_csv("eval.csv"), find_dir("FSD50K.eval_audio")),
    ]

    if not any(csv_path for csv_path, _ in subsets):
        print(f"  [skip] FSD50K ground truth CSVs not found under {DL}")
        return

    copied = skipped = 0
    for csv_path, audio_dir in subsets:
        if csv_path is None:
            continue
        if audio_dir is None:
            # CSV exists but audio not yet downloaded — still report skip gracefully
            continue

        rows = read_csv(csv_path)
        random.shuffle(rows)

        for row in rows:
            if all_full():
                break
            fname  = row.get("fname", "").strip()
            labels = row.get("labels", "").strip()
            cls = fsd50k_classify(labels)
            if cls is None or full(cls):
                continue
            # Try bare name, then .wav, then .mp3 (for preview downloads)
            f = audio_dir / fname
            if not f.is_file():
                f = audio_dir / f"{fname}.wav"
            if not f.is_file():
                f = audio_dir / f"{fname}.mp3"
            if not f.is_file():
                skipped += 1
                continue
            if try_copy(f, cls):
                copied += 1
            else:
                skipped += 1

    print(f"  FSD50K:        {copied:5d} copied  {skipped:5d} skipped")


# ── Source: Google Speech Commands ────────────────────────────────────────────

# Folder names to skip in Speech Commands — not speech content
_SC_SKIP = {"_background_noise_", "_silence_"}


def process_speech_commands() -> None:
    # Try both the versioned subfolder and the root.
    candidates = [
        DL / "speech_commands" / "v0.02",
        DL / "speech_commands",
    ]
    src = next((p for p in candidates if p.is_dir() and
                any(c.is_dir() for c in p.iterdir())), None)
    if src is None:
        print(f"  [skip] Speech Commands not found under {DL / 'speech_commands'}")
        return

    files = [
        f
        for word_dir in src.iterdir()
        if word_dir.is_dir() and word_dir.name not in _SC_SKIP
        for f in word_dir.iterdir()
        if f.suffix.lower() == ".wav"
    ]
    random.shuffle(files)

    copied = skipped = 0
    for f in files:
        if full("speech"):
            break
        if try_copy(f, "speech"):
            copied += 1
        else:
            skipped += 1

    print(f"  Speech Cmds:   {copied:5d} copied  {skipped:5d} skipped")


# ── Source: GTZAN ──────────────────────────────────────────────────────────────

def process_gtzan() -> None:
    src = DL / "gtzan" / "genres_original"
    if not src.is_dir():
        print(f"  [skip] GTZAN not found: {src}")
        return

    files = [
        f
        for genre_dir in src.iterdir() if genre_dir.is_dir()
        for f in genre_dir.iterdir()
        if f.suffix.lower() == ".wav" and f.name != "jazz.00054.wav"
    ]
    random.shuffle(files)

    copied = skipped = 0
    for f in files:
        if full("music"):
            break
        if try_copy(f, "music"):
            copied += 1
        else:
            skipped += 1

    print(f"  GTZAN:         {copied:5d} copied  {skipped:5d} skipped")


# ── Source: Common Voice ───────────────────────────────────────────────────────

def process_common_voice() -> None:
    src = DL / "common_voice" / "clips"
    if not src.is_dir():
        print(f"  [skip] Common Voice not found: {src}")
        return

    files = [f for f in src.iterdir() if f.is_file() and f.suffix.lower() == ".mp3"]
    random.shuffle(files)

    copied = skipped = 0
    for f in files:
        if full("speech"):
            break
        if try_copy(f, "speech"):
            copied += 1
        else:
            skipped += 1

    print(f"  Common Voice:  {copied:5d} copied  {skipped:5d} skipped")


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary() -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"{'Class':<10} {'Copied':>8} {'Target':>8} {'Fill%':>7} {'Train':>8} {'Test':>7}")
    print("-" * 62)
    for cls, target in TARGETS.items():
        c     = counts[cls]
        total = c["total"]
        pct   = total / target * 100 if target else 0
        warn  = "  ⚠  BELOW 70%" if pct < 70 else ""
        print(f"{cls:<10} {total:>8} {target:>8} {pct:>6.1f}% {c['train']:>8} {c['test']:>7}{warn}")
    grand = sum(c["total"] for c in counts.values())
    print("-" * 62)
    print(f"{'TOTAL':<10} {grand:>8}")
    print(sep)


# ── Zip ────────────────────────────────────────────────────────────────────────

def make_zip() -> None:
    print(f"\nZipping {OUTPUT.name}/ → {ZIP_OUT.name} ...")
    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for f in sorted(OUTPUT.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(BASE))
    mb = ZIP_OUT.stat().st_size / 1_048_576
    print(f"  Done — {ZIP_OUT.name} is {mb:.1f} MB")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print("Scanning existing output ...")
    scan_existing()

    existing = sum(c["total"] for c in counts.values())
    if existing:
        print(f"  Found {existing} existing files — they count toward targets.")

    print("\nProcessing sources:")
    process_esc50()
    process_urbansound8k()
    process_fsd50k()
    process_speech_commands()
    process_gtzan()
    process_common_voice()

    print_summary()
    make_zip()

    print("\nUpload commands:")
    print("  edge-impulse-uploader --category training ./ei_dataset/training/**/*.wav")
    print("  edge-impulse-uploader --category testing  ./ei_dataset/testing/**/*.wav")


if __name__ == "__main__":
    main()
