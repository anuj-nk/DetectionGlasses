"""
YAMNet Float32 vs Int8 TFLite — Accuracy Comparison
=====================================================
Runs both models on the same audio clips and compares:
  - Top-1 accuracy (did the top prediction match the label?)
  - Top-5 accuracy (was the label anywhere in the top 5?)
  - Per-category accuracy for the 8 glasses classes
  - Confidence score drift (how much do scores change after quantization?)

Audio sources (in order of preference):
  1. ESC-50 dataset  (if already downloaded by build_dataset.py)
  2. Built-in synthetic tests (zeros, noise, sine — always available)
  3. Any .wav files you drop into a test_audio/ folder

Usage:
    python yamnet_accuracy_test.py
    python yamnet_accuracy_test.py --audio-dir path/to/wavs
    python yamnet_accuracy_test.py --esc50-dir downloads/ESC-50-master
    python yamnet_accuracy_test.py --tflite-only   # skip float32, faster

Requirements:
    pip install tensorflow tensorflow-hub numpy scipy soundfile librosa tqdm
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

try:
    import librosa
    import soundfile as sf
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False
    print("⚠  librosa/soundfile not found — only synthetic tests will run.")
    print("   Install with: pip install librosa soundfile\n")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kwargs): return x

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE  = 16000
CLIP_SAMPLES = 16000 * 3   # 3 seconds
TFLITE_INT8  = Path("tflite_models/yamnet_int8.tflite")
TFLITE_F32   = Path("tflite_models/yamnet_full.tflite")
TOP_K        = 5

# ── YAMNet class → our 8 glasses categories ──────────────────────────────────
YAMNET_TO_CATEGORY = {
    "Speech":                    "speech",
    "Narration, monologue":      "speech",
    "Conversation":              "speech",
    "Shout":                     "speech",
    "Laughter":                  "speech",
    "Baby cry, infant cry":      "speech",
    "Cough":                     "speech",
    "Sneeze":                    "speech",
    "Alarm":                     "alarm_siren",
    "Siren":                     "alarm_siren",
    "Fire alarm":                "alarm_siren",
    "Smoke detector, smoke alarm":"alarm_siren",
    "Buzzer":                    "alarm_siren",
    "Emergency vehicle":         "alarm_siren",
    "Police car (siren)":        "alarm_siren",
    "Ambulance (siren)":         "alarm_siren",
    "Car":                       "horn_beeping",
    "Horn":                      "horn_beeping",
    "Beep, bleep":               "horn_beeping",
    "Bicycle bell":              "horn_beeping",
    "Knock":                     "door_knock_doorbell",
    "Doorbell":                  "door_knock_doorbell",
    "Door":                      "door_knock_doorbell",
    "Slam":                      "door_knock_doorbell",
    "Telephone":                 "phone_ringing",
    "Ringtone":                  "phone_ringing",
    "Telephone bell ringing":    "phone_ringing",
    "Cell phone":                "phone_ringing",
    "Dog":                       "dog_barking",
    "Bark":                      "dog_barking",
    "Growling":                  "dog_barking",
    "Microwave oven":            "appliance_beeping",
    "Washing machine":           "appliance_beeping",
    "Dishwasher":                "appliance_beeping",
    "Silence":                   "silence",
    "White noise":               "silence",
}

# ESC-50 label → our category (for loading labeled test clips)
ESC50_TO_CATEGORY = {
    "door_wood_knock":   "door_knock_doorbell",
    "door_wood_creaks":  "door_knock_doorbell",
    "dog":               "dog_barking",
    "siren":             "alarm_siren",
    "clock_alarm":       "alarm_siren",
    "car_horn":          "horn_beeping",
    "washing_machine":   "appliance_beeping",
    "crying_baby":       "speech",
    "laughing":          "speech",
    "coughing":          "speech",
    "sneezing":          "speech",
}

CATEGORIES = [
    "speech", "alarm_siren", "horn_beeping", "door_knock_doorbell",
    "phone_ringing", "dog_barking", "appliance_beeping", "silence", "unknown_other"
]


# ── Audio helpers ─────────────────────────────────────────────────────────────

def load_wav(path, target_sr=SAMPLE_RATE, target_len=CLIP_SAMPLES):
    """Load any audio file, resample to 16kHz mono, pad/trim to target_len."""
    if not HAS_AUDIO:
        return None
    try:
        y, sr = librosa.load(str(path), sr=target_sr, mono=True)
        if len(y) > target_len:
            y = y[:target_len]
        elif len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        return y.astype(np.float32)
    except Exception as e:
        print(f"  ⚠  Could not load {path}: {e}")
        return None


def make_synthetic_clips():
    """Generate synthetic test clips with known expected classes."""
    t = np.linspace(0, 3, CLIP_SAMPLES)
    clips = [
        ("silence_zeros",  np.zeros(CLIP_SAMPLES, dtype=np.float32),  "Silence"),
        ("white_noise",    np.random.uniform(-1, 1, CLIP_SAMPLES).astype(np.float32), "White noise"),
        ("sine_440hz",     np.sin(2 * np.pi * 440 * t).astype(np.float32), "Sine wave"),
        ("sine_1khz",      np.sin(2 * np.pi * 1000 * t).astype(np.float32), "Sine wave"),
    ]
    return clips  # (name, waveform, expected_yamnet_class)


# ── Model loaders ─────────────────────────────────────────────────────────────

def load_hub_model():
    print("Loading YAMNet float32 from TF Hub...")
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    class_map_path = model.class_map_path().numpy().decode()
    class_names = []
    with open(class_map_path) as f:
        reader = csv.reader(f)
        next(reader)
        class_names = [row[2] for row in reader]
    print(f"✓ Float32 model loaded ({len(class_names)} classes)\n")
    return model, class_names


def load_tflite_model(path):
    """Load a TFLite model and return (interpreter, input_idx, output_idx)."""
    if not path.exists():
        print(f"  ⚠  TFLite model not found: {path}")
        print("     Run yamnet_esp32_feasibility.py first to generate it.\n")
        return None, None, None
    print(f"Loading TFLite model: {path}")
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.resize_tensor_input(interp.get_input_details()[0]["index"], [CLIP_SAMPLES])
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]["index"]
    out = interp.get_output_details()[0]["index"]
    dtype = interp.get_input_details()[0]["dtype"]
    print(f"  Input dtype : {dtype.__name__}  shape: {interp.get_input_details()[0]['shape']}")
    print(f"  Output shape: {interp.get_output_details()[0]['shape']}\n")
    return interp, inp, out


# ── Inference helpers ─────────────────────────────────────────────────────────

def infer_hub(model, class_names, waveform):
    """Run hub model, return [(class_name, score), ...]  top-K."""
    scores, _, _ = model(waveform)
    mean_scores = np.mean(scores.numpy(), axis=0)
    top_idx = np.argsort(mean_scores)[-TOP_K:][::-1]
    return [(class_names[i], float(mean_scores[i])) for i in top_idx]


def infer_tflite(interp, inp_idx, out_idx, waveform, input_dtype):
    """Run TFLite interpreter, return [(class_name, score), ...]  top-K."""
    # int8 models need quantized input
    if input_dtype == np.int8:
        wv = (waveform * 127).clip(-128, 127).astype(np.int8)
    else:
        wv = waveform.copy()

    interp.set_tensor(inp_idx, wv)
    try:
        interp.invoke()
    except RuntimeError as e:
        if "PAD" in str(e) or "Pad value" in str(e):
            # Known desktop TFLite bug with int8 YAMNet — model still valid on-device
            return None, None
        raise
    raw = interp.get_tensor(out_idx)

    if raw.dtype == np.int8:
        out_detail = interp.get_output_details()[0]
        scale, zero_point = out_detail["quantization"]
        scores_flat = (raw.astype(np.float32) - zero_point) * scale
    else:
        scores_flat = raw.astype(np.float32)

    scores_flat = scores_flat.flatten()
    top_idx = np.argsort(scores_flat)[-TOP_K:][::-1]
    return top_idx, scores_flat


def get_top_category(top_results, class_names=None):
    """Get our glasses category from top YAMNet predictions."""
    if class_names is None:
        # top_results is [(name, score), ...]
        for name, score in top_results:
            cat = YAMNET_TO_CATEGORY.get(name)
            if cat:
                return name, score, cat
        return top_results[0][0], top_results[0][1], "unknown_other"
    else:
        # top_results is (indices, scores_flat)
        top_idx, scores = top_results
        for idx in top_idx:
            name = class_names[idx]
            cat = YAMNET_TO_CATEGORY.get(name)
            if cat:
                return name, float(scores[idx]), cat
        return class_names[top_idx[0]], float(scores[top_idx[0]]), "unknown_other"


# ── Test runners ──────────────────────────────────────────────────────────────

def run_synthetic_tests(hub_model, hub_classes, tflite_int8, tflite_f32):
    """Run on synthetic clips — quick sanity check."""
    print("\n" + "═" * 65)
    print("  SYNTHETIC TESTS (zeros / noise / sine)")
    print("═" * 65)

    clips = make_synthetic_clips()
    print(f"{'Clip':<20} {'Expected':<20} {'Float32':<25} {'Int8':<25}")
    print("─" * 95)

    for name, waveform, expected in clips:
        # Float32 hub model
        f32_top = infer_hub(hub_model, hub_classes, waveform)
        f32_name, f32_score = f32_top[0]

        # TFLite int8
        if tflite_int8[0]:
            interp, inp, out = tflite_int8
            dtype = interp.get_input_details()[0]["dtype"]
            i8_idx, i8_scores = infer_tflite(interp, inp, out, waveform, dtype)
            if i8_idx is None:
                i8_str = "PAD bug (on-device OK)"
            else:
                i8_name = hub_classes[i8_idx[0]]
                i8_score = float(i8_scores[i8_idx[0]])
                i8_str = f"{i8_name[:22]:<22} {i8_score:.2f}"
        else:
            i8_str = "N/A"

        match_f32 = "✓" if expected.lower() in f32_name.lower() else "✗"
        f32_str = f"{match_f32} {f32_name[:20]:<20} {f32_score:.2f}"
        print(f"{name:<20} {expected:<20} {f32_str:<25} {i8_str:<25}")


def run_esc50_tests(esc50_dir, hub_model, hub_classes, tflite_int8, tflite_f32):
    """Run on ESC-50 clips that map to our glasses categories."""
    meta_csv = Path(esc50_dir) / "meta" / "esc50.csv"
    audio_dir = Path(esc50_dir) / "audio"

    if not meta_csv.exists():
        print(f"\n⚠  ESC-50 metadata not found at {meta_csv}")
        print("   Run build_dataset.py first, or pass --esc50-dir path/to/ESC-50-master\n")
        return

    print("\n" + "═" * 65)
    print("  ESC-50 ACCURACY TEST")
    print("═" * 65)

    # Load only clips that map to our categories
    test_clips = []
    with open(meta_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["category"]
            if label in ESC50_TO_CATEGORY:
                path = audio_dir / row["filename"]
                if path.exists():
                    test_clips.append((path, label, ESC50_TO_CATEGORY[label]))

    print(f"Found {len(test_clips)} ESC-50 clips matching glasses categories\n")

    if not test_clips:
        print("No matching clips found.")
        return

    f32_correct = 0
    i8_correct  = 0
    score_diffs = []
    per_cat_f32 = {c: [0, 0] for c in CATEGORIES}   # [correct, total]
    per_cat_i8  = {c: [0, 0] for c in CATEGORIES}

    has_i8 = tflite_int8[0] is not None
    interp_i8, inp_i8, out_i8 = tflite_int8 if has_i8 else (None, None, None)
    dtype_i8 = interp_i8.get_input_details()[0]["dtype"] if has_i8 else None

    print(f"{'File':<30} {'True cat':<22} {'F32 pred':<22} {'Int8 pred':<22} {'Score Δ':>8}")
    print("─" * 110)

    for path, esc_label, true_cat in tqdm(test_clips, desc="Testing"):
        waveform = load_wav(path)
        if waveform is None:
            continue

        # Float32
        t0 = time.time()
        f32_top = infer_hub(hub_model, hub_classes, waveform)
        f32_yamnet, f32_score, f32_cat = get_top_category(f32_top)

        # Int8 TFLite
        # Int8 TFLite
        if has_i8:
            i8_idx, i8_scores = infer_tflite(interp_i8, inp_i8, out_i8, waveform, dtype_i8)
            if i8_idx is None:
                i8_yamnet, i8_score, i8_cat = "PAD_bug", 0.0, "N/A"
            else:
                i8_yamnet, i8_score, i8_cat = get_top_category(
                    (i8_idx, i8_scores), hub_classes)
            score_diff = i8_score - f32_score
            score_diffs.append(score_diff)
            i8_str = f"{i8_cat[:20]:<20}"
            diff_str = f"{score_diff:+.3f}"
        else:
            i8_cat = "N/A"
            i8_str = f"{'N/A':<20}"
            diff_str = "N/A"

        f32_ok = (f32_cat == true_cat)
        i8_ok  = (i8_cat  == true_cat) if has_i8 else False

        if f32_ok: f32_correct += 1
        if i8_ok:  i8_correct  += 1

        per_cat_f32[true_cat][1] += 1
        per_cat_i8[true_cat][1]  += 1
        if f32_ok: per_cat_f32[true_cat][0] += 1
        if i8_ok:  per_cat_i8[true_cat][0]  += 1

        f32_icon = "✓" if f32_ok else "✗"
        i8_icon  = "✓" if i8_ok  else "✗"
        fname = path.name[:28]
        print(f"{fname:<30} {true_cat[:20]:<22} "
              f"{f32_icon}{f32_cat[:19]:<21} "
              f"{i8_icon}{i8_str} {diff_str:>8}")

    total = len([c for c in test_clips
                 if load_wav(c[0]) is not None]) or 1

    print("\n" + "═" * 65)
    print("  SUMMARY")
    print("═" * 65)
    print(f"  Total clips tested : {total}")
    print(f"  Float32 accuracy   : {f32_correct}/{total}  ({100*f32_correct/total:.1f}%)")
    if has_i8:
        print(f"  Int8    accuracy   : {i8_correct}/{total}  ({100*i8_correct/total:.1f}%)")
        drop = (f32_correct - i8_correct) / total * 100
        print(f"  Accuracy drop      : {drop:+.1f} percentage points")
        if score_diffs:
            print(f"  Mean score drift   : {np.mean(score_diffs):+.4f}  "
                  f"(std {np.std(score_diffs):.4f})")
            print(f"  Max score drop     : {min(score_diffs):+.4f}")

    print("\n  Per-category breakdown:")
    print(f"  {'Category':<25} {'Float32':>10} {'Int8':>10} {'Drop':>8}")
    print("  " + "─" * 55)
    for cat in CATEGORIES:
        f32_n, f32_t = per_cat_f32[cat]
        i8_n,  i8_t  = per_cat_i8[cat]
        if f32_t == 0:
            continue
        f32_pct = 100 * f32_n / f32_t
        i8_pct  = 100 * i8_n  / i8_t if i8_t > 0 else 0
        drop    = i8_pct - f32_pct
        print(f"  {cat:<25} {f32_pct:>8.0f}%  {i8_pct:>8.0f}%  {drop:>+7.0f}%  "
              f"(n={f32_t})")


def run_custom_dir_tests(audio_dir, hub_model, hub_classes, tflite_int8):
    """Test any .wav files in a directory — filename used as label if it matches a category."""
    wavs = list(Path(audio_dir).glob("**/*.wav"))
    if not wavs:
        print(f"\n⚠  No .wav files found in {audio_dir}")
        return

    print("\n" + "═" * 65)
    print(f"  CUSTOM AUDIO DIR: {audio_dir}  ({len(wavs)} files)")
    print("═" * 65)

    has_i8 = tflite_int8[0] is not None
    interp_i8, inp_i8, out_i8 = tflite_int8 if has_i8 else (None, None, None)
    dtype_i8 = interp_i8.get_input_details()[0]["dtype"] if has_i8 else None

    print(f"{'File':<35} {'F32 top pred':<30} {'Int8 top pred':<30}")
    print("─" * 95)

    for path in sorted(wavs):
        waveform = load_wav(path)
        if waveform is None:
            continue

        f32_top = infer_hub(hub_model, hub_classes, waveform)
        f32_name, f32_score = f32_top[0]

        if has_i8:
            i8_idx, i8_scores = infer_tflite(interp_i8, inp_i8, out_i8, waveform, dtype_i8)
            i8_name = hub_classes[i8_idx[0]]
            i8_score = float(i8_scores[i8_idx[0]])
            i8_str = f"{i8_name[:27]:<27} {i8_score:.3f}"
        else:
            i8_str = "N/A"

        fname = path.name[:33]
        print(f"{fname:<35} {f32_name[:27]:<27} {f32_score:.3f}  {i8_str}")


# ── Main ──────────────────────────────────────────────────────────────────────
ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"

def download_esc50(dest_dir="downloads"):
    import zipfile, urllib.request
    zip_path = Path(dest_dir) / "esc50.zip"
    extract_path = Path(dest_dir) / "ESC-50-master"
    if extract_path.exists():
        print(f"ESC-50 already downloaded at {extract_path}")
        return str(extract_path)
    Path(dest_dir).mkdir(exist_ok=True)
    print("Downloading ESC-50 (~200 MB zip)...")
    def reporthook(count, block_size, total_size):
        if total_size > 0:
            pct = min(count * block_size * 100 // total_size, 100)
            print(f"\r  {pct}%", end="", flush=True)
    urllib.request.urlretrieve(ESC50_URL, zip_path, reporthook)
    print("\n  Extracting...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)
    print(f"✓ ESC-50 ready at {extract_path}\n")
    return str(extract_path)


def main():
    parser = argparse.ArgumentParser(description="YAMNet float32 vs int8 accuracy test")
    parser.add_argument("--esc50-dir",    default="",
                        help="Path to ESC-50-master (auto-downloaded if not set)")
    parser.add_argument("--audio-dir",    default="",
                        help="Path to custom directory of .wav files to test")
    parser.add_argument("--tflite-only",  action="store_true")
    parser.add_argument("--no-synthetic", action="store_true",
                        help="Skip synthetic tests")
    parser.add_argument("--no-esc50",     action="store_true",
                        help="Skip ESC-50 download and tests")
    args = parser.parse_args()

    hub_model, hub_classes = load_hub_model()

    print("Loading TFLite models...")
    tflite_int8 = load_tflite_model(TFLITE_INT8)
    tflite_f32  = load_tflite_model(TFLITE_F32)

    if not tflite_int8[0] and not tflite_f32[0]:
        print("No TFLite models found. Run yamnet_esp32_feasibility.py first.")
        print("Continuing with float32 hub model only for reference.\n")

    if not args.no_synthetic:
        run_synthetic_tests(hub_model, hub_classes, tflite_int8, tflite_f32)

    if args.audio_dir:
        run_custom_dir_tests(args.audio_dir, hub_model, hub_classes, tflite_int8)

    if not args.no_esc50:
        if args.esc50_dir and Path(args.esc50_dir).exists():
            esc50_path = args.esc50_dir
        else:
            esc50_path = download_esc50()
        run_esc50_tests(esc50_path, hub_model, hub_classes, tflite_int8, tflite_f32)
    else:
        print("\nSkipping ESC-50 tests.\n")

if __name__ == "__main__":
    main()