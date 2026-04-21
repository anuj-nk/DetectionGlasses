"""
YAMNet Accuracy Test — UrbanSound8K via soundata
=================================================
Tests float32 (TF Hub) and float32 TFLite models against
UrbanSound8K, mapping its 10 classes to your 8 glasses categories.

UrbanSound8K classes → glasses categories:
  car_horn        → horn_beeping
  dog_bark        → dog_barking
  gun_shot        → alarm_siren  (treated as danger alert)
  siren           → alarm_siren
  children_playing→ speech
  air_conditioner → unknown_other
  drilling        → unknown_other
  engine_idling   → unknown_other
  jackhammer      → unknown_other
  street_music    → unknown_other

Note on int8: the full int8 YAMNet has a known PAD kernel bug in the
desktop TFLite runtime. The float32 TFLite model runs fine and gives
a valid on-desktop comparison. int8 results are shown where possible
and marked as "PAD bug" where not.

Install:
    pip install soundata tensorflow tensorflow-hub numpy librosa tqdm

Usage:
    python yamnet_urbansound_test.py                  # auto-download + test
    python yamnet_urbansound_test.py --limit 200      # test first 200 clips
    python yamnet_urbansound_test.py --no-tflite      # hub model only
    python yamnet_urbansound_test.py --category siren # one category only
"""

import argparse
import csv
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import soundata

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE  = 16000
CLIP_SAMPLES = 16000 * 3   # pad/trim all clips to 3 seconds
TOP_K        = 5
TFLITE_INT8  = Path("tflite_models/yamnet_int8.tflite")
TFLITE_F32   = Path("tflite_models/yamnet_full.tflite")

# ── UrbanSound8K class → glasses category ────────────────────────────────────
US8K_TO_GLASSES = {
    "car_horn":          "horn_beeping",
    "dog_bark":          "dog_barking",
    "gun_shot":          "alarm_siren",
    "siren":             "alarm_siren",
    "children_playing":  "speech",
    "air_conditioner":   "unknown_other",
    "drilling":          "unknown_other",
    "engine_idling":     "unknown_other",
    "jackhammer":        "unknown_other",
    "street_music":      "unknown_other",
}

# ── YAMNet class → glasses category ──────────────────────────────────────────
YAMNET_TO_GLASSES = {
    "Speech":                     "speech",
    "Narration, monologue":       "speech",
    "Conversation":               "speech",
    "Shout":                      "speech",
    "Laughter":                   "speech",
    "Baby cry, infant cry":       "speech",
    "Cough":                      "speech",
    "Sneeze":                     "speech",
    "Child speech, kid speaking": "speech",
    "Alarm":                      "alarm_siren",
    "Siren":                      "alarm_siren",
    "Fire alarm":                 "alarm_siren",
    "Smoke detector, smoke alarm":"alarm_siren",
    "Buzzer":                     "alarm_siren",
    "Emergency vehicle":          "alarm_siren",
    "Police car (siren)":         "alarm_siren",
    "Ambulance (siren)":          "alarm_siren",
    "Gunshot, gunfire":           "alarm_siren",
    "Car":                        "horn_beeping",
    "Horn":                       "horn_beeping",
    "Beep, bleep":                "horn_beeping",
    "Bicycle bell":               "horn_beeping",
    "Car alarm":                  "horn_beeping",
    "Knock":                      "door_knock_doorbell",
    "Doorbell":                   "door_knock_doorbell",
    "Door":                       "door_knock_doorbell",
    "Slam":                       "door_knock_doorbell",
    "Telephone":                  "phone_ringing",
    "Ringtone":                   "phone_ringing",
    "Telephone bell ringing":     "phone_ringing",
    "Cell phone":                 "phone_ringing",
    "Dog":                        "dog_barking",
    "Bark":                       "dog_barking",
    "Growling":                   "dog_barking",
    "Microwave oven":             "appliance_beeping",
    "Washing machine":            "appliance_beeping",
    "Dishwasher":                 "appliance_beeping",
    "Silence":                    "silence",
    "White noise":                "silence",
}

GLASSES_CATEGORIES = [
    "speech", "alarm_siren", "horn_beeping", "door_knock_doorbell",
    "phone_ringing", "dog_barking", "appliance_beeping",
    "silence", "unknown_other",
]


# ── Audio prep ────────────────────────────────────────────────────────────────

def prepare_waveform(audio_tuple):
    """
    Takes soundata's (waveform, sr) tuple.
    Resamples to 16kHz mono, pads/trims to CLIP_SAMPLES.
    Returns float32 numpy array or None on failure.
    """
    if audio_tuple is None:
        return None
    try:
        y, sr = audio_tuple
        if y is None:
            return None
        # soundata may return stereo
        if y.ndim > 1:
            y = np.mean(y, axis=0)
        # resample if needed
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        # trim / pad to CLIP_SAMPLES
        if len(y) > CLIP_SAMPLES:
            y = y[:CLIP_SAMPLES]
        elif len(y) < CLIP_SAMPLES:
            y = np.pad(y, (0, CLIP_SAMPLES - len(y)))
        return y.astype(np.float32)
    except Exception as e:
        return None


# ── Model loading ─────────────────────────────────────────────────────────────

def load_hub_model():
    print("Loading YAMNet float32 from TF Hub...")
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    class_map_path = model.class_map_path().numpy().decode()
    class_names = []
    with open(class_map_path) as f:
        reader = csv.reader(f)
        next(reader)
        class_names = [row[2] for row in reader]
    print(f"  ✓ {len(class_names)} classes loaded\n")
    return model, class_names


def load_tflite(path: Path, label: str):
    """Load a TFLite model. Returns (interp, inp_idx, out_idx, dtype) or None."""
    if not path.exists():
        print(f"  ⚠  {label} not found at {path} — skipping")
        return None
    print(f"Loading {label}: {path}")
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.resize_tensor_input(
        interp.get_input_details()[0]["index"], [CLIP_SAMPLES])
    interp.allocate_tensors()
    inp   = interp.get_input_details()[0]["index"]
    out   = interp.get_output_details()[0]["index"]
    dtype = interp.get_input_details()[0]["dtype"]
    print(f"  dtype={dtype.__name__}  "
          f"in={interp.get_input_details()[0]['shape']}  "
          f"out={interp.get_output_details()[0]['shape']}\n")
    return interp, inp, out, dtype


# ── Inference ─────────────────────────────────────────────────────────────────

def infer_hub(model, class_names, waveform):
    """Run TF Hub model → [(class_name, score), ...] top-K."""
    scores, _, _ = model(waveform)
    mean = np.mean(scores.numpy(), axis=0)
    top  = np.argsort(mean)[-TOP_K:][::-1]
    return [(class_names[i], float(mean[i])) for i in top]


def infer_tflite(interp, inp_idx, out_idx, dtype, waveform):
    """
    Run TFLite model → (top_indices, scores_flat) or (None, None) on PAD bug.
    """
    wv = (waveform * 127).clip(-128, 127).astype(np.int8) \
         if dtype == np.int8 else waveform.copy()
    interp.set_tensor(inp_idx, wv)
    try:
        interp.invoke()
    except RuntimeError as e:
        if "PAD" in str(e) or "Pad value" in str(e):
            return None, None
        raise
    raw = interp.get_tensor(out_idx)
    if raw.dtype == np.int8:
        d = interp.get_output_details()[0]
        scale, zp = d["quantization"]
        scores = (raw.astype(np.float32) - zp) * scale
    else:
        scores = raw.astype(np.float32)
    scores = scores.flatten()
    top    = np.argsort(scores)[-TOP_K:][::-1]
    return top, scores


def top_glasses_category(results, class_names=None):
    """
    Map top predictions to glasses category.
    results: [(name, score), ...] for hub model
             (indices, scores)   for tflite
    Returns (yamnet_name, score, glasses_cat)
    """
    if class_names is None:
        # hub model format
        for name, score in results:
            cat = YAMNET_TO_GLASSES.get(name)
            if cat:
                return name, score, cat
        return results[0][0], results[0][1], "unknown_other"
    else:
        # tflite format
        top_idx, scores = results
        if top_idx is None:
            return "PAD_bug", 0.0, "PAD_bug"
        for idx in top_idx:
            name = class_names[idx]
            cat  = YAMNET_TO_GLASSES.get(name)
            if cat:
                return name, float(scores[idx]), cat
        return class_names[top_idx[0]], float(scores[top_idx[0]]), "unknown_other"


# ── Main test loop ────────────────────────────────────────────────────────────

def run_test(hub_model, hub_classes, tflite_f32, tflite_i8,
             limit=None, filter_category=None):

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("Initialising UrbanSound8K via soundata...")
    dataset = soundata.initialize("urbansound8k")

    print("Downloading / verifying dataset (skip if already present)...")
    dataset.download()
    print("✓ Dataset ready\n")

    clip_ids = dataset.clip_ids

    # Filter to only clips mapping to glasses categories of interest
    relevant_ids = []
    for cid in clip_ids:
        clip = dataset.clip(cid)
        us8k_label = clip.class_label
        glasses_cat = US8K_TO_GLASSES.get(us8k_label)
        if glasses_cat is None:
            continue
        if filter_category and glasses_cat != filter_category:
            continue
        relevant_ids.append((cid, us8k_label, glasses_cat))

    if limit:
        relevant_ids = relevant_ids[:limit]

    print(f"Testing {len(relevant_ids)} clips "
          f"({'all' if not filter_category else filter_category})\n")

    # ── Counters ──────────────────────────────────────────────────────────────
    hub_correct  = 0
    f32_correct  = 0
    i8_correct   = 0
    i8_pad_count = 0
    total        = 0

    per_cat_hub = defaultdict(lambda: [0, 0])   # [correct, total]
    per_cat_f32 = defaultdict(lambda: [0, 0])
    per_cat_i8  = defaultdict(lambda: [0, 0])
    confusion    = defaultdict(lambda: defaultdict(int))  # true → predicted

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"{'Clip ID':<25} {'US8K label':<20} {'True cat':<22} "
          f"{'HUB pred':<22} {'F32 TFLite':<22} {'Int8 TFLite':<22}")
    print("─" * 135)

    for cid, us8k_label, true_cat in tqdm(relevant_ids, desc="Inferring"):
        clip     = dataset.clip(cid)
        waveform = prepare_waveform(clip.audio)
        if waveform is None:
            continue
        total += 1

        # Hub float32
        hub_top = infer_hub(hub_model, hub_classes, waveform)
        hub_name, hub_score, hub_cat = top_glasses_category(hub_top)

        # TFLite float32
        if tflite_f32:
            interp, inp, out, dtype = tflite_f32
            f32_idx, f32_scores = infer_tflite(interp, inp, out, dtype, waveform)
            f32_name, f32_score, f32_cat = top_glasses_category(
                (f32_idx, f32_scores), hub_classes)
        else:
            f32_cat = "N/A"

        # TFLite int8
        if tflite_i8:
            interp, inp, out, dtype = tflite_i8
            i8_idx, i8_scores = infer_tflite(interp, inp, out, dtype, waveform)
            if i8_idx is None:
                i8_cat = "PAD_bug"
                i8_pad_count += 1
            else:
                _, _, i8_cat = top_glasses_category(
                    (i8_idx, i8_scores), hub_classes)
        else:
            i8_cat = "N/A"

        # Score
        hub_ok = (hub_cat == true_cat)
        f32_ok = (f32_cat == true_cat)
        i8_ok  = (i8_cat  == true_cat)

        if hub_ok: hub_correct += 1
        if f32_ok: f32_correct += 1
        if i8_ok:  i8_correct  += 1

        per_cat_hub[true_cat][1] += 1
        per_cat_f32[true_cat][1] += 1
        per_cat_i8[true_cat][1]  += 1
        if hub_ok: per_cat_hub[true_cat][0] += 1
        if f32_ok: per_cat_f32[true_cat][0] += 1
        if i8_ok:  per_cat_i8[true_cat][0]  += 1

        confusion[true_cat][hub_cat] += 1

        hub_icon = "✓" if hub_ok else "✗"
        f32_icon = "✓" if f32_ok else "✗"
        i8_icon  = "✓" if i8_ok  else ("⚠" if i8_cat == "PAD_bug" else "✗")

        print(f"{cid:<25} {us8k_label:<20} {true_cat:<22} "
              f"{hub_icon}{hub_cat:<21} "
              f"{f32_icon}{f32_cat:<21} "
              f"{i8_icon}{i8_cat:<21}")

    # ── Summary ───────────────────────────────────────────────────────────────
    if total == 0:
        print("\nNo clips processed.")
        return

    print("\n" + "═" * 70)
    print("  SUMMARY — UrbanSound8K Accuracy Test")
    print("═" * 70)
    print(f"  Total clips tested   : {total}")
    print(f"  HUB float32 accuracy : {hub_correct}/{total}  "
          f"({100*hub_correct/total:.1f}%)")

    if tflite_f32:
        print(f"  F32 TFLite accuracy  : {f32_correct}/{total}  "
              f"({100*f32_correct/total:.1f}%)")
        diff = hub_correct - f32_correct
        print(f"  Hub vs F32 TFLite    : {diff:+d} clips  "
              f"({100*diff/total:+.1f}pp)")

    if tflite_i8:
        if i8_pad_count == total:
            print(f"  Int8 TFLite          : PAD kernel bug on desktop — "
                  f"all {total} clips failed")
            print("                         Model is valid; test on XIAO for real numbers")
        else:
            print(f"  Int8 TFLite accuracy : {i8_correct}/{total}  "
                  f"({100*i8_correct/total:.1f}%)  "
                  f"({i8_pad_count} PAD failures)")

    # Per-category breakdown
    print(f"\n  {'Category':<25} {'n':>5} {'HUB':>8} "
          f"{'F32 TFLite':>12} {'Int8':>8}")
    print("  " + "─" * 62)

    # Only show categories that have test clips
    tested_cats = sorted(
        set(per_cat_hub.keys()),
        key=lambda c: -per_cat_hub[c][1]
    )
    for cat in tested_cats:
        n        = per_cat_hub[cat][1]
        hub_pct  = 100 * per_cat_hub[cat][0] / n if n else 0
        f32_pct  = 100 * per_cat_f32[cat][0] / n if n and tflite_f32 else None
        i8_pct   = 100 * per_cat_i8[cat][0]  / n if n and tflite_i8 else None

        f32_str = f"{f32_pct:>10.0f}%" if f32_pct is not None else f"{'N/A':>11}"
        i8_str  = (f"{i8_pct:>6.0f}%"
                   if (i8_pct is not None and i8_pad_count < total)
                   else f"{'PAD bug':>8}")
        print(f"  {cat:<25} {n:>5} {hub_pct:>7.0f}% {f32_str} {i8_str}")

    # Confusion matrix (hub model)
    print(f"\n  CONFUSION MATRIX (HUB float32)")
    print(f"  {'True  Pred':<22}", end="")
    pred_cats = sorted(set(
        p for preds in confusion.values() for p in preds.keys()
    ))
    for p in pred_cats:
        print(f"  {p[:10]:>10}", end="")
    print()
    print("  " + "─" * (22 + 12 * len(pred_cats)))
    for true_cat in tested_cats:
        print(f"  {true_cat:<22}", end="")
        for p in pred_cats:
            count = confusion[true_cat].get(p, 0)
            print(f"  {count:>10}", end="")
        print()

    print(f"\n  Note: UrbanSound8K covers "
          f"{len(tested_cats)} of your 8 glasses categories.")
    print(f"  Categories NOT in UrbanSound8K: "
          f"door_knock_doorbell, phone_ringing, appliance_beeping")
    print(f"  → Use ESC-50 results for those categories.\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="YAMNet accuracy test on UrbanSound8K via soundata")
    parser.add_argument("--limit",    type=int, default=None,
                        help="Max clips to test (default: all ~2900 relevant)")
    parser.add_argument("--category", default=None,
                        help="Test one glasses category only "
                             "(e.g. dog_barking, alarm_siren, horn_beeping)")
    parser.add_argument("--no-tflite", action="store_true",
                        help="Skip TFLite models, test hub model only")
    args = parser.parse_args()

    # Load models
    hub_model, hub_classes = load_hub_model()

    tflite_f32 = None
    tflite_i8  = None
    if not args.no_tflite:
        print("Loading TFLite models...")
        f32 = load_tflite(TFLITE_F32, "Float32 TFLite")
        i8  = load_tflite(TFLITE_INT8, "Int8 TFLite")
        tflite_f32 = f32
        tflite_i8  = i8

    run_test(hub_model, hub_classes, tflite_f32, tflite_i8,
             limit=args.limit,
             filter_category=args.category)


if __name__ == "__main__":
    main()