"""
YAMNet Real-Time Microphone Test
=================================
Tests YAMNet using your laptop microphone.
Continuously classifies ambient noise and prints top predictions.

Requirements:
    pip install tensorflow tensorflow-hub sounddevice numpy scipy

Usage:
    python yamnet_mic_test.py
"""

import time
import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000   # YAMNet expects 16 kHz mono
WINDOW_SEC    = 2.0     # how many seconds of audio to classify at once
HOP_SEC       = 0.5     # how often to re-run inference (seconds)
TOP_K         = 5       # how many top classes to display
MIN_SCORE     = 0.05    # ignore classes below this confidence

# Noise-to-LED color mapping for the glasses concept
# (class substring → RGB color name)
NOISE_COLOR_MAP = {
    "speech":       "🔵 BLUE   (human speech)",
    "music":        "🟣 PURPLE (music)",
    "alarm":        "🔴 RED    (alarm/danger)",
    "siren":        "🔴 RED    (alarm/danger)",
    "dog":          "🟡 YELLOW (animal)",
    "cat":          "🟡 YELLOW (animal)",
    "door":         "🟠 ORANGE (door/knock)",
    "knock":        "🟠 ORANGE (door/knock)",
    "vehicle":      "⚪ WHITE  (vehicle)",
    "car":          "⚪ WHITE  (vehicle)",
    "silence":      "⚫ OFF    (silence)",
}

def get_led_color(class_name: str) -> str:
    """Map a YAMNet class name to a glasses LED color."""
    lower = class_name.lower()
    for keyword, color in NOISE_COLOR_MAP.items():
        if keyword in lower:
            return color
    return "🟢 GREEN  (other sound)"


def load_model():
    print("Loading YAMNet from TensorFlow Hub...")
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    # Load class names from the model asset
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    class_names = []
    with open(class_map_path) as f:
        import csv
        reader = csv.reader(f)
        next(reader)  # skip header
        class_names = [row[2] for row in reader]
    print(f"✓ Loaded YAMNet ({len(class_names)} classes)\n")
    return model, class_names


def classify(model, class_names, waveform: np.ndarray):
    """Run YAMNet on a waveform and return top-k (class, score) pairs."""
    waveform = waveform.astype(np.float32)
    scores, embeddings, spectrogram = model(waveform)
    # Average scores across time frames → single clip-level prediction
    mean_scores = np.mean(scores.numpy(), axis=0)
    top_k_idx = np.argsort(mean_scores)[-TOP_K:][::-1]
    return [(class_names[i], float(mean_scores[i])) for i in top_k_idx
            if mean_scores[i] >= MIN_SCORE]


def run_mic_test(model, class_names):
    """Capture audio from the default mic and classify in a loop."""
    window_samples = int(WINDOW_SEC * SAMPLE_RATE)
    hop_samples    = int(HOP_SEC    * SAMPLE_RATE)

    print("=" * 60)
    print("  YAMNet Noise Detection — Glasses LED Demo")
    print("=" * 60)
    print(f"Listening on default microphone at {SAMPLE_RATE} Hz")
    print(f"Window: {WINDOW_SEC}s | Refresh: {HOP_SEC}s | Top-{TOP_K} shown")
    print("Press Ctrl+C to stop.\n")

    # Ring buffer to hold a rolling window of audio
    ring_buffer = np.zeros(window_samples, dtype=np.float32)

    def audio_callback(indata, frames, time_info, status):
        nonlocal ring_buffer
        chunk = indata[:, 0]  # mono
        ring_buffer = np.roll(ring_buffer, -len(chunk))
        ring_buffer[-len(chunk):] = chunk

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        blocksize=hop_samples, callback=audio_callback):
        try:
            while True:
                time.sleep(HOP_SEC)
                snapshot = ring_buffer.copy()

                results = classify(model, class_names, snapshot)
                if not results:
                    print("[No confident prediction]")
                    continue

                top_class, top_score = results[0]
                led = get_led_color(top_class)

                print(f"\n┌─ Top prediction ─────────────────────────────")
                print(f"│  {top_class:<35} {top_score:.1%}")
                print(f"│  LED → {led}")
                if len(results) > 1:
                    print(f"├─ Other candidates ───────────────────────────")
                    for name, score in results[1:]:
                        print(f"│    {name:<33} {score:.1%}")
                print(f"└──────────────────────────────────────────────")

        except KeyboardInterrupt:
            print("\n\nStopped. Bye!")


if __name__ == "__main__":
    model, class_names = load_model()
    run_mic_test(model, class_names)
