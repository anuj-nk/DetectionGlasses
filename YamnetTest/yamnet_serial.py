"""
YAMNet → Serial → XIAO NeoPixel  (with amplitude-driven brightness)
=====================================================================
Runs YAMNet on your laptop mic, sends classification + brightness
to the XIAO ESP32-S3 over USB Serial to drive the NeoPixel LEDs.

Brightness tracks how loud the sound is in real time:
  quiet / distant  →  dim
  loud  / close    →  full brightness

Protocol additions:
  C:speech:0.92:180   ← class, confidence, brightness (0-255)
  B:180               ← brightness-only update (no class change)

Usage:
    python yamnet_serial.py --port /dev/cu.usbmodem1101
    python yamnet_serial.py --port /dev/cu.usbmodem1101 --threshold 0.6
    python yamnet_serial.py --list-ports

Requirements:
    pip install tensorflow tensorflow-hub sounddevice numpy pyserial
"""

import argparse
import time
import threading
import numpy as np
import sounddevice as sd
import serial
import serial.tools.list_ports
import tensorflow_hub as hub

# ── YAMNet class → glasses category ──────────────────────────────────────────
# 9 categories tuned for a deaf wearer; priority order matters for UX:
#   danger > speech > vehicle > door_entry > phone > animal > appliance > music > silence
YAMNET_TO_CATEGORY = {
    # DANGER — fire / alarm / emergency (most critical to surface immediately)
    "Alarm":                          "danger",
    "Siren":                          "danger",
    "Fire alarm":                     "danger",
    "Smoke detector, smoke alarm":    "danger",
    "Emergency vehicle":              "danger",
    "Police car (siren)":             "danger",
    "Ambulance (siren)":              "danger",
    "Civil defense siren":            "danger",
    "Gunshot, gunfire":               "danger",
    "Explosion":                      "danger",
    "Glass":                          "danger",
    "Scream":                         "danger",
    "Yell":                           "danger",
    "Buzzer":                         "danger",
    "Alarm clock":                    "danger",

    # SPEECH — human voice (awareness of people nearby)
    "Speech":                         "speech",
    "Narration, monologue":           "speech",
    "Conversation":                   "speech",
    "Shout":                          "speech",
    "Laughter":                       "speech",
    "Baby cry, infant cry":           "speech",
    "Cough":                          "speech",
    "Sneeze":                         "speech",
    "Child speech, kid speaking":     "speech",
    "Whispering":                     "speech",
    "Crying, sobbing":                "speech",
    "Groan":                          "speech",
    "Clapping":                       "speech",
    "Crowd":                          "speech",

    # VEHICLE — traffic & transport
    "Car":                            "vehicle",
    "Truck":                          "vehicle",
    "Motorcycle":                     "vehicle",
    "Bus":                            "vehicle",
    "Horn":                           "vehicle",
    "Car alarm":                      "vehicle",
    "Beep, bleep":                    "vehicle",
    "Bicycle bell":                   "vehicle",
    "Engine":                         "vehicle",
    "Traffic noise, roadway noise":   "vehicle",
    "Skateboard":                     "vehicle",
    "Bicycle":                        "vehicle",
    "Train":                          "vehicle",
    "Airplane":                       "vehicle",

    # DOOR / ENTRY
    "Knock":                          "door_entry",
    "Doorbell":                       "door_entry",
    "Door":                           "door_entry",
    "Slam":                           "door_entry",
    "Squeak":                         "door_entry",

    # PHONE / NOTIFICATION
    "Telephone":                      "phone",
    "Ringtone":                       "phone",
    "Telephone bell ringing":         "phone",
    "Cell phone":                     "phone",

    # ANIMAL
    "Dog":                            "animal",
    "Bark":                           "animal",
    "Growling":                       "animal",
    "Cat":                            "animal",
    "Meow":                           "animal",
    "Bird":                           "animal",
    "Bird vocalization, bird call, bird song": "animal",
    "Hiss":                           "animal",

    # HOME APPLIANCE
    "Microwave oven":                 "appliance",
    "Washing machine":                "appliance",
    "Dishwasher":                     "appliance",
    "Vacuum cleaner":                 "appliance",
    "Blender":                        "appliance",
    "Toilet flush":                   "appliance",

    # MUSIC / MEDIA
    "Music":                          "music",
    "Musical instrument":             "music",
    "Singing":                        "music",
    "Song":                           "music",
    "Television":                     "music",
    "Radio":                          "music",
    "Jingle":                         "music",

    # SILENCE
    "Silence":                        "silence",
    "White noise":                    "silence",
}

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE  = 16000
WINDOW_SEC   = 2.0
HOP_SEC      = 0.5
BAUD_RATE    = 115200

# Amplitude → brightness tuning
# Adjust MIN_DB / MAX_DB to match your environment:
#   too dim overall  → lower MIN_DB (e.g. -60)
#   always max brightness → raise MIN_DB (e.g. -40)
MIN_DB         = -50.0   # dB level that maps to minimum brightness
MAX_DB         = -10.0   # dB level that maps to maximum brightness
MIN_BRIGHTNESS =  8      # never fully off when a sound is detected (0-255)
MAX_BRIGHTNESS = 255     # full brightness at loud sounds
SMOOTHING      = 0.35    # 0=instant, 1=frozen. 0.35 feels natural


# ── Amplitude helpers ─────────────────────────────────────────────────────────
_smoothed_brightness = 0.0

def compute_brightness(waveform):
    """
    RMS amplitude of waveform → smoothed NeoPixel brightness (0–255).

    Uses exponential smoothing so brightness rises/falls gradually
    rather than snapping, which feels more natural on the LEDs.
    """
    global _smoothed_brightness

    rms = float(np.sqrt(np.mean(waveform ** 2)))
    rms = max(rms, 1e-9)          # avoid log(0)
    db  = 20.0 * np.log10(rms)

    # Map dB range → 0.0–1.0
    t = (db - MIN_DB) / (MAX_DB - MIN_DB)
    t = float(np.clip(t, 0.0, 1.0))

    target = MIN_BRIGHTNESS + t * (MAX_BRIGHTNESS - MIN_BRIGHTNESS)

    # Exponential smoothing
    _smoothed_brightness = (SMOOTHING * _smoothed_brightness +
                            (1.0 - SMOOTHING) * target)
    return int(_smoothed_brightness)


def brightness_bar(brightness, width=20):
    """ASCII bar for terminal display."""
    filled = int(brightness / 255 * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {brightness:3d}"


# ── YAMNet helpers ────────────────────────────────────────────────────────────
def list_ports():
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports found.")
    else:
        print("Available serial ports:")
        for p in ports:
            print(f"  {p.device}  —  {p.description}")


def load_yamnet():
    print("Loading YAMNet...")
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    import csv
    class_map_path = model.class_map_path().numpy().decode()
    class_names = []
    with open(class_map_path) as f:
        reader = csv.reader(f)
        next(reader)
        class_names = [row[2] for row in reader]
    print(f"✓ YAMNet loaded ({len(class_names)} classes)\n")
    return model, class_names


def classify(model, class_names, waveform):
    scores, _, _ = model(waveform.astype(np.float32))
    mean_scores  = np.mean(scores.numpy(), axis=0)
    top_idx      = int(np.argmax(mean_scores))
    top_score    = float(mean_scores[top_idx])
    top_name     = class_names[top_idx]
    category     = YAMNET_TO_CATEGORY.get(top_name, "unknown_other")
    return top_name, top_score, category


# ── Main run loop ─────────────────────────────────────────────────────────────
def run(port, threshold, dry_run=False):
    model, class_names = load_yamnet()

    ser = None
    if not dry_run:
        try:
            ser = serial.Serial(port, BAUD_RATE, timeout=1)
            time.sleep(2)
            print(f"✓ Serial connected: {port} @ {BAUD_RATE}\n")
            while ser.in_waiting:
                print(f"  XIAO: {ser.readline().decode().strip()}")
        except Exception as e:
            print(f"✗ Serial error: {e}")
            print("  Run with --dry-run to test without hardware\n")
            return
    else:
        print("DRY RUN — no serial output\n")

    window_samples = int(WINDOW_SEC * SAMPLE_RATE)
    hop_samples    = int(HOP_SEC    * SAMPLE_RATE)
    ring_buffer    = np.zeros(window_samples, dtype=np.float32)
    lock           = threading.Lock()

    def audio_callback(indata, frames, time_info, status):
        chunk = indata[:, 0]
        with lock:
            nonlocal ring_buffer
            ring_buffer = np.roll(ring_buffer, -len(chunk))
            ring_buffer[-len(chunk):] = chunk

    print("=" * 60)
    print("  Noise Glasses — Live Detection + Amplitude Brightness")
    print("=" * 60)
    print(f"  Threshold  : {threshold}  |  dB range: {MIN_DB} → {MAX_DB}")
    print(f"  Window     : {WINDOW_SEC}s  |  Refresh: {HOP_SEC}s")
    print(f"  Smoothing  : {SMOOTHING}  (edit SMOOTHING in script to tune)")
    print("  Ctrl+C to stop\n")

    if ser:
        ser.write(f"T:{threshold:.2f}\n".encode())
        time.sleep(0.1)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        blocksize=hop_samples, callback=audio_callback):
        try:
            while True:
                time.sleep(HOP_SEC)

                with lock:
                    snapshot = ring_buffer.copy()

                yamnet_name, score, category = classify(model, class_names, snapshot)
                brightness = compute_brightness(snapshot)

                # Always send brightness so LEDs track volume continuously.
                # Also send class when confident enough.
                if score >= threshold and category != "silence":
                    cmd = f"C:{category}:{score:.3f}:{brightness}"
                else:
                    # Below threshold or silence — just update brightness
                    cmd = f"B:{brightness}"

                status_icon = "✓" if score >= threshold else "·"
                bar = brightness_bar(brightness)
                print(f"{status_icon} {yamnet_name:<30} {score:.2f}  "
                      f"vol {bar}  → {category}")

                if ser:
                    ser.write((cmd + "\n").encode())
                    time.sleep(0.05)
                    while ser.in_waiting:
                        reply = ser.readline().decode(errors="replace").strip()
                        if reply:
                            print(f"    XIAO: {reply}")

        except KeyboardInterrupt:
            print("\nStopped.")
            if ser:
                ser.write(b"OFF\n")
                ser.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",       default="",   help="Serial port")
    parser.add_argument("--threshold",  type=float,   default=0.50)
    parser.add_argument("--list-ports", action="store_true")
    parser.add_argument("--dry-run",    action="store_true")
    args = parser.parse_args()

    if args.list_ports:
        list_ports()
        return

    if not args.port and not args.dry_run:
        print("Specify --port or use --dry-run.")
        list_ports()
        return

    run(args.port, args.threshold, args.dry_run)


if __name__ == "__main__":
    main()
