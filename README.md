# Noise-Detection Glasses

A wearable assistive device for deaf / hard-of-hearing users that classifies ambient sounds in real time, locates roughly where each sound is coming from, and signals the wearer through NeoPixel LEDs and a haptic motor on a glasses frame. The device runs **fully on-device** on a **XIAO ESP32-S3** with an Edge Impulse (YAMNet-derived) model — no laptop required. A separate laptop-side Python suite remains for prototyping, accuracy testing, and dataset work.

---

## How it works (on-device)

The shipping firmware is `glasses_system/glasses_system.ino`. Each cycle it:

1. **Classifies** the sound into one of four classes — `alert`, `music`, `speech`, `vehicle` — using a continuous Edge Impulse model (~2 s window, refreshed every ~0.5 s slice).
2. **Locates** the sound using four PDM mics arranged in two stereo pairs (front + back). It resolves a horizontal direction (LEFT / MIDDLE / RIGHT) and whether the sound is in FRONT or BEHIND.
3. **Signals** the wearer:
   - One of **3 NeoPixels** lights up — its **position** = horizontal direction, its **color** = sound class, its **brightness** = loudness.
   - A **haptic motor** buzzes when sound comes from BEHIND, with strength scaled to rear loudness.

---

## Project overview

| Layer | Technology |
|---|---|
| Sound classification | Edge Impulse model (`Yamnet-515-Group_inferencing.h`), YAMNet-derived, 4 classes |
| Edge hardware | XIAO ESP32-S3 (8 MB flash + 8 MB PSRAM) |
| Microphones | 4× T5838 PDM MEMS (two stereo pairs, front + back) over I2S PDM @ 16 kHz |
| Visual output | 3× NeoPixel (GRBW) — direction + class + loudness |
| Haptic output | Vibration motor — rear-sound alert |
| Laptop suite | Python → YAMNet for prototyping, accuracy testing, dataset prep |

---

## Sound → LED color scheme

The on-device model emits four classes, mapped to NeoPixel colors in `CLASS_COLOR[]`:

| Class | LED color | Rationale |
|---|---|---|
| `alert` | Red | Danger / urgent (alarms, sirens) |
| `music` | Purple | Creative / non-urgent |
| `speech` | Blue | Calm, "someone talking" |
| `vehicle` | Amber | Environmental awareness |
| (silence / low confidence) | OFF | No distraction |

> The laptop-side scripts (`yamnet_serial.py`, `yamnet_mic_test.py`) use a richer 8-category map over the full 521 YAMNet classes; that broader scheme is documented in their source and in `CLAUDE.md`.

---

## Repository layout

```
DetectionGlasses/
├── glasses_system/
│   └── glasses_system.ino         # ★ On-device firmware: EI model + 4-mic direction + LEDs + haptic
└── YamnetTest/                     # Laptop-side prototyping, accuracy testing & dataset tooling
    ├── yamnet_mic_test.py          # Live mic classification on laptop (no hardware needed)
    ├── yamnet_esp32_feasibility.py # Convert YAMNet → TFLite + ESP32-S3 feasibility report
    ├── yamnet_serial.py            # Laptop → USB serial → XIAO NeoPixel bridge
    ├── yamnet_accuracy_test.py     # float32 vs int8 accuracy on ESC-50 dataset
    ├── yamnet_urbansound_test.py   # Accuracy on UrbanSound8K via soundata
    ├── neopixel_noise_test/
    │   └── neopixel_noise_test.ino # Standalone NeoPixel LED bring-up sketch
    ├── noise_glasses_ondevice/     # Earlier on-device experiment (embeddings + classifier)
    └── tflite_models/              # Generated models (gitignored)
        ├── yamnet_full.tflite      # ~3.7 MB float32
        ├── yamnet_int8.tflite      # ~0.9 MB int8 quantized (ESP32 target)
        └── yamnet_saved_model/     # Intermediate SavedModel export
```

---

## Laptop-side scripts

These Python tools are for prototyping, accuracy validation, and dataset work — they are **not** part of the on-device runtime, which is self-contained in `glasses_system.ino`.

### 1. Live mic test (`yamnet_mic_test.py`)

Runs YAMNet on your laptop mic and prints top-5 predictions every 0.5 s with LED color suggestions. Good for quick demos without hardware.

```bash
pip install tensorflow tensorflow-hub sounddevice numpy scipy
python YamnetTest/yamnet_mic_test.py
```

### 2. ESP32 feasibility + TFLite conversion (`yamnet_esp32_feasibility.py`)

Downloads YAMNet, converts to float32 and int8 TFLite, runs smoke-test inference, and prints an ESP32-S3 hardware feasibility report. Run this once to generate the model files.

```bash
pip install tensorflow tensorflow-hub numpy
python YamnetTest/yamnet_esp32_feasibility.py
```

Key findings:
- **Flash**: int8 model (~0.9 MB) fits easily in 8 MB flash
- **RAM**: YAMNet activations need ~300–400 KB — use an ESP32-S3 with 8 MB PSRAM
- Recommended board: **ESP32-S3-DevKitC-1-N8R8** (~$10)

### 3. Serial bridge (`yamnet_serial.py`)

Runs YAMNet on the laptop mic, maps predictions to categories, and sends commands over USB serial to the XIAO ESP32-S3. Brightness tracks live RMS amplitude.

```bash
pip install tensorflow tensorflow-hub sounddevice numpy pyserial

# List available serial ports
python YamnetTest/yamnet_serial.py --list-ports

# Run with hardware
python YamnetTest/yamnet_serial.py --port /dev/cu.usbmodem1101

# Test without hardware
python YamnetTest/yamnet_serial.py --port dummy --dry-run
```

Serial protocol:
```
C:speech:0.92:180    # class, confidence, brightness (0-255)
B:180                # brightness-only update
T:0.50               # set confidence threshold
OFF                  # turn LEDs off
```

### 4. Accuracy tests

**ESC-50 dataset:**
```bash
pip install tensorflow tensorflow-hub numpy scipy soundfile librosa tqdm
python YamnetTest/yamnet_accuracy_test.py --esc50-dir YamnetTest/downloads/ESC-50-master
```

**UrbanSound8K (auto-downloads):**
```bash
pip install soundata tensorflow tensorflow-hub numpy librosa tqdm
python YamnetTest/yamnet_urbansound_test.py
python YamnetTest/yamnet_urbansound_test.py --limit 200   # faster subset
```

---

## Firmware (`glasses_system/glasses_system.ino`)

The on-device sketch handles everything: PDM mic capture (on a dedicated FreeRTOS task), continuous Edge Impulse inference, direction estimation, and driving the LEDs + haptic motor. No serial bridge or laptop is needed at runtime.

### Build & flash

1. Install the Edge Impulse Arduino library exported for this project — it provides `Yamnet-515-Group_inferencing.h`.
2. Install **Adafruit NeoPixel** via the Arduino Library Manager. `ESP_I2S` ships with the ESP32 Arduino core (3.x).
3. Open `glasses_system/glasses_system.ino`, set the Arduino IDE board options below, and upload.

Tunable `#define`s live in the **USER CONFIG** block at the top of the sketch — pins, thresholds, LED brightness, haptic strength, audio gain. Set `AUDIO_DIAG 1` to stream per-mic peak-to-peak levels and the current class over Serial @ 115200 for bring-up.

### Pinout

| Signal | GPIO | Notes |
|---|---|---|
| PDM clock (shared) | GPIO1 (D0) | drives all 4 mics |
| PDM data — back pair | GPIO2 (D1) | back-left + back-right |
| PDM data — front pair | GPIO6 (D5) | front pair, L/R swapped in HW (corrected in software) |
| PDM THSEL (shared) | GPIO4 | T5838 mics; AAD unused (normal PDM streaming) |
| NeoPixel data | GPIO43 (D6) | 3 pixels, GRBW |
| Haptic motor | GPIO5 | PWM, active-high |

Mics are **T5838 PDM MEMS** in two stereo pairs. Because the front pair is mirrored relative to the back pair, the firmware pairs same-side channels `(backL, frontR)` and `(backR, frontL)` to separate left from right. If left/right come out swapped during testing, swap the two return lines in `horizontalPixel()`.

For the NeoPixel power line, a **300–500 Ω resistor** on data and a **1000 µF cap** across power/ground improve stability.

### Arduino IDE settings

| Setting | Value |
|---|---|
| Board | ESP32S3 Dev Module |
| Flash size | 8 MB |
| PSRAM | OPI PSRAM |
| USB Mode | Hardware CDC and JTAG |

---

## Roadmap

- [x] On-device Edge Impulse inference on the XIAO ESP32-S3 (no laptop at runtime)
- [x] 4-mic directional sensing (LEFT / MIDDLE / RIGHT, FRONT / BACK)
- [x] 3-pixel display: position = direction, color = class, brightness = loudness
- [x] Haptic feedback for sounds coming from behind
- [ ] Calibrate `SOUND_THRESHOLD`, `LR_THRESHOLD`, `FB_THRESHOLD`, and `LOUDNESS_MAX` on the assembled frame
- [ ] Expand/retrain the model beyond the four current classes if needed
- [ ] Final mechanical integration into the glasses temple/brow

---

## Requirements summary

| Script | Key packages |
|---|---|
| `yamnet_mic_test.py` | tensorflow, tensorflow-hub, sounddevice, scipy |
| `yamnet_esp32_feasibility.py` | tensorflow, tensorflow-hub |
| `yamnet_serial.py` | tensorflow, tensorflow-hub, sounddevice, pyserial |
| `yamnet_accuracy_test.py` | tensorflow, tensorflow-hub, librosa, soundfile, tqdm |
| `yamnet_urbansound_test.py` | tensorflow, tensorflow-hub, librosa, soundata, tqdm |
