# Noise-Detection Glasses

A wearable assistive device that classifies ambient sounds in real time and signals the wearer through NeoPixel LEDs mounted on a glasses frame. Built around a **XIAO ESP32-S3** running a quantized YAMNet model, with a laptop-side pipeline for prototyping and accuracy testing.

---

## Project overview

| Layer | Technology |
|---|---|
| Sound classification | [YAMNet](https://tfhub.dev/google/yamnet/1) (Google, 521 AudioSet classes) |
| Edge hardware | XIAO ESP32-S3 (8 MB flash + 8 MB PSRAM) |
| Microphone | I2S MEMS (e.g. INMP441) |
| Output | NeoPixel LED strip along glasses frame |
| Laptop bridge | Python → USB serial → ESP32 (for prototyping) |

---

## Sound → LED color scheme

| Sound category | LED color | Rationale |
|---|---|---|
| Silence | OFF | No distraction |
| Speech / Voice | Blue | Calm, "someone talking" |
| Music | Purple | Creative / non-urgent |
| Alarm / Siren | Red | Danger / urgent |
| Animal sounds | Yellow | Attention, non-urgent |
| Door / Knock | Orange | Action needed |
| Horn / Beeping | Orange | Action needed |
| Vehicle | White | Environmental awareness |
| Dog barking | Yellow | Attention |
| Phone ringing | Purple | Attention |
| Other | Green | Generic awareness |

---

## Repository layout

```
DetectionGlasses/
└── YamnetTest/
    ├── yamnet_mic_test.py          # Live mic classification on laptop (no hardware needed)
    ├── yamnet_esp32_feasibility.py # Convert YAMNet → TFLite + ESP32-S3 feasibility report
    ├── yamnet_serial.py            # Laptop → USB serial → XIAO NeoPixel bridge
    ├── yamnet_accuracy_test.py     # float32 vs int8 accuracy on ESC-50 dataset
    ├── yamnet_urbansound_test.py   # Accuracy on UrbanSound8K via soundata
    ├── neopixel_noise_test/
    │   └── neopixel_noise_test.ino # Arduino sketch for XIAO NeoPixel LED control
    └── tflite_models/              # Generated models (gitignored)
        ├── yamnet_full.tflite      # ~3.7 MB float32
        ├── yamnet_int8.tflite      # ~0.9 MB int8 quantized (ESP32 target)
        └── yamnet_saved_model/     # Intermediate SavedModel export
```

---

## Scripts

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

## Hardware setup

### XIAO ESP32-S3 + NeoPixel wiring

```
NeoPixel DIN  →  GPIO 48  (set LED_PIN in .ino if using another pin)
NeoPixel 5V   →  5V rail
NeoPixel GND  →  GND
```

Add a **300–500 Ω resistor** on the data line and a **1000 µF cap** across power/ground for stability.

### Arduino IDE settings

| Setting | Value |
|---|---|
| Board | ESP32S3 Dev Module |
| Flash size | 8 MB |
| PSRAM | OPI PSRAM |
| USB Mode | Hardware CDC and JTAG |

Install the **Adafruit NeoPixel** library via Arduino Library Manager.

---

## Roadmap

1. Add I2S MEMS microphone (INMP441) to the ESP32-S3
2. Port `yamnet_int8.tflite` using ESP-IDF + TFLite Micro (or Arduino TFLite library)
3. Replace the demo loop in the `.ino` with real `applyClassification(classIndex)` calls
4. Mount on glasses frame with flexible NeoPixel strip along the temple/brow

---

## Requirements summary

| Script | Key packages |
|---|---|
| `yamnet_mic_test.py` | tensorflow, tensorflow-hub, sounddevice, scipy |
| `yamnet_esp32_feasibility.py` | tensorflow, tensorflow-hub |
| `yamnet_serial.py` | tensorflow, tensorflow-hub, sounddevice, pyserial |
| `yamnet_accuracy_test.py` | tensorflow, tensorflow-hub, librosa, soundfile, tqdm |
| `yamnet_urbansound_test.py` | tensorflow, tensorflow-hub, librosa, soundata, tqdm |
