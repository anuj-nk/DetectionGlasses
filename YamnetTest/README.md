# Noise-Detection Glasses — ML & Hardware Experiments

Three self-contained experiments to validate the tech stack
before building the full glasses prototype.

---

## Experiment 1 — YAMNet Mic Test (`yamnet_mic_test.py`)

Runs YAMNet live on your **laptop microphone**, prints the top-5
predicted noise classes every 0.5 s, and maps each prediction to
a glasses LED color.

### Setup
```bash
pip install tensorflow tensorflow-hub sounddevice numpy scipy
python yamnet_mic_test.py
```

### What to try
| Sound to make | Expected top class | Expected LED |
|---|---|---|
| Silence | Silence | ⚫ OFF |
| Speak normally | Speech | 🔵 BLUE |
| Play music from phone | Music | 🟣 PURPLE |
| Knock on desk | Knock | 🟠 ORANGE |
| Whistle / tone | Sine wave / Whistle | 🟢 GREEN |
| Car video on YouTube | Vehicle / Car | ⚪ WHITE |

---

## Experiment 2 — ESP32-S3 Feasibility (`yamnet_esp32_feasibility.py`)

Downloads YAMNet, converts it to TFLite (float32 + int8 quantized),
runs a smoke-test inference, and prints a hardware feasibility report.

### Setup
```bash
pip install tensorflow tensorflow-hub numpy
python yamnet_esp32_feasibility.py
```

### Output files (written to `tflite_models/`)
| File | Size (approx) | Use |
|---|---|---|
| `yamnet_full.tflite` | ~3.7 MB | Debug / PC |
| `yamnet_int8.tflite` | ~0.9 MB | **ESP32-S3 target** |

### Key findings
- **Flash**: int8 model (~0.9 MB) easily fits in 8 MB flash ✓
- **RAM**: YAMNet activations need ~300–400 KB — use an **ESP32-S3 with 8 MB PSRAM**
- Recommended board: **ESP32-S3-DevKitC-1-N8R8** (8 MB flash + 8 MB PSRAM, ~$10)

---

## Experiment 3 — NeoPixel Test (`neopixel_noise_test/neopixel_noise_test.ino`)

Arduino sketch for the ESP32-S3 that cycles through all noise-category
colors with smooth fades. No ML needed yet — pure hardware validation.

### Wiring
```
NeoPixel DIN  →  GPIO 48 (or any free GPIO, set LED_PIN)
NeoPixel 5V   →  5V  (or USB 5V rail)
NeoPixel GND  →  GND
```
Add a 300–500 Ω resistor on the data line and a 1000 µF cap across
power/ground for stability.

### Arduino IDE settings
| Setting | Value |
|---|---|
| Board | ESP32S3 Dev Module |
| Flash size | 8MB |
| PSRAM | OPI PSRAM |
| USB Mode | Hardware CDC and JTAG |

### Library to install
`Adafruit NeoPixel` — via Arduino Library Manager

### What you'll see
The strip fades through: **OFF → Blue → Purple → Red → Yellow → Orange → White → Green**,
pausing 2 s on each color, then a rainbow sweep as a heartbeat.

---

## Next steps (full prototype)

1. Add an **I2S MEMS microphone** (e.g. INMP441) to the ESP32-S3
2. Port `yamnet_int8.tflite` using **ESP32 Arduino TFLite library** or **ESP-IDF + TFLite Micro**
3. Replace the demo loop in the `.ino` with `applyClassification(classIndex)` (stub already in the sketch)
4. Mount on glasses frame with flexible NeoPixel strip along the temple/brow

---

## Color scheme reference

| Sound category | LED color | Rationale |
|---|---|---|
| Silence | ⚫ OFF | No distraction |
| Speech / Voice | 🔵 Blue | Calm, "someone talking" |
| Music | 🟣 Purple | Creative / non-urgent |
| Alarm / Siren | 🔴 Red | Danger / urgent |
| Animal sounds | 🟡 Yellow | Attention, non-urgent |
| Door / Knock | 🟠 Orange | Action needed |
| Vehicle | ⚪ White | Environmental awareness |
| Other | 🟢 Green | Generic awareness |
