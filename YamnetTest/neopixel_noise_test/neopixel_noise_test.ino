/*
 * Noise Detection Glasses — NeoPixel Serial Receiver
 * ====================================================
 * Receives classification commands from the laptop (yamnet_serial.py)
 * over USB-Serial and drives a NeoPixel strip on the glasses frame.
 *
 * Serial protocol (115200 baud, newline-terminated):
 *   C:<category>:<confidence>:<brightness>  — new sound class detected
 *   B:<brightness>                           — volume update, keep color
 *   T:<threshold>                            — set confidence threshold
 *   OFF                                      — silence, turn LEDs off
 *
 * Categories and their LED colors:
 *   danger      → RED         (fire alarm, siren, gunshot, scream)
 *   speech      → BLUE        (conversation, baby cry, laughter)
 *   vehicle     → YELLOW      (car horn, traffic, motorcycle)
 *   door_entry  → GREEN       (doorbell, knock, slam)
 *   phone       → PURPLE      (ringtone, cell phone)
 *   animal      → ORANGE      (dog bark, cat)
 *   appliance   → CYAN        (microwave, washer)
 *   music       → WARM WHITE  (music, TV, radio)
 *   silence     → OFF
 *
 * Board: XIAO ESP32-S3
 * LED pin: GPIO 48 (change LED_PIN if wired differently)
 *
 * Install: Adafruit NeoPixel library (Library Manager)
 */

#include <Adafruit_NeoPixel.h>

#define LED_PIN    48
#define LED_COUNT  8      // adjust to your strip length
#define BAUD_RATE  115200

Adafruit_NeoPixel strip(LED_COUNT, LED_PIN, NEO_GRB + NEO_KHZ800);

// ── Category → RGB ────────────────────────────────────────────────────────────
struct RGB { uint8_t r, g, b; };

RGB categoryToRGB(const String &cat) {
    if (cat == "danger")     return {255,  15,   0};   // red
    if (cat == "speech")     return { 20,  80, 255};   // blue
    if (cat == "vehicle")    return {255, 200,   0};   // yellow
    if (cat == "door_entry") return {  0, 220,  60};   // green
    if (cat == "phone")      return {180,   0, 255};   // purple
    if (cat == "animal")     return {255, 110,   0};   // orange
    if (cat == "appliance")  return {  0, 210, 210};   // cyan
    if (cat == "music")      return {200, 200, 140};   // warm white
    return {60, 60, 60};                               // dim gray for unknown
}

// ── State ─────────────────────────────────────────────────────────────────────
String  currentCategory  = "";
uint8_t currentBrightness = 0;
String  inputBuffer      = "";

// ── Helpers ───────────────────────────────────────────────────────────────────
void setStrip(RGB c, uint8_t brightness) {
    float s = brightness / 255.0f;
    uint32_t color = strip.Color(
        (uint8_t)(c.r * s),
        (uint8_t)(c.g * s),
        (uint8_t)(c.b * s)
    );
    for (int i = 0; i < LED_COUNT; i++) {
        strip.setPixelColor(i, color);
    }
    strip.show();
}

void applyCurrentState() {
    if (currentCategory.length() == 0 || currentCategory == "silence") {
        strip.clear();
        strip.show();
    } else {
        setStrip(categoryToRGB(currentCategory), currentBrightness);
    }
}

// ── Command dispatcher ────────────────────────────────────────────────────────
void handleCommand(const String &line) {

    if (line == "OFF") {
        strip.clear();
        strip.show();
        currentCategory = "";
        Serial.println("OK:OFF");
        return;
    }

    if (line.startsWith("T:")) {
        // Threshold is stored on the laptop side; just acknowledge it.
        float t = line.substring(2).toFloat();
        Serial.print("OK:T:");
        Serial.println(t, 2);
        return;
    }

    if (line.startsWith("B:")) {
        // Brightness-only update — keep current category color.
        currentBrightness = (uint8_t)constrain(line.substring(2).toInt(), 0, 255);
        applyCurrentState();
        return;
    }

    if (line.startsWith("C:")) {
        // C:<category>:<confidence>:<brightness>
        int p1 = line.indexOf(':', 2);          // after "C:"
        int p2 = line.indexOf(':', p1 + 1);     // after category
        if (p1 < 0 || p2 < 0) {
            Serial.println("ERR:bad_C_format");
            return;
        }
        currentCategory   = line.substring(2, p1);
        float confidence  = line.substring(p1 + 1, p2).toFloat();
        currentBrightness = (uint8_t)constrain(line.substring(p2 + 1).toInt(), 0, 255);
        applyCurrentState();
        Serial.print("OK:C:");
        Serial.print(currentCategory);
        Serial.print(":");
        Serial.println(confidence, 3);
        return;
    }

    Serial.print("ERR:unknown:");
    Serial.println(line);
}

// ── Arduino entry points ──────────────────────────────────────────────────────
void setup() {
    Serial.begin(BAUD_RATE);
    strip.begin();
    strip.setBrightness(255);
    strip.clear();
    strip.show();

    // Brief startup flash so you know the sketch is running
    for (int i = 0; i < LED_COUNT; i++) {
        strip.setPixelColor(i, strip.Color(0, 80, 255));
    }
    strip.show();
    delay(300);
    strip.clear();
    strip.show();

    Serial.println("NoisyGlasses:ready");
}

void loop() {
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\n') {
            inputBuffer.trim();
            if (inputBuffer.length() > 0) {
                handleCommand(inputBuffer);
            }
            inputBuffer = "";
        } else if (c != '\r') {
            inputBuffer += c;
        }
    }
}
