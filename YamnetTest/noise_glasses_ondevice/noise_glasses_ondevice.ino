/*
 * Noise Detection Glasses — Full On-Device Pipeline
 * ==================================================
 * INMP441 → I2S → Log-Mel Spectrogram → YAMNet Backbone → Classifier Head → NeoPixel
 *
 * Wiring:
 *   INMP441 SCK → GPIO7
 *   INMP441 WS  → GPIO8
 *   INMP441 SD  → GPIO9
 *   INMP441 L/R → GND
 *   NeoPixel    → GPIO5
 *
 * Board: XIAO_ESP32S3 | Tools → PSRAM → OPI PSRAM
 */

#include <Arduino.h>
#include <driver/i2s.h>
#include <Adafruit_NeoPixel.h>
#include <LittleFS.h>
#include <math.h>
#include <esp_heap_caps.h>

#define FLATBUFFERS_SPAN_CONSTEXPR
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "embedding_mean.h"
#include "embedding_std.h"

// ── Pins ──────────────────────────────────────────────────────────────────────
#define I2S_SCK    7
#define I2S_WS     8
#define I2S_SD     9
#define LED_PIN    5
#define LED_COUNT  1

// ── Audio params ──────────────────────────────────────────────────────────────
#define SAMPLE_RATE      16000
#define NUM_SAMPLES      16000
#define FRAME_LENGTH     400
#define FRAME_STEP       160
#define FFT_LENGTH       512
#define NUM_MEL_BINS     64
#define NUM_FRAMES       96
#define MEL_LOWER_HZ     125.0f
#define MEL_UPPER_HZ     7500.0f
#define I2S_BUF_LEN      512

// ── TFLite ────────────────────────────────────────────────────────────────────
#define BACKBONE_ARENA_SIZE  (1200 * 1024)
#define HEAD_ARENA_SIZE      (64   * 1024)
#define EMBEDDING_DIM        1024
#define NUM_CLASSES          6

static uint8_t *backbone_buf   = nullptr;
static uint8_t *head_buf       = nullptr;
static uint8_t *backbone_arena = nullptr;
static uint8_t *head_arena     = nullptr;

static tflite::MicroInterpreter *backbone_interp = nullptr;
static tflite::MicroInterpreter *head_interp     = nullptr;

// ── Globals ───────────────────────────────────────────────────────────────────
Adafruit_NeoPixel led(LED_COUNT, LED_PIN, NEO_GRB + NEO_KHZ800);
static int16_t audio_buf[NUM_SAMPLES];
static float   mel_filterbank[NUM_MEL_BINS][FFT_LENGTH / 2 + 1];
static float   hann_window[FRAME_LENGTH];
static float   embedding[EMBEDDING_DIM];

// ── Class names ───────────────────────────────────────────────────────────────
const char* CLASS_NAMES[NUM_CLASSES] = {
    "animal", "appliance", "danger", "music", "speech", "vehicle"
};

// ── Category → color ─────────────────────────────────────────────────────────
void categoryToColor(const char* cat, uint8_t &r, uint8_t &g, uint8_t &b) {
    r = 0; g = 0; b = 0;
    if (strcmp(cat, "danger")    == 0) { r=255; g=15;  b=0;   return; }
    if (strcmp(cat, "speech")    == 0) { r=20;  g=80;  b=255; return; }
    if (strcmp(cat, "vehicle")   == 0) { r=255; g=200; b=0;   return; }
    if (strcmp(cat, "animal")    == 0) { r=255; g=110; b=0;   return; }
    if (strcmp(cat, "appliance") == 0) { r=0;   g=210; b=210; return; }
    if (strcmp(cat, "music")     == 0) { r=200; g=0;   b=255; return; }
}

// ── Hz ↔ Mel ──────────────────────────────────────────────────────────────────
float hzToMel(float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); }
float melToHz(float mel) { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); }

// ── Build mel filterbank ──────────────────────────────────────────────────────
void build_mel_filterbank() {
    int   num_fft_bins = FFT_LENGTH / 2 + 1;
    float mel_low      = hzToMel(MEL_LOWER_HZ);
    float mel_high     = hzToMel(MEL_UPPER_HZ);

    float mel_points[NUM_MEL_BINS + 2];
    for (int i = 0; i < NUM_MEL_BINS + 2; i++)
        mel_points[i] = mel_low + i * (mel_high - mel_low) / (NUM_MEL_BINS + 1);

    float bin_points[NUM_MEL_BINS + 2];
    for (int i = 0; i < NUM_MEL_BINS + 2; i++)
        bin_points[i] = melToHz(mel_points[i]) * FFT_LENGTH / SAMPLE_RATE;

    memset(mel_filterbank, 0, sizeof(mel_filterbank));
    for (int m = 0; m < NUM_MEL_BINS; m++) {
        float left   = bin_points[m];
        float center = bin_points[m + 1];
        float right  = bin_points[m + 2];
        for (int k = 0; k < num_fft_bins; k++) {
            float kf = (float)k;
            if (kf >= left && kf <= center && center > left)
                mel_filterbank[m][k] = (kf - left) / (center - left);
            else if (kf > center && kf <= right && right > center)
                mel_filterbank[m][k] = (right - kf) / (right - center);
        }
    }
}

// ── Build Hann window ─────────────────────────────────────────────────────────
void build_hann_window() {
    for (int i = 0; i < FRAME_LENGTH; i++)
        hann_window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (FRAME_LENGTH - 1)));
}

// ── Cooley-Tukey FFT ──────────────────────────────────────────────────────────
void compute_power_spectrum(const float* frame, float* power, int n_fft) {
    static float re[FFT_LENGTH];
    static float im[FFT_LENGTH];

    memset(re, 0, sizeof(re));
    memset(im, 0, sizeof(im));
    for (int i = 0; i < FRAME_LENGTH; i++)
        re[i] = frame[i] * hann_window[i];

    int N = n_fft;
    for (int s = 1; (1 << s) <= N; s++) {
        int   m    = 1 << s;
        int   m2   = m >> 1;
        float wRe  = 1.0f, wIm = 0.0f;
        float wprRe = cosf(M_PI / m2);
        float wprIm = -sinf(M_PI / m2);
        for (int j = 0; j < m2; j++) {
            for (int k = j; k < N; k += m) {
                int   kk  = k + m2;
                float tRe = wRe * re[kk] - wIm * im[kk];
                float tIm = wRe * im[kk] + wIm * re[kk];
                re[kk] = re[k] - tRe;
                im[kk] = im[k] - tIm;
                re[k] += tRe;
                im[k] += tIm;
            }
            float newWRe = wRe * wprRe - wIm * wprIm;
            wIm = wRe * wprIm + wIm * wprRe;
            wRe = newWRe;
        }
    }

    for (int i = 0; i <= N / 2; i++)
        power[i] = re[i] * re[i] + im[i] * im[i];
}

// ── Compute log-mel patch [NUM_FRAMES × NUM_MEL_BINS] ─────────────────────────
void compute_log_mel(const int16_t* samples, float patch[NUM_FRAMES][NUM_MEL_BINS]) {
    static float power[FFT_LENGTH / 2 + 1];
    static float frame[FRAME_LENGTH];

    for (int t = 0; t < NUM_FRAMES; t++) {
        int start = t * FRAME_STEP;
        for (int i = 0; i < FRAME_LENGTH; i++) {
            int idx = start + i;
            frame[i] = (idx < NUM_SAMPLES) ? samples[idx] / 32768.0f : 0.0f;
        }
        compute_power_spectrum(frame, power, FFT_LENGTH);
        for (int m = 0; m < NUM_MEL_BINS; m++) {
            float val = 0.0f;
            for (int k = 0; k <= FFT_LENGTH / 2; k++)
                val += mel_filterbank[m][k] * power[k];
            patch[t][m] = logf(val + 1e-6f);
        }
    }
}

// ── I2S init ──────────────────────────────────────────────────────────────────
void setup_i2s() {
    i2s_config_t cfg = {
        .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate          = SAMPLE_RATE,
        .bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count        = 8,
        .dma_buf_len          = I2S_BUF_LEN,
        .use_apll             = false,
        .tx_desc_auto_clear   = false,
        .fixed_mclk           = 0
    };
    i2s_pin_config_t pins = {
        .bck_io_num   = I2S_SCK,
        .ws_io_num    = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num  = I2S_SD
    };
    i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &pins);
    i2s_zero_dma_buffer(I2S_NUM_0);
    Serial.println("I2S OK");
}

// ── Load model from LittleFS into PSRAM ──────────────────────────────────────
uint8_t* load_model(const char* path, size_t &size_out) {
    File f = LittleFS.open(path, "r");
    if (!f) {
        Serial.printf("ERROR: cannot open %s\n", path);
        return nullptr;
    }
    size_out = f.size();
    uint8_t* buf = (uint8_t*)heap_caps_malloc(size_out, MALLOC_CAP_SPIRAM);
    if (!buf) {
        Serial.printf("ERROR: PSRAM malloc failed for %s (%u bytes)\n", path, size_out);
        return nullptr;
    }
    f.read(buf, size_out);
    f.close();
    Serial.printf("✓ Loaded %s (%u bytes)\n", path, size_out);
    return buf;
}

// ── TFLite init ───────────────────────────────────────────────────────────────
bool setup_tflite() {
    Serial.printf("Free PSRAM: %u KB\n", ESP.getFreePsram() / 1024);
    Serial.printf("Free heap:  %u KB\n", ESP.getFreeHeap()  / 1024);

    if (!LittleFS.begin(true)) {
        Serial.println("ERROR: LittleFS mount failed");
        return false;
    }

    size_t backbone_size, head_size;
    backbone_buf = load_model("/yamnet_backbone.tflite", backbone_size);
    if (!backbone_buf) return false;

    head_buf = load_model("/glasses_head.tflite", head_size);
    if (!head_buf) return false;

    // Allocate arenas explicitly in PSRAM
    backbone_arena = (uint8_t*)heap_caps_malloc(BACKBONE_ARENA_SIZE, MALLOC_CAP_SPIRAM);
    head_arena     = (uint8_t*)heap_caps_malloc(HEAD_ARENA_SIZE,     MALLOC_CAP_SPIRAM);

    if (!backbone_arena) {
        Serial.printf("ERROR: backbone arena alloc failed (%u KB)\n",
                      BACKBONE_ARENA_SIZE / 1024);
        return false;
    }
    if (!head_arena) {
        Serial.printf("ERROR: head arena alloc failed (%u KB)\n",
                      HEAD_ARENA_SIZE / 1024);
        return false;
    }
    Serial.printf("✓ Arenas allocated — backbone:%uKB head:%uKB\n",
                  BACKBONE_ARENA_SIZE / 1024, HEAD_ARENA_SIZE / 1024);

    // ── Backbone interpreter ──────────────────────────────────────────────────
    static tflite::MicroMutableOpResolver<8> backbone_resolver;
    backbone_resolver.AddConv2D();
    backbone_resolver.AddDepthwiseConv2D();
    backbone_resolver.AddFullyConnected();
    backbone_resolver.AddMean();
    backbone_resolver.AddReshape();
    backbone_resolver.AddQuantize();
    backbone_resolver.AddDequantize();
    backbone_resolver.AddLogistic();

    const tflite::Model* bm = tflite::GetModel(backbone_buf);
    if (bm->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("ERROR: backbone schema mismatch");
        return false;
    }
    static tflite::MicroInterpreter b_interp(
        bm, backbone_resolver, backbone_arena, BACKBONE_ARENA_SIZE, nullptr);
    backbone_interp = &b_interp;

    if (backbone_interp->AllocateTensors() != kTfLiteOk) {
        Serial.printf("ERROR: backbone AllocateTensors failed\n");
        Serial.printf("       arena_used_bytes: %u\n",
                      backbone_interp->arena_used_bytes());
        return false;
    }
    Serial.printf("✓ Backbone OK — arena used: %u KB\n",
                  backbone_interp->arena_used_bytes() / 1024);
    Serial.printf("  input:  [%d,%d,%d,%d] type=%d\n",
        backbone_interp->input(0)->dims->data[0],
        backbone_interp->input(0)->dims->data[1],
        backbone_interp->input(0)->dims->data[2],
        backbone_interp->input(0)->dims->data[3],
        backbone_interp->input(0)->type);
    Serial.printf("  output: [%d,%d] type=%d\n",
        backbone_interp->output(0)->dims->data[0],
        backbone_interp->output(0)->dims->data[1],
        backbone_interp->output(0)->type);

    // ── Head interpreter ──────────────────────────────────────────────────────
    static tflite::MicroMutableOpResolver<5> head_resolver;
    head_resolver.AddFullyConnected();
    head_resolver.AddSoftmax();
    head_resolver.AddQuantize();
    head_resolver.AddDequantize();
    head_resolver.AddReshape();

    const tflite::Model* hm = tflite::GetModel(head_buf);
    if (hm->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("ERROR: head schema mismatch");
        return false;
    }
    static tflite::MicroInterpreter h_interp(
        hm, head_resolver, head_arena, HEAD_ARENA_SIZE, nullptr);
    head_interp = &h_interp;

    if (head_interp->AllocateTensors() != kTfLiteOk) {
        Serial.printf("ERROR: head AllocateTensors failed\n");
        Serial.printf("       arena_used_bytes: %u\n",
                      head_interp->arena_used_bytes());
        return false;
    }
    Serial.printf("✓ Head OK — arena used: %u KB\n",
                  head_interp->arena_used_bytes() / 1024);
    Serial.printf("  input:  [%d,%d] type=%d\n",
        head_interp->input(0)->dims->data[0],
        head_interp->input(0)->dims->data[1],
        head_interp->input(0)->type);
    Serial.printf("  output: [%d,%d] type=%d\n",
        head_interp->output(0)->dims->data[0],
        head_interp->output(0)->dims->data[1],
        head_interp->output(0)->type);

    Serial.println("TFLite OK");
    return true;
}

// ── Capture 1s of audio ───────────────────────────────────────────────────────
void capture_audio() {
    int32_t raw[I2S_BUF_LEN];
    int captured = 0;
    while (captured < NUM_SAMPLES) {
        size_t bytes_read = 0;
        int to_read = min(NUM_SAMPLES - captured, I2S_BUF_LEN);
        i2s_read(I2S_NUM_0, raw, to_read * sizeof(int32_t),
                 &bytes_read, portMAX_DELAY);
        int got = bytes_read / sizeof(int32_t);
        for (int i = 0; i < got && captured < NUM_SAMPLES; i++)
            audio_buf[captured++] = (int16_t)(raw[i] >> 14);
    }
}

// ── Backbone: patch → embedding ───────────────────────────────────────────────
bool run_backbone(float patch[NUM_FRAMES][NUM_MEL_BINS]) {
    TfLiteTensor* inp = backbone_interp->input(0);
    float* inp_data   = inp->data.f;
    for (int t = 0; t < NUM_FRAMES; t++)
        for (int m = 0; m < NUM_MEL_BINS; m++)
            inp_data[t * NUM_MEL_BINS + m] = patch[t][m];

    if (backbone_interp->Invoke() != kTfLiteOk) {
        Serial.println("Backbone invoke failed");
        return false;
    }
    memcpy(embedding, backbone_interp->output(0)->data.f,
           EMBEDDING_DIM * sizeof(float));

    // DEBUG: raw embedding stats BEFORE normalization
    float emb_min = embedding[0], emb_max = embedding[0], emb_sum = 0;
    int   nonzero = 0;
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        if (embedding[i] < emb_min) emb_min = embedding[i];
        if (embedding[i] > emb_max) emb_max = embedding[i];
        emb_sum += embedding[i];
        if (embedding[i] != 0.0f) nonzero++;
    }
    Serial.printf("Raw embedding: min=%.4f max=%.4f mean=%.4f nonzero=%d/1024\n",
                  emb_min, emb_max, emb_sum / EMBEDDING_DIM, nonzero);

    // DEBUG: patch stats (is spectrogram reasonable?)
    float p_min = patch[0][0], p_max = patch[0][0], p_sum = 0;
    for (int t = 0; t < NUM_FRAMES; t++)
        for (int m = 0; m < NUM_MEL_BINS; m++) {
            float v = patch[t][m];
            if (v < p_min) p_min = v;
            if (v > p_max) p_max = v;
            p_sum += v;
        }
    Serial.printf("Patch stats:   min=%.4f max=%.4f mean=%.4f\n",
                  p_min, p_max, p_sum / (NUM_FRAMES * NUM_MEL_BINS));

    return true;
}

// ── Normalize embedding ───────────────────────────────────────────────────────
void normalize_embedding() {
    for (int i = 0; i < EMBEDDING_DIM; i++)
        embedding[i] = (embedding[i] - EMBEDDING_MEAN[i]) / EMBEDDING_STD[i];
}

// ── Head: embedding → class ───────────────────────────────────────────────────
const char* run_head() {
    TfLiteTensor* inp = head_interp->input(0);
    memcpy(inp->data.f, embedding, EMBEDDING_DIM * sizeof(float));

    if (head_interp->Invoke() != kTfLiteOk) {
        Serial.println("Head invoke failed");
        return "unknown";
    }

    float* scores = head_interp->output(0)->data.f;
    int best = 0;
    for (int i = 1; i < NUM_CLASSES; i++)
        if (scores[i] > scores[best]) best = i;

    Serial.printf("→ %s (%.2f)  [", CLASS_NAMES[best], scores[best]);
    for (int i = 0; i < NUM_CLASSES; i++)
        Serial.printf("%s:%.2f ", CLASS_NAMES[i], scores[i]);
    Serial.println("]");

    return CLASS_NAMES[best];
}

// ── setup ─────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("=== Noise Glasses On-Device ===");

    led.begin();
    led.setBrightness(180);
    led.setPixelColor(0, led.Color(0, 80, 255));
    led.show();
    delay(400);
    led.clear();
    led.show();

    build_hann_window();
    build_mel_filterbank();
    Serial.println("Mel filterbank OK");

    setup_i2s();

    if (!setup_tflite()) {
        while (true) {
            led.setPixelColor(0, led.Color(255, 0, 0)); led.show(); delay(200);
            led.clear(); led.show(); delay(200);
        }
    }
    // ── TEMP: test head with a known embedding ────────────────────────────────
    Serial.println("Testing head with zero embedding...");
    memset(embedding, 0, sizeof(embedding));
    normalize_embedding();
    const char* test_cat = run_head();
    Serial.printf("Zero embedding → %s\n", test_cat);

    // Test with random-ish embedding (all 1.0 before norm)
    for (int i = 0; i < EMBEDDING_DIM; i++) embedding[i] = 1.0f;
    normalize_embedding();
    test_cat = run_head();
    Serial.printf("Ones embedding → %s\n", test_cat);

    Serial.println("Listening...");
}

// ── loop ──────────────────────────────────────────────────────────────────────
static float patch[NUM_FRAMES][NUM_MEL_BINS];

void loop() {
    capture_audio();
    compute_log_mel(audio_buf, patch);

    if (!run_backbone(patch)) return;
    normalize_embedding();
    const char* cat = run_head();

    uint8_t r, g, b;
    categoryToColor(cat, r, g, b);
    if (r == 0 && g == 0 && b == 0)
        led.clear();
    else
        led.setPixelColor(0, led.Color(r, g, b));
    led.show();

    delay(50);
}