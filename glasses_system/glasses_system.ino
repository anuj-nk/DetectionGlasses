/* ============================================================================
 * glasses_system.ino - ESP32-S3 (arduino-esp32 3.x)
 * Noise-detection glasses for deaf / hard-of-hearing users.
 *
 * FOUR PDM mics, two stereo pairs:
 *   GPIO2 back line  -> back-left + back-right
 *   GPIO6 front line -> front pair (reads mirrored vs back; handled in SW)
 *   GPIO1 shared clock, GPIO4 shared THSEL
 *
 * Behavior:
 *   - 4 mics give direction: LEFT / MIDDLE / RIGHT, plus FRONT vs BACK
 *   - 3 NeoPixels: the lit pixel = horizontal direction
 *   - LED color   = sound class from the model (white if model is unsure)
 *   - LED brightness scales with loudness
 *   - Haptic motor buzzes for sounds coming from BEHIND
 *
 * Robustness model: the direction display fires for any loud directional
 * sound (gated by SOUND_THRESHOLD), independent of the classifier. The class
 * only sets the color. This keeps the reliable part (direction) working even
 * when the model is unsure.
 * ==========================================================================*/

#define EIDSP_QUANTIZE_FILTERBANK   0

/* ====================== USER CONFIG ====================== */

/* PDM pins (XIAO silkscreen Dx != GPIO) */
#define PDM_CLK_PIN     1    // D0 = GPIO1
#define PDM_DATA_BACK   2    // D1 = GPIO2
#define PDM_DATA_FRONT  6    // D5 = GPIO6

/* Raw 4-sample frame = [back_s0, back_s1, front_s0, front_s1].
 * Verified with mic_test.ino. */
#define CH_BACK_L    0
#define CH_BACK_R    1
#define CH_FRONT_R   2
#define CH_FRONT_L   3

/* Outputs */
#define HAPTIC_PIN        5
#define HAPTIC_ACTIVE_HIGH true
#define HAPTIC_MIN_INTENSITY 70     // weakest motor PWM at threshold
#define HAPTIC_MAX_INTENSITY 255    // strongest motor PWM for loud rear sounds
#define HAPTIC_LOUDNESS_MAX 2600    // rear p2p that maps to max vibration; tune

#define NEOPIXEL_PIN       43       // LED_DATA = GPIO43 (D6)
#define NEOPIXEL_COUNT     3        // left, middle, right
#define LED_LEFT_IDX       0        // which pixel is physically left
#define LED_MIDDLE_IDX     1
#define LED_RIGHT_IDX      2
#define NEOPIXEL_BRIGHTNESS 90      // max LED brightness when loud
#define NEOPIXEL_MIN_BRIGHTNESS 8   // dimmest visible brightness at threshold
#define LOUDNESS_MAX       2600     // p2p that maps to max LED brightness; tune

/* Audio / EI */
#define AUDIO_GAIN        8         // gain on the mono stream feeding the model

/* Direction tuning */
#define SOUND_THRESHOLD   1300      // total p2p to count as a real sound (gates room ambient)
#define LR_THRESHOLD      0.10f     // left/right bias to pick a side vs middle
#define FB_THRESHOLD      0.12f     // front/back bias to trigger back-buzz

/* Display behavior */
#define CONF_THRESHOLD    0.60f     // class confidence to trust the color (else white)
#define HOLD_MS           700       // keep a direction lit this long after sound drops

/* Diagnostics */
#define AUDIO_DIAG        1

/* Calibrations */
/* Per-channel calibration: corrects fixed sensitivity imbalance so a centered
 * sound reads equal on all four. Right side (bL, fR) reads ~20-25% low, boost it. */
#define CAL_BACK_L    1.28f   // right-back  (was reading low)
#define CAL_BACK_R    1.00f
#define CAL_FRONT_L   1.00f
#define CAL_FRONT_R   1.30f   // right-front (your weak one)

/* ====================== INCLUDES ====================== */
#include <Yamnet-515-Group_inferencing.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <ESP_I2S.h>
#include <Adafruit_NeoPixel.h>

I2SClass I2S;
Adafruit_NeoPixel strip(NEOPIXEL_COUNT, NEOPIXEL_PIN, NEO_GRB + NEO_KHZ800);

/* Class colors, index order: alert, music, speech, vehicle */
struct RGB { uint8_t r, g, b; };
static const RGB CLASS_COLOR[4] = {
    {255,   0,   0},   // alert   -> red
    {150,   0, 220},   // music   -> purple
    {  0, 100, 255},   // speech  -> blue
    {255, 140,   0},   // vehicle -> amber
};

/* ====================== EI RING BUFFER ====================== */
typedef struct {
    signed short *buffers[2];
    unsigned char buf_select, buf_ready;
    unsigned int  buf_count, n_samples;
} inference_t;
static inference_t inference;
static bool  record_status = true;
static int   print_results = -(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW);
static bool  debug_nn = false;

/* ====================== SHARED DIRECTION STATE ====================== */
static volatile float g_pp_backL  = 0;
static volatile float g_pp_backR  = 0;
static volatile float g_pp_frontL = 0;
static volatile float g_pp_frontR = 0;

/* Last classification result, refreshed each full model window */
static int   lastTop  = -1;
static float lastTopV = 0;

/* ====================== HAPTIC (non-blocking) ====================== */
static const uint16_t PAT_BACK[] = {200, 90, 200, 600};
static const uint16_t *hapPattern = nullptr;
static uint8_t hapLen = 0, hapStep = 0, hapIntensity = 0;
static uint32_t hapStepStart = 0; static bool hapOn = false;

void hapticWrite(uint8_t d){ if(!HAPTIC_ACTIVE_HIGH) d=255-d; analogWrite(HAPTIC_PIN,d); }
void hapticStart(const uint16_t*p,uint8_t n,uint8_t in){ hapPattern=p;hapLen=n;hapIntensity=in;hapStep=0;hapStepStart=millis();hapOn=true;hapticWrite(in);}
void hapticStop(){ hapPattern=nullptr;hapOn=false;hapticWrite(0);}
void hapticUpdate(){ if(!hapPattern)return; if(millis()-hapStepStart<hapPattern[hapStep])return; if(++hapStep>=hapLen){hapticStop();return;} hapStepStart=millis(); hapOn=!hapOn; hapticWrite(hapOn?hapIntensity:0); }

/* ====================== LED ====================== */
uint8_t brightnessFromLoudness(float loudness) {
    if (loudness <= SOUND_THRESHOLD) return NEOPIXEL_MIN_BRIGHTNESS;
    if (loudness > LOUDNESS_MAX) loudness = LOUDNESS_MAX;
    float t = (loudness - SOUND_THRESHOLD) / (float)(LOUDNESS_MAX - SOUND_THRESHOLD);
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    return (uint8_t)(NEOPIXEL_MIN_BRIGHTNESS + t * (NEOPIXEL_BRIGHTNESS - NEOPIXEL_MIN_BRIGHTNESS));
}

uint8_t hapticIntensityFromLoudness(float loudness) {
    if (loudness <= SOUND_THRESHOLD) return 0;
    if (loudness > HAPTIC_LOUDNESS_MAX) loudness = HAPTIC_LOUDNESS_MAX;
    float t = (loudness - SOUND_THRESHOLD) / (float)(HAPTIC_LOUDNESS_MAX - SOUND_THRESHOLD);
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    return (uint8_t)(HAPTIC_MIN_INTENSITY + t * (HAPTIC_MAX_INTENSITY - HAPTIC_MIN_INTENSITY));
}

void ledShow(int pixel, RGB c, uint8_t brightness) {
    strip.clear();
    strip.setBrightness(brightness);
    if (pixel >= 0 && pixel < NEOPIXEL_COUNT)
        strip.setPixelColor(pixel, strip.Color(c.r, c.g, c.b));
    strip.show();
}
void ledClear(){ strip.clear(); strip.show(); }

/* ====================== CAPTURE TASK ====================== */
static void audio_inference_callback(int16_t s) {
    inference.buffers[inference.buf_select][inference.buf_count++] = s;
    if (inference.buf_count >= inference.n_samples) {
        inference.buf_select ^= 1;
        inference.buf_count = 0;
        inference.buf_ready = 1;
    }
}

static void capture_samples(void *arg) {
    const int RF = 512;
    const int CH = 4;
    static int16_t buf[512 * 4];
    static float dc_est = 0.0f;

    while (record_status) {
        size_t got = I2S.readBytes((char *)buf, RF * CH * sizeof(int16_t));
        int frames = got / (CH * sizeof(int16_t));
        if (frames == 0) { delay(1); continue; }

        int32_t mn[4] = {32767,32767,32767,32767};
        int32_t mx[4] = {-32768,-32768,-32768,-32768};

        for (int f = 0; f < frames; f++) {
            int16_t bL = buf[f*CH + CH_BACK_L];
            int16_t bR = buf[f*CH + CH_BACK_R];
            int16_t fL = buf[f*CH + CH_FRONT_L];
            int16_t fR = buf[f*CH + CH_FRONT_R];

            if (bL<mn[0]) mn[0]=bL; if (bL>mx[0]) mx[0]=bL;
            if (bR<mn[1]) mn[1]=bR; if (bR>mx[1]) mx[1]=bR;
            if (fL<mn[2]) mn[2]=fL; if (fL>mx[2]) mx[2]=fL;
            if (fR<mn[3]) mn[3]=fR; if (fR>mx[3]) mx[3]=fR;

            // mono stream for the model: average the two BACK mics
            float x = ((int32_t)bL + (int32_t)bR) * 0.5f;
            dc_est += 0.0005f * (x - dc_est);
            float y = (x - dc_est) * AUDIO_GAIN;
            if (y >  32767.f) y =  32767.f;
            if (y < -32768.f) y = -32768.f;
            audio_inference_callback((int16_t)y);
        }

        // Peak-hold with slow decay so brief sounds (claps) survive long
        // enough for the slower direction read in loop() to catch them.
        const float DECAY = 0.90f;
        float pbL = mx[0]-mn[0], pbR = mx[1]-mn[1];
        float pfL = mx[2]-mn[2], pfR = mx[3]-mn[3];
        g_pp_backL  = (pbL > g_pp_backL  * DECAY) ? pbL : g_pp_backL  * DECAY;
        g_pp_backR  = (pbR > g_pp_backR  * DECAY) ? pbR : g_pp_backR  * DECAY;
        g_pp_frontL = (pfL > g_pp_frontL * DECAY) ? pfL : g_pp_frontL * DECAY;
        g_pp_frontR = (pfR > g_pp_frontR * DECAY) ? pfR : g_pp_frontR * DECAY;
    }
    vTaskDelete(NULL);
}

static bool mic_start(uint32_t n_samples) {
    inference.buffers[0] = (signed short*)malloc(n_samples*sizeof(signed short));
    inference.buffers[1] = (signed short*)malloc(n_samples*sizeof(signed short));
    if (!inference.buffers[0] || !inference.buffers[1]) return false;
    inference.buf_select=0; inference.buf_count=0; inference.buf_ready=0;
    inference.n_samples=n_samples;

    I2S.setPinsPdmRx(PDM_CLK_PIN, PDM_DATA_BACK, PDM_DATA_FRONT);
    if (!I2S.begin(I2S_MODE_PDM_RX, 16000, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO)) {
        ei_printf("I2S.begin failed\n");
        return false;
    }
    delay(200);
    record_status = true;
    xTaskCreate(capture_samples, "cap", 1024*24, NULL, 10, NULL);
    return true;
}

static bool mic_record(void) {
    if (inference.buf_ready == 1) { ei_printf("overrun\n"); return false; }
    while (inference.buf_ready == 0) delay(1);
    inference.buf_ready = 0;
    return true;
}
static int mic_get_data(size_t off, size_t len, float *out) {
    numpy::int16_to_float(&inference.buffers[inference.buf_select^1][off], out, len);
    return 0;
}

/* ====================== DIRECTION ====================== */
// returns horizontal pixel (LED_LEFT/MIDDLE/RIGHT idx) or -1 if quiet,
// sets isBack, and reports loudness from the chosen direction.
int horizontalPixel(bool *isBack, float *dirLoudness, float *outBack) {
    float bL = g_pp_backL  * CAL_BACK_L;
    float bR = g_pp_backR  * CAL_BACK_R;
    float fL = g_pp_frontL * CAL_FRONT_L;
    float fR = g_pp_frontR * CAL_FRONT_R;

    // Front pair reads mirrored vs back (bL tracks fR, bR tracks fL), so the
    // channels on the same physical side are (bL,fR) and (bR,fL).
    float sideA = bL + fR;     // one physical side
    float sideB = bR + fL;     // the other side
    float front = fL + fR;
    float back  = bL + bR;
    float total = sideA + sideB;

    float lr = 0.0f, fb = 0.0f;
    if (total > 0.0f) lr = (sideA - sideB) / total;
    if ((front + back) > 0.0f) fb = (front - back) / (front + back + 1.0f);

    if (outBack) *outBack = back;

    *isBack = false;
    *dirLoudness = total;
    if (total < SOUND_THRESHOLD) return -1;

    *isBack = (fb < -FB_THRESHOLD);

    // If left/right come out swapped during testing, swap these two lines.
    if (lr >  LR_THRESHOLD) { *dirLoudness = sideA; return LED_RIGHT_IDX; }
    if (lr < -LR_THRESHOLD) { *dirLoudness = sideB; return LED_LEFT_IDX; }
    *dirLoudness = total * 0.5f;
    return LED_MIDDLE_IDX;
}

/* ====================== SETUP ====================== */
void setup() {
    // Force haptic OFF as early as possible on every boot/upload.
    pinMode(HAPTIC_PIN, OUTPUT);
    hapticWrite(0);

    Serial.begin(115200);
    delay(300);
    hapticWrite(0);

    strip.begin(); strip.setBrightness(NEOPIXEL_MIN_BRIGHTNESS); ledClear();

    ei_printf("Glasses system starting. Classes: ");
    for (size_t i=0;i<EI_CLASSIFIER_LABEL_COUNT;i++)
        ei_printf("%s ", ei_classifier_inferencing_categories[i]);
    ei_printf("\n");

    run_classifier_init();
    if (!mic_start(EI_CLASSIFIER_SLICE_SIZE)) {
        ei_printf("ERR mic start\n");
        while (true) delay(100);
    }

    // boot blip
    ledShow(LED_MIDDLE_IDX, (RGB){60,60,60}, NEOPIXEL_BRIGHTNESS);
    hapticWrite(120); delay(150); hapticWrite(0);
    ledClear();
    ei_printf("Listening...\n");
}

/* ====================== LOOP ====================== */
void loop() {
    if (!mic_record()) return;

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_SLICE_SIZE;
    signal.get_data = &mic_get_data;
    ei_impulse_result_t result = {0};
    if (run_classifier_continuous(&signal, &result, debug_nn) != EI_IMPULSE_OK) return;

    hapticUpdate();

    // Refresh the sound CLASS only on a full model window (~2s).
    if (++print_results >= EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW) {
        print_results = 0;
        int top = -1; float topv = 0;
        for (size_t i=0;i<EI_CLASSIFIER_LABEL_COUNT;i++)
            if (result.classification[i].value > topv){ topv=result.classification[i].value; top=i; }
        lastTop = top; lastTopV = topv;
    }

    // Direction every slice (~0.5s).
    bool isBack = false;
    float dirLoudness = 0, backLoud = 0;
    int pixel = horizontalPixel(&isBack, &dirLoudness, &backLoud);

    // ---- Robust direction display with hold ----
    // Light the direction pixel for ANY loud directional sound, even if the
    // model is unsure. Color = class guess, white if not confident.
    static uint32_t lastSoundMs = 0;
    static int      heldPixel   = -1;
    static RGB      heldColor   = {255,255,255};
    static uint8_t  heldBright  = 0;

    if (pixel != -1) {                       // a sound passed SOUND_THRESHOLD
        heldPixel   = pixel;
        lastSoundMs = millis();
        heldColor   = (lastTop >= 0 && lastTopV >= CONF_THRESHOLD)
                      ? CLASS_COLOR[lastTop] : (RGB){255,255,255};
        heldBright  = brightnessFromLoudness(dirLoudness);
    }

    if (heldPixel != -1 && (millis() - lastSoundMs) < HOLD_MS) {
        ledShow(heldPixel, heldColor, heldBright);
    } else {
        heldPixel = -1;
        ledClear();
    }

    // ---- Haptic: buzz for loud sounds from behind, strength by loudness ----
    if (pixel != -1 && isBack) {
        uint8_t hi = hapticIntensityFromLoudness(backLoud);
        if (hi > 0 && hapPattern == nullptr) hapticStart(PAT_BACK, 4, hi);
    } else {
        hapticStop();
    }

#if AUDIO_DIAG
    static int diagN = 0;
    if (++diagN >= 4) {
        diagN = 0;
        const char *dir = (pixel==LED_LEFT_IDX)?"LEFT":(pixel==LED_RIGHT_IDX)?"RIGHT":(pixel==LED_MIDDLE_IDX)?"MID":"quiet";
        ei_printf("bL=%.0f bR=%.0f fL=%.0f fR=%.0f | %s%s | class=%s %.2f\n",
            g_pp_backL, g_pp_backR, g_pp_frontL, g_pp_frontR,
            dir, isBack?"+BACK":"",
            (lastTop>=0)?ei_classifier_inferencing_categories[lastTop]:"-", lastTopV);
    }
#endif
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Model is not a microphone model."
#endif
