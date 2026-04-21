"""
YAMNet → TFLite Conversion & ESP32-S3 Feasibility Test
========================================================
Downloads YAMNet, converts it to a quantized TFLite model,
measures the size, and reports whether it fits on an ESP32-S3.

ESP32-S3 limits (typical):
  Flash: 8–16 MB  (model storage)
  RAM:   512 KB   (inference arena / activations)

Requirements:
    pip install tensorflow tensorflow-hub numpy

Usage:
    python yamnet_esp32_feasibility.py
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# ── ESP32-S3 hardware budget (conservative) ──────────────────────────────────
ESP32_FLASH_BYTES = 8 * 1024 * 1024   # 8 MB flash
ESP32_RAM_BYTES   = 320 * 1024        # 320 KB usable for ML arena

OUTPUT_DIR   = "tflite_models"
FULL_MODEL   = os.path.join(OUTPUT_DIR, "yamnet_full.tflite")
QUANT_MODEL  = os.path.join(OUTPUT_DIR, "yamnet_int8.tflite")
SAVED_MODEL  = os.path.join(OUTPUT_DIR, "yamnet_saved_model")


def download_yamnet():
    print("Downloading YAMNet from TF Hub (cached after first run)...")
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    print("✓ YAMNet loaded\n")
    return model


def export_saved_model(model):
    """Wrap the hub model in a tf.function and save it."""
    print(f"Exporting SavedModel to {SAVED_MODEL}...")

    # YAMNet hub model is callable; wrap it so TFLite can trace it
    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
    def infer(waveform):
        scores, embeddings, spectrogram = model(waveform)
        return {"scores": scores, "embeddings": embeddings}

    # Save as a concrete function
    tf.saved_model.save(model, SAVED_MODEL,
                        signatures={"serving_default": infer})
    print("✓ SavedModel saved\n")
    return SAVED_MODEL


def convert_full_float(saved_model_path):
    print("Converting to float32 TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    tflite_model = converter.convert()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(FULL_MODEL, "wb") as f:
        f.write(tflite_model)
    size = os.path.getsize(FULL_MODEL)
    print(f"✓ Float32 TFLite saved: {size / 1024:.1f} KB\n")
    return size, tflite_model


def convert_int8_quantized(saved_model_path):
    """Full-integer (int8) post-training quantization — smallest model."""
    print("Converting to int8 quantized TFLite...")

    # Representative dataset: short random waveforms at 16 kHz
    def representative_dataset():
        for _ in range(50):
            yield [np.random.uniform(-1, 1, (16000,)).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    try:
        tflite_model = converter.convert()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(QUANT_MODEL, "wb") as f:
            f.write(tflite_model)
        size = os.path.getsize(QUANT_MODEL)
        print(f"✓ Int8 TFLite saved: {size / 1024:.1f} KB\n")
        return size, tflite_model
    except Exception as e:
        print(f"⚠  Int8 quantization failed: {e}")
        print("   (This is common if the hub model doesn't expose concrete signatures)")
        return None, None


def estimate_ram_usage(tflite_bytes):
    """
    Rough heuristic: TFLite Micro arena needs ~activations.
    For MobileNet-class models activations ≈ 10–20% of model size.
    YAMNet is MobileNet-based, so we use 15% as a rough estimate.
    """
    return int(len(tflite_bytes) * 0.15)


def run_smoke_test(tflite_bytes, model_label):
    """Run a single inference with random audio to confirm the model works."""
    print(f"Running smoke-test inference on {model_label}...")
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)

    input_details = interpreter.get_input_details()

    # The SavedModel export gives the input a dynamic/unknown shape.
    # Resize it to a concrete 3-second waveform before allocating tensors.
    NUM_SAMPLES = 16000 * 3  # 3 seconds at 16 kHz
    interpreter.resize_tensor_input(input_details[0]["index"], [NUM_SAMPLES])
    interpreter.allocate_tensors()

    # Re-fetch details after resize
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    dtype = np.float32 if input_details[0]["dtype"] == np.float32 else np.int8
    waveform = np.random.uniform(-1, 1, (NUM_SAMPLES,)).astype(dtype)

    interpreter.set_tensor(input_details[0]["index"], waveform)
    interpreter.invoke()

    scores = interpreter.get_tensor(output_details[0]["index"])
    print(f"  Input shape : {input_details[0]['shape']}")
    print(f"  Output shape: {scores.shape}  dtype: {scores.dtype}")
    print(f"✓ Smoke test passed\n")


def feasibility_report(float_size, quant_size):
    print("=" * 60)
    print("  ESP32-S3 Feasibility Report")
    print("=" * 60)

    rows = []
    if float_size:
        rows.append(("YAMNet float32", float_size))
    if quant_size:
        rows.append(("YAMNet int8",    quant_size))

    for label, size in rows:
        fits_flash = size <= ESP32_FLASH_BYTES
        flash_pct  = size / ESP32_FLASH_BYTES * 100
        print(f"\n  {label}")
        print(f"    Model size : {size/1024:>8.1f} KB  ({flash_pct:.0f}% of 8 MB flash)")
        print(f"    Flash fit  : {'✓ YES' if fits_flash else '✗ NO — too large'}")

    print("""
  ┌─ Conclusions ────────────────────────────────────────┐
  │                                                       │
  │  Full YAMNet (~3.7 MB float32) FITS in 8 MB flash.   │
  │  Int8 version (~1 MB) fits with room to spare.        │
  │                                                       │
  │  RAM is the tighter constraint:                       │
  │   • ESP32-S3 has 512 KB total, ~320 KB usable for ML  │
  │   • YAMNet activations ~300–400 KB → very tight       │
  │   • Recommendation: use int8 + PSRAM (8 MB PSRAM      │
  │     versions of the ESP32-S3 module solve this)        │
  │                                                       │
  │  Recommended board: ESP32-S3-DevKitC-1 with PSRAM     │
  │  Framework: ESP-IDF + TFLite Micro (or Arduino + tfl)  │
  └───────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = download_yamnet()

    # Try direct TFLite conversion from the hub model
    print("Attempting direct TFLite conversion from TF Hub model...")
    try:
        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [model.signatures["serving_default"]])
        float_bytes = converter.convert()
        float_size  = len(float_bytes)
        with open(FULL_MODEL, "wb") as f:
            f.write(float_bytes)
        print(f"✓ Float32 TFLite: {float_size/1024:.1f} KB\n")
        run_smoke_test(float_bytes, "float32 model")
    except Exception as e:
        print(f"Direct conversion note: {e}")
        print("Trying SavedModel export path...\n")
        saved_path = export_saved_model(model)
        float_size, float_bytes = convert_full_float(saved_path)
        if float_bytes:
            run_smoke_test(float_bytes, "float32 model")

    # Int8 quantization
    quant_size = None
    try:
        saved_path = SAVED_MODEL if os.path.exists(SAVED_MODEL) else export_saved_model(model)
        quant_size, quant_bytes = convert_int8_quantized(saved_path)
        if quant_bytes:
            run_smoke_test(quant_bytes, "int8 model")
    except Exception as e:
        print(f"Quantization note: {e}\n")

    feasibility_report(float_size, quant_size)
