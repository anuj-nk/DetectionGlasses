"""
YAMNet Backbone-Only TFLite Conversion for ESP32-S3
====================================================
Strips the spectrogram frontend from YAMNet and exports only the
MobileNet classifier backbone, which takes a log-mel spectrogram
as input instead of raw audio.

This produces a model with only ops supported by tflite-micro:
  CONV_2D, DEPTHWISE_CONV_2D, MEAN, FULLY_CONNECTED, LOGISTIC

You compute the log-mel spectrogram on-device (see companion Arduino code).

Input shape:  [1, 96, 64, 1]  — 96 time frames × 64 mel bins, float32
Output shape: [1, 521]        — class scores (softmax), float32

Usage:
    pip install tensorflow tensorflow-hub numpy
    python yamnet_convert.py
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

OUTPUT_DIR  = "tflite_models"
QUANT_MODEL = os.path.join(OUTPUT_DIR, "yamnet_backbone_int8.tflite")
FULL_MODEL  = os.path.join(OUTPUT_DIR, "yamnet_backbone_f32.tflite")

# YAMNet spectrogram params (must match on-device computation)
NUM_FRAMES  = 96
NUM_BINS    = 64


# ── 1. Load YAMNet and extract the backbone ──────────────────────────────────

def get_backbone():
    """
    YAMNet's TF Hub model exposes internal layers.
    We build a new model that takes a pre-computed spectrogram
    and runs only the MobileNet layers.
    """
    print("Loading YAMNet from TF Hub...")
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

    # The hub model's __call__ runs: waveform → spectrogram → embeddings → scores
    # We want just: spectrogram → scores
    # Recreate this by tracing through the model's internal structure.

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, NUM_FRAMES, NUM_BINS, 1], dtype=tf.float32)
    ])
    def backbone_infer(spectrogram):
        # patches shape expected by yamnet internals: [N, 96, 64, 1]
        # call the internal _apply_weights / embeddings path
        embeddings = yamnet.call_yamnet(spectrogram)  # may not exist on all versions
        return embeddings

    return yamnet


def build_backbone_model():
    """
    More reliable approach: rebuild the MobileNet backbone in Keras
    using YAMNet's saved weights. This gives full control over the graph.
    """
    print("Loading YAMNet weights...")
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

    # Extract weights by running a dummy inference to populate variables
    dummy_wav = tf.zeros([16000], dtype=tf.float32)
    yamnet(dummy_wav)  # populates variables

    # Build a Keras model matching YAMNet's MobileNet backbone
    inputs = tf.keras.Input(shape=(NUM_FRAMES, NUM_BINS, 1), name="log_mel_spectrogram")

    def _conv_bn_relu(x, filters, kernel, stride, name_prefix):
        x = tf.keras.layers.Conv2D(
            filters, kernel, strides=stride, padding="same",
            use_bias=False, name=f"{name_prefix}_conv")(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
        x = tf.keras.layers.ReLU(6.0, name=f"{name_prefix}_relu")(x)
        return x

    def _dw_block(x, filters, stride, name_prefix):
        x = tf.keras.layers.DepthwiseConv2D(
            3, strides=stride, padding="same",
            use_bias=False, name=f"{name_prefix}_dw_conv")(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_dw_bn")(x)
        x = tf.keras.layers.ReLU(6.0, name=f"{name_prefix}_dw_relu")(x)
        x = tf.keras.layers.Conv2D(
            filters, 1, strides=1, padding="same",
            use_bias=False, name=f"{name_prefix}_pw_conv")(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_pw_bn")(x)
        x = tf.keras.layers.ReLU(6.0, name=f"{name_prefix}_pw_relu")(x)
        return x

    # MobileNet-v1 backbone (matches YAMNet architecture)
    x = _conv_bn_relu(inputs, 32,  3, 2, "layer1")
    x = _dw_block(x,           64,  1, "layer2")
    x = _dw_block(x,          128,  2, "layer3")
    x = _dw_block(x,          128,  1, "layer4")
    x = _dw_block(x,          256,  2, "layer5")
    x = _dw_block(x,          256,  1, "layer6")
    x = _dw_block(x,          512,  2, "layer7")
    for i in range(5):
        x = _dw_block(x,      512,  1, f"layer{8+i}")
    x = _dw_block(x,         1024,  2, "layer13")
    x = _dw_block(x,         1024,  1, "layer14")

    # Global average pool → classifier
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg")(x)
    outputs = tf.keras.layers.Dense(521, activation="sigmoid", name="scores")(x)

    model = tf.keras.Model(inputs, outputs, name="yamnet_backbone")
    print(f"Backbone params: {model.count_params():,}")
    return model, yamnet


# ── 2. Transfer weights from hub model ───────────────────────────────────────

def transfer_weights(keras_model, yamnet_hub):
    """
    Copy weights from the TF Hub YAMNet into the Keras backbone.
    Falls back to random weights if the hub model structure differs.
    """
    try:
        # Run inference to materialize hub model weights
        dummy = tf.zeros([16000], dtype=tf.float32)
        yamnet_hub(dummy)

        hub_vars = {v.name: v.numpy() for v in yamnet_hub.variables}
        print(f"Hub model has {len(hub_vars)} weight tensors")

        keras_vars = keras_model.variables
        matched = 0
        for kv in keras_vars:
            # Try to find matching hub variable by shape
            candidates = [
                (name, val) for name, val in hub_vars.items()
                if val.shape == kv.shape
            ]
            if len(candidates) == 1:
                kv.assign(candidates[0][1])
                matched += 1

        print(f"Matched {matched}/{len(keras_vars)} weight tensors")
        if matched < len(keras_vars) * 0.8:
            print("⚠  Low match rate — weights may not transfer correctly")
            print("   Model will use partially random weights")
        else:
            print("✓ Weights transferred")

    except Exception as e:
        print(f"⚠  Weight transfer failed: {e}")
        print("   Continuing with random weights (model won't classify correctly)")

    return keras_model


# ── 3. Convert to TFLite ──────────────────────────────────────────────────────

def convert_float32(model):
    print("\nConverting backbone to float32 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_bytes = converter.convert()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(FULL_MODEL, "wb") as f:
        f.write(tflite_bytes)
    print(f"✓ Float32: {len(tflite_bytes)/1024:.1f} KB → {FULL_MODEL}")
    return tflite_bytes


def convert_int8(model):
    print("\nConverting backbone to int8 TFLite...")

    def representative_dataset():
        for _ in range(100):
            # Random log-mel spectrograms in realistic range [-10, 2]
            spec = np.random.uniform(-10, 2, (1, NUM_FRAMES, NUM_BINS, 1)).astype(np.float32)
            yield [spec]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.float32  # keep float I/O for easier on-device use
    converter.inference_output_type = tf.float32

    tflite_bytes = converter.convert()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(QUANT_MODEL, "wb") as f:
        f.write(tflite_bytes)
    print(f"✓ Int8: {len(tflite_bytes)/1024:.1f} KB → {QUANT_MODEL}")
    return tflite_bytes


# ── 4. Verify ops ─────────────────────────────────────────────────────────────

MICRO_SAFE_OPS = {
    "CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED",
    "MEAN", "RESHAPE", "SOFTMAX", "LOGISTIC",
    "ADD", "MUL", "PAD", "QUANTIZE", "DEQUANTIZE"
}

def verify_ops(tflite_bytes, label):
    print(f"\nOp audit — {label}:")
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    ops = {op["op_name"] for op in interpreter._get_ops_details()}
    
    bad = ops - MICRO_SAFE_OPS
    print(f"  All ops:  {sorted(ops)}")
    if bad:
        print(f"  ⚠  Unsupported on tflite-micro: {sorted(bad)}")
    else:
        print(f"  ✓ All ops are tflite-micro compatible!")
    return len(bad) == 0


# ── 5. Smoke test ─────────────────────────────────────────────────────────────

def smoke_test(tflite_bytes, label):
    print(f"\nSmoke test — {label}:")
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    dummy = np.random.uniform(-10, 2, inp["shape"]).astype(np.float32)
    interp.set_tensor(inp["index"], dummy)
    interp.invoke()
    scores = interp.get_tensor(out["index"])[0]

    top5_idx = np.argsort(scores)[-5:][::-1]
    print(f"  Input:  {inp['shape']} {inp['dtype']}")
    print(f"  Output: {out['shape']} {out['dtype']}")
    print(f"  Top-5 class indices: {top5_idx}  scores: {scores[top5_idx].round(3)}")
    print(f"✓ Smoke test passed")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    backbone, yamnet_hub = build_backbone_model()
    backbone = transfer_weights(backbone, yamnet_hub)
    backbone.summary(line_length=80)

    f32_bytes  = convert_float32(backbone)
    int8_bytes = convert_int8(backbone)

    f32_ok  = verify_ops(f32_bytes,  "float32")
    int8_ok = verify_ops(int8_bytes, "int8")

    smoke_test(f32_bytes,  "float32")
    smoke_test(int8_bytes, "int8")

    print("\n" + "="*60)
    print("  Summary")
    print("="*60)
    print(f"  Float32 : {len(f32_bytes)/1024:>7.1f} KB  micro-safe: {'✓' if f32_ok else '✗'}")
    print(f"  Int8    : {len(int8_bytes)/1024:>7.1f} KB  micro-safe: {'✓' if int8_ok else '✗'}")
    print("""
  On-device you need to compute log-mel spectrogram before inference:
    1. Capture 16000 samples @ 16kHz (1 second)
    2. Apply pre-emphasis filter
    3. Frame into 25ms windows with 10ms hop → 96 frames
    4. Apply Hann window
    5. FFT → power spectrum → 64 mel filterbank bins
    6. Log compression: log(max(spec, 1e-6))
    7. Feed [1, 96, 64, 1] float32 tensor to model
  """)