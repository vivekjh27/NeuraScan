from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import io
from PIL import Image
import base64

app = Flask(__name__)
CORS(app)

# ─── Try to load the real model ───────────────────────────────────────────────
model = None
MODEL_PATH = os.environ.get("MODEL_PATH", "model.keras")

try:
    from tensorflow.keras.models import load_model
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH, compile=False)
        print(f"[✓] Model loaded from {MODEL_PATH}")
    else:
        print(f"[!] Model file not found at '{MODEL_PATH}'. Running in DEMO mode.")
except Exception as e:
    print(f"[!] Could not load model: {e}. Running in DEMO mode.")

CLASS_LABELS = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = 224


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def demo_predict(file_bytes: bytes):
    """Cycle through classes deterministically based on image size."""
    idx = len(file_bytes) % 4
    probs = [0.05, 0.05, 0.05, 0.05]
    probs[idx] = 0.85
    return probs


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    allowed = {"png", "jpg", "jpeg", "bmp", "webp"}
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported file type: .{ext}"}), 400

    file_bytes = file.read()

    # Generate thumbnail for display
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img.thumbnail((400, 400))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Run inference
    if model is not None:
        arr = preprocess_image(file_bytes)
        preds = model.predict(arr)[0].tolist()
    else:
        preds = demo_predict(file_bytes)

    predicted_idx = int(np.argmax(preds))
    predicted_class = CLASS_LABELS[predicted_idx]
    confidence = round(float(preds[predicted_idx]) * 100, 2)

    return jsonify({
        "class": predicted_class,
        "confidence": confidence,
        "probabilities": {CLASS_LABELS[i]: round(float(preds[i]) * 100, 2) for i in range(4)},
        "image_b64": b64_img,
        "demo_mode": model is None
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
