"""
FastAPI server for frame-by-frame emotion recognition.

Run:
    uvicorn fastapi_server:app --reload --host 0.0.0.0 --port 8000

Or:
    python fastapi_server.py
"""

import json
import os
import time
import threading
import sys
from typing import Any
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

BASE_DIR = Path(__file__).resolve().parent
# On the server this file sits next to models/; in the repo it lives in api/.
APP_DIR = BASE_DIR if (BASE_DIR / "models").is_dir() else BASE_DIR.parent


MODEL_PATH = os.getenv(
    "ONNX_MODEL_PATH",
    str(APP_DIR / "models" / "raf_resnet18.onnx"),
)
MP_MODEL_PATH = os.getenv("MP_FACE_MODEL_PATH", str(APP_DIR / "models" / "blaze_face_short_range.tflite"))
FRAME_OUTPUT_JSON = os.getenv("FRAME_OUTPUT_JSON", str(APP_DIR / "results" / "frame_outputs.json"))


# RAF-DB Basic class order, matching the raf_resnet18 model.
CLASSES = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']

FACE_PAD = 0.0

# ImageNet normalization stats, used by the RGB (ResNet-style) model path.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(face_bgr: np.ndarray, input_shape: tuple[int, int, int]) -> np.ndarray:
    """Preprocess a BGR face crop. ``input_shape`` is (H, W, channels)."""
    input_height, input_width, channels = input_shape

    if channels == 1:
        gray    = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
        img     = resized.astype(np.float32) / 255.0
        return img[np.newaxis, ..., np.newaxis]      # (1, H, W, 1) NHWC float32

    if channels == 3:
        rgb     = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
        img     = resized.astype(np.float32) / 255.0
        img     = (img - IMAGENET_MEAN) / IMAGENET_STD
        chw     = np.transpose(img, (2, 0, 1))       # HWC → CHW
        return chw[np.newaxis, ...].astype(np.float32)  # (1, 3, H, W) NCHW float32

    raise ValueError(f'Unsupported ONNX input channel count: {input_shape}')


def load_model(onnx_path: str) -> ort.InferenceSession:
    available  = ort.get_available_providers()
    providers  = (
        ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if 'CUDAExecutionProvider' in available
        else ['CPUExecutionProvider']
    )
    try:
        session = ort.InferenceSession(onnx_path, providers=providers)
    except Exception as e:
        sys.exit(f"[ERROR] Could not load ONNX model:\n  {e}")

    inp = session.get_inputs()[0]
    print(f"✅ ONNX model loaded")
    print(f"   input : {inp.name}  shape: {inp.shape}")
    print(f"   device: {providers[0]}")
    return session


def predict(session: ort.InferenceSession, face_bgr: np.ndarray):
    input_meta = session.get_inputs()[0]
    input_shape = input_meta.shape

    if len(input_shape) != 4:
        raise ValueError(f'Unsupported ONNX input shape: {input_shape}')

    # FER2013 model is NHWC: (1, 48, 48, 1); RAF ResNet model is NCHW: (1, 3, 224, 224).
    if input_shape[3] == 1:
        tensor = preprocess(face_bgr, (int(input_shape[1]), int(input_shape[2]), int(input_shape[3])))
    elif input_shape[1] == 3:
        tensor = preprocess(face_bgr, (int(input_shape[2]), int(input_shape[3]), int(input_shape[1])))
    else:
        raise ValueError(f'Unsupported ONNX input shape: {input_shape}')

    out = session.run(None, {input_meta.name: tensor})[0][0]  # (7,)

    # Softmax only when the output isn't already a probability vector.
    if out.min() < 0.0 or abs(float(out.sum()) - 1.0) > 1e-3:
        exp   = np.exp(out - out.max())
        probs = exp / exp.sum()
    else:
        probs = out

    idx = int(probs.argmax())
    return CLASSES[idx], float(probs[idx]), probs


def build_detector(model_path: str):
    if not os.path.isfile(model_path):
        sys.exit(f"[ERROR] Face detector model not found: {model_path}")

    options = vision.FaceDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=0.45,
    )
    detector = vision.FaceDetector.create_from_options(options)
    print("✅ MediaPipe face detector ready")
    return detector


def get_faces(detector, frame_rgb: np.ndarray):
    h, w    = frame_rgb.shape[:2]
    boxes   = []

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    results = detector.detect(mp_image)

    if not results.detections:
        return boxes

    for det in results.detections:
        b = det.bounding_box
        xmin = max(0, int(b.origin_x - b.width * FACE_PAD))
        ymin = max(0, int(b.origin_y - b.height * FACE_PAD))
        xmax = min(w, int(b.origin_x + b.width * (1 + FACE_PAD)))
        ymax = min(h, int(b.origin_y + b.height * (1 + FACE_PAD)))

        if xmax > xmin and ymax > ymin:
            boxes.append((xmin, ymin, xmax, ymax))

    return boxes


app = FastAPI(title="ONNX Emotion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_state_lock = threading.Lock()
_session = None
_detector = None


def _append_frame_result(record: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(FRAME_OUTPUT_JSON), exist_ok=True)

    with _state_lock:
        if os.path.isfile(FRAME_OUTPUT_JSON):
            try:
                with open(FRAME_OUTPUT_JSON, "r", encoding="utf-8") as file_handle:
                    payload = json.load(file_handle)
            except Exception:
                payload = {"frames": []}
        else:
            payload = {"frames": []}

        payload.setdefault("frames", []).append(record)

        with open(FRAME_OUTPUT_JSON, "w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2)


@app.on_event("startup")
def startup_event() -> None:
    global _session, _detector

    if not os.path.isfile(MODEL_PATH):
        raise RuntimeError(f"ONNX model not found: {MODEL_PATH}")

    _session = load_model(MODEL_PATH)
    _detector = build_detector(MP_MODEL_PATH)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "mp_model_path": MP_MODEL_PATH,
    }


@app.post("/predict-frame")
async def predict_frame(
    file: UploadFile = File(...),
    frame_index: int = Form(0),
    client_id: str = Form("python-client"),
) -> JSONResponse:
    if _session is None or _detector is None:
        raise HTTPException(status_code=503, detail="Model is not ready yet")

    raw_bytes = await file.read()
    image_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image frame")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = get_faces(_detector, frame_rgb)
    detections: list[dict[str, Any]] = []

    for x1, y1, x2, y2 in boxes:
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        try:
            _, _, probs = predict(_session, crop)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

        idx = int(np.argmax(probs))
        label = CLASSES[idx]
        confidence = float(probs[idx])

        detections.append(
            {
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "label": label,
                "confidence": round(confidence, 4),
                "probabilities": [round(float(probability), 6) for probability in probs.tolist()],
            }
        )

    response_payload = {
        "client_id": client_id,
        "frame_index": frame_index,
        "frame_shape": list(frame.shape),
        "num_faces": len(detections),
        "detections": detections,
    }

    _append_frame_result(
        {
            "client_id": client_id,
            "frame_index": frame_index,
            "timestamp": time.time(),
            "frame_shape": list(frame.shape),
            "detections": detections,
        }
    )

    return JSONResponse(response_payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=8000, reload=False)