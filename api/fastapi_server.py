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
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main.app import CLASSES, build_detector, get_faces, load_model, predict


MODEL_PATH = os.getenv("POSTER_MODEL_PATH", os.path.join("models", "poster_v2_rafdb.onnx"))
MP_MODEL_PATH = os.getenv("MP_FACE_MODEL_PATH", os.path.join("models", "blaze_face_short_range.tflite"))
FRAME_OUTPUT_JSON = os.getenv("FRAME_OUTPUT_JSON", os.path.join("results", "frame_outputs.json"))


app = FastAPI(title="POSTER V2 Emotion API")

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