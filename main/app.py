"""
Controls:
    Q or ESC  — quit
    +  /  -   — raise / lower confidence threshold
"""

import sys, os, argparse, time, collections, json
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import onnxruntime as ort


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
# RAF-DB Basic native order (folders 1..7 -> index 0..6), matching the
# raf_resnet18 training config (cnn_raf/config.py EMOTION_NAMES). Short names
# are used so they line up with the COLORS keys below.
CLASSES = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']

COLORS = {
    'Angry':    (30,   30, 220),
    'Disgust':  (30,  160,  50),
    'Fear':     (160,  40, 200),
    'Happy':    (0,   200, 255),
    'Neutral':  (160, 160, 160),
    'Sad':      (200,  80,  20),
    'Surprise': (0,   210, 255),
}

FACE_PAD       = 0.0   # raf_resnet18 trained on tight aligned crops; keep boxes tight
SMOOTH_FRAMES  = 6
DEFAULT_THRESH = 0.5
FRAME_OUTPUT_JSON = os.path.join('results', 'frame_outputs.json')

# ImageNet normalization stats, used by the RGB (ResNet-style) model path.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(face_bgr: np.ndarray, input_shape: tuple[int, int, int]) -> np.ndarray:
    """Preprocess a BGR face crop for the emotion ONNX model.

    ``input_shape`` is given as (H, W, channels). Two model families are supported:

    * ``channels == 1`` — FER2013 grayscale model. Produces an NHWC tensor
      (1, H, W, 1) scaled to [0, 1]. That model has internal BatchNorm layers,
      so no external mean/std normalization is applied.
    * ``channels == 3`` — RGB ResNet-style model (e.g. raf_resnet18). Produces an
      NCHW tensor (1, 3, H, W): BGR→RGB, scaled to [0, 1] and ImageNet-normalized.
    """
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


# ══════════════════════════════════════════════════════════════════════════════
# ONNX MODEL
# ══════════════════════════════════════════════════════════════════════════════
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

    # FER2013 model is NHWC: (1, 48, 48, 1) — channels in the last dim.
    # RAF ResNet model is NCHW: (1, 3, 224, 224) — channels in dim 1.
    if input_shape[3] == 1:
        tensor = preprocess(face_bgr, (int(input_shape[1]), int(input_shape[2]), int(input_shape[3])))
    elif input_shape[1] == 3:
        tensor = preprocess(face_bgr, (int(input_shape[2]), int(input_shape[3]), int(input_shape[1])))
    else:
        raise ValueError(f'Unsupported ONNX input shape: {input_shape}')

    input_name = input_meta.name
    out = session.run(None, {input_name: tensor})[0][0]  # (7,)

    # The FER2013 graph ends in a Softmax (already a probability vector); the RAF
    # model emits raw logits. Softmax only when the output isn't already normalized.
    if out.min() < 0.0 or abs(float(out.sum()) - 1.0) > 1e-3:
        exp   = np.exp(out - out.max())
        probs = exp / exp.sum()
    else:
        probs = out

    idx = int(probs.argmax())
    return CLASSES[idx], float(probs[idx]), probs


# ══════════════════════════════════════════════════════════════════════════════
# FACE DETECTION  — MediaPipe Tasks API (FaceDetector)
# ══════════════════════════════════════════════════════════════════════════════
def build_detector(model_path: str = None):
    model_path = model_path or os.path.join('models', 'blaze_face_short_range.tflite')
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


# ══════════════════════════════════════════════════════════════════════════════
# DRAWING
# ══════════════════════════════════════════════════════════════════════════════
def draw_box(frame, box, label, conf, color, threshold):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if conf < threshold:
        cv2.putText(frame, '?', (x1 + 8, y1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (140, 140, 140), 1, cv2.LINE_AA)
        return

    text = f"{label}  {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 1)
    pad = 6
    ty  = max(y1 - th - pad * 2, 0)
    cv2.rectangle(frame,
                  (x1, ty),
                  (x1 + tw + pad * 2, ty + th + pad * 2),
                  color, -1)
    cv2.putText(frame, text,
                (x1 + pad, ty + th + pad),
                cv2.FONT_HERSHEY_DUPLEX, 0.75,
                (255, 255, 255), 1, cv2.LINE_AA)


def draw_bars(frame, probs, x=10, y=10):
    bw, bh, gap = 120, 14, 3
    for i, (cls, p) in enumerate(zip(CLASSES, probs)):
        ry = y + i * (bh + gap)
        cv2.rectangle(frame, (x, ry), (x + bw, ry + bh), (40, 40, 40), -1)
        fw = max(1, int(bw * float(p)))
        cv2.rectangle(frame, (x, ry), (x + fw, ry + bh), COLORS[cls], -1)
        cv2.putText(frame, f"{cls:<8} {p:.0%}",
                    (x + bw + 6, ry + bh - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (210, 210, 210), 1, cv2.LINE_AA)


def draw_hud(frame, fps, threshold):
    h, w = frame.shape[:2]
    cv2.putText(frame, f"FPS {fps:.1f}",
                (w - 110, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                (100, 255, 100), 1, cv2.LINE_AA)
    cv2.putText(frame, f"thresh {threshold:.2f}",
                (w - 140, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                (160, 160, 160), 1, cv2.LINE_AA)
    for i, hint in enumerate(["Q / ESC : quit", "+  /  - : threshold"]):
        cv2.putText(frame, hint,
                    (w - 170, h - 36 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (120, 120, 120), 1, cv2.LINE_AA)


def save_frame_result(json_path, frame_index, fps, threshold, frame_shape, detections):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    if os.path.isfile(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
        except Exception:
            payload = {"frames": []}
    else:
        payload = {"frames": []}

    payload.setdefault("frames", []).append({
        "frame_index": frame_index,
        "timestamp": time.time(),
        "fps": round(float(fps), 2),
        "threshold": round(float(threshold), 2),
        "frame_shape": list(frame_shape),
        "detections": detections,
    })

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
def run(onnx_path: str, camera_idx: int, threshold: float, mp_model: str = None):
    session  = load_model(onnx_path)
    detector = build_detector(mp_model)

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open camera {camera_idx}. "
                 "Try --camera 1 if you have multiple cameras.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    smoothers: dict[int, collections.deque] = {}
    fps_buf = collections.deque(maxlen=30)
    t_prev  = time.perf_counter()
    frame_index = 0

    print("[INFO] Running — press Q or ESC to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] dropped frame, retrying …")
            continue

        t_now = time.perf_counter()
        fps_buf.append(1.0 / max(t_now - t_prev, 1e-9))
        t_prev = t_now
        fps    = float(np.mean(fps_buf))
        frame_index += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes     = get_faces(detector, frame_rgb)
        frame_detections = []

        while len(smoothers) < len(boxes):
            smoothers[len(smoothers)] = collections.deque(maxlen=SMOOTH_FRAMES)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            try:
                _, _, probs = predict(session, crop)
            except Exception as e:
                print(f"[WARN] inference error: {e}")
                continue

            smoothers[i].append(probs)
            avg   = np.mean(smoothers[i], axis=0)
            idx   = int(avg.argmax())
            label = CLASSES[idx]
            conf  = float(avg[idx])
            color = COLORS[label]

            frame_detections.append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "label": label,
                "confidence": round(conf, 4),
                "probabilities": [round(float(p), 6) for p in avg.tolist()],
            })

            draw_box(frame, box, label, conf, color, threshold)
            if i == 0:
                draw_bars(frame, avg)

        draw_hud(frame, fps, threshold)
        save_frame_result(FRAME_OUTPUT_JSON, frame_index, fps, threshold, frame.shape, frame_detections)
        cv2.imshow('Emotion Recognition', frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('+'), ord('=')):
            threshold = min(0.95, round(threshold + 0.05, 2))
            print(f"[INFO] threshold → {threshold}")
        elif key == ord('-'):
            threshold = max(0.05, round(threshold - 0.05, 2))
            print(f"[INFO] threshold → {threshold}")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("[INFO] Done.")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY
# ════════════════════════════a══════════════════════════════════════════════════
if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Realtime emotion recognition — ONNX) + MediaPipe')
    ap.add_argument('--model',     required=True,
                    help='Path to models/model.onnx')
    ap.add_argument('--camera',    type=int,   default=0,
                    help='Camera index (default 0)')
    ap.add_argument('--threshold', type=float, default=DEFAULT_THRESH,
                    help=f'Min confidence threshold (default {DEFAULT_THRESH})')
    ap.add_argument('--mp-model', type=str, default=None,
                    help='Path to the MediaPipe face detector task model (.task or compatible .tflite)')
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        sys.exit(f"[ERROR] ONNX model not found: {args.model}\n"
                 "        Run export_to_onnx.py first to generate it.")

    run(args.model, args.camera, args.threshold, args.mp_model)