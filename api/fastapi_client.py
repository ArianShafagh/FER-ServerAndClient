"""
Webcam client that sends frames to the FastAPI server and optionally draws the result.

Run:
    python fastapi_client.py --server-url http://127.0.0.1:8000/predict-frame
"""

import argparse
import time

import cv2
import numpy as np
import requests


def draw_server_detections(frame: np.ndarray, detections: list[dict]) -> None:
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        label = detection["label"]
        confidence = detection["confidence"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        caption = f"{label} {confidence:.0%}"
        cv2.putText(
            frame,
            caption,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Test client for the FastAPI emotion server")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000/predict-frame")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--client-id", default="python-test-client")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    frame_index = 0
    fps_buffer = []
    last_time = time.perf_counter()

    print("Sending frames to server. Press Q or ESC to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame_index += 1
        now = time.perf_counter()
        fps_buffer.append(1.0 / max(now - last_time, 1e-9))
        if len(fps_buffer) > 30:
            fps_buffer.pop(0)
        last_time = now

        success, encoded = cv2.imencode(".jpg", frame)
        if not success:
            continue

        files = {"file": ("frame.jpg", encoded.tobytes(), "image/jpeg")}
        data = {"frame_index": str(frame_index), "client_id": args.client_id}

        try:
            response = requests.post(args.server_url, files=files, data=data, timeout=30)
            response.raise_for_status()
            result = response.json()
        except Exception as exc:
            print(f"Request failed: {exc}")
            continue

        detections = result.get("detections", [])
        if not args.no_display:
            display_frame = frame.copy()
            draw_server_detections(display_frame, detections)
            avg_fps = float(np.mean(fps_buffer)) if fps_buffer else 0.0
            cv2.putText(
                display_frame,
                f"FPS {avg_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                f"Server faces: {result.get('num_faces', 0)}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("FastAPI Client Test", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break

        print(f"frame {frame_index}: {result.get('num_faces', 0)} face(s)")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()