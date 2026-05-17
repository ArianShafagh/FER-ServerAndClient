# FER-ServerAndClient

Realtime facial emotion recognition with a FastAPI server, a webcam client, and an ONNX-based emotion model.

## Project Structure

- [run.py](run.py): starts the server and client together.
- [api/fastapi_server.py](api/fastapi_server.py): FastAPI server that accepts uploaded frames and returns detections.
- [api/fastapi_client.py](api/fastapi_client.py): webcam client that sends frames to the server.
- [main/app.py](main/app.py): shared emotion detection logic used by the server and standalone runner.
- [requirements.txt](requirements.txt): Python dependencies.
- `models/poster_v2_rafdb.onnx`: emotion classification model.
- `models/blaze_face_short_range.tflite`: face detector model.

## Requirements

Install the dependencies with:

```bash
pip install -r requirements.txt
```

If you want to use a virtual environment, activate it first and then install the requirements.

## Run the Full System

Start the server and client together from the project root:

```bash
python run.py
```

The launcher will:

1. Start the FastAPI server.
2. Wait until the `/health` endpoint is ready.
3. Start the webcam client.

## Run Components Separately

Start the server:

```bash
python api/fastapi_server.py
```

Start the client in another terminal:

```bash
python api/fastapi_client.py --server-url http://127.0.0.1:8000/predict-frame
```

## API

- `GET /health`: checks whether the server is ready.
- `POST /predict-frame`: accepts a JPEG frame and returns face boxes and emotion predictions.

## Configuration

The server reads these optional environment variables:

- `POSTER_MODEL_PATH`: path to the ONNX emotion model.
- `MP_FACE_MODEL_PATH`: path to the MediaPipe face detector model.
- `FRAME_OUTPUT_JSON`: path where frame results are written.

## Notes

- The client uses the default webcam unless you pass `--camera`.
- The server expects the ONNX model and face detector file to be present in the project.
- Large model files such as `models/poster_v2_rafdb.onnx` are intentionally kept out of Git history to avoid GitHub's 100 MB file limit.
- If you are on macOS and `python` is not available, use the interpreter inside your virtual environment.