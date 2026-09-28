# Deploying the Server on Linux

The server needs only **one Python file and the models**. You don't need the rest of the repo.

## Requirements

- **OS:** Linux x86_64 (Ubuntu 20.04+ / Debian 11+)
- **Python:** 3.9 – 3.12
- **RAM:** about 2 GB free
- **System packages:** `python3-venv`, `libgl1`, `libegl1`, `libglib2.0-0`
- **Network:** port `8000` must be reachable from the client machine.
- **Files to copy to the server:**

  ```text
  fer-server/
  ├── fastapi_server.py                  ← from api/fastapi_server.py
  └── models/
      ├── raf_resnet18.onnx              ← emotion model (download link below)
      └── blaze_face_short_range.tflite  ← face detector (from models/ in the repo)
  ```

  Download `raf_resnet18.onnx` from [Google Drive](https://drive.google.com/file/d/1H6Dh82HD8dWOgwKUbw7-VprPOlAgX8ua/view?usp=drive_link).

## Install on the server

1. Install the system packages:

   ```bash
   sudo apt update
   sudo apt install -y python3-venv libgl1 libegl1 libglib2.0-0
   ```

2. Create the folder:

   ```bash
   mkdir -p ~/fer-server/models
   ```

3. Copy the files. Run these from your own computer, in the project folder:

   ```bash
   scp api/fastapi_server.py user@SERVER_IP:~/fer-server/
   scp models/raf_resnet18.onnx models/blaze_face_short_range.tflite user@SERVER_IP:~/fer-server/models/
   ```

4. Create a virtual environment and install the Python packages:

   ```bash
   cd ~/fer-server
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install fastapi uvicorn python-multipart opencv-python mediapipe onnxruntime numpy
   ```

5. Start the server:

   ```bash
   python fastapi_server.py
   ```

6. Check that it is running:

   ```bash
   curl http://SERVER_IP:8000/health
   ```

7. Optional steps:

   ```bash
   # keep the server running after you log out
   nohup python fastapi_server.py > server.log 2>&1 &

   # open the port if a firewall is enabled
   sudo ufw allow 8000/tcp
   ```
