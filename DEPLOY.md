# Deploying the Server on Linux

## Requirements

- **OS:** Linux x86_64 (Ubuntu 20.04+ / Debian 11+)
- **Python:** 3.9 – 3.12
- **RAM:** about 2 GB free
- **System packages:** `python3-venv`, `libgl1`, `libglib2.0-0`
- **Models** (in the `models/` folder):
  - `raf_resnet18.onnx`: the emotion model. Download it from [Google Drive](https://drive.google.com/file/d/1H6Dh82HD8dWOgwKUbw7-VprPOlAgX8ua/view?usp=drive_link).
  - `blaze_face_short_range.tflite`: the face detector. It is already in the repo.
- **Network:** port `8000` must be reachable from the client machine.

## Install on the server

1. Install the system packages:

   ```bash
   sudo apt update
   sudo apt install -y python3-venv libgl1 libglib2.0-0
   ```

2. Get the code:

   ```bash
   git clone https://github.com/ArianShafagh/FER-ServerAndClient.git
   cd FER-ServerAndClient
   ```

3. Create a virtual environment and install the dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Copy the emotion model into `models/`. For example, run this from your own computer:

   ```bash
   scp raf_resnet18.onnx user@SERVER_IP:~/FER-ServerAndClient/models/
   ```

5. Start the server:

   ```bash
   python api/fastapi_server.py
   ```

6. Check that it is running:

   ```bash
   curl http://SERVER_IP:8000/health
   ```

7. Optional steps:

   ```bash
   # keep the server running after you log out
   nohup python api/fastapi_server.py > server.log 2>&1 &

   # open the port if a firewall is enabled
   sudo ufw allow 8000/tcp
   ```
