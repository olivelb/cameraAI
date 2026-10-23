# AI Smart Camera

AI-powered smart camera application that connects to a Raspberry Pi camera stream and performs real-time object detection, segmentation, tracking, face recognition, and animal recognition with a modern Gradio web UI.

## Features

- **Real-time Object Detection & Segmentation** — YOLO v8/v11/v26 with optional TensorRT acceleration
- **Multi-Object Tracking** — BoT-SORT and ByteTrack algorithms
- **Face Recognition** — RetinaFace + ArcFace via ONNX Runtime (GPU accelerated)
- **Animal Recognition** — DINOv3 embeddings for pet re-identification
- **Vision-Language Model** — Gemma4-E2B for intelligent scene descriptions and chat
- **SAM3 Segmentation** — Meta's Segment Anything Model 3 for concept-based segmentation
- **Email Alerts** — Automatic notifications when persons/cats/dogs are detected
- **Live MJPEG Stream** — Low-latency video feed via custom FastAPI endpoint

## Architecture

| Component | File | Description |
|---|---|---|
| AI Frontend | `app.py` | Gradio UI + video capture + AI processing loop |
| AI Engine | `ai.py` | Model loading, inference, VLM (Gemma4-E2B) |
| Face Recognizer | `face_recognizer.py` | RetinaFace + ArcFace ONNX pipeline |
| Animal Recognizer | `animal_recognizer.py` | DINOv3-based pet re-identification |
| Email Notifier | `notifier.py` | Gmail SMTP notification system |
| Camera Server | `server.py` | Pi-side rpicam-vid streaming + WebSocket control |
| Configuration | `config.py` / `config.yaml` | Centralized settings |
| Platform Utils | `platform_utils.py` | Windows anti-throttling optimizations |

## Quick Start

### 1. Install Dependencies

Use a dedicated Python 3.11+ environment (do not mix this application with
other ML projects). Choose a PyTorch build matching your CUDA installation
before installing the full ML requirements.

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

The UI is qualified with Gradio 6.30.0. Gradio 5's Hugging Face Hub constraint
conflicts with the project's pinned Transformers revision. The VLM uses 4-bit
CUDA inference; CPU detection is supported but the VLM requires a CUDA GPU.
See [the audit](AUDIT.md) for remaining dependency security findings and the
qualification limits of this setup.

### 2. Configure

Copy the example config and adjust for your setup:

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml` — at minimum set `pi.host` to your Raspberry Pi's IP address.

### 3. Download Model Weights

**YOLO models** are downloaded automatically by Ultralytics on first use.

**SAM3 model** must be downloaded manually:

1. Visit the [Meta SAM3 model page on Hugging Face](https://huggingface.co/facebook/sam3)
2. Request access (usually instant approval)
3. Download `sam3.pt` and place it in the project root directory:

```bash
# Option A: Direct download via huggingface-cli
pip install huggingface-hub
huggingface-cli download facebook/sam3 sam3.pt --local-dir .

# Option B: Manual download from the model page
# Save sam3.pt to: camera-client/sam3.pt
```

**InsightFace models** (for face recognition) are auto-downloaded on first use (~200MB).

**DINOv3** and **Gemma4-E2B** are auto-downloaded by HuggingFace Transformers on first use.

### 4. Run the Application

```bash
python app.py
```

The app will be available at `http://localhost:7860`.

### 5. Run the Camera Server (on Pi)

Copy `server.py`, `config.py`, `config.yaml`, `config.example.yaml`, and
`requirements-server.txt` to your Pi. Install only the lightweight Pi dependencies:

```bash
python3 -m venv camera-venv
source camera-venv/bin/activate
python -m pip install -r requirements-server.txt
python server.py
```

To run as a systemd service, copy `camera.service` too, and adjust `User`,
`WorkingDirectory`, and `ExecStart` to your installation. For the virtual environment
above, use its absolute interpreter path (for example,
`/opt/smart-camera/.venv/bin/python`). Then:

```bash
sudo cp camera.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable camera
sudo systemctl start camera
```

## Configuration

All settings are in `config.yaml`. Key options:

| Setting | Description | Default |
|---|---|---|
| `pi.host` | Raspberry Pi IP address | `camera.example.invalid` |
| `pi.auth_token` / `server.auth_token` | Shared secret for the Pi WebSocket control channel | `""` (no auth) |
| `ai.default_model` | Default YOLO model | `yolo26n.pt` |
| `ai.vlm_model_id` | VLM for scene description | `google/gemma-4-E2B-it` |
| `ai.animal_embed_model` | Animal embedding model | `facebook/dinov3-vits16-pretrain-lvd1689m` |
| `ai.person_class` / `ai.animal_classes` | COCO ids driving face/animal recognition | `0` / `[15, 16]` |
| `ai.face_recheck_interval` / `animal_recheck_interval` | Re-identify each tracked subject every N frames | `10` |
| `ai.max_cached_models` | Max models kept loaded (VRAM bound) | `2` |
| `ui.share` | Publish the UI on a public gradio.live URL | `false` |
| `ui.auth_username` / `ui.auth_password` | Login protecting the web UI | `""` (no login) |
| `server.default_bitrate` | Video stream bitrate | `2000000` (2 Mbps) |

Environment variable overrides are supported:
```bash
CAMERA__PI__HOST=192.0.2.20 python app.py
CAMERA__UI__AUTH_PASSWORD=s3cret python app.py
```

## Security notes

- **Web UI**: `ui.share: true` publishes the app (live camera feed included)
  on a public `gradio.live` URL. Only enable it together with
  `ui.auth_username`/`ui.auth_password` (or the `CAMERA__UI__AUTH_PASSWORD`
  environment variable). If `cert.pem`/`key.pem` exist in the project root,
  the UI is served over HTTPS automatically.
- Startup rejects public sharing without a complete login. The custom `/stream`
  endpoint checks the same Gradio session as the UI. An unauthenticated request
  receives HTTP 401 when login is configured.
- **Pi control channel**: set the same random secret in `pi.auth_token` and
  `server.auth_token` (e.g. `python -c "import secrets; print(secrets.token_hex(16))"`),
  then copy the updated `config.yaml` to the Pi. Update messages sent
  without the token are rejected.
- The Pi H.264/TCP video port has no authentication or transport encryption;
  the control token protects settings changes only. Keep both Pi ports on a
  trusted private network or a VPN. A missing config/dependency now fails startup
  rather than silently disabling configured authentication.
- **Email**: the app password is stored Fernet-encrypted in
  `email_config.json` (key: `.secret.key`); it is never displayed in the UI.
  Both files are git-ignored — never commit them.
- `config.yaml`, model weights, and the `faces/`/`animals/` biometric
  galleries are git-ignored by design.

## Gallery Setup

### Face Recognition
Place photos in `faces/<person_name>/`:
```
faces/
  Example Person/
    photo_001.jpg
    photo_002.jpg
  alice/
    photo_001.jpg
```

### Animal Recognition
Place photos in `animals/<pet_name>/`:
```
animals/
  Casper/
    photo_001.jpg
  Mia/
    photo_001.jpg
```

Then click **Rebuild Gallery** in the UI, or use the **Capture & Learn** button to learn from the live stream.

Gallery names must be plain names (accents and spaces are supported), without
paths or reserved filesystem characters. After updating from the pre-audit
version, rebuild the **face** gallery once: detector RGB preprocessing, anchor
coordinates, and landmark rescaling have been corrected. The original photos
and existing gallery remain intact until you explicitly rebuild them.

## Development

Run the CPU-only unit and local HTTP/WebSocket integration tests in a separate
environment. No camera, SMTP account, model download, or GPU is needed:

```bash
python -m venv .venv-test
# Activate this environment as above.
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip check
python -m pytest -q
python -m ruff check .
```

Do not install the CPU test `onnxruntime` alongside `onnxruntime-gpu` in the
same environment. CI defines Ubuntu/Python 3.11 and Windows/Python 3.13 checks;
the Linux PID lock test is deliberately skipped on Windows.

An occupied UI port now produces an actionable error; startup never kills its
owner. Windows boost is applied only at startup and uses ABOVE_NORMAL priority.
The Pi retains its PID file as an advisory-lock inode; a leftover file does not
mean a process is running. See [AUDIT.md](AUDIT.md) for findings, evidence and
prioritized improvements.

## License

Private project.
