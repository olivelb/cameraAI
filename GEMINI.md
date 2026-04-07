# Project Overview

This is an AI-powered smart camera application designed to work with a Raspberry Pi camera stream. The project uses advanced computer vision models to perform real-time object detection, segmentation, tracking, and facial/animal recognition. It features a modern web-based UI built with Gradio and supports features like email alerts and VLM (Vision-Language Model) descriptions.

## Architecture

The system is split into a camera streaming server (typically running on a Raspberry Pi) and a heavier AI processing client/frontend (running on a machine with a GPU or CPU).

*   **AI Frontend/Client (`app.py`)**: The main application interface built using Gradio. It connects to the Pi's video stream via TCP and control stream via WebSockets. It handles the display of the camera feed, user controls, and delegates frames to the AI processor.
*   **Camera Server (`server.py`)**: A lightweight Python script intended for a Raspberry Pi. It uses `rpicam-vid` to stream H.264 video natively over a TCP socket and runs a WebSocket server to receive live setting changes (resolution, FPS, bitrate). Ports and settings are configurable via `config.yaml`.
*   **AI Engine (`ai.py`)**: The core AI processing module. It manages the loading and inference of various PyTorch/TensorRT models. It supports YOLO for detection and segmentation, SAM3 for concept-based segmentation, BoT-SORT/ByteTrack for tracking, and Gemma4-E2B for visual question answering.
*   **Recognizers (`face_recognizer.py`, `animal_recognizer.py`)**: Specialized modules for re-identification. Face recognition uses RetinaFace + ArcFace via ONNX Runtime. Animal recognition uses DINOv3 embeddings via HuggingFace Transformers. Both build galleries from images stored in the `faces/` and `animals/` directories.
*   **Notification System (`notifier.py`)**: Handles sending email alerts with image attachments when specific classes (person, cat, dog) are detected. SMTP settings come from `config.yaml`.
*   **Configuration (`config.py`, `config.yaml`)**: Centralized settings system using YAML with environment variable overrides (prefix `CAMERA__`). All hardcoded values (IPs, ports, thresholds, model IDs) are externalized here.
*   **Platform Utils (`platform_utils.py`)**: Windows-specific performance optimizations (anti-EcoQoS throttling, port cleanup).

## Key Directories

*   `models/`: Directory for storing model weights (e.g., InsightFace ONNX models).
*   `faces/`: Contains galleries of known faces for the face recognition module.
*   `animals/`: Contains galleries of known pets/animals for the animal recognition module.

## Building and Running

### Running the AI Application (Client)
Ensure you have the required Python dependencies installed:

```bash
pip install -r requirements.txt
python app.py
```
The app will be available at `http://localhost:7860`.

### Running the Camera Server (Raspberry Pi)
On the Raspberry Pi, copy `server.py`, `config.py`, `config.yaml` and run:

```bash
python server.py
```

## Development Conventions

*   **Configuration First**: All configurable values live in `config.yaml`. No hardcoded IPs, ports, thresholds, or model IDs in source code.
*   **Lazy Loading**: Heavy AI libraries (torch, ultralytics, transformers) are loaded lazily in background threads to prevent blocking the main thread.
*   **Proper Logging**: All modules use Python's `logging` module. No `print()` statements for operational output.
*   **Type Hints**: All functions have type annotations for parameters and return values.
*   **DRY Helpers**: Common operations (label drawing, animal crop extraction, temporal smoothing) are extracted into reusable helper methods.
*   **Multithreading**: Extensive use of daemon threads to decouple UI, capture, AI inference, and notifications.
*   **Model Optimization**: Supports exporting `.pt` PyTorch models to TensorRT (`.engine`) for NVIDIA GPUs.
*   **Settings Persistence**: Email configuration is saved to `email_config.json` with Fernet-encrypted passwords.