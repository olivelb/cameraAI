"""
app.py — AI Vision: Smart camera application with Gradio UI.

Connects to a Raspberry Pi camera stream, runs real-time AI inference
(detection, segmentation, tracking, face/animal recognition), and
provides a web-based control panel.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from typing import Optional

import cv2
import gradio as gr
import numpy as np
from cryptography.fernet import Fernet
from fastapi.responses import StreamingResponse
from PIL import Image
from websocket import create_connection

from ai import AIProcessor
from config import cfg
from notifier import EmailNotifier
from platform_utils import apply_windows_boost, cleanup_port

# ── Apply platform optimizations early ────────────────────────────────────────
apply_windows_boost()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Ring buffer for UI log display ────────────────────────────────────────────
_UI_LOGS: list[str] = []
_UI_LOGS_LOCK = threading.Lock()


def ui_log(msg: str) -> None:
    """Log a message to both Python logging and the Gradio UI panel."""
    timestamp = time.strftime("%H:%M:%S")
    entry = f"[{timestamp}] {msg}"
    log.info(msg)
    with _UI_LOGS_LOCK:
        _UI_LOGS.append(entry)
        if len(_UI_LOGS) > cfg.ui.log_max_lines:
            _UI_LOGS.pop(0)


def get_logs() -> str:
    """Return UI log entries in reverse-chronological order."""
    with _UI_LOGS_LOCK:
        return "\n".join(_UI_LOGS[::-1])


# ── Camera source state ──────────────────────────────────────────────────────
cam_source_mode: str = "pi"

# ── Core components ──────────────────────────────────────────────────────────
log.info("Initializing AI Engine (lazy loading — app starts immediately)...")
ai = AIProcessor()
notifier = EmailNotifier()

# ── Email config & encryption ────────────────────────────────────────────────
_CONFIG_FILE = "email_config.json"
_KEY_FILE = ".secret.key"

if not os.path.exists(_KEY_FILE):
    _key = Fernet.generate_key()
    with open(_KEY_FILE, "wb") as f:
        f.write(_key)
else:
    with open(_KEY_FILE, "rb") as f:
        _key = f.read()

_cipher = Fernet(_key)

email_config: dict = {
    "enabled": False,
    "sender": "",
    "password": "",
    "receiver": "",
}

if os.path.exists(_CONFIG_FILE):
    try:
        with open(_CONFIG_FILE, "r") as f:
            data = json.load(f)
            email_config.update(data)
            # Always start with alerts disabled for safety
            email_config["enabled"] = False
            if email_config["password"]:
                try:
                    email_config["password"] = _cipher.decrypt(
                        email_config["password"].encode()
                    ).decode()
                except Exception as e:
                    log.warning("Password decryption failed: %s", e)
    except Exception as e:
        log.error("Failed to load email config: %s", e)


def save_email_settings(
    enabled: bool, sender: str, password: str, receiver: str
) -> str:
    """Save email alert settings to disk with encrypted password."""
    global email_config
    encrypted_password = ""
    if password:
        encrypted_password = _cipher.encrypt(password.encode()).decode()

    email_config = {
        "enabled": enabled,
        "sender": sender,
        "password": password,
        "receiver": receiver,
    }

    config_to_save = email_config.copy()
    config_to_save["password"] = encrypted_password

    with open(_CONFIG_FILE, "w") as f:
        json.dump(config_to_save, f)
    ui_log(f"Email settings saved. Enabled: {enabled}")
    return f"Saved! Alerts Enabled: {enabled}"


def send_test_email() -> str:
    """Send a test email to verify notification settings."""
    if not email_config["sender"] or not email_config["password"]:
        return "❌ Missing Credentials"

    ui_log(f"Sending test email from {email_config['sender']}...")
    try:
        notifier.send_email(
            email_config["sender"],
            email_config["password"],
            email_config["receiver"],
            "[Test] Camera App Notification",
            "This is a test email from your AI Camera App.\n"
            "If you see this, notifications are working!",
            None,
        )
        return "✅ Test Email Sent! Check Inbox."
    except Exception as e:
        ui_log(f"Test Email Failed: {e}")
        return f"❌ Error: {e}"


# ── Alert Manager ────────────────────────────────────────────────────────────

class AlertManager:
    """Manages detection-triggered email alerts with per-class cooldowns."""

    def __init__(self) -> None:
        self._alert_classes = cfg.alerts.class_map()
        self.last_alert: dict[str, float] = {
            name: 0.0 for _, name in self._alert_classes
        }
        self.cooldown: int = cfg.alerts.cooldown_seconds

    def check(self, detected_classes: list[int], frame_rgb: np.ndarray) -> None:
        """Check if any detected class should trigger an alert."""
        if not email_config["enabled"] or not email_config["sender"]:
            return

        now = time.time()
        for cls_id, name in self._alert_classes:
            if cls_id in detected_classes:
                if now - self.last_alert[name] > self.cooldown:
                    self._trigger(name, frame_rgb)

    def _trigger(self, name: str, frame_rgb: np.ndarray) -> None:
        """Send an alert in a background thread."""
        ui_log(f"Triggering {name} alert (cooldown refreshed)")
        self.last_alert[name] = time.time()
        threading.Thread(
            target=self._process_alert, args=(name, frame_rgb), daemon=True
        ).start()

    def _process_alert(self, name: str, frame_rgb: np.ndarray) -> None:
        """Encode image, get AI description, and send email."""
        try:
            success, buffer = cv2.imencode(
                ".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            )
            if not success:
                ui_log("Failed to encode alert image")
                return

            img_bytes = buffer.tobytes()
            img_name = f"alert_{name}_{int(time.time())}.jpg"

            pil_img = Image.fromarray(frame_rgb)
            desc = ai.describe_scene(
                pil_img, f"Describe the {name} in this image concisely."
            )

            subject = f"[Camera Alert] {name.capitalize()} Detected"
            body = f"A {name} was detected.\n\nAI Analysis:\n{desc}"

            notifier.send_email(
                email_config["sender"],
                email_config["password"],
                email_config["receiver"],
                subject,
                body,
                img_bytes,
                img_name,
            )
            ui_log(f"Alert email sent to {email_config['receiver']}")
        except Exception as e:
            ui_log(f"Alert error: {e}")


alert_manager = AlertManager()


# ── WebSocket client ─────────────────────────────────────────────────────────

def _send_ws(data: dict, label: str = "") -> str:
    """Open a fresh WebSocket connection, send data, close."""
    ui_log(f"📡 Sending {label}...")
    try:
        conn = create_connection(cfg.pi.ws_url, timeout=3)
        conn.send(json.dumps(data))
        conn.close()
        ui_log(f"✅ Sent {label}")
        return f"Sent {label}"
    except Exception as e:
        ui_log(f"❌ WS Error: {e}")
        return f"WS Error: {e}"


def send_ws_command(key: str, value: int) -> str:
    """Send a single setting update to the Pi server."""
    return _send_ws(
        {"type": "update_settings", "data": {key: value}},
        label=f"{key}={value}",
    )


def set_resolution(res_str: str) -> str:
    """Send a resolution change command to the Pi."""
    w, h = map(int, res_str.split("x"))
    result = _send_ws(
        {"type": "update_settings", "data": {"width": w, "height": h}},
        label=f"resolution {res_str}",
    )
    ui_log("🔄 Pi restarting stream — client will auto-reconnect...")
    return result


def toggle_email_state(enabled: bool) -> None:
    """Toggle email alerts on/off."""
    email_config["enabled"] = enabled
    ui_log(f"🔔 Alerts set to: {enabled}")


def change_model_logic(model_name: str, task_mode: str) -> str:
    """Switch the active YOLO model and task type."""
    return ai.load_segmentation_model(model_name, task_mode)


# ── Shared state ─────────────────────────────────────────────────────────────

class AppState:
    """Thread-safe shared state between capture, AI, and UI threads."""

    def __init__(self) -> None:
        self.conf: float = 0.5
        self.last_frame: Optional[np.ndarray] = None
        self.last_processed: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.last_res: tuple[int, int] = (0, 0)
        self.frame_id: int = 0
        self.processed_id: int = -1


state = AppState()

# ── Video capture thread (Pi stream) ─────────────────────────────────────────
_FAIL_THRESHOLD = 10


def read_stream() -> None:
    """Background thread: reads frames from the Pi TCP stream.

    Only active when cam_source_mode == 'pi'; sleeps otherwise.
    Auto-reconnects on stream loss.
    """
    cap: Optional[cv2.VideoCapture] = None
    fail_count = 0

    def connect() -> cv2.VideoCapture:
        nonlocal cap, fail_count
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        ui_log("🖥️ Connecting to Pi camera server...")
        cap = cv2.VideoCapture(cfg.pi.video_url)
        fail_count = 0
        return cap

    connect()

    while True:
        try:
            if cam_source_mode == "webcam":
                time.sleep(0.5)
                continue

            if cap is None or not cap.isOpened():
                time.sleep(2)
                connect()
                continue

            ret, frame = cap.read()
            if not ret:
                fail_count += 1
                if fail_count >= _FAIL_THRESHOLD:
                    ui_log("🔄 Pi stream lost — reconnecting...")
                    time.sleep(2)
                    connect()
                else:
                    time.sleep(0.05)
                continue

            fail_count = 0
            with state.lock:
                state.last_frame = frame
                state.frame_id += 1

        except Exception as e:
            ui_log(f"⚠️ Stream error: {e}")
            time.sleep(2)
            connect()


threading.Thread(target=read_stream, daemon=True, name="stream-reader").start()


# ── AI processing thread ────────────────────────────────────────────────────

def ai_loop() -> None:
    """Dedicated thread: pulls latest raw frame, runs AI, stores result.

    Runs as fast as the GPU allows, completely decoupled from the Gradio
    timer refresh rate.
    """
    while True:
        try:
            with state.lock:
                if state.last_frame is None or state.frame_id == state.processed_id:
                    wait_for_new = True
                else:
                    wait_for_new = False
                    frame = state.last_frame.copy()
                    current_id = state.frame_id

            if wait_for_new:
                time.sleep(0.005)
                continue

            if ai.tracking_enabled:
                processed, info, detected_classes, _ = ai.track_frame(
                    frame, conf=state.conf
                )
            else:
                processed, info, detected_classes = ai.process_frame(
                    frame, conf=state.conf
                )

            # Log resolution changes
            h, w = frame.shape[:2]
            if (w, h) != state.last_res:
                state.last_res = (w, h)
                src = "Webcam" if cam_source_mode == "webcam" else "Pi"
                ui_log(f"[Stream] {src}: {w}x{h}")

            alert_manager.check(detected_classes, frame)

            with state.lock:
                state.last_processed = processed
                state.processed_id = current_id

        except Exception as e:
            ui_log(f"[AI error] {e}")
            time.sleep(0.1)


threading.Thread(target=ai_loop, daemon=True, name="ai-loop").start()


# ── MJPEG stream generator ──────────────────────────────────────────────────

def generate_mjpeg():
    """Generator yielding MJPEG frames for the /stream endpoint."""
    last_sent_id = -1
    while True:
        with state.lock:
            if state.last_processed is None or state.processed_id == last_sent_id:
                frame_to_send = None
            else:
                frame_to_send = state.last_processed
                last_sent_id = state.processed_id

        if frame_to_send is None:
            time.sleep(0.005)
            continue

        ret, buffer = cv2.imencode(
            ".jpg",
            frame_to_send,
            [cv2.IMWRITE_JPEG_QUALITY, cfg.stream.jpeg_quality],
        )
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


# ── Chat / VLM ───────────────────────────────────────────────────────────────

def ask_vlm(message: str, history: list) -> str:
    """Answer a question about the current frame using the VLM."""
    with state.lock:
        if state.last_frame is None:
            return "No video signal."
        frame_rgb = cv2.cvtColor(state.last_frame, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(frame_rgb)
    return ai.describe_scene(pil_img, message)


# ── UI callbacks ─────────────────────────────────────────────────────────────

def update_conf(val: float) -> None:
    """Update detection confidence threshold."""
    state.conf = float(val)


def toggle_tracking(enabled: bool) -> None:
    """Enable/disable multi-object tracking."""
    ai.tracking_enabled = enabled
    ui_log(f"🎯 Tracking {'enabled' if enabled else 'disabled'}")


def change_tracker(tracker_name: str) -> None:
    """Switch tracker algorithm (BoT-SORT or ByteTrack)."""
    ai.set_tracker(tracker_name.lower().replace("-", ""))
    ui_log(f"🔀 Tracker switched to: {tracker_name}")


def toggle_face_recognition(enabled: bool) -> str:
    """Toggle face recognition overlay."""
    result = ai.set_face_recognition(enabled)
    ui_log(result)
    return result


def add_to_gallery(name: str, files) -> str:
    """Add uploaded photos to the face gallery."""
    if not name or not name.strip():
        return "❌ Enter a person name first"
    if not files:
        return "❌ Upload at least one photo"
    paths = [f.name for f in files] if hasattr(files[0], "name") else list(files)
    result = ai.add_person_to_gallery(name.strip(), paths)
    ui_log(result)
    return result


def rebuild_gallery() -> str:
    """Rebuild the face gallery from disk."""
    result = ai.rebuild_gallery()
    ui_log(result)
    return result


def start_learning(name: str) -> str:
    """Start live face learning from the camera stream."""
    result = ai.start_learning(name.strip(), count=10)
    ui_log(result)
    return result


def update_sensitivity(val: float) -> None:
    """Update face recognition sensitivity threshold."""
    ai.set_face_threshold(val)


def get_gallery_status() -> str:
    """Get current face gallery status."""
    return ai.get_gallery_status()


# ── Animal recognition callbacks ─────────────────────────────────────────────

def toggle_animal_recognition(enabled: bool) -> str:
    """Toggle animal recognition overlay."""
    result = ai.set_animal_recognition(enabled)
    ui_log(result)
    return result


def add_animal_to_gallery(name: str, files) -> str:
    """Add uploaded photos to the animal gallery."""
    if not name or not name.strip():
        return "❌ Enter a pet name first"
    if not files:
        return "❌ Upload at least one photo"
    paths = [f.name for f in files] if hasattr(files[0], "name") else list(files)
    result = ai.add_animal_to_gallery(name.strip(), paths)
    ui_log(result)
    return result


def rebuild_animal_gallery() -> str:
    """Rebuild the animal gallery from disk."""
    result = ai.rebuild_animal_gallery()
    ui_log(result)
    return result


def start_animal_learning(name: str) -> str:
    """Start live animal learning from the camera stream."""
    result = ai.start_learning_animal(name.strip(), count=10)
    ui_log(result)
    return result


def update_animal_sensitivity(val: float) -> None:
    """Update animal recognition sensitivity threshold."""
    ai.set_animal_threshold(val)


def get_animal_gallery_status() -> str:
    """Get current animal gallery status."""
    return ai.get_animal_gallery_status()


def switch_source(mode_label: str):
    """Switch camera source (currently locked to Pi)."""
    global cam_source_mode
    cam_source_mode = "pi"
    ui_log("📷 Source locked to Pi")
    return gr.update(visible=True)


# ── Gradio UI Layout ────────────────────────────────────────────────────────

with gr.Blocks(title="AI Vision", theme=gr.themes.Base()) as demo:
    gr.Markdown("# 👁️ Smart Vision")

    with gr.Row():
        with gr.Column():
            gr.HTML(
                '<div style="width:100%;text-align:center;">'
                '<img src="/stream" style="width:100%;max-width:100%;border-radius:8px;">'
                "</div>",
                label="Live AI Feed",
            )

        with gr.Column(scale=1):
            gr.Markdown("### 🎛️ Controls")

            source_radio = gr.Radio(
                ["🖥️ Pi Camera Server"],
                value="🖥️ Pi Camera Server",
                label="Source",
                visible=False,
            )

            with gr.Group(visible=True) as pi_group:
                with gr.Row():
                    res_dropdown = gr.Dropdown(
                        ["640x480", "1296x972", "1920x1080"],
                        value="640x480",
                        label="Pi Resolution",
                    )
                    task_mode = gr.Radio(
                        ["Detection", "Segmentation"],
                        value="Detection",
                        label="Task Mode",
                    )
                fps_slider = gr.Slider(
                    cfg.pi.fps_min,
                    cfg.pi.fps_max,
                    value=cfg.pi.fps_default,
                    step=1,
                    label="FPS",
                )

            gr.Markdown("### 🧠 AI Engine")
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    [
                        "yolo26n.pt", "yolo26s.pt", "yolo26m.pt",
                        "yolo26l.pt", "yolo26x.pt",
                        "yolo11n.pt", "yolo11s.pt", "yolov8n.pt",
                    ],
                    value=cfg.ai.default_model,
                    label="Model",
                    scale=3,
                )
                optimize_btn = gr.Button("⚡ Optimize (TRT)", scale=1)

            gr.Markdown("### 🎯 Tracking")
            with gr.Row():
                tracking_toggle = gr.Checkbox(
                    label="Enable Tracking", value=False, scale=1
                )
                tracker_dropdown = gr.Dropdown(
                    ["BoT-SORT", "ByteTrack"],
                    value="BoT-SORT",
                    label="Tracker Algorithm",
                    scale=2,
                )

            with gr.Accordion("👤 Face Recognition", open=False):
                face_recog_toggle = gr.Checkbox(
                    label="Enable Face Recognition", value=False
                )
                gallery_status_box = gr.Textbox(
                    label="Gallery Status",
                    interactive=False,
                    lines=3,
                    value="Gallery empty — add photos and click Rebuild",
                )
                with gr.Row():
                    person_name_input = gr.Textbox(
                        label="Person Name", placeholder="e.g. Olivier", scale=2
                    )
                    face_photos_input = gr.File(
                        label="Photos",
                        file_count="multiple",
                        file_types=[".jpg", ".jpeg", ".png", ".webp"],
                        scale=3,
                    )
                with gr.Row():
                    add_person_btn = gr.Button("➕ Add to Gallery", scale=1)
                    rebuild_gallery_btn = gr.Button("🔄 Rebuild Gallery", scale=1)
                with gr.Row():
                    learn_btn = gr.Button(
                        "📸 Capture & Learn (Live)", variant="primary", scale=2
                    )
                    sensitivity_slider = gr.Slider(
                        0.2, 0.8,
                        value=cfg.ai.face_threshold,
                        step=0.01,
                        label="Sensitivity (Threshold)",
                    )

            with gr.Accordion("🐱 Animal Recognition", open=False):
                animal_recog_toggle = gr.Checkbox(
                    label="Enable Animal Recognition", value=False
                )
                animal_gallery_status_box = gr.Textbox(
                    label="Animal Gallery Status",
                    interactive=False,
                    lines=3,
                    value="Gallery empty — add photos and click Rebuild",
                )
                with gr.Row():
                    animal_name_input = gr.Textbox(
                        label="Pet Name", placeholder="e.g. Garfield", scale=2
                    )
                    animal_photos_input = gr.File(
                        label="Photos",
                        file_count="multiple",
                        file_types=[".jpg", ".jpeg", ".png", ".webp"],
                        scale=3,
                    )
                with gr.Row():
                    add_animal_btn = gr.Button("➕ Add to Gallery", scale=1)
                    rebuild_animal_gallery_btn = gr.Button(
                        "🔄 Rebuild Gallery", scale=1
                    )
                with gr.Row():
                    animal_learn_btn = gr.Button(
                        "📸 Capture & Learn (Live)", variant="primary", scale=2
                    )
                    animal_sensitivity_slider = gr.Slider(
                        0.6, 0.99,
                        value=cfg.ai.animal_threshold,
                        step=0.01,
                        label="Sensitivity (Threshold)",
                    )

            conf_slider = gr.Slider(0.1, 1.0, value=0.5, label="Confidence")

            with gr.Accordion("📧 Email Alerts", open=False):
                email_enable = gr.Checkbox(
                    label="Enable Alerts", value=email_config["enabled"]
                )
                email_sender = gr.Textbox(
                    label="Sender Email (Gmail)", value=email_config["sender"]
                )
                email_password = gr.Textbox(
                    label="App Password",
                    type="password",
                    value=email_config["password"],
                )
                email_receiver = gr.Textbox(
                    label="Receiver Email", value=email_config["receiver"]
                )
                save_btn = gr.Button("Save Settings")
                email_status = gr.Textbox(label="Status", interactive=False)
                save_btn.click(
                    save_email_settings,
                    inputs=[email_enable, email_sender, email_password, email_receiver],
                    outputs=email_status,
                )
                test_btn = gr.Button("📨 Send Test Email")
                test_btn.click(send_test_email, outputs=email_status)

            gr.Markdown("### 📜 System Logs")
            log_output = gr.Textbox(label="Console Output", lines=5, interactive=False)
            log_timer = gr.Timer(cfg.ui.log_refresh_interval)
            log_timer.tick(get_logs, outputs=log_output)

            gr.Markdown("### 💬 Vision Assistant")
            chat = gr.ChatInterface(fn=ask_vlm, type="messages")

            info_output = gr.Textbox(label="System Status", interactive=False)

    # ── Event wiring ─────────────────────────────────────────────────────────
    source_radio.change(switch_source, inputs=source_radio, outputs=[pi_group])
    conf_slider.change(update_conf, inputs=conf_slider)
    res_dropdown.change(set_resolution, inputs=res_dropdown, outputs=info_output)
    fps_slider.release(
        lambda x: send_ws_command("fps", int(x)), inputs=fps_slider, outputs=info_output
    )
    email_enable.change(toggle_email_state, inputs=email_enable)
    model_dropdown.change(
        change_model_logic, inputs=[model_dropdown, task_mode], outputs=info_output
    )
    task_mode.change(
        change_model_logic, inputs=[model_dropdown, task_mode], outputs=info_output
    )
    optimize_btn.click(ai.export_to_tensorrt, inputs=[], outputs=info_output)
    tracking_toggle.change(toggle_tracking, inputs=tracking_toggle)
    tracker_dropdown.change(change_tracker, inputs=tracker_dropdown)
    face_recog_toggle.change(
        toggle_face_recognition, inputs=face_recog_toggle, outputs=gallery_status_box
    )
    add_person_btn.click(
        add_to_gallery,
        inputs=[person_name_input, face_photos_input],
        outputs=gallery_status_box,
    )
    rebuild_gallery_btn.click(rebuild_gallery, outputs=gallery_status_box)
    learn_btn.click(
        start_learning, inputs=person_name_input, outputs=gallery_status_box
    )
    sensitivity_slider.change(update_sensitivity, inputs=sensitivity_slider)
    animal_recog_toggle.change(
        toggle_animal_recognition,
        inputs=animal_recog_toggle,
        outputs=animal_gallery_status_box,
    )
    add_animal_btn.click(
        add_animal_to_gallery,
        inputs=[animal_name_input, animal_photos_input],
        outputs=animal_gallery_status_box,
    )
    rebuild_animal_gallery_btn.click(
        rebuild_animal_gallery, outputs=animal_gallery_status_box
    )
    animal_learn_btn.click(
        start_animal_learning,
        inputs=animal_name_input,
        outputs=animal_gallery_status_box,
    )
    animal_sensitivity_slider.change(
        update_animal_sensitivity, inputs=animal_sensitivity_slider
    )


# ── Main entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    cleanup_port(cfg.ui.server_port)

    # Get local IP for display
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"

    # Check for SSL certs
    ssl_cert = "cert.pem" if os.path.exists("cert.pem") else None
    ssl_key = "key.pem" if os.path.exists("key.pem") else None

    print("\n" + "=" * 50)
    print("🚀  SERVER STARTING  🚀")
    print(f"📱  Local URL: http://localhost:{cfg.ui.server_port}")
    print("🌍  Internet URL: (Wait for Gradio link below)")
    print("=" * 50 + "\n")

    app_server, _, _ = demo.launch(
        server_name=cfg.ui.server_host,
        server_port=cfg.ui.server_port,
        share=cfg.ui.share,
        prevent_thread_lock=True,
    )

    # Attach MJPEG stream endpoint
    app_server.add_api_route(
        "/stream",
        endpoint=lambda: StreamingResponse(
            generate_mjpeg(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        ),
        methods=["GET"],
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
