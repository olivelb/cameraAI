"""
camera_core.py — Shared camera, inference and application services.

Connects to a Raspberry Pi camera stream and runs real-time AI inference
(detection, segmentation, tracking, face/animal recognition) for both frontends.
"""

from __future__ import annotations

import json
import asyncio
import logging
import os
import threading
import time
import tempfile
from collections import deque
from pathlib import Path

from capture import open_pi_capture

import cv2
import numpy as np
from cryptography.fernet import Fernet
from PIL import Image
from websocket import create_connection

from ai import AIProcessor
from config import cfg
from notifier import EmailNotifier

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


def get_logs_and_status() -> tuple[str, str]:
    """Return (logs, current AI status) for the periodic UI refresh."""
    with state.lock:
        info = state.last_info
    return get_logs(), info


# ── Camera source state ──────────────────────────────────────────────────────
cam_source_mode: str = "pi"

# ── Core components ──────────────────────────────────────────────────────────
log.info("Initializing AI Engine (lazy loading — app starts immediately)...")
ai = AIProcessor(autoload=False)
notifier = EmailNotifier()

# ── Email config & encryption ────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
_CONFIG_FILE = _ROOT / "email_config.json"
_KEY_FILE = _ROOT / ".secret.key"
_EMAIL_LOCK = threading.RLock()


def _get_cipher() -> Fernet:
    """Create the key only when saving credentials, with restrictive permissions."""
    with _EMAIL_LOCK:
        if not _KEY_FILE.exists():
            try:
                fd = os.open(_KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                pass
            else:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(Fernet.generate_key())
        return Fernet(_KEY_FILE.read_bytes())

email_config: dict = {
    "enabled": False,
    "sender": "",
    "password": "",
    "receiver": "",
}

if os.path.exists(_CONFIG_FILE):
    try:
        with open(_CONFIG_FILE) as f:
            data = json.load(f)
            email_config.update(data)
            # Always start with alerts disabled for safety
            email_config["enabled"] = False
            if email_config["password"]:
                try:
                    if not _KEY_FILE.exists():
                        raise ValueError("Email encryption key is missing")
                    email_config["password"] = _get_cipher().decrypt(
                        email_config["password"].encode()
                    ).decode()
                except Exception as e:
                    email_config["password"] = ""
                    log.warning("Password decryption failed: %s", e)
    except Exception as e:
        log.error("Failed to load email config: %s", e)


def save_email_settings(
    enabled: bool, sender: str, password: str, receiver: str
) -> str:
    """Serialize updates so disk and memory agree after each credential save."""
    with _EMAIL_LOCK:
        return _save_email_settings(enabled, sender, password, receiver)


def _save_email_settings(
    enabled: bool, sender: str, password: str, receiver: str
) -> str:
    """Save email alert settings to disk with encrypted password.

    An empty password field keeps the previously stored password.
    """
    global email_config
    new_password = password if password else email_config.get("password", "")
    encrypted_password = ""
    if new_password:
        encrypted_password = _get_cipher().encrypt(new_password.encode()).decode()

    updated_config = {
        "enabled": enabled,
        "sender": sender,
        "password": new_password,
        "receiver": receiver,
    }

    config_to_save = updated_config.copy()
    config_to_save["password"] = encrypted_password

    fd, temporary = tempfile.mkstemp(prefix=".email-", suffix=".json", dir=_CONFIG_FILE.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(config_to_save, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, _CONFIG_FILE)
        with _EMAIL_LOCK:
            email_config = updated_config
    finally:
        Path(temporary).unlink(missing_ok=True)
    ui_log(f"Email settings saved. Enabled: {enabled}")
    return f"Saved! Alerts Enabled: {enabled}"


def send_test_email() -> str:
    """Send a test email to verify notification settings."""
    if not email_config["sender"] or not email_config["password"]:
        return "❌ Missing Credentials"

    ui_log(f"Sending test email from {email_config['sender']}...")
    try:
        notifier.send_email_sync(
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

    def check(self, detected_classes: list[int], frame_bgr: np.ndarray) -> None:
        """Check if any detected class should trigger an alert."""
        if not email_config["enabled"] or not email_config["sender"]:
            return

        now = time.time()
        for cls_id, name in self._alert_classes:
            if cls_id in detected_classes:
                if now - self.last_alert[name] > self.cooldown:
                    self._trigger(name, frame_bgr)

    def _trigger(self, name: str, frame_bgr: np.ndarray) -> None:
        """Send an alert in a background thread."""
        ui_log(f"Triggering {name} alert (cooldown refreshed)")
        self.last_alert[name] = time.time()
        threading.Thread(
            target=self._process_alert, args=(name, frame_bgr.copy()), daemon=True
        ).start()

    def _process_alert(self, name: str, frame_bgr: np.ndarray) -> None:
        """Encode image, get AI description, and send email."""
        try:
            success, buffer = cv2.imencode(".jpg", frame_bgr)
            if not success:
                ui_log("Failed to encode alert image")
                return

            img_bytes = buffer.tobytes()
            img_name = f"alert_{name}_{int(time.time())}.jpg"

            pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            desc = ai.describe_scene(
                pil_img, f"Describe the {name} in this image concisely."
            )

            subject = f"[Camera Alert] {name.capitalize()} Detected"
            body = f"A {name} was detected.\n\nAI Analysis:\n{desc}"

            notifier.send_email_sync(
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


def configure_data_directory(directory: Path) -> None:
    """Isolate galleries and email state before any worker/model starts."""
    global _CONFIG_FILE, _KEY_FILE, email_config
    if _threads_started or ai._face_recognizer is not None or ai._animal_recognizer is not None:
        raise RuntimeError("Choose the data directory before starting the application")
    directory = directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    import face_recognizer
    import animal_recognizer

    face_recognizer.FACES_DIR = directory / "faces"
    face_recognizer.GALLERY_FILE = directory / "faces" / "gallery.npz"
    animal_recognizer.ANIMALS_DIR = directory / "animals"
    animal_recognizer.GALLERY_FILE = directory / "animals" / "gallery.npz"
    ai.gallery_root = directory
    _CONFIG_FILE = directory / "email_config.json"
    _KEY_FILE = directory / ".secret.key"
    email_config = {"enabled": False, "sender": "", "receiver": "", "password": ""}
    if _CONFIG_FILE.exists():
        saved = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
        if not isinstance(saved, dict):
            raise ValueError("Invalid email settings in data directory")
        password = saved.get("password", "")
        if password:
            if not _KEY_FILE.exists():
                raise ValueError("Email encryption key is missing from data directory")
            password = _get_cipher().decrypt(password.encode()).decode()
        email_config.update(sender=saved.get("sender", ""), receiver=saved.get("receiver", ""), password=password)


# ── WebSocket client ─────────────────────────────────────────────────────────

def _send_ws(data: dict, label: str = "") -> str:
    """Open a fresh WebSocket connection, send data, close."""
    ui_log(f"📡 Sending {label}...")
    try:
        payload = dict(data)
        if cfg.pi.auth_token:
            payload["token"] = cfg.pi.auth_token
        conn = create_connection(cfg.pi.ws_url, timeout=3)
        try:
            initial = json.loads(conn.recv())
            if initial.get("type") != "current_settings":
                raise ValueError("Invalid initial settings from camera server")
            conn.send(json.dumps(payload))
            reply = json.loads(conn.recv())
            if reply.get("type") == "error":
                raise ValueError(reply.get("message", "Camera server rejected the command"))
            if reply.get("type") != "current_settings":
                raise ValueError("Camera server did not acknowledge settings")
            if any(reply.get("data", {}).get(k) != v for k, v in data.get("data", {}).items()):
                raise ValueError("Camera server rejected the requested settings")
        finally:
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
    with _EMAIL_LOCK:
        email_config["enabled"] = enabled
    ui_log(f"🔔 Alerts set to: {enabled}")


def change_model_logic(model_name: str, task_mode: str, sam3_prompt: str) -> str:
    """Switch the active YOLO/SAM model and task type."""
    return ai.load_segmentation_model(model_name, task_mode, sam3_prompt)


def update_sam3_prompt_logic(prompt: str) -> str:
    """Update the concepts SAM3 should locate and segment."""
    return ai.set_sam3_prompts(prompt)


# ── Shared state ─────────────────────────────────────────────────────────────

class AppState:
    """Thread-safe shared state between capture, AI, and UI threads."""

    def __init__(self) -> None:
        self.conf: float = 0.5
        self.last_frame: np.ndarray | None = None
        self.last_processed: np.ndarray | None = None
        self.lock = threading.Lock()
        self.last_res: tuple[int, int] = (0, 0)
        self.frame_id: int = 0
        self.processed_id: int = -1
        self.track_ids: list[int] = []
        self.detected_classes: list[int] = []
        self.last_info: str = "Model loading..."
        self.ai_enabled = True
        self.last_frame_time = 0.0
        self.last_processed_time = 0.0
        self.connection_status = "Connexion à la caméra…"
        self.frame_times: deque[float] = deque(maxlen=120)


state = AppState()

# ── Video capture thread (Pi stream) ─────────────────────────────────────────
_STOP = threading.Event()
_RECONNECT = threading.Event()


def read_stream() -> None:
    """Background thread: reads frames from the Pi TCP stream.

    Only active when cam_source_mode == 'pi'; sleeps otherwise.
    Auto-reconnects on stream loss.
    """
    cap: cv2.VideoCapture | None = None

    def connect() -> cv2.VideoCapture:
        nonlocal cap
        _RECONNECT.clear()
        if cap is not None:
            try:
                cap.release()
            except Exception as e:
                log.debug("Error releasing previous capture: %s", e)
        ui_log("🖥️ Connecting to Pi camera server...")
        with state.lock:
            state.connection_status = "Connexion à la caméra…"
        cap = open_pi_capture()
        with state.lock:
            state.connection_status = "Réception du flux…" if cap.isOpened() else "Caméra inaccessible — nouvel essai…"
        return cap

    connect()

    while not _STOP.is_set():
        try:
            if cam_source_mode == "webcam":
                time.sleep(0.5)
                continue

            if _RECONNECT.is_set() or cap is None or not cap.isOpened():
                if _STOP.wait(1):
                    break
                connect()
                continue

            ret, frame = cap.read()
            if not ret:
                with state.lock:
                    state.connection_status = "Flux interrompu — reconnexion…"
                ui_log("🔄 Pi stream lost — reconnecting...")
                if _STOP.wait(1):
                    break
                connect()
                continue

            with state.lock:
                state.last_frame = frame
                state.frame_id += 1
                state.last_frame_time = time.monotonic()
                state.frame_times.append(state.last_frame_time)
                state.connection_status = "Connectée"

        except Exception as e:
            ui_log(f"⚠️ Stream error: {e}")
            if _STOP.wait(1):
                break
            connect()

    if cap is not None:
        cap.release()


# ── AI processing thread ────────────────────────────────────────────────────

def ai_loop() -> None:
    """Dedicated thread: pulls latest raw frame, runs AI, stores result.

    Runs as fast as the GPU allows, completely decoupled from the Gradio
    timer refresh rate.
    """
    while not _STOP.is_set():
        try:
            with state.lock:
                if not state.ai_enabled or state.last_frame is None or state.frame_id == state.processed_id:
                    wait_for_new = True
                else:
                    wait_for_new = False
                    frame = state.last_frame.copy()
                    current_id = state.frame_id

            if wait_for_new:
                time.sleep(0.005)
                continue

            if ai.tracking_enabled:
                processed, info, detected_classes, track_ids = ai.track_frame(
                    frame, conf=state.conf
                )
            else:
                track_ids = []
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
                state.last_info = info
                state.processed_id = current_id
                state.track_ids = track_ids
                state.detected_classes = detected_classes
                state.last_processed_time = time.monotonic()

        except Exception as e:
            ui_log(f"[AI error] {e}")
            time.sleep(0.1)


_threads_started = False
_background_threads: list[threading.Thread] = []


def start_background_threads() -> None:
    """Start the capture and AI worker threads (idempotent).

    Called from the main entry point so that merely importing this module
    (tests, tooling) never opens sockets or spawns workers.
    """
    global _threads_started
    if _threads_started:
        return
    _threads_started = True
    ai.start_loader()
    for target, name in ((read_stream, "stream-reader"), (ai_loop, "ai-loop")):
        worker = threading.Thread(target=target, daemon=True, name=name)
        _background_threads.append(worker)
        worker.start()


# ── MJPEG stream generator ──────────────────────────────────────────────────

async def generate_mjpeg():
    """Generator yielding MJPEG frames for the /stream endpoint."""
    last_sent_id = -1
    while not _STOP.is_set():
        with state.lock:
            if state.last_processed is None or state.processed_id == last_sent_id:
                frame_to_send = None
            else:
                frame_to_send = state.last_processed
                last_sent_id = state.processed_id

        if frame_to_send is None:
            await asyncio.sleep(0.01)
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
    return ai.describe_scene(pil_img, message, history=history)


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


def configure_tracking(enabled: bool, tracker: str) -> str:
    """Apply tracking as one operation shared by the GUI and automation."""
    name = tracker.lower().replace("-", "").removesuffix(".yaml")
    if name not in {"botsort", "bytetrack"}:
        raise ValueError("Unknown tracker")
    if ai.tracker != name + ".yaml":
        ai.set_tracker(name)
    toggle_tracking(enabled)
    return f"Suivi {'activé' if enabled else 'désactivé'} — {name}"


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
