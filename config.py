"""
config.py — Centralized configuration for the camera-client application.

Loads settings from config.yaml (or config.example.yaml as fallback),
with optional environment variable overrides using the CAMERA__ prefix.

Usage:
    from config import cfg

    print(cfg.pi.host)           # "camera.example.invalid"
    print(cfg.ai.vlm_model_id)  # "google/gemma-4-E2B-it"
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent
_CONFIG_FILE = _CONFIG_DIR / "config.yaml"
_EXAMPLE_FILE = _CONFIG_DIR / "config.example.yaml"


# ── Typed settings dataclasses ────────────────────────────────────────────────

@dataclass
class PiSettings:
    host: str = "camera.example.invalid"
    video_port: int = 8888
    ws_port: int = 8889
    auth_token: str = ""
    default_resolution: list[int] = field(default_factory=lambda: [640, 480])
    fps_min: int = 5
    fps_max: int = 30
    fps_default: int = 30

    @property
    def video_url(self) -> str:
        return f"tcp://{self.host}:{self.video_port}"

    @property
    def ws_url(self) -> str:
        return f"ws://{self.host}:{self.ws_port}"


@dataclass
class ServerSettings:
    video_port: int = 8888
    ws_port: int = 8889
    auth_token: str = ""
    default_width: int = 640
    default_height: int = 480
    default_fps: int = 30
    default_bitrate: int = 2_000_000
    pid_file: str = "/tmp/camera_server.pid"


@dataclass
class AISettings:
    default_model: str = "yolo26n.pt"
    vlm_model_id: str = "google/gemma-4-E2B-it"
    animal_embed_model: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    face_threshold: float = 0.35
    animal_threshold: float = 0.85
    face_history_len: int = 5
    animal_history_len: int = 5
    detection_classes: list[int] = field(default_factory=lambda: [0, 15, 16])
    iou_threshold: float = 0.4
    person_class: int = 0
    animal_classes: list[int] = field(default_factory=lambda: [15, 16])
    face_recheck_interval: int = 10
    animal_recheck_interval: int = 10
    max_cached_models: int = 2


@dataclass
class AlertClassDef:
    id: int
    name: str


@dataclass
class AlertSettings:
    cooldown_seconds: int = 300
    alert_classes: list[dict[str, Any]] = field(default_factory=lambda: [
        {"id": 0, "name": "person"},
        {"id": 15, "name": "cat"},
        {"id": 16, "name": "dog"},
    ])

    def class_map(self) -> list[tuple[int, str]]:
        """Return alert classes as [(id, name), ...]."""
        return [(c["id"], c["name"]) for c in self.alert_classes]


@dataclass
class EmailSettings:
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    timeout_seconds: int = 30


@dataclass
class StreamSettings:
    jpeg_quality: int = 80


@dataclass
class UISettings:
    server_host: str = "127.0.0.1"
    server_port: int = 7860
    share: bool = False
    auth_username: str = ""
    auth_password: str = ""
    log_max_lines: int = 20
    log_refresh_interval: int = 1


@dataclass
class CameraSettings:
    valid_resolutions: list[list[int]] = field(default_factory=lambda: [
        [640, 480],
        [1296, 972],
        [1920, 1080],
        [2592, 1944],
    ])

    def valid_resolution_set(self) -> set[tuple[int, int]]:
        return {tuple(r) for r in self.valid_resolutions}


@dataclass
class Settings:
    pi: PiSettings = field(default_factory=PiSettings)
    server: ServerSettings = field(default_factory=ServerSettings)
    ai: AISettings = field(default_factory=AISettings)
    alerts: AlertSettings = field(default_factory=AlertSettings)
    email: EmailSettings = field(default_factory=EmailSettings)
    stream: StreamSettings = field(default_factory=StreamSettings)
    ui: UISettings = field(default_factory=UISettings)
    camera: CameraSettings = field(default_factory=CameraSettings)


# ── Loader ────────────────────────────────────────────────────────────────────

def _apply_env_overrides(data: dict, prefix: str = "CAMERA") -> dict:
    """
    Apply environment variable overrides.

    CAMERA__PI__HOST=192.0.2.30  →  data["pi"]["host"] = "192.0.2.30"
    """
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(f"{prefix}__"):
            continue
        parts = env_key[len(prefix) + 2:].lower().split("__")
        target = data
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        # Try to preserve type
        final_key = parts[-1]
        existing = target.get(final_key)
        if isinstance(existing, bool):
            target[final_key] = env_val.lower() in ("1", "true", "yes")
        elif isinstance(existing, int):
            target[final_key] = int(env_val)
        elif isinstance(existing, float):
            target[final_key] = float(env_val)
        else:
            target[final_key] = env_val
    return data


def load_settings() -> Settings:
    """Load settings from config.yaml → env overrides → Settings dataclass."""
    config_path = _CONFIG_FILE if _CONFIG_FILE.exists() else _EXAMPLE_FILE

    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        log.info("Loaded config from %s", config_path.name)
    else:
        log.warning("No config file found — using defaults")
        raw = {}

    raw = _apply_env_overrides(raw)

    # Build typed Settings
    settings = Settings()
    if "pi" in raw:
        settings.pi = PiSettings(**{k: v for k, v in raw["pi"].items() if k in PiSettings.__dataclass_fields__})
    if "server" in raw:
        settings.server = ServerSettings(**{k: v for k, v in raw["server"].items() if k in ServerSettings.__dataclass_fields__})
    if "ai" in raw:
        settings.ai = AISettings(**{k: v for k, v in raw["ai"].items() if k in AISettings.__dataclass_fields__})
    if "alerts" in raw:
        settings.alerts = AlertSettings(**{k: v for k, v in raw["alerts"].items() if k in AlertSettings.__dataclass_fields__})
    if "email" in raw:
        settings.email = EmailSettings(**{k: v for k, v in raw["email"].items() if k in EmailSettings.__dataclass_fields__})
    if "stream" in raw:
        settings.stream = StreamSettings(**{k: v for k, v in raw["stream"].items() if k in StreamSettings.__dataclass_fields__})
    if "ui" in raw:
        settings.ui = UISettings(**{k: v for k, v in raw["ui"].items() if k in UISettings.__dataclass_fields__})
    if "camera" in raw:
        settings.camera = CameraSettings(**{k: v for k, v in raw["camera"].items() if k in CameraSettings.__dataclass_fields__})

    return settings


# ── Module-level singleton ────────────────────────────────────────────────────
cfg = load_settings()
