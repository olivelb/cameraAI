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
import math
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

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
    capture_open_timeout_ms: int = 7000
    capture_read_timeout_ms: int = 3000


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

    def stream_resolution_set(self) -> set[tuple[int, int]]:
        """Restrict output to the Pi Zero's H.264 encoder envelope.

        Sensor still-photo modes such as 2592x1944 exceed this envelope.
        """
        return {(w, h) for w, h in self.valid_resolution_set() if w <= 1920 and h <= 1080}


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
    section_types = get_type_hints(Settings)
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(f"{prefix}__"):
            continue
        parts = env_key[len(prefix) + 2:].lower().split("__")
        if len(parts) != 2 or parts[0] not in section_types:
            raise ValueError(f"Unknown configuration override: {env_key}")
        section, key = parts
        hints = get_type_hints(section_types[section])
        if key not in hints:
            raise ValueError(f"Unknown configuration override: {env_key}")
        target = data.setdefault(section, {})
        if not isinstance(target, dict):
            raise ValueError(f"Configuration section {section} must be a mapping")
        kind = hints[key]
        try:
            if kind is str:
                value = env_val  # Secrets and numeric hostnames stay strings.
            elif kind is bool:
                choices = {"true": True, "1": True, "yes": True,
                           "false": False, "0": False, "no": False}
                value = choices[env_val.strip().lower()]
            elif kind in (int, float):
                value = kind(env_val)
            else:
                value = yaml.safe_load(env_val)
        except (ValueError, KeyError, yaml.YAMLError):
            raise ValueError(f"Invalid value for {section}.{key}") from None
        target[key] = value
    return data


def _matches_type(value: Any, expected: Any) -> bool:
    origin = get_origin(expected)
    args = get_args(expected)
    if expected is Any:
        return True
    if origin is list:
        return isinstance(value, list) and all(_matches_type(v, args[0]) for v in value)
    if origin is dict:
        return isinstance(value, dict) and all(
            _matches_type(k, args[0]) and _matches_type(v, args[1])
            for k, v in value.items()
        )
    if origin is types.UnionType:
        return any(_matches_type(value, kind) for kind in args)
    if expected is float:
        return type(value) in (int, float) and math.isfinite(value)
    return type(value) is expected


def validate_settings(settings: Settings) -> None:
    """Reject invalid types/ranges before they reach sockets, threads or models."""
    for section in Settings.__dataclass_fields__:
        values = getattr(settings, section)
        for key, kind in get_type_hints(type(values)).items():
            value = getattr(values, key)
            if not _matches_type(value, kind):
                raise ValueError(f"Invalid type for {section}.{key}")
            path = f"{section}.{key}"
            if key.endswith("port") and not 1 <= value <= 65535:
                raise ValueError(f"{path} must be between 1 and 65535")
            if type(value) in (int, float) and key not in ("person_class",) and value <= 0:
                raise ValueError(f"{path} must be positive")
    for key in ("face_threshold", "animal_threshold", "iou_threshold"):
        if not 0 < getattr(settings.ai, key) <= 1:
            raise ValueError(f"ai.{key} must be in (0, 1]")
    if not 1 <= settings.stream.jpeg_quality <= 100:
        raise ValueError("stream.jpeg_quality must be in [1, 100]")
    if not 1 <= settings.pi.fps_min <= settings.pi.fps_default <= settings.pi.fps_max <= 30:
        raise ValueError("pi FPS must satisfy 1 <= min <= default <= max <= 30")
    if not 1 <= settings.server.default_fps <= 30:
        raise ValueError("server.default_fps must be in [1, 30]")
    if not 100_000 <= settings.server.default_bitrate <= 25_000_000:
        raise ValueError("server.default_bitrate must be in [100000, 25000000]")
    resolutions = settings.camera.valid_resolutions
    if not resolutions or any(len(r) != 2 or any(v <= 0 for v in r) for r in resolutions):
        raise ValueError("camera.valid_resolutions must contain positive [width, height] pairs")
    valid = settings.camera.valid_resolution_set()
    if tuple(settings.pi.default_resolution) not in valid:
        raise ValueError("pi.default_resolution is not in camera.valid_resolutions")
    if (settings.server.default_width, settings.server.default_height) not in settings.camera.stream_resolution_set():
        raise ValueError("Server default resolution is outside supported H.264 video modes")
    if tuple(settings.pi.default_resolution) not in settings.camera.stream_resolution_set():
        raise ValueError("Pi default resolution is outside supported H.264 video modes")
    if settings.server.video_port == settings.server.ws_port:
        raise ValueError("Server video and WebSocket ports must differ")
    ids = settings.ai.detection_classes + settings.ai.animal_classes + [settings.ai.person_class]
    if any(v < 0 for v in ids):
        raise ValueError("AI class IDs must be nonnegative")
    for entry in settings.alerts.alert_classes:
        if (set(entry) != {"id", "name"} or type(entry["id"]) is not int
                or entry["id"] < 0 or not isinstance(entry["name"], str) or not entry["name"]):
            raise ValueError("alerts.alert_classes must contain {id: nonnegative int, name: string}")
    if bool(settings.ui.auth_username) != bool(settings.ui.auth_password):
        raise ValueError("Set both ui.auth_username and ui.auth_password")
    if settings.ui.share and not settings.ui.auth_username:
        raise ValueError("Public sharing requires ui.auth_username and ui.auth_password")


def load_settings() -> Settings:
    """Load settings from config.yaml → env overrides → Settings dataclass."""
    config_path = _CONFIG_FILE if _CONFIG_FILE.exists() else _EXAMPLE_FILE

    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
            if raw is None:
                raw = {}
        log.info("Loaded config from %s", config_path.name)
    else:
        log.warning("No config file found — using defaults")
        raw = {}

    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")
    raw = _apply_env_overrides(raw)

    # Build typed Settings
    settings = Settings()
    sections = get_type_hints(Settings)
    for section, values in raw.items():
        if section not in sections:
            raise ValueError(f"Unknown configuration section: {section}")
        kind = sections[section]
        if not isinstance(values, dict):
            raise ValueError(f"Configuration section {section} must be a mapping")
        if any(k not in kind.__dataclass_fields__ for k in values):
            raise ValueError(f"Unknown configuration key in {section}")
        setattr(settings, section, kind(**values))
    validate_settings(settings)
    return settings


# ── Module-level singleton ────────────────────────────────────────────────────
cfg = load_settings()
