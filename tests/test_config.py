"""Unit tests for config.py (no heavy dependencies required)."""

import pytest

import config


def test_defaults_are_secure():
    settings = config.Settings()
    assert settings.ui.share is False, "share must default to off"
    assert settings.ui.auth_username == ""
    assert settings.server.auth_token == ""
    assert settings.pi.auth_token == ""


def test_apply_env_overrides_types(monkeypatch):
    monkeypatch.setenv("CAMERA__PI__HOST", "192.0.2.20")
    monkeypatch.setenv("CAMERA__PI__VIDEO_PORT", "9000")
    monkeypatch.setenv("CAMERA__UI__SHARE", "true")
    monkeypatch.setenv("CAMERA__AI__FACE_THRESHOLD", "0.42")
    monkeypatch.setenv("OTHER_VAR", "ignored")

    data = {
        "pi": {"host": "1.2.3.4", "video_port": 8888},
        "ui": {"share": False},
        "ai": {"face_threshold": 0.35},
    }
    result = config._apply_env_overrides(data)

    assert result["pi"]["host"] == "192.0.2.20"
    assert result["pi"]["video_port"] == 9000
    assert isinstance(result["pi"]["video_port"], int)
    assert result["ui"]["share"] is True
    assert result["ai"]["face_threshold"] == pytest.approx(0.42)


def test_env_override_creates_missing_sections(monkeypatch):
    monkeypatch.setenv("CAMERA__PI__HOST", "198.51.100.20")
    result = config._apply_env_overrides({})
    assert result["pi"]["host"] == "198.51.100.20"


def test_alert_class_map():
    alerts = config.AlertSettings()
    assert alerts.class_map() == [(0, "person"), (15, "cat"), (16, "dog")]


def test_valid_resolution_set():
    camera = config.CameraSettings()
    assert (640, 480) in camera.valid_resolution_set()
    assert (800, 600) not in camera.valid_resolution_set()


def test_urls_built_from_host():
    pi = config.PiSettings(host="example.host", video_port=1234, ws_port=5678)
    assert pi.video_url == "tcp://example.host:1234"
    assert pi.ws_url == "ws://example.host:5678"
