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


def test_env_types_do_not_depend_on_yaml_keys(monkeypatch):
    monkeypatch.setenv("CAMERA__UI__SHARE", "false")
    monkeypatch.setenv("CAMERA__PI__VIDEO_PORT", "9001")
    monkeypatch.setenv("CAMERA__AI__FACE_THRESHOLD", "0.6")
    monkeypatch.setenv("CAMERA__AI__ANIMAL_CLASSES", "[15, 16]")
    monkeypatch.setenv("CAMERA__UI__AUTH_PASSWORD", "012345")
    data = config._apply_env_overrides({})
    assert data["ui"]["share"] is False
    assert data["ui"]["auth_password"] == "012345"  # noqa: S105 - test fixture
    assert data["pi"]["video_port"] == 9001
    assert data["ai"]["face_threshold"] == 0.6
    assert data["ai"]["animal_classes"] == [15, 16]


@pytest.mark.parametrize("key,value", [
    ("CAMERA__UI__SHARE", "flase"),
    ("CAMERA__UI__SHRAE", "true"),
    ("CAMERA__UNKNOWN__HOST", "host"),
    ("CAMERA__PI__VIDEO_PORT", "invalid"),
])
def test_bad_overrides_fail_without_leaking_values(monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        config._apply_env_overrides({})


@pytest.mark.parametrize("section,key,value", [
    ("ui", "share", "false"), ("ui", "share", True),
    ("ui", "auth_password", "secret"),
    ("ai", "face_threshold", 1.1), ("ai", "animal_threshold", float("nan")),
    ("ai", "detection_classes", [-1]), ("pi", "video_port", True),
    ("pi", "video_port", 70000), ("server", "default_fps", 0),
    ("camera", "valid_resolutions", [[640]]),
    ("alerts", "alert_classes", [{"id": True, "name": "person"}]),
])
def test_invalid_settings_rejected(section, key, value):
    settings = config.Settings()
    setattr(getattr(settings, section), key, value)
    with pytest.raises(ValueError):
        config.validate_settings(settings)


def test_load_rejects_unknown_or_malformed_yaml(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    monkeypatch.setattr(config, "_CONFIG_FILE", path)
    for source in ("[]", "pi: 42", "ui:\n  typo: true"):
        path.write_text(source, encoding="utf-8")
        with pytest.raises(ValueError):
            config.load_settings()
