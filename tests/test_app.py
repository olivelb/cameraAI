"""Construct the real UI, keeping all writes in a temporary test directory."""

import asyncio
import json

import cv2
import numpy as np
import pytest


@pytest.fixture
def application(tmp_path, monkeypatch):
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    import app
    monkeypatch.setattr(app, "_ROOT", tmp_path)
    monkeypatch.setattr(app, "_KEY_FILE", tmp_path / ".secret.key")
    monkeypatch.setattr(app, "_CONFIG_FILE", tmp_path / "email_config.json")
    monkeypatch.setattr(app, "email_config", {"enabled": False, "sender": "", "password": "", "receiver": ""})
    return app


def test_ui_constructs_without_starting_ai_loader(application):
    assert not application.ai._loader_started
    assert not application._threads_started


def test_email_save_encrypts_password_and_preserves_on_empty_input(application):
    app = application
    app.save_email_settings(False, "sender@example.test", "test-password", "receiver@example.test")
    stored = json.loads(app._CONFIG_FILE.read_text())
    assert stored["password"] != app.email_config["password"]
    assert app._get_cipher().decrypt(stored["password"].encode()).decode() == "test-password"
    app.save_email_settings(False, "sender@example.test", "", "receiver@example.test")
    assert app.email_config["password"] == "test-password"  # noqa: S105 - fixture


def test_failed_email_save_keeps_previous_memory_and_file(application, monkeypatch):
    app = application
    app.save_email_settings(False, "old", "test-password", "receiver")
    original = app._CONFIG_FILE.read_bytes()

    def fail(*args):
        raise OSError("disk error")

    monkeypatch.setattr(app.os, "replace", fail)
    with pytest.raises(OSError):
        app.save_email_settings(True, "new", "changed", "receiver")
    assert app.email_config["sender"] == "old"
    assert app._CONFIG_FILE.read_bytes() == original


def test_ws_rejected_update_is_not_reported_as_success(application, monkeypatch):
    class Connection:
        closed = False

        def recv(self):
            return json.dumps({"type": "current_settings", "data": {"fps": 30}})

        def send(self, data):
            pass

        def close(self):
            self.closed = True

    connection = Connection()
    monkeypatch.setattr(application, "create_connection", lambda *a, **kw: connection)
    assert "rejected" in application.send_ws_command("fps", 15)
    assert connection.closed


def test_mjpeg_encodes_frame_and_can_close_without_new_frames(application, monkeypatch):
    app = application
    monkeypatch.setattr(app.state, "last_processed", np.zeros((10, 10, 3), np.uint8))
    monkeypatch.setattr(app.state, "processed_id", 5)

    async def exercise():
        stream = app.generate_mjpeg()
        frame = await anext(stream)
        assert frame.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
        image = cv2.imdecode(np.frombuffer(frame.split(b"\r\n\r\n", 1)[1], np.uint8), cv2.IMREAD_COLOR)
        assert image.shape == (10, 10, 3)
        await stream.aclose()
    asyncio.run(exercise())
