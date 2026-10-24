"""Exercise every command through the real CLI process, HTTP, Qt and worker.

Camera/ML/SMTP boundaries are simulated here; persistence, PNG, validation,
transport, jobs and GUI synchronization are real. Live qualification is opt-in.
"""

import json
import os
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

import ai as ai_module
import camera_core as core
import desktop
import desktop_backend as backend
from config import Settings
from control_api import CommandBus, ControlClient, ControlServer, DEFINITIONS

CLI = Path(__file__).resolve().parents[1] / "camera_cli.py"


@pytest.fixture
def controlled_app(tmp_path, monkeypatch):
    qt = QApplication.instance() or QApplication([])
    settings = Settings()
    settings.desktop.profiles = {"local": {"host": "192.0.2.10"}, "tailscale": {"host": "camera.example.invalid"}}
    monkeypatch.setattr(core, "cfg", settings)
    monkeypatch.setattr(desktop, "cfg", settings)
    monkeypatch.setattr(backend, "cfg", settings)
    monkeypatch.setattr(core, "_STOP", threading.Event())
    monkeypatch.setattr(core, "_RECONNECT", threading.Event())
    monkeypatch.setattr(core, "_ROOT", tmp_path)
    monkeypatch.setattr(core, "_CONFIG_FILE", tmp_path / "email_config.json")
    monkeypatch.setattr(core, "_KEY_FILE", tmp_path / ".secret.key")
    monkeypatch.setattr(core, "email_config", {"enabled": False, "sender": "", "receiver": "", "password": ""})
    state = core.AppState()
    state.last_frame = np.zeros((24, 32, 3), np.uint8)
    state.last_processed = state.last_frame.copy()
    state.last_frame_time = time.monotonic()
    state.frame_id = 4
    state.processed_id = 4
    monkeypatch.setattr(core, "state", state)
    processor = ai_module.AIProcessor(autoload=False)
    processor.gallery_root = tmp_path
    monkeypatch.setattr(core, "ai", processor)
    for attribute in ("_face_recognizer", "_animal_recognizer"):
        recognizer = Mock()
        recognizer.gallery_status.return_value = "Fixture gallery"
        recognizer.build_gallery.return_value = "✅ Gallery built"
        recognizer.add_person.return_value = "✅ Person imported"
        recognizer.add_animal.return_value = "✅ Animal imported"
        setattr(processor, attribute, recognizer)
    monkeypatch.setattr(core, "ask_vlm", lambda question, history: "A test scene.")
    smtp = Mock()
    monkeypatch.setattr(core.notifier, "send_email_sync", smtp)
    camera = {"width": 640, "height": 480, "fps": 30, "bitrate": 2000000}

    def camera_settings(updates=None):
        camera.update(updates or {})
        return dict(camera)

    monkeypatch.setattr(backend, "camera_settings", camera_settings)
    monkeypatch.setattr(backend, "restart_camera", lambda: dict(camera))
    monkeypatch.setattr(backend, "tailscale_devices", lambda: [{"name": "camera", "host": "192.0.2.20", "online": True}])
    save_connection = backend.save_connection
    monkeypatch.setattr(backend, "save_connection", lambda *a, **kw: save_connection(*a, path=tmp_path / "config.yaml", **kw))

    def load(name, task="Detection", concepts=""):
        processor.segment_model_name = name
        processor.model_type = "sam3" if name == "sam3.pt" else "yolo"
        processor.segment_model = object()
        return "Loaded " + name

    monkeypatch.setattr(core, "change_model_logic", load)
    monkeypatch.setattr(processor, "export_to_tensorrt", lambda: "✅ Export success")
    descriptor = tmp_path / ".camera-control.json"
    window = desktop.CameraWindow(start_camera=False, enable_control=True, control_path=descriptor)
    window.show()

    def call(name, params=None, *, no_wait=False, timeout=10, expect=0, stdin=False):
        args = [sys.executable, str(CLI), name, "--state-file", str(descriptor), "--timeout", str(timeout)]
        if stdin:
            args.append("--stdin")
        else:
            args.extend(["--json", json.dumps(params or {})])
        if no_wait:
            args.append("--no-wait")
        process = subprocess.Popen(args, stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
        captured = []
        reader = threading.Thread(target=lambda: captured.extend(process.communicate(json.dumps(params or {}) if stdin else None)), daemon=True)
        reader.start()
        deadline = time.monotonic() + 20
        while process.poll() is None and time.monotonic() < deadline:
            qt.processEvents()
            time.sleep(.005)
        if process.poll() is None:
            process.kill()
            pytest.fail("CLI did not finish")
        reader.join(timeout=2)
        assert len(captured) == 2
        output, errors = captured
        assert not errors, errors
        assert process.returncode == expect, output
        result = json.loads(output)
        assert result["ok"] is (expect == 0), output
        return result

    yield SimpleNamespace(window=window, qt=qt, call=call, descriptor=descriptor,
                          processor=processor, camera=camera, smtp=smtp, root=tmp_path)
    end = time.monotonic() + 3
    while window.worker.busy and time.monotonic() < end:
        qt.processEvents()
        time.sleep(.01)
    window.close()
    qt.processEvents()


def test_every_function_via_cli(controlled_app):
    app = controlled_app
    photo = app.root / "fixture.png"
    cv2.imwrite(str(photo), np.zeros((24, 32, 3), np.uint8))
    cases = [
        ("schema.get", {}), ("app.start", {}), ("status.get", {}),
        ("camera.get", {}), ("camera.set", {"fps": 15}), ("camera.reconnect", {}), ("camera.restart", {}),
        ("network.get", {}), ("network.get", {"private": True}),
        ("network.use", {"profile": "local"}),
        ("network.configure", {"host": "camera.example.invalid", "profile": "tailscale"}),
        ("network.tailscale", {}), ("ai.set", {"enabled": False}), ("ai.set", {"enabled": True}),
        ("detection.set", {"confidence": .65, "classes": [0, 15, 16]}),
        ("model.get", {}), ("model.load", {"model": "fixture.pt", "task": "Segmentation"}),
        ("model.export", {}), ("sam.concepts", {"concepts": "person, cat"}),
        ("tracking.set", {"enabled": True, "tracker": "ByteTrack"}),
        ("tracking.set", {"enabled": False, "tracker": "BoT-SORT"}),
        ("alerts.get", {}),
        ("alerts.configure", {"enabled": False, "sender": "sender@example.test", "receiver": "receiver@example.test", "password": "fixture"}),
        ("alerts.enable", {"enabled": True}), ("alerts.enable", {"enabled": False}),
        ("alerts.test", {}), ("alerts.test", {"send": True}),
        ("assistant.ask", {"message": "Describe"}), ("assistant.history", {}), ("assistant.clear", {}),
        ("frame.save", {"path": str(app.root / "capture.png")}),
        ("view.get", {}), ("view.set", {"paused": True, "overlay": False, "fullscreen": True}),
        ("view.set", {"paused": False, "overlay": True, "fullscreen": False}), ("logs.get", {}),
    ]
    for kind in ("faces", "animals"):
        cases.extend([
            ("recognition.set", {"kind": kind, "enabled": True, "threshold": .7}),
            ("gallery.status", {"kind": kind}),
            ("gallery.import", {"kind": kind, "name": "Fixture", "files": [str(photo)]}),
            ("gallery.rebuild", {"kind": kind}),
            ("gallery.learn", {"kind": kind, "name": "Fixture", "count": 2}),
            ("gallery.stop", {"kind": kind}),
            ("recognition.set", {"kind": kind, "enabled": False}),
        ])
    seen = set()
    for name, params in cases:
        result = app.call(name, params, stdin="password" in params)
        seen.add(name)
        if name == "tracking.set":
            assert app.processor.tracking_enabled is params["enabled"]
            assert result["result"]["requested"] is params["enabled"]
        if name == "assistant.history":
            assert result["result"][1]["content"] == "A test scene."
        if name == "network.get" and not params:
            assert "host" not in result["result"]
    jobs = app.call("jobs.list")["result"]
    assert jobs and all("params" not in job for job in jobs)
    app.call("job.get", {"id": jobs[-1]["id"]})
    seen.update({"jobs.list", "job.get"})
    assert cv2.imread(str(app.root / "capture.png")).shape == (24, 32, 3)
    stored = json.loads((app.root / "email_config.json").read_text())
    assert stored["password"] != "fixture"  # noqa: S105 - fixture
    app.smtp.assert_called_once()
    assert app.camera["fps"] == 15
    assert app.window.commands.history == []
    app.call("app.quit")
    seen.add("app.quit")
    assert seen == set(DEFINITIONS), "Every advertised operation must be exercised from the CLI"


def test_cli_jobs_fail_truthfully_and_failed_chat_does_not_pollute_history(controlled_app, monkeypatch):
    app = controlled_app
    monkeypatch.setattr(core, "ask_vlm", lambda *args: "VLM Error: fixture failure")
    result = app.call("assistant.ask", {"message": "Describe"}, expect=1)
    assert result["error"]["code"] == "COMMAND_FAILED"
    assert app.window.commands.history == []
    app.call("frame.save", {"path": str(app.root / "absent" / "image.png")}, expect=1)
    app.call("camera.set", {"fps": True}, expect=2)
    app.call("alerts.configure", {"sender": "sender@example.test", "receiver": "receiver@example.test", "password": "fixture"}, expect=2)
    app.call("app.start", {"data_dir": str(app.root / "other-data")}, expect=2)
    monkeypatch.setattr(core, "ask_vlm", lambda *args: "")
    app.call("assistant.ask", {"message": "Describe"}, expect=1)
    assert app.window.commands.history == []


def test_cli_capture_uses_current_raw_frame_when_annotations_are_stale(controlled_app):
    app = controlled_app
    core.state.last_frame[:] = 17
    core.state.last_processed[:] = 91
    core.state.last_processed_time = time.monotonic() - 5
    target = app.root / "fresh.png"
    app.call("frame.save", {"path": str(target)})
    assert (cv2.imread(str(target)) == 17).all()
    app.call("frame.save", {"path": str(target)}, expect=1)
    core.state.last_processed_time = time.monotonic()
    app.call("frame.save", {"path": str(target), "overwrite": True})
    assert (cv2.imread(str(target)) == 91).all()


def test_cli_invalid_argument_does_not_echo_private_text():
    process = subprocess.run([sys.executable, str(CLI), "private-argument-fixture"],
                             capture_output=True, text=True, encoding="utf-8", timeout=10)
    assert process.returncode == 2
    assert "private-argument-fixture" not in process.stdout + process.stderr
    assert json.loads(process.stdout)["error"]["code"] == "CLI_ERROR"


def test_cli_timeout_keeps_job_and_preview_controls_work_during_long_job(controlled_app, monkeypatch):
    app = controlled_app
    gate = threading.Event()
    monkeypatch.setattr(core, "change_model_logic", lambda *args: gate.wait(5) or "Loaded")
    try:
        result = app.call("model.load", {"model": "fixture.pt"}, timeout=.2, expect=1)
        assert result["error"]["code"] == "WAIT_TIMEOUT"
        app.call("view.set", {"paused": True})
        assert app.window.pause.isChecked()
        assert app.call("job.get", {"id": result["job_id"]})["result"]["status"] == "running"
    finally:
        gate.set()


def test_loopback_transport_auth_validation_and_private_job_inputs(tmp_path):
    bus = CommandBus(lambda: {"ready": True})
    descriptor = tmp_path / ".camera-control.json"
    server = ControlServer(bus, descriptor)
    try:
        client = ControlClient(descriptor)
        assert client.get("/status")["result"] == {"ready": True}
        client.token = secrets.token_urlsafe()
        with pytest.raises(HTTPError) as failure:
            client.get("/status")
        assert failure.value.code == 401
        client = ControlClient(descriptor)
        with pytest.raises(ValueError):
            client.get("@example.invalid/")
        job = bus.submit("alerts.configure", {"sender": "sender@example.test", "receiver": "receiver@example.test", "password": "fixture"})
        assert "fixture" not in json.dumps(job)
        assert bus.take()[0] == job["id"]
        bus.finish(job["id"], error_message="Fixture error")
        assert "fixture" not in json.dumps(bus.get(job["id"]))
        for _ in range(32):
            bus.submit("ai.set", {"enabled": False})
        with pytest.raises(ValueError):
            bus.submit("ai.set", {"enabled": True})
    finally:
        server.close()
    assert not descriptor.exists()


def test_cli_cold_start_and_shutdown(tmp_path):
    descriptor = tmp_path / ".camera-control.json"
    command = [sys.executable, str(CLI), "app.start", "--state-file", str(descriptor), "--timeout", "25", "--json",
               json.dumps({"no_camera": True, "data_dir": str(tmp_path / "data")})]
    process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=35)
    try:
        assert process.returncode == 0, process.stdout + process.stderr
        assert json.loads(process.stdout)["result"]["running"]
        client = ControlClient(descriptor)
        assert client.call("status.get", {})["ok"]
    finally:
        if descriptor.exists():
            ControlClient(descriptor).call("app.quit", {})
    deadline = time.monotonic() + 5
    while descriptor.exists() and time.monotonic() < deadline:
        time.sleep(.05)
    assert not descriptor.exists()


def test_control_start_never_replaces_an_unrelated_existing_file(tmp_path):
    target = tmp_path / "settings.json"
    original = b'{"setting":"preserve"}'
    target.write_bytes(original)
    with pytest.raises(RuntimeError, match="not a valid descriptor"):
        ControlServer(CommandBus(lambda: {}), target)
    assert target.read_bytes() == original


def test_control_start_recovers_a_valid_stale_descriptor(tmp_path):
    target = tmp_path / ".camera-control.json"
    first = ControlServer(CommandBus(lambda: {}), target)
    stale = target.read_bytes()
    first.close()
    target.write_bytes(stale)
    second = ControlServer(CommandBus(lambda: {}), target)
    try:
        assert ControlClient(target).get("/status")["ok"]
    finally:
        second.close()
