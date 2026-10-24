"""Qt event loop, frame ownership, private settings, and command boundaries."""

import copy
import json
import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import yaml
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

import camera_core as core
import desktop_backend as backend
from desktop import CameraWindow, frame_image
import desktop as desktop_module
from config import Settings


@pytest.fixture(scope="module")
def qt():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qt, monkeypatch):
    monkeypatch.setattr(core, "_STOP", threading.Event())
    monkeypatch.setattr(core, "state", core.AppState())
    widget = CameraWindow(start_camera=False)
    widget.show()
    yield widget
    deadline = time.monotonic() + 3
    while widget.worker.busy and time.monotonic() < deadline:
        qt.processEvents()
        time.sleep(.01)
    widget.close()
    qt.processEvents()


def test_image_owns_pixels_and_preserves_bgr_channels():
    pixels = np.zeros((12, 16, 3), dtype=np.uint8)
    pixels[:] = (12, 42, 190)
    image = frame_image(pixels)
    pixels[:] = 0
    color = image.pixelColor(1, 1)
    assert (color.red(), color.green(), color.blue()) == (190, 42, 12)


def test_all_pages_construct_without_models_camera_or_web_server(window):
    assert window.tabs.count() == 7
    assert not core.ai._loader_started
    assert not core._threads_started
    assert window.token.echoMode().name == "Password"
    assert not window.email_enabled.isChecked()


def test_slow_command_does_not_block_qt_or_latest_frame(window, qt):
    gate = threading.Event()
    beats = []
    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: beats.append(time.monotonic()))
    timer.start()
    window.run("Long operation", lambda: gate.wait(2))
    with core.state.lock:
        core.state.last_frame = np.zeros((40, 60, 3), dtype=np.uint8)
        core.state.frame_id = 1
        core.state.last_frame_time = time.monotonic()
    end = time.monotonic() + .25
    while time.monotonic() < end:
        qt.processEvents()
        time.sleep(.005)
    gate.set()
    timer.stop()
    assert len(beats) >= 8
    assert not window.video.image.isNull()
    assert window._last_image_key == (False, 1)


def test_close_waits_for_pending_persistence_without_freezing(window, qt):
    gate = threading.Event()
    window.run("Saving", lambda: gate.wait(2))
    window.close()
    qt.processEvents()
    assert window.isVisible()
    gate.set()
    end = time.monotonic() + 2
    while window.isVisible() and time.monotonic() < end:
        qt.processEvents()
        time.sleep(.01)
    assert not window.isVisible()


def test_worker_rejects_backlog_and_preserves_errors():
    worker = backend.CommandWorker()
    gate = threading.Event()
    assert worker.submit("slow", lambda: gate.wait(1))
    assert not worker.submit("obsolete", lambda: None)
    gate.set()
    assert worker.results.get(timeout=2).label == "slow"
    assert worker.busy  # Until the UI consumes and acknowledges the result.
    worker.acknowledge()

    def fail():
        raise ValueError("visible error")

    assert worker.submit("failing", fail)
    assert worker.results.get(timeout=2).error == "visible error"
    worker.acknowledge()
    worker.close()


def test_connection_save_is_validated_atomic_and_preserves_other_settings(tmp_path, monkeypatch):
    candidate = copy.deepcopy(backend.cfg)
    monkeypatch.setattr(backend, "cfg", candidate)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"ai": {"default_model": "custom.pt"}, "pi": {"auth_token": "fixture"}}))
    backend.save_connection("192.0.2.10", 8888, 8889, "fixture", 7000, 10000, path=path)
    saved = yaml.safe_load(path.read_text())
    assert saved["ai"] == {"default_model": "custom.pt"}
    assert saved["pi"]["auth_token"] == "fixture"  # noqa: S105 - test fixture
    assert candidate.stream.capture_read_timeout_ms == 10000
    previous = path.read_bytes()
    with pytest.raises(ValueError):
        backend.save_connection("http://invalid/", 8888, 8889, "fixture", 7000, 3000, path=path)
    assert path.read_bytes() == previous

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(backend.os, "replace", fail)
    with pytest.raises(OSError):
        backend.save_connection("198.51.100.30", 8888, 8889, "fixture", 7000, 3000, path=path)
    assert candidate.pi.host == "192.0.2.10"
    assert path.read_bytes() == previous


def test_network_profiles_save_atomically_without_copying_tokens(tmp_path, monkeypatch):
    settings = Settings()
    monkeypatch.setattr(backend, "cfg", settings)
    path = tmp_path / "config.yaml"
    path.write_text("ai:\n  default_model: custom.pt\n", encoding="utf-8")
    backend.save_connection("camera.example.invalid", 8888, 8889, "fixture", 12000, 10000,
                            path=path, profile_id="tailscale")
    saved = yaml.safe_load(path.read_text())
    assert saved["desktop"]["active_profile"] == "tailscale"
    assert saved["desktop"]["profiles"]["tailscale"]["host"] == "camera.example.invalid"
    assert "auth_token" not in saved["desktop"]["profiles"]["tailscale"]
    previous = path.read_bytes()

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(backend.os, "replace", fail)
    with pytest.raises(OSError):
        backend.save_connection("192.0.2.10", 8888, 8889, "fixture", 10000, 3000,
                                path=path, profile_id="local")
    assert path.read_bytes() == previous
    assert settings.desktop.active_profile == "tailscale"
    assert "local" not in settings.desktop.profiles


def test_tailscale_profile_switch_restores_endpoint_and_leaves_other_profiles_intact(window, monkeypatch):
    settings = Settings()
    settings.desktop.profiles = {
        "local": {"host": "192.0.2.10", "video_port": 9000, "ws_port": 9001},
        "tailscale": {"host": "camera.example.invalid", "open_timeout_ms": 15000, "read_timeout_ms": 9000},
    }
    monkeypatch.setattr(desktop_module, "cfg", settings)
    window.network_profile.setCurrentText("Tailscale")
    assert window.host.text() == "camera.example.invalid"
    assert window.open_timeout.value() == 15000
    assert window.tailscale_button.isVisible()
    window.network_profile.setCurrentText("Réseau local")
    assert window.host.text() == "192.0.2.10"
    assert window.video_port.value() == 9000
    assert not window.tailscale_button.isVisible()
    window.network_profile.setCurrentText("Tailscale")
    assert window.host.text() == "camera.example.invalid"
    assert settings.desktop.profiles["local"]["video_port"] == 9000


def test_unconfigured_tailscale_profile_never_reuses_gateway_address(window, monkeypatch):
    monkeypatch.setattr(desktop_module, "cfg", Settings())
    window.network_profile.setCurrentText("Tailscale")
    assert window.host.text() == ""
    assert window.open_timeout.value() == 12000
    assert window.read_timeout.value() == 10000


def test_tailscale_discovery_prefers_ipv4_and_excludes_self(tmp_path, monkeypatch):
    from types import SimpleNamespace

    executable = tmp_path / "tailscale.exe"
    executable.touch()
    monkeypatch.setattr(backend.shutil, "which", lambda name: str(executable))
    response = {"BackendState": "Running", "Self": {"HostName": "this-client"}, "Peer": {
        "one": {"HostName": "camera", "DNSName": "camera.example.invalid.",
                "TailscaleIPs": ["::1", "192.0.2.20"], "Online": True},
        "two": {"HostName": "other", "DNSName": "other.example.invalid.", "Online": False},
    }}
    calls = []

    def run(command, **options):
        calls.append((command, options))
        return SimpleNamespace(returncode=0, stdout=json.dumps(response))

    monkeypatch.setattr(backend.subprocess, "run", run)
    devices = backend.tailscale_devices()
    assert devices == [{"name": "camera", "host": "192.0.2.20", "online": True},
                       {"name": "other", "host": "other.example.invalid", "online": False}]
    assert calls[0][0][1:] == ["status", "--json"]
    assert calls[0][1]["timeout"] == 10


def test_tailscale_discovery_errors_do_not_echo_cli_private_output(tmp_path, monkeypatch):
    from types import SimpleNamespace

    executable = tmp_path / "tailscale.exe"
    executable.touch()
    monkeypatch.setattr(backend.shutil, "which", lambda name: str(executable))
    monkeypatch.setattr(backend.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=1, stdout="", stderr="private-fixture-device"))
    with pytest.raises(ValueError) as failure:
        backend.tailscale_devices()
    assert "private-fixture-device" not in str(failure.value)


def test_camera_does_not_report_rejected_update_as_success(monkeypatch):
    class Connection:
        closed = False

        def recv(self):
            return json.dumps({"type": "current_settings", "data": {"fps": 30}})

        def send(self, value):
            pass

        def close(self):
            self.closed = True

    connection = Connection()
    monkeypatch.setattr(backend, "create_connection", lambda *a, **kw: connection)
    with pytest.raises(ValueError, match="tous les réglages"):
        backend.camera_settings({"fps": 15})
    assert connection.closed


def test_capture_reopens_after_first_failed_read(monkeypatch):
    stop = threading.Event()
    state = core.AppState()
    captures = []

    class Capture:
        def __init__(self):
            self.index = len(captures)
            self.released = False
            captures.append(self)

        def isOpened(self):
            return True

        def read(self):
            if self.index == 0:
                return False, None
            stop.set()
            return True, np.zeros((24, 32, 3), np.uint8)

        def release(self):
            self.released = True

    monkeypatch.setattr(core, "_STOP", stop)
    monkeypatch.setattr(core, "_RECONNECT", threading.Event())
    monkeypatch.setattr(core, "state", state)
    monkeypatch.setattr(core, "open_pi_capture", Capture)
    core.read_stream()
    assert len(captures) == 2
    assert all(capture.released for capture in captures)
    assert state.frame_id == 1
    assert state.last_frame_time > 0


def test_explicit_encoder_recovery_restores_settings_even_if_interrupted(monkeypatch):
    initial = {"width": 640, "height": 480, "fps": 30, "bitrate": 2000000}
    calls = []

    def command(updates=None):
        calls.append(updates)
        return initial.copy()

    def interrupted(seconds):
        raise RuntimeError("interrupted wait")

    monkeypatch.setattr(backend, "camera_settings", command)
    monkeypatch.setattr(backend.time, "sleep", interrupted)
    with pytest.raises(RuntimeError, match="interrupted wait"):
        backend.restart_camera()
    assert calls == [None, {"fps": 15}, initial]


def test_window_waits_for_capture_to_release_before_exiting(window, qt, monkeypatch):
    gate = threading.Event()
    thread = threading.Thread(target=lambda: gate.wait(2), daemon=True)
    monkeypatch.setattr(core, "_background_threads", [thread])
    thread.start()
    window.close()
    qt.processEvents()
    assert window.isVisible()
    assert core._STOP.is_set()
    gate.set()
    thread.join(timeout=1)
    end = time.monotonic() + 2
    while window.isVisible() and time.monotonic() < end:
        qt.processEvents()
        time.sleep(.01)
    assert not window.isVisible()
