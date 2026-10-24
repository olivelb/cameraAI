"""Exercise the actual WebSocket handler over a local socket."""

import asyncio
import json
import os
import selectors
import signal
import subprocess
import sys
import threading

import pytest
import websockets

import server


def test_control_rejects_bad_messages_and_acknowledges_updates(monkeypatch):
    monkeypatch.setattr(server, "settings", dict(server._DEFAULT_SETTINGS))
    monkeypatch.setattr(server, "_AUTH_TOKEN", "test-token")
    restarts = []
    monkeypatch.setattr(server, "start_stream", lambda: restarts.append(True))

    async def exercise():
        async with websockets.serve(server.handle_client, "127.0.0.1", 0) as listener:
            port = listener.sockets[0].getsockname()[1]
            async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                initial = json.loads(await ws.recv())["data"]
                for raw in ("invalid", b"\xff", "[]", "null", '{"type":"update_settings","data":[]}'):
                    await ws.send(raw)
                # The malformed data request above is rejected before inspecting data.
                assert json.loads(await ws.recv())["code"] == "unauthorized"
                # A rejected token must never apply this otherwise valid update.
                await ws.send(json.dumps({"type": "update_settings", "data": {"fps": 7}}))
                assert json.loads(await ws.recv())["code"] == "unauthorized"
                for changes in ({"fps": -5}, {"fps": True}, {"bitrate": 0},
                                {"width": 1920}, {"height": "1080"}):
                    await ws.send(json.dumps({"type": "update_settings", "token": "test-token", "data": changes}))
                    ack = json.loads(await asyncio.wait_for(ws.recv(), 2))
                    assert ack["data"] == initial
                assert not restarts
                await ws.send(json.dumps({"type": "update_settings", "token": "test-token",
                                          "data": {"width": 1920, "height": 1080, "fps": 15}}))
                ack = json.loads(await asyncio.wait_for(ws.recv(), 2))
                assert ack["data"]["width"] == 1920
                assert ack["data"]["height"] == 1080
                assert ack["data"]["fps"] == 15
                assert len(restarts) == 1
    asyncio.run(exercise())


def test_main_releases_resources_when_binding_fails(monkeypatch):
    events = []
    monkeypatch.setattr(server, "acquire_pid_lock", lambda: events.append("acquire"))
    monkeypatch.setattr(server, "release_pid_lock", lambda: events.append("release"))
    monkeypatch.setattr(server, "stop_stream", lambda: events.append("stop"))
    monkeypatch.setattr(server.signal, "signal", lambda *args: None)

    def fail(*args, **kwargs):
        raise OSError("port occupied")

    monkeypatch.setattr(server.websockets, "serve", fail)
    with pytest.raises(OSError):
        asyncio.run(server.main())
    assert events == ["acquire", "stop", "release"]


def test_h264_modes_exclude_full_sensor_photo_resolution():
    assert (1920, 1080) in server._VALID_RESOLUTIONS
    assert (2592, 1944) not in server._VALID_RESOLUTIONS


def test_rpicam_command_does_not_require_privileged_niceness():
    cmd = server.build_rpicam_cmd()
    assert cmd[0] == "rpicam-vid"
    assert "nice" not in cmd


def test_control_stays_responsive_during_slow_child_cleanup(monkeypatch):
    monkeypatch.setattr(server, "settings", dict(server._DEFAULT_SETTINGS))
    monkeypatch.setattr(server, "_AUTH_TOKEN", "")
    server._stream_lock.acquire()
    # A timeout release prevents the old blocking implementation deadlocking pytest.
    release = threading.Timer(1, server._stream_lock.release)
    release.start()

    async def exercise():
        async with websockets.serve(server.handle_client, "127.0.0.1", 0) as listener:
            port = listener.sockets[0].getsockname()[1]
            async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                await ws.recv()
                await ws.send(json.dumps({"type": "update_settings", "data": {"fps": 15}}))
                assert json.loads(await ws.recv())["data"]["fps"] == 15
                assert server._stream_lock.locked(), "Control waited for process cleanup"
    try:
        asyncio.run(exercise())
    finally:
        release.cancel()
        release.join()
        if server._stream_lock.locked():
            server._stream_lock.release()


@pytest.mark.skipif(sys.platform == "win32", reason="Pi advisory lock uses Linux fcntl")
def test_pid_lock_excludes_second_process_and_survives_stale_pid(tmp_path, monkeypatch):
    lock = tmp_path / "camera.pid"
    lock.write_text("99999999")
    monkeypatch.setattr(server, "_PID_FILE", str(lock))
    server.acquire_pid_lock()
    try:
        assert lock.read_text() == str(os.getpid())
        code = (
            "import server; server._PID_FILE = " + repr(str(lock))
            + "; server.acquire_pid_lock()"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
        assert result.returncode != 0
        assert "BlockingIOError" in result.stderr
    finally:
        server.release_pid_lock()
    server.acquire_pid_lock()
    server.release_pid_lock()
    assert lock.exists()  # Keep the inode, not a race-prone unlink.


@pytest.mark.skipif(sys.platform == "win32", reason="Native Linux SIGTERM/selector regression")
def test_sigterm_wakes_an_idle_server(tmp_path):
    code = (
        "import asyncio, server; server._WS_PORT=0; server._PID_FILE="
        + repr(str(tmp_path / "idle-server.pid"))
        + "; server.start_watchdog=lambda: print('READY',flush=True); asyncio.run(server.main())"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(10), "Test server did not become ready"
        assert process.stdout.readline().strip() == "READY"
        process.send_signal(signal.SIGTERM)
        _, errors = process.communicate(timeout=3)
        assert process.returncode == 0, errors
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()
