#!/usr/bin/env python3
"""
server.py — Raspberry Pi Camera Streaming Server.

Architecture:
  - rpicam-vid with --listen opens a TCP server directly on the video port
    (no socat, no pipe — eliminates a process and buffering overhead)
  - asyncio WebSocket server handles live settings changes
  - PID file prevents duplicate instances
  - Graceful SIGTERM/SIGINT shutdown

Camera: OV5647 (Pi Camera v1) — 640x480 @ 30fps native
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import threading as _threading

import websockets
from config import cfg

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
_VIDEO_PORT = cfg.server.video_port
_WS_PORT = cfg.server.ws_port
_PID_FILE = cfg.server.pid_file
_AUTH_TOKEN = cfg.server.auth_token
_DEFAULT_SETTINGS = {
    "width": cfg.server.default_width,
    "height": cfg.server.default_height,
    "fps": cfg.server.default_fps,
    "bitrate": cfg.server.default_bitrate,
}
_VALID_RESOLUTIONS = cfg.camera.valid_resolution_set()


# ── PID file — prevents duplicate instances ──────────────────────────────────

_pid_lock_fd: int | None = None


def acquire_pid_lock() -> None:
    """Hold a Linux advisory lock for this process's complete lifetime."""
    import fcntl

    global _pid_lock_fd
    if _pid_lock_fd is not None:
        raise RuntimeError("PID lock already acquired")
    fd = os.open(_PID_FILE, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
    except BaseException:
        os.close(fd)
        raise
    _pid_lock_fd = fd


def release_pid_lock() -> None:
    """Release only our lock. Keep the inode to avoid unlink/reacquire races."""
    global _pid_lock_fd
    if _pid_lock_fd is not None:
        os.close(_pid_lock_fd)
        _pid_lock_fd = None


# ── Stream settings ──────────────────────────────────────────────────────────
settings: dict[str, int] = dict(_DEFAULT_SETTINGS)

# Watchdog control
_watchdog_stop = _threading.Event()
_watchdog_restart = _threading.Event()
_stream_lock = _threading.Lock()
_stream_process: subprocess.Popen | None = None


def build_rpicam_cmd() -> list[str]:
    """Build the rpicam-vid command optimized for the Pi.

    Key optimizations:
      --nopreview     Saves ~5% ARM CPU
      --denoise off   Saves ~8% ARM CPU
      --profile baseline  Simplest H.264 profile — fastest VPU path
      --inline        SPS/PPS in every keyframe — allows mid-stream joins
      --flush         No OS buffering latency
      --intra 2*fps   Keyframe every 2s — fewer expensive I-frames
      nice -n -10     Raises scheduling priority

    Returns an argument list (executed without a shell).
    """
    w = settings["width"]
    h = settings["height"]
    fps = settings["fps"]
    br = settings["bitrate"]
    intra = fps * 2

    return [
        "nice", "-n", "-10", "rpicam-vid",
        "--nopreview",
        "--width", str(w),
        "--height", str(h),
        "--framerate", str(fps),
        "--bitrate", str(br),
        "--profile", "baseline",
        "--level", "4",
        "--codec", "h264",
        "--inline",
        "--flush",
        "--denoise", "off",
        "--intra", str(intra),
        "-t", "0",
        "-o", f"tcp://0.0.0.0:{_VIDEO_PORT}",
        "--listen",
    ]


def _kill_process(proc: subprocess.Popen) -> None:
    """Terminate our child session, escalating if it ignores SIGTERM."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=3)
    except ProcessLookupError:
        proc.wait(timeout=3)


def _watchdog() -> None:
    """Watchdog thread — keeps rpicam-vid alive.

    rpicam-vid --listen exits after each client disconnects. This loop
    detects the exit and restarts it immediately so the port is always
    available for the next connection.
    """
    global _stream_process
    log.info("Stream watchdog started")

    while not _watchdog_stop.is_set():
        if _watchdog_restart.is_set():
            _watchdog_stop.wait(timeout=0.3)
            _watchdog_restart.clear()
            with _stream_lock:
                if _stream_process is not None:
                    log.info("Restarting stream with new settings...")
                    _kill_process(_stream_process)
                    _stream_process = None

        with _stream_lock:
            if _stream_process is None or _stream_process.poll() is not None:
                cmd = build_rpicam_cmd()
                log.info(
                    "Starting stream: %dx%d @ %dfps  %dkbps",
                    settings["width"],
                    settings["height"],
                    settings["fps"],
                    settings["bitrate"] // 1000,
                )
                try:
                    _stream_process = subprocess.Popen(cmd, start_new_session=True)
                except OSError:
                    log.exception("Could not start rpicam-vid; retrying")
        if _stream_process is None:
            _watchdog_stop.wait(timeout=2)

        _watchdog_stop.wait(timeout=0.5)

    with _stream_lock:
        if _stream_process is not None:
            _kill_process(_stream_process)
            _stream_process = None
    log.info("Stream watchdog stopped")


def start_stream() -> None:
    """Signal the watchdog to restart rpicam-vid with current settings."""
    _watchdog_restart.set()


def stop_stream() -> None:
    """Stop the watchdog and terminate rpicam-vid."""
    _watchdog_stop.set()


def start_watchdog() -> _threading.Thread:
    """Launch the watchdog thread (call once at startup)."""
    _watchdog_stop.clear()
    _watchdog_restart.clear()
    t = _threading.Thread(target=_watchdog, daemon=True, name="stream-watchdog")
    t.start()
    return t


# ── WebSocket control server ────────────────────────────────────────────────
VALID_KEYS = {"width", "height", "fps", "bitrate"}


async def handle_client(websocket) -> None:
    """Handle a WebSocket control client."""
    client = websocket.remote_address
    log.info("Client connected: %s", client)

    try:
        await websocket.send(
            json.dumps({"type": "current_settings", "data": settings})
        )
    except Exception as e:
        log.warning("Could not send initial settings to %s: %s", client, e)

    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError as e:
                log.warning("Malformed JSON from %s: %s", client, e)
                continue

            if not isinstance(msg, dict):
                continue

            if msg.get("type") != "update_settings":
                continue

            if _AUTH_TOKEN and msg.get("token") != _AUTH_TOKEN:
                log.warning(
                    "update_settings from %s rejected: missing or invalid token",
                    client,
                )
                continue

            updates = msg.get("data", {})
            if not isinstance(updates, dict):
                continue
            changed = False

            pending: dict[str, int] = {}
            for key, value in updates.items():
                if key not in VALID_KEYS:
                    log.warning("Unknown setting '%s' — ignored", key)
                    continue
                if type(value) is not int:
                    log.warning(
                        "Invalid type for '%s': expected int, got %s",
                        key,
                        type(value).__name__,
                    )
                    continue
                if key == "fps" and not 1 <= value <= 30:
                    log.warning("FPS outside [1, 30] — ignored")
                    continue
                if key == "bitrate" and not 100_000 <= value <= 25_000_000:
                    log.warning("Bitrate outside [100000, 25000000] — ignored")
                    continue
                pending[key] = value

            # Validate the resulting resolution pair, even when only one
            # dimension is updated (merge pending over current settings)
            if "width" in pending or "height" in pending:
                pair = (
                    pending.get("width", settings["width"]),
                    pending.get("height", settings["height"]),
                )
                if pair not in _VALID_RESOLUTIONS:
                    log.warning(
                        "Invalid resolution %dx%d — not a native mode. Ignored.",
                        pair[0],
                        pair[1],
                    )
                    pending.pop("width", None)
                    pending.pop("height", None)

            with _stream_lock:
                for key, value in pending.items():
                    if settings[key] != value:
                        log.info("  %s: %s → %s", key, settings[key], value)
                        settings[key] = value
                        changed = True

            if changed:
                start_stream()

            # Ack with the authoritative settings so the client can resync
            try:
                await websocket.send(
                    json.dumps({"type": "current_settings", "data": settings})
                )
            except Exception as e:
                log.warning("Could not send settings ack to %s: %s", client, e)

    except websockets.exceptions.ConnectionClosed:
        log.info("Client disconnected: %s", client)


# ── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    """Start the camera server."""
    acquire_pid_lock()
    watchdog = None
    shutdown = asyncio.Event()

    def _shutdown(signum, frame):
        log.info("Received signal %s. Shutting down...", signum)
        shutdown.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    log.info("WebSocket control server listening on 0.0.0.0:%d", _WS_PORT)
    log.info("Video stream (H.264/TCP) available on port %d", _VIDEO_PORT)

    # Bind all interfaces: the Pi must be reachable from LAN/Tailscale clients
    try:
        async with websockets.serve(
            handle_client, "0.0.0.0", _WS_PORT, max_size=4096  # noqa: S104
        ):
            watchdog = start_watchdog()
            await shutdown.wait()
    finally:
        stop_stream()
        if watchdog is not None:
            await asyncio.to_thread(watchdog.join, 10)
        release_pid_lock()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        log.exception("Fatal error: %s", e)
        sys.exit(1)
