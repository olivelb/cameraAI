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

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
# Import config only if available (server may run on Pi without full deps)
try:
    from config import cfg

    _VIDEO_PORT = cfg.server.video_port
    _WS_PORT = cfg.server.ws_port
    _PID_FILE = cfg.server.pid_file
    _DEFAULT_SETTINGS = {
        "width": cfg.server.default_width,
        "height": cfg.server.default_height,
        "fps": cfg.server.default_fps,
        "bitrate": cfg.server.default_bitrate,
    }
    _VALID_RESOLUTIONS = cfg.camera.valid_resolution_set()
except ImportError:
    log.warning("config.py not found — using built-in defaults")
    _VIDEO_PORT = 8888
    _WS_PORT = 8889
    _PID_FILE = "/tmp/camera_server.pid"
    _DEFAULT_SETTINGS = {
        "width": 640,
        "height": 480,
        "fps": 30,
        "bitrate": 2_000_000,
    }
    _VALID_RESOLUTIONS = {
        (640, 480),
        (1296, 972),
        (1920, 1080),
        (2592, 1944),
    }


# ── PID file — prevents duplicate instances ──────────────────────────────────

def acquire_pid_lock() -> None:
    """Write our PID to the lock file. Exit if another instance is running."""
    if os.path.exists(_PID_FILE):
        try:
            with open(_PID_FILE) as f:
                old_pid = int(f.read().strip())
            os.kill(old_pid, 0)
            log.error(
                "Another server instance is already running (PID %d). Exiting.",
                old_pid,
            )
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            log.warning("Stale PID file found. Cleaning up.")

    with open(_PID_FILE, "w") as f:
        f.write(str(os.getpid()))


def release_pid_lock() -> None:
    """Remove the PID lock file."""
    try:
        os.remove(_PID_FILE)
    except FileNotFoundError:
        pass


# ── Stream settings ──────────────────────────────────────────────────────────
settings: dict[str, int] = dict(_DEFAULT_SETTINGS)

# Watchdog control
_watchdog_stop = _threading.Event()
_watchdog_restart = _threading.Event()
_stream_lock = _threading.Lock()
_stream_process: subprocess.Popen | None = None


def build_rpicam_cmd() -> str:
    """Build the rpicam-vid command optimized for the Pi.

    Key optimizations:
      --nopreview     Saves ~5% ARM CPU
      --denoise off   Saves ~8% ARM CPU
      --profile baseline  Simplest H.264 profile — fastest VPU path
      --inline        SPS/PPS in every keyframe — allows mid-stream joins
      --flush         No OS buffering latency
      --intra 2*fps   Keyframe every 2s — fewer expensive I-frames
      nice -n -10     Raises scheduling priority
    """
    w = settings["width"]
    h = settings["height"]
    fps = settings["fps"]
    br = settings["bitrate"]
    intra = fps * 2

    return (
        f"nice -n -10 rpicam-vid"
        f" --nopreview"
        f" --width {w}"
        f" --height {h}"
        f" --framerate {fps}"
        f" --bitrate {br}"
        f" --profile baseline"
        f" --level 4"
        f" --codec h264"
        f" --inline"
        f" --flush"
        f" --denoise off"
        f" --intra {intra}"
        f" -t 0"
        f" -o tcp://0.0.0.0:{_VIDEO_PORT}"
        f" --listen"
    )


def _kill_process(proc: subprocess.Popen) -> None:
    """Kill a process group, ignoring errors."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=3)
    except Exception as e:
        log.debug("Kill process: %s", e)


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
                _stream_process = subprocess.Popen(
                    cmd,
                    shell=True,
                    preexec_fn=os.setsid,
                    stderr=subprocess.DEVNULL,
                )

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

            if msg.get("type") != "update_settings":
                continue

            updates = msg.get("data", {})
            changed = False

            pending: dict[str, int] = {}
            for key, value in updates.items():
                if key not in VALID_KEYS:
                    log.warning("Unknown setting '%s' — ignored", key)
                    continue
                if not isinstance(value, int):
                    log.warning(
                        "Invalid type for '%s': expected int, got %s",
                        key,
                        type(value).__name__,
                    )
                    continue
                pending[key] = value

            # Validate resolution pair
            if "width" in pending and "height" in pending:
                pair = (pending["width"], pending["height"])
                if pair not in _VALID_RESOLUTIONS:
                    log.warning(
                        "Invalid resolution %dx%d — not a native mode. Ignored.",
                        pair[0],
                        pair[1],
                    )
                    pending.pop("width")
                    pending.pop("height")

            for key, value in pending.items():
                if settings[key] != value:
                    log.info("  %s: %s → %s", key, settings[key], value)
                    settings[key] = value
                    changed = True

            if changed:
                start_stream()

    except websockets.exceptions.ConnectionClosed:
        log.info("Client disconnected: %s", client)


# ── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    """Start the camera server."""
    acquire_pid_lock()
    start_watchdog()

    def _shutdown(signum, frame):
        log.info("Received signal %s. Shutting down...", signum)
        stop_stream()
        release_pid_lock()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    log.info("WebSocket control server listening on 0.0.0.0:%d", _WS_PORT)
    log.info("Video stream (H.264/TCP) available on port %d", _VIDEO_PORT)

    async with websockets.serve(handle_client, "0.0.0.0", _WS_PORT):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except SystemExit:
        pass
    except Exception as e:
        log.exception("Fatal error: %s", e)
        stop_stream()
        release_pid_lock()
        sys.exit(1)
