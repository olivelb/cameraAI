"""
platform_utils.py — Platform-specific performance optimizations.

Extracted from app.py to keep the main module clean.
Currently handles Windows 11 anti-throttling boost only.
"""

from __future__ import annotations

import logging
import os
import socket
import sys

log = logging.getLogger(__name__)


def apply_windows_boost() -> None:
    """Apply Windows 11 anti-EcoQoS / anti-throttling optimizations.

    - Forces 1ms timer resolution
    - Sets ABOVE_NORMAL priority class
    - Prevents thread-to-E-core shunting
    - Prevents display/system sleep
    """
    if sys.platform != "win32":
        return

    try:
        import ctypes
        from ctypes import wintypes

        import psutil

        # 1. Force extreme timer resolution
        ctypes.windll.winmm.timeBeginPeriod(1)

        # 2. Avoid REALTIME: inference must not starve input or system services.
        p = psutil.Process(os.getpid())
        p.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)

        # 3. Prevent display/system sleep
        ctypes.windll.kernel32.SetThreadExecutionState(
            0x80000000 | 0x00000001  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )

        # 4. Disable EcoQoS power throttling
        class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
            _fields_ = [
                ("Version", wintypes.ULONG),
                ("ControlMask", wintypes.ULONG),
                ("StateMask", wintypes.ULONG),
            ]

        state = PROCESS_POWER_THROTTLING_STATE()
        state.Version = 1
        state.ControlMask = 1  # PROCESS_POWER_THROTTLING_EXECUTION_SPEED
        state.StateMask = 0    # 0 = Disable Throttling

        ctypes.windll.kernel32.SetProcessInformation(
            ctypes.windll.kernel32.GetCurrentProcess(),
            77,  # ProcessPowerThrottling
            ctypes.byref(state),
            ctypes.sizeof(state),
        )
        log.info("⚡ Windows 11 anti-throttling boost activated")
    except Exception as e:
        log.warning("Could not activate Windows boost: %s", e)


def _listening_pids(netstat_output: str, port: int, own_pid: int) -> list[str]:
    """Parse `netstat -ano` output; return PIDs LISTENING on `port`.

    Only TCP lines in LISTENING state whose *local* address ends with the
    exact port are considered, so ESTABLISHED connections to/from the port
    (e.g. a browser) are never matched. The caller's own PID is excluded.
    """
    pids: list[str] = []
    for line in netstat_output.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0].upper() != "TCP":
            continue
        local_addr, state, pid = parts[1], parts[3], parts[4]
        if state.upper() != "LISTENING":
            continue
        if local_addr.rsplit(":", 1)[-1] != str(port):
            continue
        if pid == str(own_pid):
            continue
        pids.append(pid)
    return pids


def cleanup_port(port: int) -> None:
    """Refuse an occupied port without terminating unrelated processes."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if sys.platform == "win32":
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            probe.bind(("127.0.0.1", port))
    except OSError as e:
        raise RuntimeError(
            f"Port {port} is unavailable. Stop its owner or change ui.server_port."
        ) from e
