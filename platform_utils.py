"""
platform_utils.py — Platform-specific performance optimizations.

Extracted from app.py to keep the main module clean.
Currently handles Windows 11 anti-throttling boost only.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time

log = logging.getLogger(__name__)


def apply_windows_boost() -> None:
    """Apply Windows 11 anti-EcoQoS / anti-throttling optimizations.

    - Forces 1ms timer resolution
    - Sets REALTIME priority class
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

        # 2. REALTIME priority
        p = psutil.Process(os.getpid())
        p.nice(psutil.REALTIME_PRIORITY_CLASS)

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
    """Kill stale listeners holding a port (Windows only).

    Used to reclaim port 7860 on startup when a previous instance
    didn't shut down cleanly. Only processes actually LISTENING on the
    exact port are targeted — never the current process, never clients
    with established connections.
    """
    if sys.platform != "win32":
        return

    log.info("Checking for stale listeners on port %d...", port)
    try:
        # Kill stray Gradio tunnel binaries
        subprocess.run(
            ["taskkill", "/IM", "frpc_windows_amd64_v0.3.exe", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Find processes listening on the port and kill them
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout

        pids = _listening_pids(result, port, os.getpid())
        if not pids:
            return

        for pid in pids:
            log.warning("Killing stale listener (PID %s) on port %d", pid, port)
            subprocess.run(
                ["taskkill", "/F", "/PID", pid],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        time.sleep(1)  # Let OS release the port
    except FileNotFoundError:
        pass  # netstat/taskkill unavailable — nothing to clean
    except Exception as e:
        log.debug("Port cleanup error: %s", e)
