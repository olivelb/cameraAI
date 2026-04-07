"""
platform_utils.py — Platform-specific performance optimizations.

Extracted from app.py to keep the main module clean.
Currently handles Windows 11 anti-throttling boost only.
"""

from __future__ import annotations

import logging
import os
import re
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


def cleanup_port(port: int) -> None:
    """Kill stale processes holding a port (Windows only).

    Used to reclaim port 7860 on startup when a previous instance
    didn't shut down cleanly.
    """
    if sys.platform != "win32":
        return

    log.info("Checking for stale processes on port %d...", port)
    try:
        # Kill stray Gradio tunnel binaries
        subprocess.run(
            "taskkill /IM frpc_windows_amd64_v0.3.exe /F",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Find and kill processes blocking the port
        result = subprocess.check_output(
            f"netstat -ano | findstr :{port}", shell=True
        ).decode()

        for line in result.strip().split("\n"):
            parts = re.split(r"\s+", line.strip())
            if len(parts) > 4:
                pid = parts[-1]
                if pid != "0":
                    log.warning("Killing stale process (PID %s) on port %d", pid, port)
                    subprocess.run(
                        f"taskkill /F /PID {pid}",
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )

        time.sleep(1)  # Let OS release the port
    except subprocess.CalledProcessError:
        pass  # Port is likely free
    except Exception as e:
        log.debug("Port cleanup error: %s", e)
