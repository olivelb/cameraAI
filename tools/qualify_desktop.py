"""Opt-in native GUI + actual camera/GPU qualification. No images are saved."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

import camera_core as core
from desktop import CameraWindow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    app = QApplication([])
    window = CameraWindow()
    window.show()
    started = last = time.monotonic()
    beats = []
    samples = []
    first_frame = None

    def heartbeat():
        nonlocal last
        now = time.monotonic()
        beats.append(now - last)
        last = now

    def sample():
        nonlocal first_frame
        now = time.monotonic()
        with core.state.lock:
            state = core.state
            item = {"seconds": round(now - started, 2), "frames": state.frame_id,
                    "processed_id": state.processed_id,
                    "frame_age": round(now - state.last_frame_time, 3) if state.last_frame_time else None,
                    "status": state.connection_status}
            if state.frame_id and first_frame is None:
                first_frame = now - started
        samples.append(item)
        if now - started >= args.seconds:
            fast.stop()
            monitor.stop()
            ordered = sorted(beats)
            result = {"seconds": args.seconds, "first_frame_seconds": first_frame,
                      "qt_heartbeat_count": len(beats),
                      "qt_heartbeat_p95_ms": round(ordered[int(.95 * (len(ordered) - 1))] * 1000, 2),
                      "qt_heartbeat_max_ms": round(max(beats) * 1000, 2),
                      "device": core.ai.device, "model": core.ai.segment_model_name,
                      "samples": samples}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            print(json.dumps({key: value for key, value in result.items() if key != "samples"}), flush=True)
            window.close()

    fast = QTimer()
    fast.setInterval(10)
    fast.timeout.connect(heartbeat)
    fast.start()
    monitor = QTimer()
    monitor.setInterval(1000)
    monitor.timeout.connect(sample)
    monitor.start()
    code = app.exec()
    return code or (0 if first_frame is not None and core.ai.segment_model is not None else 1)


if __name__ == "__main__":
    raise SystemExit(main())
