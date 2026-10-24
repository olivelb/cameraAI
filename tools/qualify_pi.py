"""Opt-in hardware checks: python -m tools.qualify_pi --help.

Reads the configured camera by default. --profiles changes settings temporarily
and restores the initial values in finally. Images stay in the output directory.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from capture import open_pi_capture

import cv2
from websocket import create_connection

from config import cfg


def control(updates: dict | None = None, *, token: str | None = None) -> dict:
    connection = create_connection(cfg.pi.ws_url, timeout=5)
    try:
        initial = json.loads(connection.recv())
        if updates is None:
            return initial["data"]
        connection.send(json.dumps({
            "type": "update_settings", "data": updates,
            "token": cfg.pi.auth_token if token is None else token,
        }))
        reply = json.loads(connection.recv())
        if reply.get("type") == "error":
            return reply
        return reply["data"]
    finally:
        connection.close()


def capture(seconds: float, image_path: Path) -> dict:
    start = time.monotonic()
    video = open_pi_capture()
    result = {"open_seconds": time.monotonic() - start, "opened": video.isOpened()}
    count = 0
    first = last = None
    frame = None
    shapes: set[tuple[int, ...]] = set()
    deadline = time.monotonic() + seconds
    try:
        while video.isOpened() and time.monotonic() < deadline:
            ok, current = video.read()
            if not ok:
                result["read_failed"] = True
                break
            now = time.monotonic()
            first = now if first is None else first
            last = now
            count += 1
            shapes.add(current.shape)
            frame = current
    finally:
        video.release()
    result.update({
        "frames": count,
        "first_frame_seconds": first - start if first is not None else None,
        "received_fps": (count - 1) / (last - first) if count > 1 and last > first else None,
        "shapes": sorted(shapes),
    })
    if frame is not None:
        result["image_saved"] = bool(cv2.imwrite(str(image_path), frame))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=3)
    parser.add_argument("--profiles", nargs="*", help="Temporarily test WIDTHxHEIGHT@FPS")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    initial = control()
    result = {"initial_settings": initial, "profiles": [], "restored": None}
    try:
        for index, profile in enumerate(args.profiles or ["current"]):
            if profile != "current":
                resolution, fps = profile.split("@")
                width, height = map(int, resolution.split("x"))
                expected = {"width": width, "height": height, "fps": int(fps)}
                acknowledged = control(expected)
                if any(acknowledged.get(k) != v for k, v in expected.items()):
                    raise RuntimeError(f"Profile rejected: {profile}")
                time.sleep(2)
            else:
                acknowledged = initial
            entry = {"requested": profile, "acknowledged": acknowledged}
            entry.update(capture(args.seconds, args.output / f"frame-{index}.jpg"))
            expected_shape = [acknowledged["height"], acknowledged["width"], 3]
            entry["shape_matches_settings"] = bool(entry["shapes"]) and all(
                list(shape) == expected_shape for shape in entry["shapes"]
            )
            result["profiles"].append(entry)
            print(json.dumps(entry), flush=True)
            (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            time.sleep(3)
    finally:
        if args.profiles:
            result["restored"] = control(initial)
            if result["restored"] != initial:
                raise RuntimeError("Could not restore the initial camera settings")
        (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if all(p.get("frames", 0) > 0 and p["shape_matches_settings"] for p in result["profiles"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
