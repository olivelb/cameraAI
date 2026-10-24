"""Application operations shared by native UI automation and the JSON CLI."""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path

import cv2

import desktop_backend as backend
from control_api import validate_params


def checked(value):
    """Translate legacy backend failures into failed jobs, never false success."""
    if isinstance(value, str) and value.startswith((
        "❌", "VLM Error:", "VLM unavailable", "VLM requires", "Failed to", "No video signal.", "WS Error:",
    )):
        raise ValueError(value)
    return value


class ApplicationCommands:
    def __init__(self, core):
        self.core = core
        self.history = []
        self._view_lock = threading.Lock()
        self._view = {"paused": False, "overlay": True, "fullscreen": False}

    def publish_view(self, **values):
        with self._view_lock:
            self._view.update(values)

    def view(self):
        with self._view_lock:
            return dict(self._view)

    def redact(self, text):
        cfg = self.core.cfg
        values = [cfg.pi.host, cfg.pi.auth_token, cfg.server.auth_token,
                  cfg.ui.auth_password, str(self.core._ROOT)]
        values += [item["host"] for item in cfg.desktop.profiles.values()]
        values += [str(value) for key, value in self.core.email_config.items() if key != "enabled"]
        for value in sorted(set(values), key=len, reverse=True):
            if value:
                text = text.replace(value, "<private>")
        return re.sub(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]+", "<email>", text)

    def status(self):
        core, ai = self.core, self.core.ai
        with core.state.lock:
            state = core.state
            shape = list(state.last_frame.shape) if state.last_frame is not None else None
            age = round(time.monotonic() - state.last_frame_time, 3) if state.last_frame_time else None
            frame_id, processed = state.frame_id, state.processed_id
            info, track_ids = state.last_info, list(state.track_ids)
            classes, enabled = list(state.detected_classes), state.ai_enabled
        return {
            "camera": {"connected": age is not None and age < 3, "frame_age_seconds": age,
                       "shape": shape, "frame_id": frame_id, "processed_id": processed},
            "ai": {"enabled": enabled, "ready": ai.segment_model is not None,
                   "model": Path(ai.segment_model_name).name, "type": ai.model_type,
                   "device": ai.device, "confidence": core.state.conf, "classes": list(core.cfg.ai.detection_classes),
                   "info": self.redact(info)},
            "tracking": {"requested": ai.tracking_enabled,
                         "effective": ai.tracking_enabled and ai.model_type == "yolo" and enabled and ai.segment_model is not None,
                         "tracker": ai.tracker, "ids": track_ids, "detected_classes": classes},
            "recognition": {"faces": ai.face_recognition_enabled, "animals": ai.animal_recognition_enabled},
            "learning": {"faces_remaining": ai.learning_count, "animals_remaining": ai.animal_learning_count},
            "alerts": self.alerts_status(), "preview": self.view(),
            "network": {"active_profile": core.cfg.desktop.active_profile},
        }

    def alerts_status(self):
        email = self.core.email_config
        return {"enabled": bool(email["enabled"]),
                "configured": bool(email["sender"] and email["receiver"] and email["password"])}

    def execute(self, name, params):
        validate_params(name, params)
        core, ai, cfg = self.core, self.core.ai, self.core.cfg
        p = params
        if name == "camera.get":
            return backend.camera_settings()
        if name == "camera.set":
            return backend.camera_settings(p)
        if name == "camera.reconnect":
            core._RECONNECT.set()
            return {"reconnect_requested": True}
        if name == "camera.restart":
            return backend.restart_camera()
        if name == "network.get":
            result = {"active_profile": cfg.desktop.active_profile,
                      "profiles": list(cfg.desktop.profiles), "token_configured": bool(cfg.pi.auth_token)}
            if p.get("private"):
                result.update(host=cfg.pi.host, video_port=cfg.pi.video_port, ws_port=cfg.pi.ws_port,
                              saved_profiles=cfg.desktop.profiles)
            return result
        if name == "network.use":
            profile = p["profile"]
            if profile == "current":
                values = {"host": cfg.pi.host, "video_port": cfg.pi.video_port, "ws_port": cfg.pi.ws_port,
                          "open_timeout_ms": cfg.stream.capture_open_timeout_ms, "read_timeout_ms": cfg.stream.capture_read_timeout_ms}
            elif profile in cfg.desktop.profiles:
                values = dict(cfg.desktop.profiles[profile])
            else:
                raise ValueError("Profile not configured; use network.configure")
            return self.execute("network.configure", {**values, "profile": profile})
        if name == "network.configure":
            message = backend.save_connection(
                p["host"], p.get("video_port", cfg.pi.video_port), p.get("ws_port", cfg.pi.ws_port),
                p.get("token", cfg.pi.auth_token), p.get("open_timeout_ms", cfg.stream.capture_open_timeout_ms),
                p.get("read_timeout_ms", cfg.stream.capture_read_timeout_ms), profile_id=p.get("profile", "current"),
            )
            core._RECONNECT.set()
            return {"message": message, "active_profile": cfg.desktop.active_profile}
        if name == "network.tailscale":
            return backend.tailscale_devices()
        if name == "ai.set":
            with core.state.lock:
                core.state.ai_enabled = p["enabled"]
            return {"enabled": core.state.ai_enabled}
        if name == "detection.set":
            if "confidence" in p:
                core.update_conf(p["confidence"])
            if "classes" in p:
                cfg.ai.detection_classes = list(p["classes"])
            return {"confidence": core.state.conf, "classes": list(cfg.ai.detection_classes)}
        if name == "model.get":
            return {"model": Path(ai.segment_model_name).name, "type": ai.model_type, "device": ai.device,
                    "installed": sorted(path.name for pattern in ("*.pt", "*.engine") for path in core._ROOT.glob(pattern))}
        if name == "model.load":
            message = checked(core.change_model_logic(p["model"], p.get("task", "Detection"), p.get("concepts", "")))
            return {"message": message, **self.execute("model.get", {})}
        if name == "model.export":
            return {"message": checked(ai.export_to_tensorrt()), **self.execute("model.get", {})}
        if name == "sam.concepts":
            return {"message": checked(core.update_sam3_prompt_logic(p["concepts"])), "concepts": list(ai.sam3_prompts)}
        if name == "tracking.set":
            message = core.configure_tracking(p["enabled"], p.get("tracker", "BoT-SORT" if ai.tracker == "botsort.yaml" else "ByteTrack"))
            return {"message": message, **self.status()["tracking"]}
        if name == "recognition.set":
            animal = p["kind"] == "animals"
            recognizer = ai._animal_recognizer if animal else ai._face_recognizer
            if "threshold" in p:
                setattr(cfg.ai, "animal_threshold" if animal else "face_threshold", p["threshold"])
                if recognizer is not None:
                    recognizer.threshold = p["threshold"]
            message = checked((core.toggle_animal_recognition if animal else core.toggle_face_recognition)(p["enabled"]))
            return {"message": message, "enabled": ai.animal_recognition_enabled if animal else ai.face_recognition_enabled}
        if name == "gallery.status":
            recognizer = ai._animal_recognizer if p["kind"] == "animals" else ai._face_recognizer
            directory = ai.gallery_root / p["kind"]
            return {"loaded": recognizer is not None,
                    "ready": bool(recognizer.is_ready()) if recognizer is not None else False,
                    "entries": sorted(path.name for path in directory.iterdir() if path.is_dir()) if directory.exists() else [],
                    "status": recognizer.gallery_status() if recognizer is not None else "Not loaded"}
        if name == "gallery.import":
            action = core.add_animal_to_gallery if p["kind"] == "animals" else core.add_to_gallery
            return {"message": checked(action(p["name"], p["files"]))}
        if name == "gallery.rebuild":
            action = core.rebuild_animal_gallery if p["kind"] == "animals" else core.rebuild_gallery
            return {"message": checked(action())}
        if name == "gallery.learn":
            action = ai.start_learning_animal if p["kind"] == "animals" else ai.start_learning
            return {"message": checked(action(p["name"], count=p.get("count", 10)))}
        if name == "gallery.stop":
            with ai._infer_lock:
                if p["kind"] == "animals":
                    ai.animal_learning_count = 0
                else:
                    ai.learning_count = 0
            return {"stopped": True}
        if name == "alerts.get":
            return self.alerts_status()
        if name == "alerts.configure":
            message = core.save_email_settings(p.get("enabled", False), p["sender"], p.get("password", ""), p["receiver"])
            return {"message": checked(message), **self.alerts_status()}
        if name == "alerts.enable":
            if p["enabled"] and not self.alerts_status()["configured"]:
                raise ValueError("Configure email credentials before enabling alerts")
            core.toggle_email_state(p["enabled"])
            return self.alerts_status()
        if name == "alerts.test":
            if not self.alerts_status()["configured"]:
                raise ValueError("Configure email credentials first")
            if not p.get("send", False):
                return {"validated": True, "sent": False}
            return {"message": checked(core.send_test_email())}
        if name == "assistant.ask":
            answer = checked(core.ask_vlm(p["message"], list(self.history)))
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Assistant returned no final answer")
            self.history.extend([{"role": "user", "content": p["message"]}, {"role": "assistant", "content": answer}])
            del self.history[:-20]
            return {"answer": answer, "messages": len(self.history)}
        if name == "assistant.history":
            return list(self.history)
        if name == "assistant.clear":
            self.history.clear()
            return {"cleared": True}
        if name == "frame.save":
            with core.state.lock:
                annotated = (p.get("annotated", True) and core.state.ai_enabled
                             and core.state.last_processed is not None
                             and time.monotonic() - core.state.last_processed_time < 1)
                frame = core.state.last_processed if annotated else core.state.last_frame
                if frame is None:
                    raise ValueError("No frame available")
                frame = frame.copy()
            path = Path(p["path"]).expanduser().resolve()
            if path.suffix.lower() != ".png":
                raise ValueError("Frame output must be a PNG file")
            ok, pixels = cv2.imencode(".png", frame)
            if not ok:
                raise ValueError("PNG encoding failed")
            with path.open("wb" if p.get("overwrite") else "xb") as stream:
                stream.write(pixels.tobytes())
            return {"path": str(path), "shape": list(frame.shape)}
        if name == "logs.get":
            text = "\n".join(core.get_logs().splitlines()[:p.get("limit", 20)])
            return {"text": text if p.get("private") else self.redact(text)}
        raise ValueError("Command is handled by the desktop UI or transport")
