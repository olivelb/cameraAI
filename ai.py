"""
ai.py — Core AI processing engine.

Manages model loading (YOLO, SAM3), inference (detection, segmentation,
tracking), face/animal recognition overlays, and VLM scene description
(Gemma4-E2B).

Heavy libraries (torch, ultralytics, transformers) are loaded lazily in
a background thread so the Gradio UI starts instantly.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from config import cfg

log = logging.getLogger(__name__)

# ── Lazy import placeholders ─────────────────────────────────────────────────
torch = None
YOLO = None
SAM = None


class AIProcessor:
    """Central AI inference engine for detection, tracking, and recognition."""

    def __init__(self) -> None:
        log.info("AIProcessor initializing...")
        self.device: str = "cpu"

        self.loaded_models: dict = {}
        self.segment_model = None
        self.segment_model_name: str = cfg.ai.default_model
        self.model_type: str = "yolo"

        # Tracking state
        self.tracking_enabled: bool = False
        self.tracker: str = "botsort.yaml"

        # Face recognition
        self.face_recognition_enabled: bool = False
        self._face_recognizer = None
        self._fr_lock = threading.Lock()
        self.learning_name: str = ""
        self.learning_count: int = 0
        self.learning_max: int = 5
        self.face_history: dict[int, deque] = {}

        # Animal recognition
        self.animal_recognition_enabled: bool = False
        self._animal_recognizer = None
        self._ar_lock = threading.Lock()
        self.animal_learning_name: str = ""
        self.animal_learning_count: int = 0
        self.animal_learning_max: int = 5
        self.animal_history: dict[int, deque] = {}

        # VLM (Gemma4-E2B)
        self._vlm_pipeline = None
        self._vlm_lock = threading.Lock()

        # Start background loader
        threading.Thread(
            target=self._async_loader, daemon=True, name="ai-loader"
        ).start()

    # ── Background loader ────────────────────────────────────────────────────

    def _async_loader(self) -> None:
        """Import heavy libraries and load default model in background."""
        log.info("Background loader: importing PyTorch & Ultralytics...")
        global torch, YOLO, SAM

        import torch as _torch
        from ultralytics import YOLO as _YOLO, SAM as _SAM

        torch = _torch
        YOLO = _YOLO
        SAM = _SAM

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("AI backend ready. Device: %s", self.device)

        if self.device == "cpu":
            log.warning("CUDA not available — AI processing will be slow")

        self.load_segmentation_model(self.segment_model_name)

    # ── Model management ─────────────────────────────────────────────────────

    def load_segmentation_model(
        self, model_name: str, task_type: str = "Detection"
    ) -> str:
        """Load a YOLO or SAM model by name and task type."""
        if YOLO is None:
            log.info("Waiting for libraries to load...")
            while YOLO is None:
                time.sleep(0.5)

        log.info("Loading model: %s (%s)...", model_name, task_type)
        try:
            base_name = (
                model_name.replace(".pt", "")
                .replace(".engine", "")
                .replace("-seg", "")
            )
            ext = ".engine" if model_name.endswith(".engine") else ".pt"

            if task_type == "Segmentation":
                target_name = f"{base_name}-seg{ext}"
            else:
                target_name = f"{base_name}{ext}"

            # Prefer TensorRT engine if available
            engine_name = target_name.replace(".pt", ".engine")
            if target_name.endswith(".pt") and os.path.exists(engine_name):
                log.info("Found TensorRT engine: %s — using it instead", engine_name)
                target_name = engine_name

            self.segment_model_name = target_name

            # Check cache
            if target_name in self.loaded_models:
                log.info("Using cached model: %s", target_name)
                self.segment_model = self.loaded_models[target_name]
                self.model_type = "sam" if "sam" in target_name.lower() else "yolo"
                return f"Loaded Cached {target_name}"

            if "sam" in target_name.lower():
                self.segment_model = SAM(target_name)
                self.model_type = "sam"
            else:
                self.segment_model = YOLO(target_name)
                self.model_type = "yolo"

            self.loaded_models[target_name] = self.segment_model
            log.info("Model %s loaded successfully", target_name)
            return f"Loaded {target_name}"

        except Exception as e:
            log.error("Model load error: %s", e)
            return str(e)

    def set_tracker(self, tracker_name: str) -> None:
        """Set the active tracker (botsort or bytetrack)."""
        self.tracker = f"{tracker_name}.yaml"
        log.info("Tracker set to: %s", self.tracker)

    def export_to_tensorrt(self) -> str:
        """Export current .pt model to TensorRT .engine format."""
        if not self.segment_model or not self.segment_model_name.endswith(".pt"):
            return "❌ Load a PyTorch (.pt) model first!"

        try:
            log.info("Starting TensorRT export for %s...", self.segment_model_name)
            self.segment_model.export(format="engine", device=0, half=True, workspace=4)
            engine_name = self.segment_model_name.replace(".pt", ".engine")
            self.load_segmentation_model(engine_name)
            return f"✅ Export Success! Switched to {engine_name}"
        except Exception as e:
            log.error("TRT export failed: %s", e)
            return f"❌ Export Failed: {e}"

    # ── Face recognition ─────────────────────────────────────────────────────

    def _get_face_recognizer(self):
        """Lazy-load FaceRecognizer on first use."""
        with self._fr_lock:
            if self._face_recognizer is None:
                try:
                    from face_recognizer import FaceRecognizer
                    self._face_recognizer = FaceRecognizer()
                    log.info("FaceRecognizer loaded")
                except Exception as e:
                    log.error("FaceRecognizer load failed: %s", e)
                    return None
            return self._face_recognizer

    def set_face_recognition(self, enabled: bool) -> str:
        """Enable or disable face recognition overlay."""
        self.face_recognition_enabled = enabled
        if enabled:
            fr = self._get_face_recognizer()
            if fr is None:
                return "❌ Face recognizer failed to load"
            return f"👤 Face recognition ON\n{fr.gallery_status()}"
        return "👤 Face recognition OFF"

    def set_face_threshold(self, val: float) -> None:
        """Set face recognition similarity threshold."""
        fr = self._get_face_recognizer()
        if fr:
            fr.threshold = float(val)

    def rebuild_gallery(self) -> str:
        """Rebuild face gallery from the faces/ directory."""
        fr = self._get_face_recognizer()
        if fr is None:
            return "❌ Face recognizer not available"
        return fr.build_gallery()

    def add_person_to_gallery(self, name: str, image_paths: list[str]) -> str:
        """Add a person to the gallery from uploaded images."""
        fr = self._get_face_recognizer()
        if fr is None:
            return "❌ Face recognizer not available"
        return fr.add_person(name, image_paths)

    def start_learning(self, name: str, count: int = 5) -> str:
        """Start capturing face samples from the live stream."""
        if not name:
            return "❌ Name required"
        self.learning_name = name
        self.learning_count = count
        self.learning_max = count
        return f"📸 Learning {name} in next {count} frames..."

    def get_gallery_status(self) -> str:
        """Get face gallery status."""
        fr = self._get_face_recognizer()
        if fr is None:
            return "Models loading..."
        return fr.gallery_status()

    # ── Animal recognition ───────────────────────────────────────────────────

    def _get_animal_recognizer(self):
        """Lazy-load AnimalRecognizer on first use."""
        with self._ar_lock:
            if self._animal_recognizer is None:
                try:
                    from animal_recognizer import AnimalRecognizer
                    self._animal_recognizer = AnimalRecognizer(device=self.device)
                    log.info("AnimalRecognizer loaded")
                except Exception as e:
                    log.error("AnimalRecognizer load failed: %s", e)
                    return None
            return self._animal_recognizer

    def set_animal_recognition(self, enabled: bool) -> str:
        """Enable or disable animal recognition overlay."""
        self.animal_recognition_enabled = enabled
        if enabled:
            ar = self._get_animal_recognizer()
            if ar is None:
                return "❌ Animal recognizer failed to load"
            return f"🐱 Animal recognition ON\n{ar.gallery_status()}"
        return "🐱 Animal recognition OFF"

    def set_animal_threshold(self, val: float) -> None:
        """Set animal recognition similarity threshold."""
        ar = self._get_animal_recognizer()
        if ar:
            ar.threshold = float(val)

    def rebuild_animal_gallery(self) -> str:
        """Rebuild animal gallery from the animals/ directory."""
        ar = self._get_animal_recognizer()
        if ar is None:
            return "❌ Animal recognizer not available"
        return ar.build_gallery()

    def add_animal_to_gallery(self, name: str, image_paths: list[str]) -> str:
        """Add an animal to the gallery from uploaded images."""
        ar = self._get_animal_recognizer()
        if ar is None:
            return "❌ Animal recognizer not available"
        return ar.add_animal(name, image_paths)

    def start_learning_animal(self, name: str, count: int = 5) -> str:
        """Start capturing animal samples from the live stream."""
        if not name:
            return "❌ Name required"
        self.animal_learning_name = name
        self.animal_learning_count = count
        self.animal_learning_max = count
        return f"📸 Learning Animal {name} in next {count} frames..."

    def get_animal_gallery_status(self) -> str:
        """Get animal gallery status."""
        ar = self._get_animal_recognizer()
        if ar is None:
            return "Models loading..."
        return ar.gallery_status()

    # ── Helpers (DRY) ────────────────────────────────────────────────────────

    def _is_engine_model(self) -> bool:
        return self.segment_model_name.endswith(".engine")

    @staticmethod
    def _extract_animal_crop(
        frame_bgr: np.ndarray,
        box: np.ndarray,
        masks,
        mask_index: int,
    ) -> np.ndarray:
        """Extract a masked animal crop from a frame.

        If a segmentation mask is available for this detection, it is applied
        to zero out background pixels. Otherwise returns a plain bbox crop.
        """
        x1, y1, x2, y2 = box
        crop = frame_bgr[y1:y2, x1:x2].copy()

        if masks is not None and len(masks.xy) > mask_index:
            poly = masks.xy[mask_index]
            if len(poly) > 0:
                m = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                poly_shifted = poly - np.array([x1, y1])
                cv2.fillPoly(m, [np.int32(poly_shifted)], 255)
                crop = cv2.bitwise_and(crop, crop, mask=m)

        return crop

    @staticmethod
    def _draw_label(
        frame: np.ndarray,
        x1: int,
        y1: int,
        label_text: str,
        box_color: tuple[int, int, int],
    ) -> None:
        """Draw a label with background rectangle above a bounding box."""
        t_size = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )[0]
        cv2.rectangle(
            frame,
            (int(x1), int(y1 - t_size[1] - 4)),
            (int(x1 + t_size[0]), int(y1)),
            box_color,
            -1,
        )
        cv2.putText(
            frame,
            label_text,
            (int(x1), int(y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _resolve_person_label(
        self,
        frame_bgr: np.ndarray,
        box: np.ndarray,
        track_id: Optional[int],
    ) -> tuple[str, tuple[int, int, int]]:
        """Identify a person via face recognition with temporal smoothing."""
        x1, y1, x2, y2 = box
        fr = self._get_face_recognizer()
        fr_ready = fr is not None and fr.is_ready()

        if not fr_ready:
            tid_str = f"#{track_id}" if track_id is not None else ""
            return f"Person {tid_str}", (255, 140, 0)

        name, conf = fr.identify(frame_bgr, (x1, y1, x2, y2))
        final_name = name

        # Temporal smoothing via majority vote
        if track_id is not None:
            if track_id not in self.face_history:
                self.face_history[track_id] = deque(
                    maxlen=cfg.ai.face_history_len
                )
            self.face_history[track_id].append(name)

            history = list(self.face_history[track_id])
            if history:
                counts = {n: history.count(n) for n in set(history)}
                most_common = max(counts, key=counts.get)
                if counts[most_common] >= len(history) / 2:
                    final_name = most_common

        if final_name != "Unknown":
            return f"{final_name} ({conf:.2f})", (0, 255, 0)

        tid_str = f"#{track_id}" if track_id is not None else ""
        return f"Person {tid_str}", (255, 140, 0)

    def _resolve_animal_label(
        self,
        frame_bgr: np.ndarray,
        box: np.ndarray,
        masks,
        mask_index: int,
        class_name: str,
        track_id: Optional[int],
    ) -> tuple[str, tuple[int, int, int]]:
        """Identify an animal via DINOv3 with temporal smoothing."""
        ar = self._get_animal_recognizer()
        ar_ready = ar is not None and ar.is_ready()

        if not ar_ready:
            tid_str = f"#{track_id}" if track_id is not None else ""
            return f"{class_name} {tid_str}", (255, 100, 100)

        crop = self._extract_animal_crop(frame_bgr, box, masks, mask_index)
        name, conf = ar.identify(crop)
        final_name = name

        # Temporal smoothing
        if track_id is not None:
            if track_id not in self.animal_history:
                self.animal_history[track_id] = deque(
                    maxlen=cfg.ai.animal_history_len
                )
            self.animal_history[track_id].append(name)

            history = list(self.animal_history[track_id])
            if history:
                counts = {n: history.count(n) for n in set(history)}
                most_common = max(counts, key=counts.get)
                if counts[most_common] >= len(history) / 2:
                    final_name = most_common

        if final_name != "Unknown":
            return f"{final_name} ({conf:.2f})", (150, 0, 200)

        tid_str = f"#{track_id}" if track_id is not None else ""
        return f"{class_name} {tid_str}", (150, 100, 200)

    def _draw_recognition_overlays(
        self,
        annotated_frame: np.ndarray,
        results,
        frame_bgr: np.ndarray,
        track_ids: list[int],
    ) -> None:
        """Draw face/animal recognition labels on annotated frame."""
        if not (self.face_recognition_enabled or self.animal_recognition_enabled):
            return

        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        names_dict = results[0].names
        masks = results[0].masks

        # Clean stale track history
        if track_ids:
            current_ids = set(track_ids)
            for history_dict in (self.face_history, self.animal_history):
                stale = [tid for tid in history_dict if tid not in current_ids]
                for tid in stale:
                    del history_dict[tid]

        for i, (cls_id, box) in enumerate(zip(classes, boxes_xyxy)):
            x1, y1, x2, y2 = box
            class_name = names_dict.get(cls_id, str(cls_id))
            tid = track_ids[i] if (track_ids and i < len(track_ids)) else None

            if cls_id == 0 and self.face_recognition_enabled:
                label, color = self._resolve_person_label(frame_bgr, box, tid)
            elif cls_id in [15, 16] and self.animal_recognition_enabled:
                label, color = self._resolve_animal_label(
                    frame_bgr, box, masks, i, class_name, tid
                )
            else:
                tid_str = f"#{tid}" if tid is not None else ""
                label = f"{class_name} {tid_str}"
                color = (255, 100, 100)

            self._draw_label(annotated_frame, x1, y1, label, color)

    def _run_learning_logic(self, results, frame_bgr: np.ndarray) -> None:
        """Capture samples for face/animal learning if active."""
        if self.learning_count <= 0 and self.animal_learning_count <= 0:
            return

        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        masks = results[0].masks

        # Human face capture
        if self.learning_count > 0:
            largest_box = None
            max_area = 0
            for cls_id, box in zip(classes, boxes_xyxy):
                if cls_id == 0:
                    x1, y1, x2, y2 = box
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        largest_box = box

            if largest_box is not None:
                fr = self._get_face_recognizer()
                if fr:
                    progress = self.learning_max - self.learning_count + 1
                    res = fr.capture_face(
                        frame_bgr, tuple(largest_box), self.learning_name
                    )
                    log.info(
                        "Learning Person (%d/%d): %s",
                        progress, self.learning_max, res,
                    )
                    self.learning_count -= 1
                    if self.learning_count == 0:
                        log.info("Learning person complete — rebuilding gallery")
                        threading.Thread(target=fr.build_gallery).start()

        # Animal capture
        if self.animal_learning_count > 0:
            largest_box = None
            max_area = 0
            best_index = -1
            for i, (cls_id, box) in enumerate(zip(classes, boxes_xyxy)):
                if cls_id in [15, 16]:
                    x1, y1, x2, y2 = box
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        largest_box = box
                        best_index = i

            if largest_box is not None and best_index >= 0:
                ar = self._get_animal_recognizer()
                if ar:
                    crop = self._extract_animal_crop(
                        frame_bgr, largest_box, masks, best_index
                    )
                    progress = self.animal_learning_max - self.animal_learning_count + 1
                    res = ar.capture_animal(crop, self.animal_learning_name)
                    log.info(
                        "Learning Animal (%d/%d): %s",
                        progress, self.animal_learning_max, res,
                    )
                    self.animal_learning_count -= 1
                    if self.animal_learning_count == 0:
                        log.info("Learning animal complete — rebuilding gallery")
                        threading.Thread(target=ar.build_gallery).start()

    # ── Frame processing ─────────────────────────────────────────────────────

    def process_frame(
        self, frame_bgr: np.ndarray, conf: float = 0.5
    ) -> tuple[np.ndarray, str, list[int]]:
        """Run detection (no tracking) on a single frame."""
        if not self.segment_model:
            return frame_bgr, "Model Loading...", []

        start = time.time()
        try:
            if self.model_type == "yolo":
                use_half = self.device == "cuda" and not self._is_engine_model()

                results = self.segment_model(
                    frame_bgr,
                    conf=conf,
                    verbose=False,
                    iou=cfg.ai.iou_threshold,
                    device=self.device,
                    half=use_half,
                    classes=cfg.ai.detection_classes,
                )

                show_default = not (
                    self.face_recognition_enabled or self.animal_recognition_enabled
                )
                annotated_frame = results[0].plot(labels=show_default)

                if not show_default and len(results[0].boxes) > 0:
                    self._draw_recognition_overlays(
                        annotated_frame, results, frame_bgr, []
                    )

                self._run_learning_logic(results, frame_bgr)

                count = len(results[0])
                fps = 1.0 / (time.time() - start)
                h, w = frame_bgr.shape[:2]
                detected_classes = (
                    results[0].boxes.cls.cpu().numpy().astype(int).tolist()
                )
                return (
                    annotated_frame,
                    f"Res: {w}x{h} | Obj: {count} | FPS: {fps:.1f} ({self.device})",
                    detected_classes,
                )

            elif self.model_type == "sam":
                results = self.segment_model(
                    frame_bgr, verbose=False, device=self.device
                )
                annotated_frame = results[0].plot()
                fps = 1.0 / (time.time() - start)
                h, w = frame_bgr.shape[:2]
                return (
                    annotated_frame,
                    f"Res: {w}x{h} | SAM Active | FPS: {fps:.1f} ({self.device})",
                    [],
                )

            else:
                return frame_bgr, f"Unknown model type: {self.model_type}", []

        except Exception as e:
            log.error("Inference error: %s", e)
            return frame_bgr, f"Error: {e}", []

    def track_frame(
        self, frame_bgr: np.ndarray, conf: float = 0.5
    ) -> tuple[np.ndarray, str, list[int], list[int]]:
        """Run multi-object tracking on a single frame.

        Tracking state persists across calls (persist=True).
        """
        if not self.segment_model:
            return frame_bgr, "Model Loading...", [], []

        if self.model_type == "sam":
            result = self.process_frame(frame_bgr, conf)
            return (*result, [])

        start = time.time()
        try:
            use_half = self.device == "cuda" and not self._is_engine_model()

            results = self.segment_model.track(
                frame_bgr,
                persist=True,
                tracker=self.tracker,
                conf=conf,
                iou=cfg.ai.iou_threshold,
                verbose=False,
                device=self.device,
                half=use_half,
                classes=cfg.ai.detection_classes,
            )

            show_default = not (
                self.face_recognition_enabled or self.animal_recognition_enabled
            )
            annotated_frame = results[0].plot(labels=show_default)

            fps = 1.0 / (time.time() - start)
            h, w = frame_bgr.shape[:2]
            count = len(results[0])

            detected_classes = (
                results[0].boxes.cls.cpu().numpy().astype(int).tolist()
            )

            track_ids = []
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.cpu().numpy().astype(int).tolist()

            # Draw recognition overlays
            if not show_default:
                self._draw_recognition_overlays(
                    annotated_frame, results, frame_bgr, track_ids
                )

            self._run_learning_logic(results, frame_bgr)

            tracker_label = "BoT-SORT" if "botsort" in self.tracker else "ByteTrack"
            status = (
                f"Res: {w}x{h} | Tracked: {count} | "
                f"FPS: {fps:.1f} | {tracker_label}"
            )
            return annotated_frame, status, detected_classes, track_ids

        except Exception as e:
            log.error("Tracking error: %s", e)
            return frame_bgr, f"Tracking Error: {e}", [], []

    # ── VLM (Gemma4-E2B) ────────────────────────────────────────────────────

    def describe_scene(
        self, frame_pil: Image.Image, prompt: str = "Describe this image concisely."
    ) -> str:
        """Describe the current frame using the Gemma4-E2B VLM.

        Lazily loads the model on first call using the transformers
        ``pipeline("any-to-any", ...)`` API.
        """
        with self._vlm_lock:
            if self._vlm_pipeline is None:
                log.info("Loading VLM (%s)...", cfg.ai.vlm_model_id)
                try:
                    from transformers import pipeline as hf_pipeline

                    self._vlm_pipeline = hf_pipeline(
                        task="any-to-any",
                        model=cfg.ai.vlm_model_id,
                        device_map="auto",
                        torch_dtype="auto",
                    )
                    log.info("VLM loaded successfully")
                except Exception as e:
                    log.error("Failed to load VLM: %s", e)
                    return f"Failed to load VLM: {e}"

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": frame_pil},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            output = self._vlm_pipeline(messages, max_new_tokens=256)
            # Extract the generated text from pipeline output
            if isinstance(output, list) and len(output) > 0:
                result = output[0]
                if isinstance(result, dict) and "generated_text" in result:
                    generated = result["generated_text"]
                    # The generated_text may include the full conversation;
                    # extract only the assistant's reply
                    if isinstance(generated, list):
                        for msg in reversed(generated):
                            if isinstance(msg, dict) and msg.get("role") == "assistant":
                                content = msg.get("content", "")
                                if isinstance(content, str):
                                    return content
                                if isinstance(content, list):
                                    return " ".join(
                                        c.get("text", "")
                                        for c in content
                                        if isinstance(c, dict) and c.get("type") == "text"
                                    )
                    elif isinstance(generated, str):
                        return generated
                return str(output)
            return "No response from VLM"
        except Exception as e:
            log.error("VLM error: %s", e)
            return f"VLM Error: {e}"
