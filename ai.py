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

import cv2
import numpy as np
from PIL import Image

from config import cfg
from gallery_utils import gallery_directory

log = logging.getLogger(__name__)

# ── Lazy import placeholders ─────────────────────────────────────────────────
torch = None
YOLO = None
SAM = None


class AIProcessor:
    """Central AI inference engine for detection, tracking, and recognition."""

    def __init__(self, *, autoload: bool = True) -> None:
        log.info("AIProcessor initializing...")
        self.device: str = "cpu"

        self.loaded_models: dict = {}
        self.segment_model = None
        self.segment_model_name: str = cfg.ai.default_model
        self.model_type: str = "yolo"
        self.sam3_prompts: list[str] = ["person", "cat", "dog"]
        self._loader_error: str | None = None

        # Guards every use/swap of segment_model (inference threads vs.
        # model switching vs. VLM-driven release)
        self._infer_lock = threading.RLock()

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
        self._frame_no: int = 0
        # track_id -> (frame_no, label, color): re-identify only every
        # cfg.ai.face_recheck_interval frames per track
        self._face_cache: dict[int, tuple[int, str, tuple[int, int, int]]] = {}

        # Animal recognition
        self.animal_recognition_enabled: bool = False
        self._animal_recognizer = None
        self._ar_lock = threading.Lock()
        self.animal_learning_name: str = ""
        self.animal_learning_count: int = 0
        self.animal_learning_max: int = 5
        self.animal_history: dict[int, deque] = {}
        self._animal_cache: dict[int, tuple[int, str, tuple[int, int, int]]] = {}

        # VLM (Gemma4-E2B)
        self._vlm_model = None
        self._vlm_processor = None
        self._vlm_lock = threading.Lock()
        self._vlm_gen_lock = threading.Lock()

        self._loader_lock = threading.Lock()
        self._loader_started = False
        if autoload:
            self.start_loader()

    def start_loader(self) -> None:
        """Start model loading exactly once, explicitly at application startup."""
        with self._loader_lock:
            if not self._loader_started:
                self._loader_started = True
                threading.Thread(
                    target=self._async_loader, daemon=True, name="ai-loader"
                ).start()

    # ── Background loader ────────────────────────────────────────────────────

    def _async_loader(self) -> None:
        """Import heavy libraries and load default model in background."""
        log.info("Background loader: importing PyTorch & Ultralytics...")
        global torch, YOLO, SAM

        try:
            import torch as _torch
            from ultralytics import YOLO as _YOLO, SAM as _SAM
        except Exception as e:
            self._loader_error = (
                f"Failed to import AI libraries (torch/ultralytics): {e}"
            )
            log.error(self._loader_error)
            return

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
        self, model_name: str, task_type: str = "Detection", sam3_prompt: str = ""
    ) -> str:
        """Load a YOLO or SAM model by name and task type."""
        if YOLO is None:
            log.info("Waiting for libraries to load...")
            deadline = time.time() + 120
            while YOLO is None and self._loader_error is None:
                if time.time() > deadline:
                    return "❌ AI libraries are taking too long to load — check logs"
                time.sleep(0.5)
            if self._loader_error:
                return f"❌ {self._loader_error}"

        log.info("Loading model: %s (%s)...", model_name, task_type)
        try:
            if sam3_prompt.strip():
                self.set_sam3_prompts(sam3_prompt)

            # SAM models use their own checkpoint names (for example
            # ``sam3.pt``).  Unlike YOLO, they must not be rewritten to a
            # ``-seg`` variant when the UI task toggle changes.
            is_sam_model = "sam" in model_name.lower()
            if is_sam_model:
                target_name = model_name
            else:
                base_name = (
                    model_name.replace(".pt", "")
                    .replace(".engine", "")
                    .replace("-seg", "")
                )
                ext = ".engine" if model_name.endswith(".engine") else ".pt"
                target_name = (
                    f"{base_name}-seg{ext}"
                    if task_type == "Segmentation"
                    else f"{base_name}{ext}"
                )

            if is_sam_model and not os.path.isfile(target_name):
                message = (
                    f"SAM checkpoint not found: {target_name}. "
                    "Select the installed SAM 2 Tiny model or place the "
                    "SAM 3 checkpoint in the project root."
                )
                log.error(message)
                return f"❌ {message}"

            # Prefer TensorRT engine if available
            engine_name = target_name.replace(".pt", ".engine")
            if (
                not is_sam_model
                and self.device == "cuda"
                and target_name.endswith(".pt")
                and os.path.exists(engine_name)
            ):
                log.info("Found TensorRT engine: %s — using it instead", engine_name)
                target_name = engine_name

            with self._infer_lock:
                # Check cache
                if target_name in self.loaded_models:
                    log.info("Using cached model: %s", target_name)
                    self.segment_model = self.loaded_models[target_name]
                    self.segment_model_name = target_name
                    self.model_type = (
                        "sam3"
                        if target_name.lower() == "sam3.pt"
                        else "sam" if "sam" in target_name.lower() else "yolo"
                    )
                    self._clear_track_history()
                    return f"Loaded Cached {target_name}"

                if target_name.lower() == "sam3.pt":
                    from ultralytics.models.sam import SAM3SemanticPredictor

                    model = SAM3SemanticPredictor(
                        overrides={
                            "conf": 0.25,
                            "task": "segment",
                            "mode": "predict",
                            "model": target_name,
                            "imgsz": 644,
                            "device": self.device,
                            "save": False,
                            "verbose": False,
                        }
                    )
                    model_type = "sam3"
                elif "sam" in target_name.lower():
                    model = SAM(target_name)
                    model_type = "sam"
                else:
                    if target_name.endswith(".engine"):
                        task = "segment" if task_type == "Segmentation" else "detect"
                        model = YOLO(target_name, task=task)
                    else:
                        model = YOLO(target_name)
                    model_type = "yolo"

                self.segment_model = model
                self.segment_model_name = target_name
                self.model_type = model_type
                self._clear_track_history()
                self._cache_model(target_name, self.segment_model)
            log.info("Model %s loaded successfully", target_name)
            return f"Loaded {target_name}"

        except Exception as e:
            log.error("Model load error: %s", e)
            return f"❌ Model load failed: {e}"

    def _cache_model(self, name: str, model) -> None:
        """Cache a loaded model, evicting oldest entries to bound VRAM."""
        self.loaded_models[name] = model
        max_n = max(1, cfg.ai.max_cached_models)
        while len(self.loaded_models) > max_n:
            evictable = [
                k for k in self.loaded_models if k != self.segment_model_name
            ]
            if not evictable:
                break
            oldest = evictable[0]
            log.info("Evicting cached model to free memory: %s", oldest)
            del self.loaded_models[oldest]
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def set_sam3_prompts(self, prompt_text: str) -> str:
        """Update SAM3 concept prompts from a comma-separated UI value."""
        prompts = [part.strip() for part in prompt_text.split(",") if part.strip()]
        if not prompts:
            return "❌ Enter at least one SAM3 concept, e.g. person, cat, dog"
        self.sam3_prompts = prompts
        return f"SAM3 concepts: {', '.join(prompts)}"

    def set_tracker(self, tracker_name: str) -> None:
        """Set the active tracker (botsort or bytetrack)."""
        if tracker_name not in {"botsort", "bytetrack"}:
            raise ValueError("Unknown tracker")
        with self._infer_lock:
            self.tracker = f"{tracker_name}.yaml"
            # persist=True keeps existing Ultralytics tracker objects even when
            # the filename changes. Rebuild them without registering callbacks twice.
            for model in self.loaded_models.values():
                predictor = getattr(model, "predictor", None)
                if predictor is not None and hasattr(predictor, "trackers"):
                    from ultralytics.trackers.track import on_predict_start

                    predictor.args.tracker = self.tracker
                    on_predict_start(predictor, persist=False)
            self._clear_track_history()
        log.info("Tracker set to: %s", self.tracker)

    def export_to_tensorrt(self) -> str:
        """Export current .pt model to TensorRT .engine format."""
        if not self.segment_model or not self.segment_model_name.endswith(".pt"):
            return "❌ Load a PyTorch (.pt) model first!"
        if self.device != "cuda":
            return "❌ TensorRT export requires a CUDA GPU"

        if self.model_type != "yolo":
            return "❌ TensorRT export is only supported for YOLO models"
        try:
            log.info("Starting TensorRT export for %s...", self.segment_model_name)
            with self._infer_lock:
                self.segment_model.export(format="engine", device=0, half=True, workspace=4)
                engine_name = self.segment_model_name.replace(".pt", ".engine")
                mode = "Segmentation" if "-seg" in engine_name else "Detection"
                result = self.load_segmentation_model(engine_name, mode)
                if result.startswith("❌"):
                    return result
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
        try:
            gallery_directory(os.path.join(os.path.dirname(__file__), "faces"), name)
        except ValueError as e:
            return f"❌ {e}"
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

    def _make_animal_crop_fn(self):
        """Build a callable isolating the largest animal in a photo.

        Uses the active YOLO model (segmentation masks included) so
        gallery embeddings live in the same domain as the masked runtime
        crops. Returns None when no suitable model is loaded.
        """
        if self.model_type != "yolo" or self.segment_model is None:
            return None

        def crop_animal(img_bgr: np.ndarray):
            try:
                with self._infer_lock:
                    results = self.segment_model(
                        img_bgr,
                        conf=0.3,
                        verbose=False,
                        device=self.device,
                        classes=cfg.ai.animal_classes,
                    )
                boxes = results[0].boxes
                if len(boxes) == 0:
                    return None
                xyxy = boxes.xyxy.cpu().numpy().astype(int)
                areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                idx = int(np.argmax(areas))
                return self._extract_animal_crop(
                    img_bgr, xyxy[idx], results[0].masks, idx
                )
            except Exception as e:
                log.warning("Animal gallery crop failed: %s", e)
                return None

        return crop_animal

    def rebuild_animal_gallery(self) -> str:
        """Rebuild animal gallery from the animals/ directory."""
        ar = self._get_animal_recognizer()
        if ar is None:
            return "❌ Animal recognizer not available"
        crop_fn = self._make_animal_crop_fn()
        if crop_fn is None:
            log.warning(
                "No YOLO model active — embedding full photos; switch to a "
                "YOLO model and rebuild for best animal recognition accuracy"
            )
        return ar.build_gallery(crop_fn=crop_fn)

    def add_animal_to_gallery(self, name: str, image_paths: list[str]) -> str:
        """Add an animal to the gallery from uploaded images."""
        ar = self._get_animal_recognizer()
        if ar is None:
            return "❌ Animal recognizer not available"
        return ar.add_animal(name, image_paths, crop_fn=self._make_animal_crop_fn())

    def start_learning_animal(self, name: str, count: int = 5) -> str:
        """Start capturing animal samples from the live stream."""
        try:
            gallery_directory(os.path.join(os.path.dirname(__file__), "animals"), name)
        except ValueError as e:
            return f"❌ {e}"
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
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return frame_bgr[0:0, 0:0].copy()
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
        track_id: int | None,
    ) -> tuple[str, tuple[int, int, int]]:
        """Identify a person via face recognition with temporal smoothing.

        Results are cached per track id and refreshed only every
        ``cfg.ai.face_recheck_interval`` frames to keep the frame rate up.
        """
        if track_id is not None:
            hit = self._face_cache.get(track_id)
            if (
                hit is not None
                and self._frame_no - hit[0] < max(1, cfg.ai.face_recheck_interval)
            ):
                return hit[1], hit[2]

        x1, y1, x2, y2 = box
        fr = self._get_face_recognizer()
        fr_ready = fr is not None and fr.is_ready()

        if not fr_ready:
            tid_str = f"#{track_id}" if track_id is not None else ""
            label, color = f"Person {tid_str}", (255, 140, 0)
        else:
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
                label, color = f"{final_name} ({conf:.2f})", (0, 255, 0)
            else:
                tid_str = f"#{track_id}" if track_id is not None else ""
                label, color = f"Person {tid_str}", (255, 140, 0)

        if track_id is not None:
            self._face_cache[track_id] = (self._frame_no, label, color)
        return label, color

    def _resolve_animal_label(
        self,
        frame_bgr: np.ndarray,
        box: np.ndarray,
        masks,
        mask_index: int,
        class_name: str,
        track_id: int | None,
    ) -> tuple[str, tuple[int, int, int]]:
        """Identify an animal via DINOv3 with temporal smoothing.

        Results are cached per track id and refreshed only every
        ``cfg.ai.animal_recheck_interval`` frames to keep the frame rate up.
        """
        if track_id is not None:
            hit = self._animal_cache.get(track_id)
            if (
                hit is not None
                and self._frame_no - hit[0] < max(1, cfg.ai.animal_recheck_interval)
            ):
                return hit[1], hit[2]

        ar = self._get_animal_recognizer()
        ar_ready = ar is not None and ar.is_ready()

        if not ar_ready:
            tid_str = f"#{track_id}" if track_id is not None else ""
            label, color = f"{class_name} {tid_str}", (255, 100, 100)
        else:
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
                label, color = f"{final_name} ({conf:.2f})", (150, 0, 200)
            else:
                tid_str = f"#{track_id}" if track_id is not None else ""
                label, color = f"{class_name} {tid_str}", (150, 100, 200)

        if track_id is not None:
            self._animal_cache[track_id] = (self._frame_no, label, color)
        return label, color

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

        # Clean stale track history and identification caches
        self._prune_track_history(set(track_ids))

        for i, (cls_id, box) in enumerate(zip(classes, boxes_xyxy, strict=False)):
            x1, y1, x2, y2 = box
            class_name = names_dict.get(cls_id, str(cls_id))
            tid = track_ids[i] if (track_ids and i < len(track_ids)) else None

            if cls_id == cfg.ai.person_class and self.face_recognition_enabled:
                label, color = self._resolve_person_label(frame_bgr, box, tid)
            elif cls_id in cfg.ai.animal_classes and self.animal_recognition_enabled:
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
            for cls_id, box in zip(classes, boxes_xyxy, strict=False):
                if cls_id == cfg.ai.person_class:
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
            for i, (cls_id, box) in enumerate(zip(classes, boxes_xyxy, strict=False)):
                if cls_id in cfg.ai.animal_classes:
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

    def _prune_track_history(self, current_ids: set[int]) -> None:
        for per_track in (
            self.face_history, self.animal_history, self._face_cache, self._animal_cache,
        ):
            for tid in set(per_track) - current_ids:
                del per_track[tid]

    def _clear_track_history(self) -> None:
        self._prune_track_history(set())

    def process_frame(
        self, frame_bgr: np.ndarray, conf: float = 0.5
    ) -> tuple[np.ndarray, str, list[int]]:
        """Run detection (no tracking) on a single frame."""
        if not self.segment_model:
            return frame_bgr, "Model Loading...", []

        self._infer_lock.acquire()
        try:
            self._frame_no += 1
            start = time.time()
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

            elif self.model_type == "sam3":
                # Run SAM3 on every received frame. If processing is slower
                # than the camera feed, the AI loop naturally uses the newest
                # available frame rather than queueing stale work.
                self.segment_model.set_image(frame_bgr)
                results = self.segment_model(text=self.sam3_prompts)
                result = results[0]
                annotated_frame = result.plot()
                count = len(result)
                fps = 1.0 / (time.time() - start)
                h, w = frame_bgr.shape[:2]
                return (
                    annotated_frame,
                    f"Res: {w}x{h} | SAM3: {count} | "
                    f"FPS: {fps:.1f} ({self.device})",
                    [],
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
        finally:
            self._infer_lock.release()

    def track_frame(
        self, frame_bgr: np.ndarray, conf: float = 0.5
    ) -> tuple[np.ndarray, str, list[int], list[int]]:
        """Run multi-object tracking on a single frame.

        Tracking state persists across calls (persist=True).
        """
        if not self.segment_model:
            return frame_bgr, "Model Loading...", [], []

        if self.model_type in {"sam", "sam3"}:
            result = self.process_frame(frame_bgr, conf)
            return (*result, [])

        self._infer_lock.acquire()
        try:
            self._frame_no += 1
            start = time.time()
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
            self._prune_track_history(set(track_ids))

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
        finally:
            self._infer_lock.release()

    # ── VLM (Gemma4-E2B) ────────────────────────────────────────────────────

    def describe_scene(
        self,
        frame_pil: Image.Image,
        prompt: str = "",
        history: list | None = None,
    ) -> str:
        """Describe the current frame using the Gemma4-E2B VLM.

        Args:
            frame_pil: current frame as a PIL image (RGB).
            prompt: question or instruction for the model.
            history: optional prior chat messages (``{"role", "content"}``
                dicts, text only) used as conversation context.

        Lazily loads Gemma in 4-bit NF4 precision. The E2B model has 5.1B
        total parameters (including embeddings), so generic ``device_map``
        loading in BF16 spills it across both VRAM and system RAM.
        """
        if torch is None:
            return "VLM unavailable: AI libraries are still loading"
        if self.device != "cuda":
            return "VLM requires a CUDA GPU for 4-bit inference"
        with self._vlm_lock:
            if self._vlm_model is None:
                log.info("Loading VLM (%s)...", cfg.ai.vlm_model_id)
                try:
                    from transformers import (
                        AutoModelForMultimodalLM,
                        AutoProcessor,
                        BitsAndBytesConfig,
                    )

                    # SAM3 and Gemma cannot comfortably coexist at full
                    # precision on an 8 GB GPU. Release SAM3 before loading
                    # the assistant; selecting it again reloads it on demand.
                    # The inference lock keeps the AI loop from using the
                    # model while it is being torn down.
                    if self.model_type == "sam3":
                        log.info("Releasing SAM3 GPU memory for the VLM")
                        with self._infer_lock:
                            self.loaded_models.pop(self.segment_model_name, None)
                            self.segment_model = None
                            self.segment_model_name = ""
                            self.model_type = "none"
                        if torch is not None and torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                    self._vlm_processor = AutoProcessor.from_pretrained(
                        cfg.ai.vlm_model_id
                    )
                    self._vlm_model = AutoModelForMultimodalLM.from_pretrained(
                        cfg.ai.vlm_model_id,
                        quantization_config=quantization_config,
                        # ``auto`` offloads part of Gemma E2B to RAM despite
                        # the 4-bit model fitting on the 8 GB 4070. Force a
                        # single-GPU placement to avoid the huge RAM spike and
                        # very slow CPU/GPU transfers.
                        device_map={"": 0},
                        dtype=torch.bfloat16,
                        low_cpu_mem_usage=True,
                    )
                    log.info("VLM loaded successfully")
                except Exception as e:
                    self._vlm_model = None
                    self._vlm_processor = None
                    log.error("Failed to load VLM: %s", e)
                    return f"Failed to load VLM: {e}"

        try:
            messages: list[dict] = []
            for h in history or []:
                role = h.get("role")
                content = h.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                if role == "user":
                    messages.append(
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": str(content)}],
                        }
                    )
                elif role == "assistant":
                    messages.append({"role": "assistant", "content": str(content)})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": frame_pil},
                        {"type": "text", "text": prompt},
                    ],
                }
            )
            # One generation at a time: chat and alert threads serialize here
            with self._vlm_gen_lock:
                inputs = self._vlm_processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                ).to(self._vlm_model.device)
                input_len = inputs["input_ids"].shape[-1]
                with torch.inference_mode():
                    output_ids = self._vlm_model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=False,
                        repetition_penalty=1.08,
                    )
                response = self._vlm_processor.decode(
                    output_ids[0][input_len:], skip_special_tokens=False
                )
            parsed = self._vlm_processor.parse_response(response)
            if isinstance(parsed, str):
                return parsed
            if isinstance(parsed, dict):
                content = parsed.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return " ".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
            if isinstance(parsed, list):
                for message in reversed(parsed):
                    if isinstance(message, dict) and message.get("role") == "assistant":
                        content = message.get("content", "")
                        if isinstance(content, str):
                            return content
                        if isinstance(content, list):
                            return " ".join(
                                part.get("text", "")
                                for part in content
                                if isinstance(part, dict) and part.get("type") == "text"
                            )
            return str(parsed)
        except Exception as e:
            log.error("VLM error: %s", e)
            return f"VLM Error: {e}"
