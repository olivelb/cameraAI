"""
animal_recognizer.py — Animal re-identification using DINOv3.

Uses facebook/dinov3-small via transformers to extract deep features
from animals (cats/dogs). Works best with segmented crops but also
supports bounding-box crops.

Gallery workflow:
  1. Drop photos into  animals/<name>/photo.jpg  (5-20 photos per pet)
  2. Call build_gallery() — computes mean embedding per animal
  3. At runtime, identify(crop_bgr) embeds the crop and returns
     (name, confidence) via cosine similarity.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import cfg

log = logging.getLogger(__name__)

ANIMALS_DIR = Path(__file__).parent / "animals"
GALLERY_FILE = ANIMALS_DIR / "gallery.npz"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


class AnimalRecognizer:
    """DINOv3-based animal re-identification engine."""

    def __init__(self, device: str = "cpu") -> None:
        self._lock = threading.Lock()
        self.device: str = device
        self.extractor = None
        self.processor = None
        self._gallery: dict[str, np.ndarray] = {}
        self.enabled: bool = False
        self.threshold: float = cfg.ai.animal_threshold
        self._model_loading: bool = False

        self._try_load_gallery()
        threading.Thread(
            target=self._load_model, daemon=True, name="dinov3-loader"
        ).start()

    def _load_model(self) -> None:
        """Download and load the DINOv3 model in background."""
        with self._lock:
            if self.extractor is not None or self._model_loading:
                return
            self._model_loading = True

        try:
            model_id = cfg.ai.animal_embed_model
            log.info("Loading %s for animal recognition...", model_id)

            import torch
            from transformers import AutoImageProcessor, AutoModel

            if self.device == "cuda" and not torch.cuda.is_available():
                self.device = "cpu"

            processor = AutoImageProcessor.from_pretrained(model_id)
            extractor = AutoModel.from_pretrained(model_id).to(self.device).eval()

            with self._lock:
                self.processor = processor
                self.extractor = extractor
                self._model_loading = False

            log.info("AnimalRecognizer loaded on %s", self.device)
        except Exception as e:
            log.error("AnimalRecognizer load failed: %s", e)
            with self._lock:
                self._model_loading = False

    def is_ready(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        return self.extractor is not None

    def _embed(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Compute a normalized embedding for an animal crop."""
        if not self.is_ready() or image_bgr.size == 0:
            return None

        try:
            import torch

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            inputs = self.processor(images=image_rgb, return_tensors="pt").to(
                self.device
            )
            with torch.no_grad():
                outputs = self.extractor(**inputs)
                emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

            emb /= np.linalg.norm(emb) + 1e-8
            return emb
        except Exception as e:
            log.error("Embedding error: %s", e)
            return None

    def build_gallery(self) -> str:
        """Scan animals/<name>/ subfolders and build mean embeddings."""
        if not self.is_ready():
            return "❌ Models still loading..."

        ANIMALS_DIR.mkdir(parents=True, exist_ok=True)
        gallery: dict[str, np.ndarray] = {}
        total_photos = 0

        for animal_dir in sorted(ANIMALS_DIR.iterdir()):
            if not animal_dir.is_dir() or animal_dir.name.startswith("."):
                continue
            name = animal_dir.name
            embeddings: list[np.ndarray] = []

            for img_path in sorted(animal_dir.glob("*")):
                if img_path.suffix.lower() not in {
                    ".jpg", ".jpeg", ".png", ".webp"
                }:
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                emb = self._embed(img)
                if emb is not None:
                    embeddings.append(emb)
                    total_photos += 1

            if embeddings:
                mean_emb = np.mean(embeddings, axis=0)
                mean_emb /= np.linalg.norm(mean_emb) + 1e-8
                gallery[name] = mean_emb
                log.info("  %s: %d photos → 1 embedding", name, len(embeddings))

        with self._lock:
            self._gallery = gallery

        self._save_gallery()
        names = list(gallery.keys())
        return f"✅ Gallery built: {names} ({total_photos} photos)"

    def add_animal(self, name: str, image_paths: list[str]) -> str:
        """Add or update an animal in the gallery from uploaded images."""
        if not self.is_ready():
            return "❌ Models loading..."

        animal_dir = ANIMALS_DIR / name
        animal_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for i, path in enumerate(image_paths):
            img = cv2.imread(path)
            if img is None:
                continue
            dest = animal_dir / f"photo_{int(time.time() * 1000)}_{i}{Path(path).suffix}"
            cv2.imwrite(str(dest), img)
            saved += 1

        if saved == 0:
            return f"❌ No valid images for {name}"

        threading.Thread(target=self.build_gallery, daemon=True).start()
        return f"✅ Added {saved} photos for {name}. Rebuilding gallery..."

    def capture_animal(self, image_bgr: np.ndarray, name: str) -> str:
        """Save an animal crop from the live stream to disk."""
        if not name:
            return "❌ Name required"

        animal_dir = ANIMALS_DIR / name
        animal_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time() * 1000)
        filename = f"capture_{timestamp}.jpg"
        path = animal_dir / filename
        cv2.imwrite(str(path), image_bgr)
        return f"📸 Saved {filename}"

    def identify(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        """Identify an animal crop against the gallery.

        Returns (name, confidence) or ("Unknown", similarity).
        """
        if not self.is_ready() or crop_bgr.size == 0:
            return "Unknown", 0.0

        with self._lock:
            if not self._gallery:
                return "Unknown", 0.0

        emb = self._embed(crop_bgr)
        if emb is None:
            return "Unknown", 0.0

        with self._lock:
            gallery = dict(self._gallery)

        best_name, best_sim = "Unknown", -1.0
        for name, ref_emb in gallery.items():
            sim = _cosine(emb, ref_emb)
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_sim >= self.threshold:
            return best_name, best_sim
        return "Unknown", best_sim

    def gallery_status(self) -> str:
        """Return a human-readable gallery status string."""
        with self._lock:
            if not self._gallery:
                return "Gallery empty — add photos and click Rebuild"
            lines = [f"  • {name}" for name in sorted(self._gallery)]
            return "Known animals:\n" + "\n".join(lines)

    def _save_gallery(self) -> None:
        """Persist gallery embeddings to disk."""
        try:
            ANIMALS_DIR.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = dict(self._gallery)
            np.savez(str(GALLERY_FILE), **data)
            log.info("Animal gallery saved to %s", GALLERY_FILE)
        except Exception as e:
            log.warning("Could not save gallery: %s", e)

    def _try_load_gallery(self) -> None:
        """Load persisted gallery from disk if available."""
        if not GALLERY_FILE.exists():
            return
        try:
            data = np.load(str(GALLERY_FILE))
            with self._lock:
                self._gallery = {k: data[k] for k in data.files}
            log.info("Animal gallery loaded: %s", list(self._gallery.keys()))
        except Exception as e:
            log.warning("Could not load gallery: %s", e)
