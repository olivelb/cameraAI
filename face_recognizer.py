"""
face_recognizer.py — GPU-accelerated face recognition via ONNX inference.

Uses insightface's pre-trained ONNX models directly via onnxruntime,
bypassing the insightface Python wrapper.

Models (auto-downloaded on first use, ~200 MB total):
  - det_10g.onnx     : RetinaFace face detector
  - w600k_r50.onnx   : ArcFace R50 face embedding (512-dim)

Gallery workflow:
  1. Drop photos into  faces/<name>/photo.jpg  (5-20 photos per person)
  2. Call build_gallery() — computes mean embedding per person
  3. At runtime, identify(frame_bgr, bbox) crops the person, detects the
     face, embeds it, and returns (name, confidence) via cosine similarity.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

from config import cfg

log = logging.getLogger(__name__)

# ── Model paths ──────────────────────────────────────────────────────────────
_MODEL_DIR = Path(__file__).parent / "models" / "insightface"
_ZIP_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
)
_DET_PATH = _MODEL_DIR / "det_10g.onnx"
_REC_PATH = _MODEL_DIR / "w600k_r50.onnx"

# ── Gallery paths ────────────────────────────────────────────────────────────
FACES_DIR = Path(__file__).parent / "faces"
GALLERY_FILE = FACES_DIR / "gallery.npz"


# ── Model download ───────────────────────────────────────────────────────────

def _download_and_extract(url: str, dest_dir: Path) -> None:
    """Download and extract insightface model weights."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "buffalo_l.zip"

    log.info("Downloading models from %s...", url)
    urllib.request.urlretrieve(url, zip_path)
    log.info("Downloaded %s (%d MB)", zip_path.name, zip_path.stat().st_size // 1024 // 1024)

    log.info("Extracting models...")
    with zipfile.ZipFile(zip_path, "r") as z:
        for file in z.namelist():
            if file.endswith("det_10g.onnx"):
                with z.open(file) as src, open(dest_dir / "det_10g.onnx", "wb") as dst:
                    shutil.copyfileobj(src, dst)
            elif file.endswith("w600k_r50.onnx"):
                with z.open(file) as src, open(dest_dir / "w600k_r50.onnx", "wb") as dst:
                    shutil.copyfileobj(src, dst)

    zip_path.unlink()
    log.info("Models extracted and ready")


def _ort_session(path: Path) -> ort.InferenceSession:
    """Create an ONNX Runtime session, preferring CUDA then CPU."""
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(path), providers=providers)
    used = sess.get_providers()[0]
    log.info("Loaded %s on %s", path.name, used)
    return sess


# ── RetinaFace pre/post-processing ──────────────────────────────────────────

_DET_INPUT_SIZE = (640, 640)
_DET_MEAN = np.array([127.5, 127.5, 127.5], dtype=np.float32)
_DET_STD = 128.0
_STRIDES = [8, 16, 32]
_NUM_ANCH = 2


def _preprocess_det(img_bgr: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Resize + normalize for RetinaFace."""
    h, w = img_bgr.shape[:2]
    scale_x = _DET_INPUT_SIZE[0] / w
    scale_y = _DET_INPUT_SIZE[1] / h
    resized = cv2.resize(img_bgr, _DET_INPUT_SIZE)
    blob = (resized.astype(np.float32) - _DET_MEAN) / _DET_STD
    blob = blob.transpose(2, 0, 1)[np.newaxis]  # NCHW
    return blob, scale_x, scale_y


def _decode_det(
    outputs: list[np.ndarray],
    scale_x: float,
    scale_y: float,
    score_thresh: float = 0.5,
    nms_thresh: float = 0.4,
) -> list[tuple[int, int, int, int]]:
    """Decode RetinaFace outputs → list of (x1, y1, x2, y2) in original coords."""
    scores_list: list[np.ndarray] = []
    boxes_list: list[np.ndarray] = []
    num_strides = len(_STRIDES)

    for i, stride in enumerate(_STRIDES):
        score_idx = i
        box_idx = i + num_strides

        fh = _DET_INPUT_SIZE[1] // stride
        fw = _DET_INPUT_SIZE[0] // stride

        raw_score = outputs[score_idx]
        if raw_score.shape[1] == 1:
            score = raw_score[:, 0]
        else:
            score = raw_score[:, 1]

        box = outputs[box_idx]

        cx = (np.arange(fw) + 0.5) * stride
        cy = (np.arange(fh) + 0.5) * stride
        cx, cy = np.meshgrid(cx, cy)
        centres = np.stack([cx.ravel(), cy.ravel()], axis=1)
        centres = np.repeat(centres, _NUM_ANCH, axis=0)

        x1 = centres[:, 0] - box[:, 0] * stride
        y1 = centres[:, 1] - box[:, 1] * stride
        x2 = centres[:, 0] + box[:, 2] * stride
        y2 = centres[:, 1] + box[:, 3] * stride

        mask = score > score_thresh
        scores_list.append(score[mask])
        boxes_list.append(np.stack([x1, y1, x2, y2], axis=1)[mask])

    if not scores_list:
        return []

    scores = np.concatenate(scores_list)
    boxes = np.concatenate(boxes_list)

    keep = cv2.dnn.NMSBoxes(
        boxes[:, :4].tolist(), scores.tolist(), score_thresh, nms_thresh
    )
    if len(keep) == 0:
        return []

    keep = keep.flatten()
    result = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        result.append((
            int(x1 / scale_x), int(y1 / scale_y),
            int(x2 / scale_x), int(y2 / scale_y),
        ))
    return result


# ── ArcFace pre/post-processing ──────────────────────────────────────────────

_REC_INPUT_SIZE = (112, 112)


def _align_face(
    img_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int
) -> Optional[np.ndarray]:
    """Crop and resize face region to 112×112 for ArcFace."""
    h, w = img_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    face = img_bgr[y1:y2, x1:x2]
    face = cv2.resize(face, _REC_INPUT_SIZE)
    return face


def _apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization to Y channel."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge((l_ch, a_ch, b_ch))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _preprocess_rec(face_bgr: np.ndarray) -> np.ndarray:
    """Pre-process a 112×112 face crop for ArcFace."""
    face_bgr = _apply_clahe(face_bgr)
    face = face_bgr[:, :, ::-1].astype(np.float32)  # BGR→RGB
    face = (face - 127.5) / 128.0
    return face.transpose(2, 0, 1)[np.newaxis]  # NCHW


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ── Main class ───────────────────────────────────────────────────────────────

class FaceRecognizer:
    """GPU-accelerated face recognition using RetinaFace + ArcFace ONNX models.

    Usage::

        fr = FaceRecognizer()
        fr.build_gallery()
        name, conf = fr.identify(frame_bgr, (x1, y1, x2, y2))
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._det_sess: Optional[ort.InferenceSession] = None
        self._rec_sess: Optional[ort.InferenceSession] = None
        self._gallery: dict[str, np.ndarray] = {}
        self.enabled: bool = False
        self.threshold: float = cfg.ai.face_threshold
        self._load_models()
        self._try_load_gallery()

    # ── Model loading ────────────────────────────────────────────────────────

    def _load_models(self) -> None:
        """Load RetinaFace detector and ArcFace recognizer."""
        try:
            if not _DET_PATH.exists() or not _REC_PATH.exists():
                _download_and_extract(_ZIP_URL, _MODEL_DIR)

            self._det_sess = _ort_session(_DET_PATH)
            self._rec_sess = _ort_session(_REC_PATH)
            log.info("FaceRecognizer models loaded")
        except Exception as e:
            log.error("FaceRecognizer model load failed: %s", e)

    def is_ready(self) -> bool:
        """Check if both models are loaded."""
        return self._det_sess is not None and self._rec_sess is not None

    # ── Gallery management ───────────────────────────────────────────────────

    def build_gallery(self, faces_dir: Optional[Path] = None) -> str:
        """Scan faces/<name>/ subfolders, compute mean ArcFace embedding per person.

        Saves result to faces/gallery.npz for fast startup.
        """
        if not self.is_ready():
            return "❌ Models not loaded"

        faces_dir = Path(faces_dir) if faces_dir else FACES_DIR
        if not faces_dir.exists():
            faces_dir.mkdir(parents=True)
            return "❌ faces/ directory was empty — add photos first"

        gallery: dict[str, np.ndarray] = {}
        total_photos = 0

        for person_dir in sorted(faces_dir.iterdir()):
            if not person_dir.is_dir() or person_dir.name.startswith("."):
                continue
            name = person_dir.name
            embeddings: list[np.ndarray] = []

            for img_path in sorted(person_dir.glob("*")):
                if img_path.suffix.lower() not in {
                    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
                }:
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    log.warning("Could not read %s", img_path)
                    continue
                emb = self._embed_best_face(img)
                if emb is not None:
                    embeddings.append(emb)
                    total_photos += 1
                else:
                    log.warning("No face detected in %s", img_path.name)

            if embeddings:
                mean_emb = np.mean(embeddings, axis=0)
                mean_emb /= np.linalg.norm(mean_emb) + 1e-8
                gallery[name] = mean_emb
                log.info("  %s: %d photos → 1 embedding", name, len(embeddings))
            else:
                log.warning("  %s: no usable photos", name)

        with self._lock:
            self._gallery = gallery

        self._save_gallery()
        names = list(gallery.keys())
        return f"✅ Gallery built: {names} ({total_photos} photos)"

    def add_person(self, name: str, image_paths: list[str]) -> str:
        """Add or update a person in the gallery from uploaded image paths."""
        if not self.is_ready():
            return "❌ Models not loaded"

        person_dir = FACES_DIR / name
        person_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for i, path in enumerate(image_paths):
            img = cv2.imread(path)
            if img is None:
                continue
            dest = person_dir / f"photo_{i:03d}{Path(path).suffix}"
            cv2.imwrite(str(dest), img)
            saved += 1

        if saved == 0:
            return f"❌ No valid images for {name}"

        return self.build_gallery()

    def capture_face(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[int, int, int, int],
        name: str,
    ) -> str:
        """Save the face in bbox to faces/<name>/captured_<timestamp>.jpg."""
        if not name:
            return "❌ Name required"

        x1, y1, x2, y2 = bbox
        h, w = frame_bgr.shape[:2]

        # Add margin
        bh, bw = y2 - y1, x2 - x1
        x1 = max(0, x1 - int(bw * 0.2))
        y1 = max(0, y1 - int(bh * 0.2))
        x2 = min(w, x2 + int(bw * 0.2))
        y2 = min(h, y2 + int(bh * 0.2))

        face_img = frame_bgr[y1:y2, x1:x2]
        if face_img.size == 0:
            return "❌ Invalid face region"

        timestamp = int(time.time() * 1000)
        person_dir = FACES_DIR / name
        person_dir.mkdir(parents=True, exist_ok=True)

        filename = f"capture_{timestamp}.jpg"
        path = person_dir / filename
        cv2.imwrite(str(path), face_img)
        log.info("Captured face for %s: %s", name, path)
        return f"📸 Saved {filename}"

    def gallery_status(self) -> str:
        """Return a human-readable gallery status string."""
        with self._lock:
            if not self._gallery:
                return "Gallery empty — add photos and click Rebuild"
            lines = [f"  • {name}" for name in sorted(self._gallery)]
            return "Known people:\n" + "\n".join(lines)

    def _save_gallery(self) -> None:
        """Persist gallery embeddings to disk."""
        try:
            FACES_DIR.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = dict(self._gallery)
            np.savez(str(GALLERY_FILE), **data)
            log.info("Face gallery saved to %s", GALLERY_FILE)
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
            log.info("Face gallery loaded: %s", list(self._gallery.keys()))
        except Exception as e:
            log.warning("Could not load gallery: %s", e)

    # ── Inference ────────────────────────────────────────────────────────────

    def _detect_faces(
        self, img_bgr: np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        """Run RetinaFace on the full image, return list of (x1, y1, x2, y2)."""
        blob, sx, sy = _preprocess_det(img_bgr)
        input_name = self._det_sess.get_inputs()[0].name
        outputs = [
            o.numpy() if hasattr(o, "numpy") else o
            for o in self._det_sess.run(None, {input_name: blob})
        ]

        flat = []
        for i, o in enumerate(outputs):
            if hasattr(o, "numpy"):
                o = o.numpy()
            if o.ndim == 3:
                o = o[0]
            flat.append(o)

        return _decode_det(flat, sx, sy)

    def _embed(self, face_bgr: np.ndarray) -> np.ndarray:
        """Run ArcFace on a 112×112 face crop, return normalized 512-dim embedding."""
        blob = _preprocess_rec(face_bgr)
        input_name = self._rec_sess.get_inputs()[0].name
        emb = self._rec_sess.run(None, {input_name: blob})[0][0]
        emb /= np.linalg.norm(emb) + 1e-8
        return emb

    def _embed_best_face(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Detect the largest face in img_bgr and return its embedding."""
        faces = self._detect_faces(img_bgr)
        if not faces:
            return None
        best = max(faces, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        face = _align_face(img_bgr, *best)
        if face is None:
            return None
        return self._embed(face)

    def identify(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> tuple[str, float]:
        """Identify a person by detecting their face within the given bbox.

        Returns (name, confidence) — name is "Unknown" if no match above threshold.
        """
        if not self.is_ready():
            return "Unknown", 0.0

        with self._lock:
            if not self._gallery:
                return "Unknown", 0.0

        x1, y1, x2, y2 = bbox
        h = y2 - y1
        y1_exp = max(0, y1 - int(h * 0.15))
        crop = frame_bgr[y1_exp:y2, x1:x2]
        if crop.size == 0:
            return "Unknown", 0.0

        faces = self._detect_faces(crop)
        if not faces:
            return "Unknown", 0.0

        best_face_box = max(faces, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        face = _align_face(crop, *best_face_box)
        if face is None:
            return "Unknown", 0.0

        emb = self._embed(face)

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
