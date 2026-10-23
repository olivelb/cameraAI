"""Filesystem regressions; use temporary galleries, never personal photos."""

import threading

import numpy as np
import pytest

from gallery_utils import gallery_directory, save_gallery


@pytest.mark.parametrize("name", [
    "", " ", "..", "../outside", "a/b", "a\\b", "C:outside", "/absolute",
    "name.", "name ", "CON", "NUL.jpg", "hidden\x00name", ".hidden",
])
def test_gallery_paths_reject_escape_and_reserved_names(tmp_path, name):
    with pytest.raises(ValueError):
        gallery_directory(tmp_path, name)


def test_gallery_names_support_spaces_and_accents(tmp_path):
    assert gallery_directory(tmp_path, "Chloé Dupont") == tmp_path / "Chloé Dupont"


def test_gallery_rejects_symlink_escape(tmp_path):
    root = tmp_path / "gallery"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (root / "escape").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlinks require Windows developer mode or elevated rights")
    with pytest.raises(ValueError):
        gallery_directory(root, "escape")


def test_atomic_gallery_failure_preserves_previous_archive(tmp_path, monkeypatch):
    path = tmp_path / "gallery.npz"
    save_gallery(path, {"file": np.array([1., 0.])})
    import gallery_utils

    def fail(*args):
        raise OSError("simulated disk error")

    monkeypatch.setattr(gallery_utils.os, "replace", fail)
    with pytest.raises(OSError):
        save_gallery(path, {"Mia": np.array([0., 1.])})
    with np.load(path, allow_pickle=False) as data:
        assert data.files == ["file"]
    assert not list(tmp_path.glob(".gallery-*"))


def test_animal_capture_reports_write_failure(tmp_path, monkeypatch):
    import animal_recognizer as module
    monkeypatch.setattr(module, "ANIMALS_DIR", tmp_path)
    monkeypatch.setattr(module.cv2, "imwrite", lambda *args: False)
    recognizer = module.AnimalRecognizer.__new__(module.AnimalRecognizer)
    assert recognizer.capture_animal(np.zeros((5, 5, 3), np.uint8), "Mia").startswith("❌")


def test_face_capture_rejects_traversal(tmp_path, monkeypatch):
    import face_recognizer as module
    monkeypatch.setattr(module, "FACES_DIR", tmp_path / "faces")
    recognizer = module.FaceRecognizer.__new__(module.FaceRecognizer)
    recognizer._lock = threading.Lock()
    result = recognizer.capture_face(np.zeros((10, 10, 3), np.uint8), (0, 0, 5, 5), "../escape")
    assert result.startswith("❌")
    assert not (tmp_path / "escape").exists()
