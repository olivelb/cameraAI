import numpy as np

import ai as module


def test_sam3_tracking_falls_back_to_segmentation(monkeypatch):
    processor = module.AIProcessor(autoload=False)
    processor.segment_model = object()
    processor.model_type = "sam3"
    frame = np.zeros((10, 10, 3), np.uint8)
    monkeypatch.setattr(processor, "process_frame", lambda *args: (frame, "SAM3", []))
    result = processor.track_frame(frame)
    assert result[1:] == ("SAM3", [], [])


def test_failed_model_switch_preserves_active_state(monkeypatch):
    processor = module.AIProcessor(autoload=False)
    previous = object()
    processor.segment_model = previous
    processor.segment_model_name = "previous.pt"

    def fail(name):
        raise RuntimeError("checkpoint invalid")

    monkeypatch.setattr(module, "YOLO", fail)
    result = processor.load_segmentation_model("missing.pt")
    assert result.startswith("❌")
    assert processor.segment_model is previous
    assert processor.segment_model_name == "previous.pt"


def test_cpu_does_not_auto_select_tensorrt(monkeypatch):
    processor = module.AIProcessor(autoload=False)
    loaded = []
    monkeypatch.setattr(module.os.path, "exists", lambda *args: True)
    monkeypatch.setattr(module, "YOLO", lambda name: loaded.append(name) or object())
    processor.load_segmentation_model("yolo26n.pt")
    assert loaded == ["yolo26n.pt"]


def test_model_file_picker_does_not_treat_parent_directory_as_sam(tmp_path, monkeypatch):
    processor = module.AIProcessor(autoload=False)
    loaded = []
    monkeypatch.setattr(module, "YOLO", lambda name: loaded.append(name) or object())
    model_path = str(tmp_path / "sam-gallery" / "yolo26n.pt")
    assert processor.load_segmentation_model(model_path).startswith("Loaded")
    assert loaded == [model_path]
    assert processor.model_type == "yolo"


def test_crop_clamps_negative_boxes_and_masks():
    frame = np.full((10, 20, 3), 255, np.uint8)
    crop = module.AIProcessor._extract_animal_crop(frame, [-10, -10, 30, 30], None, 0)
    assert crop.shape == frame.shape
    assert module.AIProcessor._extract_animal_crop(frame, [15, 0, 10, 5], None, 0).size == 0


def test_learning_rejects_paths_without_loading_models():
    processor = module.AIProcessor(autoload=False)
    assert processor.start_learning("../escape").startswith("❌")
    assert processor.start_learning_animal("../escape").startswith("❌")


def test_track_cache_cleared_when_all_tracks_disappear():
    processor = module.AIProcessor(autoload=False)
    processor._face_cache[1] = (1, "Alice", (0, 255, 0))
    processor._animal_cache[2] = (1, "Mia", (0, 255, 0))
    processor._prune_track_history(set())
    assert not processor._face_cache
    assert not processor._animal_cache


def test_cpu_vlm_fails_before_loading_or_releasing_models(monkeypatch):
    processor = module.AIProcessor(autoload=False)
    monkeypatch.setattr(module, "torch", object())
    processor.model_type = "sam3"
    processor.segment_model = object()
    previous = processor.segment_model
    assert "CUDA" in processor.describe_scene(None)
    assert processor.segment_model is previous


def test_tensorrt_segmentation_task_is_explicit(monkeypatch):
    processor = module.AIProcessor(autoload=False)
    processor.device = "cuda"
    loaded = []
    monkeypatch.setattr(module, "YOLO", lambda name, *, task: loaded.append((name, task)) or object())
    processor.load_segmentation_model("yolo26n-seg.engine", "Segmentation")
    assert loaded == [("yolo26n-seg.engine", "segment")]
