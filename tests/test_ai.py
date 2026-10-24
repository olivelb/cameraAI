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


def test_vlm_parser_receives_prompt_prefix_and_returns_only_answer(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    class Batch(dict):
        def to(self, device):
            return self

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            return Batch(input_ids=np.array([[101, 102, 103]]))

        def decode(self, tokens, **kwargs):
            return "<assistant><think>" if len(tokens) == 3 else "generated response"

        def parse_response(self, response, *, prefix):
            assert prefix == "<assistant><think>"
            assert response == "generated response"
            return {"role": "assistant", "thinking": "internal", "content": [{"type": "text", "text": "A room."}]}

    processor = module.AIProcessor(autoload=False)
    processor.device = "cuda"
    processor._vlm_processor = Processor()
    processor._vlm_model = SimpleNamespace(device="cuda", generate=lambda **kw: np.array([[101, 102, 103, 104]]))
    monkeypatch.setattr(module, "torch", SimpleNamespace(inference_mode=nullcontext))
    assert processor.describe_scene(None, "Describe") == "A room."


def test_failed_learning_capture_does_not_claim_a_sample(monkeypatch):
    from types import SimpleNamespace

    processor = module.AIProcessor(autoload=False)
    processor.learning_count = processor.learning_max = 1
    recognizer = SimpleNamespace(capture_face=lambda *args: "❌ No face")
    monkeypatch.setattr(processor, "_get_face_recognizer", lambda: recognizer)
    def tensor(array):
        return SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.array(array)))
    result = SimpleNamespace(boxes=SimpleNamespace(xyxy=tensor([[0, 0, 10, 10]]),
                                                  cls=tensor([module.cfg.ai.person_class])), masks=None)
    processor._run_learning_logic([result], np.zeros((10, 10, 3), np.uint8))
    assert processor.learning_count == 1


def test_assistant_reclaims_inactive_cyclic_models_before_allocating_weights(monkeypatch):
    import sys
    import weakref
    from types import SimpleNamespace

    class CachedModel:
        pass

    processor = module.AIProcessor(autoload=False)
    processor.device = "cuda"
    processor.segment_model_name = "active.engine"
    processor.segment_model = object()
    inactive = CachedModel()
    inactive.cycle = inactive
    released = weakref.ref(inactive)
    processor.loaded_models = {"active.engine": processor.segment_model, "sam3.pt": inactive}
    del inactive

    def allocate(**kwargs):
        assert released() is None
        assert processor.segment_model is processor.loaded_models["active.engine"]
        raise RuntimeError("Fixture stops before weight loading")

    monkeypatch.setattr(module, "torch", SimpleNamespace(bfloat16="bf16", cuda=SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None)))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForMultimodalLM=object(), AutoProcessor=object(), BitsAndBytesConfig=allocate))
    assert "Fixture stops before weight loading" in processor.describe_scene(None)
