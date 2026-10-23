import numpy as np

from face_recognizer import _decode_det, _preprocess_det


def detector_outputs(with_landmarks=True):
    sizes = [(640 // stride) ** 2 * 2 for stride in (8, 16, 32)]
    scores = [np.zeros((n, 1), np.float32) for n in sizes]
    boxes = [np.ones((n, 4), np.float32) for n in sizes]
    landmarks = [np.zeros((n, 10), np.float32) for n in sizes]
    # First anchor at grid column 3, row 2 -> (24, 16), without half-stride offset.
    index = (2 * 80 + 3) * 2
    scores[0][index, 0] = .9
    landmarks[0][index] = [1, 2] * 5
    return scores + boxes + (landmarks if with_landmarks else [])


def test_detector_coordinates_and_landmarks_use_original_image_scale():
    faces = _decode_det(detector_outputs(), scale_x=2., scale_y=4.)
    assert len(faces) == 1
    assert faces[0][:4] == (8, 2, 16, 6)
    np.testing.assert_allclose(faces[0][4], [[16, 8]] * 5)


def test_missing_landmarks_allow_bbox_alignment_fallback():
    assert _decode_det(detector_outputs(False), 1., 1.)[0][4] is None


def test_detector_receives_rgb_channels():
    frame = np.zeros((640, 640, 3), np.uint8)
    frame[:, :, 2] = 255
    blob, _, _ = _preprocess_det(frame)
    assert blob[0, 0, 0, 0] > 0
    assert blob[0, 2, 0, 0] < 0
