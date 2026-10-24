"""Shared camera opening policy for the live client and hardware checks."""

from config import cfg

def open_pi_capture():
    """Open the configured raw H.264 camera with finite connection/read waits."""
    import cv2

    return cv2.VideoCapture(cfg.pi.video_url, cv2.CAP_FFMPEG, [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, cfg.stream.capture_open_timeout_ms,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, cfg.stream.capture_read_timeout_ms,
    ])
