"""Desktop boundaries: bounded background commands and private client settings."""

from __future__ import annotations

import copy
import json
import os
import queue
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from websocket import create_connection

from config import PiSettings, StreamSettings, cfg, validate_settings


@dataclass
class Completion:
    label: str
    value: Any = None
    error: str = ""


class CommandWorker:
    """One daemon worker, one command at a time, no GUI objects in the worker.

    Poll completions on the GUI thread. Reject excess commands rather than queueing
    obsolete sliders/model changes behind a potentially long GPU operation.
    """

    def __init__(self):
        self.results: queue.Queue[Completion] = queue.Queue()
        self._commands: queue.Queue = queue.Queue(maxsize=1)
        self._busy = False
        self._closed = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True, name="desktop-commands")
        self._thread.start()

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def submit(self, label: str, action: Callable) -> bool:
        with self._lock:
            if self._busy or self._closed:
                return False
            self._busy = True
        self._commands.put_nowait((label, action))
        return True

    def _run(self):
        while True:
            command = self._commands.get()
            if command is None:
                return
            label, action = command
            try:
                completed = Completion(label, value=action())
            except Exception as exc:
                completed = Completion(label, error=str(exc))
            self.results.put(completed)
            if self._closed:
                return

    def acknowledge(self):
        """The UI has consumed the result; a new command can now be accepted."""
        with self._lock:
            self._busy = False

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._commands.put_nowait(None)
        except queue.Full:
            pass  # The pending command will observe _closed after finishing.


def camera_settings(updates: dict | None = None) -> dict:
    """Read or atomically update the camera, requiring an authenticated ACK."""
    connection = create_connection(cfg.pi.ws_url, timeout=5)
    try:
        initial = json.loads(connection.recv())
        if initial.get("type") != "current_settings":
            raise ValueError("Réponse initiale de la caméra invalide")
        connection.send(json.dumps({
            "type": "update_settings", "data": updates or {}, "token": cfg.pi.auth_token,
        }))
        reply = json.loads(connection.recv())
        if reply.get("type") != "current_settings":
            raise ValueError(reply.get("message", "Commande refusée par la caméra"))
        values = reply["data"]
        if any(values.get(key) != value for key, value in (updates or {}).items()):
            raise ValueError("La caméra n’a pas appliqué tous les réglages")
        return values
    finally:
        connection.close()


def restart_camera() -> dict:
    """Explicit encoder recovery using the deployed protocol; restore in finally."""
    initial = camera_settings()
    try:
        camera_settings({"fps": 15 if initial["fps"] != 15 else 20})
        time.sleep(2)
    finally:
        restored = camera_settings(initial)
        if restored != initial:
            raise RuntimeError("Les réglages initiaux n’ont pas pu être rétablis")
    return restored


def save_connection(host: str, video_port: int, ws_port: int, token: str,
                    open_timeout: int, read_timeout: int, *, path: Path | None = None) -> str:
    """Validate, atomically persist, then publish settings; preserve other sections."""
    host = host.strip()
    if not re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9.-]{0,251}[a-zA-Z0-9])?", host):
        raise ValueError("Indique une adresse IPv4 ou un nom d’hôte, sans http:// ni port")
    if video_port == ws_port:
        raise ValueError("Les ports vidéo et commandes doivent être différents")
    candidate = copy.deepcopy(cfg)
    candidate.pi = PiSettings(**{**vars(cfg.pi), "host": host, "video_port": video_port,
                                 "ws_port": ws_port, "auth_token": token})
    candidate.stream = StreamSettings(**{**vars(cfg.stream),
                                         "capture_open_timeout_ms": open_timeout,
                                         "capture_read_timeout_ms": read_timeout})
    validate_settings(candidate)
    root = Path(__file__).resolve().parent
    path = path or root / "config.yaml"
    source = path if path.exists() else root / "config.example.yaml"
    data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    data.setdefault("pi", {}).update({key: getattr(candidate.pi, key)
                                     for key in ("host", "video_port", "ws_port", "auth_token")})
    data.setdefault("stream", {}).update({key: getattr(candidate.stream, key)
                                         for key in ("capture_open_timeout_ms", "capture_read_timeout_ms")})
    fd, temporary = tempfile.mkstemp(prefix=".connection-", suffix=".yaml", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            yaml.safe_dump(data, stream, allow_unicode=True, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    cfg.pi = candidate.pi
    cfg.stream = candidate.stream
    return f"Connexion enregistrée : {host}"
