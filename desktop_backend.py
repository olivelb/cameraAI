"""Desktop boundaries: bounded background commands and private client settings."""

from __future__ import annotations

import copy
import json
import os
import queue
import ipaddress
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from websocket import create_connection

from config import (
    NETWORK_PROFILE_IDS, PiSettings, StreamSettings, cfg,
    validate_connection_profile, validate_settings,
)


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
                    open_timeout: int, read_timeout: int, *, path: Path | None = None,
                    profile_id: str = "current") -> str:
    """Validate, atomically persist, then publish settings; preserve other sections."""
    host = host.strip()
    profile = validate_connection_profile({
        "host": host, "video_port": video_port, "ws_port": ws_port,
        "open_timeout_ms": open_timeout, "read_timeout_ms": read_timeout,
    })
    if profile_id not in NETWORK_PROFILE_IDS:
        raise ValueError("Unknown network profile")
    candidate = copy.deepcopy(cfg)
    candidate.pi = PiSettings(**{**vars(cfg.pi), "host": host, "video_port": video_port,
                                 "ws_port": ws_port, "auth_token": token})
    candidate.stream = StreamSettings(**{**vars(cfg.stream),
                                         "capture_open_timeout_ms": open_timeout,
                                         "capture_read_timeout_ms": read_timeout})
    candidate.desktop.active_profile = profile_id
    if profile_id != "current":
        candidate.desktop.profiles[profile_id] = asdict(profile)
    validate_settings(candidate)
    root = Path(__file__).resolve().parent
    path = path or root / "config.yaml"
    source = path if path.exists() else root / "config.example.yaml"
    data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    data.setdefault("pi", {}).update({key: getattr(candidate.pi, key)
                                     for key in ("host", "video_port", "ws_port", "auth_token")})
    data.setdefault("stream", {}).update({key: getattr(candidate.stream, key)
                                         for key in ("capture_open_timeout_ms", "capture_read_timeout_ms")})
    data["desktop"] = asdict(candidate.desktop)
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
    cfg.desktop = candidate.desktop
    return "Connexion et profil enregistrés localement"


def tailscale_devices() -> list[dict]:
    """Read local CLI state. Never log peer data, stderr, or authentication URLs."""
    candidates = [shutil.which("tailscale")]
    for variable in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        base = os.environ.get(variable)
        if base:
            candidates.append(str(Path(base) / "Tailscale" / "tailscale.exe"))
    executable = next((path for path in candidates if path and Path(path).is_file()), None)
    if not executable:
        raise ValueError("Client Tailscale introuvable. Installe-le ou saisis l’adresse manuellement.")
    try:
        process = subprocess.run(
            [executable, "status", "--json"], capture_output=True, text=True, encoding="utf-8",
            timeout=10, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise ValueError("Impossible de lire Tailscale. Vérifie que le client est démarré.") from None
    if process.returncode:
        raise ValueError("Tailscale ne répond pas. Vérifie le client et sa connexion.")
    try:
        status = json.loads(process.stdout)
        if not isinstance(status, dict) or status.get("BackendState") != "Running":
            raise ValueError("Tailscale doit être connecté avant de rechercher les appareils.")
        peers = status.get("Peer") or {}
        if not isinstance(peers, dict):
            raise ValueError("Réponse Tailscale invalide")
        devices = []
        for peer in peers.values():
            if not isinstance(peer, dict):
                continue
            name = peer.get("HostName") or peer.get("DNSName")
            dns = str(peer.get("DNSName") or "").rstrip(".")
            addresses = peer.get("TailscaleIPs") or []
            ipv4 = ""
            for address in addresses if isinstance(addresses, list) else []:
                try:
                    if ipaddress.ip_address(address).version == 4:
                        ipv4 = address
                        break
                except (ValueError, TypeError):
                    continue
            host = ipv4 or dns
            if name and host:
                devices.append({"name": str(name), "host": host, "online": bool(peer.get("Online"))})
    except (json.JSONDecodeError, TypeError):
        raise ValueError("Réponse Tailscale invalide") from None
    return sorted(devices, key=lambda item: (not item["online"], item["name"].casefold()))
