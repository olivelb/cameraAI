"""Authenticated loopback control and a discoverable JSON command contract.

No Qt, AI, or user configuration is imported by the CLI transport.
"""

from __future__ import annotations

import copy
import hmac
import json
import os
import re
import secrets
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import error, request
from uuid import uuid4

STATE_FILE = Path(__file__).resolve().parent / ".camera-control.json"
VERSION = 1


def field(kind, *, required=False, **options):
    return {"type": kind, "_required": required, **options}


BOOL = field("boolean")
TEXT = field("string", maxLength=4096)
KIND = field("string", required=True, enum=["faces", "animals"])
PROFILE = field("string", enum=["current", "local", "vpn", "tailscale"])
DEFINITIONS = {
    "schema.get": ("List commands and JSON argument schemas; works offline.", {}),
    "status.get": ("Read live state without exposing credentials or network addresses.", {}),
    "jobs.list": ("List recent command states, excluding their input parameters.", {}),
    "job.get": ("Read a command result by job ID.", {"id": field("string", required=True)}),
    "app.start": ("Start the desktop app with this Python environment.", {"no_camera": BOOL, "data_dir": TEXT}),
    "app.quit": ("Close the desktop app after pending work and camera release.", {}),
    "camera.get": ("Read authenticated camera settings.", {}),
    "camera.set": ("Apply video settings; reconnects automatically.", {
        "width": field("integer", minimum=1, maximum=1920),
        "height": field("integer", minimum=1, maximum=1080),
        "fps": field("integer", minimum=1, maximum=30),
        "bitrate": field("integer", minimum=100000, maximum=25000000),
    }),
    "camera.reconnect": ("Reconnect the local reader, without changing the camera.", {}),
    "camera.restart": ("Explicitly restart the encoder and restore its video settings.", {}),
    "network.get": ("List local profiles; addresses require private=true, tokens are never returned.", {"private": BOOL}),
    "network.use": ("Activate a previously saved local profile.", {"profile": {**PROFILE, "_required": True}}),
    "network.configure": ("Save a connection locally. Supply secrets through stdin or a private JSON file.", {
        "host": {**TEXT, "_required": True}, "profile": PROFILE,
        "video_port": field("integer", minimum=1, maximum=65535),
        "ws_port": field("integer", minimum=1, maximum=65535),
        "open_timeout_ms": field("integer", minimum=1000, maximum=60000),
        "read_timeout_ms": field("integer", minimum=1000, maximum=60000), "token": TEXT,
    }),
    "network.tailscale": ("List Tailscale devices locally; output is private.", {}),
    "ai.set": ("Enable or disable inference; capture remains running.", {"enabled": {**BOOL, "_required": True}}),
    "detection.set": ("Set detection confidence and optional class IDs.", {
        "confidence": field("number", minimum=.1, maximum=1),
        "classes": field("array", items={"type": "integer", "minimum": 0}, maxItems=100),
    }),
    "model.get": ("Read current model state and installed model filenames.", {}),
    "model.load": ("Load detection, segmentation, SAM2, SAM3, or an installed TensorRT engine.", {
        "model": {**TEXT, "_required": True},
        "task": field("string", enum=["Detection", "Segmentation"]), "concepts": TEXT,
    }),
    "model.export": ("Export the active supported PyTorch model to TensorRT.", {}),
    "sam.concepts": ("Set comma-separated SAM3 concepts.", {"concepts": {**TEXT, "_required": True}}),
    "tracking.set": ("Apply tracking immediately and report effective support.", {
        "enabled": {**BOOL, "_required": True}, "tracker": field("string", enum=["BoT-SORT", "ByteTrack"]),
    }),
    "recognition.set": ("Enable face/animal recognition and set similarity threshold.", {
        "kind": KIND, "enabled": {**BOOL, "_required": True},
        "threshold": field("number", minimum=.01, maximum=1),
    }),
    "gallery.status": ("Inspect a gallery without downloading models.", {"kind": KIND}),
    "gallery.import": ("Import photos into a named gallery entry.", {
        "kind": KIND, "name": {**TEXT, "_required": True},
        "files": field("array", required=True, items={"type": "string"}, minItems=1, maxItems=100),
    }),
    "gallery.rebuild": ("Rebuild embeddings from the gallery photos.", {"kind": KIND}),
    "gallery.learn": ("Start live learning; progress is reported by status.get.", {
        "kind": KIND, "name": {**TEXT, "_required": True}, "count": field("integer", minimum=1, maximum=100),
    }),
    "gallery.stop": ("Stop pending live learning without deleting captured photos.", {"kind": KIND}),
    "alerts.get": ("Read enabled/configured flags; never returns email addresses or passwords.", {}),
    "alerts.configure": ("Save encrypted SMTP credentials locally, without sending email.", {
        "enabled": BOOL, "sender": {**TEXT, "_required": True},
        "receiver": {**TEXT, "_required": True}, "password": TEXT,
    }),
    "alerts.enable": ("Enable or disable automatic detection emails.", {"enabled": {**BOOL, "_required": True}}),
    "alerts.test": ("Validate email settings without sending; send=true explicitly sends a real email.", {"send": BOOL}),
    "assistant.ask": ("Ask the VLM about the current image; preserves successful conversation turns.", {
        "message": {**TEXT, "_required": True, "minLength": 1},
    }),
    "assistant.history": ("Read the local assistant conversation; output is private.", {}),
    "assistant.clear": ("Clear only the in-memory conversation.", {}),
    "frame.save": ("Save a PNG frame; refuses to overwrite unless explicitly requested.", {
        "path": {**TEXT, "_required": True}, "annotated": BOOL, "overwrite": BOOL,
    }),
    "view.get": ("Read current preview options.", {}),
    "view.set": ("Set preview pause, annotations, or fullscreen while inference continues.", {
        "paused": BOOL, "overlay": BOOL, "fullscreen": BOOL,
    }),
    "logs.get": ("Read recent logs with network details and credentials redacted by default.", {
        "limit": field("integer", minimum=1, maximum=200), "private": BOOL,
    }),
}


def manifest():
    commands = {}
    for name, (description, fields) in DEFINITIONS.items():
        properties = {key: {k: v for k, v in value.items() if k != "_required"}
                      for key, value in fields.items()}
        commands[name] = {"description": description, "parameters": {
            "type": "object", "properties": properties,
            "required": [key for key, value in fields.items() if value.get("_required")],
            "additionalProperties": False,
        }}
    return {"version": VERSION, "commands": commands}


def validate_params(name, params):
    if name not in DEFINITIONS:
        raise ValueError("Unknown command; use schema.get")
    if not isinstance(params, dict):
        raise ValueError("Parameters must be a JSON object")
    fields = DEFINITIONS[name][1]
    if set(params) - fields.keys():
        raise ValueError("Unknown parameter")

    def check(value, schema):
        kind = schema["type"]
        valid = {"string": type(value) is str, "boolean": type(value) is bool,
                 "integer": type(value) is int, "number": type(value) in (int, float),
                 "array": type(value) is list}.get(kind, False)
        if not valid:
            raise ValueError(f"Expected {kind}")
        if "enum" in schema and value not in schema["enum"]:
            raise ValueError("Value is not in the allowed choices")
        if kind in ("integer", "number"):
            import math
            if not math.isfinite(value) or value < schema.get("minimum", value) or value > schema.get("maximum", value):
                raise ValueError("Number outside allowed range")
        if kind == "string" and not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", 4096):
            raise ValueError("Invalid text length")
        if kind == "array":
            if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 100):
                raise ValueError("Invalid list length")
            for item in value:
                check(item, schema["items"])

    for key, schema in fields.items():
        if schema.get("_required") and key not in params:
            raise ValueError(f"Missing parameter: {key}")
        if key in params:
            try:
                check(params[key], schema)
            except ValueError as exc:
                raise ValueError(f"{key}: {exc}") from None
    return params


class CommandBus:
    """Bounded jobs. Inputs remain private in memory and are never returned."""

    def __init__(self, status_reader):
        self.status_reader = status_reader
        self._lock = threading.Lock()
        self._jobs = OrderedDict()
        self._pending = []
        self.closed = False

    def submit(self, name, params):
        validate_params(name, params)
        with self._lock:
            if self.closed:
                raise ValueError("Application is closing")
            if len(self._pending) >= 32:
                raise ValueError("Command queue is full")
            while len(self._jobs) >= 128:
                terminal = next((key for key, job in self._jobs.items() if job["status"] in ("succeeded", "failed")), None)
                if terminal is None:
                    raise ValueError("Too many active jobs")
                del self._jobs[terminal]
            job_id = uuid4().hex
            self._jobs[job_id] = {"id": job_id, "command": name, "status": "queued",
                                  "result": None, "error": None, "_params": copy.deepcopy(params)}
            self._pending.append(job_id)
            return self.get_unlocked(job_id)

    def take(self, *, ui_only=False):
        with self._lock:
            for job_id in self._pending:
                job = self._jobs[job_id]
                if ui_only and job["command"] not in ("view.get", "view.set", "app.quit"):
                    continue
                self._pending.remove(job_id)
                job["status"] = "running"
                return job_id, job["command"], copy.deepcopy(job["_params"])
        return None

    def finish(self, job_id, *, result=None, error_message=None):
        with self._lock:
            job = self._jobs[job_id]
            job.update(status="failed" if error_message else "succeeded", result=result,
                       error={"code": "COMMAND_FAILED", "message": error_message} if error_message else None)
            job.pop("_params", None)

    def get_unlocked(self, job_id):
        return copy.deepcopy({key: value for key, value in self._jobs[job_id].items() if not key.startswith("_")})

    def get(self, job_id):
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError("Unknown or expired job")
            return self.get_unlocked(job_id)

    def jobs(self):
        with self._lock:
            return [{"id": job["id"], "command": job["command"], "status": job["status"]}
                    for job in self._jobs.values()]

    def close(self):
        with self._lock:
            self.closed = True
            for job_id in self._pending:
                self._jobs[job_id].update(status="failed", error={"code": "APP_CLOSED", "message": "Application closed"})
                self._jobs[job_id].pop("_params", None)
            self._pending.clear()


class ControlServer:
    def __init__(self, bus, path=STATE_FILE):
        self.bus = bus
        self.path = Path(path)
        self.token = secrets.token_urlsafe(32)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, payload):
                data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def authorized(self):
                self.connection.settimeout(5)
                supplied = self.headers.get("Authorization", "")
                return hmac.compare_digest(supplied.encode(), ("Bearer " + outer.token).encode())

            def do_GET(self):
                if not self.authorized():
                    self.reply(401, {"ok": False, "error": {"code": "UNAUTHORIZED", "message": "Local control token required"}})
                    return
                try:
                    if self.path == "/schema":
                        value = manifest()
                    elif self.path == "/status":
                        value = outer.bus.status_reader()
                    elif self.path == "/jobs":
                        value = outer.bus.jobs()
                    elif self.path.startswith("/jobs/"):
                        value = outer.bus.get(self.path.removeprefix("/jobs/"))
                    else:
                        raise KeyError("Unknown endpoint")
                    self.reply(200, {"ok": True, "result": value})
                except KeyError:
                    self.reply(404, {"ok": False, "error": {"code": "NOT_FOUND", "message": "Unknown endpoint or job"}})

            def do_POST(self):
                if not self.authorized():
                    self.reply(401, {"ok": False, "error": {"code": "UNAUTHORIZED", "message": "Local control token required"}})
                    return
                if self.path != "/commands":
                    self.reply(404, {"ok": False, "error": {"code": "NOT_FOUND", "message": "Unknown endpoint"}})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 1 <= length <= 65536:
                        raise ValueError("Request size must be in [1, 65536]")
                    data = json.loads(self.rfile.read(length))
                    if not isinstance(data, dict) or set(data) != {"command", "params"}:
                        raise ValueError("Expected command and params")
                    job = outer.bus.submit(data["command"], data["params"])
                    self.reply(202, {"ok": True, "result": job})
                except (ValueError, TypeError, UnicodeError):
                    self.reply(400, {"ok": False, "error": {"code": "INVALID_REQUEST", "message": "Invalid command or parameters; see schema.get"}})

        if self.path.exists():
            try:
                previous = ControlClient(self.path)
            except (OSError, ValueError):
                raise RuntimeError("Existing control file is not a valid descriptor; choose another path") from None
            try:
                previous.get("/status")
            except error.HTTPError:
                raise RuntimeError("Existing control endpoint rejected authentication; choose another path") from None
            except (OSError, error.URLError):
                self.path.unlink(missing_ok=True)
            else:
                raise RuntimeError("A local control server is already running")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        descriptor = {"version": VERSION, "host": "127.0.0.1", "port": self.server.server_port,
                      "token": self.token, "pid": os.getpid()}
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(descriptor, stream)
        except BaseException:
            self.server.server_close()
            raise
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=.05), daemon=True, name="local-control")
        self.thread.start()

    def close(self):
        self.bus.close()
        self.server.shutdown()
        self.server.server_close()
        try:
            descriptor = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(descriptor, dict) and descriptor.get("token") == self.token:
                self.path.unlink(missing_ok=True)
        except (OSError, ValueError):
            pass


class ControlClient:
    def __init__(self, path=STATE_FILE):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if (not isinstance(data, dict) or data.get("version") != VERSION or data.get("host") != "127.0.0.1"
                or type(data.get("port")) is not int or not 1 <= data["port"] <= 65535
                or not isinstance(data.get("token"), str) or len(data["token"]) < 24):
            raise ValueError("Invalid local control descriptor")
        self.base = f"http://127.0.0.1:{data['port']}"
        self.token = data["token"]
        self.opener = request.build_opener(request.ProxyHandler({}))

    def send(self, path, data=None):
        if not re.fullmatch(r"/(schema|status|commands|jobs(?:/[a-f0-9]{32})?)", path):
            raise ValueError("Invalid local control path")
        payload = json.dumps(data).encode() if data is not None else None
        message = request.Request(self.base + path, data=payload, headers={  # noqa: S310 - fixed loopback and validated path
            "Authorization": "Bearer " + self.token, "Content-Type": "application/json",
        })
        with self.opener.open(message, timeout=5) as response:  # noqa: S310 - descriptor restricts loopback
            return json.load(response)

    def get(self, path):
        return self.send(path)

    def call(self, name, params, *, wait=True, timeout=120):
        validate_params(name, params)
        direct = {"status.get": "/status", "jobs.list": "/jobs", "schema.get": "/schema"}
        if name in direct:
            return self.get(direct[name])
        if name == "job.get":
            return self.get("/jobs/" + params["id"])
        accepted = self.send("/commands", {"command": name, "params": params})
        job = accepted["result"]
        if name == "app.quit":
            return {"ok": True, "job_id": job["id"], "result": {"closing_requested": True}}
        if not wait:
            return {"ok": True, "job": job}
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.get("/jobs/" + job["id"])["result"]
            if job["status"] in ("succeeded", "failed"):
                return {"ok": job["status"] == "succeeded", "job_id": job["id"],
                        "result": job["result"], "error": job["error"]}
            time.sleep(.1)
        return {"ok": False, "job_id": job["id"], "error": {"code": "WAIT_TIMEOUT", "message": "Job still running; poll job.get, do not resubmit"}}
