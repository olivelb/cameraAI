"""JSON CLI for the running native app. Discover everything with schema.get."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError

from control_api import ControlClient, DEFINITIONS, STATE_FILE, manifest, validate_params


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("Invalid command line; use --help and schema.get")


def main(argv=None):
    for stream in (sys.stdin, sys.stdout):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = Parser(description=__doc__)
    parser.add_argument("command", choices=sorted(DEFINITIONS))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--json", dest="params", help="JSON parameters; use --stdin or --file for secrets")
    source.add_argument("--stdin", action="store_true", help="Read one JSON object from stdin")
    source.add_argument("--file", type=Path, help="Read parameters from a local private JSON file")
    parser.add_argument("--state-file", type=Path, default=STATE_FILE)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--no-wait", action="store_true", help="Return an asynchronous job ID")
    try:
        args = parser.parse_args(argv)
        if not math.isfinite(args.timeout) or not .1 <= args.timeout <= 3600:
            raise ValueError("Timeout must be in [.1, 3600] seconds")
        raw = sys.stdin.read(65537) if args.stdin else args.file.read_text(encoding="utf-8") if args.file else args.params or "{}"
        if len(raw.encode("utf-8")) > 65536:
            raise ValueError("Parameters exceed 64 KiB")
        params = json.loads(raw)
        validate_params(args.command, params)
        if args.params and any(key in params for key in ("password", "token")):
            raise ValueError("Supply passwords and tokens with --stdin or --file, never command arguments")
        if args.command == "schema.get":
            result = {"ok": True, "result": manifest()}
        elif args.command == "app.start":
            try:
                ControlClient(args.state_file).get("/status")
            except (OSError, ValueError, URLError):
                command = [sys.executable, str(Path(__file__).with_name("desktop.py")), "--control-file", str(args.state_file)]
                if params.get("no_camera"):
                    command.append("--no-camera")
                if params.get("data_dir"):
                    command.extend(["--data-dir", params["data_dir"]])
                with Path(__file__).with_name("cli-launch.log").open("ab") as log:
                    child = subprocess.Popen(command, cwd=Path(__file__).resolve().parent, stdout=log, stderr=log,
                                             creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
                deadline = time.monotonic() + args.timeout
                while time.monotonic() < deadline:
                    try:
                        ControlClient(args.state_file).get("/status")
                        break
                    except (OSError, ValueError, URLError):
                        if child.poll() is not None:
                            raise ValueError("Application exited; see local cli-launch.log") from None
                        time.sleep(.1)
                else:
                    raise ValueError("Startup timeout; the application may still be starting")
            else:
                if params.get("data_dir"):
                    raise ValueError("An application already uses this control file; choose a separate --state-file for an isolated data directory")
            result = {"ok": True, "result": {"running": True}}
        else:
            result = ControlClient(args.state_file).call(args.command, params, wait=not args.no_wait, timeout=args.timeout)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0 if result.get("ok") else 1
    except (OSError, ValueError, TypeError, URLError):
        # Never print authentication headers, addresses, stderr, or credential values.
        exc = sys.exception()
        message = str(exc) if isinstance(exc, ValueError) and not isinstance(exc, json.JSONDecodeError) else "Application unavailable or invalid JSON; use app.start and schema.get"
        print(json.dumps({"ok": False, "error": {"code": "CLI_ERROR", "message": message}}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
