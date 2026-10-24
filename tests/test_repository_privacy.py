"""Published sources use documentation addresses and fictional identities only."""

import ipaddress
import re
import subprocess
from pathlib import Path

import yaml

from config import PiSettings


def published_text_files():
    root = Path(__file__).resolve().parents[1]
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root,
    ).decode("utf-8").split("\0")
    for name in set(names):
        path = root / name
        if not name or not path.is_file():
            continue
        try:
            yield name, path.read_text(encoding="utf-8")
        except UnicodeError:
            continue


def test_no_private_network_addresses_in_published_files():
    examples = [ipaddress.ip_network(block) for block in (
        "192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24",
    )]
    violations = []
    for name, text in published_text_files():
        for match in re.finditer(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])", text):
            try:
                address = ipaddress.ip_address(match.group())
            except ValueError:
                continue
            if (not address.is_global and not address.is_loopback and not address.is_unspecified
                    and not any(address in block for block in examples)):
                violations.append(name)
    assert not violations, f"Private addresses in: {sorted(set(violations))}"


def test_no_personal_email_or_user_home_paths_in_published_files():
    violations = []
    email = re.compile(r"[\w.!#$%&'*+/=?^`{|}~-]+@([\w-]+(?:\.[\w-]+)+)")
    home = re.compile(r"/home/[A-Za-z0-9_.-]+|[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/][^/\\\s]+")
    tailnet = re.compile(r"\b[a-zA-Z0-9][a-zA-Z0-9.-]*\.ts\.net\b")
    for name, text in published_text_files():
        domains = [match.group(1).lower() for match in email.finditer(text)]
        if any(not domain.endswith((".test", ".invalid", ".example", "example.com", "example.org", "example.net"))
               for domain in domains) or home.search(text) or tailnet.search(text):
            violations.append(name)
    assert not violations, f"Personal email or home paths in: {sorted(set(violations))}"


def test_camera_defaults_are_explicit_placeholders():
    example = Path(__file__).resolve().parents[1] / "config.example.yaml"
    assert PiSettings().host.endswith(".invalid")
    assert yaml.safe_load(example.read_text(encoding="utf-8"))["pi"]["host"] == PiSettings().host


def test_private_settings_and_diagnostics_are_not_published():
    forbidden = {"config.yaml", "email_config.json", ".secret.key", ".camera-control.json"}
    violations = [name for name, _ in published_text_files()
                  if Path(name).name in forbidden or name.startswith(".audit-tmp/")
                  or Path(name).name.startswith((".connection-", ".profile-", ".email-", ".control-"))]
    assert not violations, f"Private files are published: {violations}"
