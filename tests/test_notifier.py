import ssl

import pytest

import notifier


def test_smtp_validates_server_certificate_before_login(monkeypatch):
    events = []

    class SMTP:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def starttls(self, *, context):
            assert context.verify_mode == ssl.CERT_REQUIRED
            assert context.check_hostname
            events.append("tls")

        def login(self, *args):
            events.append("login")

        def send_message(self, message):
            assert message["To"] == "receiver@example.test"
            events.append("send")

    monkeypatch.setattr(notifier.smtplib, "SMTP", SMTP)
    notifier.EmailNotifier().send_email_sync(
        "sender@example.test", "test-only", "receiver@example.test", "Subject", "Body"
    )
    assert events == ["tls", "login", "send"]


def test_smtp_failure_reaches_sync_caller(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("SMTP unavailable")

    monkeypatch.setattr(notifier.smtplib, "SMTP", fail)
    with pytest.raises(OSError, match="SMTP unavailable"):
        notifier.EmailNotifier().send_email_sync("sender", "test-only", "receiver", "Subject", "Body")
