"""
notifier.py — Email notification system for camera alerts.

Sends emails with optional image attachments via Gmail SMTP.
All sending is done in background threads to avoid blocking the UI.
"""

from __future__ import annotations

import logging
import smtplib
import threading
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config import cfg

log = logging.getLogger(__name__)


class EmailNotifier:
    """Threaded email sender for camera alert notifications."""

    def __init__(self) -> None:
        self.smtp_server: str = cfg.email.smtp_server
        self.smtp_port: int = cfg.email.smtp_port
        self.timeout: int = cfg.email.timeout_seconds

    def send_email(
        self,
        sender_email: str,
        sender_password: str,
        receiver_email: str,
        subject: str,
        body: str,
        image_data: bytes | None = None,
        image_name: str = "alert.jpg",
    ) -> None:
        """Send an email in a background thread (fire-and-forget).

        Errors are logged but not raised. Use :meth:`send_email_sync`
        when delivery failures must reach the caller.
        """
        thread = threading.Thread(
            target=self._send_email_logged,
            args=(
                sender_email, sender_password, receiver_email,
                subject, body, image_data, image_name,
            ),
            daemon=True,
        )
        thread.start()

    def _send_email_logged(self, *args, **kwargs) -> None:
        """Thread wrapper: send and log failures instead of raising."""
        try:
            self.send_email_sync(*args, **kwargs)
        except Exception as e:
            log.error("Failed to send email: %s", e)

    def send_email_sync(
        self,
        sender_email: str,
        sender_password: str,
        receiver_email: str,
        subject: str,
        body: str,
        image_data: bytes | None = None,
        image_name: str = "alert.jpg",
    ) -> None:
        """Compose and send the email inline; raise on failure."""
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        if image_data:
            image = MIMEImage(image_data, name=image_name)
            msg.attach(image)

        with smtplib.SMTP(
            self.smtp_server, self.smtp_port, timeout=self.timeout
        ) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)

        log.info("Email sent successfully to %s", receiver_email)
