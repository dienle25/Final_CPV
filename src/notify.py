"""Optional Gmail SMTP notification.

Why SMTP instead of Gmail API/Stringee:
SMTP demonstrates the notification layer with far less setup. It is deliberately
kept outside the critical detection path: a network or credential failure must
never stop video inference or corrupt the local violation database.
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


class EmailNotifier:
    def __init__(self, enabled: bool | None = None) -> None:
        load_dotenv()
        env_enabled = os.getenv("EMAIL_ENABLED", "false").lower() in {"1", "true", "yes"}
        self.enabled = env_enabled if enabled is None else enabled
        self.host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.port = int(os.getenv("SMTP_PORT", "587"))
        self.username = os.getenv("SMTP_USERNAME", "")
        self.password = os.getenv("SMTP_APP_PASSWORD", "")
        self.recipient = os.getenv("ALERT_EMAIL_TO", "")

    def configured(self) -> bool:
        return bool(self.enabled and self.username and self.password and self.recipient)

    def send_violation(self, event: dict[str, Any]) -> bool:
        if not self.configured():
            print("[notify] Email disabled or SMTP variables are incomplete.")
            return False

        message = EmailMessage()
        plate = event.get("plate_text") or "UNREAD"
        message["Subject"] = f"[Helmet MVP] Violation detected - {plate}"
        message["From"] = self.username
        message["To"] = self.recipient
        message.set_content(
            "\n".join(
                [
                    "Helmet violation confirmed by temporal tracking.",
                    f"Violation ID: {event.get('id')}",
                    f"Track ID: {event.get('track_id')}",
                    f"Time (UTC): {event.get('detected_at_utc')}",
                    f"Video time: {event.get('video_time_s')} s",
                    f"Detection confidence: {event.get('confidence')}",
                    f"License plate: {plate}",
                    "",
                    "This is a classroom MVP; the result requires human review.",
                ]
            )
        )

        image_path = Path(str(event.get("image_path", "")))
        if image_path.exists():
            mime, _ = mimetypes.guess_type(image_path.name)
            major, minor = (mime or "image/jpeg").split("/", 1)
            message.add_attachment(
                image_path.read_bytes(),
                maintype=major,
                subtype=minor,
                filename=image_path.name,
            )

        try:
            with smtplib.SMTP(self.host, self.port, timeout=15) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(message)
            print(f"[notify] Sent email for violation {event.get('id')}")
            return True
        except Exception as exc:
            # Notification is best-effort and must not crash the AI pipeline.
            print(f"[notify] Email failed: {type(exc).__name__}: {exc}")
            return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a test SMTP notification")
    parser.add_argument("--image", default="")
    args = parser.parse_args()
    event = {
        "id": "TEST",
        "track_id": 1,
        "detected_at_utc": "test",
        "video_time_s": 0,
        "confidence": 0.99,
        "plate_text": "TEST123",
        "image_path": args.image,
    }
    ok = EmailNotifier(enabled=True).send_violation(event)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
