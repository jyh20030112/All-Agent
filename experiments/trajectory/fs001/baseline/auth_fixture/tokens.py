"""Deterministic token values for trajectory experiment FS-001."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

FROZEN_NOW = datetime(2030, 1, 1, 12, 0, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class Token:
    kind: str
    subject: str
    issued_at: datetime
    expires_at: datetime

    @property
    def expired(self) -> bool:
        return self.expires_at <= FROZEN_NOW

    def is_expired_at(self, now: datetime) -> bool:
        return self.expires_at <= now

    def to_payload(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "subject": self.subject,
            "issued_at": self.issued_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
        }


def expired_access_token(subject: str = "user-123") -> Token:
    return Token(
        kind="access",
        subject=subject,
        issued_at=FROZEN_NOW - timedelta(hours=2),
        expires_at=FROZEN_NOW - timedelta(hours=1),
    )


def valid_refresh_token(subject: str = "user-123") -> Token:
    return Token(
        kind="refresh",
        subject=subject,
        issued_at=FROZEN_NOW - timedelta(days=1),
        expires_at=FROZEN_NOW + timedelta(days=6),
    )


def issue_access_token(subject: str, now: datetime) -> Token:
    return Token(
        kind="access",
        subject=subject,
        issued_at=now,
        expires_at=now + timedelta(hours=1),
    )
