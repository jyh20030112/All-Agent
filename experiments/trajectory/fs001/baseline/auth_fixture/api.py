"""Stable public behavior for trajectory experiment FS-001."""

from __future__ import annotations

from datetime import datetime

from auth_fixture.tokens import FROZEN_NOW, Token, issue_access_token
from auth_fixture.validation import (
    SubjectMismatch,
    validate_access_token,
    validate_refresh_token,
)


def access_protected_resource(
    access_token: Token,
    now: datetime = FROZEN_NOW,
) -> dict[str, str]:
    validate_access_token(access_token, now)
    return {"resource": "protected", "subject": access_token.subject}


def refresh_session(
    access_token: Token,
    refresh_token: Token,
    now: datetime = FROZEN_NOW,
) -> Token:
    validate_access_token(access_token, now)
    validate_refresh_token(refresh_token, now)
    if access_token.subject != refresh_token.subject:
        raise SubjectMismatch("access and refresh token subjects differ")
    return issue_access_token(access_token.subject, now)
