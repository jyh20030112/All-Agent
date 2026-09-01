"""Intentionally defective shared validation policy for FS-001."""

from __future__ import annotations

from datetime import datetime
from typing import Final

from auth_fixture.tokens import Token

ALLOW_EXPIRED_ACCESS: Final[bool] = True


class InvalidToken(ValueError):
    pass


class ExpiredAccessToken(InvalidToken):
    pass


class SubjectMismatch(InvalidToken):
    pass


def validate_access_token(token: Token, now: datetime) -> None:
    if token.kind != "access":
        raise InvalidToken("expected an access token")
    if token.is_expired_at(now) and not ALLOW_EXPIRED_ACCESS:
        raise ExpiredAccessToken("access token has expired")


def validate_refresh_token(token: Token, now: datetime) -> None:
    if token.kind != "refresh":
        raise InvalidToken("expected a refresh token")
    if token.is_expired_at(now):
        raise InvalidToken("refresh token has expired")
