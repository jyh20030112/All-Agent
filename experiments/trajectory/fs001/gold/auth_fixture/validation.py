"""Gold path-sensitive validation policy for FS-001."""

from __future__ import annotations

from datetime import datetime

from auth_fixture.tokens import Token


class InvalidToken(ValueError):
    pass


class ExpiredAccessToken(InvalidToken):
    pass


class SubjectMismatch(InvalidToken):
    pass


def validate_access_token(
    token: Token,
    now: datetime,
    *,
    allow_expired: bool = False,
) -> None:
    if token.kind != "access":
        raise InvalidToken("expected an access token")
    if token.is_expired_at(now) and not allow_expired:
        raise ExpiredAccessToken("access token has expired")


def validate_refresh_token(token: Token, now: datetime) -> None:
    if token.kind != "refresh":
        raise InvalidToken("expected a refresh token")
    if token.is_expired_at(now):
        raise InvalidToken("refresh token has expired")
