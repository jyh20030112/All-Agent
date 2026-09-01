"""PROTOTYPE authentication fixture for trajectory experiment FS-001."""

from auth_fixture.api import access_protected_resource, refresh_session
from auth_fixture.tokens import FROZEN_NOW, Token
from auth_fixture.validation import ExpiredAccessToken, InvalidToken, SubjectMismatch

__all__ = [
    "ExpiredAccessToken",
    "FROZEN_NOW",
    "InvalidToken",
    "SubjectMismatch",
    "Token",
    "access_protected_resource",
    "refresh_session",
]
