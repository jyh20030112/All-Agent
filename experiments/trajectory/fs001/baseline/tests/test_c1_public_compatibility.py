from __future__ import annotations

import inspect
import unittest

from auth_fixture.api import access_protected_resource, refresh_session
from auth_fixture.tokens import (
    FROZEN_NOW,
    issue_access_token,
)
from auth_fixture.validation import ExpiredAccessToken, InvalidToken, SubjectMismatch


class PublicCompatibilityConstraint(unittest.TestCase):
    def test_public_contract_is_stable(self) -> None:
        self.assertEqual(
            tuple(inspect.signature(access_protected_resource).parameters),
            ("access_token", "now"),
        )
        self.assertEqual(
            tuple(inspect.signature(refresh_session).parameters),
            ("access_token", "refresh_token", "now"),
        )
        replacement = issue_access_token("user-123", FROZEN_NOW)
        self.assertEqual(
            tuple(sorted(replacement.to_payload())),
            ("expires_at", "issued_at", "kind", "subject"),
        )
        self.assertTrue(issubclass(ExpiredAccessToken, InvalidToken))
        self.assertTrue(issubclass(SubjectMismatch, InvalidToken))


if __name__ == "__main__":
    unittest.main()
