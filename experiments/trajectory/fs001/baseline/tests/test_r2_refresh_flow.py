from __future__ import annotations

import unittest

from auth_fixture.api import refresh_session
from auth_fixture.tokens import (
    FROZEN_NOW,
    expired_access_token,
    valid_refresh_token,
)


class RefreshFlowRequirement(unittest.TestCase):
    def test_valid_refresh_renews_expired_access_session(self) -> None:
        replacement = refresh_session(
            expired_access_token(),
            valid_refresh_token(),
            FROZEN_NOW,
        )
        self.assertEqual(replacement.kind, "access")
        self.assertEqual(replacement.subject, "user-123")
        self.assertGreater(replacement.expires_at, FROZEN_NOW)


if __name__ == "__main__":
    unittest.main()
