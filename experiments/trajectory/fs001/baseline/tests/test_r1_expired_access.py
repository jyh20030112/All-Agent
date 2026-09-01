from __future__ import annotations

import unittest

from auth_fixture.api import access_protected_resource
from auth_fixture.tokens import FROZEN_NOW, expired_access_token
from auth_fixture.validation import ExpiredAccessToken


class ExpiredAccessRequirement(unittest.TestCase):
    def test_expired_access_is_rejected(self) -> None:
        with self.assertRaises(ExpiredAccessToken):
            access_protected_resource(expired_access_token(), FROZEN_NOW)


if __name__ == "__main__":
    unittest.main()
