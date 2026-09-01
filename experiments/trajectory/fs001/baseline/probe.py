"""Emit deterministic public-contract facts for FS-001."""

from __future__ import annotations

import inspect
import json

from auth_fixture.api import access_protected_resource, refresh_session
from auth_fixture.tokens import (
    FROZEN_NOW,
    issue_access_token,
)

replacement = issue_access_token("user-123", FROZEN_NOW)
print(
    json.dumps(
        {
            "access_signature": str(inspect.signature(access_protected_resource)),
            "refresh_signature": str(inspect.signature(refresh_session)),
            "token_payload_keys": sorted(replacement.to_payload()),
        },
        sort_keys=True,
    )
)
