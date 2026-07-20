class SessionError(RuntimeError):
    """Base error for durable Session encoding and storage."""


class SessionSerializationError(SessionError):
    """A Session payload is invalid, unsupported, or not JSON-compatible."""


class SessionStorageError(SessionError):
    """A Session could not be read or atomically persisted."""


class SessionConflictError(SessionStorageError):
    """A branch head changed before a conditional append could commit."""


class SessionLockTimeoutError(SessionStorageError):
    """A Session journal lock could not be acquired before its deadline."""
