import asyncio

from ejagent.session.types import AgentSession


class MemorySessionStorage:
    """Process-local Session storage with copy isolation."""

    def __init__(self) -> None:
        self._sessions: dict[str, AgentSession] = {}
        self._lock = asyncio.Lock()

    async def load(self, session_id: str) -> AgentSession | None:
        async with self._lock:
            session = self._sessions.get(session_id)
            return session.snapshot() if session is not None else None

    async def save(self, session: AgentSession) -> None:
        async with self._lock:
            self._sessions[session.session_id] = session.snapshot()
