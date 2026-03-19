from __future__ import annotations

import uuid

from .state import SessionState


def create_session(user_query: str) -> SessionState:
    """Create a new runtime session with a generated identifier."""

    return SessionState(
        session_id=uuid.uuid4().hex,
        user_query=user_query.strip(),
    )
