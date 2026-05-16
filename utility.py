"""Shared utility functions for Kaito-AI.

Provides helpers for ID generation, API-key validation, and
mode-aware memory lookup.
"""

import logging
import uuid
from typing import TYPE_CHECKING

import requests

from config import Mode
from database.memory import get_rag_memory, get_search_memory

if TYPE_CHECKING:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

logger = logging.getLogger(__name__)

# Timeout (seconds) for outbound HTTP requests.
_API_TIMEOUT: int = 10


def generate_unique_id() -> str:
    """Generate a random UUID4 string."""
    return str(uuid.uuid4())


def validate_groq_key(api_key: str) -> bool:
    """Validate a Groq API key by querying the models endpoint.

    Args:
        api_key: The Groq API key to validate.

    Returns:
        ``True`` if the key is accepted by the Groq API, ``False`` otherwise.
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers=headers,
            timeout=_API_TIMEOUT,
        )
        return response.status_code == 200
    except requests.RequestException as exc:
        logger.warning("Groq API key validation failed: %s", exc)
        return False


async def get_memory_for_mode(mode: Mode) -> "AsyncSqliteSaver":
    """Return the SQLite checkpointer for the given chat *mode*.

    Args:
        mode: :class:`Mode.SEARCH` or :class:`Mode.RAG`.

    Returns:
        The corresponding ``AsyncSqliteSaver`` instance.
    """
    if mode == Mode.RAG:
        return await get_rag_memory()
    return await get_search_memory()
