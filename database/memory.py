"""LangGraph checkpoint memory management using SQLite.

Provides a singleton ``AsyncSqliteSaver`` instance for conversation state
persistence across API requests with async streaming support.
"""

import logging

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from config import CHATBOT_DB_PATH

logger = logging.getLogger(__name__)

_MEMORY_SAVER = None


async def get_memory() -> AsyncSqliteSaver:
    """Return the async SQLite checkpointer for all agents."""
    global _MEMORY_SAVER
    if _MEMORY_SAVER is None:
        logger.info("Initializing async memory at %s", CHATBOT_DB_PATH)
        conn = await aiosqlite.connect(CHATBOT_DB_PATH, check_same_thread=False)
        _MEMORY_SAVER = AsyncSqliteSaver(conn)
    return _MEMORY_SAVER
