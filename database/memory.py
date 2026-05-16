"""LangGraph checkpoint memory management using SQLite.

Provides singleton ``AsyncSqliteSaver`` instances for the search and RAG agents
so that conversation state persists across API requests and supports async streaming.
"""

import logging
import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from config import RAG_DB_PATH, SEARCH_DB_PATH

logger = logging.getLogger(__name__)

_SEARCH_SAVER = None
_RAG_SAVER = None

async def get_search_memory() -> AsyncSqliteSaver:
    """Return the async SQLite checkpointer for the **search** agent."""
    global _SEARCH_SAVER
    if _SEARCH_SAVER is None:
        logger.info("Initializing async search memory at %s", SEARCH_DB_PATH)
        conn = await aiosqlite.connect(SEARCH_DB_PATH, check_same_thread=False)
        _SEARCH_SAVER = AsyncSqliteSaver(conn)
    return _SEARCH_SAVER

async def get_rag_memory() -> AsyncSqliteSaver:
    """Return the async SQLite checkpointer for the **RAG** agent."""
    global _RAG_SAVER
    if _RAG_SAVER is None:
        logger.info("Initializing async RAG memory at %s", RAG_DB_PATH)
        conn = await aiosqlite.connect(RAG_DB_PATH, check_same_thread=False)
        _RAG_SAVER = AsyncSqliteSaver(conn)
    return _RAG_SAVER
