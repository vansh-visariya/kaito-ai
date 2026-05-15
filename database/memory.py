"""LangGraph checkpoint memory management using SQLite.

Provides singleton ``SqliteSaver`` instances for the search and RAG agents
so that conversation state persists across API requests.
"""

import logging
import sqlite3
from functools import lru_cache

from langgraph.checkpoint.sqlite import SqliteSaver

from config import RAG_DB_PATH, SEARCH_DB_PATH

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_search_memory() -> SqliteSaver:
    """Return the SQLite checkpointer for the **search** agent.

    The connection is created once and reused on subsequent calls thanks
    to ``@lru_cache``.
    """
    logger.info("Initializing search memory at %s", SEARCH_DB_PATH)
    conn = sqlite3.connect(database=SEARCH_DB_PATH, check_same_thread=False)
    return SqliteSaver(conn=conn)


@lru_cache(maxsize=1)
def get_rag_memory() -> SqliteSaver:
    """Return the SQLite checkpointer for the **RAG** agent.

    The connection is created once and reused on subsequent calls thanks
    to ``@lru_cache``.
    """
    logger.info("Initializing RAG memory at %s", RAG_DB_PATH)
    conn = sqlite3.connect(database=RAG_DB_PATH, check_same_thread=False)
    return SqliteSaver(conn=conn)
