"""Centralized configuration for Kaito-AI.

This module defines all application constants, the Mode enum, and a single
entry-point for setting environment variables so that no other module touches
``os.environ`` directly.
"""

import logging
import os
from enum import Enum
from typing import Optional

# Protobuf compatibility fix
# chromadb bundles opentelemetry-proto whose _pb2.py files were generated
# with an old protoc version; the pure-Python protobuf implementation is
# fully compatible and avoids the "Descriptors cannot be created directly"
# TypeError that appears with protobuf >= 4 on Python 3.13.
# This MUST be set before any chromadb / opentelemetry import.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)


# Enums
class Mode(str, Enum):
    """Operational mode of the chatbot."""

    SEARCH = "search"
    RAG = "rag"


# Application Constants
DEFAULT_MODEL: str = "llama-3.1-8b-instant"
DEFAULT_CHUNK_SIZE: int = 1000
DEFAULT_CHUNK_OVERLAP: int = 200
DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_RETRIEVER_K: int = 3
MAX_GENERATION_RETRIES: int = 3

VECTOR_STORE_DIR: str = "./chroma_langchain_db"
SEARCH_DB_PATH: str = "database/search_chatbot.db"
RAG_DB_PATH: str = "database/rag_chatbot.db"

LANGCHAIN_PROJECT_NAME: str = "chatbot"

THREAD_PREFIX: dict[Mode, str] = {
    Mode.SEARCH: "search_",
    Mode.RAG: "rag_",
}


# Environment helpers
def configure_environment(
    groq_api_key: str,
    tavily_api_key: str,
    langchain_api_key: Optional[str] = None,
) -> None:
    """Set all required environment variables in a single place.

    This is the **only** function in the codebase that should write to
    ``os.environ``.  Every other module should call this instead of
    setting env vars itself.
    """
    os.environ["GROQ_API_KEY"] = groq_api_key
    os.environ["TAVILY_API_KEY"] = tavily_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT_NAME
    if langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = langchain_api_key


def get_thread_mode(thread_id: str) -> Mode:
    """Determine the chat mode from a thread-ID prefix."""
    if thread_id.startswith(THREAD_PREFIX[Mode.RAG]):
        return Mode.RAG
    return Mode.SEARCH
