"""Centralized configuration for Kaito-AI.

This module defines all application constants and a single
entry-point for setting environment variables so that no other module touches
``os.environ`` directly.
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)


# Application Constants
DEFAULT_MODEL: str = "openai/gpt-oss-20b"
DEFAULT_CHUNK_SIZE: int = 1000
DEFAULT_CHUNK_OVERLAP: int = 200
DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_RETRIEVER_K: int = 3

DAILY_TOKEN_LIMIT: int = 50000

from pathlib import Path

# Data directory for persistent storage (crucial for Docker/Render deployments)
DATA_DIR = Path(os.environ.get("DATA_DIR", "."))

VECTOR_STORE_DIR: str = str(DATA_DIR / "chroma_langchain_db")
CHATBOT_DB_PATH: str = str(DATA_DIR / "database" / "chatbot.db")
USERS_DB_PATH: str = str(DATA_DIR / "database" / "users.db")
SESSIONS_FILE_PATH: str = str(DATA_DIR / "database" / "sessions.json")
UPLOADS_DIR_PATH: str = str(DATA_DIR / "uploads")

LANGSMITH_PROJECT: str = os.environ.get("LANGSMITH_PROJECT")

# Server-level API keys (set via .env, NOT user-configurable)
SERVER_GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
SERVER_TAVILY_API_KEY: str = os.environ.get("TAVILY_API_KEY", "")
SERVER_LANGCHAIN_API_KEY: str | None = os.environ.get("LANGCHAIN_API_KEY")


def configure_environment() -> None:
    """Ensure server API keys are in os.environ at startup.

    This is the **only** place env vars may be written for Groq/Tavily/LangChain.
    """
    if SERVER_GROQ_API_KEY:
        os.environ["GROQ_API_KEY"] = SERVER_GROQ_API_KEY
    if SERVER_TAVILY_API_KEY:
        os.environ["TAVILY_API_KEY"] = SERVER_TAVILY_API_KEY

    # LangSmith tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
    if SERVER_LANGCHAIN_API_KEY:
        os.environ["LANGCHAIN_API_KEY"] = SERVER_LANGCHAIN_API_KEY

    # Protobuf implementation fix — MUST be before any chromadb import
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
