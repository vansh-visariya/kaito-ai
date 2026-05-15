"""Unified agent module for Kaito-AI.

Both the Search agent and the RAG agent are ReAct tool-calling agents
built with ``langgraph.prebuilt.create_react_agent``.  They share one
web-search tool factory and one ``_AgentWrapper`` class.

Agents
------
create_search_agent — web search only (no documents).
    Tools: tavily_search

create_rag_agent — PDF document retrieval + web search fallback.
    Tools: document_retriever, tavily_search

Both expose the same interface:
    agent.invoke({"question": "..."}) -> {"generation": "...", "question": "..."}
    agent.get_state(config)           -> LangGraph snapshot (for history)
"""

import logging
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools.retriever import create_retriever_tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent

from config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RETRIEVER_K,
    VECTOR_STORE_DIR,
    configure_environment,
)
from database.memory import get_rag_memory, get_search_memory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared: web search tool
# ---------------------------------------------------------------------------
def make_web_search_tool() -> TavilySearch:
    """Return a configured Tavily web-search tool (4 results)."""
    return TavilySearch(max_results=4)


# ---------------------------------------------------------------------------
# RAG-only: embeddings + vector store
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_embeddings() -> HuggingFaceEmbeddings:
    """Load and cache the HuggingFace embedding model (once per process)."""
    logger.info("Loading embedding model: %s", DEFAULT_EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)


def build_vector_store(file_paths: list[str]) -> Chroma:
    """Create a ChromaDB vector store from a list of PDF file paths.

    Args:
        file_paths: Absolute paths to PDF files on disk.

    Returns:
        A populated :class:`Chroma` vector store.

    Raises:
        ValueError: If no text could be extracted from the supplied PDFs.
    """
    all_docs = []
    for path in file_paths:
        logger.info("Loading PDF: %s", path)
        pages = PyPDFLoader(path).load()
        logger.info("  -> %d page(s)", len(pages))
        all_docs.extend(pages)

    if not all_docs:
        raise ValueError("No content could be extracted from the uploaded PDF(s).")

    logger.info(
        "Loaded %d page(s) from %d PDF(s).", len(all_docs), len(file_paths)
    )

    splits = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    ).split_documents(all_docs)
    logger.info("Created %d text chunks.", len(splits))

    return Chroma.from_documents(
        documents=splits,
        embedding=_load_embeddings(),
        persist_directory=VECTOR_STORE_DIR,
    )


# ---------------------------------------------------------------------------
# Shared: agent wrapper
# ---------------------------------------------------------------------------
class _AgentWrapper:
    """Adapts a ReAct agent's messages interface to the question/generation
    interface used by ``api.py``.

    Input  (invoke):  {"question": "..."}
    Output (invoke):  {"generation": "...", "question": "..."}

    ``get_state`` is a pass-through so conversation-history loading in
    ``api.py`` (which reads ``state.values["messages"]``) keeps working.
    """

    def __init__(self, agent) -> None:
        self._agent = agent

    def invoke(self, inputs: dict, config: dict | None = None) -> dict:
        question = inputs.get("question", "")
        result = self._agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config,
        )
        # The last AIMessage with non-empty content is the final answer.
        ai_msgs = [
            m for m in result["messages"]
            if isinstance(m, AIMessage) and m.content
        ]
        generation = (
            ai_msgs[-1].content
            if ai_msgs
            else "Sorry, I couldn't generate a response."
        )
        return {"generation": generation, "question": question}

    def get_state(self, config: dict):
        return self._agent.get_state(config)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
_SEARCH_SYSTEM = SystemMessage(content="""\
You are a knowledgeable AI assistant with access to a live web-search tool.

Rules:
- Use tavily_search for current events, recent news, or anything your
  training data may not cover reliably.
- For well-established facts you are confident about, answer directly
  without searching.
- Keep answers concise, accurate, and well-structured.
""")

_RAG_SYSTEM = SystemMessage(content="""\
You are an AI assistant with access to two tools:

1. document_retriever — searches the uploaded PDF documents.
2. tavily_search      — searches the live web.

Rules:
- ALWAYS call document_retriever first for any question.
- Read the retrieved passages carefully and base your answer on them.
- Only call tavily_search if the documents do not contain the needed info.
- Quote or paraphrase specific details from retrieved passages when possible.
- If neither source has the answer, say so honestly.
""")


# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------
def create_search_agent(
    groq_api_key: str,
    model_name: str,
    tavily_api_key: str,
) -> _AgentWrapper:
    """Build a ReAct search agent with web-search as its only tool.

    Args:
        groq_api_key:   Groq API key for LLM inference.
        model_name:     Groq model identifier (e.g. ``llama-3.1-8b-instant``).
        tavily_api_key: Tavily API key for web search.

    Returns:
        An :class:`_AgentWrapper` ready to handle chat requests.
    """
    configure_environment(groq_api_key, tavily_api_key)

    llm    = ChatGroq(model=model_name)
    memory = get_search_memory()
    tools  = [make_web_search_tool()]

    agent = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=memory,
        state_modifier=_SEARCH_SYSTEM,
    )
    logger.info("Search agent compiled (model=%s).", model_name)
    return _AgentWrapper(agent)


def create_rag_agent(
    groq_api_key: str,
    model_name: str,
    file_paths: list[str],
    tavily_api_key: str,
) -> _AgentWrapper:
    """Build a ReAct RAG agent with document retrieval + web-search tools.

    Args:
        groq_api_key:   Groq API key for LLM inference.
        model_name:     Groq model identifier.
        file_paths:     Absolute paths to PDF files on disk.
        tavily_api_key: Tavily API key for web-search fallback.

    Returns:
        An :class:`_AgentWrapper` ready to handle chat requests.
    """
    configure_environment(groq_api_key, tavily_api_key)

    llm    = ChatGroq(model=model_name)
    memory = get_rag_memory()

    # Build ChromaDB vector store and wrap as a tool
    vector_store = build_vector_store(file_paths)
    retriever    = vector_store.as_retriever(
        search_kwargs={"k": DEFAULT_RETRIEVER_K}
    )
    retriever_tool = create_retriever_tool(
        retriever,
        name="document_retriever",
        description=(
            "Search and retrieve relevant passages from the uploaded PDF "
            "documents. Use this tool first for any question about the files. "
            "Input: a natural-language search query."
        ),
    )

    tools = [retriever_tool, make_web_search_tool()]

    agent = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=memory,
        state_modifier=_RAG_SYSTEM,
    )
    logger.info("RAG agent compiled (model=%s, %d file(s)).", model_name, len(file_paths))
    return _AgentWrapper(agent)
