"""Unified agent module for Kaito-AI.

Agents
------
create_search_agent — web search only.       Tools: [tavily_search]
create_rag_agent    — docs + web fallback.   Tools: [document_retriever, tavily_search]
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.retrievers import BaseRetriever
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

# Constants
SUMMARISE_AFTER = 20   # number of Human+AI messages before summarisation kicks in
BM25_WEIGHT     = 0.4  # weight given to BM25 results vs vector results (0.6)


# Shared: web search tool
def make_web_search_tool() -> TavilySearch:
    """Return a configured Tavily web-search tool (4 results)."""
    return TavilySearch(max_results=4)


# RAG-only: embeddings & reranker
@lru_cache(maxsize=1)
def _load_embeddings() -> HuggingFaceEmbeddings:
    logger.info("Loading embedding model: %s", DEFAULT_EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _load_reranker():
    logger.info("Loading cross-encoder reranker...")
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# RAG-only: hybrid BM25 + vector retriever (#6)
class _HybridRetriever(BaseRetriever):
    """Weighted ensemble of BM25 (keyword) + ChromaDB (semantic) retrievers.

    Since ``EnsembleRetriever`` is not available in the installed version of
    langchain, this class implements the same reciprocal-rank fusion logic
    manually.

    Attributes:
        bm25_retriever:    BM25Retriever built from document splits.
        vector_retriever:  Chroma as_retriever().
        bm25_weight:       Score weight for BM25 results (vector gets 1-weight).
        k:                 Number of final documents to return.
    """

    bm25_retriever:   Any
    vector_retriever: Any
    bm25_weight:      float = 0.4
    k:                int   = DEFAULT_RETRIEVER_K

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> list[Document]:  # type: ignore[override]
        bm25_docs   = self.bm25_retriever.invoke(query)
        vector_docs = self.vector_retriever.invoke(query)

        # Reciprocal-rank fusion score
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        def _score(docs, weight):
            for rank, doc in enumerate(docs):
                key = doc.page_content[:200]   # use content snippet as key
                scores[key]  = scores.get(key, 0.0) + weight / (rank + 1)
                doc_map[key] = doc

        _score(bm25_docs,   self.bm25_weight)
        _score(vector_docs, 1.0 - self.bm25_weight)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # Take top K * 2 from hybrid search for reranking
        top_docs = [doc_map[key] for key, _ in ranked[: self.k * 2]]

        if not top_docs:
            return []

        # Cross-encoder reranking
        reranker = _load_reranker()
        pairs = [[query, doc.page_content] for doc in top_docs]
        rerank_scores = reranker.predict(pairs)

        # Sort documents by cross-encoder score descending
        reranked = sorted(zip(rerank_scores, top_docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in reranked[: self.k]]

    async def _aget_relevant_documents(self, query: str) -> list[Document]:  # type: ignore[override]
        return self._get_relevant_documents(query)


def _load_and_split(file_paths: list[str]) -> list[Document]:
    """Load PDFs and split into chunks."""
    all_docs: list[Document] = []
    for path in file_paths:
        logger.info("Loading PDF: %s", path)
        pages = PyPDFLoader(path).load()
        logger.info("  -> %d page(s)", len(pages))
        all_docs.extend(pages)

    if not all_docs:
        raise ValueError("No content could be extracted from the uploaded PDF(s).")

    splits = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    ).split_documents(all_docs)
    logger.info("Created %d text chunks from %d PDF(s).", len(splits), len(file_paths))
    return splits


def delete_document_from_vector_store(file_path: str, vector_store_dir: str):
    """Delete all chunks associated with a specific file from Chroma."""
    vector_store = Chroma(
        persist_directory=vector_store_dir,
        embedding_function=_load_embeddings(),
    )
    # Use underlying chromadb collection to delete by metadata
    try:
        vector_store._collection.delete(where={"source": file_path})
        logger.info("Deleted %s from vector store %s", file_path, vector_store_dir)
    except Exception as exc:
        logger.warning("Failed to delete %s from vector store: %s", file_path, exc)


def build_hybrid_retriever(file_paths: list[str], vector_store_dir: str) -> _HybridRetriever:
    """Build and return a hybrid BM25 + ChromaDB retriever.

    Args:
        file_paths: Absolute paths to PDF files on disk.
        vector_store_dir: Directory to persist the Chroma database.

    Returns:
        A :class:`_HybridRetriever` combining keyword and semantic search.
    """
    splits = _load_and_split(file_paths)

    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=_load_embeddings(),
        persist_directory=vector_store_dir,
    )

    bm25   = BM25Retriever.from_documents(splits, k=DEFAULT_RETRIEVER_K)
    vector = vector_store.as_retriever(search_kwargs={"k": DEFAULT_RETRIEVER_K})

    logger.info("Hybrid retriever ready (BM25 weight=%.1f, vector weight=%.1f).",
                BM25_WEIGHT, 1.0 - BM25_WEIGHT)

    return _HybridRetriever(
        bm25_retriever=bm25,
        vector_retriever=vector,
        bm25_weight=BM25_WEIGHT,
        k=DEFAULT_RETRIEVER_K,
    )


# Source extraction helper
def _extract_sources(messages: list) -> list[dict]:
    """Pull document citations out of ToolMessage artifacts."""
    sources: list[dict] = []
    seen: set[tuple]    = set()

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        for doc in getattr(msg, "artifact", None) or []:
            if not hasattr(doc, "metadata"):
                continue
            raw  = doc.metadata.get("source", "")
            page = doc.metadata.get("page", 0)
            file = Path(raw).name if raw else ""
            key  = (file, page)
            if file and key not in seen:
                seen.add(key)
                sources.append({"file": file, "page": page + 1})

    return sources


# Conversation summarisation helper (#5)
async def _maybe_summarise(agent, llm, config: dict | None) -> None:
    """Summarise old messages when a thread grows beyond SUMMARISE_AFTER turns."""
    if not config:
        return
    try:
        snapshot = await agent.aget_state(config)
    except Exception:
        return

    messages = snapshot.values.get("messages", [])
    convo = [m for m in messages if isinstance(m, (HumanMessage, AIMessage)) and m.content]

    if len(convo) <= SUMMARISE_AFTER:
        return

    to_summarise = convo[:-6]
    recent       = convo[-6:]

    convo_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content[:600]}"
        for m in to_summarise
    )

    try:
        result = await llm.ainvoke([
            SystemMessage(content=(
                "Summarise the conversation below in 4–6 sentences. "
                "Preserve key facts, questions, and answers. Be concise."
            )),
            HumanMessage(content=convo_text),
        ])
        summary = result.content
    except Exception as exc:
        logger.warning("Summarisation LLM call failed: %s", exc)
        return

    new_messages = [
        SystemMessage(content=f"[Earlier conversation summary]: {summary}"),
        *recent,
    ]
    try:
        await agent.aupdate_state(config, {"messages": new_messages})
        logger.info("Thread summarised — %d messages → summary + 6 recent.", len(to_summarise))
    except Exception as exc:
        logger.warning("update_state failed: %s", exc)


# Shared: agent wrapper
class _AgentWrapper:
    """Adapts the ReAct agent to the question/generation interface.

    Input  (invoke):  {"question": "..."}
    Output (invoke):  {"generation": "...", "question": "...", "sources": [...]}
    """

    def __init__(self, agent, llm) -> None:
        self._agent = agent
        self._llm   = llm

    async def aget_state(self, config: dict):
        return await self._agent.aget_state(config)

    async def ainvoke(self, inputs: dict, config: dict | None = None) -> dict:
        await _maybe_summarise(self._agent, self._llm, config)      # #5

        question = inputs.get("question", "")
        result   = await self._agent.ainvoke(
            {"messages": [HumanMessage(content=question)]},
            config=config,
        )
        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage) and m.content]
        generation = ai_msgs[-1].content if ai_msgs else "Sorry, I couldn't generate a response."
        return {
            "generation": generation,
            "question":   question,
            "sources":    _extract_sources(result["messages"]),
        }

    async def astream_events(self, inputs: dict, config: dict | None = None):
        """Async generator of raw LangGraph events for SSE streaming."""
        await _maybe_summarise(self._agent, self._llm, config)      # #5

        question = inputs.get("question", "")
        async for event in self._agent.astream_events(
            {"messages": [HumanMessage(content=question)]},
            config=config,
            version="v2",
        ):
            yield event

    def get_state(self, config: dict):
        return self._agent.get_state(config)


# System prompts
_SEARCH_SYSTEM = SystemMessage(content="""\
You are a knowledgeable AI assistant.
Answer the user's questions accurately. Use the provided web search tool if you need current events or if you are unsure.
Keep answers concise and well-structured.
""")

_RAG_SYSTEM = SystemMessage(content="""\
You are an AI assistant.
You have access to a document retriever and a web search tool.
ALWAYS search the documents first for any user query.
Base your answer primarily on retrieved passages and quote specific details when possible.
Only use web search if the documents don't contain the needed info.
""")


# Public factories
async def create_search_agent(
    groq_api_key: str,
    model_name:   str,
    tavily_api_key: str,
) -> _AgentWrapper:
    """Build a ReAct search agent (web-search only)."""
    configure_environment(groq_api_key, tavily_api_key)

    llm    = ChatGroq(model=model_name, streaming=True)
    memory = await get_search_memory()
    tools  = [make_web_search_tool()]

    agent = create_react_agent(
        model=llm, tools=tools, checkpointer=memory, prompt=_SEARCH_SYSTEM
    )
    logger.info("Search agent compiled (model=%s).", model_name)
    return _AgentWrapper(agent, llm)


async def create_rag_agent(
    groq_api_key:   str,
    model_name:     str,
    file_paths:     list[str],
    tavily_api_key: str,
    vector_store_dir: str,
) -> _AgentWrapper:
    """Build a ReAct RAG agent with hybrid retrieval + web-search fallback."""
    configure_environment(groq_api_key, tavily_api_key)

    llm    = ChatGroq(model=model_name, streaming=True)
    memory = await get_rag_memory()

    # #6 — hybrid BM25 + vector retriever
    retriever = build_hybrid_retriever(file_paths, vector_store_dir)

    retriever_tool = create_retriever_tool(
        retriever,
        name="document_retriever",
        description=(
            "Search and retrieve relevant passages from the uploaded PDF documents "
            "using hybrid keyword + semantic search. Use this tool first for any "
            "question about the uploaded files. Input: a natural-language search query."
        ),
        response_format="content_and_artifact",
    )

    tools = [retriever_tool, make_web_search_tool()]

    agent = create_react_agent(
        model=llm, tools=tools, checkpointer=memory, prompt=_RAG_SYSTEM
    )
    logger.info("RAG agent compiled (model=%s, %d file(s), hybrid retrieval).",
                model_name, len(file_paths))
    return _AgentWrapper(agent, llm)
