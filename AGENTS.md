# Memory

## Project Overview
Kaito-AI is an AI-powered chatbot with a **FastAPI backend** and a **vanilla HTML/CSS/JS frontend** that combines live web search with PDF document analysis (RAG). Powered by Groq LLMs and LangGraph ReAct agents.

**Two operational modes:**
- **Search Mode** (`search_*` threads): LangGraph ReAct agent with Tavily web search tool
- **RAG Mode** (`rag_*` threads): Hybrid retrieval (BM25 + ChromaDB vector + cross-encoder reranking) with web fallback

**Key stack:** FastAPI, LangGraph, Groq LLMs, ChromaDB, SQLite (LangGraph AsyncSqliteSaver), Tavily API, HuggingFace embeddings/reranker, vanilla HTML/CSS/JS frontend with marked.js + highlight.js

### Project Structure
```
kaito-ai/
├── api.py                  # FastAPI backend — all 13 REST endpoints, session management, SSE streaming
├── config.py               # App constants, Mode enum, configure_environment(), get_thread_mode()
├── utility.py              # UUID generation, Groq key validation, mode-aware memory lookup
├── pyproject.toml          # Project metadata & dependencies (uv / pip)
├── requirements.txt        # Pinned dependency versions
│
├── agent/
│   ├── __init__.py
│   └── agent.py            # Unified agent module
│                           #   _HybridRetriever — BM25 + vector reciprocal-rank fusion + cross-encoder reranking
│                           #   _AgentWrapper     — adapts ReAct messages → question/generation interface + summarisation + SSE
│                           #   make_web_search_tool()          — shared TavilySearch(max_results=4)
│                           #   build_hybrid_retriever()        — BM25 + ChromaDB + reranker pipeline
│                           #   add_document_to_vector_store()  — load+split+embed single PDF
│                           #   delete_document_from_vector_store() — delete chunks by source metadata
│                           #   create_search_agent()           — tools: [tavily_search]
│                           #   create_rag_agent()              — tools: [document_retriever, tavily_search]
│
├── database/
│   ├── __init__.py
│   └── memory.py           # AsyncSqliteSaver singletons: get_search_memory() → search_chatbot.db, get_rag_memory() → rag_chatbot.db
│
└── frontend/
    ├── index.html          # Single-page app shell (config modal, sidebar, chat area, upload overlay)
    ├── style.css           # Dark-mode design system with animations
    └── app.js              # Client-side logic: SSE streaming, markdown rendering, thread/docs management
```

## Code Style Guidelines
- Use descriptive variable names
- Follow existing patterns in the codebase
- Extract complex conditions into meaningful boolean variables

## Architecture Notes

### Session & Multi-User Model
- Sessions stored in-memory (`SESSIONS` dict) + JSON persistence (`database/sessions.json`)
- Session ID = username from config form, stored as HTTP-only cookie (`session_id`)
- Each session has independent: API keys (in-memory only, not persisted), agent graphs (lazy-init), thread list, uploaded docs, vector store dir
- Vector stores and uploads are per-session: `chroma_langchain_db_{session_id}`, `uploads/{session_id}/`
- API keys never persisted to disk — must be re-entered on server restart

### Agent Architecture
- Two separate LangGraph ReAct agents, built lazily and cached on the Session object
- **Search agent**: ChatGroq + TavilySearch tool + search_chatbot.db memory + `_SEARCH_SYSTEM` prompt
- **RAG agent**: ChatGroq + document_retriever tool + TavilySearch tool + rag_chatbot.db memory + `_RAG_SYSTEM` prompt
- Thread mode determined by ID prefix: `search_*` or `rag_*`
- `_AgentWrapper` adapts the LangGraph `{messages: [...]}` interface to `{question, generation, sources}` used by the API
- Automatic summarisation when conversation exceeds 20 Human/AI message pairs (keeps last 6 messages, replaces rest with a SystemMessage summary)

### Hybrid Retrieval Pipeline (RAG only)
`_HybridRetriever` implements reciprocal-rank fusion manually (EnsembleRetriever unavailable in installed langchain version):
1. BM25 keyword search (rank-bm25, weight 0.4) + ChromaDB semantic search (all-mpnet-base-v2, weight 0.6)
2. Reciprocal-rank fusion scoring → take top K*2 candidates
3. Cross-encoder reranking (ms-marco-MiniLM-L-6-v2) → return top K

### Data Flow (SSE streaming)
```
User types message → app.js handleSend()
→ POST /api/chat/stream {message, thread_id}
→ api.py chat_stream()
  → _get_graph_for_thread() → resolve search or RAG agent
  → graph.astream_events({"question": message}, config={thread_id})
    → _maybe_summarise() if conversation too long
    → agent.ainvoke({"messages": [HumanMessage]})
    → yield SSE tokens on on_chat_model_stream
    → capture source citations on on_tool_end (document_retriever)
    → yield done event with thread_id, mode, sources
→ app.js consumes ReadableStream → incremental markdown render → final display with sources
```

### Key Design Patterns
- **Singleton**: `_load_embeddings()`, `_load_reranker()` (LRU-cached), `get_search_memory()`, `get_rag_memory()` (module globals)
- **Adapter**: `_AgentWrapper` adapts LangGraph output to API contract
- **Strategy**: `Mode` enum + `_get_graph_for_thread()` selects agent strategy based on thread prefix
- **Dependency Injection**: FastAPI `Depends(get_session)` for session resolution
- **Lazy Initialization**: Agent graphs built on first use, cached on Session object

### Known Quirks
- `build_hybrid_retriever()` calls `_load_and_split()` + `Chroma().add_documents()` at agent creation time, which re-adds chunks already added by `add_document_to_vector_store()` during upload. Chunks are added twice (once during upload, once during agent creation).
- `config.py` docstring claims `configure_environment()` is the only function that writes to `os.environ`, but `api.py` line 22 also directly sets `os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]`.
- `MAX_GENERATION_RETRIES = 3` is defined in `config.py` and documented in README but is never imported or used anywhere in the codebase.
- The `LICENSE` file contains Apache 2.0, but README and `pyproject.toml` claim MIT.

## Common Workflows

### Running the server
```bash
uv sync                          # install dependencies
uv run uvicorn api:app --reload --port 8000
```

### Running tests
```bash
# No test suite exists yet
```

### Linting / formatting
```bash
uv run ruff check .              # lint
uv run ruff format .             # format
uv run mypy .                    # type-check (if installed via dev deps)
```
