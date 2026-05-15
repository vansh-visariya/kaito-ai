# 🤖 Kaito-AI — Intelligent Search & Document Analysis

An AI-powered chatbot with a **FastAPI backend** and a **static HTML/CSS/JS frontend** that combines live web search with PDF document analysis (RAG). Powered by Groq LLMs and LangGraph ReAct agents.

---

## ✨ Features

### 🔍 Search Mode
- **Smart web search** — the agent decides when to hit Tavily vs. answer from its own knowledge
- **Conversation memory** — full thread history persisted in SQLite across requests
- **Multi-thread** — create and switch between unlimited conversation threads

### 📄 Document Analysis (RAG Mode)
- **Upload PDFs** — one or more documents processed on upload
- **Semantic search** — HuggingFace embeddings (`all-mpnet-base-v2`) + ChromaDB vector store
- **Tool-calling agent** — LLM explicitly calls `document_retriever`, falls back to web search only when needed
- **Separate thread** — each PDF upload starts a fresh RAG conversation

### 💬 Thread Management
- Create, switch, and delete conversation threads
- Clean up empty threads in one click
- Mode badge shows whether a thread is Search or RAG

### ⚙️ Tech Stack
| Layer | Technology |
|---|---|
| LLM | Groq (`llama-3.1-8b-instant`, Gemma 2, Mixtral, …) |
| Agent orchestration | LangGraph `create_react_agent` |
| Web search | Tavily API |
| PDF loading | LangChain `PyPDFLoader` |
| Embeddings | HuggingFace `sentence-transformers/all-mpnet-base-v2` |
| Vector store | ChromaDB |
| Memory | LangGraph `SqliteSaver` (SQLite) |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML / CSS / JS (dark-mode UI) |

---

## 🏗️ Project Structure

```
kaito-ai/
├── api.py                  # FastAPI backend — all REST endpoints
├── config.py               # App constants, Mode enum, env helpers
├── utility.py              # ID generation, Groq key validation, memory helpers
├── requirements.txt        # Pinned dependencies
├── pyproject.toml          # Project metadata (uv / pip)
│
├── agent/
│   ├── __init__.py
│   └── agent.py            # Unified agent module
│                           #   make_web_search_tool()   — shared Tavily tool
│                           #   build_vector_store()     — PDF → ChromaDB
│                           #   create_search_agent()    — tools: [tavily_search]
│                           #   create_rag_agent()       — tools: [document_retriever, tavily_search]
│
├── database/
│   ├── __init__.py
│   └── memory.py           # SqliteSaver singletons for search & RAG threads
│
└── frontend/
    ├── index.html          # Single-page app shell
    ├── style.css           # Dark-mode design system
    └── app.js              # All API calls, chat rendering, thread management
```

---

## 🚀 Quick Start

### Prerequisites
- Python ≥ 3.10
- [`uv`](https://github.com/astral-sh/uv) (recommended) **or** `pip`
- A [Groq API key](https://console.groq.com/) — **required**
- A [Tavily API key](https://tavily.com/) — required for web search
- A [LangSmith API key](https://smith.langchain.com/) — optional (tracing)

### 1. Clone
```bash
git clone https://github.com/vansh-visariya/kaito-ai.git
cd kaito-ai
```

### 2. Install dependencies

**With uv (recommended):**
```bash
uv sync
```

**With pip:**
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

`.env` variables:
```env
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly-...
# LANGCHAIN_API_KEY=ls__...   # optional
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python   # required for chromadb on Python 3.13
```

### 4. Run the server

```bash
uv run uvicorn api:app --reload --port 8000
```

or with plain Python:
```bash
uvicorn api:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

---

## 📖 How to Use

### First Launch — Configuration
The app opens a configuration modal. Enter:
- **Groq API Key** *(required)*
- **Model name** — default `llama-3.1-8b-instant`; any Groq model works
- **Tavily API Key** *(for web search)*
- **LangSmith API Key** *(optional — enables tracing at smith.langchain.com)*

Click **Connect**. The key is validated against the Groq API before proceeding.

### Search Mode 🔍
Type any question in the chat box. The agent:
1. Decides whether to answer from its own knowledge or call `tavily_search`
2. Fetches live web results if needed
3. Generates a final answer with conversation history

Best for: current events, news, general knowledge, coding help.

### RAG Mode 📄
Click the paperclip icon → select one or more PDF files → press Send.  
The app:
1. Saves uploads to temporary files, loads them with `PyPDFLoader`
2. Splits pages into chunks, embeds them with HuggingFace, stores in ChromaDB
3. Starts a new `rag_` prefixed conversation thread
4. On every question, the agent calls `document_retriever` first, falls back to Tavily only if the docs don't have the answer

Best for: research papers, contracts, manuals, reports.

### Thread Management
| Action | How |
|---|---|
| New chat | Click **+ New Chat** in the sidebar |
| Switch thread | Click any thread in the sidebar list |
| Delete thread | Hover over a thread → click **×** |
| Clean empty threads | **Clean Empty Threads** button at the bottom of sidebar |
| Clear all documents | **Clear All Docs** button (resets RAG, starts new search thread) |
| Reconfigure API keys | **Reconfigure** button at the bottom of sidebar |

---

## 🔌 REST API Reference

The FastAPI backend exposes the following endpoints (also served at `/docs` via Swagger UI):

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/config` | Set API keys & model; validates Groq key |
| `GET` | `/api/config/status` | Check if session is configured |
| `GET` | `/api/threads` | List all conversation threads |
| `POST` | `/api/threads/new` | Start a new search thread |
| `POST` | `/api/threads/select` | Switch to an existing thread |
| `DELETE` | `/api/threads/{id}` | Delete a specific thread |
| `DELETE` | `/api/threads` | Delete all empty threads |
| `POST` | `/api/chat` | Send a message, get a response |
| `GET` | `/api/chat/{id}/history` | Load message history for a thread |
| `POST` | `/api/documents/upload` | Upload PDFs → build RAG chain |
| `GET` | `/api/documents` | List uploaded document names |
| `DELETE` | `/api/documents` | Clear all documents & vector store |

---

## 🛠️ Configuration Reference

### Supported Groq Models
| Model | Notes |
|---|---|
| `llama-3.1-8b-instant` | Default — fast, good quality |
| `llama-3.3-70b-versatile` | Strongest reasoning |
| `gemma2-9b-it` | Google Gemma 2 |
| `mixtral-8x7b-32768` | Long context (32k tokens) |

Any model available on [console.groq.com/docs/models](https://console.groq.com/docs/models) can be entered.

### Chunking & Retrieval Defaults (`config.py`)
| Setting | Default | Description |
|---|---|---|
| `DEFAULT_CHUNK_SIZE` | `1000` | Characters per chunk |
| `DEFAULT_CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `DEFAULT_EMBEDDING_MODEL` | `sentence-transformers/all-mpnet-base-v2` | HuggingFace model |
| `DEFAULT_RETRIEVER_K` | `3` | Top-k chunks retrieved per query |
| `MAX_GENERATION_RETRIES` | `3` | Max agent tool-call iterations |

---

## 🧠 How the Agents Work

Both agents are **LangGraph ReAct agents** (`create_react_agent`). The LLM is given tools and autonomously decides when and how to call them.

### Search Agent
```
User question
      │
      ▼
  LLM decides: need web search?
      ├─ No  → answers from internal knowledge
      └─ Yes → calls tavily_search → generates answer
```
**Memory**: `database/search_chatbot.db`

### RAG Agent
```
User question
      │
      ▼
  LLM calls document_retriever("query")
      │
      ├─ Relevant chunks found → generates answer from docs
      └─ Not found → calls tavily_search → generates answer from web
```
**Memory**: `database/rag_chatbot.db`

Both agents share:
- `make_web_search_tool()` — single Tavily tool factory
- `_AgentWrapper` — adapts the ReAct `messages` interface to the `question/generation` interface used by the API

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'feat: add my feature'`)
4. Push and open a Pull Request

---

## 📝 License

MIT — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **[LangChain](https://github.com/langchain-ai/langchain)** — LLM framework & tooling
- **[LangGraph](https://github.com/langchain-ai/langgraph)** — ReAct agent orchestration
- **[Groq](https://groq.com/)** — Ultra-fast LLM inference
- **[Tavily](https://tavily.com/)** — AI-optimised web search
- **[ChromaDB](https://www.trychroma.com/)** — Local vector store
- **[HuggingFace](https://huggingface.co/)** — Sentence embeddings
- **[FastAPI](https://fastapi.tiangolo.com/)** — Modern Python web framework

---

*Built with Python, FastAPI, and LangGraph*
