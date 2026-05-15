"""Kaito-AI — FastAPI backend.

Replaces the Streamlit app.py with a proper REST API that serves the
static HTML/CSS/JS frontend and handles all chatbot logic.

Run with:
    uvicorn api:app --reload --port 8000
"""

import gc
import logging
import os
import shutil
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# pysqlite3 shim — Linux only
# ---------------------------------------------------------------------------
if sys.platform == "linux":
    try:
        __import__("pysqlite3")
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# Protobuf fix — MUST be before any chromadb import
# ---------------------------------------------------------------------------
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from agent.agent import create_search_agent, create_rag_agent
from config import (
    Mode,
    THREAD_PREFIX,
    VECTOR_STORE_DIR,
    configure_environment,
    get_thread_mode,
)
from database.memory import get_rag_memory, get_search_memory
from utility import generate_unique_id, get_memory_for_mode, validate_groq_key

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Kaito-AI API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory session store (single-user; extend with Redis for multi-user)
# ---------------------------------------------------------------------------
class Session:
    def __init__(self) -> None:
        self.groq_api_key: str = ""
        self.model_name: str = "llama-3.1-8b-instant"
        self.tavily_api_key: str = ""
        self.langchain_api_key: Optional[str] = None
        self.search_graph = None
        self.rag_graph = None
        self.current_mode: Mode = Mode.SEARCH
        self.uploaded_docs: list[str] = []
        self.thread_list: list[str] = []
        self.current_thread_id: str = ""

    def is_configured(self) -> bool:
        return bool(self.groq_api_key)


SESSION = Session()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class ConfigRequest(BaseModel):
    groq_api_key: str
    model_name: str = "llama-3.1-8b-instant"
    tavily_api_key: str = ""
    langchain_api_key: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


class ThreadDeleteRequest(BaseModel):
    thread_id: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_session() -> Session:
    if not SESSION.is_configured():
        raise HTTPException(status_code=401, detail="Not configured. POST /api/config first.")
    return SESSION


def _create_thread_id(mode: Mode) -> str:
    return f"{THREAD_PREFIX[mode]}{generate_unique_id()}"


def _get_search_graph(session: Session):
    if not session.search_graph:
        session.search_graph = create_search_agent(
            session.groq_api_key, session.model_name, session.tavily_api_key
        )
    return session.search_graph


def _get_rag_graph(session: Session):
    return session.rag_graph


def _get_graph_for_thread(session: Session, thread_id: str):
    mode = get_thread_mode(thread_id)
    if mode == Mode.RAG:
        if not session.rag_graph:
            raise HTTPException(status_code=400, detail="RAG graph not available. Upload documents first.")
        return session.rag_graph
    return _get_search_graph(session)


def _load_conversation(session: Session, thread_id: str) -> list[dict]:
    try:
        graph = _get_graph_for_thread(session, thread_id)
    except HTTPException:
        return []
    state = graph.get_state(config={"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])
    result = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})
    return result


def _retrieve_all_threads() -> list[str]:
    threads: set[str] = set()
    for mem in (get_search_memory(), get_rag_memory()):
        try:
            for checkpoint in mem.list(None):
                threads.add(checkpoint.config["configurable"]["thread_id"])
        except Exception:
            pass
    return list(threads)


def _delete_thread_from_db(thread_id: str) -> bool:
    mode = get_thread_mode(thread_id)
    memory = get_memory_for_mode(mode)
    try:
        cursor = memory.conn.cursor()
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        memory.conn.commit()
        return True
    except Exception as exc:
        logger.exception("Failed to delete thread %s: %s", thread_id, exc)
        return False


def _thread_preview(session: Session, thread_id: str) -> str:
    mode = get_thread_mode(thread_id)
    icon = "📄" if mode == Mode.RAG else "🔍"
    messages = _load_conversation(session, thread_id)
    if messages:
        content = messages[0]["content"]
        preview = content[:40] + "..." if len(content) > 40 else content
        return f"{icon} {preview}"
    return f"{icon} Thread {thread_id[-8:]}"


# ---------------------------------------------------------------------------
# Routes — Config
# ---------------------------------------------------------------------------
@app.post("/api/config")
async def configure(req: ConfigRequest):
    """Set API keys and model. Must be called before any chat."""
    if not validate_groq_key(req.groq_api_key):
        raise HTTPException(status_code=400, detail="Invalid Groq API key.")

    SESSION.groq_api_key = req.groq_api_key
    SESSION.model_name = req.model_name
    SESSION.tavily_api_key = req.tavily_api_key
    SESSION.langchain_api_key = req.langchain_api_key

    # Reset graphs so they're rebuilt with new keys
    SESSION.search_graph = None
    SESSION.rag_graph = None

    configure_environment(
        groq_api_key=req.groq_api_key,
        tavily_api_key=req.tavily_api_key,
        langchain_api_key=req.langchain_api_key,
    )

    # Bootstrap a default thread if none exists
    if not SESSION.current_thread_id:
        SESSION.thread_list = _retrieve_all_threads()
        if not SESSION.thread_list:
            tid = _create_thread_id(Mode.SEARCH)
            SESSION.thread_list.append(tid)
        SESSION.current_thread_id = SESSION.thread_list[-1]

    return {
        "status": "ok",
        "model": req.model_name,
        "current_thread_id": SESSION.current_thread_id,
    }


@app.get("/api/config/status")
async def config_status():
    """Check whether the session is configured."""
    return {
        "configured": SESSION.is_configured(),
        "model": SESSION.model_name,
        "current_thread_id": SESSION.current_thread_id,
        "mode": SESSION.current_mode.value,
    }


# ---------------------------------------------------------------------------
# Routes — Threads
# ---------------------------------------------------------------------------
@app.get("/api/threads")
async def list_threads():
    session = _require_session()
    threads = []
    for tid in reversed(session.thread_list):
        threads.append({
            "id": tid,
            "preview": _thread_preview(session, tid),
            "mode": get_thread_mode(tid).value,
            "active": tid == session.current_thread_id,
        })
    return {"threads": threads}


@app.post("/api/threads/new")
async def new_thread():
    session = _require_session()
    tid = _create_thread_id(Mode.SEARCH)
    session.current_thread_id = tid
    session.current_mode = Mode.SEARCH
    if tid not in session.thread_list:
        session.thread_list.append(tid)
    return {"thread_id": tid, "mode": Mode.SEARCH.value}


@app.post("/api/threads/select")
async def select_thread(req: ThreadDeleteRequest):
    session = _require_session()
    thread_id = req.thread_id
    if thread_id not in session.thread_list:
        raise HTTPException(status_code=404, detail="Thread not found.")
    session.current_thread_id = thread_id
    session.current_mode = get_thread_mode(thread_id)
    messages = _load_conversation(session, thread_id)
    return {"thread_id": thread_id, "messages": messages, "mode": session.current_mode.value}


@app.delete("/api/threads/{thread_id}")
async def delete_thread(thread_id: str):
    session = _require_session()
    if thread_id not in session.thread_list:
        raise HTTPException(status_code=404, detail="Thread not found.")
    if len(session.thread_list) <= 1:
        raise HTTPException(status_code=400, detail="Cannot delete the only thread.")
    _delete_thread_from_db(thread_id)
    session.thread_list.remove(thread_id)
    if session.current_thread_id == thread_id:
        session.current_thread_id = session.thread_list[-1]
        session.current_mode = get_thread_mode(session.current_thread_id)
    return {"deleted": thread_id, "current_thread_id": session.current_thread_id}


@app.delete("/api/threads")
async def delete_empty_threads():
    session = _require_session()
    deleted = []
    for tid in list(session.thread_list):
        if tid == session.current_thread_id:
            continue
        if not _load_conversation(session, tid):
            _delete_thread_from_db(tid)
            session.thread_list.remove(tid)
            deleted.append(tid)
    return {"deleted": deleted, "count": len(deleted)}


# ---------------------------------------------------------------------------
# Routes — Chat
# ---------------------------------------------------------------------------
@app.post("/api/chat")
async def chat(req: ChatRequest):
    session = _require_session()
    thread_id = req.thread_id or session.current_thread_id

    if thread_id not in session.thread_list:
        session.thread_list.append(thread_id)

    graph = _get_graph_for_thread(session, thread_id)
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = graph.invoke({"question": req.message}, config=config)
        response = result.get("generation", "Sorry, I couldn't generate a response.")
    except Exception as exc:
        logger.exception("Chat invoke failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    session.current_thread_id = thread_id
    return {
        "thread_id": thread_id,
        "response": response,
        "mode": get_thread_mode(thread_id).value,
    }


@app.get("/api/chat/{thread_id}/history")
async def chat_history(thread_id: str):
    session = _require_session()
    messages = _load_conversation(session, thread_id)
    return {"thread_id": thread_id, "messages": messages}


# ---------------------------------------------------------------------------
# Routes — Documents (RAG)
# ---------------------------------------------------------------------------
@app.post("/api/documents/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    """Save uploaded PDFs to temp files, build the RAG chain, then clean up."""
    import tempfile

    session = _require_session()

    # (original_name, tmp_path) pairs — kept alive until chain is built
    uploads: list[tuple[str, str]] = []

    try:
        for upload in files:
            if not (upload.filename or "").lower().endswith(".pdf"):
                raise HTTPException(
                    status_code=400,
                    detail=f"{upload.filename!r} is not a PDF.",
                )
            content = await upload.read()
            if not content:
                raise HTTPException(
                    status_code=400,
                    detail=f"{upload.filename!r} is empty.",
                )
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(content)
            tmp.close()
            uploads.append((upload.filename, tmp.name))
            logger.info("Saved upload %r → %s (%d bytes)", upload.filename, tmp.name, len(content))

        # Pass the real on-disk paths; agentic_rag will open them with PyPDFLoader
        tmp_paths = [path for _, path in uploads]
        saved     = [name for name, _ in uploads]

        session.rag_graph = create_rag_agent(
            session.groq_api_key,
            session.model_name,
            tmp_paths,
            session.tavily_api_key,
        )
        session.current_mode = Mode.RAG

        for name in saved:
            if name not in session.uploaded_docs:
                session.uploaded_docs.append(name)

        tid = _create_thread_id(Mode.RAG)
        session.current_thread_id = tid
        if tid not in session.thread_list:
            session.thread_list.append(tid)

        return {"uploaded": saved, "thread_id": tid, "mode": Mode.RAG.value}

    finally:
        # Always remove temp files, even if an error occurred
        for _, path in uploads:
            try:
                os.unlink(path)
            except OSError:
                pass


@app.get("/api/documents")
async def list_documents():
    session = _require_session()
    return {"documents": session.uploaded_docs}


@app.delete("/api/documents")
async def clear_documents():
    session = _require_session()
    session.rag_graph = None
    gc.collect()
    if os.path.exists(VECTOR_STORE_DIR):
        shutil.rmtree(VECTOR_STORE_DIR)
    session.uploaded_docs = []
    session.current_mode = Mode.SEARCH

    # Switch to a new search thread
    tid = _create_thread_id(Mode.SEARCH)
    session.current_thread_id = tid
    if tid not in session.thread_list:
        session.thread_list.append(tid)

    return {"cleared": True, "thread_id": tid}


# ---------------------------------------------------------------------------
# Serve static frontend
# ---------------------------------------------------------------------------
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
