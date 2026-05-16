"""Kaito-AI — FastAPI backend.

Replaces the Streamlit app.py with a proper REST API that serves the
static HTML/CSS/JS frontend and handles all chatbot logic.

Improvements in this version:
#1 Streaming Responses (SSE) via /api/chat/stream
#4 Multi-User Session Support via Cookies

Run with:
    uvicorn api:app --reload --port 8000
"""

import asyncio
import gc
import json
import logging
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Optional

# pysqlite3 shim — Linux only
if sys.platform == "linux":
    try:
        __import__("pysqlite3")
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    except ImportError:
        pass

# Protobuf fix — MUST be before any chromadb import
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from fastapi import FastAPI, HTTPException, UploadFile, File, Cookie, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
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
from utility import generate_unique_id, get_memory_for_mode, validate_groq_key

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)

# App
app = FastAPI(title="Kaito-AI API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Multi-User Session Store (#4)
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


SESSIONS: dict[str, Session] = {}


def get_session(session_id: Optional[str] = Cookie(default=None)) -> Session:
    """FastAPI Dependency to get the current user's session."""
    if not session_id or session_id not in SESSIONS:
        raise HTTPException(status_code=401, detail="No session found. Please configure API keys first.")
    session = SESSIONS[session_id]
    if not session.is_configured():
        raise HTTPException(status_code=401, detail="Session not configured.")
    return session


# Pydantic schemas
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


# Helpers
def _create_thread_id(mode: Mode) -> str:
    return f"{THREAD_PREFIX[mode]}{generate_unique_id()}"


async def _get_search_graph(session: Session):
    if not session.search_graph:
        session.search_graph = await create_search_agent(
            session.groq_api_key, session.model_name, session.tavily_api_key
        )
    return session.search_graph


async def _get_graph_for_thread(session: Session, thread_id: str):
    mode = get_thread_mode(thread_id)
    if mode == Mode.RAG:
        if not session.rag_graph:
            raise HTTPException(status_code=400, detail="RAG graph not available. Upload documents first.")
        return session.rag_graph
    return await _get_search_graph(session)


async def _load_conversation(session: Session, thread_id: str) -> list[dict]:
    try:
        graph = await _get_graph_for_thread(session, thread_id)
    except HTTPException:
        return []
    state = await graph.aget_state(config={"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])
    result = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})
    return result


async def _delete_thread_from_db(thread_id: str) -> bool:
    mode = get_thread_mode(thread_id)
    memory = await get_memory_for_mode(mode)
    try:
        async with memory.conn.cursor() as cursor:
            await cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        await memory.conn.commit()
        return True
    except Exception as exc:
        logger.exception("Failed to delete thread %s: %s", thread_id, exc)
        return False


async def _thread_preview(session: Session, thread_id: str) -> str:
    mode = get_thread_mode(thread_id)
    icon = "📄" if mode == Mode.RAG else "🔍"
    messages = await _load_conversation(session, thread_id)
    if messages:
        content = messages[0]["content"]
        preview = content[:40] + "..." if len(content) > 40 else content
        return f"{icon} {preview}"
    return f"{icon} Thread {thread_id[-8:]}"


# Routes — Config
@app.post("/api/config")
async def configure(req: ConfigRequest, response: Response):
    """Set API keys and model. Creates a new session and returns a cookie."""
    if not validate_groq_key(req.groq_api_key):
        raise HTTPException(status_code=400, detail="Invalid Groq API key.")

    session_id = str(uuid.uuid4())
    session = Session()

    session.groq_api_key = req.groq_api_key
    session.model_name = req.model_name
    session.tavily_api_key = req.tavily_api_key
    session.langchain_api_key = req.langchain_api_key

    # Save session
    SESSIONS[session_id] = session

    configure_environment(
        groq_api_key=req.groq_api_key,
        tavily_api_key=req.tavily_api_key,
        langchain_api_key=req.langchain_api_key,
    )

    # Bootstrap a default thread
    tid = _create_thread_id(Mode.SEARCH)
    session.thread_list.append(tid)
    session.current_thread_id = tid

    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        samesite="lax",
    )

    return {
        "status": "ok",
        "model": req.model_name,
        "current_thread_id": session.current_thread_id,
    }


@app.get("/api/config/status")
async def config_status(session_id: Optional[str] = Cookie(default=None)):
    """Check whether the session is configured."""
    if not session_id or session_id not in SESSIONS:
        raise HTTPException(status_code=401, detail="No session")
    session = SESSIONS[session_id]
    if not session.is_configured():
        raise HTTPException(status_code=401, detail="Session not configured")

    return {
        "configured": True,
        "model": session.model_name,
        "current_thread_id": session.current_thread_id,
        "mode": session.current_mode.value,
    }


# Routes — Threads
@app.get("/api/threads")
async def list_threads(session: Session = Depends(get_session)):
    threads = []
    for tid in reversed(session.thread_list):
        threads.append({
            "id": tid,
            "preview": await _thread_preview(session, tid),
            "mode": get_thread_mode(tid).value,
            "active": tid == session.current_thread_id,
        })
    return {"threads": threads}


@app.post("/api/threads/new")
async def new_thread(session: Session = Depends(get_session)):
    tid = _create_thread_id(Mode.SEARCH)
    session.current_thread_id = tid
    session.current_mode = Mode.SEARCH
    if tid not in session.thread_list:
        session.thread_list.append(tid)
    return {"thread_id": tid, "mode": Mode.SEARCH.value}


@app.post("/api/threads/select")
async def select_thread(req: ThreadDeleteRequest, session: Session = Depends(get_session)):
    thread_id = req.thread_id
    if thread_id not in session.thread_list:
        raise HTTPException(status_code=404, detail="Thread not found.")
    session.current_thread_id = thread_id
    session.current_mode = get_thread_mode(thread_id)
    messages = await _load_conversation(session, thread_id)
    return {"thread_id": thread_id, "messages": messages, "mode": session.current_mode.value}


@app.delete("/api/threads/{thread_id}")
async def delete_thread(thread_id: str, session: Session = Depends(get_session)):
    if thread_id not in session.thread_list:
        raise HTTPException(status_code=404, detail="Thread not found.")
    if len(session.thread_list) <= 1:
        raise HTTPException(status_code=400, detail="Cannot delete the only thread.")
    await _delete_thread_from_db(thread_id)
    session.thread_list.remove(thread_id)
    if session.current_thread_id == thread_id:
        session.current_thread_id = session.thread_list[-1]
        session.current_mode = get_thread_mode(session.current_thread_id)
    return {"deleted": thread_id, "current_thread_id": session.current_thread_id}


@app.delete("/api/threads")
async def delete_empty_threads(session: Session = Depends(get_session)):
    deleted = []
    for tid in list(session.thread_list):
        if tid == session.current_thread_id:
            continue
        if not await _load_conversation(session, tid):
            await _delete_thread_from_db(tid)
            session.thread_list.remove(tid)
            deleted.append(tid)
    return {"deleted": deleted, "count": len(deleted)}


# Routes — Chat
@app.post("/api/chat")
async def chat(req: ChatRequest, session: Session = Depends(get_session)):
    """Blocking chat — returns full response in one JSON object."""
    thread_id = req.thread_id or session.current_thread_id

    if thread_id not in session.thread_list:
        session.thread_list.append(thread_id)

    graph = await _get_graph_for_thread(session, thread_id)
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = await graph.ainvoke({"question": req.message}, config=config)
        response = result.get("generation", "Sorry, I couldn't generate a response.")
        sources  = result.get("sources", [])
    except Exception as exc:
        logger.exception("Chat invoke failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    session.current_thread_id = thread_id
    return {
        "thread_id": thread_id,
        "response": response,
        "sources": sources,
        "mode": get_thread_mode(thread_id).value,
    }


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, session: Session = Depends(get_session)):
    """SSE streaming chat — sends tokens as they are generated."""
    thread_id = req.thread_id or session.current_thread_id

    if thread_id not in session.thread_list:
        session.thread_list.append(thread_id)

    graph  = await _get_graph_for_thread(session, thread_id)
    config = {"configurable": {"thread_id": thread_id}}

    async def generate():
        sources: list[dict] = []
        seen_sources: set[tuple] = set()

        try:
            async for event in graph.astream_events(
                {"question": req.message}, config=config
            ):
                kind = event["event"]

                # —— Stream final-answer tokens only ———————————————
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content and not chunk.tool_call_chunks:
                        token = chunk.content if isinstance(chunk.content, str) else ""
                        if token:
                            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

                # —— Capture document citations from retriever tool ———————
                elif kind == "on_tool_end":
                    if event.get("name") == "document_retriever":
                        output = event["data"].get("output")
                        docs = output[1] if isinstance(output, tuple) else []
                        for doc in docs:
                            if not hasattr(doc, "metadata"):
                                continue
                            raw  = doc.metadata.get("source", "")
                            page = doc.metadata.get("page", 0)
                            file = Path(raw).name if raw else ""
                            key  = (file, page)
                            if file and key not in seen_sources:
                                seen_sources.add(key)
                                sources.append({"file": file, "page": page + 1})

        except Exception as exc:
            logger.exception("SSE stream error: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
            return

        session.current_thread_id = thread_id
        yield f"data: {json.dumps({'type': 'done', 'thread_id': thread_id, 'mode': get_thread_mode(thread_id).value, 'sources': sources})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable Nginx buffering
        },
    )


@app.get("/api/chat/{thread_id}/history")
async def chat_history(thread_id: str, session: Session = Depends(get_session)):
    messages = await _load_conversation(session, thread_id)
    return {"thread_id": thread_id, "messages": messages}


# Routes — Documents (RAG)
@app.post("/api/documents/upload")
async def upload_documents(files: list[UploadFile] = File(...), session: Session = Depends(get_session)):
    """Save uploaded PDFs to temp files, build the RAG chain, then clean up."""
    import tempfile

    uploads: list[tuple[str, str]] = []

    try:
        for upload in files:
            if not (upload.filename or "").lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail=f"{upload.filename!r} is not a PDF.")
            content = await upload.read()
            if not content:
                raise HTTPException(status_code=400, detail=f"{upload.filename!r} is empty.")
            
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(content)
            tmp.close()
            uploads.append((upload.filename, tmp.name))
            logger.info("Saved upload %r → %s (%d bytes)", upload.filename, tmp.name, len(content))

        tmp_paths = [path for _, path in uploads]
        saved     = [name for name, _ in uploads]

        session.rag_graph = await create_rag_agent(
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
        for _, path in uploads:
            try:
                os.unlink(path)
            except OSError:
                pass


@app.get("/api/documents")
async def list_documents(session: Session = Depends(get_session)):
    return {"documents": session.uploaded_docs}


@app.delete("/api/documents")
async def clear_documents(session: Session = Depends(get_session)):
    session.rag_graph = None
    gc.collect()
    # In a multi-user environment, wiping the entire VECTOR_STORE_DIR breaks 
    # other sessions that are using it! For production this needs to be 
    # a per-session directory. For MVP, we will still wipe it to clear space.
    if os.path.exists(VECTOR_STORE_DIR):
        shutil.rmtree(VECTOR_STORE_DIR)
    session.uploaded_docs = []
    session.current_mode = Mode.SEARCH

    tid = _create_thread_id(Mode.SEARCH)
    session.current_thread_id = tid
    if tid not in session.thread_list:
        session.thread_list.append(tid)

    return {"cleared": True, "thread_id": tid}


# Serve static frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
