"""Kaito-AI — FastAPI backend.
Run with:
    uvicorn api:app --reload --port 8000
"""

import gc
import json
import logging
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Annotated, Optional, TypedDict

# pysqlite3 shim — Linux only
if sys.platform == "linux":
    try:
        __import__("pysqlite3")
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    except ImportError:
        pass

# Protobuf fix — MUST be before any chromadb import
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from fastapi import Cookie, Depends, FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from agent.agent import create_agent
from config import (
    DEFAULT_MODEL,
    VECTOR_STORE_DIR,
    SESSIONS_FILE_PATH,
    UPLOADS_DIR_PATH,
    configure_environment,
)
from database.memory import get_memory
from database.users import check_rate_limit, create_session, increment_tokens, login, logout, register, validate_session
from utility import generate_unique_id

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configure server environment (API keys, LangSmith) ────────────────────
configure_environment()

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Kaito-AI API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════
# SESSION MODEL
# ══════════════════════════════════════════════════════════════════════════
class Session:
    def __init__(self, user_id: int, username: str, email: str) -> None:
        self.user_id = user_id
        self.username = username
        self.email = email
        self.model_name: str = DEFAULT_MODEL
        self.graph = None  # single unified agent
        self.uploaded_docs: list[str] = []
        self.thread_list: list[str] = []
        self.current_thread_id: str = ""


SESSIONS: dict[int, Session] = {}  # keyed by user_id

SESSIONS_FILE = Path(SESSIONS_FILE_PATH)
SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)


def save_sessions():
    data = {}
    for uid, s in SESSIONS.items():
        data[str(uid)] = {
            "username": s.username,
            "email": s.email,
            "thread_list": s.thread_list,
            "current_thread_id": s.current_thread_id,
            "uploaded_docs": s.uploaded_docs,
            "model_name": s.model_name,
        }
    SESSIONS_FILE.parent.mkdir(exist_ok=True)
    SESSIONS_FILE.write_text(json.dumps(data))


def load_sessions():
    if not SESSIONS_FILE.exists():
        return
    try:
        data = json.loads(SESSIONS_FILE.read_text())
        for uid_str, sdata in data.items():
            uid = int(uid_str)
            s = Session(
                user_id=uid,
                username=sdata.get("username", ""),
                email=sdata.get("email", ""),
            )
            s.thread_list = sdata.get("thread_list", [])
            s.current_thread_id = sdata.get("current_thread_id", "")
            s.uploaded_docs = sdata.get("uploaded_docs", [])
            s.model_name = sdata.get("model_name", DEFAULT_MODEL)
            SESSIONS[uid] = s
        logger.info("Loaded %d sessions from disk.", len(SESSIONS))
    except Exception as exc:
        logger.error("Failed to load sessions: %s", exc)


load_sessions()


def _get_or_create_session(user_id: int, username: str, email: str) -> Session:
    """Get existing session or create a new one for the user."""
    if user_id in SESSIONS:
        session = SESSIONS[user_id]
        # Update username/email in case they changed
        session.username = username
        session.email = email
        return session
    session = Session(user_id, username, email)
    SESSIONS[user_id] = session
    return session


def get_session(session_token: Optional[str] = Cookie(default=None)) -> Session:
    """FastAPI Dependency: validate session token and return Session."""
    if not session_token:
        raise HTTPException(status_code=401, detail="Not authenticated. Please log in.")

    user = validate_session(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")

    return _get_or_create_session(user["id"], user["username"], user["email"])


# ══════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ══════════════════════════════════════════════════════════════════════════
class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


class ThreadDeleteRequest(BaseModel):
    thread_id: str


class ThreadBranchRequest(BaseModel):
    thread_id: str
    edit_index: int


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════
def _create_thread_id() -> str:
    return f"thread_{generate_unique_id()}"


async def _get_or_build_graph(session: Session):
    """Get or build the unified agent graph for the session."""
    if session.graph:
        return session.graph

    session_uploads_dir = UPLOADS_DIR / str(session.user_id)
    file_paths = []
    if session_uploads_dir.exists():
        file_paths = [str(p.absolute()) for p in session_uploads_dir.glob("*.pdf")]

    if file_paths:
        session.graph = await create_agent(
            session.model_name,
            session.user_id,
            file_paths,
        )
    else:
        session.graph = await create_agent(session.model_name, session.user_id)

    return session.graph



class _DummyState(TypedDict):
    messages: Annotated[list, add_messages]


async def _read_thread_state(thread_id: str) -> list:
    memory = await get_memory()

    builder = StateGraph(_DummyState)
    builder.add_node("dummy", lambda x: x)
    builder.set_entry_point("dummy")
    graph = builder.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": thread_id}}
    state = await graph.aget_state(config)
    return state.values.get("messages", [])


async def _load_conversation(thread_id: str) -> list[dict]:
    try:
        messages = await _read_thread_state(thread_id)
    except Exception as exc:
        logger.error("Failed to read thread state for %s: %s", thread_id, exc)
        return []

    result = []
    pending_sources = []
    from langchain_core.messages import ToolMessage
    from agent.agent import _extract_sources

    for msg in messages:
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content, "id": getattr(msg, "id", None)})
            pending_sources = []
        elif isinstance(msg, ToolMessage):
            # Extract sources and keep them for the next AI message
            extracted = _extract_sources([msg])
            for s in extracted:
                if s not in pending_sources:
                    pending_sources.append(s)
        elif isinstance(msg, AIMessage):
            result.append({
                "role": "assistant", 
                "content": msg.content, 
                "id": getattr(msg, "id", None),
                "sources": pending_sources
            })
            pending_sources = []
    return result


async def _delete_thread_from_db(thread_id: str) -> bool:
    memory = await get_memory()
    try:
        async with memory.conn.cursor() as cursor:
            await cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        await memory.conn.commit()
        return True
    except Exception as exc:
        logger.exception("Failed to delete thread %s: %s", thread_id, exc)
        return False


async def _thread_preview(thread_id: str) -> str:
    messages = await _load_conversation(thread_id)
    if messages:
        content = messages[0]["content"]
        preview = content[:40] + "..." if len(content) > 40 else content
        return f"💬 {preview}"
    return f"💬 Thread {thread_id[-8:]}"


# ══════════════════════════════════════════════════════════════════════════
# ROUTES — AUTH
# ══════════════════════════════════════════════════════════════════════════
@app.post("/api/auth/register")
async def auth_register(req: RegisterRequest, response: Response):
    """Register a new user, auto-login, and return session cookie."""
    try:
        register(req.email, req.password, req.username)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Auto-login after registration
    user = login(req.email, req.password)
    if not user:
        raise HTTPException(status_code=500, detail="Registration succeeded but login failed.")

    token = create_session(user["id"])
    session = _get_or_create_session(user["id"], user["username"], user["email"])

    # Bootstrap a default thread if none exists
    if not session.thread_list:
        tid = _create_thread_id()
        session.thread_list.append(tid)
        session.current_thread_id = tid

    save_sessions()

    response.set_cookie(
        key="session_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=7 * 86400,  # 7 days
    )

    return {
        "status": "ok",
        "username": user["username"],
        "current_thread_id": session.current_thread_id,
    }


@app.post("/api/auth/login")
async def auth_login(req: LoginRequest, response: Response):
    """Login with email and password."""
    user = login(req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = create_session(user["id"])
    session = _get_or_create_session(user["id"], user["username"], user["email"])

    # Bootstrap a default thread if none exists
    if not session.thread_list:
        tid = _create_thread_id()
        session.thread_list.append(tid)
        session.current_thread_id = tid

    save_sessions()

    response.set_cookie(
        key="session_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=7 * 86400,  # 7 days
    )

    return {
        "status": "ok",
        "username": user["username"],
        "current_thread_id": session.current_thread_id,
    }


@app.post("/api/auth/logout")
async def auth_logout(
    response: Response,
    session_token: Optional[str] = Cookie(default=None),
):
    """Logout: clear cookie and delete session token."""
    if session_token:
        logout(session_token)
    response.delete_cookie(key="session_token")
    return {"status": "ok"}


@app.get("/api/auth/status")
async def auth_status(session_token: Optional[str] = Cookie(default=None)):
    """Check whether the user is authenticated."""
    if not session_token:
        raise HTTPException(status_code=401, detail="No session")

    user = validate_session(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Session expired")

    session = _get_or_create_session(user["id"], user["username"], user["email"])

    has_docs = len(session.uploaded_docs) > 0

    return {
        "authenticated": True,
        "username": user["username"],
        "current_thread_id": session.current_thread_id,
        "has_documents": has_docs,
    }


# ══════════════════════════════════════════════════════════════════════════
# ROUTES — THREADS
# ══════════════════════════════════════════════════════════════════════════
@app.get("/api/threads")
async def list_threads(session: Session = Depends(get_session)):
    threads = []
    for tid in reversed(session.thread_list):
        threads.append({
            "id": tid,
            "preview": await _thread_preview(tid),
            "active": tid == session.current_thread_id,
        })
    return {"threads": threads}


@app.post("/api/threads/new")
async def new_thread(session: Session = Depends(get_session)):
    tid = _create_thread_id()
    session.current_thread_id = tid
    if tid not in session.thread_list:
        session.thread_list.append(tid)
    save_sessions()
    return {"thread_id": tid}


@app.post("/api/threads/select")
async def select_thread(req: ThreadDeleteRequest, session: Session = Depends(get_session)):
    thread_id = req.thread_id
    if thread_id not in session.thread_list:
        raise HTTPException(status_code=404, detail="Thread not found.")
    session.current_thread_id = thread_id
    save_sessions()
    messages = await _load_conversation(thread_id)
    return {"thread_id": thread_id, "messages": messages}


@app.delete("/api/threads/{thread_id}")
async def delete_thread(thread_id: str, session: Session = Depends(get_session)):
    if thread_id not in session.thread_list:
        raise HTTPException(status_code=404, detail="Thread not found.")
    if len(session.thread_list) <= 1:
        raise HTTPException(status_code=400, detail="Cannot delete the only thread.")
    await _delete_thread_from_db(thread_id)
    session.thread_list.remove(thread_id)
    if session.current_thread_id == thread_id:
        if session.thread_list:
            session.current_thread_id = session.thread_list[-1]
        else:
            tid = _create_thread_id()
            session.thread_list.append(tid)
            session.current_thread_id = tid
    save_sessions()
    return {"deleted": thread_id, "current_thread_id": session.current_thread_id}


@app.delete("/api/threads")
async def delete_empty_threads(session: Session = Depends(get_session)):
    deleted = []
    for tid in list(session.thread_list):
        if tid == session.current_thread_id:
            continue
        if not await _load_conversation(tid):
            await _delete_thread_from_db(tid)
            session.thread_list.remove(tid)
            deleted.append(tid)
    save_sessions()
    return {"deleted": deleted}


@app.post("/api/threads/branch")
async def branch_thread(req: ThreadBranchRequest, session: Session = Depends(get_session)):
    """Branch a thread at a specific edit point to allow regeneration."""
    try:
        check_rate_limit(session.user_id)
    except ValueError as e:
        raise HTTPException(status_code=429, detail=str(e))

    old_thread_id = req.thread_id
    if old_thread_id not in session.thread_list:
        raise HTTPException(status_code=404, detail="Thread not found.")

    # Read old state
    messages = await _read_thread_state(old_thread_id)
    
    # We want to keep all messages up to `edit_index` (exclusive for the user message we are replacing)
    # The frontend edit_index corresponds to the index of the message in the UI array.
    # Actually, if the frontend just passes `edit_index` as the literal index in the UI's message array,
    # it corresponds directly to the AI/Human messages in our `_load_conversation` output.
    # Let's count them:
    kept_messages = []
    ui_index = 0
    from langchain_core.messages import ToolMessage
    for msg in messages:
        if isinstance(msg, (HumanMessage, AIMessage)):
            if ui_index == req.edit_index:
                break
            ui_index += 1
            kept_messages.append(msg)
        elif isinstance(msg, ToolMessage):
            # Keep tool messages if they belong to the last AIMessage we kept.
            # If we haven't hit edit_index yet, we keep everything.
            kept_messages.append(msg)

    # Create new thread
    new_thread_id = _create_thread_id()
    session.thread_list.append(new_thread_id)
    session.current_thread_id = new_thread_id
    save_sessions()

    # Write truncated history to the new thread
    if kept_messages:
        graph = await _get_or_build_graph(session)
        config = {"configurable": {"thread_id": new_thread_id}}
        await graph.aupdate_state(config, {"messages": kept_messages})

    return {"thread_id": new_thread_id}

# ══════════════════════════════════════════════════════════════════════════
# ROUTES — CHAT
# ══════════════════════════════════════════════════════════════════════════
@app.post("/api/chat")
async def chat(req: ChatRequest, session: Session = Depends(get_session)):
    """Blocking chat — returns full response in one JSON object."""
    try:
        check_rate_limit(session.user_id)
    except ValueError as e:
        raise HTTPException(status_code=429, detail=str(e))

    thread_id = req.thread_id or session.current_thread_id

    if thread_id not in session.thread_list:
        session.thread_list.append(thread_id)

    graph = await _get_or_build_graph(session)
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = await graph.ainvoke({"question": req.message}, config=config)
        response = result.get("generation", "Sorry, I couldn't generate a response.")
        sources  = result.get("sources", [])
        
        # Count tokens
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and hasattr(msg, "response_metadata"):
                tokens = msg.response_metadata.get("token_usage", {}).get("total_tokens", 0)
                if tokens:
                    increment_tokens(session.user_id, tokens)
                break
    except Exception as exc:
        logger.exception("Chat invoke failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    session.current_thread_id = thread_id
    return {
        "thread_id": thread_id,
        "response": response,
        "sources": sources,
    }


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, session: Session = Depends(get_session)):
    """SSE streaming chat — sends tokens as they are generated."""
    try:
        check_rate_limit(session.user_id)
    except ValueError as e:
        raise HTTPException(status_code=429, detail=str(e))

    thread_id = req.thread_id or session.current_thread_id

    if thread_id not in session.thread_list:
        session.thread_list.append(thread_id)

    graph  = await _get_or_build_graph(session)
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

                # —— Track token usage ——————————————————————————————
                elif kind == "on_chat_model_end":
                    msg = event["data"]["output"]
                    if hasattr(msg, "response_metadata"):
                        tokens = msg.response_metadata.get("token_usage", {}).get("total_tokens", 0)
                        if tokens:
                            increment_tokens(session.user_id, tokens)

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
        yield f"data: {json.dumps({'type': 'done', 'thread_id': thread_id, 'sources': sources})}\n\n"

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
    messages = await _load_conversation(thread_id)
    return {"thread_id": thread_id, "messages": messages}


# ══════════════════════════════════════════════════════════════════════════
# ROUTES — DOCUMENTS (RAG)
# ══════════════════════════════════════════════════════════════════════════
UPLOADS_DIR = Path(UPLOADS_DIR_PATH)


@app.post("/api/documents/upload")
async def upload_documents(files: list[UploadFile] = File(...), session: Session = Depends(get_session)):
    """Save uploaded PDFs to session dir, rebuild the unified agent."""
    session_uploads_dir = UPLOADS_DIR / str(session.user_id)
    session_uploads_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    for upload in files:
        if not (upload.filename or "").lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{upload.filename!r} is not a PDF.")
        content = await upload.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"{upload.filename!r} is empty.")

        file_path = session_uploads_dir / upload.filename
        file_path.write_bytes(content)
        saved_files.append(upload.filename)
        logger.info("Saved upload %r → %s (%d bytes)", upload.filename, file_path, len(content))

        # Add to vector store individually
        from agent.agent import add_document_to_vector_store
        add_document_to_vector_store(str(file_path.absolute()), session.user_id)

    all_paths = [str(p.absolute()) for p in session_uploads_dir.glob("*.pdf")]

    # Rebuild the unified agent with document retriever
    session.graph = await create_agent(
        session.model_name,
        session.user_id,
        all_paths,
    )

    for name in saved_files:
        if name not in session.uploaded_docs:
            session.uploaded_docs.append(name)

    save_sessions()

    tid = _create_thread_id()
    session.current_thread_id = tid
    if tid not in session.thread_list:
        session.thread_list.append(tid)

    return {"uploaded": session.uploaded_docs, "thread_id": tid}


@app.delete("/api/documents/{filename}")
async def delete_document(filename: str, session: Session = Depends(get_session)):
    session_uploads_dir = UPLOADS_DIR / str(session.user_id)
    file_path = session_uploads_dir / filename

    if file_path.exists():
        # 1. Delete from vector store
        from agent.agent import delete_document_from_vector_store
        delete_document_from_vector_store(str(file_path.absolute()), session.user_id)

        # 2. Delete file
        file_path.unlink(missing_ok=True)

    if filename in session.uploaded_docs:
        session.uploaded_docs.remove(filename)
        save_sessions()

    all_paths = [str(p.absolute()) for p in session_uploads_dir.glob("*.pdf")]

    if not all_paths:
        # No documents left -> rebuild agent without retriever
        session.graph = await create_agent(session.model_name, session.user_id)
        save_sessions()
        return {"deleted": filename, "has_documents": False}
    else:
        session.graph = await create_agent(
            session.model_name,
            session.user_id,
            all_paths,
        )
        return {"deleted": filename, "has_documents": True}


@app.get("/api/documents")
async def list_documents(session: Session = Depends(get_session)):
    return {"documents": session.uploaded_docs}


@app.delete("/api/documents")
async def clear_documents(session: Session = Depends(get_session)):
    session.graph = None
    gc.collect()

    # Delete this user's chunks from the unified vector store
    from agent.agent import delete_document_from_vector_store
    session_uploads_dir = UPLOADS_DIR / str(session.user_id)
    if session_uploads_dir.exists():
        for pdf in session_uploads_dir.glob("*.pdf"):
            delete_document_from_vector_store(str(pdf.absolute()), session.user_id)
        shutil.rmtree(session_uploads_dir, ignore_errors=True)

    session.uploaded_docs = []
    save_sessions()

    # Rebuild agent without documents
    session.graph = await create_agent(session.model_name, session.user_id)

    tid = _create_thread_id()
    session.current_thread_id = tid
    if tid not in session.thread_list:
        session.thread_list.append(tid)

    return {"cleared": True, "thread_id": tid}


# ══════════════════════════════════════════════════════════════════════════
# SERVE STATIC FRONTEND
# ══════════════════════════════════════════════════════════════════════════
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
