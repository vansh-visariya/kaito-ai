"""Kaito-AI — Streamlit chatbot with web search and document analysis (RAG).

This is the main entry-point.  It renders the Streamlit UI and orchestrates
the two operational modes via :class:`GraphManager`.
"""

# ---------------------------------------------------------------------------
# pysqlite3 shim — required **only** on Streamlit Cloud (Linux).
# On Windows / macOS the stdlib sqlite3 works out of the box.
# ---------------------------------------------------------------------------
import sys

if sys.platform == "linux":
    try:
        __import__("pysqlite3")
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# Protobuf compatibility fix — MUST be set before any chromadb import.
#
# chromadb bundles opentelemetry-proto whose _pb2.py files were compiled
# with an old protoc. The pure-Python protobuf implementation handles these
# gracefully without the "Descriptors cannot be created directly" TypeError.
# Setting it here (before every other import) guarantees correct ordering
# even if Streamlit's module cache loads chromadb through another path.
# ---------------------------------------------------------------------------
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import gc
import logging
import shutil
from typing import Optional

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from agent.search import model_create
from config import (
    Mode,
    THREAD_PREFIX,
    VECTOR_STORE_DIR,
    configure_environment,
    get_thread_mode,
)
from database.memory import get_rag_memory, get_search_memory
from rag.agentic_rag import create_rag_chain
from utility import generate_unique_id, get_memory_for_mode, validate_groq_key

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ChatBot",
    page_icon="🤖",
    layout="wide",
)
st.title("🤖 ChatBot")


# ---------------------------------------------------------------------------
# GraphManager — lazy-initialised graph registry
# ---------------------------------------------------------------------------
class GraphManager:
    """Manages LangGraph workflows for both search and RAG modes.

    Lazily creates each graph on first access and caches it for the
    lifetime of the Streamlit session.
    """

    def __init__(
        self,
        groq_api_key: str,
        model_name: str,
        tavily_api_key: str,
    ) -> None:
        self.groq_api_key = groq_api_key
        self.model_name = model_name
        self.tavily_api_key = tavily_api_key
        self.search_graph = None
        self.rag_graph = None
        self.current_mode: Mode = Mode.SEARCH
        self.uploaded_docs: list[str] = []

    # -- accessors ---------------------------------------------------------

    def get_search_graph(self):
        """Return the search-agent graph, creating it if needed."""
        if not self.search_graph:
            self.search_graph = model_create(
                self.groq_api_key, self.model_name, self.tavily_api_key,
            )
        return self.search_graph

    def get_rag_graph(self, documents=None):
        """Return the RAG-agent graph, (re)creating it when *documents* are provided."""
        if documents:
            self.uploaded_docs = documents
            self.rag_graph = create_rag_chain(
                self.groq_api_key, self.model_name, documents, self.tavily_api_key,
            )
        return self.rag_graph

    def get_graph_for_thread(self, thread_id: str):
        """Select the correct graph for a given *thread_id*."""
        mode = get_thread_mode(thread_id)
        if mode == Mode.RAG:
            if not self.rag_graph:
                st.error("RAG graph not available — please upload documents first.")
                return None
            return self.rag_graph
        return self.get_search_graph()

    def set_mode(self, mode: Mode) -> None:
        """Set the active operational mode."""
        self.current_mode = mode


# ---------------------------------------------------------------------------
# Sidebar — API key configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🔑 Configuration")
    groq_api_key = st.text_input("Enter your Groq API key", type="password")
    model_name = st.text_input("Enter the model you want to use", value="llama-3.1-8b-instant")
    langchain_api_key = st.text_input("Enter your LangChain API key", type="password")
    tavily_api_key = st.text_input("Enter your Tavily API key", type="password")

# Set env vars centrally
configure_environment(
    groq_api_key=groq_api_key or "",
    tavily_api_key=tavily_api_key or "",
    langchain_api_key=langchain_api_key or None,
)

# Validate & initialise
if groq_api_key and validate_groq_key(groq_api_key):
    if "graph_manager" not in st.session_state:
        st.session_state.graph_manager = GraphManager(
            groq_api_key, model_name, tavily_api_key,
        )
    graph_manager: GraphManager = st.session_state.graph_manager
else:
    st.error("Invalid Groq API key. Please check and try again.")
    st.stop()


# ---------------------------------------------------------------------------
# Thread helpers
# ---------------------------------------------------------------------------
def create_thread_id(mode: Mode) -> str:
    """Create a new thread ID prefixed with the mode."""
    return f"{THREAD_PREFIX[mode]}{generate_unique_id()}"


def delete_thread(thread_id: str) -> bool:
    """Delete a thread's checkpoints from the appropriate database."""
    mode = get_thread_mode(thread_id)
    memory = get_memory_for_mode(mode)
    try:
        cursor = memory.conn.cursor()
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        memory.conn.commit()
        return True
    except Exception as exc:
        st.error(f"Error deleting thread: {exc}")
        logger.exception("Failed to delete thread %s", thread_id)
        return False


def load_conversation(thread_id: str) -> list[dict]:
    """Load and format a thread's conversation for display."""
    graph = st.session_state.graph_manager.get_graph_for_thread(thread_id)
    if not graph:
        return []

    state = graph.get_state(config={"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])

    display: list[dict] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            display.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            display.append({"role": "assistant", "content": msg.content})
    return display


def retrieve_all_threads() -> list[str]:
    """Collect every thread ID across both databases."""
    threads: set[str] = set()
    for mem in (get_search_memory(), get_rag_memory()):
        for checkpoint in mem.list(None):
            threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(threads)


def get_thread_preview(thread_id: str) -> str:
    """Return a human-friendly label for the thread selector."""
    mode = get_thread_mode(thread_id)
    icon = "📄" if mode == Mode.RAG else "🔍"

    messages = load_conversation(thread_id)
    if messages:
        content = messages[0]["content"]
        preview = content[:30] + "..." if len(content) > 30 else content
        return f"{icon} {preview}"
    return f"{icon} Thread {thread_id[-8:]}"


# ---------------------------------------------------------------------------
# Session state bootstrap
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_list" not in st.session_state:
    st.session_state.thread_list = retrieve_all_threads()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = create_thread_id(Mode.SEARCH)

if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []


# ---------------------------------------------------------------------------
# Document management
# ---------------------------------------------------------------------------
def delete_document_from_storage() -> bool:
    """Clear the ChromaDB vector store and invalidate the RAG graph."""
    try:
        st.session_state.graph_manager.rag_graph = None
        gc.collect()

        if os.path.exists(VECTOR_STORE_DIR):
            shutil.rmtree(VECTOR_STORE_DIR)

        if st.session_state.uploaded_documents:
            st.warning("Vector store cleared. Please re-upload remaining documents.")
            st.session_state.uploaded_documents = []

        return True
    except Exception as exc:
        st.error(f"Error deleting from storage: {exc}")
        logger.exception("Vector store deletion failed")
        return False


def clear_all_documents() -> None:
    """Remove all documents and switch to search mode if necessary."""
    try:
        delete_document_from_storage()

        if get_thread_mode(st.session_state.thread_id) == Mode.RAG:
            new_id = create_thread_id(Mode.SEARCH)
            st.session_state.thread_id = new_id
            st.session_state.messages = []
            if new_id not in st.session_state.thread_list:
                st.session_state.thread_list.append(new_id)

        st.success("✅ All documents cleared")
    except Exception as exc:
        st.error(f"❌ Error clearing documents: {exc}")
        logger.exception("clear_all_documents failed")


# ---------------------------------------------------------------------------
# Sidebar — document management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Document Management")

    if st.session_state.uploaded_documents:
        st.write("**Uploaded Documents:**")
        for idx, name in enumerate(st.session_state.uploaded_documents, 1):
            st.write(f"📄{idx} {name}")

        if st.button("🗑️ Clear All Documents"):
            clear_all_documents()
            st.rerun()
    else:
        st.info("No documents uploaded yet")


# ---------------------------------------------------------------------------
# Sidebar — thread management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("💬 Thread Management")

    mode_icon = "📄" if graph_manager.current_mode == Mode.RAG else "🔍"
    st.info(f"Current Mode: {mode_icon} {graph_manager.current_mode.value.upper()}")

    # New chat
    if st.button("🆕 New Chat"):
        graph_manager.set_mode(Mode.SEARCH)
        new_id = create_thread_id(Mode.SEARCH)
        st.session_state.thread_id = new_id
        st.session_state.messages = []
        if new_id not in st.session_state.thread_list:
            st.session_state.thread_list.append(new_id)
        st.rerun()

    # Thread selector
    if st.session_state.thread_list:
        col1, col2 = st.columns([3, 1])

        with col1:
            reversed_list = st.session_state.thread_list[::-1]
            current_index = 0
            if st.session_state.thread_id in reversed_list:
                current_index = reversed_list.index(st.session_state.thread_id)

            selected_thread = st.selectbox(
                "Select conversation",
                options=reversed_list,
                format_func=get_thread_preview,
                index=current_index,
                key="thread_selector",
            )

        with col2:
            if st.button("🗑️", help="Delete selected thread"):
                if len(st.session_state.thread_list) > 1:
                    if delete_thread(selected_thread):
                        st.session_state.thread_list.remove(selected_thread)

                        if selected_thread == st.session_state.thread_id:
                            remaining = [
                                t for t in st.session_state.thread_list
                                if t != selected_thread
                            ]
                            if remaining:
                                new_current = remaining[-1]
                                st.session_state.thread_id = new_current
                                st.session_state.messages = load_conversation(new_current)
                                graph_manager.current_mode = get_thread_mode(new_current)

                        st.success("✅ Thread deleted!")
                        st.rerun()
                else:
                    st.error("❌ Cannot delete the only remaining thread")

        # Switch thread
        if selected_thread != st.session_state.thread_id:
            st.session_state.thread_id = selected_thread
            st.session_state.messages = load_conversation(selected_thread)
            graph_manager.current_mode = get_thread_mode(selected_thread)
            st.rerun()


# ---------------------------------------------------------------------------
# Sidebar — thread cleanup
# ---------------------------------------------------------------------------
with st.sidebar:
    with st.expander("🧹 Thread Cleanup"):
        st.write(f"Total threads: {len(st.session_state.thread_list)}")

        if st.button("Delete All Empty Threads"):
            # Build list of IDs to delete *before* mutating the list
            to_delete = [
                tid for tid in st.session_state.thread_list
                if tid != st.session_state.thread_id and not load_conversation(tid)
            ]
            for tid in to_delete:
                if delete_thread(tid):
                    st.session_state.thread_list.remove(tid)

            if to_delete:
                st.success(f"✅ Deleted {len(to_delete)} empty thread(s)")
                st.rerun()


# ---------------------------------------------------------------------------
# Chat display
# ---------------------------------------------------------------------------
messages = load_conversation(st.session_state.thread_id)
if messages not in st.session_state.messages:
    st.session_state.messages = messages
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])


# ---------------------------------------------------------------------------
# Chat input handling
# ---------------------------------------------------------------------------
user_input: Optional[str] = None
chat_input = st.chat_input("Enter your query", accept_file=True, file_type=["pdf"])

# File upload → RAG mode
if chat_input and chat_input.get("files"):
    documents = chat_input["files"]

    for doc in documents:
        if doc.name not in st.session_state.uploaded_documents:
            st.session_state.uploaded_documents.append(doc.name)

    graph_manager.get_rag_graph(documents)
    graph_manager.set_mode(Mode.RAG)

    new_id = create_thread_id(Mode.RAG)
    st.session_state.thread_id = new_id
    st.session_state.messages = []
    if new_id not in st.session_state.thread_list:
        st.session_state.thread_list.append(new_id)

    st.success(f"✅ Uploaded {len(documents)} document(s). Started new RAG chat.")
    st.rerun()

# Text input
if chat_input and chat_input.text:
    user_input = chat_input.text

if user_input:
    if st.session_state.thread_id not in st.session_state.thread_list:
        st.session_state.thread_list.append(st.session_state.thread_id)

    current_graph = graph_manager.get_graph_for_thread(st.session_state.thread_id)
    if not current_graph:
        st.error("Graph not available for this thread type.")
        st.stop()

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            result = current_graph.invoke({"question": user_input}, config=config)
            response = result.get("generation", "Sorry, I couldn't generate a response.")
            st.markdown(response)

            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": response})

    st.rerun()
