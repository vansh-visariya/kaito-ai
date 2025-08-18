__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
## this is for streamlit deploy

import streamlit as st
from agent.search import model_create
from utility import generate_unique_id, validate_groq_key, get_memory_for_mode
from langchain_core.messages import HumanMessage, AIMessage
from rag.agentic_rag import create_rag_chain
from database.get_sql import get_search_memory, get_rag_memory
import shutil
import os
import gc

st.set_page_config(
    page_title="ChatBot",       # Title on the browser tab
    page_icon="🤖",                    # Icon on the browser tab (emoji or URL)
    layout="wide",                     # "centered" (default) or "wide"
)

st.title("🤖 ChatBot")

## Graph Manager for proper separation
class GraphManager:
    def __init__(self, groq_api_key, model_name, tavily_api_key):
        self.groq_api_key = groq_api_key
        self.model_name = model_name
        self.tavily_api_key = tavily_api_key
        self.search_graph = None
        self.rag_graph = None
        self.current_mode = "search"
        self.uploaded_docs = []
        
    def get_search_graph(self):
        if not self.search_graph:
            self.search_graph = model_create(self.groq_api_key, self.model_name, self.tavily_api_key)
        return self.search_graph
    
    def get_rag_graph(self, documents=None):
        if documents:
            self.uploaded_docs = documents
            self.rag_graph = create_rag_chain(self.groq_api_key, self.model_name, documents, self.tavily_api_key)
        return self.rag_graph
    
    def get_graph_for_thread(self, thread_id):
        mode = get_thread_mode(thread_id)
        if mode == "rag":
            if not self.rag_graph: ## if mode is rag but the rag graph is not present then return none, graph is deleted due to deletion of documents
                st.error("RAG graph not available. Please upload documents first.")
                return None
            return self.rag_graph
        else:
            return self.get_search_graph()
    
    def get_graph_for_mode(self, mode):  ## can delete this function but not sure now
        if mode == "rag" and self.rag_graph:
            return self.rag_graph
        return self.get_search_graph()
    
    def set_mode(self, mode):
        self.current_mode = mode

## User data collection
with st.sidebar:
    st.header("🔑 Configuration")
    groq_api_key = st.text_input("Enter your Groq API key", type="password")
    model_name = st.text_input("Enter the model you want to use", value="gemma2-9b-it")
    langchain_api_key = st.text_input("Enter your LangChain API key", type="password")
    tavily_api_key = st.text_input("Enter your Tavily API key", type="password")

# LangSmith configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_PROJECT"] = "chatbot"

## Validate groq key and initialize graph manager
if groq_api_key and validate_groq_key(groq_api_key):
    if 'graph_manager' not in st.session_state:
        st.session_state.graph_manager = GraphManager(groq_api_key, model_name, tavily_api_key)
    graph_manager = st.session_state.graph_manager
else:
    st.error("Invalid Groq API key. Please check and try again.")
    st.stop()

## Thread Management Functions
def get_thread_mode(thread_id): ## get the mode of the thread
    return "rag" if thread_id.startswith("rag_") else "search"

def create_thread_id(mode):  ## create thread id based on mode
    return f"{mode}_{generate_unique_id()}"

def delete_thread(thread_id):  ## delete thread from the database
    mode = get_thread_mode(thread_id)
    memory = get_memory_for_mode(mode)
    try:
        conn = memory.conn  ## get the connection from the memory
        cursor = conn.cursor()  ## create a cursor for executing SQL statements
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error deleting thread: {e}")
        return False

def load_conversation(thread_id):
    # Get the appropriate graph for this thread
    graph = st.session_state.graph_manager.get_graph_for_thread(thread_id)
    if not graph:
        return []
    
    # Get conversation from graph's memory
    state = graph.get_state(config={'configurable': {'thread_id': thread_id}})
    messages = state.values.get('messages', [])
    
    # Convert LangChain messages to display format
    display_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            display_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            display_messages.append({"role": "assistant", "content": msg.content})
    
    return display_messages

def retrieve_all_threads():
    all_threads = set()
    ## Retrieve all threads from both memories
    search_memory = get_search_memory()
    for m in search_memory.list(None):
        all_threads.add(m.config['configurable']['thread_id'])

    rag_memory = get_rag_memory()
    for m in rag_memory.list(None):
        all_threads.add(m.config['configurable']['thread_id'])

    return list(all_threads)

def get_thread_preview(thread_id): ## get the preview of the thread (makes the selectbox look better)
    mode = get_thread_mode(thread_id)
    mode_icon = "📄" if mode == "rag" else "🔍"
    
    messages = load_conversation(thread_id)
    if messages and len(messages) > 0:
        first_msg = messages[0]
        content = first_msg["content"]
        preview = content[:30] + "..." if len(content) > 30 else content
        return f"{mode_icon} {preview}"
    
    return f"{mode_icon} Thread {thread_id[-8:]}"


## Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if 'thread_list' not in st.session_state:
    st.session_state.thread_list = retrieve_all_threads()

if 'thread_id' not in st.session_state:
    st.session_state.thread_id = create_thread_id("search")

if 'uploaded_documents' not in st.session_state:
    st.session_state.uploaded_documents = []

## Document Management Functions
def delete_document_from_storage():
    try:
        # Delete the vector store
        # This is because Chroma doesn't easily support selective document deletion
        vector_store_path = "./chroma_langchain_db"
        st.session_state.graph_manager.rag_graph = None
        gc.collect()
        if os.path.exists(vector_store_path):
            shutil.rmtree(vector_store_path)
            
        # If there are remaining documents, recreate the vector store
        if st.session_state.uploaded_documents:
            st.warning("Vector store cleared. Please re-upload remaining documents.")
            st.session_state.uploaded_documents = []
            st.session_state.graph_manager.rag_graph = None
        
        return True
    except Exception as e:
        st.error(f"Error deleting from storage: {e}")
        return False

def clear_all_documents():
    try:
        # Clear vector store
        delete_document_from_storage()
        
        # Switch to search mode if current thread is RAG
        if get_thread_mode(st.session_state.thread_id) == "rag":
            new_thread_id = create_thread_id("search")
            st.session_state.thread_id = new_thread_id
            st.session_state.messages = []
            if new_thread_id not in st.session_state.thread_list:
                st.session_state.thread_list.append(new_thread_id)
        
        st.success("✅ All documents cleared")
    except Exception as e:
        st.error(f"❌ Error clearing documents: {e}")

## Document Management UI
with st.sidebar:
    st.header("📄 Document Management")
    
    if st.session_state.uploaded_documents:
        st.write("**Uploaded Documents:**")
        for i, doc_name in enumerate(st.session_state.uploaded_documents):
            st.write(f"📄{i+1} {doc_name}")
        
        if st.button("🗑️ Clear All Documents"):
            clear_all_documents()
            st.rerun()
    else:
        st.info("No documents uploaded yet")

## Thread Management UI
with st.sidebar:
    st.header("💬 Thread Management")
    
    # Current mode indicator
    current_mode = st.session_state.graph_manager.current_mode
    mode_icon = "📄" if current_mode == "rag" else "🔍"
    st.info(f"Current Mode: {mode_icon} {current_mode.upper()}")
    
    # New chat button
    if st.button("🆕 New Chat"):
        st.session_state.graph_manager.set_mode("search")
        new_thread_id = create_thread_id("search")
        st.session_state.thread_id = new_thread_id
        st.session_state.messages = []
        if new_thread_id not in st.session_state.thread_list:
            st.session_state.thread_list.append(new_thread_id)
        st.rerun()
    
    # Thread selection
    if st.session_state.thread_list:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Get current index for the selectbox
            current_index = 0
            if st.session_state.thread_id in st.session_state.thread_list:
                reversed_list = st.session_state.thread_list[::-1]
                current_index = reversed_list.index(st.session_state.thread_id)
            
            selected_thread = st.selectbox(
                "Select conversation",
                options=st.session_state.thread_list[::-1],  # Most recent first
                format_func=get_thread_preview,
                index=current_index,
                key="thread_selector"
            )
        
        with col2:
            if st.button("🗑️", help="Delete selected thread"):
                # Allow deletion of current thread if more than 1 thread exists
                if len(st.session_state.thread_list) > 1:
                    if delete_thread(selected_thread):
                        st.session_state.thread_list.remove(selected_thread)
                        
                        # If deleted thread was current thread, load the most recent remaining thread
                        if selected_thread == st.session_state.thread_id:
                            # Get the most recent remaining thread
                            remaining_threads = [t for t in st.session_state.thread_list if t != selected_thread]
                            if remaining_threads:
                                new_current_thread = remaining_threads[-1]  # Most recent
                                st.session_state.thread_id = new_current_thread
                                st.session_state.messages = load_conversation(new_current_thread)
                                
                                # Update mode based on new thread
                                thread_mode = get_thread_mode(new_current_thread)
                                st.session_state.graph_manager.current_mode = thread_mode
                        
                        st.success("✅ Thread deleted!")
                        st.rerun()
                else:
                    st.error("❌ Cannot delete the only remaining thread")
        
        # Load selected thread ONLY if it's different from current
        if selected_thread != st.session_state.thread_id:
            st.session_state.thread_id = selected_thread
            st.session_state.messages = load_conversation(selected_thread)
            
            # Update current mode based on thread
            thread_mode = get_thread_mode(selected_thread)
            st.session_state.graph_manager.current_mode = thread_mode
            st.rerun()

## Thread Cleanup
with st.sidebar:   ## delete all the empty threads that were created but not used
    with st.expander("🧹 Thread Cleanup"):
        st.write(f"Total threads: {len(st.session_state.thread_list)}")
        
        if st.button("Delete All Empty Threads"):
            deleted_count = 0
            for thread_id in st.session_state.thread_list:
                if thread_id != st.session_state.thread_id:
                    messages = load_conversation(thread_id)
                    if not messages:
                        if delete_thread(thread_id):
                            st.session_state.thread_list.remove(thread_id)
                            deleted_count += 1
            
            if deleted_count > 0:
                st.success(f"✅ Deleted {deleted_count} empty threads")
                st.rerun()

## Display chat history
messages = load_conversation(st.session_state.thread_id)
if messages not in st.session_state.messages:
    st.session_state.messages = messages
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

## Chat Input Handling
user_input = None
chat_input = st.chat_input("Enter your query", accept_file=True, file_type=['pdf'])

# Handle file upload
if chat_input and chat_input.get('files'):
    documents = chat_input['files']
    
    # Add to document list
    for doc in documents:
        if doc.name not in st.session_state.uploaded_documents:
            st.session_state.uploaded_documents.append(doc.name)
    
    # Create/update RAG graph
    st.session_state.graph_manager.get_rag_graph(documents)
    st.session_state.graph_manager.set_mode("rag")
    
    # Create new RAG thread
    new_thread_id = create_thread_id("rag")
    st.session_state.thread_id = new_thread_id
    st.session_state.messages = []
    if new_thread_id not in st.session_state.thread_list:
        st.session_state.thread_list.append(new_thread_id)
    
    st.success(f"✅ Uploaded {len(documents)} document(s). Started new RAG chat.")
    st.rerun()

# Handle text input
if chat_input and chat_input.text:
    user_input = chat_input.text

if user_input:
    # Ensure thread is in thread list
    if st.session_state.thread_id not in st.session_state.thread_list:
        st.session_state.thread_list.append(st.session_state.thread_id)
    
    # Get the appropriate graph for this specific thread
    current_graph = st.session_state.graph_manager.get_graph_for_thread(st.session_state.thread_id)
    
    if not current_graph:
        st.error("Graph not available for this thread type.")
        st.stop()
    
    CONFIG = {'configurable': {'thread_id': st.session_state.thread_id}}
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            # Get thread mode to determine input format
            thread_mode = get_thread_mode(st.session_state.thread_id)
            graph_input = {
                "question": user_input
            }
            result = current_graph.invoke(graph_input, config=CONFIG)
            
            # Extract response, defaulting to a generic message if generation is missing or empty
            response_content = result.get("generation", "Sorry, I couldn't generate a response.")

            st.markdown(response_content)
            
            # Update session state messages
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": response_content})
    st.rerun()
