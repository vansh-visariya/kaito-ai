import streamlit as st
from agent.search import model_create
from utility import generate_unique_id, validate_groq_key, get_memory_for_mode
from langchain_core.messages import HumanMessage, AIMessage
from rag.agentic_rag import create_rag_chain
from database.get_sql import get_search_memory, get_rag_memory
import os

# LangSmith configuration
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if langchain_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = "chatbot"

st.title("🤖 Advanced Chatbot")

## Enhanced Graph Manager for proper separation
class GraphManager:
    def __init__(self, groq_api_key, model_name):
        self.groq_api_key = groq_api_key
        self.model_name = model_name
        self.search_graph = None
        self.rag_graph = None
        self.current_mode = "search"
        self.uploaded_docs = []
        
    def get_search_graph(self):
        if not self.search_graph:
            self.search_graph = model_create(self.groq_api_key, self.model_name)
        return self.search_graph
    
    def get_rag_graph(self, documents=None):
        if documents:
            self.uploaded_docs = documents
            self.rag_graph = create_rag_chain(self.groq_api_key, self.model_name, documents)
        return self.rag_graph
    
    def get_graph_for_thread(self, thread_id):
        """Get the appropriate graph based on thread ID"""
        mode = get_thread_mode(thread_id)
        if mode == "rag":
            if not self.rag_graph:
                # If RAG graph doesn't exist but thread is RAG, there's an issue
                st.error("RAG graph not available. Please upload documents first.")
                return None
            return self.rag_graph
        else:
            return self.get_search_graph()
    
    def get_graph_for_mode(self, mode):
        if mode == "rag" and self.rag_graph:
            return self.rag_graph
        return self.get_search_graph()
    
    def get_current_graph(self):
        return self.get_graph_for_mode(self.current_mode)
    
    def set_mode(self, mode):
        self.current_mode = mode
        

## User data collection
with st.sidebar:
    st.header("🔑 Configuration")
    groq_api_key = st.text_input("Enter your Groq API key", type="password")
    model_name = st.text_input("Enter the model you want to use", value="gemma2-9b-it")

## Validate groq key and initialize graph manager
if groq_api_key and validate_groq_key(groq_api_key):
    if 'graph_manager' not in st.session_state:
        st.session_state.graph_manager = GraphManager(groq_api_key, model_name)
    graph_manager = st.session_state.graph_manager
else:
    st.error("❌ Invalid Groq API key. Please check and try again.")
    st.stop()

## Thread Management Functions
def get_thread_mode(thread_id):
    """Extract mode from thread ID"""
    return "rag" if thread_id.startswith("rag_") else "search"

def create_thread_id(mode):
    """Create a properly formatted thread ID"""
    return f"{mode}_{generate_unique_id()}"

def delete_thread(thread_id):
    """Delete a specific thread from the appropriate database"""
    mode = get_thread_mode(thread_id)
    memory = get_memory_for_mode(mode)
    try:
        conn = memory.conn
        cursor = conn.cursor()
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error deleting thread: {e}")
        return False

def load_conversation(thread_id):
    """Load conversation from the appropriate graph's memory"""
    try:
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
    except Exception as e:
        st.error(f"Error loading conversation: {e}")
        return []

def retrieve_all_threads():
    """Retrieve all threads from both databases"""
    all_threads = set()
    
    try:
        search_memory = get_search_memory()
        for m in search_memory.list(None):
            all_threads.add(m.config['configurable']['thread_id'])
    except:
        pass
    
    try:
        rag_memory = get_rag_memory()
        for m in rag_memory.list(None):
            all_threads.add(m.config['configurable']['thread_id'])
    except:
        pass
    
    return list(all_threads)

def get_thread_preview(thread_id):
    """Get thread preview with mode indicator"""
    try:
        mode = get_thread_mode(thread_id)
        mode_icon = "📄" if mode == "rag" else "🔍"
        
        messages = load_conversation(thread_id)
        if messages and len(messages) > 0:
            first_msg = messages[0]
            content = first_msg["content"]
            preview = content[:30] + "..." if len(content) > 30 else content
            return f"{mode_icon} {preview}"
        
        return f"{mode_icon} Thread {thread_id[-8:]}"
    except:
        return f"Thread {thread_id[-8:]}"

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
def delete_document_from_storage(doc_name):
    """Delete document from vector store"""
    try:
        import shutil
        import os
        
        # For now, we'll clear the entire vector store when any document is deleted
        # This is because Chroma doesn't easily support selective document deletion
        vector_store_path = "./chroma_langchain_db"
        if os.path.exists(vector_store_path):
            shutil.rmtree(vector_store_path)
            
        # If there are remaining documents, recreate the vector store
        if st.session_state.uploaded_documents:
            # We need to recreate the RAG graph with remaining documents
            # This requires storing the actual document objects, not just names
            st.warning("Vector store cleared. Please re-upload remaining documents.")
            st.session_state.uploaded_documents = []
            st.session_state.graph_manager.rag_graph = None
        
        return True
    except Exception as e:
        st.error(f"Error deleting from storage: {e}")
        return False

def delete_document(doc_name, index):
    """Delete a specific document"""
    try:
        # Remove from session state
        st.session_state.uploaded_documents.pop(index)
        
        # Delete from actual storage
        delete_document_from_storage(doc_name)
        
        # If no documents left, clear RAG mode
        if not st.session_state.uploaded_documents:
            st.session_state.graph_manager.rag_graph = None
            # Switch current thread to search mode if it was RAG
            if get_thread_mode(st.session_state.thread_id) == "rag":
                # Create new search thread
                new_thread_id = create_thread_id("search")
                st.session_state.thread_id = new_thread_id
                st.session_state.messages = []
                if new_thread_id not in st.session_state.thread_list:
                    st.session_state.thread_list.append(new_thread_id)
        
        st.success(f"✅ Deleted {doc_name}")
    except Exception as e:
        st.error(f"❌ Error deleting document: {e}")

def clear_all_documents():
    """Clear all documents and vector store"""
    try:
        # Clear document list
        st.session_state.uploaded_documents = []
        
        # Clear vector store
        delete_document_from_storage("all")
        
        # Clear RAG graph
        st.session_state.graph_manager.rag_graph = None
        
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
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"📄 {doc_name}")
            with col2:
                if st.button("🗑️", key=f"del_doc_{i}", help=f"Delete {doc_name}"):
                    delete_document(doc_name, i)
                    st.rerun()
        
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
with st.sidebar:
    with st.expander("🧹 Thread Cleanup"):
        st.write(f"Total threads: {len(st.session_state.thread_list)}")
        
        if st.button("Delete All Empty Threads"):
            deleted_count = 0
            for thread_id in st.session_state.thread_list.copy():
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
    
    # Create user message
    user_message = HumanMessage(content=user_input)
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                # Get thread mode to determine input format
                thread_mode = get_thread_mode(st.session_state.thread_id)
                
                if thread_mode == "rag":
                    # For RAG: Use the question format and let the graph handle history via checkpointer
                    graph_input = {
                        "question": user_input,
                        "messages": [user_message]  # Current message
                    }
                else:
                    # For Search: Use the question format and let the graph handle history via checkpointer
                    graph_input = {
                        "question": user_input,
                        "messages": [user_message]  # Current message
                    }
                
                # Invoke the graph - it will automatically load conversation history via checkpointer
                result = current_graph.invoke(graph_input, config=CONFIG)
                
                # Extract response
                response_content = result.get("generation", "Sorry, I couldn't generate a response.")
                
                # Display response
                st.markdown(response_content)
                
                # Update session state messages for display purposes
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.messages.append({"role": "assistant", "content": response_content})
                
            except Exception as e:
                error_msg = f"❌ Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Rerun to refresh the display
    st.rerun()