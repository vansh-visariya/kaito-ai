import streamlit as st
from backend import model_create
from utility import generate_unique_id, validate_groq_key
from langchain_core.messages import HumanMessage

st.title("chatbot")

## function for storing all threads in the conversion
def add_thread_id(thread_id):
    if thread_id not in st.session_state['thread_list']:
        st.session_state['thread_list'].append(thread_id)

## user data collection
with st.sidebar:
    groq_api_key = st.text_input("Enter your API key", type="password")
    model_name = st.text_input("Enter the model you want to use", value="gemma2-9b-it")

## validate groq key
if groq_api_key and validate_groq_key(groq_api_key):
    graph,memory = model_create(groq_api_key, model_name)
else:
    st.error("Invalid Groq API key. Please check and try again.")
    st.stop()

## delete thread from sqlite database
def delete_thread(thread_id, memory):
    """Delete a specific thread from the database"""
    try:
        # SqliteSaver doesn't have a direct delete method
        # Need to access the underlying SQLite connection
        conn = memory.conn
        cursor = conn.cursor()
        
        # Delete from checkpoints table (where conversation data is stored)
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        conn.commit()
        
        return True
    except Exception as e:
        st.error(f"Error deleting thread: {e}")
        return False

## load conversation from sqlite database
def load_conversation(thread_id):
    state = graph.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])

## retrieve all threads from sqlite database
def retrieve_threads():
    all_threads = set()
    for m in memory.list(None):
        all_threads.add(m.config['configurable']['thread_id'])
    return list(all_threads)

## setup for session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if 'thread_list' not in st.session_state:
    st.session_state['thread_list'] = retrieve_threads()

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_unique_id()
    add_thread_id(st.session_state['thread_id'])

def get_thread_preview(thread_id):
    """Get first message of thread for preview"""
    try:
        messages = load_conversation(thread_id)
        if messages:
            first_msg = messages[0]
            if hasattr(first_msg, 'content'):
                return first_msg.content[:30] + "..." if len(first_msg.content) > 30 else first_msg.content
        return f"Thread {thread_id[:8]}"
    except:
        return f"Thread {thread_id[:8]}"

## new chat button
## Enhanced thread management in sidebar
with st.sidebar:
    st.subheader("Thread Management")
    
    # New chat button
    if st.button("🆕 New Chat"):
        thread = generate_unique_id()
        st.session_state['thread_id'] = thread
        st.session_state.messages = []
        add_thread_id(st.session_state['thread_id'])
        st.rerun()
    
    # Thread selection with delete option
    if st.session_state.get('thread_list'):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_thread = st.selectbox(
                "Select conversation",
                options=st.session_state['thread_list'][::-1],
                format_func=get_thread_preview
            )
        
        with col2:
            if st.button("🗑️", help="Delete selected thread"):
                if selected_thread != st.session_state['thread_id']:
                    if delete_thread(selected_thread, memory):
                        st.session_state['thread_list'].remove(selected_thread)
                        st.success("Thread deleted!")
                        st.rerun()
                else:
                    st.error("Cannot delete current thread")


# Add to sidebar 
with st.expander("🧹 Thread Cleanup"):
    st.write(f"Total threads: {len(st.session_state.get('thread_list', []))}")
    
    if st.button("Delete All Empty Threads"):
        deleted_count = 0
        for thread_id in st.session_state['thread_list'].copy():
            if thread_id != st.session_state['thread_id']:
                messages = load_conversation(thread_id)
                if not messages:  # Empty thread
                    if delete_thread(thread_id, memory):
                        st.session_state['thread_list'].remove(thread_id)
                        deleted_count += 1
        
        if deleted_count > 0:
            st.success(f"Deleted {deleted_count} empty threads")
            st.rerun()


if selected_thread != st.session_state['thread_id']:
    st.session_state['thread_id'] = selected_thread
    messages = load_conversation(selected_thread)
    conv = []
    for m in messages:
        if isinstance(m, HumanMessage):
            conv.append({"role": "user", "content": m.content})
        else:
            conv.append({"role": "assistant", "content": m.content})
    st.session_state.messages = conv


## display chat history
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

### user input
user_input = st.chat_input("Enter your query")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    ## config for langgraph
    CONFIG = {'configurable':{'thread_id': st.session_state['thread_id']}}

    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            ai_message = st.write_stream(
                message_chunk.content for message_chunk, metadata in graph.stream(
                    {'messages': [HumanMessage(content=user_input)]},
                    config= CONFIG,
                    stream_mode= 'messages'
                )
            )
            st.session_state.messages.append({"role": "assistant", "content": ai_message})
