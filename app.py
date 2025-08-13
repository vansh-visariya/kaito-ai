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



## new chat button
if st.sidebar.button("new chat"):
    thread = generate_unique_id()
    st.session_state['thread_id'] = thread
    st.session_state.messages = []
    add_thread_id(st.session_state['thread_id'])

## load previous conversations
selected_thread = st.sidebar.selectbox(
    "Select a conversation thread",
    options=st.session_state['thread_list'][::-1]
)

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
