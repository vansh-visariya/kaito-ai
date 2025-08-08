import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import os

st.title("chatbot")

### getting api from user
groq_api_key = st.text_input("enter you api key", type="password")
os.environ["GROQ_API_KEY"] = groq_api_key

### select which model to use
option = st.selectbox(
    'which model you want to use',
    ['openai/gpt-oss-20b', 'deepseek-r1-distill-llama-70b', 'qwen/qwen3-32b', 'gemma2-9b-it','llama-3.1-8b-instant']
)
llm = ChatGroq(model=option)

### define state
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

### define chatbot
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot") 
graph_builder.add_edge("chatbot", END)      ### START -> chatbot -> END

graph = graph_builder.compile()

### user input
user_input = st.chat_input("Enter your query")

if user_input:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    for message in st.session_state.messages:
        st.chat_message(message["role"]).write(message["content"])

    for event in graph.stream({"messages": st.session_state.messages}):
        for value in event.values():
            st.session_state.messages.append({"role": "assistant", "content": value["messages"][-1].content})
            st.chat_message("assistant").write(value["messages"][-1].content)
