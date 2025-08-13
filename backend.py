import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import os

def model_create(groq_api_key, model_name):

    os.environ["GROQ_API_KEY"] = groq_api_key
    llm = ChatGroq(model=model_name)

    ### define state
    class State(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    graph_builder = StateGraph(State)

    ### define chatbot
    def chatbot(state: State):
        return {"messages": [llm.invoke(state["messages"])]}
    
    ## database creation
    conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
    memory = SqliteSaver(conn = conn)

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot") 
    graph_builder.add_edge("chatbot", END)      ### START -> chatbot -> END

    graph = graph_builder.compile(checkpointer=memory)
    return graph,memory