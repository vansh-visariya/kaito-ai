from langchain_groq import ChatGroq
from typing_extensions import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
import sqlite3
import os
from dotenv import load_dotenv
load_dotenv()

def model_create(groq_api_key, model_name): 
    ## define state
    class State(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    search_tool = TavilySearchResults()

    os.environ["GROQ_API_KEY"] = groq_api_key
    llm = ChatGroq(model=model_name)
    tools = [search_tool]
    agent = create_react_agent(llm, tools=tools)

    def agent_node(state: State) -> State:
        response = agent.invoke({"messages": state['messages']})
        return {"messages": response["messages"]}
    

    graph_builder = StateGraph(State)
    conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
    memory = SqliteSaver(conn=conn)

    graph_builder.add_node("agent", agent_node)
    graph_builder.add_edge(START, "agent")
    graph_builder.add_edge("agent", END)

    graph = graph_builder.compile(checkpointer=memory)
    return graph, memory
