from langchain_groq import ChatGroq
from typing_extensions import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import Tool
import sqlite3
import os
from dotenv import load_dotenv
load_dotenv()

def model_create(groq_api_key, model_name): 
    ## define state
    class State(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
        next_action: Optional[str]
        search_results: Optional[str]

    search_tool = TavilySearchResults(tavily_api_key=os.getenv("TAVILY_API_KEY"))

    def summarize(results):
        if not results or not isinstance(results, list):
            return "nothing found."
        summary = ""
        for item in results[:3]:
            title = item.get("title", "No title")
            url = item.get("url", "")
            content = item.get("content", "").strip().split("\n")[0]
            summary += f"- **{title}**: {content[:150].strip()}... [Read more]({url})\n"
        return summary

    def web_search_node(state: State):
        query = state["messages"][-1].content
        raw_results = search_tool.run(query)
        summary = summarize(raw_results)
        return {
            "search_results": summary,
            "messages": [AIMessage(content=summary)]
        }

    os.environ["GROQ_API_KEY"] = groq_api_key
    llm = ChatGroq(model=model_name)
    tools = [Tool(name="Web Search", func=search_tool.run, description="Search the web and return the precise search results as per the user query.")]
    agent = create_react_agent(llm, tools=tools)

    def router(state: State):
        last_msg = state["messages"][-1].content.lower()
        if "news" in last_msg or "search" in last_msg or "current" in last_msg:
            return "web_search"
        return "llm_response"

    def llm_response(state: State):
        return {"messages":llm.invoke(state["messages"])}

    graph_builder = StateGraph(State)
    conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
    memory = SqliteSaver(conn=conn)

    graph_builder.add_node("web_search", web_search_node)
    graph_builder.add_node("llm_response", llm_response)

    graph_builder.add_conditional_edges(START, router, {"web_search": "web_search", "llm_response": "llm_response"})
    graph_builder.add_edge("web_search", END)
    graph_builder.add_edge("llm_response", END)

    graph = graph_builder.compile(checkpointer=memory)
    return graph, memory