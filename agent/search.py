from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages 
from langchain.schema.messages import BaseMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent

load_dotenv()

search_tool = TavilySearchResults()
tools = [search_tool]

llm = ChatGroq( model="gemma2-9b-it")

prompt = """You are a helpful and efficient assistant with access to a tools. Use your internal knowledge to answer questions whenever possible. Only use the web search tool if:

The question requires up-to-date or real-time information (e.g., current events, weather, stock prices).

The information is niche, uncommon, or likely outside your training data.

Accuracy is critical and cannot be guaranteed without verification.

Do not use the tools for general knowledge or questions you can confidently answer without it. When using the tool, explain why you’re doing so. 
"""
agent_executor = create_react_agent(llm, tools, prompt=prompt)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

graph = StateGraph(AgentState)

def agent(state: AgentState) -> AgentState:
    response = agent_executor.invoke({"messages": state['messages']})
    return {"messages": response["messages"]}

graph.add_node("agent", agent)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)

workflow = graph.compile()