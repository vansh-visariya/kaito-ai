"""Web-search agent built with LangGraph.

Workflow
--------
1. **Router** — an LLM classifier decides whether the question can be
   answered from internal knowledge or requires a live web search.
2. **Web search** (conditional) — queries the Tavily API for current info.
3. **Generate** — produces the final answer, optionally augmented with
   search results and full conversation history.

The compiled graph is persisted via an SQLite checkpointer so that
conversation state survives across Streamlit reruns.
"""

import json
import logging
from typing import List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from config import configure_environment
from database.memory import get_search_memory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
ROUTER_PROMPT = PromptTemplate(
    template="""\
You are a router agent. Your purpose is to decide if a question can be \
answered from your internal knowledge or if it requires a web search.

Your internal knowledge is static and does not include information about \
current events, news, or any developments after your last training cut-off.

Analyze the user's question. If the question:
- Asks for "the latest news", "today's update", or current events.
- Refers to a specific future date (like the current year 2025).
- Inquires about a person or topic for which information changes rapidly.
You MUST perform a web search.

User Question: {question}

Based on the analysis, does this question require a web search to provide \
a relevant and up-to-date answer?
Give a binary score 'yes' for "I can answer from internal knowledge" or \
'no' for "I need a web search".

Provide the binary score as a JSON with a single key 'score' and no \
preamble or explanation.
""",
    input_variables=["question"],
)

GENERATE_PROMPT = PromptTemplate(
    template="""\
You are a knowledgeable and helpful assistant. Answer the user's question \
as accurately and helpfully as possible.

Use the information from your own knowledge. If the provided search \
results are relevant or necessary for answering the question, you may \
refer to them — but only use them when needed. If the answer can be \
confidently given without them, do not rely on the search results.

chat_history: {messages}
question: {question}
search_results: {search_results}
""",
    input_variables=["messages", "question", "search_results"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_llm_score(raw: str, default: str = "no") -> str:
    """Extract the ``score`` value from a JSON LLM response.

    Args:
        raw: Raw LLM output (expected ``{"score": "yes"|"no"}``).
        default: Fallback if parsing fails.

    Returns:
        Lowercase score string.
    """
    try:
        return json.loads(raw).get("score", default).lower()
    except (json.JSONDecodeError, AttributeError) as exc:
        logger.warning("Failed to parse LLM score: %s (raw: %.100s)", exc, raw)
        return default


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------
class SearchState(TypedDict):
    """LangGraph state for the search agent."""

    generation: str
    question: str
    search_results: Optional[list[str]]
    messages: Annotated[List[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------
def model_create(
    groq_api_key: str,
    model_name: str,
    tavily_api_key: str,
) -> StateGraph:
    """Build and compile the search-agent LangGraph.

    Args:
        groq_api_key: Groq API key for LLM inference.
        model_name: Groq model identifier (e.g. ``llama-3.1-8b-instant``).
        tavily_api_key: Tavily API key for web search.

    Returns:
        A compiled LangGraph workflow backed by an SQLite checkpointer.
    """
    configure_environment(groq_api_key, tavily_api_key)

    llm = ChatGroq(model=model_name)
    search_tool = TavilySearch()
    memory = get_search_memory()

    # -- node functions ----------------------------------------------------

    def can_answer(state: SearchState) -> bool:
        """Router: can the LLM answer from internal knowledge?"""
        chain = ROUTER_PROMPT | llm | StrOutputParser()
        result = chain.invoke({"question": state["question"]})
        return _parse_llm_score(result) == "yes"

    def web_search(state: SearchState) -> dict:
        """Fetch results from Tavily and extract title/content snippets."""
        logger.info("Web search: %s", state["question"][:80])
        raw_results = search_tool.invoke(state["question"])

        snippets: list[str] = []
        if "results" in raw_results:
            for r in raw_results["results"]:
                snippets.append(f"title: {r['title']}\ncontent: {r['content']}")

        return {"search_results": snippets}

    def generate(state: SearchState) -> dict:
        """Produce the final answer using history + optional search context."""
        chain = GENERATE_PROMPT | llm | StrOutputParser()

        question = state["question"]
        search_results = state.get("search_results", [])
        messages = state.get("messages", [])

        answer = chain.invoke({
            "messages": messages,
            "question": question,
            "search_results": "\n\n".join(search_results),
        })

        return {
            "generation": answer,
            "search_results": search_results,
            "question": question,
            "messages": [
                HumanMessage(content=question),
                AIMessage(content=answer),
            ],
        }

    # -- graph wiring ------------------------------------------------------
    graph = StateGraph(SearchState)
    graph.add_node("web_search", web_search)
    graph.add_node("generate", generate)

    graph.add_conditional_edges(
        START,
        lambda state: "generate" if can_answer(state) else "web_search",
        {"web_search": "web_search", "generate": "generate"},
    )
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)

    workflow = graph.compile(checkpointer=memory)
    logger.info("Search agent compiled successfully.")
    return workflow