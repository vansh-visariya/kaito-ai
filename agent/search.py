from typing_extensions import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import os
from database.get_sql import get_sqlite_connection

memory = get_sqlite_connection()

load_dotenv()

def model_create(groq_api_key, model_name):
    os.environ["GROQ_API_KEY"] = groq_api_key
    llm = ChatGroq(model=model_name)

    search_tool = TavilySearch()

    class AgentState(TypedDict):
        generated_answer: str
        question: str
        search_results: Optional[list[str]]

    graph = StateGraph(AgentState)

    def can_answer(state: AgentState) -> bool:
        prompt = PromptTemplate(
            template="""You are a router agent. Your purpose is to decide if a question can be answered from your internal knowledge or if it requires a web search.

    Your internal knowledge is static and does not include information about current events, news, or any developments after your last training cut-off.

    Analyze the user's question. If the question:
    - Asks for "the latest news", "today's update", or current events.
    - Refers to a specific future date (like the current year 2025).
    - Inquires about a person or topic for which information changes rapidly.
    You MUST perform a web search.

    User Question: {question}

    Based on the analysis, does this question require a web search to provide a relevant and up-to-date answer?
    Give a binary score 'yes' for "I can answer from internal knowledge" or 'no' for "I need a web search".

    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
    """,
            input_variables=["question"],
        )
        
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"question": state['question']})
        
        try:
            score = json.loads(result)
            return score.get("score", "no").lower() == "yes"
        except:
            return False  

    def web_search(state: AgentState):
        raw_results = search_tool.invoke(state['question'])
        
        search_snippets = []
        if 'results' in raw_results:
            for result in raw_results['results']:
                search_snippets.append("title: " + result['title'] + "\ncontent: " + result['content'])
                
        state['search_results'] = search_snippets
        return state

    def generate(state: AgentState):
        prompt = PromptTemplate(
            template="""You are a helpful assistant. Answer the user's question using the provided search results as context.
            question: {question}
            search_results: {search_results}

            Answer:
            """,
            input_variables=["question", "search_results"],
        )
        chain = prompt | llm | StrOutputParser()
        
        search_results = state.get('search_results', [])
        
        answer = chain.invoke({"question": state['question'], "search_results": "\n\n".join(search_results)})
        state['generated_answer'] = answer
        state['search_results'] = search_results
        
        return state

    graph.add_node("web_search", web_search)
    graph.add_node("generate", generate)

    graph.add_conditional_edges(
        START,
        lambda state: "generate" if can_answer(state) else "web_search",
        {"web_search": "web_search", "generate": "generate"}
    )
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)

    workflow = graph.compile(checkpointer=memory)
    return workflow