from typing_extensions import TypedDict, Optional, Annotated
from typing import List
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph.message import add_messages
import json
import os
from database.get_sql import get_search_memory

memory = get_search_memory()

load_dotenv()

def model_create(groq_api_key, model_name):
    os.environ["GROQ_API_KEY"] = groq_api_key
    llm = ChatGroq(model=model_name)

    search_tool = TavilySearch()

    class AgentState(TypedDict):
        generation: str
        question: str
        search_results: Optional[list[str]]
        messages: Annotated[List[BaseMessage],add_messages]

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
            template="""
        You are a knowledgeable and helpful assistant. Answer the user's question as accurately and helpfully as possible.

        Use the information from your own knowledge. If the provided search results are relevant or necessary for answering the question, you may refer to them — but only use them when needed. If the answer can be confidently given without them, do not rely on the search results.

        chat_history: this is the conversation history. You can refer to it to answer the user's question of personal context {messages}
        question: {question}
        search_results: {search_results}
        """,
            input_variables=["messages","question", "search_results"],
            )
        chain = prompt | llm | StrOutputParser()
        question = state['question']
        search_results = state.get('search_results', [])
        messages = state.get('messages', [])

        search_results = state.get('search_results', [])
        
        answer = chain.invoke({"messages": messages,"question": question, "search_results": "\n\n".join(search_results)})
        messages.append(HumanMessage(content=question))
        messages.append(AIMessage(content=answer))

        return {
        "generation": answer,
        "search_results": search_results,
        "question": question,
        "messages": messages
        }

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