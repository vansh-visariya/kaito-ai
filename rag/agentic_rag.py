"""Agentic RAG (Retrieval-Augmented Generation) with self-correction.

Implements a sophisticated document-analysis pipeline using LangGraph:

1. **Retrieve** — fetch top-k chunks from a ChromaDB vector store.
2. **Grade documents** — LLM filters out irrelevant chunks.
3. **Generate** — produce an answer grounded in relevant context.
4. **Grade generation** — verify the answer is grounded *and* useful;
   loop back through web-search or query-rewriting if not.
5. **Update memory** — persist the Q&A pair in conversation history.

A ``MAX_GENERATION_RETRIES`` guard prevents infinite self-correction loops.
"""

import json
import logging
import os
import tempfile
from functools import lru_cache
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RETRIEVER_K,
    MAX_GENERATION_RETRIES,
    VECTOR_STORE_DIR,
    configure_environment,
)
from database.memory import get_rag_memory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
RELEVANCE_PROMPT = PromptTemplate(
    template="""\
You are a grader assessing relevance of a retrieved document to a user question.

Retrieved document: {document}

User question: {question}

If the document contains information related to the question, grade it as relevant.
Give a binary score 'yes' or 'no' to indicate whether the document is relevant.

Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
""",
    input_variables=["question", "document"],
)

REWRITER_PROMPT = PromptTemplate(
    template="""\
You are a question re-writer that converts an input question to a better \
version optimized for vectorstore retrieval.

Look at the input and try to reason about the underlying semantic intent.

Here is the initial question:
{question}

Provide an improved question without any preamble.
""",
    input_variables=["question"],
)

GROUNDED_PROMPT = PromptTemplate(
    template="""\
You are a grader assessing whether an answer is grounded in / supported \
by a set of retrieved facts.

Retrieved facts: {documents}

Answer: {generation}

Give a binary score 'yes' or 'no' to indicate whether the answer is \
grounded in the retrieved facts.

Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
""",
    input_variables=["generation", "documents"],
)

ANSWERS_QUESTION_PROMPT = PromptTemplate(
    template="""\
You are a grader assessing whether an answer addresses / resolves a question.

User question: {question}

Answer: {generation}

Give a binary score 'yes' or 'no' to indicate whether the answer resolves the question.

Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
""",
    input_variables=["generation", "question"],
)

RAG_GENERATE_PROMPT = PromptTemplate(
    template="""\
You are an assistant for question-answering tasks. Use the following \
pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
You also have the conversation history for personal context.

chat_history: {messages}
Question: {question}

Context: {context}

Answer:
""",
    input_variables=["messages", "question", "context"],
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


def _format_documents(documents: List) -> str:
    """Join document contents into a single context string."""
    return "\n\n".join(
        doc.page_content if hasattr(doc, "page_content") else str(doc)
        for doc in documents
    )


# ---------------------------------------------------------------------------
# Embedding & vector-store setup
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def setup_embeddings() -> HuggingFaceEmbeddings:
    """Initialise and cache the HuggingFace embedding model."""
    logger.info("Loading embedding model: %s", DEFAULT_EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)


def setup_vector_store(files) -> Chroma:
    """Create a ChromaDB vector store from uploaded PDF files.

    Args:
        files: File-like objects with ``.name`` and ``.getvalue()`` (PDF).

    Returns:
        A populated :class:`Chroma` vector store.
    """
    all_docs = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        all_docs.extend(loader.load())
        os.remove(tmp_path)

    logger.info("Loaded %d pages from %d PDF(s).", len(all_docs), len(files))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )
    splits = splitter.split_documents(all_docs)
    logger.info("Created %d text chunks.", len(splits))

    return Chroma.from_documents(
        documents=splits,
        embedding=setup_embeddings(),
        persist_directory=VECTOR_STORE_DIR,
    )


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------
class RAGState(TypedDict):
    """LangGraph state for the agentic-RAG workflow."""

    question: str
    documents: List
    generation: str
    messages: Annotated[List[BaseMessage], add_messages]
    retry_count: int


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------
def create_rag_chain(
    groq_api_key: str,
    model_name: str,
    files,
    tavily_api_key: str,
):
    """Build and compile the agentic-RAG LangGraph.

    Args:
        groq_api_key: Groq API key for LLM inference.
        model_name: Groq model identifier.
        files: File-like objects with ``.name`` and ``.getvalue()`` (PDF).
        tavily_api_key: Tavily API key for web-search fallback.

    Returns:
        A compiled LangGraph workflow backed by an SQLite checkpointer.
    """
    configure_environment(groq_api_key, tavily_api_key)

    llm = ChatGroq(model=model_name)
    tavily_retriever = TavilySearchAPIRetriever(k=DEFAULT_RETRIEVER_K)
    memory = get_rag_memory()

    vector_store = setup_vector_store(files)
    retriever = vector_store.as_retriever(search_kwargs={"k": DEFAULT_RETRIEVER_K})
    rag_chain = RAG_GENERATE_PROMPT | llm | StrOutputParser()

    # -- grading helpers ---------------------------------------------------

    def _check_relevance(doc, question: str) -> bool:
        chain = RELEVANCE_PROMPT | llm | StrOutputParser()
        result = chain.invoke({"question": question, "document": doc.page_content})
        return _parse_llm_score(result, default="yes") == "yes"

    def _rewrite_question(question: str) -> str:
        chain = REWRITER_PROMPT | llm | StrOutputParser()
        return chain.invoke({"question": question})

    def _check_grounded(generation: str, documents: List) -> bool:
        chain = GROUNDED_PROMPT | llm | StrOutputParser()
        result = chain.invoke({
            "generation": generation,
            "documents": _format_documents(documents),
        })
        return _parse_llm_score(result, default="yes") == "yes"

    def _check_answers_question(generation: str, question: str) -> bool:
        chain = ANSWERS_QUESTION_PROMPT | llm | StrOutputParser()
        result = chain.invoke({"generation": generation, "question": question})
        return _parse_llm_score(result, default="yes") == "yes"

    # -- node functions ----------------------------------------------------

    def retrieve(state: RAGState) -> dict:
        """Retrieve documents from the vector store."""
        question = state["question"]
        logger.info("Retrieving docs for: %s", question[:80])
        return {
            "question": question,
            "documents": retriever.invoke(question),
            "retry_count": 0,
        }

    def grade_documents(state: RAGState) -> dict:
        """Filter retrieved documents by LLM-assessed relevance."""
        question = state["question"]
        docs = state["documents"]
        filtered = [d for d in docs if _check_relevance(d, question)]
        logger.info("Grading: %d/%d docs relevant.", len(filtered), len(docs))
        return {"question": question, "documents": filtered}

    def transform_query(state: RAGState) -> dict:
        """Rewrite the user query for better retrieval."""
        original = state["question"]
        rewritten = _rewrite_question(original)
        logger.info("Rewrite: '%.40s' → '%.40s'", original, rewritten)
        return {"question": rewritten, "documents": state["documents"]}

    def web_search(state: RAGState) -> dict:
        """Fallback: search the web when local docs are insufficient."""
        question = state["question"]
        logger.info("Web-search fallback: %s", question[:80])
        return {"question": question, "documents": tavily_retriever.invoke(question)}

    def generate(state: RAGState) -> dict:
        """Generate an answer using the RAG chain."""
        question = state["question"]
        documents = state["documents"]
        answer = rag_chain.invoke({
            "messages": state["messages"],
            "context": _format_documents(documents),
            "question": question,
        })
        return {
            "question": question,
            "documents": documents,
            "generation": answer,
            "retry_count": state.get("retry_count", 0) + 1,
        }

    def update_memory(state: RAGState) -> dict:
        """Persist the Q&A pair in conversation history."""
        return {
            "messages": [
                HumanMessage(content=state["question"]),
                AIMessage(content=state["generation"]),
            ],
        }

    # -- routing functions -------------------------------------------------

    def decide_to_generate(state: RAGState) -> str:
        """Route after grading: generate if docs exist, else rewrite."""
        return "generate" if state["documents"] else "transform_query"

    def grade_generation(state: RAGState) -> str:
        """Post-generation QA: grounded? useful? Retry-limited."""
        if state.get("retry_count", 0) >= MAX_GENERATION_RETRIES:
            logger.warning("Max retries (%d) hit — accepting current answer.", MAX_GENERATION_RETRIES)
            return "useful"

        if not _check_grounded(state["generation"], state["documents"]):
            logger.info("Not grounded — routing to web search.")
            return "not_grounded"

        if not _check_answers_question(state["generation"], state["question"]):
            logger.info("Doesn't answer question — rewriting query.")
            return "not_useful"

        return "useful"

    # -- graph wiring ------------------------------------------------------
    workflow = StateGraph(RAGState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
    workflow.add_node("update_memory", update_memory)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"generate": "generate", "transform_query": "transform_query"},
    )
    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "generate")
    workflow.add_conditional_edges(
        "generate",
        grade_generation,
        {
            "not_grounded": "web_search",
            "not_useful": "transform_query",
            "useful": "update_memory",
        },
    )
    workflow.add_edge("update_memory", END)

    app = workflow.compile(checkpointer=memory)
    logger.info("RAG agent compiled successfully.")
    return app
