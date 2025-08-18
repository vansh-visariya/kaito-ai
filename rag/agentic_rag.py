from typing import List
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from database.get_sql import get_rag_memory
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph.message import add_messages
import tempfile
import os

load_dotenv()

def create_rag_chain(groq_api_key, model_name, files):
    os.environ["GROQ_API_KEY"] = groq_api_key
    llm = ChatGroq(model=model_name)
    memory = get_rag_memory()

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    def setup_vector_store():
        ## Load documents
        all_docs = []
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file_path = tmp_file.name

            loader = PyPDFLoader(tmp_file_path) 
            documents = loader.load()
            all_docs.extend(documents)
            os.remove(tmp_file_path)
        
        ## Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(all_docs)
        
        ## Create vector store
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_langchain_db" 
        )
        
        return vector_store

    # Initialize vector store
    vector_store = setup_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) ## get only top 3 documents

    # Define the state structure
    class GraphState(TypedDict):
        question: str    ## User question
        documents: List  ## List of documents have (page_content, metadata)
        generation: str   ## Generated answer
        messages: Annotated[List[BaseMessage],add_messages]

    def retrieve_from_vector_store(question: str):
        docs = retriever.invoke(question)
        return docs

    def is_relevant(doc, question: str) -> bool:
        # Simple relevance check
        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question.
            
            Retrieved document: {document}
            
            User question: {question}
            
            If the document contains information related to the question, grade it as relevant.
            Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.
            
            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
            """,
            input_variables=["question", "document"],
        )
        
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"question": question, "document": doc.page_content})
        
        try:
            import json
            score = json.loads(result)
            return score.get("score", "no").lower() == "yes"
        except:
            return True  # Default to relevant if parsing fails

    def rewriter(question: str) -> str:
        prompt = PromptTemplate(
            template="""You are a question re-writer that converts an input question to a better version that is optimized for vectorstore retrieval.
            
            Look at the input and try to reason about the underlying semantic intent / meaning.
            
            Here is the initial question:
            {question}
            
            Provide an improved question without any preamble.
            """,
            input_variables=["question"],
        )
        
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"question": question})

    def is_grounded(generation: str, documents: List) -> bool:
        prompt = PromptTemplate(
            template="""You are a grader assessing whether an answer is grounded in / supported by a set of retrieved facts.
            
            Retrieved facts: {documents}
            
            Answer: {generation}
            
            Give a binary score 'yes' or 'no' to indicate whether the answer is grounded in the retrieved facts.
            
            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
            """,
            input_variables=["generation", "documents"],
        )
        
        chain = prompt | llm | StrOutputParser()
        docs_content = "\n\n".join([doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in documents])
        result = chain.invoke({"generation": generation, "documents": docs_content})
        
        try:
            import json
            score = json.loads(result)
            return score.get("score", "no").lower() == "yes"
        except:
            return True 

    def answers_question(generation: str, question: str) -> bool:
        prompt = PromptTemplate(
            template="""You are a grader assessing whether an answer addresses / resolves a question.
            
            User question: {question}
            
            Answer: {generation}
            
            Give a binary score 'yes' or 'no' to indicate whether the answer resolves the question.
            
            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
            """,
            input_variables=["generation", "question"],
        )
        
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"generation": generation, "question": question})
        
        try:
            import json
            score = json.loads(result)
            return score.get("score", "no").lower() == "yes"
        except:
            return True  

    # Create RAG chain
    def create_rag_chain():
        prompt = PromptTemplate(
            template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know.
            
            Question: {question}
            
            Context: {context}
            
            Answer:
            """,
            input_variables=["question", "context"],
        )
        
        return prompt | llm | StrOutputParser()

    rag_chain = create_rag_chain()

    ## graph node functions
    def retrieve(state: GraphState):
        question = state["question"]
        documents = retrieve_from_vector_store(question)
        return {"question": question, "documents": documents}

    def grade_documents(state: GraphState):
        question = state["question"]
        documents = state["documents"]
        
        filtered_docs = []
        for d in documents:
            if is_relevant(d, question):
                filtered_docs.append(d)
        
        return {"question": question, "documents": filtered_docs}

    def transform_query(state: GraphState):
        question = state["question"]
        better_question = rewriter(question)
        return {"question": better_question, "documents": state["documents"]}

    def web_search(state: GraphState):
        question = state["question"]
        
        # Use the retriever which directly returns Document objects
        retriever = TavilySearchAPIRetriever(k=3)
        documents = retriever.invoke(question)
        
        return {"question": question, "documents": documents}

    def generate(state: GraphState):
        question = state["question"]
        documents = state["documents"]
        
        # Format context
        context = "\n\n".join([
            doc.page_content if hasattr(doc, 'page_content') else str(doc) 
            for doc in documents
        ])
        
        # Generate answer
        answer = rag_chain.invoke({"context": context, "question": question})
        
        return {"question": question, "documents": documents, "generation": answer}
    
    def update_memory(state: GraphState):
        messages = state.get('messages', [])
        messages.append(HumanMessage(content=state['question']))
        messages.append(AIMessage(content=state['generation']))
        return {"messages": messages}

    def grade_generation(state: GraphState):
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        
        if not is_grounded(generation, documents):
            ans =  "not grounded"
            return ans
        
        if not answers_question(generation, question):
            ans = "not useful"
            return ans
        
        ans = "useful"
        if ans == "useful":
            update_memory(state)
            return ans

    def decide_to_generate(state: GraphState):
        if state["documents"]:
            return "generate"
        else:
            return "transform_query"

    # Build the graph
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)

    # Add edges
    workflow.add_edge(START, "retrieve")

    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"generate": "generate", "transform_query": "transform_query"}
    )

    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "generate")

    workflow.add_conditional_edges(
        "generate",
        grade_generation,
        {
            "not_grounded": "web_search",
            "not_useful": "transform_query",
            "useful": END,
        },
    )

    # Compile the graph
    app = workflow.compile(checkpointer=memory)
    return app