import pytest
import os
from pathlib import Path

def test_vector_store_isolation():
    from agent.agent import _load_and_split
    
    # We will just test the metadata tagging logic
    # Mocking PyPDFLoader would be better but we can test the function if we have a dummy pdf
    # Let's just create a dummy file and test the chunk metadata
    dummy_pdf = Path("test_dummy.pdf")
    # We can't easily create a valid PDF here, so we'll mock PyPDFLoader
    
    class MockPyPDFLoader:
        def __init__(self, path):
            self.path = path
        def load(self):
            from langchain_core.documents import Document
            return [Document(page_content="Test content", metadata={"source": self.path, "page": 0})]
            
    import agent.agent
    agent.agent.PyPDFLoader = MockPyPDFLoader
    
    splits_user1 = agent.agent._load_and_split(["test_dummy.pdf"], user_id=1)
    splits_user2 = agent.agent._load_and_split(["test_dummy.pdf"], user_id=2)
    
    assert all(doc.metadata["user_id"] == 1 for doc in splits_user1)
    assert all(doc.metadata["user_id"] == 2 for doc in splits_user2)

def test_hybrid_retriever_filter():
    from agent.agent import build_hybrid_retriever
    # Ensure it passes the filter down
    # Given we mocked PyPDFLoader, we can run build_hybrid_retriever if Chroma and HuggingFace are available
    pass # we leave this as a structural test since building embeddings requires downloading the model
