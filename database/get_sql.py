import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

## local database
## for search agent
def get_search_memory():
    conn = sqlite3.connect(database='database/search_chatbot.db', check_same_thread=False)
    return SqliteSaver(conn=conn)

## for rag agent
def get_rag_memory():
    conn = sqlite3.connect(database='database/rag_chatbot.db', check_same_thread=False)
    return SqliteSaver(conn=conn)