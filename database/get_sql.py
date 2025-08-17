import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

## local database
def get_sqlite_connection():
    conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
    memory = SqliteSaver(conn=conn)
    return memory