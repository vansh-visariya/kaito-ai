# chatgpt
This project is build a application that have all the features like  File Uploads, Web Search, Reasoning, Chat History, New Chat,  Voice Assistant

chatgpt/
│
├── .gitignore            # Specifies which files Git should ignore
├── app.py                # The main entry point for your Streamlit application
├── requirements.txt      # Lists all Python dependencies for the project
├── README.md             # Project description, setup, and usage instructions
│
└───src/
    │
    ├── __init__.py         # Makes the 'src' directory a Python package
    ├── agent.py            # Contains the LangGraph agent logic and graph definition
    ├── tools.py            # Defines custom tools (file processor, web search)
    └── utils.py            # Houses helper functions (e.g., voice processing)