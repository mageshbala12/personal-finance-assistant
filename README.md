# 💰 Personal Finance Assistant

A personal finance chatbot powered by Google Gemini AI and RAG.

## Features
- Upload bank statements (PDF or TXT)
- Ask questions about YOUR personal finances
- Multi-turn conversation with memory
- Source references for every answer
- View retrieved document chunks

## Tech Stack
- Python
- Google Gemini AI
- LangChain
- ChromaDB
- Streamlit

## Stages
- [x] Stage 0 - Basic Chatbot
- [x] Stage 1 - RAG with documents
- [ ] Stage 2 - AI Agents
- [ ] Stage 3 - Agentic AI with MCP

## How to Run Locally
```bash
# Activate virtual environment
venv\Scripts\activate

# Run RAG chatbot
streamlit run src/rag_chatbot.py
```