Day 4 Documentation — RAG Concepts & LangChain Setup
Personal Finance Assistant Project

🎯 Objective
Understand RAG architecture deeply and set up LangChain framework with Gemini integration.

📚 Part 1 — RAG Architecture
What is RAG?
RAG = Retrieval Augmented Generation
Word	Meaning
Retrieval	Fetch relevant information from your documents
Augmented	Add that information to your question
Generation	Gemini generates answer using your information
Two Phases of RAG
Phase 1 — Indexing (Done once):
Your PDF/Text File
       ↓
Load Document
       ↓
Clean & Preprocess Text
       ↓
Split into Chunks
       ↓
Convert Chunks to Embeddings
       ↓
Store in Vector Database (ChromaDB)
       ↓
✅ Ready to answer questions
Phase 2 — Querying (Every time user asks):
User Question
       ↓
Convert Question to Embedding
       ↓
Search Vector DB for Similar Chunks
       ↓
Retrieve Top 3-5 Relevant Chunks
       ↓
Question + Chunks → Gemini
       ↓
Answer shown to user

How Embeddings Work
Embeddings convert text to numbers that capture meaning:
"Zomato food order"     → [0.2, 0.8, 0.1, 0.9]
"food delivery expense" → [0.21, 0.79, 0.11, 0.88]
"salary credit"         → [0.9, 0.1, 0.7, 0.2]
First two are numerically close → similar meaning → found together in search. Third is numerically far → different meaning → not returned for food queries.
This is called Semantic Search — finding content by meaning, not exact words.

RAG vs Fine Tuning
    RAG	Fine Tuning
Cost	Very cheap	Very expensive
Time	Minutes	Days/weeks
Update docs	Just re-index	Retrain model
Private data	✅ Safe	⚠️ Risk of leakage
Best for	Personal documents	Changing model behavior
Conclusion: RAG is always the right choice for personal documents.

📚 Part 2 — LangChain Overview
What is LangChain?
A framework with pre-built components (like LEGO blocks) for building AI applications.
LangChain Components Used in Our Project
Component	Purpose	Stage Used
Document Loaders	Load PDFs, text files	Stage 1 RAG
Text Splitters	Split documents into chunks	Stage 1 RAG
Embeddings	Convert text to numbers	Stage 1 RAG
Vector Stores	Store and search embeddings	Stage 1 RAG
Chains	Connect components together	Stage 1 RAG
Agents	Give AI ability to use tools	Stage 2 Agents
LangGraph	Multi-agent orchestration	Stage 3 Agentic AI

⚙️ Part 3 — Installation & Setup
Packages Installed
bash
pip install langchain langchain-community langchain-google-genai chromadb pypdf langchain-text-splitters
Package	Purpose
langchain	Core LangChain framework
langchain-community	Community integrations, document loaders
langchain-google-genai	LangChain integration for Gemini
chromadb	Vector database to store embeddings
pypdf	Read and extract text from PDF files
langchain-text-splitters	Text splitting utilities
pip Upgrade
bash
python.exe -m pip install --upgrade pip
Updated pip from 24.2 to 26.0.1

Verification File — src/test_langchain.py
python
# Test LangChain installation
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from dotenv import load_dotenv
import os
load_dotenv()
print("✅ LangChain imported successfully")
print("✅ ChromaDB imported successfully")
# Test Gemini connection via LangChain
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY")
)
response = llm.invoke("What is RAG in AI? Answer in one sentence.")
print(f"\n✅ Gemini via LangChain working!")
print(f"Response: {response.content}")
```
### Updated `requirements.txt`
```
google-generativeai
streamlit
python-dotenv
langchain
langchain-community
langchain-google-genai
langchain-text-splitters
chromadb
pypdf

⚠️ Issues Faced & Solutions
Issue	Solution
ModuleNotFoundError: No module named 'langchain.text_splitter'	LangChain moved this module. Changed import to from langchain_text_splitters import RecursiveCharacterTextSplitter and ran pip install langchain-text-splitters
pip notice about new version	Ran python.exe -m pip install --upgrade pip to upgrade

✅ Day 4 Checklist
    • Understand RAG architecture — indexing and querying phases
    • Understand embeddings and semantic search
    • Understand why RAG beats fine tuning for personal documents
    • Understand what LangChain is and its components
    • Switched to feature/rag branch
    • Installed all LangChain packages
    • Fixed langchain.text_splitter import error
    • test_langchain.py ran successfully
    • Gemini verified working via LangChain
    • Updated requirements.txt
    • Committed and pushed to GitHub
