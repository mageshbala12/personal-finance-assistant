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