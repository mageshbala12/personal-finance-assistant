import chromadb
from chromadb.config import Settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from document_loader import load_document, preprocess_document
from text_chunker import create_chunks
from dotenv import load_dotenv
import os
import shutil

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "bank_statements"

# ── Step 1: Create embedding model ────────────────────────────────────
def get_embedding_model():
    """
    Initialize and return Google embedding model.
    """
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )


# ── Step 2: Create vector store ───────────────────────────────────────
def create_vector_store(chunks, embedding_model):
    """
    Create ChromaDB vector store from document chunks.
    Stores embeddings on disk so they persist between sessions.

    Args:
        chunks         : List of LangChain Document objects
        embedding_model: Google embedding model instance

    Returns:
        ChromaDB vector store instance
    """
    print("\n🗄️  Creating ChromaDB vector store...")

    # Delete existing database if it exists
    # This ensures fresh start when re-indexing documents
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
        print("🗑️  Deleted existing database")

    # Create new vector store from chunks
    # This automatically:
    # 1. Generates embeddings for each chunk
    # 2. Stores chunks + embeddings in ChromaDB
    # 3. Saves to disk at CHROMA_DB_PATH
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME
    )

    print(f"✅ Vector store created!")
    print(f"📁 Saved to: {CHROMA_DB_PATH}/")
    print(f"📦 Total chunks stored: {len(chunks)}")

    return vector_store


# ── Step 3: Load existing vector store ───────────────────────────────
def load_vector_store(embedding_model):
    """
    Load existing ChromaDB vector store from disk.
    Use this instead of recreating every time.

    Args:
        embedding_model: Google embedding model instance

    Returns:
        ChromaDB vector store instance
    """
    print("\n📂 Loading existing vector store...")

    if not os.path.exists(CHROMA_DB_PATH):
        raise FileNotFoundError(
            f"No vector store found at {CHROMA_DB_PATH}. "
            f"Run create_vector_store() first."
        )

    vector_store = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME
    )

    print(f"✅ Vector store loaded from {CHROMA_DB_PATH}/")
    return vector_store


# ── Step 4: Search vector store ───────────────────────────────────────
def search_vector_store(vector_store, query, top_k=3):
    """
    Search vector store for chunks relevant to query.

    Args:
        vector_store: ChromaDB vector store instance
        query       : User's question as string
        top_k       : Number of chunks to retrieve (default 3)

    Returns:
        List of relevant Document objects with scores
    """
    print(f"\n🔍 Searching for: '{query}'")
    print(f"📊 Retrieving top {top_k} chunks...")

    # similarity_search_with_score returns chunks AND their scores
    results = vector_store.similarity_search_with_score(
        query=query,
        k=top_k
    )

    print(f"✅ Found {len(results)} relevant chunks")
    return results


# ── Step 5: Inspect search results ────────────────────────────────────
def inspect_search_results(results):
    """
    Display search results in readable format.
    Shows which chunks were retrieved and their similarity scores.
    """
    print("\n" + "="*50)
    print("📋 SEARCH RESULTS")
    print("="*50)

    for i, (doc, score) in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Similarity Score : {round(score, 4)}")
        print(f"Source           : {doc.metadata.get('source', 'unknown')}")
        print(f"Content:")
        print("-"*30)
        print(doc.page_content)
        print("-"*30)


# ── Main: Run all steps ───────────────────────────────────────────────
if __name__ == "__main__":

    # Get embedding model
    embedding_model = get_embedding_model()

    # Load and process document
    file_path = "data/sample_statement.txt"
    documents = load_document(file_path)
    cleaned_docs = preprocess_document(documents)
    chunks = create_chunks(cleaned_docs)

    # Create vector store
    vector_store = create_vector_store(chunks, embedding_model)

    print("\n" + "="*50)
    print("🧪 TESTING SEARCH QUERIES")
    print("="*50)

    # Test Query 1 — Food expenses
    results1 = search_vector_store(
        vector_store,
        "How much did I spend on Zomato food orders?",
        top_k=3
    )
    inspect_search_results(results1)

    # Test Query 2 — SIP investments
    results2 = search_vector_store(
        vector_store,
        "What are my SIP investments this month?",
        top_k=3
    )
    inspect_search_results(results2)

    # Test Query 3 — Salary
    results3 = search_vector_store(
        vector_store,
        "When was my salary credited?",
        top_k=2
    )
    inspect_search_results(results3)

    # Test loading from disk
    print("\n" + "="*50)
    print("💾 TESTING PERSISTENCE")
    print("="*50)
    print("Loading vector store from disk...")
    loaded_store = load_vector_store(embedding_model)
    results4 = search_vector_store(
        loaded_store,
        "What is my account balance?",
        top_k=2
    )
    inspect_search_results(results4)

    print("\n✅ ChromaDB setup complete!")
    print("🔜 Next step: Build full RAG pipeline (Day 9)")