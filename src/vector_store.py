from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
import time

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────
FAISS_DB_PATH   = "faiss_db"
COLLECTION_NAME = "bank_statements"

# ── Step 1: Get embedding model ───────────────────────────────────────
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
    Create FAISS vector store from document chunks.

    Args:
        chunks         : List of LangChain Document objects
        embedding_model: Google embedding model instance

    Returns:
        FAISS vector store instance
    """
    print("\n🗄️  Creating FAISS vector store...")
    print(f"📦 Total chunks to store: {len(chunks)}")

    # Batch settings for rate limiting
    BATCH_SIZE  = 50
    BATCH_DELAY = 65

    if len(chunks) <= BATCH_SIZE:
        # Small document — process all at once
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embedding_model
        )
        print(f"✅ All {len(chunks)} chunks stored!")

    else:
        # Large document — process in batches
        print(f"📦 Large document — batches of {BATCH_SIZE}")

        first_batch  = chunks[:BATCH_SIZE]
        vector_store = FAISS.from_documents(
            documents=first_batch,
            embedding=embedding_model
        )
        print(f"✅ Batch 1 done ({len(first_batch)} chunks)")

        remaining = chunks[BATCH_SIZE:]
        batch_num = 2

        while remaining:
            print(f"⏳ Waiting {BATCH_DELAY}s...")
            time.sleep(BATCH_DELAY)
            batch     = remaining[:BATCH_SIZE]
            remaining = remaining[BATCH_SIZE:]
            vector_store.add_documents(batch)
            print(f"✅ Batch {batch_num} done ({len(batch)} chunks)")
            batch_num += 1

    print(f"✅ Vector store created successfully!")
    return vector_store


# ── Step 3: Load existing vector store ───────────────────────────────
def load_vector_store(embedding_model):
    """
    Load existing FAISS vector store from disk.
    """
    print("\n📂 Loading FAISS vector store...")

    if not os.path.exists(FAISS_DB_PATH):
        raise FileNotFoundError(
            f"No vector store found at {FAISS_DB_PATH}. "
            f"Run create_vector_store() first."
        )

    vector_store = FAISS.load_local(
        folder_path=FAISS_DB_PATH,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

    print(f"✅ FAISS vector store loaded!")
    return vector_store


# ── Step 4: Search vector store ───────────────────────────────────────
def search_vector_store(vector_store, query, top_k=3):
    """
    Search FAISS for chunks relevant to query.

    Args:
        vector_store: FAISS vector store instance
        query       : User question as string
        top_k       : Number of chunks to retrieve

    Returns:
        List of (Document, score) tuples
    """
    print(f"\n🔍 Searching for: '{query}'")
    print(f"📊 Retrieving top {top_k} chunks...")

    results = vector_store.similarity_search_with_score(
        query=query,
        k=top_k
    )

    print(f"✅ Found {len(results)} relevant chunks")
    return results


# ── Main: Test FAISS setup ────────────────────────────────────────────
if __name__ == "__main__":

    from document_loader import load_document, preprocess_document
    from text_chunker import create_chunks

    embedding_model = get_embedding_model()

    file_path    = "data/sample_statement.txt"
    documents    = load_document(file_path)
    cleaned_docs = preprocess_document(documents)
    chunks       = create_chunks(cleaned_docs)

    vector_store = create_vector_store(chunks, embedding_model)

    results = search_vector_store(
        vector_store,
        "How much did I spend on Zomato?",
        top_k=3
    )

    print("\n" + "="*50)
    print("📋 SEARCH RESULTS")
    print("="*50)
    for i, (doc, score) in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Score  : {round(score, 4)}")
        print(f"Content: {doc.page_content}")

    print("\n✅ FAISS setup complete!")


