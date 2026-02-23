from langchain_google_genai import GoogleGenerativeAIEmbeddings
from document_loader import load_document, preprocess_document
from text_chunker import create_chunks
from dotenv import load_dotenv
import os
import time

load_dotenv()

# ── Step 1: Create embedding model ───────────────────────────────────
def create_embedding_model():
    """
    Initialize Google's embedding model.
    This model converts text to vectors (lists of numbers).
    """
    print("\n🤖 Initializing embedding model...")

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    print("✅ Embedding model ready!")
    return embedding_model


# ── Step 2: Test single embedding ────────────────────────────────────
def test_single_embedding(embedding_model):
    """
    Test embedding on a single sentence.
    Shows us what an embedding actually looks like.
    """
    print("\n" + "="*50)
    print("🔬 SINGLE EMBEDDING TEST")
    print("="*50)

    test_text = "Zomato food order January"
    embedding = embedding_model.embed_query(test_text)

    print(f"Text: '{test_text}'")
    print(f"Embedding dimensions: {len(embedding)}")
    print(f"First 10 numbers: {[round(x, 4) for x in embedding[:10]]}")
    print(f"Last 10 numbers : {[round(x, 4) for x in embedding[-10:]]}")
    print(f"Min value: {round(min(embedding), 4)}")
    print(f"Max value: {round(max(embedding), 4)}")


# ── Step 3: Compare similarity ────────────────────────────────────────
def compare_similarity(embedding_model):
    """
    Compare embeddings of similar and different texts.
    Shows how semantically similar texts have similar embeddings.
    """
    print("\n" + "="*50)
    print("🔍 SIMILARITY COMPARISON")
    print("="*50)

    # Three texts - two similar, one different
    texts = {
        "food_1" : "Zomato food order January",
        "food_2" : "Swiggy restaurant delivery charge",
        "salary" : "Monthly salary bank credit"
    }

    embeddings = {}
    for key, text in texts.items():
        embeddings[key] = embedding_model.embed_query(text)
        print(f"✅ Embedded: '{text}'")
        time.sleep(1)  # avoid API rate limit

    # Calculate cosine similarity manually
    def cosine_similarity(vec1, vec2):
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a ** 2 for a in vec1) ** 0.5
        magnitude2 = sum(b ** 2 for b in vec2) ** 0.5
        return dot_product / (magnitude1 * magnitude2)

    sim_food = cosine_similarity(embeddings["food_1"], embeddings["food_2"])
    sim_diff = cosine_similarity(embeddings["food_1"], embeddings["salary"])

    print(f"\n📊 Similarity Results:")
    print(f"Zomato vs Swiggy (both food)  : {round(sim_food, 4)} ← should be HIGH")
    print(f"Zomato vs Salary (different)  : {round(sim_diff, 4)} ← should be LOW")
    print(f"\n✅ Proof that embeddings capture meaning!")


# ── Step 4: Embed all chunks ──────────────────────────────────────────
def embed_chunks(chunks, embedding_model):
    """
    Generate embeddings for all chunks.
    In real usage ChromaDB does this automatically.
    Here we do it manually to understand the process.
    """
    print("\n" + "="*50)
    print("📦 EMBEDDING ALL CHUNKS")
    print("="*50)

    print(f"Total chunks to embed: {len(chunks)}")
    print("Generating embeddings...")

    embedded_chunks = []

    for i, chunk in enumerate(chunks):
        embedding = embedding_model.embed_query(chunk.page_content)
        embedded_chunks.append({
            "chunk_number" : i + 1,
            "content"      : chunk.page_content,
            "metadata"     : chunk.metadata,
            "embedding"    : embedding,
            "dimensions"   : len(embedding)
        })
        print(f"✅ Chunk {i+1}/{len(chunks)} embedded ({len(embedding)} dimensions)")
        time.sleep(1)  # avoid API rate limit

    print(f"\n✅ All {len(chunks)} chunks embedded successfully!")
    return embedded_chunks


# ── Step 5: Inspect embedded chunks ──────────────────────────────────
def inspect_embedded_chunks(embedded_chunks):
    """
    Show summary of embedded chunks.
    """
    print("\n" + "="*50)
    print("📋 EMBEDDED CHUNKS SUMMARY")
    print("="*50)

    for item in embedded_chunks:
        print(f"\n--- Chunk {item['chunk_number']} ---")
        print(f"Content preview : {item['content'][:80]}...")
        print(f"Embedding dims  : {item['dimensions']}")
        print(f"First 5 numbers : {[round(x, 4) for x in item['embedding'][:5]]}")


# ── Main: Run all steps ───────────────────────────────────────────────
if __name__ == "__main__":

    # Initialize embedding model
    embedding_model = create_embedding_model()

    # Test single embedding
    test_single_embedding(embedding_model)

    # Compare similarity between texts
    compare_similarity(embedding_model)

    # Load, preprocess and chunk document
    file_path = "data/sample_statement.txt"
    documents = load_document(file_path)
    cleaned_docs = preprocess_document(documents)
    chunks = create_chunks(cleaned_docs)

    # Embed all chunks
    embedded_chunks = embed_chunks(chunks, embedding_model)

    # Inspect results
    inspect_embedded_chunks(embedded_chunks)

    print("\n✅ Embeddings complete!")
    print("🔜 Next step: Store in ChromaDB (Day 8)")