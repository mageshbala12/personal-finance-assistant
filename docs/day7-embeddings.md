📘 Day 7 Documentation — Embeddings
Personal Finance Assistant Project

🎯 Objective
Understand embeddings deeply, convert text chunks into vectors using Google's embedding model, and prove that similar texts produce similar embeddings.

📚 Part 1 — Key Concepts
What are Embeddings?
Embeddings convert text into lists of numbers (vectors) that capture meaning — not just characters.
Why not simple number mapping?
Simple approach (wrong):
"Zomato" → 1
"Swiggy" → 2
"Salary" → 3
Problem: Numbers have no meaningful relationship to content
Embedding approach (correct):
"Zomato food order"     → [0.2, 0.8, 0.1, 0.9, 0.3, ...]
"Swiggy food delivery"  → [0.21, 0.79, 0.12, 0.88, 0.31, ...]
"Monthly salary credit" → [0.9, 0.1, 0.7, 0.2, 0.8, ...]

Zomato and Swiggy → numerically CLOSE → similar meaning ✅
Salary → numerically FAR from both → different meaning ✅

What Does an Embedding Look Like?
An embedding is a vector — a list of floating point numbers. Google's gemini-embedding-001 model produces 768 numbers per text:
python"01-Jan Zomato Food Order 850" →

[0.023, -0.156, 0.891, 0.234, -0.567, 0.123,
 0.445, -0.234, 0.678, 0.012, -0.345, 0.789,
 ... 768 numbers total ...]
```

Each number captures a different **dimension of meaning** — topic, sentiment, context, word relationships.

---

### How Similarity is Measured — Cosine Similarity
```
Question embedding:  [0.21, 0.79, 0.11, 0.88]
Chunk 1 embedding:   [0.20, 0.80, 0.10, 0.90] → similarity: 0.99 ✅ very similar
Chunk 2 embedding:   [0.90, 0.10, 0.70, 0.20] → similarity: 0.21 ❌ not similar
Chunks with highest similarity scores are retrieved and sent to Gemini.

Why Google's Embedding Model?
ReasonDetailFreeIncluded with Gemini API keySame providerNo extra accounts neededHigh qualityTrained on massive multilingual dataIndian language supportWorks with Hindi, Tamil etc.
Model used: models/gemini-embedding-001

Note: Original code used models/embedding-001 which gave 404 error. Listed available models and found correct name models/gemini-embedding-001.


💻 Part 2 — Code Created
File: src/embeddings_manager.py
pythonfrom langchain_google_genai import GoogleGenerativeAIEmbeddings
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

    texts = {
        "food_1" : "Zomato food order January",
        "food_2" : "Swiggy restaurant delivery charge",
        "salary" : "Monthly salary bank credit"
    }

    embeddings = {}
    for key, text in texts.items():
        embeddings[key] = embedding_model.embed_query(text)
        print(f"✅ Embedded: '{text}'")
        time.sleep(1)

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
        time.sleep(1)

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

    embedding_model = create_embedding_model()
    test_single_embedding(embedding_model)
    compare_similarity(embedding_model)

    file_path = "data/sample_statement.txt"
    documents = load_document(file_path)
    cleaned_docs = preprocess_document(documents)
    chunks = create_chunks(cleaned_docs)

    embedded_chunks = embed_chunks(chunks, embedding_model)
    inspect_embedded_chunks(embedded_chunks)

    print("\n✅ Embeddings complete!")
    print("🔜 Next step: Store in ChromaDB (Day 8)")

🔍 Part 3 — Detailed Code Explanation
Imports
pythonfrom langchain_google_genai import GoogleGenerativeAIEmbeddings
from document_loader import load_document, preprocess_document
from text_chunker import create_chunks
from dotenv import load_dotenv
import os
import time
ImportPurposeGoogleGenerativeAIEmbeddingsLangChain wrapper for Google's embedding modelload_document, preprocess_documentOur Day 5 functions — reused herecreate_chunksOur Day 6 function — reused hereosRead environment variablestimetime.sleep() to avoid hitting API rate limits
Key concept — reusing our own code:
pythonfrom document_loader import load_document, preprocess_document
from text_chunker import create_chunks
We are building a pipeline — each day's code builds on the previous day. Instead of rewriting, we import and reuse. This is professional software development practice.

Function 1 — create_embedding_model()
pythonembedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)
GoogleGenerativeAIEmbeddings — LangChain's wrapper class for Google's embedding API. Handles all the complexity of making API calls and returning vectors.
model="models/gemini-embedding-001" — Specifies which Google embedding model to use. This model was found by listing available models:
pythonfor model in genai.list_models():
    if 'embedContent' in model.supported_generation_methods:
        print(model.name)
Why store as variable and return?
pythonembedding_model = create_embedding_model()
We create the model once and reuse it for all embedding operations. Creating it multiple times would be wasteful — each creation makes an API connection.

Function 2 — test_single_embedding()
pythontest_text = "Zomato food order January"
embedding = embedding_model.embed_query(test_text)
embed_query() — converts a single text string into a vector. Returns a Python list of floating point numbers.
pythonprint(f"Embedding dimensions: {len(embedding)}")
len(embedding) counts the numbers in the vector. Google's model always returns 768 dimensions.
pythonprint(f"First 10 numbers: {[round(x, 4) for x in embedding[:10]]}")
round(x, 4) → rounds each number to 4 decimal places for readable output.
embedding[:10] → first 10 numbers from the 768-number vector.
[round(x, 4) for x in ...] → list comprehension applying rounding to each number.
pythonprint(f"Min value: {round(min(embedding), 4)}")
print(f"Max value: {round(max(embedding), 4)}")
Embedding values typically range between -1 and +1. Negative values are as meaningful as positive — they represent different directions in the meaning space.

Function 3 — compare_similarity()
pythontexts = {
    "food_1" : "Zomato food order January",
    "food_2" : "Swiggy restaurant delivery charge",
    "salary" : "Monthly salary bank credit"
}
Dictionary — stores key-value pairs. Key is our label ("food_1"), value is the text. Makes it easy to reference each text by name.
pythonembeddings = {}
for key, text in texts.items():
    embeddings[key] = embedding_model.embed_query(text)
    print(f"✅ Embedded: '{text}'")
    time.sleep(1)
texts.items() → loops through dictionary giving both key and value at the same time.
embeddings[key] = ... → stores each embedding in a new dictionary using same key.
time.sleep(1) → pauses 1 second between API calls to avoid rate limit errors.
After this loop:
pythonembeddings = {
    "food_1": [0.2, 0.8, 0.1, ...],   # 768 numbers
    "food_2": [0.21, 0.79, 0.12, ...], # 768 numbers
    "salary": [0.9, 0.1, 0.7, ...]     # 768 numbers
}
Cosine Similarity function:
pythondef cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a ** 2 for a in vec1) ** 0.5
    magnitude2 = sum(b ** 2 for b in vec2) ** 0.5
    return dot_product / (magnitude1 * magnitude2)
This is a nested function — a function defined inside another function. It's only accessible within compare_similarity().
Breaking down the math:
zip(vec1, vec2) — pairs up numbers from both vectors:
pythonvec1 = [0.2, 0.8, 0.1]
vec2 = [0.21, 0.79, 0.12]
zip  = [(0.2, 0.21), (0.8, 0.79), (0.1, 0.12)]
dot_product — multiplies paired numbers and sums them:
python(0.2 × 0.21) + (0.8 × 0.79) + (0.1 × 0.12) = 0.686
magnitude — length of the vector:
pythonsum(a ** 2 for a in vec1) ** 0.5
= (0.2² + 0.8² + 0.1²) ** 0.5
= (0.04 + 0.64 + 0.01) ** 0.5
= 0.69 ** 0.5
= 0.831
** 0.5 means square root (0.5 power = square root).
Final result:
pythonreturn dot_product / (magnitude1 * magnitude2)
Result is always between -1 and 1:

1.0 → identical meaning
0.8+ → very similar meaning
0.5 → somewhat related
0.2- → very different meaning


Function 4 — embed_chunks()
pythonembedded_chunks = []

for i, chunk in enumerate(chunks):
    embedding = embedding_model.embed_query(chunk.page_content)
    embedded_chunks.append({
        "chunk_number" : i + 1,
        "content"      : chunk.page_content,
        "metadata"     : chunk.metadata,
        "embedding"    : embedding,
        "dimensions"   : len(embedding)
    })
    time.sleep(1)
embedded_chunks.append({...}) — adds a dictionary to the list for each chunk. Each dictionary contains all information about that chunk including its embedding.
Why store everything in a dictionary?
Keeps all related data together:
python{
    "chunk_number": 1,
    "content"     : "HDFC BANK - ACCOUNT STATEMENT...",
    "metadata"    : {"source": "data/sample_statement.txt"},
    "embedding"   : [0.023, -0.156, 0.891, ...],  # 768 numbers
    "dimensions"  : 768
}
Important note: In Day 8 with ChromaDB we won't need to manually embed chunks — ChromaDB does it automatically. We do it manually here purely to understand what's happening under the hood.

Function 5 — inspect_embedded_chunks()
pythonprint(f"Content preview : {item['content'][:80]}...")
print(f"Embedding dims  : {item['dimensions']}")
print(f"First 5 numbers : {[round(x, 4) for x in item['embedding'][:5]]}")
```

`item['content'][:80]` → first 80 characters of chunk content as preview.
`item['embedding'][:5]` → first 5 numbers from 768-number embedding.
`...` at the end → visually indicates content is truncated.

---

## 🗺️ RAG Pipeline Progress
```
✅ Step 1: Load Document       ← Day 5
✅ Step 2: Preprocess Text     ← Day 5
✅ Step 3: Split into Chunks   ← Day 6
✅ Step 4: Create Embeddings   ← Done today!
⏳ Step 5: Store in ChromaDB   ← Day 8
⏳ Step 6: Query & Retrieve    ← Day 9
⏳ Step 7: Generate Answer     ← Day 9
⏳ Step 8: Integrate into UI   ← Day 10

💡 Key Python Concepts Learned
ConceptExampleMeaningDictionary{"key": "value"}Store key-value pairsdict.items()for k, v in dict.items()Loop through key and value togetherNested functiondef func() inside def func()Function only accessible within parent functionzip()zip(vec1, vec2)Pair up elements from two lists** 0.5value ** 0.5Square root** 2value ** 2Square (power of 2)time.sleep(1)Pause 1 secondAvoid API rate limit errorsround(x, 4)round(0.12345, 4)Round to 4 decimal places

⚠️ Issues Faced & Solutions
IssueSolution404 NOT_FOUND models/embedding-001Listed available embedding models using genai.list_models(). Found correct model name models/gemini-embedding-001 and updated code

✅ Day 7 Checklist

 Understand what embeddings are and why they work
 Understand what a vector is and its dimensions
 Understand cosine similarity concept and math
 Created src/embeddings_manager.py
 Fixed embedding model name from embedding-001 to gemini-embedding-001
 Tested single embedding — confirmed 768 dimensions
 Compared similarity — food texts scored HIGH, salary scored LOW
 Embedded all chunks manually to understand the process
 Committed and pushed to GitHub
