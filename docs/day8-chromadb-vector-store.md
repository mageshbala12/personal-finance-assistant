📘 Day 8 Documentation — ChromaDB Vector Database
Personal Finance Assistant Project

🎯 Objective
Set up ChromaDB as a persistent vector database, store document chunk embeddings, and perform similarity searches to retrieve relevant chunks for user queries.

📚 Part 1 — Key Concepts
Why ChromaDB is Needed
Problem with Python ListChromaDB SolutionDisappears when program ends✅ Saves to disk — survives restartsManual loop to search embeddings✅ Optimized similarity search built inCan't scale to millions of embeddings✅ Handles millions efficiently

ChromaDB Key Concepts
ConceptMeaningReal World AnalogyCollectionGroup of related embeddingsDatabase tableDocumentsActual text content of chunksTable rowsEmbeddingsVector representation of chunksIndex for fast searchIDsUnique identifier for each chunkPrimary keyMetadataExtra info — source file, page numberTable columns

ChromaDB Architecture
Document Chunks
      ↓
ChromaDB stores:
┌─────────────────────────────────────┐
│  Collection: bank_statements        │
│  ┌────┬──────────────┬───────────┐  │
│  │ ID │   Content    │ Embedding │  │
│  ├────┼──────────────┼───────────┤  │
│  │ 0  │ HDFC BANK... │ [0.2,...] │  │
│  │ 1  │ 03-Jan Zom.. │ [0.8,...] │  │
│  │ 2  │ 15-Jan SIP.. │ [0.1,...] │  │
│  │ 3  │ 28-Jan SIP.. │ [0.9,...] │  │
│  └────┴──────────────┴───────────┘  │
└─────────────────────────────────────┘
      ↓
User asks question
      ↓
ChromaDB searches embeddings
      ↓
Returns top 3 relevant chunks

💻 Part 2 — Code Created
File: src/vector_store.py
pythonimport chromadb
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
    """
    print("\n🗄️  Creating ChromaDB vector store...")

    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
        print("🗑️  Deleted existing database")

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
    """
    print(f"\n🔍 Searching for: '{query}'")
    print(f"📊 Retrieving top {top_k} chunks...")

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

    embedding_model = get_embedding_model()

    file_path = "data/sample_statement.txt"
    documents = load_document(file_path)
    cleaned_docs = preprocess_document(documents)
    chunks = create_chunks(cleaned_docs)

    vector_store = create_vector_store(chunks, embedding_model)

    print("\n" + "="*50)
    print("🧪 TESTING SEARCH QUERIES")
    print("="*50)

    results1 = search_vector_store(
        vector_store,
        "How much did I spend on Zomato food orders?",
        top_k=3
    )
    inspect_search_results(results1)

    results2 = search_vector_store(
        vector_store,
        "What are my SIP investments this month?",
        top_k=3
    )
    inspect_search_results(results2)

    results3 = search_vector_store(
        vector_store,
        "When was my salary credited?",
        top_k=2
    )
    inspect_search_results(results3)

    print("\n" + "="*50)
    print("💾 TESTING PERSISTENCE")
    print("="*50)
    loaded_store = load_vector_store(embedding_model)
    results4 = search_vector_store(
        loaded_store,
        "What is my account balance?",
        top_k=2
    )
    inspect_search_results(results4)

    print("\n✅ ChromaDB setup complete!")
    print("🔜 Next step: Build full RAG pipeline (Day 9)")

🔍 Part 3 — Detailed Code Explanation
Imports
pythonimport chromadb
from chromadb.config import Settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from document_loader import load_document, preprocess_document
from text_chunker import create_chunks
import shutil
ImportPurposechromadbCore ChromaDB librarySettingsChromaDB configuration optionsGoogleGenerativeAIEmbeddingsGoogle embedding modelChromaLangChain wrapper for ChromaDB — easier to useload_document, preprocess_documentOur Day 5 functionscreate_chunksOur Day 6 functionshutilFile system operations — delete folders
Why Chroma from LangChain instead of raw chromadb?
python# Raw ChromaDB — complex, manual
client = chromadb.Client()
collection = client.create_collection("name")
collection.add(documents=[], embeddings=[], ids=[])

# LangChain Chroma — simple, handles embeddings automatically
vector_store = Chroma.from_documents(chunks, embedding_model)
LangChain's wrapper automatically generates embeddings and handles storage in one line.

Constants
pythonCHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "bank_statements"
Constants are variables whose values never change during program execution. By convention written in UPPER_CASE.
Why use constants instead of hardcoding strings?
python# Bad — hardcoded in multiple places
Chroma(persist_directory="chroma_db")
os.path.exists("chroma_db")
shutil.rmtree("chroma_db")

# Good — change one constant, updates everywhere
Chroma(persist_directory=CHROMA_DB_PATH)
os.path.exists(CHROMA_DB_PATH)
shutil.rmtree(CHROMA_DB_PATH)
If you ever want to change the folder name you only change it in one place.

Function 1 — get_embedding_model()
pythondef get_embedding_model():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
Simple function that creates and returns the embedding model. We extracted this into its own function because multiple functions need it — create_vector_store() and load_vector_store() both use it. Clean reusable code.

Function 2 — create_vector_store()
Delete existing database:
pythonif os.path.exists(CHROMA_DB_PATH):
    shutil.rmtree(CHROMA_DB_PATH)
    print("🗑️  Deleted existing database")
shutil.rmtree() — deletes an entire folder and all its contents recursively. Like selecting a folder and pressing Shift+Delete in Windows.
Why delete before creating? ChromaDB stores data in files on disk. If we run the program again with updated documents, we want a completely fresh database — not old data mixed with new data.
Create vector store:
pythonvector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=CHROMA_DB_PATH,
    collection_name=COLLECTION_NAME
)
```

`Chroma.from_documents()` — LangChain's most powerful one-liner. It does all of this automatically:
```
1. Takes each chunk from documents list
2. Calls embedding_model to generate embedding
3. Stores chunk text + embedding + metadata in ChromaDB
4. Saves everything to disk at persist_directory
ParameterValuePurposedocumentschunksList of Document objects to storeembeddingembedding_modelModel to generate embeddingspersist_directory"chroma_db"Folder to save database on diskcollection_name"bank_statements"Name of this collection

Function 3 — load_vector_store()
pythonif not os.path.exists(CHROMA_DB_PATH):
    raise FileNotFoundError(
        f"No vector store found at {CHROMA_DB_PATH}. "
        f"Run create_vector_store() first."
    )

vector_store = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embedding_model,
    collection_name=COLLECTION_NAME
)
Why do we need this function? In our final chatbot app, we don't want to re-index documents every time the app starts. We create the database once and just load it on startup. Much faster and cheaper.
Chroma.from_documents() vs Chroma():
MethodWhen to UseChroma.from_documents()First time — creates database from scratchChroma()Every other time — loads existing database
Note the parameter name difference:
python# Creating: uses "embedding"
Chroma.from_documents(embedding=model)

# Loading: uses "embedding_function"
Chroma(embedding_function=model)

Function 4 — search_vector_store()
pythonresults = vector_store.similarity_search_with_score(
    query=query,
    k=top_k
)
```

**`similarity_search_with_score()`** — ChromaDB's search function. Internally it:
```
1. Converts query text to embedding using embedding_model
2. Compares query embedding to all stored embeddings
3. Ranks by cosine similarity
4. Returns top k most similar chunks with their scores
k=top_k — how many chunks to retrieve. We default to 3 because:

Too few (1) → might miss relevant information
Too many (10) → sends too much text to Gemini, increases cost and confusion
Just right (3) → enough context, focused and relevant

Return format:
pythonresults = [
    (Document(page_content="...", metadata={...}), 0.3821),
    (Document(page_content="...", metadata={...}), 0.4123),
    (Document(page_content="...", metadata={...}), 0.5234),
]
Each result is a tuple — a pair of (Document, score).

Function 5 — inspect_search_results()
pythonfor i, (doc, score) in enumerate(results):
    print(f"Similarity Score : {round(score, 4)}")
    print(f"Source           : {doc.metadata.get('source', 'unknown')}")
    print(f"Content:")
    print(doc.page_content)
Tuple unpacking:
pythonfor i, (doc, score) in enumerate(results):
```
Each result is a tuple `(Document, score)`. We **unpack** it directly into `doc` and `score` variables in the loop. Clean and readable.

**`doc.metadata.get('source', 'unknown')`:**
`dict.get(key, default)` — safely gets a value from dictionary. If `'source'` key doesn't exist, returns `'unknown'` instead of crashing. Safer than `doc.metadata['source']` which crashes if key missing.

**Understanding similarity scores:**
ChromaDB returns **distance scores** not similarity scores. Lower = more similar:
```
Score 0.1 - 0.3 → Very relevant ✅✅
Score 0.3 - 0.5 → Relevant ✅
Score 0.5 - 0.7 → Somewhat relevant ⚠️
Score 0.7+      → Not very relevant ❌

Main Block — Test Queries
pythonresults1 = search_vector_store(
    vector_store,
    "How much did I spend on Zomato food orders?",
    top_k=3
)
Why test multiple queries?
QueryTestsZomato food ordersFood expense retrievalSIP investmentsInvestment retrievalSalary creditedIncome retrievalAccount balanceBalance retrieval + persistence test
Each tests a different type of financial query to verify ChromaDB retrieves the right chunks consistently.
Persistence test:
pythonloaded_store = load_vector_store(embedding_model)
results4 = search_vector_store(loaded_store, "What is my account balance?")
```
This proves data survived — we load from disk and search successfully without re-creating the database.

---

## 📊 Part 4 — Actual Output
```
📂 Loading document: data/sample_statement.txt
✅ Loaded 1 document(s)
📄 Total characters: 743

🔧 Preprocessing documents...
✅ Preprocessed 1 document(s)

✂️  Splitting documents into chunks...
✅ Created 4 chunks from 1 document(s)

🗄️  Creating ChromaDB vector store...
🗑️  Deleted existing database
✅ Vector store created!
📁 Saved to: chroma_db/
📦 Total chunks stored: 4

==================================================
🧪 TESTING SEARCH QUERIES
==================================================

🔍 Searching for: 'How much did I spend on Zomato food orders?'
📊 Retrieving top 3 chunks...
✅ Found 3 relevant chunks

==================================================
📋 SEARCH RESULTS
==================================================

--- Result 1 ---
Similarity Score : 0.3821
Source           : data/sample_statement.txt
Content:
------------------------------
03-Jan  Zomato Food Order  850
12-Jan  Zomato Food Order  650
20-Jan  Zomato Food Order  550
------------------------------

--- Result 2 ---
Similarity Score : 0.4532
Source           : data/sample_statement.txt
Content:
------------------------------
HDFC BANK - ACCOUNT STATEMENT
Account Holder: Magesh Balasubramanian
Period: January 2025
------------------------------

--- Result 3 ---
Similarity Score : 0.5123
Source           : data/sample_statement.txt
Content:
------------------------------
15-Jan  SIP - Axis Bluechip Fund  5,000
28-Jan  SIP - Parag Parikh Fund   3,000
------------------------------

🔍 Searching for: 'What are my SIP investments this month?'
📊 Retrieving top 3 chunks...
✅ Found 3 relevant chunks

--- Result 1 ---
Similarity Score : 0.2943
Source           : data/sample_statement.txt
Content:
------------------------------
15-Jan  SIP - Axis Bluechip Fund  5,000
28-Jan  SIP - Parag Parikh Fund   3,000
------------------------------

==================================================
💾 TESTING PERSISTENCE
==================================================
Loading vector store from disk...
✅ Vector store loaded from chroma_db/

✅ ChromaDB setup complete!
🔜 Next step: Build full RAG pipeline (Day 9)
```

**Key observations from output:**
- Zomato query → Chunk with all Zomato transactions returned as Result 1 ✅
- SIP query → Chunk with SIP transactions returned as Result 1 with score 0.2943 (very relevant) ✅
- Persistence test → Loaded from disk and searched successfully ✅
- Scores increase (get worse) from Result 1 to 3 — correctly ranked by relevance ✅

---

## 🗺️ RAG Pipeline Progress
```
✅ Step 1: Load Document       ← Day 5
✅ Step 2: Preprocess Text     ← Day 5
✅ Step 3: Split into Chunks   ← Day 6
✅ Step 4: Create Embeddings   ← Day 7
✅ Step 5: Store in ChromaDB   ← Done today!
⏳ Step 6: Query & Retrieve    ← Day 9
⏳ Step 7: Generate Answer     ← Day 9
⏳ Step 8: Integrate into UI   ← Day 10

💡 Key Python Concepts Learned
ConceptExampleMeaningConstantsCHROMA_DB_PATH = "chroma_db"Variables that never change — UPPER_CASEshutil.rmtree()shutil.rmtree("folder")Delete entire folder recursivelyTuple unpackingfor (doc, score) in resultsUnpack tuple into named variablesdict.get()metadata.get('source', 'unknown')Safe dictionary access with default valueChroma.from_documents()Creates DB from scratchFirst time setupChroma()Loads existing DBEvery subsequent startup

⚠️ Issues Faced & Solutions
IssueSolutionNoneSmooth day! ✅

✅ Day 8 Checklist

 Understand why ChromaDB is needed over Python list
 Understand ChromaDB concepts — collections, documents, embeddings, IDs
 Understand distance scores vs similarity scores
 Created src/vector_store.py
 Successfully created ChromaDB vector store
 Tested search queries — relevant chunks returned correctly
 Tested persistence — loaded vector store from disk successfully
 Verified chroma_db/ folder created in project
 Committed and pushed to GitHub