📘 Day 11 Documentation — Production Deployment & Troubleshooting
Personal Finance Assistant Project

🎯 Objective
Deploy the RAG chatbot to production on Render, solve deployment challenges including ChromaDB persistence, Pydantic conflicts, and API rate limits.

📚 Part 1 — Deployment Challenges Overview
What's Different from Day 2 Deployment
AspectDay 2 (Simple Chatbot)Day 11 (RAG Chatbot)Packages3 packages10+ packagesDatabaseNoneVector database neededFile uploadsNonePDF/TXT handlingMemoryMinimalMore for embeddingsComplexityLowHigh

Challenges Faced and Solutions
ChallengeSolutionRender persistent disk not freeAccepted re-indexing on restart (ChromaDB in memory)rag_chatbot.py not foundChanged Render branch from main to feature/ragChromaDB Pydantic conflictReplaced ChromaDB with FAISS vector databaseChatbot only answered from documentsAdded hybrid response logic for general + personal questionsAPI rate limit (429 error)Added friendly error handler, plan to switch model tomorrow

🛠️ Part 2 — All Changes Made Today
Change 1 — Updated requirements.txt
google-generativeai
streamlit
python-dotenv
langchain
langchain-community
langchain-google-genai
langchain-text-splitters
langchain-core
faiss-cpu
pypdf
Key change: Removed chromadb completely. Added faiss-cpu as replacement.

Change 2 — Updated .streamlit/config.toml
toml[server]
headless = true
port = 10000
maxUploadSize = 50

[browser]
gatherUsageStats = false
maxUploadSize = 50 → allows files up to 50MB.

Change 3 — Replaced ChromaDB with FAISS
Why FAISS over ChromaDB?
ChromaDBFAISSPydantic compatibility❌ Conflicts on Render✅ No conflictsSpeedGood✅ Faster for small datasetsDisk persistenceNeeds paid diskIn-memory works fineProduction useCommon✅ Used by Facebook/Meta
New File: src/vector_store.py (Complete Replacement)
pythonfrom langchain_google_genai import GoogleGenerativeAIEmbeddings
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
    Processes in batches to avoid API rate limits.
    """
    print("\n🗄️  Creating FAISS vector store...")
    print(f"📦 Total chunks to store: {len(chunks)}")

    BATCH_SIZE  = 50
    BATCH_DELAY = 65

    if len(chunks) <= BATCH_SIZE:
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embedding_model
        )
        print(f"✅ All {len(chunks)} chunks stored!")

    else:
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

Change 4 — Hybrid Response in rag_pipeline.py
Problem: RAG was only answering from documents. General finance questions like "What is EPF?" returned "I couldn't find that in your documents."
Solution: Updated system prompt to handle both personal and general finance questions.
Updated build_rag_prompt() system message:
pythonsystem_message = SystemMessage(content="""
    You are a helpful personal finance assistant for Indian users.
    You have two sources of knowledge:
    1. The user's personal financial documents (provided as context)
    2. Your general finance knowledge from training

    Rules:
    1. If the question is about the USER'S PERSONAL finances
       (their spending, transactions, balance, investments)
       → Answer ONLY from the provided context
       → If not found in context say
         "I couldn't find that in your uploaded document"

    2. If the question is a GENERAL finance question
       (what is EPF, how does SIP work, what is XIRR etc.)
       → Answer from your general knowledge
       → Clearly say "Based on general finance knowledge:"

    3. If answer requires calculation use numbers from context
       and show your working clearly

    4. Always mention specific amounts and dates from context
       when answering personal finance questions

    5. Format currency as ₹ with Indian number format

    6. Be concise, practical and helpful for Indian users
    """)

Change 5 — Rate Limit Error Handler in rag_pipeline.py
Added friendly error message instead of app crash when API limit is hit:
pythonprint("🤔 Generating answer...")
try:
    response = llm.invoke(messages)
except Exception as e:
    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
        return {
            "question" : question,
            "answer"   : "⚠️ API rate limit reached. Please wait a few minutes and try again. This happens on the free tier when too many requests are made.",
            "sources"  : [],
            "chunks"   : []
        }
    raise e
```

---

### Change 6 — Updated `.gitignore`
```
faiss_db/
chroma_db/
```

---

### Change 7 — Render Configuration

| Setting | Old Value | New Value |
|---------|-----------|-----------|
| Branch | `main` | `feature/rag` |
| Start Command | `streamlit run src/chatbot.py` | `streamlit run src/rag_chatbot.py` |

---

## 🔍 Part 3 — Detailed Code Explanation

### FAISS vs ChromaDB

**Why FAISS works where ChromaDB failed:**
```
ChromaDB on Render:
───────────────────
ChromaDB uses Pydantic v1 internally
Render's environment has Pydantic v2
Conflict → ConfigError on startup ❌

FAISS on Render:
─────────────────
FAISS is a pure C++ library with Python bindings
No Pydantic dependency at all
Zero conflicts → Works perfectly ✅

FAISS Key Functions Explained
FAISS.from_documents():
pythonvector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embedding_model
)
Creates FAISS index from scratch. Automatically:

Generates embedding for each chunk
Builds FAISS index in memory
Ready for similarity search immediately

FAISS.load_local():
pythonvector_store = FAISS.load_local(
    folder_path=FAISS_DB_PATH,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)
allow_dangerous_deserialization=True → FAISS uses Python's pickle to save/load indexes. LangChain warns about this because pickle files could theoretically contain malicious code. Since we created the file ourselves it's safe to allow.
similarity_search_with_score():
pythonresults = vector_store.similarity_search_with_score(
    query=query,
    k=top_k
)
```
Works identically to ChromaDB's version. Returns list of `(Document, score)` tuples. **Note:** FAISS scores are L2 distances — lower is better (opposite of cosine similarity where higher is better).

---

### Hybrid Response Logic Explained
```
Question comes in
      ↓
Is it about USER'S personal data?
(spending, balance, transactions)
      ↓
YES → Search ChromaDB/FAISS
      → Answer from document context only
      → "I couldn't find that" if not in docs

NO → General finance question?
(EPF, SIP mechanics, tax rules)
      ↓
YES → Answer from Gemini training knowledge
      → Prefix with "Based on general finance knowledge:"
```

**Why this matters:**
```
Without hybrid logic:
"What is EPF?" → "I couldn't find that in your documents" ❌

With hybrid logic:
"What is EPF?" → "Based on general finance knowledge:
                  EPF (Employee Provident Fund) is..." ✅

"How much did I spend on Zomato?" → "You spent ₹2,050..." ✅

Rate Limit Handler Explained
pythontry:
    response = llm.invoke(messages)
except Exception as e:
    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
        return {
            "answer": "⚠️ API rate limit reached..."
        }
    raise e
```

**`try/except` pattern:**
```
try block    → attempt the risky operation
except block → handle specific errors gracefully
raise e      → re-raise any OTHER errors we didn't expect
```

**`"429" in str(e)`** → converts exception to string and checks if it contains "429" (HTTP rate limit status code).

**`raise e`** → if error is NOT a rate limit error we don't want to silently swallow it. Re-raising lets it propagate so we can see and fix unexpected errors.

---

## 📊 Part 4 — Deployment Journey Summary
```
Attempt 1: Deploy with chromadb
→ ❌ Pydantic v1/v2 conflict

Attempt 2: Pin chromadb==0.4.24
→ ❌ Still pydantic conflict

Attempt 3: chromadb in-memory mode
→ ❌ Still pydantic conflict (deeper issue)

Attempt 4: Replace with FAISS
→ ✅ Works perfectly!

Additional fixes:
→ ✅ Hybrid response for general questions
→ ✅ Rate limit error handler added
→ ⏳ Model switch pending (tomorrow)
```

---

## 💡 Key Concepts Learned Today

| Concept | Explanation |
|---------|------------|
| Dependency conflicts | Different packages needing different versions of same library |
| Pydantic v1 vs v2 | Major breaking changes between versions caused ChromaDB to fail |
| FAISS | Facebook's vector similarity search library — faster and conflict-free |
| L2 distance | FAISS scoring — lower score = more similar (opposite of cosine) |
| Hybrid RAG | Combining document retrieval with general LLM knowledge |
| Rate limits | Free tier API quotas — per minute AND per day limits |
| `try/except/raise` | Handle known errors gracefully, re-raise unexpected ones |
| Branch deployment | Render can deploy from any GitHub branch |

---

## ⚠️ Issues Faced & Solutions

| Issue | Root Cause | Solution |
|-------|-----------|---------|
| `File does not exist: src/rag_chatbot.py` | Render deploying `main` branch, RAG code on `feature/rag` | Changed Render branch to `feature/rag` |
| `pydantic ConfigError chroma_server_nofile` | ChromaDB uses Pydantic v1, Render has v2 | Replaced ChromaDB with FAISS |
| `pydantic ConfigError chroma_db_impl` | Same Pydantic conflict, different attribute | Confirmed FAISS replacement needed |
| Chatbot only answers from documents | RAG system prompt too restrictive | Added hybrid response logic |
| `429 RESOURCE_EXHAUSTED` daily limit | `gemini-2.5-flash-lite` only 20 requests/day free | Added error handler, will switch model tomorrow |

---

## 🔜 Pending for Tomorrow
```
1. Switch LLM model to higher quota model
   (gemini-3-flash-preview or equivalent)

2. Test full production app after model switch

3. Merge feature/rag into main branch

4. Stage 1 complete review

5. Begin planning Stage 2 — AI Agents

✅ Day 11 Checklist

 Updated requirements.txt with FAISS
 Updated .streamlit/config.toml
 Replaced ChromaDB with FAISS in vector_store.py
 Updated .gitignore with faiss_db/
 Fixed Render branch to feature/rag
 Updated Render start command
 Added hybrid response logic
 Added rate limit error handler
 App successfully deployed on Render
 Document upload working in production
 RAG answers working in production
 General finance questions working
 Model switch pending (tomorrow)
 Merge to main pending (tomorrow)


Production Note

ChromaDB was replaced with FAISS which runs in memory on Render free tier. Data resets on app restart — users need to re-upload documents. For permanent storage, Pinecone cloud vector DB will be integrated in a future upgrade when moving to paid tier.

