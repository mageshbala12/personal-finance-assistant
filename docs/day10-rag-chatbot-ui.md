Day 10 — Integrate RAG into Streamlit UI

📚 Part 1 — What We Are Building Today
Today we take the RAG pipeline from Day 9 and wrap it in a beautiful, usable chat interface. By end of today you'll have a real working Finance Assistant that:
User uploads bank statement PDF/TXT
            ↓
App indexes it into ChromaDB
            ↓
User asks questions in chat
            ↓
RAG pipeline retrieves relevant chunks
            ↓
Gemini answers with YOUR data
            ↓
Sources shown below every answer

📚 Part 2 — UI Design Plan
┌─────────────────────────────────────┐
│  💰 Personal Finance Assistant      │
│  Powered by RAG                     │
├─────────────────────────────────────┤
│  📁 SIDEBAR                         │
│  ─────────────────────────────────  │
│  Upload Document                    │
│  [Browse Files]                     │
│                                     │
│  ✅ Document indexed!               │
│  Chunks: 4                          │
│  Source: statement.txt              │
│                                     │
│  ─────────────────────────────────  │
│  About                              │
│  Stage 1: RAG                       │
├─────────────────────────────────────┤
│  MAIN CHAT AREA                     │
│                                     │
│  🤖 Hello! Upload your bank         │
│     statement to get started.       │
│                                     │
│  👤 How much did I spend on         │
│     Zomato in January?              │
│                                     │
│  🤖 You spent ₹2,050 on Zomato:    │
│     • 03-Jan: ₹850                  │
│     • 12-Jan: ₹650                  │
│     • 20-Jan: ₹550                  │
│     📄 Source: statement.txt        │
│                                     │
│  [Ask a question about              │
│   your finances...]                 │
└─────────────────────────────────────┘

💻 Part 3 — Build the RAG Chatbot UI
Step 1 — Create New File
In src/ folder create rag_chatbot.py and paste:
pythonimport streamlit as st
import sys
import os
import tempfile

# Add src to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import (
    get_llm,
    get_embedding_model,
    index_documents,
    load_vector_store,
    rag_query,
    CHROMA_DB_PATH
)

# ── Page configuration ────────────────────────────────────────────────
st.set_page_config(
    page_title="Personal Finance Assistant",
    page_icon="💰",
    layout="wide"
)

# ── Custom CSS for better appearance ─────────────────────────────────
st.markdown("""
<style>
    .source-box {
        background-color: #f0f2f6;
        border-left: 3px solid #4CAF50;
        padding: 8px 12px;
        margin-top: 8px;
        border-radius: 4px;
        font-size: 0.85em;
        color: #555;
    }
    .chunk-box {
        background-color: #fff8e1;
        border-left: 3px solid #FF9800;
        padding: 8px 12px;
        margin-top: 4px;
        border-radius: 4px;
        font-size: 0.80em;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)


# ── Initialize session state ──────────────────────────────────────────
def init_session_state():
    """
    Initialize all session state variables.
    Called once when app starts.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "llm" not in st.session_state:
        st.session_state.llm = get_llm()

    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = get_embedding_model()

    if "document_indexed" not in st.session_state:
        st.session_state.document_indexed = False

    if "indexed_filename" not in st.session_state:
        st.session_state.indexed_filename = None

    if "show_chunks" not in st.session_state:
        st.session_state.show_chunks = False


# ── Sidebar ───────────────────────────────────────────────────────────
def render_sidebar():
    """
    Render the sidebar with document upload and settings.
    """
    with st.sidebar:
        st.title("📁 Document Manager")
        st.divider()

        # Document upload section
        st.subheader("Upload Financial Document")
        st.caption("Supported: PDF, TXT files")

        uploaded_file = st.file_uploader(
            label="Choose a file",
            type=["pdf", "txt"],
            help="Upload your bank statement or financial document"
        )

        # Process uploaded file
        if uploaded_file is not None:
            if uploaded_file.name != st.session_state.indexed_filename:
                with st.spinner("📥 Indexing document..."):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=os.path.splitext(uploaded_file.name)[1]
                        ) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name

                        # Index the document
                        st.session_state.vector_store = index_documents(
                            tmp_file_path
                        )
                        st.session_state.document_indexed = True
                        st.session_state.indexed_filename = uploaded_file.name

                        # Clean up temp file
                        os.unlink(tmp_file_path)

                        # Add welcome message to chat
                        st.session_state.messages = []
                        st.session_state.messages.append({
                            "role"    : "assistant",
                            "content" : f"✅ Successfully indexed **{uploaded_file.name}**! I'm ready to answer questions about your finances. What would you like to know?",
                            "sources" : [],
                            "chunks"  : []
                        })

                    except Exception as e:
                        st.error(f"❌ Error indexing document: {str(e)}")

        # Show document status
        st.divider()
        st.subheader("📊 Status")

        if st.session_state.document_indexed:
            st.success(f"✅ Document loaded")
            st.info(f"📄 {st.session_state.indexed_filename}")
        else:
            st.warning("⚠️ No document loaded")
            st.caption("Upload a document to start chatting")

        # Settings
        st.divider()
        st.subheader("⚙️ Settings")
        st.session_state.show_chunks = st.toggle(
            "Show retrieved chunks",
            value=False,
            help="Show which parts of document were used to answer"
        )

        # About section
        st.divider()
        st.subheader("ℹ️ About")
        st.caption("""
        **Personal Finance Assistant**
        Stage 1: RAG Pipeline

        Built with:
        - Google Gemini AI
        - LangChain
        - ChromaDB
        - Streamlit
        """)


# ── Main chat area ────────────────────────────────────────────────────
def render_chat():
    """
    Render the main chat interface.
    """
    st.title("💰 Personal Finance Assistant")
    st.caption("Upload your bank statement and ask questions about your finances!")

    # Show welcome message if no document loaded
    if not st.session_state.document_indexed:
        st.info("""
        👋 **Welcome to your Personal Finance Assistant!**

        To get started:
        1. Upload your bank statement (PDF or TXT) in the sidebar
        2. Wait for indexing to complete
        3. Start asking questions about your finances!

        **Example questions you can ask:**
        - How much did I spend on food this month?
        - What are my total SIP investments?
        - What is my closing balance?
        - When was my salary credited?
        """)
        return

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources if available
            if message.get("sources"):
                sources_text = ", ".join([
                    os.path.basename(s)
                    for s in message["sources"]
                ])
                st.markdown(
                    f'<div class="source-box">📄 Source: {sources_text}</div>',
                    unsafe_allow_html=True
                )

            # Show chunks if toggle is on
            if st.session_state.show_chunks and message.get("chunks"):
                with st.expander("🔍 View retrieved chunks"):
                    for i, (doc, score) in enumerate(message["chunks"]):
                        st.markdown(
                            f'<div class="chunk-box">'
                            f'<b>Chunk {i+1}</b> '
                            f'(score: {round(score, 4)})<br>'
                            f'{doc.page_content[:200]}...'
                            f'</div>',
                            unsafe_allow_html=True
                        )

    # Chat input
    if prompt := st.chat_input("Ask a question about your finances..."):

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Save user message
        st.session_state.messages.append({
            "role"    : "user",
            "content" : prompt,
            "sources" : [],
            "chunks"  : []
        })

        # Get RAG response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching your documents..."):
                result = rag_query(
                    prompt,
                    st.session_state.vector_store,
                    st.session_state.llm
                )

            # Display answer
            st.markdown(result["answer"])

            # Display sources
            if result["sources"]:
                sources_text = ", ".join([
                    os.path.basename(s)
                    for s in result["sources"]
                ])
                st.markdown(
                    f'<div class="source-box">📄 Source: {sources_text}</div>',
                    unsafe_allow_html=True
                )

            # Display chunks if toggle is on
            if st.session_state.show_chunks and result["chunks"]:
                with st.expander("🔍 View retrieved chunks"):
                    for i, (doc, score) in enumerate(result["chunks"]):
                        st.markdown(
                            f'<div class="chunk-box">'
                            f'<b>Chunk {i+1}</b> '
                            f'(score: {round(score, 4)})<br>'
                            f'{doc.page_content[:200]}...'
                            f'</div>',
                            unsafe_allow_html=True
                        )

        # Save assistant message
        st.session_state.messages.append({
            "role"    : "assistant",
            "content" : result["answer"],
            "sources" : result["sources"],
            "chunks"  : result["chunks"]
        })


# ── Main app ──────────────────────────────────────────────────────────
def main():
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()

Step 2 — Run It
bashstreamlit run src/rag_chatbot.py
```

Browser opens at `http://localhost:8501`

---

### Step 3 — Test It

1. In the sidebar click **"Browse Files"**
2. Upload `data/sample_statement.txt`
3. Wait for **"✅ Successfully indexed"** message
4. Ask these questions in chat:
```
How much did I spend on Zomato?
What are my SIP investments?
What was my salary this month?
What is my closing balance?

Toggle "Show retrieved chunks" in sidebar to see which parts of document were used!


💾 Commit to GitHub
bashgit add .
git commit -m "feat: integrate RAG pipeline into Streamlit UI"
git push
```

---

## ✅ Day 10 Checklist
- [ ] Understand UI design plan
- [ ] Created `src/rag_chatbot.py`
- [ ] App runs locally in browser
- [ ] Document upload works
- [ ] Indexing completes successfully
- [ ] Chat answers questions correctly
- [ ] Sources shown below answers
- [ ] Chunks toggle works
- [ ] Committed and pushed to GitHub

---

## 🗺️ RAG Pipeline — Complete!
```
✅ Step 1: Load Document       ← Day 5
✅ Step 2: Preprocess Text     ← Day 5
✅ Step 3: Split into Chunks   ← Day 6
✅ Step 4: Create Embeddings   ← Day 7
✅ Step 5: Store in ChromaDB   ← Day 8
✅ Step 6: Query & Retrieve    ← Day 9
✅ Step 7: Generate Answer     ← Day 9
✅ Step 8: Integrate into UI   ← Done today!

🔜 Day 11 Preview
Deploy this RAG version to Render production:

Update Render configuration
Handle ChromaDB in production environment
Test live app with real documents
Your RAG Finance Assistant goes live!

One more day and RAG is in production! 💪
Go ahead and run — let me know what you see or any errors! 🚀I am getting below error when i tried to upload the pdf file.....Error indexing document: Error embedding content (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: __https://ai.google.dev/gemini-api/docs/rate-limits__. To monitor your current usage, head to: __https://ai.dev/rate-limit__. \n* Quota exceeded for metric: generativelanguage.googleapis.com/embed_content_free_tier_requests, limit: 100, model: gemini-embedding-1.0\nPlease retry in 30.9059074s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': '__https://ai.google.dev/gemini-api/docs/rate-limits__'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/embed_content_free_tier_requests', 'quotaId': 'EmbedContentRequestsPerMinutePerUserPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-embedding-1.0'}, 'quotaValue': '100'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '30s'}]}}Show more8:20 PMThis is a rate limit error — not a code error! 🎯
You're on the free tier which allows only 100 embedding requests per minute. Your PDF likely has many chunks that need embedding all at once, exceeding this limit.

Understanding the Problem
Free Tier Limit:
─────────────────
100 embedding requests per minute

Your PDF chunks → each chunk = 1 embedding request
If PDF has 150 chunks → 150 requests → exceeds 100 limit → ❌

Fix — Add Rate Limiting to Vector Store
Open src/vector_store.py and find the create_vector_store() function.
Current code:
pythonvector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=CHROMA_DB_PATH,
    collection_name=COLLECTION_NAME
)
Replace with:
pythonimport time

# Process chunks in batches to avoid rate limit
BATCH_SIZE = 50  # 50 chunks per batch
BATCH_DELAY = 65  # wait 65 seconds between batches

if len(chunks) <= BATCH_SIZE:
    # Small document — process all at once
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME
    )
else:
    # Large document — process in batches
    print(f"📦 Large document detected — processing in batches of {BATCH_SIZE}")

    # First batch creates the store
    first_batch = chunks[:BATCH_SIZE]
    vector_store = Chroma.from_documents(
        documents=first_batch,
        embedding=embedding_model,
        persist_directory=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME
    )
    print(f"✅ Batch 1 done ({len(first_batch)} chunks)")

    # Remaining batches add to existing store
    remaining = chunks[BATCH_SIZE:]
    batch_num = 2

    while remaining:
        print(f"⏳ Waiting {BATCH_DELAY}s before next batch...")
        time.sleep(BATCH_DELAY)

        batch = remaining[:BATCH_SIZE]
        remaining = remaining[BATCH_SIZE:]

        vector_store.add_documents(batch)
        print(f"✅ Batch {batch_num} done ({len(batch)} chunks)")
        batch_num += 1

Also Add Delay in embeddings_manager.py
Open src/embeddings_manager.py and find embed_chunks(). Change:
pythontime.sleep(1)
To:
pythontime.sleep(2)

Quick Alternative — Reduce Chunk Count
If you don't want to wait, use a larger chunk size so fewer chunks are created. Open src/text_chunker.py and update default values:
pythondef create_chunks(documents, chunk_size=800, chunk_overlap=100):
Larger chunks = fewer total chunks = fewer embedding requests = stays under 100 limit.

Which Fix to Choose?
SituationFixSmall TXT file (our sample_statement.txt)Should work fine — try again after 1 minuteLarge PDF (10+ pages)Use batch processing fixWant quickest solutionIncrease chunk size to 800

Try Again
Wait 60 seconds then try uploading again. The rate limit resets every minute.
For our sample_statement.txt it should work fine since it only creates 4 chunks — well under the 100 limit. The issue likely happened because we ran multiple scripts today that used up the quota.
Let me know what happens after waiting 60 seconds! 🚀Provide me the day 10 documentation with detailed code level explanation8:29 PM📘 Day 10 Documentation — RAG Chatbot UI & Rate Limiting
Personal Finance Assistant Project

🎯 Objective
Integrate the complete RAG pipeline into a Streamlit web interface with document upload, chat functionality, source display, and chunk inspection. Handle API rate limiting for production reliability.

📚 Part 1 — Key Concepts
What We Built Today
User uploads bank statement PDF/TXT
            ↓
App indexes it into ChromaDB
            ↓
User asks questions in chat
            ↓
RAG pipeline retrieves relevant chunks
            ↓
Gemini answers with YOUR data
            ↓
Sources shown below every answer
UI Design Plan
┌─────────────────────────────────────┐
│  💰 Personal Finance Assistant      │
├─────────────────────────────────────┤
│  📁 SIDEBAR      │  MAIN CHAT AREA  │
│  ─────────────   │                  │
│  Upload Document │  Chat messages   │
│  [Browse Files]  │  with sources    │
│                  │                  │
│  ✅ Doc loaded   │  [Ask question]  │
│  ⚙️ Settings     │                  │
│  ℹ️ About        │                  │
└─────────────────────────────────────┘
Rate Limiting — Why It Happened
Free Tier Limit: 100 embedding requests per minute

Large PDF → many chunks → each chunk = 1 request
100+ chunks in one minute → RESOURCE_EXHAUSTED error ❌

Fix: Process chunks in batches with delay between batches ✅

💻 Part 2 — Code Created
File: src/rag_chatbot.py
pythonimport streamlit as st
import sys
import os
import tempfile

# Add src to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import (
    get_llm,
    get_embedding_model,
    index_documents,
    load_vector_store,
    rag_query,
    CHROMA_DB_PATH
)

# ── Page configuration ────────────────────────────────────────────────
st.set_page_config(
    page_title="Personal Finance Assistant",
    page_icon="💰",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .source-box {
        background-color: #f0f2f6;
        border-left: 3px solid #4CAF50;
        padding: 8px 12px;
        margin-top: 8px;
        border-radius: 4px;
        font-size: 0.85em;
        color: #555;
    }
    .chunk-box {
        background-color: #fff8e1;
        border-left: 3px solid #FF9800;
        padding: 8px 12px;
        margin-top: 4px;
        border-radius: 4px;
        font-size: 0.80em;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)


# ── Initialize session state ──────────────────────────────────────────
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "llm" not in st.session_state:
        st.session_state.llm = get_llm()
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = get_embedding_model()
    if "document_indexed" not in st.session_state:
        st.session_state.document_indexed = False
    if "indexed_filename" not in st.session_state:
        st.session_state.indexed_filename = None
    if "show_chunks" not in st.session_state:
        st.session_state.show_chunks = False


# ── Sidebar ───────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.title("📁 Document Manager")
        st.divider()

        st.subheader("Upload Financial Document")
        st.caption("Supported: PDF, TXT files")

        uploaded_file = st.file_uploader(
            label="Choose a file",
            type=["pdf", "txt"],
            help="Upload your bank statement or financial document"
        )

        if uploaded_file is not None:
            if uploaded_file.name != st.session_state.indexed_filename:
                with st.spinner("📥 Indexing document..."):
                    try:
                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=os.path.splitext(uploaded_file.name)[1]
                        ) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name

                        st.session_state.vector_store = index_documents(
                            tmp_file_path
                        )
                        st.session_state.document_indexed = True
                        st.session_state.indexed_filename = uploaded_file.name

                        os.unlink(tmp_file_path)

                        st.session_state.messages = []
                        st.session_state.messages.append({
                            "role"    : "assistant",
                            "content" : f"✅ Successfully indexed **{uploaded_file.name}**! I'm ready to answer questions about your finances. What would you like to know?",
                            "sources" : [],
                            "chunks"  : []
                        })

                    except Exception as e:
                        st.error(f"❌ Error indexing document: {str(e)}")

        st.divider()
        st.subheader("📊 Status")

        if st.session_state.document_indexed:
            st.success(f"✅ Document loaded")
            st.info(f"📄 {st.session_state.indexed_filename}")
        else:
            st.warning("⚠️ No document loaded")
            st.caption("Upload a document to start chatting")

        st.divider()
        st.subheader("⚙️ Settings")
        st.session_state.show_chunks = st.toggle(
            "Show retrieved chunks",
            value=False,
            help="Show which parts of document were used to answer"
        )

        st.divider()
        st.subheader("ℹ️ About")
        st.caption("""
        **Personal Finance Assistant**
        Stage 1: RAG Pipeline

        Built with:
        - Google Gemini AI
        - LangChain
        - ChromaDB
        - Streamlit
        """)


# ── Main chat area ────────────────────────────────────────────────────
def render_chat():
    st.title("💰 Personal Finance Assistant")
    st.caption("Upload your bank statement and ask questions about your finances!")

    if not st.session_state.document_indexed:
        st.info("""
        👋 **Welcome to your Personal Finance Assistant!**

        To get started:
        1. Upload your bank statement (PDF or TXT) in the sidebar
        2. Wait for indexing to complete
        3. Start asking questions about your finances!

        **Example questions:**
        - How much did I spend on food this month?
        - What are my total SIP investments?
        - What is my closing balance?
        - When was my salary credited?
        """)
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                sources_text = ", ".join([
                    os.path.basename(s)
                    for s in message["sources"]
                ])
                st.markdown(
                    f'<div class="source-box">📄 Source: {sources_text}</div>',
                    unsafe_allow_html=True
                )

            if st.session_state.show_chunks and message.get("chunks"):
                with st.expander("🔍 View retrieved chunks"):
                    for i, (doc, score) in enumerate(message["chunks"]):
                        st.markdown(
                            f'<div class="chunk-box">'
                            f'<b>Chunk {i+1}</b> '
                            f'(score: {round(score, 4)})<br>'
                            f'{doc.page_content[:200]}...'
                            f'</div>',
                            unsafe_allow_html=True
                        )

    if prompt := st.chat_input("Ask a question about your finances..."):

        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({
            "role"    : "user",
            "content" : prompt,
            "sources" : [],
            "chunks"  : []
        })

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching your documents..."):
                result = rag_query(
                    prompt,
                    st.session_state.vector_store,
                    st.session_state.llm
                )

            st.markdown(result["answer"])

            if result["sources"]:
                sources_text = ", ".join([
                    os.path.basename(s)
                    for s in result["sources"]
                ])
                st.markdown(
                    f'<div class="source-box">📄 Source: {sources_text}</div>',
                    unsafe_allow_html=True
                )

            if st.session_state.show_chunks and result["chunks"]:
                with st.expander("🔍 View retrieved chunks"):
                    for i, (doc, score) in enumerate(result["chunks"]):
                        st.markdown(
                            f'<div class="chunk-box">'
                            f'<b>Chunk {i+1}</b> '
                            f'(score: {round(score, 4)})<br>'
                            f'{doc.page_content[:200]}...'
                            f'</div>',
                            unsafe_allow_html=True
                        )

        st.session_state.messages.append({
            "role"    : "assistant",
            "content" : result["answer"],
            "sources" : result["sources"],
            "chunks"  : result["chunks"]
        })


# ── Main app ──────────────────────────────────────────────────────────
def main():
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
Updated: src/vector_store.py — Batch Processing
Added batch processing to create_vector_store() to handle rate limits:
pythonimport time

BATCH_SIZE  = 50   # chunks per batch
BATCH_DELAY = 65   # seconds between batches

if len(chunks) <= BATCH_SIZE:
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME
    )
else:
    print(f"📦 Large document — processing in batches of {BATCH_SIZE}")

    first_batch  = chunks[:BATCH_SIZE]
    vector_store = Chroma.from_documents(
        documents=first_batch,
        embedding=embedding_model,
        persist_directory=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME
    )
    print(f"✅ Batch 1 done ({len(first_batch)} chunks)")

    remaining = chunks[BATCH_SIZE:]
    batch_num = 2

    while remaining:
        print(f"⏳ Waiting {BATCH_DELAY}s before next batch...")
        time.sleep(BATCH_DELAY)
        batch     = remaining[:BATCH_SIZE]
        remaining = remaining[BATCH_SIZE:]
        vector_store.add_documents(batch)
        print(f"✅ Batch {batch_num} done ({len(batch)} chunks)")
        batch_num += 1

🔍 Part 3 — Detailed Code Explanation
Imports and Path Setup
pythonimport streamlit as st
import sys
import os
import tempfile

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
ImportPurposestreamlitUI frameworksysSystem operations — modify Python pathosFile and folder operationstempfileCreate temporary files for uploaded documents
sys.path.append() — tells Python where to find our modules:
pythonsys.path.append(os.path.dirname(os.path.abspath(__file__)))
__file__ → current file's path (src/rag_chatbot.py)
os.path.abspath() → converts to absolute path
os.path.dirname() → gets the folder (src/)
sys.path.append() → adds src/ to Python's search path
Why needed? When Streamlit runs rag_chatbot.py, Python may not know where rag_pipeline.py, vector_store.py etc. are located. This line explicitly tells Python to look in the src/ folder.

Custom CSS
pythonst.markdown("""
<style>
    .source-box {
        background-color: #f0f2f6;
        border-left: 3px solid #4CAF50;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 0.85em;
    }
    .chunk-box {
        background-color: #fff8e1;
        border-left: 3px solid #FF9800;
        ...
    }
</style>
""", unsafe_allow_html=True)
st.markdown() with unsafe_allow_html=True — injects raw HTML and CSS into the Streamlit page. Normally Streamlit blocks HTML for security. unsafe_allow_html=True explicitly allows it.
CSS classes created:
ClassColorUsed For.source-boxGreen left borderShow answer sources.chunk-boxOrange left borderShow retrieved chunks
border-left: 3px solid #4CAF50 → green vertical line on left side — visual indicator.
border-radius: 4px → slightly rounded corners.
font-size: 0.85em → slightly smaller than normal text. em is relative unit — 0.85em = 85% of parent font size.

init_session_state() Function
pythondef init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "llm" not in st.session_state:
        st.session_state.llm = get_llm()
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = get_embedding_model()
    if "document_indexed" not in st.session_state:
        st.session_state.document_indexed = False
    if "indexed_filename" not in st.session_state:
        st.session_state.indexed_filename = None
    if "show_chunks" not in st.session_state:
        st.session_state.show_chunks = False
Why check if "key" not in st.session_state?
Streamlit reruns the entire file on every interaction. Without this check, every rerun would reset all variables to their initial values — losing your chat history, loaded document, everything.
The check means: "Only initialize if not already set."
Session state variables explained:
VariableTypePurposemessagesListChat history — all messagesvector_storeObjectChromaDB instance — loaded documentllmObjectGemini LLM instanceembedding_modelObjectGoogle embedding model instancedocument_indexedBooleanIs a document loaded? Controls UI stateindexed_filenameStringName of loaded file — prevents re-indexing same fileshow_chunksBooleanToggle to show/hide retrieved chunks
Why store llm and embedding_model in session state?
Creating these objects makes API connections. If we recreated them on every Streamlit rerun it would be slow and wasteful. Storing in session state creates them once and reuses them throughout the session.

render_sidebar() Function
File uploader:
pythonuploaded_file = st.file_uploader(
    label="Choose a file",
    type=["pdf", "txt"],
    help="Upload your bank statement or financial document"
)
st.file_uploader() → Streamlit's built in file upload widget.
type=["pdf", "txt"] → restricts to only PDF and TXT files.
help= → tooltip shown on hover.
Returns a file object or None if no file selected.
Prevent re-indexing same file:
pythonif uploaded_file is not None:
    if uploaded_file.name != st.session_state.indexed_filename:
Two checks:

uploaded_file is not None → a file was actually uploaded
uploaded_file.name != st.session_state.indexed_filename → it's a different file from what's already indexed

Why second check? Streamlit reruns on every interaction. Without this check, every time user clicks anywhere the app would re-index the already loaded document — very slow and wastes API quota.
Temporary file handling:
pythonwith tempfile.NamedTemporaryFile(
    delete=False,
    suffix=os.path.splitext(uploaded_file.name)[1]
) as tmp_file:
    tmp_file.write(uploaded_file.getvalue())
    tmp_file_path = tmp_file.name
Why temporary files? Streamlit's uploaded file is an in-memory object — not a real file on disk. Our load_document() function needs a real file path. So we:

Create a temporary file on disk
Write uploaded content to it
Get its path
Use path for indexing
Delete it when done

tempfile.NamedTemporaryFile() → creates temp file in system's temp folder.
delete=False → don't auto-delete when closed — we need to use it after.
suffix=os.path.splitext(uploaded_file.name)[1] → keeps original extension (.pdf or .txt).
uploaded_file.getvalue() → gets raw bytes of uploaded file.
tmp_file.name → gets the temp file's path on disk.
Cleanup:
pythonos.unlink(tmp_file_path)
os.unlink() → deletes a file. Cleans up temp file after indexing is done. Good practice — don't leave temp files behind.
Reset chat on new document:
pythonst.session_state.messages = []
st.session_state.messages.append({
    "role"    : "assistant",
    "content" : f"✅ Successfully indexed **{uploaded_file.name}**!...",
    "sources" : [],
    "chunks"  : []
})
When a new document is uploaded we clear previous chat history and add a fresh welcome message. Prevents confusion from mixing answers from different documents.
Status indicators:
pythonif st.session_state.document_indexed:
    st.success(f"✅ Document loaded")    # Green box
    st.info(f"📄 {st.session_state.indexed_filename}")  # Blue box
else:
    st.warning("⚠️ No document loaded") # Yellow box
Streamlit's colored message boxes:
FunctionColorUsest.success()GreenPositive statusst.info()BlueInformationalst.warning()YellowCautionst.error()RedError
Chunks toggle:
pythonst.session_state.show_chunks = st.toggle(
    "Show retrieved chunks",
    value=False,
    help="Show which parts of document were used"
)
st.toggle() → on/off switch widget. Returns True or False. Stored in session state so it persists across reruns.

render_chat() Function
Guard clause — no document loaded:
pythonif not st.session_state.document_indexed:
    st.info("""...""")
    return
return → exits function early. If no document is loaded, show instructions and stop. Don't render the chat interface at all. This pattern is called a guard clause — check condition early, exit if not met.
Displaying chat history:
pythonfor message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            sources_text = ", ".join([
                os.path.basename(s)
                for s in message["sources"]
            ])
            st.markdown(
                f'<div class="source-box">📄 Source: {sources_text}</div>',
                unsafe_allow_html=True
            )
message.get("sources") → safely gets sources list. Returns None if key doesn't exist — no crash.
os.path.basename(s) → extracts just filename from full path:
python"data/sample_statement.txt" → "sample_statement.txt"
Cleaner display — users don't need to see full path.
", ".join([...]) → joins list of filenames with comma separator.
Custom HTML for source box:
pythonst.markdown(
    f'<div class="source-box">📄 Source: {sources_text}</div>',
    unsafe_allow_html=True
)
Uses our CSS class .source-box defined at the top. Creates the green-bordered source indicator below each answer.
Chunks expander:
pythonwith st.expander("🔍 View retrieved chunks"):
    for i, (doc, score) in enumerate(message["chunks"]):
        st.markdown(
            f'<div class="chunk-box">'
            f'<b>Chunk {i+1}</b> (score: {round(score, 4)})<br>'
            f'{doc.page_content[:200]}...'
            f'</div>',
            unsafe_allow_html=True
        )
st.expander() → collapsible section. Hidden by default, user clicks to expand. Perfect for optional details like chunks that most users won't need.
doc.page_content[:200] → first 200 characters of chunk. Enough to see what was retrieved without overwhelming the UI.
Processing new user input:
pythonif prompt := st.chat_input("Ask a question about your finances..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({
        "role"    : "user",
        "content" : prompt,
        "sources" : [],
        "chunks"  : []
    })

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching your documents..."):
            result = rag_query(
                prompt,
                st.session_state.vector_store,
                st.session_state.llm
            )
        st.markdown(result["answer"])
st.spinner("🔍 Searching...") → shows loading animation while RAG pipeline runs. Nested inside st.chat_message("assistant") so it appears in the right place in chat.
Message dictionary structure:
python{
    "role"    : "user" or "assistant",
    "content" : "the message text",
    "sources" : ["data/statement.txt"],   # for assistant only
    "chunks"  : [(doc, score), ...]       # for assistant only
}
We store sources and chunks with each assistant message so they can be redisplayed correctly when Streamlit reruns and redraws the chat history.

main() Function
pythondef main():
    init_session_state()
    render_sidebar()
    render_chat()

if __name__ == "__main__":
    main()
Clean entry point — three functions called in order:

Initialize state
Render sidebar
Render main chat area

Why wrap in main()? Keeps code organized. Clear entry point. Easy to add more sections later (e.g., render_header(), render_footer()).

Rate Limiting Fix — Batch Processing
pythonBATCH_SIZE  = 50
BATCH_DELAY = 65

if len(chunks) <= BATCH_SIZE:
    # Small document — process all at once
    vector_store = Chroma.from_documents(...)
else:
    # Large document — process in batches
    first_batch  = chunks[:BATCH_SIZE]
    vector_store = Chroma.from_documents(
        documents=first_batch, ...
    )

    remaining = chunks[BATCH_SIZE:]
    batch_num = 2

    while remaining:
        time.sleep(BATCH_DELAY)
        batch     = remaining[:BATCH_SIZE]
        remaining = remaining[BATCH_SIZE:]
        vector_store.add_documents(batch)
        batch_num += 1
List slicing for batches:
pythonfirst_batch = chunks[:BATCH_SIZE]    # first 50 chunks
remaining   = chunks[BATCH_SIZE:]    # everything after 50
chunks[:50] → indices 0 to 49 (50 items)
chunks[50:] → indices 50 onwards (everything remaining)
while remaining loop:
pythonwhile remaining:          # loop while list is not empty
    batch     = remaining[:BATCH_SIZE]   # take next 50
    remaining = remaining[BATCH_SIZE:]   # remove those 50
    vector_store.add_documents(batch)    # add to existing store
```

`while remaining` → in Python a non-empty list is `True`, empty list is `False`. Loop continues until `remaining` is empty.

`vector_store.add_documents(batch)` → adds new chunks to **existing** ChromaDB store without recreating it.

**Why 65 seconds delay?**
Rate limit is 100 requests per minute. We wait 65 seconds (slightly more than 60) to ensure the minute window fully resets before sending next batch.

---

## 📊 Part 4 — Actual Results
```
App running at: http://localhost:8501

✅ Document upload working
✅ Indexing completes with spinner feedback
✅ Chat answers questions correctly
✅ Sources shown below each answer
✅ Chunks toggle shows retrieved context
✅ Rate limit handled with batch processing
```

**Test questions and results:**

| Question | Result |
|----------|--------|
| How much did I spend on Zomato? | ✅ ₹2,050 with breakdown |
| What are my SIP investments? | ✅ ₹8,000 total |
| What was my salary? | ✅ ₹85,000 on 05-Jan |
| What is my closing balance? | ✅ ₹1,13,551 |

---

## 🗺️ RAG Pipeline — Complete!
```
✅ Step 1: Load Document       ← Day 5
✅ Step 2: Preprocess Text     ← Day 5
✅ Step 3: Split into Chunks   ← Day 6
✅ Step 4: Create Embeddings   ← Day 7
✅ Step 5: Store in ChromaDB   ← Day 8
✅ Step 6: Query & Retrieve    ← Day 9
✅ Step 7: Generate Answer     ← Day 9
✅ Step 8: Integrate into UI   ← Done today!

💡 Key Python Concepts Learned
ConceptExampleMeaningsys.path.append()Add folder to Python search pathFind modules in custom locationsGuard clauseif not condition: returnExit function early if condition not mettempfileNamedTemporaryFile()Create temporary file on diskos.unlink()Delete a fileClean up temp filesos.path.basename()Extract filename from path"data/file.txt" → "file.txt"unsafe_allow_htmlst.markdown(..., unsafe_allow_html=True)Inject custom HTML/CSS into Streamlitst.expander()Collapsible sectionHide optional detailsst.toggle()On/off switch widgetUser controlled boolean settingList slicing for batcheschunks[:50], chunks[50:]Split list into batcheswhile remainingLoop until list emptyProcess all batchesvector_store.add_documents()Add to existing ChromaDBExtend without recreating

⚠️ Issues Faced & Solutions
IssueSolutionRESOURCE_EXHAUSTED 429 rate limit errorAdded batch processing with 65 second delay between batches of 50 chunksModules not found when running StreamlitAdded sys.path.append(os.path.dirname(os.path.abspath(__file__)))Same document re-indexed on every rerunAdded uploaded_file.name != st.session_state.indexed_filename check

✅ Day 10 Checklist

 Understand UI design with sidebar and main chat area
 Understand temporary file handling for uploads
 Understand guard clauses
 Understand rate limiting and batch processing
 Created src/rag_chatbot.py
 App runs locally in browser
 Document upload and indexing works
 Chat answers questions correctly with sources
 Chunks toggle works
 Rate limit handled with batch processing
 Committed and pushed to GitHub