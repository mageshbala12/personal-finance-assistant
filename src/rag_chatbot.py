import streamlit as st
import sys
import os
import tempfile

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import (
    get_llm,
    get_embedding_model,
    index_documents,
    load_vector_store,
    rag_query,
    extract_text_from_response,
    CHROMA_DB_PATH
)
from langchain_core.messages import HumanMessage, SystemMessage

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
    .mode-badge-rag {
        background-color: #e8f5e9;
        border: 1px solid #4CAF50;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75em;
        color: #2e7d32;
    }
    .mode-badge-general {
        background-color: #e3f2fd;
        border: 1px solid #2196F3;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75em;
        color: #1565c0;
    }
</style>
""", unsafe_allow_html=True)


# ── Initialize session state ──────────────────────────────────────────
def init_session_state():
    """
    Initialize all session state variables.
    """
    if "messages" not in st.session_state:
        # Start with welcome message
        st.session_state.messages = [
            {
                "role"    : "assistant",
                "content" : """👋 **Welcome to your Personal Finance Assistant!**

I can help you with:
- 💬 **General finance questions** — EPF, SIP, NPS, tax planning and more
- 📄 **Your personal finances** — Upload your bank statement to ask about your spending, investments and balance

**Supported file formats:**
- 📄 PDF — Bank statements, fund factsheets
- 📊 CSV — Zerodha tradebook, Groww statement
- 📗 Excel — HDFC, ICICI, SBI bank exports
- 📝 TXT — Any text based statement

Feel free to ask any finance question — no document upload required!""",
                "sources" : [],
                "chunks"  : [],
                "mode"    : "general"
            }
        ]

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "llm" not in st.session_state:
        st.session_state.llm = get_llm()

    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = get_embedding_model()

    if "document_indexed" not in st.session_state:
        st.session_state.document_indexed = False

    if "indexed_filenames" not in st.session_state:
        st.session_state.indexed_filenames = []

    if "show_chunks" not in st.session_state:
        st.session_state.show_chunks = False

    # ── NEW: Chat history for memory ──────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


# ── General finance chat with memory ─────────────────────────────────
def general_finance_chat(question, chat_history, llm):
    """
    Answer general finance questions using Gemini
    with full conversation memory.

    Args:
        question    : Current user question
        chat_history: List of previous (question, answer) tuples
        llm         : Gemini LLM instance

    Returns:
        Answer string
    """
    # Build conversation history for context
    history_text = ""
    if chat_history:
        history_text = "\n\nPrevious conversation:\n"
        for prev_q, prev_a in chat_history[-5:]:  # last 5 turns
            history_text += f"User: {prev_q}\nAssistant: {prev_a}\n\n"

    system_message = SystemMessage(content="""
    You are a helpful personal finance assistant for Indian users.
    You have deep knowledge of Indian finance topics including:
    EPF, PPF, NPS, SIP, mutual funds, tax planning, ELSS,
    fixed deposits, insurance, stock market, and budgeting.

    Rules:
    1. Answer clearly and practically for Indian context
    2. Use ₹ for currency formatting
    3. Reference previous conversation context when relevant
    4. If asked to elaborate or explain more — refer to
       your previous answer in this conversation
    5. Be concise but complete
    """)

    human_message = HumanMessage(content=f"""
    {history_text}
    Current question: {question}
    """)

    try:
        response = llm.invoke([system_message, human_message])
        return extract_text_from_response(response)
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            return "⚠️ API rate limit reached. Please wait a few minutes and try again."
        raise e


# ── RAG chat with memory ──────────────────────────────────────────────
def rag_finance_chat(question, chat_history, vector_store, llm):
    """
    Answer questions using RAG pipeline with conversation memory.

    Args:
        question    : Current user question
        chat_history: List of previous (question, answer) tuples
        vector_store: FAISS vector store instance
        llm         : Gemini LLM instance

    Returns:
        Dictionary with answer, sources, chunks
    """
    from vector_store import search_vector_store
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    # Build conversation history
    history_text = ""
    if chat_history:
        history_text = "\n\nPrevious conversation:\n"
        for prev_q, prev_a in chat_history[-5:]:
            history_text += f"User: {prev_q}\nAssistant: {prev_a}\n\n"

    # Search for relevant chunks
    relevant_chunks = search_vector_store(
        vector_store,
        question,
        top_k=3
    )

    # Build context from chunks
    context_parts = []
    for i, (doc, score) in enumerate(relevant_chunks):
        context_parts.append(f"[Context {i+1}]\n{doc.page_content}")
    context_text = "\n\n".join(context_parts)

    system_message = SystemMessage(content="""
    You are a helpful personal finance assistant for Indian users.
    You have two sources of knowledge:
    1. User's personal financial documents (provided as context)
    2. General Indian finance knowledge from training

    Rules:
    1. If question is about USER'S PERSONAL finances
       → Answer from document context
       → If not found say "I couldn't find that in your document"

    2. If question is GENERAL finance
       → Answer from general knowledge
       → Prefix with "Based on general finance knowledge:"

    3. Use previous conversation context for follow-up questions
       If user says "elaborate", "explain more", "tell me more"
       → Refer to your previous answer and expand on it

    4. Show calculation working when doing math

    5. Format currency as ₹ with Indian number format
    """)

    human_message = HumanMessage(content=f"""
    {history_text}
    Relevant sections from financial documents:
    {context_text}

    Current question: {question}
    """)

    try:
        response = llm.invoke([system_message, human_message])
        answer   = extract_text_from_response(response)
        sources  = list(set([
            doc.metadata.get('source', 'unknown')
            for doc, score in relevant_chunks
        ]))
        return {
            "answer"  : answer,
            "sources" : sources,
            "chunks"  : relevant_chunks
        }
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            return {
                "answer"  : "⚠️ API rate limit reached. Please wait a few minutes and try again.",
                "sources" : [],
                "chunks"  : []
            }
        raise e

# ── Sidebar ───────────────────────────────────────────────────────────
def render_sidebar():
    """
    Render sidebar with optional document upload.
    """
    with st.sidebar:
        st.title("📁 Document Manager")
        st.divider()

        # ── Document upload — OPTIONAL ────────────────────────────────
        st.subheader("📄 Upload Document (Optional)")
        st.caption("Supports PDF, TXT, CSV, Excel — Zerodha, Groww, HDFC and more!")

        uploaded_files = st.file_uploader(
            label="Choose files",
            type=["pdf", "txt", "csv", "xlsx", "xls"],
            help="Optional — upload your bank statement or trading report (PDF, TXT, CSV, Excel)",
            accept_multiple_files=True
        )

        if uploaded_files:
            # Get names of newly uploaded files
            uploaded_names = [f.name for f in uploaded_files]

            # Check if files changed since last index
            if uploaded_names != st.session_state.get("indexed_filenames", []):
                with st.spinner(f"📥 Indexing {len(uploaded_files)} file(s)..."):
                    try:
                        all_chunks = []

                        for uploaded_file in uploaded_files:
                            # Save each file temporarily
                            tmp_file = tempfile.NamedTemporaryFile(
                                delete=False,
                                suffix=os.path.splitext(uploaded_file.name)[1]
                            )
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                            tmp_file.close()  # ← Close explicitly before processing!

                            # Load and process each file
                            from document_loader import load_document, preprocess_document
                            from text_chunker import create_chunks

                            try:
                                documents    = load_document(tmp_file_path)
                                cleaned_docs = preprocess_document(documents)
                                chunks       = create_chunks(cleaned_docs)
                                all_chunks.extend(chunks)
                                print(f"✅ Processed: {uploaded_file.name} ({len(chunks)} chunks)")
                            finally:
                                # Always delete temp file even if error occurs
                                try:
                                    os.unlink(tmp_file_path)
                                except:
                                    pass  # Ignore if already deleted
                            print(f"✅ Processed: {uploaded_file.name} ({len(chunks)} chunks)")

                        # Store ALL chunks in one vector store
                        from vector_store import create_vector_store
                        embedding_model = st.session_state.embedding_model
                        st.session_state.vector_store = create_vector_store(
                            all_chunks,
                            embedding_model
                        )

                        st.session_state.document_indexed  = True
                        st.session_state.indexed_filenames = uploaded_names
                        st.session_state.indexed_filename  = ", ".join(uploaded_names)

                        # Add success message to chat
                        file_list = "\n".join([f"  - {n}" for n in uploaded_names])
                        st.session_state.messages.append({
                            "role"    : "assistant",
                            "content" : f"✅ **{len(uploaded_files)} file(s) indexed successfully!**\n\n{file_list}\n\nYou can now ask questions across all your documents!",
                            "sources" : [],
                            "chunks"  : [],
                            "mode"    : "general"
                        })

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        # ── Status ────────────────────────────────────────────────────
        st.divider()
        st.subheader("📊 Status")

        if st.session_state.document_indexed:
            st.success(f"✅ {len(st.session_state.indexed_filenames)} file(s) loaded")
            for fname in st.session_state.indexed_filenames:
                st.info(f"📄 {fname}")
        else:
            st.info("💬 General finance mode")
            st.caption("No document uploaded — answering from general knowledge")

                # ── Settings ──────────────────────────────────────────────────
        st.divider()
        st.subheader("⚙️ Settings")

        st.session_state.show_chunks = st.toggle(
            "Show retrieved chunks",
            value=False,
            help="Show document sections used to answer"
        )

        # ── Clear chat ────────────────────────────────────────────────
        st.divider()
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

        # ── About ─────────────────────────────────────────────────────
        st.divider()
        st.subheader("ℹ️ About")
        st.caption("""
        **Personal Finance Assistant**
        Stage 1: RAG Pipeline

        Built with:
        - Google Gemini AI
        - LangChain + FAISS
        - Streamlit
        """)


# ── Main chat area ────────────────────────────────────────────────────
def render_chat():
    """
    Render main chat interface.
    Works with or without uploaded document.
    """
    st.title("💰 Personal Finance Assistant")
    st.caption("Ask any finance question — upload your bank statement for personal insights!")

    # ── Display chat history ──────────────────────────────────────────
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show mode badge
            if message["role"] == "assistant" and message.get("mode"):
                if message["mode"] == "rag":
                    st.markdown(
                        '<span class="mode-badge-rag">📄 From your document</span>',
                        unsafe_allow_html=True
                    )
                elif message["mode"] == "general":
                    st.markdown(
                        '<span class="mode-badge-general">💡 General knowledge</span>',
                        unsafe_allow_html=True
                    )

            # Show sources
            if message.get("sources"):
                sources_text = ", ".join([
                    os.path.basename(s)
                    for s in message["sources"]
                ])
                st.markdown(
                    f'<div class="source-box">📄 Source: {sources_text}</div>',
                    unsafe_allow_html=True
                )

            # Show chunks if toggle on
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

    # ── Chat input ────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask a finance question..."):

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Save user message
        st.session_state.messages.append({
            "role"    : "user",
            "content" : prompt,
            "sources" : [],
            "chunks"  : [],
            "mode"    : ""
        })

        # ── Decide which mode to use ──────────────────────────────────
        with st.chat_message("assistant"):

            if st.session_state.document_indexed:
                # RAG mode — document uploaded
                with st.spinner("🔍 Searching your documents..."):
                    result = rag_finance_chat(
                        prompt,
                        st.session_state.chat_history,
                        st.session_state.vector_store,
                        st.session_state.llm
                    )

                st.markdown(result["answer"])
                mode = "rag"

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

                # Save to session
                st.session_state.messages.append({
                    "role"    : "assistant",
                    "content" : result["answer"],
                    "sources" : result["sources"],
                    "chunks"  : result["chunks"],
                    "mode"    : mode
                })

                # Update chat history for memory
                st.session_state.chat_history.append(
                    (prompt, result["answer"])
                )

            else:
                # General mode — no document uploaded
                with st.spinner("💭 Thinking..."):
                    answer = general_finance_chat(
                        prompt,
                        st.session_state.chat_history,
                        st.session_state.llm
                    )

                st.markdown(answer)
                st.markdown(
                    '<span class="mode-badge-general">💡 General knowledge</span>',
                    unsafe_allow_html=True
                )

                # Save to session
                st.session_state.messages.append({
                    "role"    : "assistant",
                    "content" : answer,
                    "sources" : [],
                    "chunks"  : [],
                    "mode"    : "general"
                })

                # Update chat history for memory
                st.session_state.chat_history.append(
                    (prompt, answer)
                )


# ── Main app ──────────────────────────────────────────────────────────
def main():
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()