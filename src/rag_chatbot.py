import streamlit as st
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