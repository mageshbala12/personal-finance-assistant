from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from vector_store import (
    get_embedding_model,
    create_vector_store,
    load_vector_store,
    search_vector_store
)
from document_loader import load_document, preprocess_document
from text_chunker import create_chunks
from dotenv import load_dotenv
import os

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────
CHROMA_DB_PATH = "chroma_db"
TOP_K_CHUNKS   = 3

# ── Step 1: Initialize LLM ────────────────────────────────────────────
def get_llm():
    """
    Initialize and return Gemini LLM.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.2
    )


# ── Step 2: Index documents ───────────────────────────────────────────
def index_documents(file_path):
    """
    Complete indexing pipeline:
    Load → Preprocess → Chunk → Store in ChromaDB

    Run this once whenever documents change.

    Args:
        file_path: Path to document file

    Returns:
        ChromaDB vector store instance
    """
    print("\n📥 Starting document indexing...")

    # Load and preprocess
    documents    = load_document(file_path)
    cleaned_docs = preprocess_document(documents)

    # Chunk
    chunks = create_chunks(cleaned_docs)

    # Store in ChromaDB
    embedding_model = get_embedding_model()
    vector_store    = create_vector_store(chunks, embedding_model)

    print("✅ Document indexing complete!")
    return vector_store


# ── Step 3: Build RAG prompt ──────────────────────────────────────────
def build_rag_prompt(question, relevant_chunks):
    """
    Build a prompt that includes:
    - System instruction for finance assistant
    - Retrieved context from documents
    - User question

    This is called Prompt Engineering — carefully
    crafting the prompt to get best answers from LLM.

    Args:
        question       : User's question
        relevant_chunks: List of (Document, score) tuples from ChromaDB

    Returns:
        List of LangChain message objects
    """

    # Extract text from chunks
    context_parts = []
    for i, (doc, score) in enumerate(relevant_chunks):
        context_parts.append(f"[Context {i+1}]\n{doc.page_content}")

    context_text = "\n\n".join(context_parts)

    # System message — tells Gemini its role and rules
    system_message = SystemMessage(content="""
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
    # Human message — context + question
    human_message = HumanMessage(content=f"""
    Here are relevant sections from your financial documents:

    {context_text}

    Based on the above information, please answer this question:
    {question}
    """)

    return [system_message, human_message]


# ── Step 4: RAG query ─────────────────────────────────────────────────
def rag_query(question, vector_store, llm):
    """
    Complete RAG query pipeline:
    Search ChromaDB → Build prompt → Get Gemini answer

    Args:
        question    : User's question
        vector_store: ChromaDB vector store instance
        llm         : Gemini LLM instance

    Returns:
        Dictionary with answer and source chunks
    """
    print(f"\n❓ Question: {question}")
    print("-"*50)

    # Step 1: Retrieve relevant chunks
    relevant_chunks = search_vector_store(
        vector_store,
        question,
        top_k=TOP_K_CHUNKS
    )

    # Step 2: Build RAG prompt
    messages = build_rag_prompt(question, relevant_chunks)

    # Step 3: Get answer from Gemini
    print("🤔 Generating answer...")
    response = llm.invoke(messages)

    # Step 4: Extract sources
    sources = list(set([
        doc.metadata.get('source', 'unknown')
        for doc, score in relevant_chunks
    ]))

    return {
        "question" : question,
        "answer"   : response.content,
        "sources"  : sources,
        "chunks"   : relevant_chunks
    }


# ── Step 5: Display result ────────────────────────────────────────────
def display_result(result):
    """
    Display RAG query result in clean readable format.
    """
    print("\n" + "="*50)
    print("💬 RAG RESPONSE")
    print("="*50)
    print(f"\n❓ Question: {result['question']}")
    print(f"\n💡 Answer:\n{result['answer']}")
    print(f"\n📄 Sources: {', '.join(result['sources'])}")
    print("="*50)


# ── Main: Full RAG pipeline test ──────────────────────────────────────
if __name__ == "__main__":

    # Initialize components
    embedding_model = get_embedding_model()
    llm             = get_llm()

    # Index documents (comment this out after first run)
    file_path   = "data/sample_statement.txt"
    vector_store = index_documents(file_path)

    # OR load existing vector store (uncomment after first run)
    # vector_store = load_vector_store(embedding_model)

    print("\n" + "="*50)
    print("🧪 TESTING RAG PIPELINE")
    print("="*50)

    # Test questions
    test_questions = [
        "How much did I spend on Zomato in January?",
        "What are my SIP investments and total amount?",
        "What was my salary this month?",
        "How much did I spend on groceries?",
        "What is my closing balance?",
        "What percentage of my salary did I save?"
    ]

    for question in test_questions:
        result = rag_query(question, vector_store, llm)
        display_result(result)
        print()

    print("✅ RAG Pipeline test complete!")
    print("🔜 Next step: Integrate into Streamlit UI (Day 10)")