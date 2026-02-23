from langchain_text_splitters import RecursiveCharacterTextSplitter
from document_loader import load_document, preprocess_document
import os

# ── Step 1: Create the text splitter ─────────────────────────────────
def create_chunks(documents, chunk_size=300, chunk_overlap=50):
    """
    Split documents into smaller chunks for RAG pipeline.

    Args:
        documents   : List of LangChain Document objects
        chunk_size  : Maximum characters per chunk (default 300)
        chunk_overlap: Characters repeated between chunks (default 50)

    Returns:
        List of chunk Document objects
    """
    print(f"\n✂️  Splitting documents into chunks...")
    print(f"📐 Chunk size: {chunk_size} characters")
    print(f"🔁 Chunk overlap: {chunk_overlap} characters")

    # Create the splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # Split all documents into chunks
    chunks = text_splitter.split_documents(documents)

    print(f"\n✅ Created {len(chunks)} chunks from {len(documents)} document(s)")

    return chunks


# ── Step 2: Inspect chunks ────────────────────────────────────────────
def inspect_chunks(chunks):
    """
    Print all chunks so we can see exactly how document was split.
    """
    print("\n" + "="*50)
    print("🔍 CHUNK INSPECTION")
    print("="*50)

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Characters: {len(chunk.page_content)}")
        print(f"Metadata: {chunk.metadata}")
        print(f"Content:")
        print("-"*30)
        print(chunk.page_content)
        print("-"*30)


# ── Step 3: Chunk statistics ──────────────────────────────────────────
def chunk_statistics(chunks):
    """
    Show statistics about the chunks created.
    Helps us understand if chunk size is appropriate.
    """
    print("\n" + "="*50)
    print("📊 CHUNK STATISTICS")
    print("="*50)

    chunk_lengths = [len(chunk.page_content) for chunk in chunks]

    print(f"Total chunks     : {len(chunks)}")
    print(f"Smallest chunk   : {min(chunk_lengths)} characters")
    print(f"Largest chunk    : {max(chunk_lengths)} characters")
    print(f"Average chunk    : {sum(chunk_lengths) // len(chunk_lengths)} characters")


# ── Main: Run all steps ───────────────────────────────────────────────
if __name__ == "__main__":

    # Load and preprocess document first
    file_path = "data/sample_statement.txt"
    documents = load_document(file_path)
    cleaned_docs = preprocess_document(documents)

    # Create chunks
    chunks = create_chunks(cleaned_docs)

    # Inspect chunks
    inspect_chunks(chunks)

    # Show statistics
    chunk_statistics(chunks)

    print("\n✅ Text chunking complete!")
    print("🔜 Next step: Create embeddings (Day 7)")