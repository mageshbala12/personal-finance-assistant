from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os
import re

load_dotenv()

# ── Step 1: Load the document ─────────────────────────────────────────
def load_document(file_path):
    """
    Load a document from a file path.
    Supports .txt and .pdf files.
    Returns a list of Document objects.
    """
    print(f"\n📂 Loading document: {file_path}")

    # Check file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load based on file type
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    documents = loader.load()

    print(f"✅ Loaded {len(documents)} document(s)")
    print(f"📄 Total characters: {sum(len(doc.page_content) for doc in documents)}")

    return documents


# ── Step 2: Inspect the document ─────────────────────────────────────
def inspect_document(documents):
    """
    Print document details so we can see what we loaded.
    Very useful for debugging and understanding raw data.
    """
    print("\n" + "="*50)
    print("📋 DOCUMENT INSPECTION")
    print("="*50)

    for i, doc in enumerate(documents):
        print(f"\n--- Document {i+1} ---")
        print(f"Metadata: {doc.metadata}")
        print(f"Content Length: {len(doc.page_content)} characters")
        print(f"\nFirst 500 characters of content:")
        print("-"*30)
        print(doc.page_content[:500])
        print("-"*30)


# ── Step 3: Preprocess the document ──────────────────────────────────
def preprocess_document(documents):
    """
    Clean and preprocess document text.
    Removes extra whitespace, empty lines, and normalizes formatting.
    """
    print("\n🔧 Preprocessing documents...")

    cleaned_documents = []

    for doc in documents:
        content = doc.page_content

        # Remove extra whitespace between words
        content = re.sub(r' +', ' ', content)

        # Remove more than 2 consecutive newlines
        content = re.sub(r'\n{3,}', '\n\n', content)

        # Strip leading and trailing whitespace from each line
        lines = [line.strip() for line in content.split('\n')]

        # Remove completely empty lines at start and end
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()

        # Join lines back together
        content = '\n'.join(lines)

        # Update document content
        doc.page_content = content
        cleaned_documents.append(doc)

    print(f"✅ Preprocessed {len(cleaned_documents)} document(s)")
    return cleaned_documents


# ── Main: Run all steps ───────────────────────────────────────────────
if __name__ == "__main__":

    # Path to our sample bank statement
    file_path = "data/sample_statement.txt"

    # Step 1: Load
    documents = load_document(file_path)

    # Step 2: Inspect raw document
    print("\n📌 RAW DOCUMENT (before preprocessing):")
    inspect_document(documents)

    # Step 3: Preprocess
    cleaned_docs = preprocess_document(documents)

    # Step 4: Inspect cleaned document
    print("\n📌 CLEANED DOCUMENT (after preprocessing):")
    inspect_document(cleaned_docs)

    print("\n✅ Document loading and preprocessing complete!")
    print("🔜 Next step: Split into chunks (Day 6)")