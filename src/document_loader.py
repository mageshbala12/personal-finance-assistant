from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os
import re
from langchain_core.documents import Document
import csv
import pandas as pd

load_dotenv()


# ── Step 1: Load the document ─────────────────────────────────────────

def load_csv(file_path):
    """
    Load CSV file and convert to LangChain Documents.
    Each row becomes meaningful text for embedding.

    Handles Zerodha, Groww, HDFC and other
    common Indian broker/bank CSV formats.
    """
    print(f"📊 Loading CSV file...")

    documents = []

    try:
        # Read CSV with pandas for better handling
        df = pd.read_csv(file_path)

        print(f"📋 Columns found: {list(df.columns)}")
        print(f"📋 Total rows: {len(df)}")

        # Convert each row to a readable text format
        rows_text = []
        for index, row in df.iterrows():
            # Convert row to "Column: Value" format
            row_text = " | ".join([
                f"{col}: {val}"
                for col, val in row.items()
                if pd.notna(val) and str(val).strip() != ""
            ])
            if row_text:
                rows_text.append(row_text)

        # Join all rows into one document
        full_text = f"File: {os.path.basename(file_path)}\n"
        full_text += f"Columns: {', '.join(df.columns)}\n\n"
        full_text += "\n".join(rows_text)

        documents.append(Document(
            page_content=full_text,
            metadata={"source": file_path, "type": "csv"}
        ))

    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")

    return documents


def load_excel(file_path):
    """
    Load Excel file and convert to LangChain Documents.
    Handles .xlsx and .xls formats.
    Supports multiple sheets.
    """
    print(f"📊 Loading Excel file...")

    documents = []

    try:
        # Read all sheets
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names

        print(f"📋 Sheets found: {sheet_names}")

        for sheet_name in sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            if df.empty:
                continue

            print(f"📋 Sheet '{sheet_name}': {len(df)} rows")

            # Convert each row to readable text
            rows_text = []
            for index, row in df.iterrows():
                row_text = " | ".join([
                    f"{col}: {val}"
                    for col, val in row.items()
                    if pd.notna(val) and str(val).strip() != ""
                ])
                if row_text:
                    rows_text.append(row_text)

            # Create document for each sheet
            full_text = f"File: {os.path.basename(file_path)}\n"
            full_text += f"Sheet: {sheet_name}\n"
            full_text += f"Columns: {', '.join(str(c) for c in df.columns)}\n\n"
            full_text += "\n".join(rows_text)

            documents.append(Document(
                page_content=full_text,
                metadata={
                    "source" : file_path,
                    "sheet"  : sheet_name,
                    "type"   : "excel"
                }
            ))

    except Exception as e:
        raise ValueError(f"Error reading Excel file: {str(e)}")

    return documents

def load_document(file_path):
    """
    Load a document from a file path.
    Supports .txt, .pdf, .csv and .xlsx files.
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
        documents = loader.load()

    elif file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
        documents = loader.load()

    elif file_extension == ".csv":
        documents = load_csv(file_path)

    elif file_extension in [".xlsx", ".xls"]:
        documents = load_excel(file_path)

    else:
        raise ValueError(
            f"Unsupported file type: {file_extension}. "
            f"Supported: .txt, .pdf, .csv, .xlsx, .xls"
        )

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