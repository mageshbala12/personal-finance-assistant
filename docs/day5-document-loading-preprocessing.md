Day 5 Documentation — Document Loading & Text Preprocessing
Personal Finance Assistant Project

🎯 Objective
Load financial documents into LangChain, inspect raw content, and preprocess text to prepare it for chunking in the next step.

📚 Part 1 — Key Concepts
What is a Document Loader?
LangChain tool that reads files and converts them into Document objects.
Raw File (PDF/Text)          LangChain Document Object
────────────────────         ──────────────────────────
Just bytes on disk    →      page_content: "text here..."
Not usable by AI             metadata: {source: "file.txt"}
LangChain Document Loader:

Reads the file
Extracts the text
Wraps it in a Document object with content + metadata
Hands it to the next step in the pipeline

What is Metadata?
Extra information about the document like filename and page number.
python# Text file metadata
{'source': 'data/sample_statement.txt'}

# PDF file metadata
{'source': 'data/report.pdf', 'page': 0}
Used later to tell users where answers came from.
What is Preprocessing?
Cleaning raw text before AI processing:

Remove extra whitespace
Remove excessive blank lines
Normalize text formatting
Strip leading/trailing spaces from each line


💻 Part 2 — Code Created
File: src/document_loader.py
pythonfrom langchain_community.document_loaders import TextLoader
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

🔍 Part 3 — Detailed Code Explanation
Imports
ImportPurposeTextLoaderReads .txt files and wraps content in Document objectPyPDFLoaderReads .pdf files, handles binary PDF format complexityload_dotenvLoads API key from .env file into memoryosCheck file existence, split file extensionreRegular expressions for pattern-based text cleaning

Function 1 — load_document()
File existence check:
pythonif not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")
Verifies file exists before loading. Gives clear error message instead of cryptic Python crash.
File extension detection:
pythonfile_extension = os.path.splitext(file_path)[1].lower()
os.path.splitext("data/sample.txt") returns ("data/sample", ".txt"). [1] gets the extension. .lower() handles both .TXT and .txt.
Loader selection:
pythonif file_extension == ".txt":
    loader = TextLoader(file_path, encoding="utf-8")
elif file_extension == ".pdf":
    loader = PyPDFLoader(file_path)
Different file types need different loaders. encoding="utf-8" supports English, Hindi, Tamil and most languages. Without this, Windows may throw encoding errors.
Document loading:
pythondocuments = loader.load()
Returns a list of Document objects. Each Document has:

doc.page_content → actual text content
doc.metadata → information about the document


Function 2 — inspect_document()
Purpose: Purely a debugging and learning tool — not part of final RAG pipeline. Used to verify content loaded correctly before processing.
enumerate():
pythonfor i, doc in enumerate(documents):
Gives both index number (i) and document (doc) while looping. Without enumerate you'd only get the document, not the position number.
String slicing:
pythonprint(doc.page_content[:500])
Gets first 500 characters. [:500] means "from beginning up to character 500". Keeps output manageable.

Function 3 — preprocess_document()
Remove extra spaces:
pythoncontent = re.sub(r' +', ' ', content)
```
`re.sub(pattern, replacement, text)` → finds pattern and replaces it.
`r' +'` means "one or more spaces" → replaced with single space.
```
"Zomato    Food    Order"  →  "Zomato Food Order"
Remove excessive blank lines:
pythoncontent = re.sub(r'\n{3,}', '\n\n', content)
\n = newline character. {3,} = 3 or more times.
3+ consecutive newlines → replaced with 2 newlines.
Strip each line:
pythonlines = [line.strip() for line in content.split('\n')]
content.split('\n') → splits text into list of lines.
line.strip() → removes leading/trailing whitespace from each line.
[... for line in ...] → list comprehension — compact loop.
Equivalent to:
pythonlines = []
for line in content.split('\n'):
    lines.append(line.strip())
Remove empty lines at start/end:
pythonwhile lines and not lines[0]:
    lines.pop(0)
while lines and not lines[-1]:
    lines.pop()
lines[-1] → last element (-1 counts from end).
lines.pop(0) → removes first element.
lines.pop() → removes last element.
Keeps looping until no empty lines remain at edges.
Rejoin lines:
pythoncontent = '\n'.join(lines)
Opposite of split() — joins list back into single string with newlines between each line.

Main Block — if __name__ == "__main__"
pythonif __name__ == "__main__":
```

**Most important Python pattern to understand:**

| Scenario | `__name__` value |
|----------|-----------------|
| File run directly (`python src/document_loader.py`) | `"__main__"` |
| File imported by another file | Module name |

This means test code only runs when file is executed directly — NOT when imported. Critical because in Day 9 we'll import `load_document()` from this file into the RAG pipeline without triggering the test code.

---

## 🗺️ RAG Pipeline Progress
```
✅ Step 1: Load Document       ← Done today
✅ Step 2: Preprocess Text     ← Done today
⏳ Step 3: Split into Chunks   ← Day 6
⏳ Step 4: Create Embeddings   ← Day 7
⏳ Step 5: Store in ChromaDB   ← Day 8
⏳ Step 6: Query & Retrieve    ← Day 9
⏳ Step 7: Generate Answer     ← Day 9
⏳ Step 8: Integrate into UI   ← Day 10

💡 Key Python Concepts Learned
ConceptExampleMeaningf-stringf"Found {len(docs)} docs"Embed variable in stringList comprehension[x.strip() for x in lines]Compact loop to transform listString slicingtext[:500]Get first 500 charactersenumerate()for i, doc in enumerate(docs)Loop with index and valuere.sub()re.sub(r' +', ' ', text)Find and replace using patternraiseraise FileNotFoundError(...)Trigger error with custom message__name__if __name__ == "__main__"Run only when executed directly

⚠️ Issues Faced & Solutions
IssueSolutionNoneSmooth day! ✅

✅ Day 5 Checklist

 Understand what Document Loader does
 Understand Document object — page_content and metadata
 Understand what preprocessing does and why
 Created src/document_loader.py
 Successfully loaded sample_statement.txt
 Inspected raw document output
 Preprocessed document and inspected cleaned output
 Understood every function and Python concept in the code
 Committed and pushed to GitHub