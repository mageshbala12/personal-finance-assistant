📘 Day 6 Documentation — Text Chunking Strategies
Personal Finance Assistant Project

🎯 Objective
Split preprocessed documents into smaller chunks that can be efficiently stored in a vector database and retrieved during RAG queries.

📚 Part 1 — Key Concepts
Why Chunking is Needed
ProblemExplanationToken LimitEvery LLM has maximum input size. Large documents exceed this limitCostSending entire document every query = expensive. Only relevant chunks = cheapAccuracyToo much irrelevant text confuses AI. Relevant chunks = accurate answers
Chunking Flow
Large Document (743 characters)
         ↓
Split into smaller chunks
         ↓
Chunk 1: "HDFC BANK header info..."
Chunk 2: "01-Jan to 10-Jan transactions..."
Chunk 3: "11-Jan to 20-Jan transactions..."
Chunk 4: "21-Jan to 31-Jan transactions..."
         ↓
Store each chunk separately in ChromaDB
         ↓
Only retrieve relevant chunks per query

Chunking Strategies
Strategy 1 — Fixed Size Chunking:
Splits text into equal sized chunks regardless of content. Problem: cuts mid-word or mid-sentence losing context.
"HDFC BANK Jan 2025. Zomato 85"  ← cuts mid-word!
Strategy 2 — Recursive Character Text Splitting ✅ (What We Use):
Tries to split on natural boundaries in priority order:
1. Paragraphs (\n\n)   → best boundary
2. Lines (\n)          → good boundary
3. Sentences (". ")    → acceptable boundary
4. Words (" ")         → last resort
5. Characters ("")     → absolute last resort
Strategy 3 — Semantic Chunking:
Groups sentences by meaning. More advanced — used in later stages.

Two Critical Parameters
Chunk Size — how many characters per chunk:
Too small (50 chars)  : "01-Jan Zomato" → loses context
Too large (2000 chars): Entire document → defeats purpose
Just right (300 chars): Few transactions → enough context
Chunk Overlap — characters repeated between chunks:
Without overlap:
Chunk 1: "Transaction A. Transaction B."
Chunk 2: "Transaction C. Transaction D."
Problem: Answer spanning boundary is missed!

With overlap (50 chars):
Chunk 1: "Transaction A. Transaction B."
Chunk 2: "Transaction B. Transaction C. Transaction D."
          ↑ repeated     ↑ new content
Benefit: No information lost at boundaries!

💻 Part 2 — Code Created
File: src/text_chunker.py
pythonfrom langchain_text_splitters import RecursiveCharacterTextSplitter
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

🔍 Part 3 — Detailed Code Explanation
Imports
pythonfrom langchain_text_splitters import RecursiveCharacterTextSplitter
from document_loader import load_document, preprocess_document
import os
ImportPurposeRecursiveCharacterTextSplitterLangChain's smart text splitter that respects natural boundariesload_document, preprocess_documentImporting our own functions from Day 5's document_loader.pyosOperating system utilities
Key concept — importing our own functions:
pythonfrom document_loader import load_document, preprocess_document
This is how Python reuses code across files. We wrote load_document() in document_loader.py — instead of rewriting it, we import it here. This is why if __name__ == "__main__" was important in Day 5 — it prevents test code from running when we import.

Function 1 — create_chunks()
Function signature with default parameters:
pythondef create_chunks(documents, chunk_size=300, chunk_overlap=50):
chunk_size=300 and chunk_overlap=50 are default parameter values. If caller doesn't provide them, these defaults are used. Caller can override:
pythoncreate_chunks(docs)                    # uses 300, 50
create_chunks(docs, chunk_size=500)    # uses 500, 50
create_chunks(docs, 500, 100)          # uses 500, 100
Args and Returns in docstring:
python"""
Args:
    documents   : List of LangChain Document objects
    chunk_size  : Maximum characters per chunk (default 300)
    chunk_overlap: Characters repeated between chunks (default 50)
Returns:
    List of chunk Document objects
"""
Professional way to document what a function expects and returns. Makes code self-documenting.
Creating the splitter:
pythontext_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `chunk_size` | 300 | Maximum 300 characters per chunk |
| `chunk_overlap` | 50 | 50 characters repeated between chunks |
| `length_function` | `len` | Measure chunk size by character count |
| `separators` | `["\n\n", "\n", ". ", " ", ""]` | Priority order for splitting boundaries |

**Separators priority explained:**
```
"\n\n" → blank line between paragraphs (best split point)
"\n"   → end of a line (good split point)
". "   → end of sentence (acceptable split point)
" "    → between words (last resort)
""     → between characters (absolute last resort)
Splitting the documents:
pythonchunks = text_splitter.split_documents(documents)
split_documents() vs split_text():
MethodInputOutputPreserves Metadatasplit_text()Plain stringList of strings❌ Nosplit_documents()Document objectsDocument objects✅ Yes
Always use split_documents() — preserves metadata so each chunk remembers which file it came from.

Function 2 — inspect_chunks()
pythonfor i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---")
    print(f"Characters: {len(chunk.page_content)}")
    print(f"Metadata: {chunk.metadata}")
    print(f"Content:")
    print(chunk.page_content)
Each chunk is still a LangChain Document object with:

chunk.page_content → the actual text of this chunk
chunk.metadata → source file information carried over from original document
len(chunk.page_content) → character count of this specific chunk

Why inspect chunks? To visually verify the splitting makes logical sense. If a chunk cuts mid-transaction it means chunk size needs adjustment.

Function 3 — chunk_statistics()
pythonchunk_lengths = [len(chunk.page_content) for chunk in chunks]

print(f"Total chunks     : {len(chunks)}")
print(f"Smallest chunk   : {min(chunk_lengths)} characters")
print(f"Largest chunk    : {max(chunk_lengths)} characters")
print(f"Average chunk    : {sum(chunk_lengths) // len(chunk_lengths)} characters")
List comprehension to get all lengths:
pythonchunk_lengths = [len(chunk.page_content) for chunk in chunks]
# Result: [287, 243, 198, 187]
Built-in Python functions:
FunctionExampleResultlen(chunks)len([287, 243, 198, 187])4min(chunk_lengths)min([287, 243, 198, 187])187max(chunk_lengths)max([287, 243, 198, 187])287sum(chunk_lengths)sum([287, 243, 198, 187])915
Integer division //:
pythonsum(chunk_lengths) // len(chunk_lengths)
915 // 4 = 228  ← rounds down, no decimals
```
Regular division `/` would give `228.75` — we use `//` for clean whole numbers.

**What good statistics look like:**
```
✅ Good:
Total chunks     : 4
Smallest chunk   : 187 characters
Largest chunk    : 287 characters  ← close to chunk_size (300)
Average chunk    : 243 characters

❌ Warning signs:
Largest chunk: 1500 characters     ← way over chunk_size, separator not found
Smallest chunk: 5 characters       ← too small, chunk_size too low

Main Block
pythonif __name__ == "__main__":
    file_path = "data/sample_statement.txt"
    documents = load_document(file_path)
    cleaned_docs = preprocess_document(documents)
    chunks = create_chunks(cleaned_docs)
    inspect_chunks(chunks)
    chunk_statistics(chunks)
```

**Full pipeline so far:**
```
load_document()       → raw Document objects
      ↓
preprocess_document() → cleaned Document objects
      ↓
create_chunks()       → list of chunk Document objects
      ↓
inspect_chunks()      → visual verification
      ↓
chunk_statistics()    → numerical verification
```

---

## 🧪 Experiments Done

**Experiment 1 — Very small chunks (chunk_size=100):**
More chunks created, less context per chunk. AI may struggle to understand transaction context.

**Experiment 2 — Very large chunks (chunk_size=600):**
Fewer chunks, more context per chunk. May include irrelevant transactions in same chunk.

**Experiment 3 — No overlap (chunk_overlap=0):**
Chunks don't share content. Risk of losing information at chunk boundaries.

**Conclusion:** `chunk_size=300, chunk_overlap=50` is the sweet spot for our bank statement.

---

## 🗺️ RAG Pipeline Progress
```
✅ Step 1: Load Document       ← Day 5
✅ Step 2: Preprocess Text     ← Day 5
✅ Step 3: Split into Chunks   ← Done today!
⏳ Step 4: Create Embeddings   ← Day 7
⏳ Step 5: Store in ChromaDB   ← Day 8
⏳ Step 6: Query & Retrieve    ← Day 9
⏳ Step 7: Generate Answer     ← Day 9
⏳ Step 8: Integrate into UI   ← Day 10

💡 Key Python Concepts Learned
ConceptExampleMeaningDefault parametersdef func(size=300)Parameter has default value if not providedImporting own functionsfrom document_loader import load_documentReuse code across filessplit_documents()Preserves metadataAlways use over split_text()Integer division //915 // 4 = 228Divide and round down, no decimalsBuilt-in functionsmin(), max(), sum()Python list operationsDocstring with Args/Returns"""Args: ... Returns: ..."""Professional function documentation

⚠️ Issues Faced & Solutions
IssueSolutionNoneSmooth day! ✅

✅ Day 6 Checklist

 Understand why chunking is needed
 Understand chunk size and chunk overlap concepts
 Understand RecursiveCharacterTextSplitter strategy
 Understand separator priority order
 Created src/text_chunker.py
 Successfully split bank statement into chunks
 Inspected individual chunks
 Reviewed chunk statistics
 Tried different chunk size experiments
 Committed and pushed to GitHub