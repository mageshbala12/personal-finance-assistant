Day 9 Documentation — Full RAG Pipeline
Personal Finance Assistant Project

🎯 Objective
Connect all RAG components built in Days 5-8 into a single end-to-end pipeline that answers questions about personal financial documents using Gemini AI.

📚 Part 1 — Key Concepts
What Changed from Day 2 Chatbot
Day 2 Chatbot (No RAG):
────────────────────────
User: "How much did I spend on Zomato?"
Bot:  "I don't have access to your personal data" ❌

Day 9 RAG Pipeline:
────────────────────
User: "How much did I spend on Zomato?"
Bot:  "You spent ₹2,050 on Zomato in January 2025:
       - 03-Jan: ₹850
       - 12-Jan: ₹650
       - 20-Jan: ₹550" ✅
Complete RAG Flow
User Question
      ↓
Search ChromaDB → Get relevant chunks
      ↓
Build prompt with question + chunks
      ↓
Send to Gemini
      ↓
Get accurate answer about YOUR finances
      ↓
Show answer with source reference

RAG Limitations Discovered Today
Type of QuestionRAG ResultWhy"What did I spend on Zomato?"✅ CorrectAll data in one chunk"What are my SIP investments?"✅ CorrectAll SIP data in one chunk"What was my salary?"✅ CorrectSingle transaction, easy to find"What % of salary did I save?"❌ FailedData spread across chunks, needs calculation
Key insight: RAG retrieves text. When answer requires combining information from multiple chunks or complex calculations — RAG struggles. This is exactly what Stage 2 AI Agents solves.

Prompt Engineering
Carefully crafting the prompt to get best answers from LLM:
System Message → Tells Gemini its role and rules
Human Message  → Context from documents + User question
Rule that made the difference:
❌ "Answer ONLY based on context"
   → Too strict, refuses to calculate even when data is present

✅ "If answer requires calculation use numbers from context"
   → Encourages Gemini to calculate when data is available

💻 Part 2 — Code Created
File: src/rag_pipeline.py
pythonfrom langchain_google_genai import ChatGoogleGenerativeAI
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
    """
    print("\n📥 Starting document indexing...")

    documents    = load_document(file_path)
    cleaned_docs = preprocess_document(documents)
    chunks       = create_chunks(cleaned_docs)

    embedding_model = get_embedding_model()
    vector_store    = create_vector_store(chunks, embedding_model)

    print("✅ Document indexing complete!")
    return vector_store


# ── Step 3: Build RAG prompt ──────────────────────────────────────────
def build_rag_prompt(question, relevant_chunks):
    """
    Build a prompt that includes system instruction,
    retrieved context and user question.
    """

    context_parts = []
    for i, (doc, score) in enumerate(relevant_chunks):
        context_parts.append(f"[Context {i+1}]\n{doc.page_content}")

    context_text = "\n\n".join(context_parts)

    system_message = SystemMessage(content="""
    You are a helpful personal finance assistant for Indian users.
    You have been provided with relevant sections from the user's
    financial documents as context.

    Rules:
    1. Answer based on the provided context
    2. If exact answer exists in context state it directly
    3. If answer requires calculation use the numbers
       from context to calculate and show your working
    4. If information is completely missing from context say
       "I couldn't find that information in your documents"
    5. Always mention specific amounts and dates from context
    6. Be concise and clear in your answers
    7. Format currency as ₹ with Indian number format
    """)

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
    """
    print(f"\n❓ Question: {question}")
    print("-"*50)

    relevant_chunks = search_vector_store(
        vector_store,
        question,
        top_k=TOP_K_CHUNKS
    )

    messages = build_rag_prompt(question, relevant_chunks)

    print("🤔 Generating answer...")
    response = llm.invoke(messages)

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

    embedding_model = get_embedding_model()
    llm             = get_llm()

    file_path    = "data/sample_statement.txt"
    vector_store = index_documents(file_path)

    # OR load existing (uncomment after first run):
    # vector_store = load_vector_store(embedding_model)

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

🔍 Part 3 — Detailed Code Explanation
Imports
pythonfrom langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from vector_store import (
    get_embedding_model,
    create_vector_store,
    load_vector_store,
    search_vector_store
)
from document_loader import load_document, preprocess_document
from text_chunker import create_chunks
ImportPurposeChatGoogleGenerativeAIGemini LLM for generating answersHumanMessage, SystemMessageLangChain message types for structured promptsget_embedding_model, create_vector_storeOur Day 8 functionsload_document, preprocess_documentOur Day 5 functionscreate_chunksOur Day 6 function
Key learning — importing from our own modules:
pythonfrom vector_store import (
    get_embedding_model,
    create_vector_store,
    load_vector_store,
    search_vector_store
)
We import multiple functions from one file using parentheses. This is called multi-line import — clean and readable. All our previous days' work reused here without rewriting a single line.
Why langchain_core.messages not langchain.schema?
LangChain reorganized its modules in newer versions. langchain_core is now the correct location for core components like messages. Always check import errors — they often just mean the module moved.

Function 1 — get_llm()
pythondef get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.2
    )
```

**`temperature=0.2`** — This is a very important parameter:
```
Temperature controls randomness/creativity of responses:

temperature=0.0 → Completely deterministic
                  Same question always gets same answer
                  Best for: factual questions, calculations

temperature=0.2 → Mostly deterministic with slight variation
                  Our choice — factual but naturally worded
                  Best for: finance Q&A

temperature=0.7 → More creative and varied
                  Best for: creative writing, brainstorming

temperature=1.0 → Maximum creativity/randomness
                  Best for: poetry, fiction
For finance questions we want accurate, consistent answers — hence low temperature of 0.2.

Function 2 — index_documents()
pythondef index_documents(file_path):
    documents    = load_document(file_path)
    cleaned_docs = preprocess_document(documents)
    chunks       = create_chunks(cleaned_docs)
    embedding_model = get_embedding_model()
    vector_store    = create_vector_store(chunks, embedding_model)
    return vector_store
```

This function is the **complete indexing pipeline** in one place. Notice how it calls our previous days' functions in sequence:
```
load_document()        ← Day 5
      ↓
preprocess_document()  ← Day 5
      ↓
create_chunks()        ← Day 6
      ↓
get_embedding_model()  ← Day 7
      ↓
create_vector_store()  ← Day 8
      ↓
returns vector_store
This is called function composition — building complex behavior by combining simpler functions. Each function does one job. Together they form a pipeline.

Function 3 — build_rag_prompt()
This is the most important function in the RAG pipeline — prompt engineering.
Building context from chunks:
pythoncontext_parts = []
for i, (doc, score) in enumerate(relevant_chunks):
    context_parts.append(f"[Context {i+1}]\n{doc.page_content}")

context_text = "\n\n".join(context_parts)
```

`context_parts` → empty list to collect formatted chunks.
`f"[Context {i+1}]\n{doc.page_content}"` → labels each chunk with a number so Gemini knows where each piece of information comes from.

Result looks like:
```
[Context 1]
03-Jan Zomato Food Order 850
12-Jan Zomato Food Order 650

[Context 2]
15-Jan SIP Axis Bluechip 5000
28-Jan SIP Parag Parikh 3000

[Context 3]
HDFC BANK header information...
"\n\n".join(context_parts) → joins all context parts with blank line between them for readability.
System Message:
pythonsystem_message = SystemMessage(content="""
You are a helpful personal finance assistant...
Rules:
1. Answer based on the provided context
2. If exact answer exists state it directly
3. If answer requires calculation use numbers from context
...
""")
SystemMessage → invisible to user, sets Gemini's behavior and rules. Like giving instructions to an employee before they start work.
Key rules explained:
RulePurposeAnswer based on contextPrevents Gemini from hallucinatingCalculate when neededAllows math on available dataSay "couldn't find" if missingHonest response, no guessingMention amounts and datesForces specific, verifiable answersFormat as ₹India-specific formatting
Human Message:
pythonhuman_message = HumanMessage(content=f"""
Here are relevant sections from your financial documents:
{context_text}
Based on the above information, please answer this question:
{question}
""")
```

`HumanMessage` → the actual question with context injected. Gemini sees both the retrieved chunks AND the question together. This is the **augmentation** part of Retrieval Augmented Generation.

**Why two separate messages?**
```
SystemMessage → Persistent instructions (role, rules, behavior)
HumanMessage  → Per-query content (context + question)

This separation is important:
- System stays same for every query
- Human message changes per query
- Gemini treats them differently internally

Function 4 — rag_query()
pythondef rag_query(question, vector_store, llm):

    # Step 1: Retrieve
    relevant_chunks = search_vector_store(
        vector_store, question, top_k=TOP_K_CHUNKS
    )

    # Step 2: Augment
    messages = build_rag_prompt(question, relevant_chunks)

    # Step 3: Generate
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
```

**Three steps of RAG in one function:**
```
Retrieve  → search_vector_store()
Augment   → build_rag_prompt()
Generate  → llm.invoke()
Source extraction:
pythonsources = list(set([
    doc.metadata.get('source', 'unknown')
    for doc, score in relevant_chunks
]))
Breaking this down inside out:
doc.metadata.get('source', 'unknown') → get source filename from each chunk's metadata.
[... for doc, score in relevant_chunks] → list comprehension over all retrieved chunks.
set([...]) → removes duplicates. If 3 chunks all came from same file, set keeps only one copy.
list(set([...])) → converts set back to list for consistent handling.
Result: ["data/sample_statement.txt"] — tells user which document the answer came from.
Return dictionary:
pythonreturn {
    "question" : question,
    "answer"   : response.content,
    "sources"  : sources,
    "chunks"   : relevant_chunks
}
Returns everything as a dictionary so the caller can use any part — display the answer, show sources, inspect chunks for debugging.

Function 5 — display_result()
pythondef display_result(result):
    print(f"\n❓ Question: {result['question']}")
    print(f"\n💡 Answer:\n{result['answer']}")
    print(f"\n📄 Sources: {', '.join(result['sources'])}")
result['question'] → access dictionary value by key.
', '.join(result['sources']) → joins list of sources with comma separator.
Separation of concerns:
rag_query() handles logic. display_result() handles display. Two separate functions — each does one job. This makes it easy to later replace display_result() with a Streamlit UI display function without touching the logic.

Main Block — Test Questions
pythontest_questions = [
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
```

**Why test 6 different questions?**

| Question | Tests |
|----------|-------|
| Zomato spending | Multi-transaction aggregation |
| SIP investments | Investment data retrieval |
| Salary | Single transaction retrieval |
| Groceries | Specific category retrieval |
| Closing balance | Balance information retrieval |
| Savings percentage | Cross-chunk calculation (limitation test) |

The last question intentionally tests RAG's limitation — revealing why AI Agents are needed in Stage 2.

---

## 📊 Part 4 — Actual Results

| Question | Answer Quality | Reason |
|----------|---------------|--------|
| Zomato spending | ✅ ₹2,050 with breakdown | All transactions in one chunk |
| SIP investments | ✅ ₹8,000 total | SIP data in one chunk |
| Salary | ✅ ₹85,000 on 05-Jan | Single clear transaction |
| Groceries | ✅ ₹3,200 DMart on 25-Jan | Single transaction found |
| Closing balance | ✅ ₹1,13,551 | Explicitly in document |
| Savings % | ⚠️ Partial/incorrect | Data spread across chunks |

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
⏳ Step 8: Integrate into UI   ← Day 10

💡 Key Python Concepts Learned
ConceptExampleMeaningMulti-line importfrom module import (a, b, c)Import multiple items cleanlyTemperaturetemperature=0.2Controls LLM randomnessFunction compositionCalling functions inside functionsBuild complex behavior from simple partsSystemMessagePersistent AI instructionsSets role and rules for every queryHumanMessagePer-query contentChanges with each questionset()Remove duplicates from listUnique values onlySeparation of concernsLogic vs Display in separate functionsEach function does one jobTuple unpacking in comprehensionfor doc, score in resultsUnpack while iterating

⚠️ Issues Faced & Solutions
IssueSolutionModuleNotFoundError: No module named 'langchain.schema'Changed import to from langchain_core.messages import HumanMessage, SystemMessageSavings % question returned "couldn't find"Updated prompt rules to encourage calculation from available context dataSavings % still partially incorrectIdentified as RAG limitation — data spread across chunks. Will be solved in Stage 2 with AI Agents

✅ Day 9 Checklist

 Understand how all RAG pieces connect
 Understand prompt engineering — SystemMessage vs HumanMessage
 Understand temperature parameter
 Understand function composition
 Created src/rag_pipeline.py
 Fixed langchain.schema import error
 Successfully indexed document
 5 out of 6 test questions answered correctly
 Understood RAG limitation for cross-chunk calculations
 Committed and pushed to GitHub