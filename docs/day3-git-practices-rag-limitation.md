Day 3 Documentation — Git Best Practices & RAG Limitation
Personal Finance Assistant Project

🎯 Objective
Learn professional Git workflows, understand branching strategy, and experience the limitation of the current chatbot that RAG will solve.

📋 Part 1 — Git Best Practices
Concept 1 — Branching Strategy
In real projects, code is never written directly on main branch.
main branch        → Production code (always stable and deployed)
feature branches   → Where new features are built and tested
Workflow:
main (stable) ──────────────────────────→ production
                  ↑               ↑
            feature/rag    feature/agents
            (build here)   (build here)
Commands used:
Create and switch to new branch:
bash
git checkout -b feature/rag
Verify current branch:
bash
git branch
```
Output:
```
* feature/rag
  main
The * indicates current branch.
Push branch to GitHub:
bash
git push -u origin feature/rag
```
---
### Concept 2 — Professional Commit Messages
**Format:**
```
type: short description
```
**Types:**
| Type | When to Use |
|------|------------|
| `feat` | New feature added |
| `fix` | Bug fix |
| `docs` | Documentation update |
| `refactor` | Code restructure without feature change |
| `test` | Adding or updating tests |
**Examples:**
```
feat: add PDF document loader
fix: resolve ChromaDB connection error
docs: update README with RAG instructions
refactor: simplify embedding pipeline
test: add unit tests for chunking

Concept 3 — Daily Git Workflow
Follow this every single day:
bash
# 1. Start of day - pull latest code
git pull
# 2. Make sure you're on right branch
git branch
# 3. Write your code...
# 4. Check what changed
git status
# 5. See exactly what changed line by line
git diff
# 6. Stage your changes
git add .
# 7. Commit with proper message
git commit -m "feat: add document loader for PDF files"
# 8. Push to GitHub
git push
```
---
### Concept 4 — Updated `.gitignore`
Updated `.gitignore` with comprehensive rules:
```
# Virtual Environment
venv/
env/
.env
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
*.egg-info/
dist/
build/
# VS Code
.vscode/
# Data files - never push personal financial data to GitHub!
data/
*.pdf
*.csv
*.xlsx
# Vector database files (will be created in Stage 1)
chroma_db/
*.sqlite3
# OS files
.DS_Store
Thumbs.db
    Important: data/ folder is in .gitignore — personal bank statements and financial PDFs must never be pushed to GitHub even in a private repository.
Committed with:
bash
git add .
git commit -m "docs: update gitignore with comprehensive rules"
git push
```
---
## 📋 Part 2 — Experiencing the RAG Limitation
### Step 1 — Created Sample Bank Statement
Created `data/sample_statement.txt`:
```
HDFC BANK - ACCOUNT STATEMENT
Account Holder: Magesh Balasubramanian
Account Number: XXXX1234
Period: January 2025
DATE        DESCRIPTION                  DEBIT      CREDIT     BALANCE
01-Jan      Opening Balance                                     50,000
03-Jan      Zomato Food Order            850                    49,150
05-Jan      Salary Credit                           85,000      1,34,150
07-Jan      Amazon Shopping              2,300                  1,31,850
10-Jan      Electricity Bill             1,450                  1,30,400
12-Jan      Zomato Food Order            650                    1,29,750
15-Jan      SIP - Axis Bluechip Fund     5,000                  1,24,750
18-Jan      Petrol                       2,000                  1,22,750
20-Jan      Zomato Food Order            550                    1,22,200
22-Jan      Netflix Subscription         649                    1,21,551
25-Jan      Grocery - DMart              3,200                  1,18,351
28-Jan      SIP - Parag Parikh Fund      3,000                  1,15,351
30-Jan      Restaurant - Dinner          1,800                  1,13,551
31-Jan      Closing Balance                                     1,13,551
    Note: This file stays only on local machine — never pushed to GitHub because data/ is in .gitignore.

Step 2 — Tested Chatbot Limitations
Ran chatbot locally:
bash
streamlit run src/chatbot.py
```
Asked these questions and observed responses:
| Question | Expected Answer | Chatbot Response |
|----------|----------------|-----------------|
| How much did I spend on Zomato in January 2025? | ₹2,050 (850+650+550) | ❌ No access to personal data |
| What is my total SIP investment this month? | ₹8,000 (5000+3000) | ❌ No access to personal data |
| How much did I spend on food and dining? | ₹3,850 (850+650+550+1800) | ❌ No access to personal data |
| What percentage of my salary did I save? | ~34% | ❌ No access to personal data |
---
### Step 3 — Understanding the Gap
**Current Architecture (Stage 0):**
```
Your Question → Gemini → Answer
                ↑
         Only knows general
         internet training data
         Cannot access your
         personal documents
```
**What RAG Solves (Stage 1):**
```
Your Statement → Read → Store → Search
                                  ↓
Your Question ──────────────→ Find relevant parts
                                  ↓
                    Question + Your Data → Gemini → Answer

💡 Key Concepts Learned
Branching — Never code directly on main. Feature branches keep production stable while you build new features.
Commit messages — Follow type: description format for clean, readable Git history.
Daily workflow — pull → branch check → code → status → diff → add → commit → push.
gitignore — Personal and sensitive data files must never be pushed to GitHub under any circumstances.
RAG limitation — Current chatbot only knows general finance from Gemini's training. It cannot answer questions about personal finances without RAG.

⚠️ Issues Faced & Solutions
Issue	Solution
None today	Smooth sailing! ✅

✅ Day 3 Checklist
    • Created feature/rag branch
    • Understand branching concept
    • Updated .gitignore with comprehensive rules
    • Follow professional commit message format
    • Understand daily Git workflow
    • Created sample bank statement in data/
    • Tested chatbot limitations with personal finance questions
    • Clearly understand WHY RAG is needed
