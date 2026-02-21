Day 1 Documentation — Stage 0 Setup
Personal Finance Assistant Project

🎯 Objective
Set up the complete development environment, connect to Gemini AI API, and push the project to GitHub.

🛠️ Prerequisites
    • Windows OS
    • VS Code installed
    • Python installed

📋 Steps Followed
Step 1 — Verify Python Installation
Opened VS Code terminal (Ctrl + backtick) and ran:
bash
python --version
Expected output: Python 3.11.x or similar

Step 2 — Create Project Folder
Created project folder at the desired location:
bash
mkdir C:\D_Drive\02Magesh\AI\projects\personal-finance-assistant
cd C:\D_Drive\02Magesh\AI\projects\personal-finance-assistant
code .
This opens the project folder in VS Code.

Step 3 — Create Virtual Environment
Created and activated a virtual environment:
bash
python -m venv venv
venv\Scripts\activate
✅ Success indicator: (venv) appears at the start of the terminal line.
    Why virtual environment? Keeps project dependencies isolated from other Python projects on your machine.

Step 4 — Install Dependencies
Installed required packages for Stage 0:
bash
pip install google-generativeai streamlit python-dotenv
```
Packages installed:
- `google-generativeai` — Gemini API SDK
- `streamlit` — UI framework
- `python-dotenv` — Secure API key management

---
### Step 5 — Get Gemini API Key
1. Went to **https://aistudio.google.com/app/apikey**
2. Signed in with Google account
3. Clicked **"Create API Key"**
4. Copied the generated key (looks like `AIzaSy...`)
---
### Step 6 — Store API Key Securely
Created `.env` file in project root and added:
```
GEMINI_API_KEY=your_actual_key_here
```
Created `.gitignore` file and added:
```
venv/
.env
__pycache__/
*.pyc
    Why .gitignore? Prevents secret API key and unnecessary files from being uploaded to GitHub.

Step 7 — Create Project Structure
bash
mkdir src data docs
```
Final project structure:
```
personal-finance-assistant/
│
├── src/                  ← All Python code
├── data/                 ← PDFs and documents
├── docs/                 ← Notes and documentation
├── .env                  ← Secret keys (never share)
├── .gitignore            ← Git ignore rules
└── venv/                 ← Virtual environment

Step 8 — First Gemini API Call
Created src/test_gemini.py:
python
import google.generativeai as genai
from dotenv import load_dotenv
import os
# Load your API key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
# Configure Gemini
genai.configure(api_key=api_key)
# Choose the model
model = genai.GenerativeModel("gemini-2.5-flash-lite")
# Send a message
response = model.generate_content("What are the top 3 things a beginner should know about personal finance in India?")
# Print the response
print(response.text)
Ran it:
bash
python src/test_gemini.py
✅ Success indicator: Gemini responds with finance tips.
    Note: To find available models on your API key, use this script:
    python
    for model in genai.list_models():
    if "generateContent" in model.supported_generation_methods:
        print(model.name)
    We used gemini-2.5-flash-lite as it was the latest stable model available.

Step 9 — Git & GitHub Setup
Install Git:
    • Downloaded from https://git-scm.com/download/win
    • Installed with all default options
    • Fully restarted VS Code after installation
Configure Git identity:
bash
git config --global user.name "Magesh Balasubramanian"
git config --global user.email "your_email@gmail.com"
Verify Git config:
bash
git config --global --list
Initialize and commit:
bash
git init
git add .
git commit -m "Initial project setup - Stage 0"
Create GitHub repository:
    1. Went to https://github.com
    2. Clicked "+" → "New repository"
    3. Name: personal-finance-assistant
    4. Visibility: Private
    5. ❌ Did NOT check "Add a README file"
    6. Clicked "Create repository"
Push to GitHub:
bash
git remote add origin https://github.com/YOUR_USERNAME/personal-finance-assistant.git
git branch -M main
git push -u origin main
Authenticated via browser popup (Option 1 — Sign in via Browser).
✅ Success indicator: Project files visible at https://github.com/YOUR_USERNAME/personal-finance-assistant

⚠️ Issues Faced & Solutions
Issue	Solution
gemini-1.5-flash model not found	Listed available models and switched to gemini-2.5-flash-lite
git not recognized in terminal	Installed Git from git-scm.com, fully restarted VS Code
Git commit failed — Author unknown	Ran git config --global user.name and user.email
GitHub OAuth authorization disabled	Used browser sign-in option instead

✅ Day 1 Checklist
    • Python version confirmed
    • Project folder created and opened in VS Code
    • Virtual environment created and activated
    • Packages installed
    • Gemini API key obtained and saved in .env
    • .gitignore created
    • Project folder structure created
    • First Gemini API call working with gemini-2.5-flash-lite
    • Git installed and configured
    • Code pushed to GitHub
