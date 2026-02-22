Day 2 Documentation — Streamlit Chatbot & Production Deployment
Personal Finance Assistant Project

🎯 Objective
Build a Finance Chatbot UI using Streamlit, understand the code deeply, and deploy it to production on Render with automatic GitHub integration.

🛠️ Prerequisites
    • Day 1 setup completed
    • GitHub repository created
    • Gemini API key available
    • Render account (free)

📋 Part 1 — Build the Chatbot
Step 1 — Create Chatbot File
In VS Code Explorer panel:
    1. Click on src folder
    2. Click "New File" icon
    3. Name it chatbot.py
Step 2 — Write Chatbot Code
Created src/chatbot.py with the following code:
python
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st
import os
# ── Load API key from .env ──────────────────────────────────────────
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Please set it in .env or environment variables.")
genai.configure(api_key=api_key)
# ── Model setup ─────────────────────────────────────────────────────
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash-lite",
    system_instruction="""
    You are a helpful personal finance assistant for Indian users.
    You help with topics like budgeting, savings, investments, 
    mutual funds, SIP, fixed deposits, tax planning, and general 
    money management.
    Always give practical, easy to understand advice.
    If asked about anything unrelated to finance, politely 
    redirect the conversation back to finance topics.
    """
)
# ── Page configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="Personal Finance Assistant",
    page_icon="💰",
    layout="centered"
)
st.title("💰 Personal Finance Assistant")
st.caption("Ask me anything about budgeting, investments, SIP, tax planning and more!")
# ── Initialize chat history in session state ─────────────────────────
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])
if "messages" not in st.session_state:
    st.session_state.messages = []
# ── Display existing chat messages ───────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
# ── Handle new user input ─────────────────────────────────────────────
if prompt := st.chat_input("Ask a finance question..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    # Save user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    # Send to Gemini and get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat.send_message(prompt)
            st.markdown(response.text)
    # Save assistant response to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.text
    })
Step 3 — Run Locally
bash
streamlit run src/chatbot.py
App opens at http://localhost:8501

📚 Part 2 — Code Understanding
Code Breakdown
Imports — Bring in required libraries:
    • google.generativeai — Gemini AI SDK
    • streamlit — UI framework
    • python-dotenv — Secure API key management
    • os — Read environment variables
API Key Loading — Loads key from .env locally and from Render environment variables in production. Raises clear error if key is missing.
System Instruction — Private instructions to Gemini that:
    • Sets persona (personal finance assistant for Indian users)
    • Defines scope (SIP, mutual funds, tax planning etc.)
    • Sets behavior (practical, easy to understand)
    • Sets boundary (redirect non-finance questions)
Session State — Special Streamlit dictionary that survives page reruns:
    • st.session_state.chat → Gemini's conversation memory
    • st.session_state.messages → Screen display history
Chat Flow — On every message:
    1. Display user message on screen
    2. Save to session state
    3. Send to Gemini with full history
    4. Display Gemini response
    5. Save response to session state

🚀 Part 3 — Production Deployment
Step 1 — Create Streamlit Config
bash
mkdir .streamlit
Created .streamlit/config.toml:
toml
[server]
headless = true
port = 10000
[browser]
gatherUsageStats = false
Why? headless = true runs without browser on server. port = 10000 is Render's expected port.
Step 2 — Create requirements.txt
Initially generated with:
bash
pip freeze > requirements.txt
```
Had dependency conflicts so simplified to:
```
google-generativeai
streamlit
python-dotenv
Why simplified? pip freeze captures all packages with exact versions that sometimes conflict. Using only package names without versions lets Render resolve compatible versions automatically.
Step 3 — Create README.md
Created in project root:
markdown
# 💰 Personal Finance Assistant
A personal finance chatbot powered by Google Gemini AI.
## Features
- Ask questions about budgeting, SIP, mutual funds, tax planning
- Multi-turn conversation with memory
- Focused on Indian personal finance
## Tech Stack
- Python
- Google Gemini AI
- Streamlit
- LangChain (coming soon)
## Stages
- [x] Stage 0 - Basic Chatbot
- [ ] Stage 1 - RAG with documents
- [ ] Stage 2 - AI Agents
- [ ] Stage 3 - Agentic AI with MCP
Step 4 — Push to GitHub
bash
git add .
git commit -m "Production ready - Stage 0 basic chatbot"
git push
```
### Step 5 — Deploy on Render
**Create Account:**
1. Go to **https://render.com**
2. Sign up using GitHub account
**Connect GitHub:**
1. Click **"New +"** → **"Web Service"**
2. Click **"GitHub"** to authorize Render
3. Select your GitHub account
4. Choose `personal-finance-assistant` repository
5. Click **"Install & Authorize"**
**Configure Service:**
| Field | Value |
|-------|-------|
| Name | `personal-finance-assistant` |
| Region | Singapore |
| Branch | `main` |
| Runtime | `Python 3` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `streamlit run src/chatbot.py` |
| Instance Type | Free |
**Add Environment Variable:**
- Click **"Advanced"** → **"Add Environment Variable"**
- Key: `GEMINI_API_KEY`
- Value: your Gemini API key
**Deploy:**
- Click **"Deploy Web Service"**
- Wait 3-5 minutes for deployment to complete
**Manual Redeploy (if needed):**
- Go to service dashboard
- Click **"Manual Deploy"** → **"Deploy latest commit"**
### Step 6 — Live URL
After successful deployment:
```
https://personal-finance-assistant-xxxx.onrender.com

🔄 Future Deployments
Every time you improve the app and push to GitHub, Render automatically redeploys:
bash
git add .
git commit -m "your message"
git push
Same URL throughout all stages — RAG, Agents, MCP!

⚠️ Issues Faced & Solutions
Issue	Solution
Dependency conflict in requirements.txt	Simplified to only 3 core packages without version numbers
Render not auto-deploying	Used "Manual Deploy" → "Deploy latest commit"
GitHub repositories not found in Render	Clicked GitHub button to authorize Render access to repositories

🔐 API Key Security
Location	Safety
Render environment variables	✅ Encrypted, safe
.env file (local)	✅ Safe, never pushed to GitHub
Hardcoded in code	❌ Never do this
Pushed to GitHub	❌ Never do this

✅ Day 2 Checklist
    • chatbot.py created with multi-turn conversation
    • Streamlit UI running locally at localhost:8501
    • Code understood piece by piece
    • .streamlit/config.toml created for production
    • requirements.txt created and conflict resolved
    • README.md created
    • Code pushed to GitHub
    • Render account created and GitHub connected
    • Environment variable set on Render
    • App deployed and live on production URL
