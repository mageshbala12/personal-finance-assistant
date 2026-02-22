Code Breakdown — chatbot.py

📦 Piece 1 — Imports
pythonimport google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st
import os
What is this doing?
Think of imports like bringing tools into your workshop before starting work.
ImportPurposegoogle.generativeai as genaiBrings in the Gemini AI SDK. We give it a short nickname genai so we don't have to type the full name every timefrom dotenv import load_dotenvBrings in the function that reads your .env file and loads your API keyimport streamlit as stBrings in Streamlit UI framework. Nicknamed st for shortimport osBuilt-in Python library to interact with your operating system — we use it to read environment variables

🔑 Piece 2 — Load API Key
pythonload_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
What is this doing?
load_dotenv() opens your .env file and loads everything in it into memory.
os.getenv("GEMINI_API_KEY") reads the value of GEMINI_API_KEY from that memory.
genai.configure(api_key=...) hands that key to Gemini SDK so every request you make is authenticated.
Why not just hardcode the key directly?
You could write api_key="AIzaSy..." directly — it would work. But if you push that to GitHub, your key is exposed to the world. Anyone can use it and rack up charges on your account. The .env approach keeps it safe.

🤖 Piece 3 — Model Setup with System Instruction
pythonmodel = genai.GenerativeModel(
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
What is this doing?
genai.GenerativeModel() creates your AI model instance — think of it as hiring an employee.
model_name tells which Gemini model to use — like choosing a junior vs senior employee.
system_instruction is the most important part — this is your private instructions to the AI that the user never sees. It sets:

Persona → who the AI is (personal finance assistant)
Scope → what topics it covers (SIP, mutual funds, tax)
Behavior → how it should act (practical, easy to understand)
Boundary → what it should NOT do (redirect non-finance questions)

Try this experiment: Remove the system_instruction and ask about biryani recipe — it will answer! Add it back and it redirects. That's the power of system instructions.

🖥️ Piece 4 — Page Configuration
pythonst.set_page_config(
    page_title="Personal Finance Assistant",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Personal Finance Assistant")
st.caption("Ask me anything about budgeting, investments, SIP, tax planning and more!")
What is this doing?
st.set_page_config() configures the browser tab — title and icon you see in the browser tab.
st.title() displays the big heading on the page.
st.caption() displays the small subtitle text below the heading.
This is pure UI — nothing AI related here. Streamlit makes UI incredibly simple. What would take 50 lines of HTML/CSS takes 3 lines here.

🧠 Piece 5 — Session State (Most Important Concept!)
pythonif "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

if "messages" not in st.session_state:
    st.session_state.messages = []
What is this doing?
This is the most important concept in Streamlit to understand.
Every time you type a message and press Enter, Streamlit reruns the entire Python file from top to bottom. Every variable gets reset. So how does it remember previous messages?
That's what st.session_state solves — it's a special dictionary that survives reruns. Anything you store in it persists across reruns.
st.session_state.chat → stores the Gemini chat session with full conversation history. This is what gives Gemini memory of previous messages.
st.session_state.messages → stores the display history — the list of messages shown on screen.
Why two separate histories?
VariablePurposest.session_state.chatGemini's memory — sent to AI with every messagest.session_state.messagesScreen display — what you see in the UI
They serve different purposes but stay in sync.

💬 Piece 6 — Display Existing Messages
pythonfor message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
What is this doing?
Remember — Streamlit reruns the file on every message. So every time it reruns, this loop redraws all previous messages on screen from the saved history.
st.chat_message(message["role"]) creates a chat bubble — automatically shows user bubble on right, assistant bubble on left based on the role value ("user" or "assistant").
st.markdown() renders the text with formatting — bold, bullet points, etc.

⌨️ Piece 7 — Handle New User Input
pythonif prompt := st.chat_input("Ask a finance question..."):
What is this doing?
st.chat_input() renders the text input box at the bottom of the screen.
:= is called the walrus operator — it does two things at once:

Captures whatever the user typed into prompt
Checks if it's not empty

So the entire block inside only runs when the user actually types something and presses Enter.

📤 Piece 8 — Send Message and Get Response
python    # Display user message
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
```

**What is this doing?** This is the **heart of the chatbot** — 4 things happen in sequence:

**1 — Display user message** on screen immediately so the user sees what they typed.

**2 — Save user message** to `st.session_state.messages` so it survives the next rerun.

**3 — Send to Gemini** using `st.session_state.chat.send_message(prompt)` — this sends the new message AND the full conversation history to Gemini. `st.spinner("Thinking...")` shows a loading animation while waiting for response. `response.text` contains Gemini's reply.

**4 — Save assistant response** to history so it also survives the next rerun.

---

## 🗺️ Complete Flow Diagram
```
User types message
        ↓
Streamlit captures it
        ↓
Display user message on screen
        ↓
Save to session_state.messages
        ↓
Send to Gemini (with full history)
        ↓
Gemini returns response
        ↓
Display response on screen
        ↓
Save to session_state.messages
        ↓
Page reruns → loop redraws all messages
