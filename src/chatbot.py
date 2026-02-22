import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st
import os

# ── Load API key from .env ──────────────────────────────────────────
# Works locally (loads from .env) AND in production (loads from Render environment variables)
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
# session_state persists data across reruns of the app
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