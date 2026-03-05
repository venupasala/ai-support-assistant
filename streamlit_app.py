"""
streamlit_app.py
~~~~~~~~~~~~~~~~
Beautiful Streamlit frontend for the AI Customer Support RAG system.
Connects to the FastAPI backend running at http://localhost:8000.
"""

from __future__ import annotations

import time
from pathlib import Path

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="AI Support Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #e8e8f0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* Chat bubbles */
    .chat-user {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 12px;
    }
    .chat-user .bubble {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: #fff;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 18px;
        max-width: 70%;
        box-shadow: 0 4px 15px rgba(99,102,241,0.3);
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .chat-bot {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 12px;
    }
    .chat-bot .bubble {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        color: #e8e8f0;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 18px;
        max-width: 75%;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Source badge */
    .source-badge {
        display: inline-block;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.4);
        color: #a5b4fc;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
        margin-top: 6px;
    }
    .conf-badge {
        display: inline-block;
        background: rgba(16,185,129,0.15);
        border: 1px solid rgba(16,185,129,0.4);
        color: #6ee7b7;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
        margin-left: 6px;
        margin-top: 6px;
    }

    /* Status card */
    .status-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 16px;
    }

    /* Upload area */
    .stFileUploader > div {
        background: rgba(255,255,255,0.04);
        border: 2px dashed rgba(99,102,241,0.5);
        border-radius: 12px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: #fff;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99,102,241,0.45);
    }

    /* Input */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 12px;
        color: #e8e8f0;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []  # List[{role, content, source, confidence}]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def api_status():
    try:
        r = requests.get(f"{API_BASE}/status", timeout=4)
        return r.json() if r.ok else None
    except Exception:
        return None


def api_upload(file_bytes: bytes, filename: str) -> dict:
    r = requests.post(
        f"{API_BASE}/upload",
        files={"file": (filename, file_bytes, "application/octet-stream")},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def api_ask(question: str) -> dict:
    r = requests.post(
        f"{API_BASE}/ask",
        json={"question": question},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def api_reset() -> dict:
    r = requests.delete(f"{API_BASE}/reset", timeout=10)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🤖 AI Support Assistant")
    st.markdown("*Powered by RAG + ChromaDB*")
    st.divider()

    # Status
    status = api_status()
    if status:
        st.markdown(
            f"""
            <div class="status-card">
              <b>🟢 Backend Online</b><br>
              <small>{status['document_chunks']} chunks indexed</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="status-card">
              <b>🔴 Backend Offline</b><br>
              <small>Start: <code>uvicorn app:app --reload</code></small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF, TXT, or Markdown",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("⬆️ Upload & Index", use_container_width=True):
        if not uploaded_files:
            st.warning("Please select files first.")
        else:
            for uf in uploaded_files:
                with st.spinner(f"Indexing {uf.name}…"):
                    try:
                        res = api_upload(uf.read(), uf.name)
                        st.success(
                            f"✅ **{uf.name}** → {res['chunks_added']} chunks"
                        )
                    except Exception as e:
                        st.error(f"❌ {uf.name}: {e}")
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear All Documents", use_container_width=True):
        try:
            api_reset()
            st.session_state.messages = []
            st.success("Vector store cleared.")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown(
        """
        **How to use:**
        1. Start the FastAPI server
        2. Upload your support documents
        3. Ask questions below!
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main chat view
# ---------------------------------------------------------------------------

st.markdown(
    """
    <h1 style="
        text-align:center;
        background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 4px;
    ">
        AI Customer Support Assistant
    </h1>
    <p style="text-align:center; color:#9ca3af; margin-bottom: 24px; font-size:0.95rem;">
        Ask anything about your uploaded documents
    </p>
    """,
    unsafe_allow_html=True,
)

# Chat history
chat_container = st.container()
with chat_container:
    if not st.session_state.messages:
        st.markdown(
            """
            <div style="text-align:center; padding: 60px 20px; color:#6b7280;">
                <div style="font-size:3rem;">💬</div>
                <p>No conversation yet.<br>Upload documents and ask your first question!</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user"><div class="bubble">🧑 {msg["content"]}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                badges = ""
                if msg.get("source") and msg["source"] != "N/A":
                    badges += f'<span class="source-badge">📄 {msg["source"]}</span>'
                if msg.get("confidence"):
                    pct = int(float(msg["confidence"]) * 100)
                    badges += f'<span class="conf-badge">🎯 {pct}% confidence</span>'

                st.markdown(
                    f"""
                    <div class="chat-bot">
                      <div class="bubble">
                        🤖 {msg["content"]}
                        <br>{badges}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# Question input
st.markdown("<br>", unsafe_allow_html=True)
with st.form("question_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        question = st.text_input(
            "Ask a question",
            placeholder="e.g. What is the refund policy?",
            label_visibility="collapsed",
        )
    with col2:
        submitted = st.form_submit_button("Send ➤", use_container_width=True)

if submitted and question.strip():
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Thinking…"):
        try:
            result = api_ask(question)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "source": result.get("source", "N/A"),
                    "confidence": result.get("confidence", 0.0),
                }
            )
        except Exception as e:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"⚠️ Error: {e}",
                    "source": "N/A",
                    "confidence": 0.0,
                }
            )
    st.rerun()
