import streamlit as st
import os
import requests

# Page config MUST be the first command
st.set_page_config(
    page_title="Multimodel Dengue Report RAG Assistant",
    page_icon="🩺",
    layout="wide"
)

import importlib
import rag_pipeline
import ingest
importlib.reload(rag_pipeline)
importlib.reload(ingest)

from rag_pipeline import generate_answer, load_vectorstore
from ingest import init_directories, clean_directories, ingest_documents

init_directories()

# ── Sky Blue Healthcare Theme CSS ────────────────────────────────────────────
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Entire App Background: #F8FCFF */
    .stApp {
        background-color: #F8FCFF !important;
        color: #0F172A !important;
    }

    /* Centered Fixed-Width Container */
    .block-container {
        max-width: 1280px !important;
        margin: 0 auto !important;
        padding-top: 1.5rem !important;
        padding-bottom: 5.5rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }

    /* Constrain bottom chat input to match exact container width */
    div[data-testid="stBottom"] > div {
        max-width: 1280px !important;
        margin: 0 auto !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }

    /* ── Header Banner (Blue Gradient) ── */
    .sky-header-banner {
        background: linear-gradient(135deg, #0284C7 0%, #0EA5E9 45%, #38BDF8 100%);
        border-radius: 18px;
        padding: 1.5rem 2rem;
        color: #FFFFFF;
        box-shadow: 0 8px 24px -4px rgba(14, 165, 233, 0.25), 0 2px 6px -1px rgba(15, 23, 42, 0.06);
        margin-bottom: 1.25rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.25);
    }
    .header-title-text {
        font-size: 1.95rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0;
        letter-spacing: -0.025em;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.6rem;
    }
    .header-subtitle-text {
        font-size: 0.95rem;
        color: #E0F2FE;
        margin: 0.35rem 0 0 0;
        font-weight: 500;
        opacity: 0.95;
        letter-spacing: 0.01em;
    }

    /* ── Top Modern Cards (Equal Height, Rounded, Border #BAE6FD, Soft Shadow) ── */
    div[data-testid="stHorizontalBlock"] {
        align-items: stretch !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
        display: flex !important;
        flex-direction: column !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border-radius: 16px !important;
        border: 1px solid #BAE6FD !important;
        box-shadow: 0 4px 16px -2px rgba(14, 165, 233, 0.08) !important;
        padding: 0.75rem 0.85rem !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        display: flex !important;
        flex-direction: column !important;
        flex: 1 1 100% !important;
        height: 100% !important;
        min-height: 205px !important;
        box-sizing: border-box !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] > div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 8px 24px -4px rgba(14, 165, 233, 0.16) !important;
        transform: translateY(-2px) !important;
        border-color: #38BDF8 !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] > div[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"] {
        display: flex !important;
        flex-direction: column !important;
        flex: 1 1 100% !important;
        height: 100% !important;
        justify-content: flex-start !important;
    }

    /* Column Headers */
    .col-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0369A1;
        margin-bottom: 0.65rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* ── Pill-Style Status Badges ── */
    .pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.82rem;
        font-weight: 700;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        white-space: nowrap;
        margin-top: 0.3rem;
        transition: all 0.15s ease-in-out;
    }
    .pill-green {
        background-color: #DCFCE7 !important;
        color: #15803D !important;
        border: 1px solid #86EFAC !important;
        box-shadow: 0 2px 6px rgba(34, 197, 94, 0.12) !important;
    }
    .pill-blue {
        background-color: #E0F2FE !important;
        color: #0369A1 !important;
        border: 1px solid #BAE6FD !important;
        box-shadow: 0 2px 6px rgba(14, 165, 233, 0.12) !important;
    }
    .pill-yellow {
        background-color: #FEF3C7 !important;
        color: #92400E !important;
        border: 1px solid #FDE68A !important;
    }
    .pill-red {
        background-color: #FEE2E2 !important;
        color: #991B1B !important;
        border: 1px solid #FECACA !important;
    }
    .pill-file {
        background-color: #F0F9FF !important;
        color: #0284C7 !important;
        border: 1px solid #BAE6FD !important;
        box-shadow: 0 2px 6px rgba(14, 165, 233, 0.08) !important;
        font-size: 0.78rem !important;
    }
    .status-caption {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 0.35rem;
        font-weight: 500;
    }

    /* ── Selectbox Styling ── */
    div[data-testid="stSelectbox"] > div {
        border-radius: 10px !important;
        border-color: #BAE6FD !important;
    }
    div[data-testid="stSelectbox"] > div:hover {
        border-color: #0EA5E9 !important;
    }

    /* ── Upload Area (Dotted Sky Blue Border, Compact Height) ── */
    div[data-testid="stFileUploader"] {
        margin-bottom: 0.35rem !important;
    }
    div[data-testid="stFileUploader"] section {
        background-color: #F0F9FF !important;
        border: 1.5px dashed #38BDF8 !important;
        border-radius: 12px !important;
        padding: 0.45rem 0.75rem !important;
        transition: all 0.2s ease-in-out !important;
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: #0EA5E9 !important;
        background-color: #E0F2FE !important;
    }
    div[data-testid="stFileUploader"] section [data-testid="stMarkdownContainer"] p {
        color: #0369A1 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        margin: 0 !important;
    }
    div[data-testid="stFileUploader"] section small {
        display: none !important;
    }
    div[data-testid="stFileUploader"] button {
        background-color: #FFFFFF !important;
        border: 1px solid #BAE6FD !important;
        color: #0284C7 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }

    /* ── Process Documents Button (Sky Blue Gradient) ── */
    .stButton > button {
        background: linear-gradient(135deg, #2563EB 0%, #0EA5E9 60%, #38BDF8 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        padding: 0.5rem 1.25rem !important;
        box-shadow: 0 4px 14px rgba(14, 165, 233, 0.3) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        margin-top: 0.35rem !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #0284C7 60%, #0EA5E9 100%) !important;
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.45) !important;
        transform: translateY(-2px) !important;
        color: #FFFFFF !important;
    }

    /* Horizontal Divider */
    hr {
        margin: 1.25rem 0 !important;
        border-color: #BAE6FD !important;
    }

    /* ── Chat Messages ── */
    /* User Message Bubble */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        flex-direction: row-reverse !important;
        text-align: right !important;
        background: transparent !important;
        padding: 0.4rem 0 !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, #1E40AF 0%, #2563EB 60%, #0EA5E9 100%) !important;
        color: #FFFFFF !important;
        border-radius: 18px 18px 4px 18px !important;
        padding: 0.8rem 1.3rem !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.22) !important;
        max-width: 80% !important;
        margin-left: auto !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] p {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        margin: 0 !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="chatAvatarIcon-user"] {
        background: #0284C7 !important;
        color: #FFFFFF !important;
    }

    /* Assistant Message Card */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: transparent !important;
        padding: 0.5rem 0 !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
        background: #FFFFFF !important;
        border: 1px solid #BAE6FD !important;
        border-radius: 16px !important;
        padding: 1.25rem 1.5rem !important;
        box-shadow: 0 4px 16px rgba(14, 165, 233, 0.06) !important;
        max-width: 96% !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="chatAvatarIcon-assistant"] {
        background: #E0F2FE !important;
        color: #0284C7 !important;
    }

    /* Status Expander Box (Assessment Complete) */
    div[data-testid="stStatusWidget"] {
        border-radius: 14px !important;
        border: 1px solid #BAE6FD !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.04) !important;
    }

    /* ── Retrieved Patient Card ── */
    .retrieved-patient-box {
        background-color: #E0F2FE !important;
        border: 1px solid #BAE6FD !important;
        color: #0369A1 !important;
        border-radius: 12px !important;
        padding: 0.65rem 1rem !important;
        font-weight: 700 !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.75rem !important;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.08) !important;
        font-size: 0.9rem !important;
    }

    /* ── Answer Card with Subtle Fade-In Animation ── */
    @keyframes fadeInAnswer {
        from {
            opacity: 0;
            transform: translateY(6px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .answer-card-box {
        background-color: #F0F9FF !important;
        border-left: 5px solid #0EA5E9 !important;
        border-radius: 14px !important;
        padding: 1.15rem 1.4rem !important;
        margin-top: 0.5rem !important;
        box-shadow: 0 4px 14px rgba(14, 165, 233, 0.06) !important;
        color: #0F172A !important;
        font-size: 0.95rem !important;
        line-height: 1.65 !important;
        white-space: pre-wrap !important;
        animation: fadeInAnswer 0.35s ease-out forwards;
    }

    /* ── Chat Question Box (Rounded, Blue border & button) ── */
    div[data-testid="stChatInput"] {
        background: #FFFFFF !important;
        border: 2px solid #BAE6FD !important;
        border-radius: 18px !important;
        padding: 0.35rem 0.65rem !important;
        box-shadow: 0 6px 24px rgba(14, 165, 233, 0.12) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div[data-testid="stChatInput"]:focus-within {
        border-color: #0EA5E9 !important;
        box-shadow: 0 6px 28px rgba(14, 165, 233, 0.25) !important;
    }
    div[data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #2563EB 0%, #0EA5E9 100%) !important;
        color: #FFFFFF !important;
        border-radius: 50% !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.3) !important;
        transition: transform 0.15s ease-in-out !important;
    }
    div[data-testid="stChatInput"] button:hover {
        transform: scale(1.08) !important;
    }
    div[data-testid="stChatInput"] button svg {
        fill: #FFFFFF !important;
    }

    /* ── Small Footer ── */
    .app-footer {
        text-align: center;
        color: #64748B;
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #E2E8F0;
    }

    /* Responsive */
    @media (max-width: 1024px) {
        .block-container {
            max-width: 95% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        div[data-testid="stBottom"] > div {
            max-width: 95% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ── Header Banner (Blue Gradient) ────────────────────────────────────────────
st.markdown("""
<div class="sky-header-banner">
    <h1 class="header-title-text">🩺 Multimodel Dengue Report RAG Assistant</h1>
    <p class="header-subtitle-text">AI-Powered Clinical Decision Support System</p>
</div>
""", unsafe_allow_html=True)

# ── Horizontal Top Row: 4 Modern Cards (Exact Structure, Equal Heights) ──────
col1, col2, col3, col4 = st.columns([1.1, 1.8, 1.1, 1.1], gap="medium")

# Column 1: Configuration
with col1:
    with st.container(border=True):
        st.markdown('<div class="col-header">⚙️ Configuration</div>', unsafe_allow_html=True)
        selected_model = st.selectbox(
            "Select LLM Model",
            ["tinyllama", "phi3", "gemma2:2b", "llama3"],
            index=0,
            label_visibility="collapsed"
        )
        st.markdown(
            f'<span class="pill-badge pill-blue">🤖 Model: {selected_model}</span>',
            unsafe_allow_html=True
        )

# Column 2: Document Upload
with col2:
    with st.container(border=True):
        st.markdown('<div class="col-header">📄 Document Upload</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload PDF Reports",
            type="pdf",
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        process_clicked = st.button("🚀 Process Documents", use_container_width=True)

# Column 3: Index Status
with col3:
    with st.container(border=True):
        st.markdown('<div class="col-header">🗄️ Index Status</div>', unsafe_allow_html=True)
        if load_vectorstore() is not None:
            st.markdown('<span class="pill-badge pill-green">🟢 FAISS Ready</span>', unsafe_allow_html=True)
            st.markdown('<div class="status-caption">Vector index active</div>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill-badge pill-red">🔴 FAISS Not Loaded</span>', unsafe_allow_html=True)
            st.markdown('<div class="status-caption">Upload & process report</div>', unsafe_allow_html=True)

# Column 4: Ollama Status
with col4:
    with st.container(border=True):
        st.markdown('<div class="col-header">🤖 Ollama Status</div>', unsafe_allow_html=True)
        ollama_running = False
        available_models = []
        try:
            res = requests.get("http://127.0.0.1:11434/api/tags", timeout=1.5)
            if res.status_code == 200:
                ollama_running = True
                available_models = [m['name'] for m in res.json().get('models', [])]
        except Exception:
            ollama_running = False

        if ollama_running:
            if any(m.startswith(selected_model) for m in available_models):
                st.markdown('<span class="pill-badge pill-green">🟢 Ollama Running</span>', unsafe_allow_html=True)
                st.markdown(f'<div class="status-caption">{selected_model} ready</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="pill-badge pill-yellow">🟡 {selected_model} Missing</span>', unsafe_allow_html=True)
                st.markdown(f'<div class="status-caption">Run: ollama pull {selected_model}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill-badge pill-red">🔴 Ollama Offline</span>', unsafe_allow_html=True)
            st.markdown('<div class="status-caption">Run: ollama serve</div>', unsafe_allow_html=True)

# Divider
st.markdown("---")

# ── Document Ingestion Processing ────────────────────────────────────────────
if process_clicked:
    if uploaded_files:
        with st.spinner("Clearing old data and ingesting new files..."):
            st.cache_resource.clear()
            clean_directories()

            for uploaded_file in uploaded_files:
                with open(os.path.join("reports", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())

            from ingest import ingest_documents
            success, indexed_patients = ingest_documents()

            if success:
                st.cache_resource.clear()
                st.toast("✅ Documents Processed Successfully", icon="🟢")
                st.success("✅ Documents Processed Successfully")
            else:
                st.error("Failed to ingest documents.")
    else:
        st.warning("Please upload a PDF report first.")

# ── Session State Management (Latest Q&A Only) ──────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Input query - fixed at bottom with rounded styling and improved placeholder
if prompt := st.chat_input("Ask about diagnosis, platelet count, risk level, recommendations, or patient details..."):
    # Keep only the latest prompt
    st.session_state.messages = [{"role": "user", "content": prompt}]

# ── Chat Area Rendering ──────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    # Render messages (only latest pair)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    # Trigger generation when user enters a prompt
    if len(st.session_state.messages) == 1 and st.session_state.messages[-1]["role"] == "user":
        prompt = st.session_state.messages[-1]["content"]

        with st.chat_message("assistant"):
            with st.status("Analyzing clinical records...", expanded=False) as status:
                answer, retrieved_patient_ui = generate_answer(prompt, model_name=selected_model)
                status.update(label="Assessment Complete ✅", state="complete")

            # Show Retrieved Patient Card
            if retrieved_patient_ui:
                patient_card_html = f'<div class="retrieved-patient-box">📋 {retrieved_patient_ui}</div>'
                st.markdown(patient_card_html, unsafe_allow_html=True)

            # Show Answer Card with subtle fade-in animation
            answer_card_html = f'<div class="answer-card-box">{answer}</div>'
            st.markdown(answer_card_html, unsafe_allow_html=True)

            # Save latest state
            final_output = f"{retrieved_patient_ui}\n\n{answer}" if retrieved_patient_ui else answer
            st.session_state.messages.append({"role": "assistant", "content": final_output})

# ── Small Footer ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Dengue Clinical Intelligence Assistant v1.0 • Powered by FAISS + Ollama
</div>
""", unsafe_allow_html=True)
