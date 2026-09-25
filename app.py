import streamlit as st
import os
import re
import requests

# Page config MUST be the first command
st.set_page_config(
    page_title="Multimodal Dengue Report RAG Assistant",
    page_icon="🩺",
    layout="wide"
)

import importlib
import rag_pipeline
import ingest
importlib.reload(rag_pipeline)
importlib.reload(ingest)

from rag_pipeline import generate_answer, load_vectorstore, get_active_report_meta
from ingest import init_directories, clean_directories, ingest_documents

init_directories()

# ── Modern Healthcare AI Application Theme (Sky Blue + Mint Green) ───────────
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Entire App Background: Light Background #F8FCFF */
    .stApp {
        background-color: #F8FCFF !important;
        color: #0F172A !important;
    }

    /* Centered Fixed-Width Container */
    .block-container {
        max-width: 1280px !important;
        margin: 0 auto !important;
        padding-top: 1.25rem !important;
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

    /* ── Header Banner (Blue → Sky Blue → Mint Green Gradient) ── */
    .sky-header-banner {
        background: linear-gradient(135deg, #0EA5E9 0%, #38BDF8 50%, #4ADE80 100%) !important;
        border-radius: 20px !important;
        padding: 1.6rem 2.2rem !important;
        color: #FFFFFF !important;
        box-shadow: 0 10px 28px -4px rgba(14, 165, 233, 0.28), 0 4px 12px -2px rgba(74, 222, 128, 0.18) !important;
        margin-bottom: 1.25rem !important;
        text-align: center !important;
        border: 1px solid rgba(255, 255, 255, 0.35) !important;
    }
    .header-title-text {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        margin: 0 !important;
        letter-spacing: -0.025em !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 0.6rem !important;
    }
    .header-subtitle-text {
        font-size: 0.98rem !important;
        color: #F0FDF4 !important;
        margin: 0.35rem 0 0 0 !important;
        font-weight: 600 !important;
        opacity: 0.95 !important;
        letter-spacing: 0.01em !important;
    }

    /* ── Top Modern Cards (Equal Height, Rounded 18px-20px, Soft Shadows) ── */
    div[data-testid="stHorizontalBlock"] {
        align-items: stretch !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
        display: flex !important;
        flex-direction: column !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border-radius: 18px !important;
        border: 1.5px solid #BAE6FD !important;
        box-shadow: 0 4px 18px -2px rgba(14, 165, 233, 0.08) !important;
        padding: 0.85rem 1rem !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        display: flex !important;
        flex-direction: column !important;
        flex: 1 1 100% !important;
        height: 100% !important;
        min-height: 205px !important;
        box-sizing: border-box !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] > div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 8px 26px -2px rgba(14, 165, 233, 0.16) !important;
        transform: translateY(-3px) !important;
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
        font-size: 1.02rem;
        font-weight: 700;
        color: #0369A1;
        margin-bottom: 0.65rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* ── Pill-Style Status Badges (Clean, Modern Healthcare) ── */
    .pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.84rem;
        font-weight: 700;
        padding: 0.38rem 0.85rem;
        border-radius: 9999px;
        white-space: nowrap;
        margin-top: 0.35rem;
        transition: all 0.2s ease-in-out;
    }
    .pill-green {
        background-color: #ECFDF5 !important;
        color: #15803D !important;
        border: 1.5px solid #86EFAC !important;
        box-shadow: 0 2px 8px rgba(74, 222, 128, 0.18) !important;
    }
    .pill-model, .pill-blue {
        background-color: #F0F9FF !important;
        color: #0369A1 !important;
        border: 1.5px solid #BAE6FD !important;
        box-shadow: 0 2px 8px rgba(56, 189, 248, 0.15) !important;
    }
    .pill-yellow {
        background-color: #FEF3C7 !important;
        color: #92400E !important;
        border: 1.5px solid #FDE68A !important;
    }
    .pill-red {
        background-color: #FEE2E2 !important;
        color: #991B1B !important;
        border: 1.5px solid #FECACA !important;
    }
    .pill-file {
        background-color: #F0F9FF !important;
        color: #0284C7 !important;
        border: 1px solid #BAE6FD !important;
        box-shadow: 0 2px 6px rgba(14, 165, 233, 0.08) !important;
        font-size: 0.8rem !important;
    }
    .status-caption {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 0.35rem;
        font-weight: 500;
    }

    /* ── Small Patient Badge ── */
    .patient-pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: #FFFFFF;
        border: 1.5px solid #BAE6FD;
        border-radius: 9999px;
        padding: 0.45rem 1.15rem;
        font-size: 0.88rem;
        font-weight: 700;
        color: #0369A1;
        box-shadow: 0 4px 14px rgba(14, 165, 233, 0.08);
        margin-bottom: 0.75rem;
        transition: all 0.25s ease;
    }
    .patient-pill-badge:hover {
        border-color: #38BDF8;
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(14, 165, 233, 0.14);
    }

    /* ── Upload Area (Larger rounded area, gradient border, soft hover) ── */
    div[data-testid="stFileUploader"] {
        margin-bottom: 0.35rem !important;
    }
    div[data-testid="stFileUploader"] section {
        background-color: #F8FCFF !important;
        border: 2px dashed #7DD3FC !important;
        border-radius: 18px !important;
        padding: 0.65rem 0.9rem !important;
        transition: all 0.25s ease-in-out !important;
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: #38BDF8 !important;
        background-color: #F0F9FF !important;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.22) !important;
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
        border-radius: 12px !important;
        font-weight: 600 !important;
    }

    /* Process Documents Button */
    .stButton > button {
        background: linear-gradient(135deg, #0EA5E9 0%, #38BDF8 60%, #4ADE80 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 14px !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        padding: 0.55rem 1.3rem !important;
        box-shadow: 0 4px 16px rgba(14, 165, 233, 0.25) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        margin-top: 0.35rem !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0284C7 0%, #0EA5E9 60%, #22C55E 100%) !important;
        box-shadow: 0 8px 24px rgba(14, 165, 233, 0.38) !important;
        transform: translateY(-2px) !important;
        color: #FFFFFF !important;
    }

    /* Uploaded File Delete 'X' Button */
    div[data-testid="stColumn"] div:has(> button[key*="remove_"]) button,
    button[key*="remove_"] {
        background-color: #FEE2E2 !important;
        color: #DC2626 !important;
        border: 1px solid #FECACA !important;
        border-radius: 10px !important;
        padding: 0.1rem 0.4rem !important;
        font-size: 0.85rem !important;
        font-weight: 800 !important;
        min-height: 28px !important;
        height: 28px !important;
        line-height: 1 !important;
        transition: all 0.15s ease !important;
    }
    div[data-testid="stColumn"] div:has(> button[key*="remove_"]) button:hover,
    button[key*="remove_"]:hover {
        background-color: #EF4444 !important;
        color: #FFFFFF !important;
        border-color: #DC2626 !important;
    }

    /* Selectbox */
    div[data-testid="stSelectbox"] > div {
        border-radius: 12px !important;
        border-color: #BAE6FD !important;
    }
    div[data-testid="stSelectbox"] > div:hover {
        border-color: #38BDF8 !important;
    }

    /* Horizontal Divider */
    hr {
        margin: 1.15rem 0 !important;
        border-color: #BAE6FD !important;
    }

    /* ── Smooth Answer Reveal Animation & Pulse Typing ── */
    @keyframes smoothAnswerReveal {
        0% {
            opacity: 0;
            transform: translateY(12px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }
    @keyframes pulseTyping {
        0%, 100% { opacity: 0.75; transform: scale(0.997); }
        50% { opacity: 1; transform: scale(1); }
    }

    /* Status Expander Box (Typing Indicator: Analyzing Patient Report...) */
    div[data-testid="stStatusWidget"] {
        border-radius: 18px !important;
        border: 1.5px solid #BAE6FD !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 4px 16px rgba(14, 165, 233, 0.06) !important;
        animation: pulseTyping 1.8s infinite ease-in-out !important;
        margin-bottom: 0.75rem !important;
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
        background: linear-gradient(135deg, #0EA5E9 0%, #2563EB 100%) !important;
        color: #FFFFFF !important;
        border-radius: 20px 20px 4px 20px !important;
        padding: 0.85rem 1.35rem !important;
        box-shadow: 0 4px 16px rgba(14, 165, 233, 0.22) !important;
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
        background: #0EA5E9 !important;
        color: #FFFFFF !important;
    }

    /* Assistant Message Container */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: transparent !important;
        padding: 0.5rem 0 !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
        background: transparent !important;
        padding: 0 !important;
        max-width: 100% !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="chatAvatarIcon-assistant"] {
        background: #E0F2FE !important;
        color: #0284C7 !important;
    }

    /* ── Retrieved Patient Box ── */
    .retrieved-patient-box {
        background-color: #FFFFFF !important;
        border: 1.5px solid #BAE6FD !important;
        color: #0369A1 !important;
        border-radius: 18px !important;
        padding: 0.75rem 1.25rem !important;
        font-weight: 700 !important;
        margin-top: 0.35rem !important;
        margin-bottom: 0.75rem !important;
        box-shadow: 0 4px 16px rgba(14, 165, 233, 0.08) !important;
        font-size: 0.92rem !important;
        animation: smoothAnswerReveal 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards !important;
        transition: all 0.25s ease !important;
    }
    .retrieved-patient-box:hover {
        border-color: #38BDF8 !important;
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.12) !important;
        transform: translateY(-2px) !important;
    }

    /* ── Answer Section & Cards (White Background, Light Blue Border, Rounded Corners 18px-20px, Soft Shadows) ── */
    .answer-card-box {
        background-color: #FFFFFF !important;
        border: 1.5px solid #BAE6FD !important;
        border-left: 5px solid #38BDF8 !important;
        border-radius: 18px !important;
        padding: 1.25rem 1.6rem !important;
        margin-top: 0.5rem !important;
        box-shadow: 0 4px 18px rgba(14, 165, 233, 0.08) !important;
        color: #0F172A !important;
        font-size: 0.95rem !important;
        line-height: 1.65 !important;
        white-space: pre-wrap !important;
        animation: smoothAnswerReveal 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards !important;
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease !important;
    }
    .answer-card-box:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 26px rgba(14, 165, 233, 0.14) !important;
        border-color: #38BDF8 !important;
    }

    /* Structured Section Cards */
    .section-card {
        background-color: #FFFFFF !important;
        border: 1.5px solid #BAE6FD !important;
        border-radius: 18px !important;
        padding: 1.15rem 1.5rem !important;
        margin-bottom: 0.85rem !important;
        box-shadow: 0 4px 16px rgba(14, 165, 233, 0.07) !important;
        transition: all 0.25s ease !important;
        animation: smoothAnswerReveal 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards !important;
    }
    .section-card:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(14, 165, 233, 0.14) !important;
        border-color: #38BDF8 !important;
    }
    .section-card-title {
        font-size: 1rem !important;
        font-weight: 700 !important;
        color: #0369A1 !important;
        margin-bottom: 0.45rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.45rem !important;
    }
    .section-card-body {
        font-size: 0.94rem !important;
        color: #1E293B !important;
        line-height: 1.65 !important;
        white-space: pre-wrap !important;
    }

    /* ── Clinical Evidence Card (Clean White, Rounded 18px, Light Blue Border) ── */
    .clinical-evidence-card {
        background-color: #FFFFFF !important;
        border: 1.5px solid #BAE6FD !important;
        border-radius: 18px !important;
        box-shadow: 0 4px 18px -2px rgba(14, 165, 233, 0.08) !important;
        padding: 1.35rem 1.7rem !important;
        color: #0F172A !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        transition: all 0.25s ease !important;
        animation: smoothAnswerReveal 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards !important;
    }
    .clinical-evidence-card:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px -2px rgba(14, 165, 233, 0.14) !important;
        border-color: #38BDF8 !important;
    }
    .evidence-header-label {
        font-size: 0.96rem !important;
        font-weight: 700 !important;
        color: #0369A1 !important;
        margin-bottom: 0.35rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.4rem !important;
    }
    .evidence-patient-name {
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        color: #0F172A !important;
        padding-left: 0.15rem !important;
    }
    .evidence-val-text {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #1E293B !important;
        padding-left: 0.15rem !important;
    }
    .evidence-bullet-list {
        list-style: none !important;
        padding-left: 0.15rem !important;
        margin: 0.25rem 0 0 0 !important;
    }
    .evidence-bullet-list li {
        font-size: 0.92rem !important;
        color: #334155 !important;
        line-height: 1.65 !important;
        position: relative !important;
        padding-left: 1.2rem !important;
    }
    .evidence-bullet-list li::before {
        content: "•" !important;
        color: #38BDF8 !important;
        font-weight: 900 !important;
        font-size: 1.25rem !important;
        position: absolute !important;
        left: 0.1rem !important;
        top: -0.15rem !important;
    }

    /* ── Question Input (Rounded 20px, Blue glow on focus, Modern send button) ── */
    div[data-testid="stChatInput"] {
        background: #FFFFFF !important;
        border: 2px solid #BAE6FD !important;
        border-radius: 20px !important;
        padding: 0.4rem 0.75rem !important;
        box-shadow: 0 4px 18px rgba(14, 165, 233, 0.08) !important;
        transition: all 0.25s ease-in-out !important;
    }
    div[data-testid="stChatInput"]:focus-within {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 0 4px rgba(56, 189, 248, 0.25), 0 8px 26px rgba(14, 165, 233, 0.14) !important;
    }
    div[data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #0EA5E9 0%, #38BDF8 60%, #4ADE80 100%) !important;
        color: #FFFFFF !important;
        border-radius: 50% !important;
        border: none !important;
        box-shadow: 0 3px 10px rgba(14, 165, 233, 0.28) !important;
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease !important;
    }
    div[data-testid="stChatInput"] button:hover {
        transform: scale(1.1) !important;
        box-shadow: 0 4px 14px rgba(14, 165, 233, 0.4) !important;
    }
    div[data-testid="stChatInput"] button svg {
        fill: #FFFFFF !important;
    }

    /* ── Tiny Footer ── */
    .app-footer {
        text-align: center;
        color: #64748B;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 2.5rem;
        padding-top: 1.25rem;
        border-top: 1px solid #E2E8F0;
        letter-spacing: 0.02em;
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

# ── Header Section (Gradient Banner: Blue → Sky Blue → Mint Green) ────────────
st.markdown("""
<div class="sky-header-banner">
    <h1 class="header-title-text">🩺 Multimodal Dengue Report RAG Assistant</h1>
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
            f'<span class="pill-badge pill-model">Model Selected ✅ <span style="font-weight:600; opacity:0.85;">({selected_model})</span></span>',
            unsafe_allow_html=True
        )

# Column 2: Document Upload
with col2:
    with st.container(border=True):
        st.markdown('<div class="col-header">📄 Document Upload</div>', unsafe_allow_html=True)
        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = 0

        uploaded_files = st.file_uploader(
            "Upload Medical Reports (PDF, TXT, DOCX)",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"uploader_{st.session_state.uploader_key}"
        )

        # Show visible file card with 'X' (remove) button next to each uploaded report
        if uploaded_files:
            for idx, file in enumerate(uploaded_files):
                size_kb = len(file.getvalue()) / 1024
                size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
                col_finfo, col_fdel = st.columns([0.80, 0.20])
                with col_finfo:
                    st.markdown(
                        f'<div class="pill-badge pill-file" style="width:100%; overflow:hidden; text-overflow:ellipsis;" title="{file.name}">'
                        f'📄 <b>{file.name}</b> <span style="opacity:0.75;">({size_str})</span></div>',
                        unsafe_allow_html=True
                    )
                with col_fdel:
                    if st.button("✕", key=f"remove_uploaded_{idx}_{file.name[:8]}", help=f"Remove {file.name}", use_container_width=True):
                        st.session_state.uploader_key += 1
                        clean_directories()
                        st.cache_resource.clear()
                        st.session_state.messages = []
                        st.rerun()
        else:
            # If files were previously ingested and reside in reports/
            active_files = [f for f in os.listdir("reports") if os.path.isfile(os.path.join("reports", f)) and not f.endswith(".json")] if os.path.exists("reports") else []
            if active_files:
                for idx, fname in enumerate(active_files):
                    fpath = os.path.join("reports", fname)
                    fsize = os.path.getsize(fpath) / 1024
                    size_str = f"{fsize:.1f} KB" if fsize < 1024 else f"{fsize/1024:.1f} MB"
                    col_finfo, col_fdel = st.columns([0.80, 0.20])
                    with col_finfo:
                        st.markdown(
                            f'<div class="pill-badge pill-file" style="width:100%; overflow:hidden; text-overflow:ellipsis;" title="{fname}">'
                            f'📄 <b>{fname}</b> <span style="opacity:0.75;">({size_str})</span></div>',
                            unsafe_allow_html=True
                        )
                    with col_fdel:
                        if st.button("✕", key=f"remove_active_{idx}_{fname[:8]}", help=f"Remove {fname}", use_container_width=True):
                            clean_directories()
                            st.cache_resource.clear()
                            st.session_state.messages = []
                            st.rerun()

        process_clicked = st.button("🚀 Process Documents", use_container_width=True)

# Column 3: Index Status
with col3:
    with st.container(border=True):
        st.markdown('<div class="col-header">🗄️ Index Status</div>', unsafe_allow_html=True)
        if load_vectorstore() is not None:
            st.markdown('<span class="pill-badge pill-green">FAISS Ready ✅</span>', unsafe_allow_html=True)
            st.markdown('<div class="status-caption">Active report index ready</div>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill-badge pill-red">FAISS Not Loaded ⏳</span>', unsafe_allow_html=True)
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
                st.markdown('<span class="pill-badge pill-green">Ollama Running ✅</span>', unsafe_allow_html=True)
                st.markdown(f'<div class="status-caption">{selected_model} ready</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="pill-badge pill-yellow">Ollama Running (Model Missing) ⚠️</span>', unsafe_allow_html=True)
                st.markdown(f'<div class="status-caption">Run: ollama pull {selected_model}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill-badge pill-red">Ollama Offline ❌</span>', unsafe_allow_html=True)
            st.markdown('<div class="status-caption">Run: ollama serve</div>', unsafe_allow_html=True)

# Divider
st.markdown("---")

# ── Small Patient Badge (Current Patient: Rahul (D001) or Active Patient) ────
active_meta = get_active_report_meta()
current_patient_name = active_meta.get("patient_name")
current_patient_id = active_meta.get("patient_id")

if not current_patient_name or current_patient_name in ["Not specified", "Extracted", "Unknown"]:
    current_patient_name = "Rahul"
if not current_patient_id or current_patient_id in ["Not specified", "Extracted", "Unknown"]:
    current_patient_id = "D001"

st.markdown(
    f'<div style="margin-bottom: 0.65rem;">'
    f'<span class="patient-pill-badge">👤 Current Patient: <strong>{current_patient_name} ({current_patient_id})</strong></span>'
    f'</div>',
    unsafe_allow_html=True
)

# ── Document Ingestion Processing ────────────────────────────────────────────
if process_clicked:
    if uploaded_files:
        with st.spinner("Clearing previous session and ingesting new report..."):
            st.cache_resource.clear()
            clean_directories()
            st.session_state.messages = []

            for uploaded_file in uploaded_files:
                dest_path = os.path.join("reports", uploaded_file.name)
                with open(dest_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            from ingest import ingest_documents
            success, meta_info = ingest_documents()

            if success:
                st.cache_resource.clear()
                p_name = meta_info.get("patient_name", "Extracted")
                p_id = meta_info.get("patient_id", "Extracted")
                st.toast(f"✅ Ingested: {p_name} ({p_id})", icon="🟢")
                st.success(f"✅ Report processed successfully! Active: **{p_name}** ({p_id}) | Chunks: {meta_info.get('chunk_count', 0)}")
                st.rerun()
            else:
                st.error("Failed to ingest documents.")
    else:
        st.warning("Please upload a medical report (PDF, TXT, DOCX) first.")

# ── Session State Management (Latest Q&A Only) ──────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Input query - rounded input box, blue glow on focus, modern send button
if prompt := st.chat_input("Ask about diagnosis, platelet count, risk level, recommendations, or patient details..."):
    # Keep only the latest prompt
    st.session_state.messages = [{"role": "user", "content": prompt}]

# ── Answer Section Helper (Formats Clean White Cards with Light Blue Border) ─
def render_styled_answer_cards(answer_text: str) -> str:
    """
    Renders the Answer Section cards with:
    - White background
    - Light blue border
    - Rounded corners (18px-20px)
    - Soft shadows
    Preserves all section headers:
    ✅ Direct Answer, 📋 Patient Details, 🔬 Clinical Findings,
    ⚠️ Health Insights, 💡 Suggestions, 📌 Recommendations, 🏥 Follow-up Advice
    """
    known_section_patterns = [
        ("✅ Direct Answer", re.compile(r'^(?:✅\s*)?Direct\s*Answer[:\s\-]*$', re.I | re.M)),
        ("📋 Patient Details", re.compile(r'^(?:📋\s*)?Patient\s*Details[:\s\-]*$', re.I | re.M)),
        ("🔬 Clinical Findings", re.compile(r'^(?:🔬\s*)?Clinical\s*Findings[:\s\-]*$', re.I | re.M)),
        ("⚠️ Health Insights", re.compile(r'^(?:⚠️\s*)?Health\s*Insights[:\s\-]*$', re.I | re.M)),
        ("💡 Suggestions", re.compile(r'^(?:💡\s*)?Suggestions?[:\s\-]*$', re.I | re.M)),
        ("📌 Recommendations", re.compile(r'^(?:📌\s*)?Recommendations?[:\s\-]*$', re.I | re.M)),
        ("🏥 Follow-up Advice", re.compile(r'^(?:🏥\s*)?Follow-up\s*Advice[:\s\-]*$', re.I | re.M)),
    ]

    matches = []
    for title, rgx in known_section_patterns:
        for m in rgx.finditer(answer_text):
            matches.append((m.start(), m.end(), title))

    if not matches:
        # Standard answer card with clean white background and light blue border
        return f'<div class="answer-card-box">{answer_text}</div>'

    matches.sort(key=lambda x: x[0])
    cards_html = []

    # Check for preamble before first matched section
    if matches[0][0] > 0:
        preamble = answer_text[:matches[0][0]].strip()
        if preamble:
            cards_html.append(
                f'<div class="section-card"><div class="section-card-body">{preamble}</div></div>'
            )

    for i, (start, end, title) in enumerate(matches):
        next_start = matches[i + 1][0] if i + 1 < len(matches) else len(answer_text)
        body = answer_text[end:next_start].strip()
        cards_html.append(
            f'<div class="section-card">'
            f'<div class="section-card-title">{title}</div>'
            f'<div class="section-card-body">{body}</div>'
            f'</div>'
        )

    return "".join(cards_html)

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
            # Small animated typing indicator
            with st.status("🔍 Analyzing Patient Report...", expanded=False) as status:
                answer, retrieved_patient_ui, evidence_list = generate_answer(prompt, model_name=selected_model)
                status.update(label="Assessment Complete ✅", state="complete")

            # Show Retrieved Patient Card
            if retrieved_patient_ui:
                patient_card_html = f'<div class="retrieved-patient-box">📋 {retrieved_patient_ui}</div>'
                st.markdown(patient_card_html, unsafe_allow_html=True)

            # Show Answer Section in modern redesigned cards
            answer_cards_html = render_styled_answer_cards(answer)
            if hasattr(st, "html"):
                st.html(answer_cards_html)
            else:
                st.markdown(answer_cards_html, unsafe_allow_html=True)

            # Show Clean Clinical Evidence Card
            if evidence_list:
                if isinstance(evidence_list, dict):
                    ev_name = evidence_list.get("patient_name", "Not specified")
                    ev_findings = evidence_list.get("findings", [])
                    ev_diag = evidence_list.get("diagnosis", "Suspected Dengue Fever")
                    ev_recs = evidence_list.get("recommendations", [])
                else:
                    ev_name = "Not specified"
                    ev_findings = []
                    ev_diag = "Suspected Dengue Fever"
                    ev_recs = []

                findings_html = "".join([f"<li>{item}</li>" for item in ev_findings]) if ev_findings else "<li>No laboratory findings specified</li>"
                recs_html = "".join([f"<li>{item}</li>" for item in ev_recs]) if ev_recs else "<li>Follow standard clinical care guidance</li>"

                clinical_card_html = (
                    f'<div class="clinical-evidence-card">'
                    f'<div class="evidence-header-label">👤 Patient Name</div>'
                    f'<div class="evidence-patient-name">{ev_name}</div>'
                    f'<div class="evidence-header-label" style="margin-top: 1.1rem;">🩸 Clinical Findings</div>'
                    f'<ul class="evidence-bullet-list">{findings_html}</ul>'
                    f'<div class="evidence-header-label" style="margin-top: 1.1rem;">🩺 Diagnosis</div>'
                    f'<div class="evidence-val-text">{ev_diag}</div>'
                    f'<div class="evidence-header-label" style="margin-top: 1.1rem;">💊 Recommendation</div>'
                    f'<ul class="evidence-bullet-list">{recs_html}</ul>'
                    f'</div>'
                )
                with st.expander("📋 View Clinical Evidence", expanded=False):
                    if hasattr(st, "html"):
                        st.html(clinical_card_html)
                    else:
                        st.markdown(clinical_card_html, unsafe_allow_html=True)

            # Save latest state
            final_output = f"{retrieved_patient_ui}\n\n{answer}" if retrieved_patient_ui else answer
            st.session_state.messages.append({"role": "assistant", "content": final_output})

# ── Tiny Footer (Exact text as requested) ────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Powered by AWS Bedrock Knowledge Base + RAG
</div>
""", unsafe_allow_html=True)
