import streamlit as st
import os
import re
import io
import requests
from datetime import datetime

# Page config MUST be the first command
st.set_page_config(
    page_title="Multimodal Dengue Report RAG Assistant",
    page_icon="🩺",
    layout="wide"
)

import importlib
import rag_pipeline
import ingest
import utils.multilingual as multilingual
importlib.reload(rag_pipeline)
importlib.reload(ingest)
importlib.reload(multilingual)

from rag_pipeline import generate_answer, load_vectorstore, get_active_report_meta, clean_simple_direct_answer
from ingest import init_directories, clean_directories, ingest_documents
from utils.multilingual import (
    SUPPORTED_LANGUAGES,
    detect_language,
    translate_text,
    text_to_speech_audio,
    transcribe_audio_bytes,
)

init_directories()

# ── ReportLab PDF Export Utility (Dynamic Import with Fallback) ───────────────
def generate_answer_pdf(question: str, answer: str, patient_name: str) -> bytes:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=45,
            leftMargin=45,
            topMargin=45,
            bottomMargin=45
        )
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'DocTitle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=17,
            leading=21,
            textColor=colors.HexColor('#050814'),
            spaceAfter=4
        )
        subtitle_style = ParagraphStyle(
            'DocSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#65A30D'),
            spaceAfter=14
        )
        meta_label = ParagraphStyle(
            'MetaLabel',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=9,
            leading=13,
            textColor=colors.HexColor('#365314')
        )
        meta_val = ParagraphStyle(
            'MetaVal',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            leading=13,
            textColor=colors.HexColor('#111827')
        )
        section_head = ParagraphStyle(
            'SecHead',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=12,
            leading=16,
            textColor=colors.HexColor('#365314'),
            spaceBefore=12,
            spaceAfter=6
        )
        body_style = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9.5,
            leading=14.5,
            textColor=colors.HexColor('#111827'),
            spaceAfter=5
        )

        story = []
        story.append(Paragraph("🩺 Multimodal Dengue Report RAG Assistant", title_style))
        story.append(Paragraph("Clinical Decision Support System • Grounded Assessment Report", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#84CC16'), spaceAfter=12))

        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        meta_data = [
            [
                Paragraph("<b>Patient Name:</b>", meta_label), Paragraph(str(patient_name), meta_val),
                Paragraph("<b>Timestamp:</b>", meta_label), Paragraph(timestamp_str, meta_val)
            ],
            [
                Paragraph("<b>Platform:</b>", meta_label), Paragraph("AWS Bedrock Knowledge Base", meta_val),
                Paragraph("<b>Status:</b>", meta_label), Paragraph("Clinically Grounded Response", meta_val)
            ]
        ]
        t = Table(meta_data, colWidths=[80, 180, 80, 182])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F7FEE7')),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#D9F99D')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E8F9D7')),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(t)
        story.append(Spacer(1, 14))

        # Query Section
        safe_q = question.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        story.append(Paragraph("Clinical Query", section_head))
        story.append(Paragraph(f"<b>Q:</b> {safe_q}", body_style))
        story.append(Spacer(1, 10))

        # Answer Section
        story.append(Paragraph("Synthesized Grounded Response", section_head))
        for line in answer.split('\n'):
            line_clean = line.strip()
            if line_clean:
                safe_l = line_clean.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                safe_l = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', safe_l)
                story.append(Paragraph(safe_l, body_style))
            else:
                story.append(Spacer(1, 4))

        story.append(Spacer(1, 18))
        story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor('#CBD5E1'), spaceAfter=8))
        footer_text = Paragraph(
            '<font size=7 color="#64748B">CONFIDENTIAL CLINICAL RECORD • For Decision Support Only • Powered by AWS Bedrock Knowledge Base + RAG</font>',
            ParagraphStyle('Footer', parent=styles['Normal'], alignment=1)
        )
        story.append(footer_text)

        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes
    except Exception as e:
        # Fallback if ReportLab is not available
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        txt_content = (
            f"MULTIMODAL DENGUE REPORT RAG ASSISTANT\n"
            f"Clinical Decision Support Assessment Report\n"
            f"====================================================\n"
            f"Patient: {patient_name}\n"
            f"Timestamp: {timestamp_str}\n"
            f"System: AWS Bedrock Knowledge Base + RAG\n"
            f"====================================================\n\n"
            f"CLINICAL QUERY:\n{question}\n\n"
            f"SYNTHESIZED GROUNDED RESPONSE:\n{answer}\n\n"
            f"====================================================\n"
            f"CONFIDENTIAL MEDICAL INFORMATION - FOR CLINICAL DECISION SUPPORT ONLY\n"
        )
        return txt_content.encode("utf-8")

# ── Chain360 Inspired Theme (Vibrant Lime Green #74D116, Pitch Black #0D0D0D, Pure White #FFFFFF) ──
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500;600;700&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* ── 1. GLOBAL BACKGROUND: Soft gradient canvas — elegant, not overwhelming ── */
    html, body, #root, .stApp, [data-testid="stAppViewContainer"], .main,
    section[data-testid="stMain"], div.stMainBlockContainer {
        background: linear-gradient(145deg, #F4FDE8 0%, #E8F9D0 40%, #DCFCA7 100%) !important;
        color: #0D0D0D !important;
    }



    /* Transparent Streamlit Header */
    header[data-testid="stHeader"] {
        background: transparent !important;
        background-color: transparent !important;
    }
    header[data-testid="stHeader"] svg {
        color: #0D0D0D !important;
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


    /* ── 2. HEADER HERO BANNER — Deep Forest Green, matches UI flow ── */
    .saas-hero-banner {
        background: linear-gradient(135deg, #14532d 0%, #166534 50%, #15803d 100%) !important;
        border-radius: 22px !important;
        padding: 2.2rem 2.5rem !important;
        color: #FFFFFF !important;
        box-shadow: 0 12px 40px rgba(0,0,0,0.15), 0 0 0 1px rgba(116,209,22,0.3) !important;
        margin-bottom: 1.5rem !important;
        text-align: center !important;
        border: 2px solid rgba(116,209,22,0.5) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    .saas-hero-banner::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 50% 0%, rgba(116,209,22,0.2) 0%, transparent 55%);
        pointer-events: none;
    }
    .hero-tagline-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        background: rgba(255,255,255,0.12);
        border: 1.5px solid rgba(255,255,255,0.35);
        color: #D9F99D;
        padding: 0.38rem 1.1rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.85rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.15);
    }
    .hero-pulse-dot {
        width: 8px;
        height: 8px;
        background: #86EFAC;
        border-radius: 50%;
        box-shadow: 0 0 10px #86EFAC;
        display: inline-block;
        animation: heroPulse 1.8s infinite ease-in-out;
    }
    @keyframes heroPulse {
        0%, 100% { transform: scale(0.9); opacity: 0.75; }
        50% { transform: scale(1.35); opacity: 1; box-shadow: 0 0 16px #86EFAC; }
    }
    .hero-title-text {
        font-size: 2.35rem !important;
        font-weight: 900 !important;
        color: #FFFFFF !important;
        margin: 0 !important;
        letter-spacing: -0.03em !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 0.75rem !important;
        text-shadow: 0 2px 8px rgba(0,0,0,0.25) !important;
    }
    .neon-highlight {
        color: #BBF7D0 !important;
    }
    .hero-subtitle-text {
        font-size: 1.05rem !important;
        color: #BBF7D0 !important;
        margin: 0.55rem 0 0.85rem 0 !important;
        font-weight: 500 !important;
        opacity: 0.9 !important;
        letter-spacing: 0.015em !important;
    }
    .hero-chip-row {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.65rem;
        flex-wrap: wrap;
        margin-top: 0.85rem;
    }
    .hero-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(255,255,255,0.15);
        border: 1.5px solid rgba(255,255,255,0.3);
        color: #F0FDF4;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 0.32rem 0.85rem;
        border-radius: 9999px;
        backdrop-filter: blur(4px);
    }

    /* ── 3. CLEAN PANEL DESIGN — Equal size cards, no nested box feel ── */
    div[data-testid="stHorizontalBlock"] {
        align-items: stretch !important;
        gap: 1rem !important;
    }
    /* Equal-height columns */
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
        display: flex !important;
        flex-direction: column !important;
        flex: 1 1 0 !important;
        min-width: 0 !important;
    }

    /* TOP-LEVEL column cards: full height, equal, green top border */
    html body [data-testid="stAppViewContainer"] [data-testid="stColumn"] > div:first-child,
    html body [data-testid="stAppViewContainer"] div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF !important;
        border-radius: 18px !important;
        border: 2px solid #D9F99D !important;
        border-top: 4px solid #74D116 !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08) !important;
        padding: 1.1rem 1.2rem 1.2rem 1.2rem !important;
        transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
        display: flex !important;
        flex-direction: column !important;
        flex: 1 1 auto !important;
        box-sizing: border-box !important;
        width: 100% !important;
    }

    html body [data-testid="stAppViewContainer"] [data-testid="stColumn"] > div:first-child:hover,
    html body [data-testid="stAppViewContainer"] div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        background: #F7FEE7 !important;
        background-color: #F7FEE7 !important;
        border-color: #84CC16 !important;
        border-top-color: #4D9C00 !important;
        box-shadow: 0 10px 30px rgba(116,209,22,0.22), 0 2px 10px rgba(0,0,0,0.07) !important;
        transform: translateY(-2px) !important;
    }

    /* Flatten all inner nested containers — no box-in-box */
    html body [data-testid="stAppViewContainer"] div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlockBorderWrapper"],
    html body [data-testid="stAppViewContainer"] [data-testid="stColumn"] > div:first-child div[data-testid="stVerticalBlock"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        border-radius: 0 !important;
        flex: unset !important;
    }



    /* Column Headers */
    .col-header {
        font-size: 1.05rem !important;
        font-weight: 900 !important;
        color: #0D0D0D !important;
        margin-bottom: 0.75rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.45rem !important;
        letter-spacing: -0.01em !important;
    }

    /* ── 4. STATUS BADGES — Modern Light Mint & Lime Theme matching UI flow ── */
    .product-badge-stack {
        display: flex !important;
        flex-direction: column !important;
        gap: 0.5rem !important;
        margin-top: 0.35rem !important;
    }

    .product-badge {
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        background: linear-gradient(135deg, #F0FDF4 0%, #ECFCCB 100%) !important;
        background-color: #F7FEE7 !important;
        border: 1.5px solid #84CC16 !important;
        border-radius: 9999px !important;
        padding: 0.45rem 0.95rem !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        color: #14532D !important;
        box-shadow: 0 2px 8px rgba(132, 204, 22, 0.16), 0 1px 3px rgba(0, 0, 0, 0.04) !important;
        transition: all 0.2s ease-in-out !important;
    }

    .product-badge:hover {
        transform: translateY(-1.5px) !important;
        box-shadow: 0 4px 14px rgba(132, 204, 22, 0.3) !important;
        border-color: #65A30D !important;
    }

    .product-badge-label {
        color: #166534 !important;
        font-weight: 700 !important;
        font-size: 0.82rem !important;
        margin-left: 0.4rem !important;
        margin-right: auto !important;
    }

    .product-badge-state {
        color: #15803D !important;
        font-weight: 800 !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.01em !important;
    }

    .product-badge.badge-pending {
        background: linear-gradient(135deg, #F0FDF4 0%, #ECFCCB 100%) !important;
        background-color: #F7FEE7 !important;
        border-color: #84CC16 !important;
        box-shadow: 0 2px 8px rgba(132, 204, 22, 0.16), 0 1px 3px rgba(0, 0, 0, 0.04) !important;
    }
    .product-badge.badge-pending .product-badge-label {
        color: #166534 !important;
    }
    .product-badge.badge-pending .product-badge-state {
        color: #15803D !important;
    }

    .product-badge.badge-offline {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%) !important;
        background-color: #FEF2F2 !important;
        border-color: #F87171 !important;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.18), 0 1px 3px rgba(0, 0, 0, 0.04) !important;
    }
    .product-badge.badge-offline .product-badge-label {
        color: #991B1B !important;
    }
    .product-badge.badge-offline .product-badge-state {
        color: #DC2626 !important;
    }

    .badge-dot {
        width: 8px;
        height: 8px;
        background-color: #16A34A;
        border-radius: 50%;
        box-shadow: 0 0 6px rgba(22, 163, 74, 0.7);
        display: inline-block;
        flex-shrink: 0;
        animation: badgePulse 2s infinite ease-in-out;
    }
    @keyframes badgePulse {
        0%, 100% { transform: scale(0.95); opacity: 0.85; }
        50% { transform: scale(1.2); opacity: 1; box-shadow: 0 0 10px rgba(22, 163, 74, 0.9); }
    }
    .dot-amber {
        background-color: #16A34A !important;
        box-shadow: 0 0 6px rgba(22, 163, 74, 0.7) !important;
    }
    .dot-red {
        background-color: #EF4444 !important;
        box-shadow: 0 0 6px rgba(239, 68, 68, 0.7) !important;
    }

    .pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.85rem;
        font-weight: 700;
        padding: 0.45rem 1.05rem;
        border-radius: 9999px;
        white-space: nowrap;
        margin-top: 0.4rem;
        background: linear-gradient(135deg, #F0FDF4 0%, #ECFCCB 100%) !important;
        background-color: #F7FEE7 !important;
        color: #14532D !important;
        border: 1.5px solid #84CC16 !important;
        box-shadow: 0 2px 8px rgba(132, 204, 22, 0.16) !important;
    }
    .pill-file {
        background: linear-gradient(135deg, #F0FDF4 0%, #ECFCCB 100%) !important;
        background-color: #F7FEE7 !important;
        color: #14532D !important;
        border: 1.5px solid #84CC16 !important;
        border-radius: 9999px !important;
        box-shadow: 0 2px 8px rgba(132, 204, 22, 0.16) !important;
        font-size: 0.84rem !important;
        padding: 0.35rem 0.75rem !important;
    }
    .file-item-card {
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        background: linear-gradient(135deg, #F0FDF4 0%, #ECFCCB 100%) !important;
        background-color: #F7FEE7 !important;
        color: #14532D !important;
        border: 1.5px solid #84CC16 !important;
        border-radius: 9999px !important;
        box-shadow: 0 2px 6px rgba(132, 204, 22, 0.15) !important;
        padding: 0.35rem 0.75rem !important;
        font-size: 0.82rem !important;
        margin-top: 0.35rem !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }
    .file-item-name {
        font-weight: 700 !important;
        color: #14532D !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        display: inline-block !important;
        max-width: 100% !important;
    }
    .file-item-size {
        font-size: 0.75rem !important;
        color: #4D7C0F !important;
        font-weight: 600 !important;
        flex-shrink: 0 !important;
        margin-left: 0.35rem !important;
        white-space: nowrap !important;
    }
    .status-caption {
        font-size: 0.78rem;
        color: #4B5563;
        margin-top: 0.4rem;
        font-weight: 600;
    }

    /* ── 5. SUGGESTED QUESTIONS TITLE ── */
    .suggested-q-title {
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        color: #0284C7 !important;
        margin-top: 1.25rem !important;
        margin-bottom: 0.65rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }

    /* ── SUGGESTED QUESTIONS CHIPS (Exact Reference Match: White Rounded Cards) ── */
    button[key*="chip_q_"],
    div[class*="st-key-chip_q_"] button {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        border: 1.5px solid #E2E8F0 !important;
        border-radius: 12px !important;
        color: #1E293B !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 0.5rem 0.75rem !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
        transition: all 0.15s ease !important;
        min-height: 42px !important;
        height: 42px !important;
        white-space: nowrap !important;
        cursor: pointer !important;
        width: 100% !important;
    }
    button[key*="chip_q_"] *,
    div[class*="st-key-chip_q_"] button * {
        color: #1E293B !important;
        font-weight: 600 !important;
    }
    button[key*="chip_q_"]:hover,
    div[class*="st-key-chip_q_"] button:hover {
        background: #ECFCCB !important;
        background-color: #ECFCCB !important;
        border-color: #74D116 !important;
        box-shadow: 0 4px 14px rgba(116, 209, 22, 0.25) !important;
        transform: translateY(-2px) !important;
    }
    button[key*="chip_q_"]:hover *,
    div[class*="st-key-chip_q_"] button:hover * {
        color: #14532D !important;
        font-weight: 700 !important;
    }

    /* ── RECENT QUESTIONS SECTION & ASK BUTTONS ── */
    .recent-q-item {
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        color: #111827 !important;
        display: flex !important;
        align-items: center !important;
        min-height: 40px !important;
        height: 40px !important;
        line-height: 40px !important;
    }

    /* Reset column borders inside the expander */
    div[data-testid="stExpander"] [data-testid="stColumn"] > div:first-child {
        background: transparent !important;
        border: none !important;
        border-top: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }

    /* Ask ↗ Button styling inside Recent Questions */
    div[data-testid="stExpander"] button,
    div[class*="st-key-ask_rq_"] button,
    button[key*="ask_rq_"] {
        background: linear-gradient(135deg, #f0fde4 0%, #e8fad4 100%) !important;
        background-color: #f0fde4 !important;
        border: 2px solid #74D116 !important;
        border-radius: 9999px !important;
        color: #14532D !important;
        font-weight: 800 !important;
        font-size: 0.92rem !important;
        min-height: 38px !important;
        height: 38px !important;
        padding: 0 1.25rem !important;
        box-shadow: 0 2px 8px rgba(116, 209, 22, 0.2) !important;
        transition: all 0.18s ease !important;
        cursor: pointer !important;
        width: 100% !important;
    }

    /* Force text color to deep dark green #14532D so it is 100% visible */
    div[data-testid="stExpander"] button *,
    div[data-testid="stExpander"] button p,
    div[data-testid="stExpander"] button span,
    div[class*="st-key-ask_rq_"] button *,
    div[class*="st-key-ask_rq_"] button p,
    div[class*="st-key-ask_rq_"] button span,
    button[key*="ask_rq_"] *,
    button[key*="ask_rq_"] p,
    button[key*="ask_rq_"] span {
        color: #14532D !important;
        font-weight: 800 !important;
        font-size: 0.92rem !important;
        visibility: visible !important;
        opacity: 1 !important;
    }

    div[data-testid="stExpander"] button:hover,
    div[class*="st-key-ask_rq_"] button:hover,
    button[key*="ask_rq_"]:hover {
        background: #74D116 !important;
        background-color: #74D116 !important;
        border-color: #4D9C00 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 14px rgba(116, 209, 22, 0.45) !important;
        transform: translateY(-1px) !important;
    }

    div[data-testid="stExpander"] button:hover *,
    div[data-testid="stExpander"] button:hover p,
    div[data-testid="stExpander"] button:hover span,
    div[class*="st-key-ask_rq_"] button:hover *,
    div[class*="st-key-ask_rq_"] button:hover p,
    div[class*="st-key-ask_rq_"] button:hover span,
    button[key*="ask_rq_"]:hover * {
        color: #FFFFFF !important;
    }

    /* ── GENERAL SECONDARY BUTTONS ── */
    [data-testid="stBaseButton-secondary"]:not([key*="chip_q_"]):not([key*="ask_rq_"]),
    [data-testid="baseButton-secondary"]:not([key*="chip_q_"]):not([key*="ask_rq_"]),
    .stButton > button[data-testid="stBaseButton-secondary"]:not([key*="chip_q_"]):not([key*="ask_rq_"]),
    .stButton > button[data-testid="baseButton-secondary"]:not([key*="chip_q_"]):not([key*="ask_rq_"]) {
        background: linear-gradient(135deg, #f0fde4 0%, #e8fad4 100%) !important;
        border: 2px solid #74D116 !important;
        border-radius: 9999px !important;
        color: #1a5c00 !important;
        font-weight: 700 !important;
        font-size: 0.87rem !important;
        padding: 0.5rem 0.9rem !important;
        box-shadow: 0 2px 8px rgba(116,209,22,0.2) !important;
        transition: all 0.18s ease !important;
        min-height: 38px !important;
        height: auto !important;
        white-space: normal !important;
        cursor: pointer !important;
        width: 100% !important;
    }

    [data-testid="stBaseButton-secondary"]:not([key*="chip_q_"]):not([key*="ask_rq_"]):hover,
    [data-testid="baseButton-secondary"]:not([key*="chip_q_"]):not([key*="ask_rq_"]):hover,
    .stButton > button[data-testid="stBaseButton-secondary"]:not([key*="chip_q_"]):not([key*="ask_rq_"]):hover,
    .stButton > button[data-testid="baseButton-secondary"]:not([key*="chip_q_"]):not([key*="ask_rq_"]):hover,
    div[data-testid="stDownloadButton"] button:hover {
        background: #74D116 !important;
        background-color: #74D116 !important;
        border-color: #4D9C00 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 16px rgba(116, 209, 22, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    [data-testid="stBaseButton-secondary"]:not([key*="chip_q_"]):not([key*="ask_rq_"]):hover *,
    [data-testid="baseButton-secondary"]:not([key*="chip_q_"]):not([key*="ask_rq_"]):hover *,
    .stButton > button[data-testid="stBaseButton-secondary"]:not([key*="chip_q_"]):not([key*="ask_rq_"]):hover *,
    .stButton > button[data-testid="baseButton-secondary"]:not([key*="chip_q_"]):not([key*="ask_rq_"]):hover *,
    div[data-testid="stDownloadButton"] button:hover * {
        color: #FFFFFF !important;
    }



    /* ── 6. PRIMARY BUTTON (Vibrant Lime Green #74D116 as in reference "Swap" button) ── */
    html body [data-testid="stAppViewContainer"] button[kind="primary"],
    html body [data-testid="stAppViewContainer"] button[data-testid="baseButton-primary"],
    html body [data-testid="stAppViewContainer"] button[data-testid="stBaseButton-primary"],
    html body [data-testid="stAppViewContainer"] .stButton > button[kind="primary"],
    html body [data-testid="stAppViewContainer"] div[class*="st-key-btn_process"] button {
        background: #74D116 !important;
        background-color: #74D116 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
        padding: 0.75rem 1.5rem !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25) !important;
        letter-spacing: 0.02em !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        margin-top: 0.35rem !important;
    }
    html body [data-testid="stAppViewContainer"] button[kind="primary"] *,
    html body [data-testid="stAppViewContainer"] button[kind="primary"] p {
        color: #FFFFFF !important;
        font-weight: 800 !important;
    }
    html body [data-testid="stAppViewContainer"] button[kind="primary"]:hover {
        background: #62B80E !important;
        background-color: #62B80E !important;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35) !important;
        transform: translateY(-2px) !important;
        color: #FFFFFF !important;
    }

    /* Uploaded File Delete 'X' Button */
    button[key*="remove_"],
    div[class*="st-key-remove_"] button {
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
    button[key*="remove_"] *,
    div[class*="st-key-remove_"] button * {
        color: #DC2626 !important;
    }
    button[key*="remove_"]:hover,
    div[class*="st-key-remove_"] button:hover {
        background-color: #EF4444 !important;
        color: #FFFFFF !important;
        border-color: #DC2626 !important;
    }
    button[key*="remove_"]:hover *,
    div[class*="st-key-remove_"] button:hover * {
        color: #FFFFFF !important;
    }

    /* ── Hide ALL tooltips, help icons and popup boxes ── */
    button[data-testid="stTooltipIcon"],
    [data-testid="stTooltipHoverTarget"],
    .stTooltipIcon,
    [data-testid="stWidgetLabel"] button,
    label + div > svg,
    div[data-testid="stTooltipIcon"],
    [data-testid="stTooltip"],
    [role="tooltip"],
    div[class*="tooltip"],
    div[class*="Tooltip"],
    [data-baseweb="tooltip"],
    [data-baseweb="popover"] {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        opacity: 0 !important;
        pointer-events: none !important;
        max-width: 0 !important;
        max-height: 0 !important;
        overflow: hidden !important;
    }

    /* ── Selectbox: Clean green border, zero arc artifact ── */
    div[data-testid="stSelectbox"] { width: 100% !important; }

    /* Outer wrapper: rounded green border, clips inner content */
    div[data-testid="stSelectbox"] > div:last-child {
        border: 2px solid #74D116 !important;
        border-radius: 10px !important;
        background: #FFFFFF !important;
        overflow: hidden !important;
        box-shadow: none !important;
        padding: 2px !important;
        cursor: pointer !important;
        transition: all 0.22s ease !important;
    }
    div[data-testid="stSelectbox"] > div:last-child:hover,
    div[data-testid="stSelectbox"] > div:last-child:focus-within {
        background: #ECFCCB !important;
        background-color: #ECFCCB !important;
        border-color: #4D9C00 !important;
        box-shadow: 0 0 0 3px rgba(116,209,22,0.25), 0 4px 12px rgba(116,209,22,0.15) !important;
        transform: translateY(-1px) !important;
    }

    /* Kill ALL inner borders — every layer — so no arc can appear */
    div[data-testid="stSelectbox"] [data-baseweb="select"],
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div > div,
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div > div > div,
    div[data-testid="stSelectbox"] [data-baseweb="select"] input,
    div[data-testid="stSelectbox"] > div > div,
    div[data-testid="stSelectbox"] > div > div > div,
    div[data-testid="stSelectbox"] > div > div > div > div,
    div[data-testid="stSelectbox"] > div > div > div > div > div {
        border: none !important;
        border-color: transparent !important;
        border-width: 0 !important;
        outline: none !important;
        box-shadow: none !important;
        border-radius: 8px !important;
        background: transparent !important;
    }

    /* Dropdown menu options hover */
    ul[data-baseweb="menu"] li:hover,
    li[role="option"]:hover,
    li[role="option"][aria-selected="true"] {
        background-color: #D9F99D !important;
        color: #14532D !important;
        font-weight: 700 !important;
    }

    /* ── Toggle Switch Hover & Clickable Styling (Translate Responses & Voice Assistant Mode) ── */
    div[data-testid="stToggle"] {
        cursor: pointer !important;
        padding: 0.45rem 0.75rem !important;
        border-radius: 14px !important;
        border: 1.5px solid transparent !important;
        transition: all 0.22s ease !important;
    }
    div[data-testid="stToggle"]:hover {
        background-color: #ECFCCB !important;
        background: #ECFCCB !important;
        border-color: #84CC16 !important;
        box-shadow: 0 4px 12px rgba(116, 209, 22, 0.2) !important;
        transform: translateY(-1px) !important;
    }
    div[data-testid="stToggle"] label,
    div[data-testid="stToggle"] label * {
        cursor: pointer !important;
    }
    div[data-testid="stToggle"]:hover label p,
    div[data-testid="stToggle"]:hover label span {
        color: #14532D !important;
        font-weight: 700 !important;
    }
    /* Toggle active track green */
    div[data-testid="stToggle"] input:checked ~ div[class*="StyledTrack"],
    div[data-testid="stToggle"] input:checked + div,
    div[data-testid="stToggle"] [data-baseweb="checkbox"] [aria-checked="true"] {
        background-color: #74D116 !important;
    }

    /* ── Expander Summary Header Hover ── */
    div[data-testid="stExpander"] details summary {
        border-radius: 12px !important;
        padding: 0.5rem 0.85rem !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }
    div[data-testid="stExpander"] details summary:hover {
        background-color: #ECFCCB !important;
        background: #ECFCCB !important;
        color: #14532D !important;
    }
    div[data-testid="stExpander"] details summary:hover * {
        color: #14532D !important;
    }

    /* ── File Uploader Dropzone Hover ── */
    [data-testid="stFileUploaderDropzone"] {
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        border-radius: 14px !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        background-color: #ECFCCB !important;
        background: #ECFCCB !important;
        border-color: #74D116 !important;
        box-shadow: 0 4px 14px rgba(116, 209, 22, 0.18) !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        background-color: #74D116 !important;
        background: #74D116 !important;
        color: #FFFFFF !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover * {
        color: #FFFFFF !important;
    }

    /* Horizontal Divider */
    hr {
        margin: 1.25rem 0 !important;
        border: none !important;
        border-top: 2px solid #74D116 !important;
    }

    /* ── Voice Assistant Styling ── */
    div[data-testid="stAudioInput"] {
        background-color: #FFFFFF !important;
        border: 2px solid #D9F99D !important;
        border-radius: 16px !important;
        padding: 0.45rem 0.85rem !important;
        transition: all 0.22s ease !important;
        cursor: pointer !important;
    }
    div[data-testid="stAudioInput"]:hover {
        background-color: #F7FEE7 !important;
        background: #F7FEE7 !important;
        border-color: #84CC16 !important;
        box-shadow: 0 4px 16px rgba(132, 204, 22, 0.22) !important;
        transform: translateY(-1px) !important;
    }

    /* ── 7. CHATBOT WIDGET — Reference Match: White Pill Input + Orange Circular Chatbot Button (#FE5100) ── */
    div[data-testid="stBottom"] {
        background: transparent !important;
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
        display: flex !important;
        align-items: flex-end !important;
        justify-content: flex-end !important;
        padding: 0 1.5rem 1.25rem 0 !important;
        pointer-events: none !important;
    }
    div[data-testid="stBottom"]::before,
    div[data-testid="stBottom"]::after,
    div[data-testid="stBottom"] > div::before {
        display: none !important;
        background: transparent !important;
    }

    /* Container holding the white pill input on the left + the circular chatbot button on the right */
    div[data-testid="stBottom"] > div {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        max-width: 440px !important;
        width: 440px !important;
        min-width: 0 !important;
        flex-shrink: 0 !important;
        margin-left: auto !important;
        margin-right: 0 !important;
        padding: 0 !important;
        pointer-events: auto !important;
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        gap: 12px !important;
    }

    /* Floating circular chatbot assistant button (Complete Solid Green + Question mark symbol) */
    div[data-testid="stBottom"] > div::after {
        content: "" !important;
        width: 48px !important;
        height: 48px !important;
        min-width: 48px !important;
        min-height: 48px !important;
        border-radius: 50% !important;
        background-color: #74D116 !important;
        background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='3.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3'/%3E%3Cline x1='12' y1='17' x2='12.01' y2='17' stroke-width='3.8'/%3E%3C/svg%3E") no-repeat center / 24px 24px,
                    linear-gradient(135deg, #74D116 0%, #15803D 100%) !important;
        box-shadow: 0 4px 16px rgba(22, 163, 74, 0.45), 0 2px 6px rgba(0, 0, 0, 0.15) !important;
        flex-shrink: 0 !important;
        cursor: pointer !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        display: block !important;
    }
    div[data-testid="stBottom"] > div::after:hover {
        transform: scale(1.08) !important;
        box-shadow: 0 6px 22px rgba(22, 163, 74, 0.65) !important;
    }

    /* Intermediate wrapper reset */
    div[data-testid="stBottom"] > div > div,
    div[data-testid="stChatInputContainer"] {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        flex: 1 1 auto !important;
        min-width: 0 !important;
    }

    /* Outer stChatInput wrapper */
    div[data-testid="stChatInput"] {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
        width: 100% !important;
        flex: 1 1 auto !important;
        min-width: 0 !important;
    }

    /* THE GREEN CAPSULE INPUT PILL */
    div[data-testid="stChatInput"] > div {
        background: #74D116 !important;
        background-color: #74D116 !important;
        border: 2px solid #5EA808 !important;
        border-radius: 9999px !important;
        min-height: 48px !important;
        height: 48px !important;
        padding: 0 8px 0 20px !important;
        box-shadow: 0 4px 16px rgba(116, 209, 22, 0.35), 0 1px 3px rgba(0, 0, 0, 0.08) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: space-between !important;
        box-sizing: border-box !important;
        overflow: visible !important;
        width: 100% !important;
    }
    div[data-testid="stChatInput"] > div:focus-within {
        border-color: #3B7E00 !important;
        box-shadow: 0 0 0 3px rgba(116, 209, 22, 0.45), 0 6px 20px rgba(0, 0, 0, 0.12) !important;
    }

    /* Strip all inner div layers completely — zero overlap */
    div[data-testid="stChatInput"] > div div {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        border-color: transparent !important;
        box-shadow: none !important;
        outline: none !important;
        border-radius: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        min-height: unset !important;
    }

    /* Inner row flex container */
    div[data-testid="stChatInput"] [class*="e1p9v2yr3"],
    div[data-testid="stChatInput"] [class*="e1p9v2yr4"] {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        flex: 1 1 auto !important;
        min-width: 0 !important;
        height: 100% !important;
    }

    div[data-testid="stChatInput"] [class*="e1p9v2yr6"] {
        display: flex !important;
        align-items: center !important;
        flex-shrink: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Solid black text styling inside the green pill */
    div[data-testid="stChatInput"] textarea,
    textarea[data-testid="stChatInputTextArea"] {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        border-color: transparent !important;
        outline: none !important;
        box-shadow: none !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        color: #000000 !important;
        caret-color: #000000 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        height: 24px !important;
        min-height: 24px !important;
        max-height: 28px !important;
        line-height: 24px !important;
        resize: none !important;
        padding: 0 !important;
        margin: 0 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        width: 100% !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder,
    textarea[data-testid="stChatInputTextArea"]::placeholder {
        color: #000000 !important;
        opacity: 0.8 !important;
        font-size: 0.90rem !important;
        font-weight: 600 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        line-height: 24px !important;
    }

    /* Crisp white circular send button with black arrow inside green pill */
    div[data-testid="stChatInput"] button,
    button[data-testid="stChatInputSubmitButton"],
    div[data-testid="stChatInput"] > div button {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border-radius: 50% !important;
        border: none !important;
        height: 34px !important;
        width: 34px !important;
        min-height: 34px !important;
        min-width: 34px !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15) !important;
        transition: all 0.15s ease !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        flex-shrink: 0 !important;
        padding: 0 !important;
        margin: 0 0 0 6px !important;
        cursor: pointer !important;
    }
    div[data-testid="stChatInput"] button:hover,
    button[data-testid="stChatInputSubmitButton"]:hover,
    div[data-testid="stChatInput"] > div button:hover {
        background: #050814 !important;
        background-color: #050814 !important;
        color: #FFFFFF !important;
        transform: scale(1.08) !important;
    }
    div[data-testid="stChatInput"] button:hover svg,
    button[data-testid="stChatInputSubmitButton"]:hover svg {
        fill: #FFFFFF !important;
        color: #FFFFFF !important;
    }

    /* Native clean up arrow SVG in solid black */
    div[data-testid="stChatInput"] button svg,
    button[data-testid="stChatInputSubmitButton"] svg {
        display: block !important;
        fill: #000000 !important;
        color: #000000 !important;
        width: 17px !important;
        height: 17px !important;
    }
    div[data-testid="stChatInput"] button::after,
    button[data-testid="stChatInputSubmitButton"]::after {
        display: none !important;
        content: none !important;
    }



    /* ── 8. ANSWER SECTION & CARDS (Pure White, 8px Green Left Border, Rounded 20px) ── */
    @keyframes smoothAnswerReveal {
        0% { opacity: 0; transform: translateY(14px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    .answer-card-box {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        border: 2px solid #D9F99D !important;
        border-left: 8px solid #84CC16 !important;
        border-radius: 20px !important;
        padding: 1.6rem 2rem !important;
        margin-top: 0.75rem !important;
        margin-bottom: 1.15rem !important;
        box-shadow: 0 16px 38px -4px rgba(5, 8, 20, 0.12), 0 4px 18px rgba(132, 204, 22, 0.2) !important;
        color: #111827 !important;
        font-size: 0.98rem !important;
        line-height: 1.8 !important;
        white-space: pre-wrap !important;
        animation: smoothAnswerReveal 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards !important;
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease !important;
    }
    .answer-card-box:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 22px 46px -4px rgba(5, 8, 20, 0.18), 0 6px 26px rgba(132, 204, 22, 0.32) !important;
        border-color: #84CC16 !important;
    }

    /* Structured Section Cards */
    .section-card {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FCF5 100%) !important;
        background-color: #FFFFFF !important;
        border: 2px solid #D9F99D !important;
        border-left: 8px solid #84CC16 !important;
        border-radius: 20px !important;
        padding: 1.55rem 2rem !important;
        margin-bottom: 1.25rem !important;
        box-shadow: 0 14px 34px -4px rgba(5, 8, 20, 0.1), 0 4px 18px rgba(132, 204, 22, 0.18) !important;
        transition: all 0.25s ease !important;
        animation: smoothAnswerReveal 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards !important;
    }
    .section-card:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 20px 44px -4px rgba(5, 8, 20, 0.16), 0 6px 26px rgba(132, 204, 22, 0.3) !important;
        border-color: #84CC16 !important;
    }
    .section-card-title {
        font-size: 1.15rem !important;
        font-weight: 800 !important;
        color: #14532D !important;
        margin-bottom: 0.75rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.6rem !important;
        border-bottom: 2px dashed #D9F99D !important;
        padding-bottom: 0.55rem !important;
        letter-spacing: -0.01em !important;
    }
    .section-card-body {
        font-size: 0.98rem !important;
        color: #1F2937 !important;
        line-height: 1.8 !important;
        white-space: pre-wrap !important;
    }

    /* Clinical Evidence Card */
    .clinical-evidence-card {
        background-color: #FFFFFF !important;
        border: 2px solid #D9F99D !important;
        border-left: 8px solid #84CC16 !important;
        border-radius: 20px !important;
        box-shadow: 0 14px 34px -4px rgba(5, 8, 20, 0.1), 0 4px 18px rgba(132, 204, 22, 0.18) !important;
        padding: 1.55rem 1.95rem !important;
        color: #111827 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        transition: all 0.25s ease !important;
        animation: smoothAnswerReveal 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards !important;
    }
    .evidence-header-label {
        font-size: 1rem !important;
        font-weight: 800 !important;
        color: #14532D !important;
        margin-bottom: 0.35rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.45rem !important;
    }
    .evidence-patient-name {
        font-size: 1.15rem !important;
        font-weight: 900 !important;
        color: #050814 !important;
        padding-left: 0.15rem !important;
    }
    .evidence-val-text {
        font-size: 0.98rem !important;
        font-weight: 700 !important;
        color: #111827 !important;
        padding-left: 0.15rem !important;
    }
    .evidence-bullet-list {
        list-style: none !important;
        padding-left: 0.15rem !important;
        margin: 0.35rem 0 0 0 !important;
    }
    .evidence-bullet-list li {
        font-size: 0.94rem !important;
        color: #374151 !important;
        line-height: 1.7 !important;
        position: relative !important;
        padding-left: 1.25rem !important;
    }
    .evidence-bullet-list li::before {
        content: "•" !important;
        color: #84CC16 !important;
        font-weight: 900 !important;
        font-size: 1.35rem !important;
        position: absolute !important;
        left: 0.1rem !important;
        top: -0.15rem !important;
    }

    /* ── Chat User Bubble ── */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
        background: #050814 !important;
        color: #FFFFFF !important;
        border-radius: 20px 20px 4px 20px !important;
        padding: 0.95rem 1.45rem !important;
        box-shadow: 0 6px 20px rgba(5, 8, 20, 0.35) !important;
        border: 2px solid #84CC16 !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] p {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="chatAvatarIcon-user"] {
        background: #84CC16 !important;
        color: #FFFFFF !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="chatAvatarIcon-assistant"] {
        background: #050814 !important;
        color: #84CC16 !important;
        border: 1.5px solid #84CC16 !important;
    }

    /* ── Upload Area ── */
    div[data-testid="stFileUploader"] section {
        background-color: #FFFFFF !important;
        border: 2px dashed #84CC16 !important;
        border-radius: 20px !important;
        padding: 0.85rem 1.15rem !important;
        box-shadow: 0 4px 14px rgba(132, 204, 22, 0.12) !important;
        transition: all 0.25s ease-in-out !important;
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: #65A30D !important;
        background-color: #F0FDF4 !important;
        box-shadow: 0 0 22px rgba(132, 204, 22, 0.28) !important;
        transform: translateY(-2px) !important;
    }
    div[data-testid="stFileUploader"] section [data-testid="stMarkdownContainer"] p {
        color: #365314 !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        margin: 0 !important;
    }

    /* ── Expanders (Clean Card Design Matching Screenshot) ── */
    div[data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04) !important;
        overflow: hidden !important;
        margin-top: 0.75rem !important;
    }
    div[data-testid="stExpander"] summary {
        background-color: #F8FAFC !important;
        padding: 0.65rem 1rem !important;
        font-weight: 700 !important;
        color: #334155 !important;
        border-bottom: 1px solid #F1F5F9 !important;
    }
    div[data-testid="stExpander"] summary:hover {
        color: #0284C7 !important;
    }
    div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        padding: 0.75rem 1rem !important;
    }

    /* ── App Footer ── */
    .app-footer {
        text-align: center;
        color: #1F2937;
        font-size: 0.88rem;
        font-weight: 700;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 2px solid #D9F99D;
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
        .hero-title-text {
            font-size: 1.8rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ── Header Hero Banner (Dark #050814 with Neon Highlights #84CC16) ────────────
st.markdown("""
<div class="saas-hero-banner">
    <div class="hero-tagline-badge">
        <span class="hero-pulse-dot"></span>
        <span>AI-POWERED CLINICAL DECISION SUPPORT PLATFORM</span>
        <span style="opacity: 0.4;">|</span>
        <span style="color: #D9F99D;">AWS BEDROCK RAG</span>
    </div>
    <h1 class="hero-title-text">
        <span style="color:#74D116;">🩺 Multimodal Dengue Report RAG Assistant</span>
    </h1>
    <p class="hero-subtitle-text">
        Clinical Decision Support System • High-Precision Diagnostics & Grounded Evidence Synthesis
    </p>
    <div class="hero-chip-row">
        <span class="hero-chip">🟢 Bedrock Knowledge Base Active</span>
        <span class="hero-chip">⚡ Real-Time Multimodal Retrieval</span>
        <span class="hero-chip">🛡️ Clinical Guardrails Synced</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize required session state keys safely
if "active_patient_display" not in st.session_state:
    st.session_state.active_patient_display = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "recent_questions" not in st.session_state or not st.session_state.recent_questions:
    st.session_state.recent_questions = [
        "What is the diagnosis?",
        "What is the platelet count?",
        "What is the risk level?"
    ]

# ── Horizontal Top Row: 4 Modern Cards (Exact Structure, Equal Heights) ──────
col1, col2, col3, col4 = st.columns([1.0, 2.1, 1.0, 1.0], gap="medium")

# Column 1: Configuration
with col1:
    with st.container(border=True, key="card_config"):
        st.markdown('<div class="col-header">⚙️ Configuration</div>', unsafe_allow_html=True)
        selected_model = st.selectbox(
            "Select LLM Model",
            ["tinyllama", "phi3", "gemma2:2b", "llama3"],
            index=0,
            label_visibility="collapsed"
        )
        st.markdown(
            f'''<div class="product-badge" style="margin-top: 0.65rem;">
                <span class="badge-dot"></span>
                <span class="product-badge-label">Active Model</span>
                <span class="product-badge-state">🟢 {selected_model}</span>
            </div>''',
            unsafe_allow_html=True
        )

# Column 2: Document Upload
with col2:
    with st.container(border=True, key="card_upload"):
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
                col_finfo, col_fdel = st.columns([0.88, 0.12])
                with col_finfo:
                    st.markdown(
                        f'<div class="file-item-card" title="{file.name} ({size_str})">'
                        f'<div style="display:flex; align-items:center; gap:0.4rem; min-width:0; overflow:hidden;">'
                        f'<span style="flex-shrink:0;">📄</span>'
                        f'<span class="file-item-name">{file.name}</span>'
                        f'</div>'
                        f'<span class="file-item-size">{size_str}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                with col_fdel:
                    if st.button("✕", key=f"remove_uploaded_{idx}_{file.name[:8]}", use_container_width=True):
                        st.session_state.uploader_key += 1
                        clean_directories()
                        st.cache_resource.clear()
                        st.session_state.messages = []
                        st.session_state.active_patient_display = None
                        st.rerun()
        else:
            # If files were previously ingested and reside in reports/
            active_files = [f for f in os.listdir("reports") if os.path.isfile(os.path.join("reports", f)) and not f.endswith(".json")] if os.path.exists("reports") else []
            if active_files:
                for idx, fname in enumerate(active_files):
                    fpath = os.path.join("reports", fname)
                    fsize = os.path.getsize(fpath) / 1024
                    size_str = f"{fsize:.1f} KB" if fsize < 1024 else f"{fsize/1024:.1f} MB"
                    col_finfo, col_fdel = st.columns([0.88, 0.12])
                    with col_finfo:
                        st.markdown(
                            f'<div class="file-item-card" title="{fname} ({size_str})">'
                            f'<div style="display:flex; align-items:center; gap:0.4rem; min-width:0; overflow:hidden;">'
                            f'<span style="flex-shrink:0;">📄</span>'
                            f'<span class="file-item-name">{fname}</span>'
                            f'</div>'
                            f'<span class="file-item-size">{size_str}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    with col_fdel:
                        if st.button("✕", key=f"remove_active_{idx}_{fname[:8]}", use_container_width=True):
                            clean_directories()
                            st.cache_resource.clear()
                            st.session_state.messages = []
                            st.session_state.active_patient_display = None
                            st.session_state.report_processed = False
                            st.rerun()

        process_clicked = st.button("🚀 Process Documents", type="primary", use_container_width=True, key="btn_process_docs")

# Column 3: Index Status
with col3:
    with st.container(border=True, key="card_index"):
        st.markdown('<div class="col-header">🗄️ Index Status</div>', unsafe_allow_html=True)
        if load_vectorstore() is not None:
            faiss_badge = '<div class="product-badge"><span class="badge-dot"></span><span class="product-badge-label">FAISS Vector</span><span class="product-badge-state">🟢 Ready</span></div>'
        else:
            faiss_badge = '<div class="product-badge"><span class="badge-dot"></span><span class="product-badge-label">FAISS Vector</span><span class="product-badge-state">🟢 Standby</span></div>'
        
        bedrock_badge = '<div class="product-badge"><span class="badge-dot"></span><span class="product-badge-label">AWS Bedrock</span><span class="product-badge-state">🟢 Connected</span></div>'
        kb_badge = '<div class="product-badge"><span class="badge-dot"></span><span class="product-badge-label">Knowledge Base</span><span class="product-badge-state">🟢 Active</span></div>'

        st.markdown(f'<div class="product-badge-stack">{faiss_badge}{bedrock_badge}{kb_badge}</div>', unsafe_allow_html=True)

# Column 4: AI / Ollama Status
with col4:
    with st.container(border=True, key="card_ollama"):
        st.markdown('<div class="col-header">🤖 AI Engine Status</div>', unsafe_allow_html=True)
        ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        ollama_running = False
        available_models = []
        try:
            res = requests.get(f"{ollama_base}/api/tags", timeout=1.5)
            if res.status_code == 200:
                ollama_running = True
                available_models = [m['name'] for m in res.json().get('models', [])]
        except Exception:
            ollama_running = False

        if ollama_running:
            ollama_badge = '<div class="product-badge"><span class="badge-dot"></span><span class="product-badge-label">Ollama LLM</span><span class="product-badge-state">🟢 Active</span></div>'
            if any(m.startswith(selected_model) for m in available_models):
                model_badge = f'<div class="product-badge"><span class="badge-dot"></span><span class="product-badge-label">{selected_model}</span><span class="product-badge-state">🟢 Ready</span></div>'
                st.markdown(f'<div class="product-badge-stack">{ollama_badge}{model_badge}</div>', unsafe_allow_html=True)
            else:
                model_badge = f'<div class="product-badge badge-pending"><span class="badge-dot dot-amber"></span><span class="product-badge-label">{selected_model}</span><span class="product-badge-state state-pending">⚠️ Missing</span></div>'
                st.markdown(f'<div class="product-badge-stack">{ollama_badge}{model_badge}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="status-caption">Run: <code>ollama pull {selected_model}</code></div>', unsafe_allow_html=True)
        else:
            # Cloud Deployment Mode: Grounded Clinical RAG Engine
            cloud_badge = '<div class="product-badge"><span class="badge-dot"></span><span class="product-badge-label">Cloud RAG</span><span class="product-badge-state">🟢 Active</span></div>'
            guard_badge = '<div class="product-badge"><span class="badge-dot"></span><span class="product-badge-label">Clinical AI</span><span class="product-badge-state">🟢 Ready</span></div>'
            st.markdown(f'<div class="product-badge-stack">{cloud_badge}{guard_badge}</div>', unsafe_allow_html=True)
            st.markdown('<div class="status-caption">Mode: <b>Cloud Clinical Grounded RAG</b></div>', unsafe_allow_html=True)

# Divider
st.markdown("---")

# ── Document Ingestion Processing ────────────────────────────────────────────
if "report_processed" not in st.session_state:
    if load_vectorstore() is not None:
        st.session_state.report_processed = True
        if not st.session_state.get("active_patient_display"):
            meta = get_active_report_meta()
            p_name = meta.get("patient_name")
            p_id = meta.get("patient_id")
            if p_name and p_name not in ["Not specified", "Unknown"]:
                if p_id and p_id not in ["Not specified", "Extracted", "Unknown"]:
                    st.session_state.active_patient_display = f"{p_name} ({p_id})"
                else:
                    st.session_state.active_patient_display = p_name
    else:
        st.session_state.report_processed = False

if process_clicked:
    if uploaded_files:
        with st.spinner("Clearing previous session and ingesting new report..."):
            st.cache_resource.clear()
            clean_directories()
            st.session_state.messages = []
            st.session_state.active_patient_display = None
            st.session_state.report_processed = False

            for uploaded_file in uploaded_files:
                dest_path = os.path.join("reports", uploaded_file.name)
                with open(dest_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            from ingest import ingest_documents
            try:
                success, meta_info = ingest_documents()
            except Exception as e:
                logger.error(f"Ingestion error: {e}")
                st.error(f"Error processing documents: {e}")
                success, meta_info = False, {}

            if success:
                st.cache_resource.clear()
                p_name = meta_info.get("patient_name", "Extracted")
                p_id = meta_info.get("patient_id", "Not specified")
                if p_id and p_id not in ["Not specified", "Extracted", "Unknown"]:
                    st.session_state.active_patient_display = f"{p_name} ({p_id})"
                else:
                    st.session_state.active_patient_display = p_name
                st.session_state.report_processed = True
                st.toast(f"✅ Ingested: {st.session_state.active_patient_display}", icon="🟢")
                st.success(f"✅ Report processed successfully! Active: **{st.session_state.active_patient_display}** | Chunks: {meta_info.get('chunk_count', 0)}")
                st.rerun()
            else:
                if not meta_info:
                    pass
                else:
                    st.error("Failed to ingest documents.")
    else:
        st.warning("Please upload a medical report (PDF, TXT, DOCX) first.")

# ── Dynamic Current Patient Badge (Strictly shown AFTER processing report) ───
if st.session_state.get("report_processed") and st.session_state.get("active_patient_display"):
    st.markdown(
        f'<div style="margin-bottom: 0.65rem;">'
        f'<span class="patient-pill-badge">👤 Current Patient: <strong>{st.session_state.active_patient_display}</strong></span>'
        f'</div>',
        unsafe_allow_html=True
    )

# ── Session State Management (Latest Q&A Only & Recent Questions) ───────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "recent_questions" not in st.session_state or not st.session_state.recent_questions:
    st.session_state.recent_questions = [
        "What is the diagnosis?",
        "What is the platelet count?",
        "What is the risk level?"
    ]

# ── Multilingual Voice Assistant Controls (Sky Blue Healthcare Card) ────────
with st.container(border=True, key="card_voice"):
    col_vlang, col_vtrans, col_vmode = st.columns([0.34, 0.33, 0.33])
    with col_vlang:
        selected_language = st.selectbox(
            "🌐 Language",
            options=["English", "Hindi", "Kannada", "Telugu", "Tamil", "Marathi"],
            index=0,
            key="ui_language_select"
        )
    with col_vtrans:
        st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
        translate_responses = st.toggle(
            "🌐 Translate Responses",
            value=(selected_language != "English"),
            key="ui_translate_responses_toggle"
        )
    with col_vmode:
        st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
        voice_assistant_mode = st.toggle(
            "🎙️ Voice Assistant Mode",
            value=False,
            key="ui_voice_assistant_mode_toggle"
        )

    # Voice Input Control
    st.markdown(
        f'<div style="font-size: 0.88rem; font-weight: 700; color: #365314; margin-top: 0.35rem; margin-bottom: 0.25rem;">'
        f'🎤 Voice Input <span style="font-weight: 500; font-size: 0.8rem; color: #4B5563;">(Click mic to speak in <b>{selected_language}</b>)</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    audio_voice_file = st.audio_input(
        label=f"Speak question in {selected_language}",
        label_visibility="collapsed",
        key=f"audio_mic_input_{selected_language}"
    )

    if audio_voice_file is not None:
        raw_audio_bytes = audio_voice_file.getvalue()
        audio_fingerprint = hash(raw_audio_bytes)
        if st.session_state.get("last_processed_audio_hash") != audio_fingerprint:
            st.session_state.last_processed_audio_hash = audio_fingerprint
            with st.spinner(f"Transcribing voice input in {selected_language}..."):
                v_ok, v_text = transcribe_audio_bytes(raw_audio_bytes, lang=selected_language)
            if v_ok and v_text.strip():
                if voice_assistant_mode:
                    if v_text not in st.session_state.recent_questions:
                        st.session_state.recent_questions.insert(0, v_text)
                        st.session_state.recent_questions = st.session_state.recent_questions[:5]
                    st.session_state.messages = [{"role": "user", "content": v_text}]
                    st.session_state.auto_speak_answer = True
                    st.rerun()
                else:
                    st.session_state.pending_voice_query = v_text
                    st.rerun()
            else:
                st.warning(f"Voice recognition: {v_text}. Please speak clearly into the microphone.")

    # Show pending voice query if voice assistant mode is not automatically submitting
    if st.session_state.get("pending_voice_query"):
        with st.container(border=True, key="card_pending_voice"):
            st.markdown(
                f'<div style="font-size: 0.92rem; color: #365314; font-weight: 600; margin-bottom: 0.35rem;">'
                f'🎤 <b>Voice Input Recognized:</b> "{st.session_state.pending_voice_query}"'
                f'</div>',
                unsafe_allow_html=True
            )
            col_v1, col_v2 = st.columns([0.3, 0.7])
            with col_v1:
                if st.button("🚀 Submit Question", key="btn_submit_pending_voice", use_container_width=True):
                    v_prompt = st.session_state.pending_voice_query
                    del st.session_state.pending_voice_query
                    if v_prompt not in st.session_state.recent_questions:
                        st.session_state.recent_questions.insert(0, v_prompt)
                        st.session_state.recent_questions = st.session_state.recent_questions[:5]
                    st.session_state.messages = [{"role": "user", "content": v_prompt}]
                    st.rerun()
            with col_v2:
                if st.button("✕ Discard", key="btn_discard_pending_voice"):
                    del st.session_state.pending_voice_query
                    st.rerun()

# ── Smart Suggestions Dynamic Generator ─────────────────────────────────────
def get_smart_suggestions() -> list:
    """
    Returns dynamic suggested questions based on the active report content.
    Returns list of tuples: (compact_pill_label, full_question)
    """
    base_suggestions = [
        ("Diagnosis", "What is the diagnosis?"),
        ("Risk Factors", "Why is the patient at risk?"),
        ("Platelets", "What is the platelet count?"),
        ("Platelet Normalcy", "Is the platelet count normal?"),
        ("Risk", "What is the risk level?")
    ]

    active_files = [f for f in os.listdir("reports") if os.path.isfile(os.path.join("reports", f)) and not f.endswith(".json")] if os.path.exists("reports") else []
    if not active_files:
        return base_suggestions

    combined_text = ""
    for af in active_files:
        p = os.path.join("reports", af)
        try:
            from ingest import extract_text_from_file
            combined_text += " " + extract_text_from_file(p)
        except Exception:
            pass

    if not combined_text:
        return base_suggestions

    return base_suggestions

# ── Suggested Questions Section ──────────────────────────────────────────────
st.markdown("""
<div class="suggested-q-title">💡 Suggested Questions</div>
""", unsafe_allow_html=True)

smart_queries = get_smart_suggestions()

# Fallback if empty
if not smart_queries:
    smart_queries = [
        ("Diagnosis", "What is the diagnosis?"),
        ("Risk Factors", "Why is the patient at risk?"),
        ("Platelets", "What is the platelet count?"),
        ("Platelet Normalcy", "Is the platelet count normal?"),
        ("Risk", "What is the risk level?")
    ]

sq_cols = st.columns(len(smart_queries))
for idx, (col, (pill_label, full_q)) in enumerate(zip(sq_cols, smart_queries)):
    with col:
        # No help= to avoid label rendering conflicts
        if st.button(pill_label, key=f"chip_q_{idx}", use_container_width=True):
            if full_q not in st.session_state.recent_questions:
                st.session_state.recent_questions.insert(0, full_q)
                st.session_state.recent_questions = st.session_state.recent_questions[:5]
            st.session_state.messages = [{"role": "user", "content": full_q}]
            st.rerun()


# ── Recent Questions (Last 5 Questions Stored — Always Visible as in Reference) ───
if not st.session_state.get("recent_questions"):
    st.session_state.recent_questions = [
        "What is the diagnosis?",
        "What is the platelet count?",
        "What is the risk level?"
    ]

with st.expander("🕒 Recent Questions", expanded=True):
    for idx, rq in enumerate(st.session_state.recent_questions):
        col_rq_text, col_rq_btn = st.columns([0.84, 0.16])
        with col_rq_text:
            st.markdown(f'<div class="recent-q-item">• {rq}</div>', unsafe_allow_html=True)
        with col_rq_btn:
            if st.button("Ask ↗", key=f"ask_rq_btn_{idx}", use_container_width=True):
                st.session_state.messages = [{"role": "user", "content": rq}]
                st.rerun()

# ── Input query (Reference Match: Capsule input with 'Ask question...') ─────
if prompt := st.chat_input("Ask question..."):
    # Add to recent questions (max 5)
    if prompt not in st.session_state.recent_questions:
        st.session_state.recent_questions.insert(0, prompt)
        st.session_state.recent_questions = st.session_state.recent_questions[:5]
    # Keep only the latest prompt
    st.session_state.messages = [{"role": "user", "content": prompt}]

# ── Answer Section Helper (Formats Clean Single Direct Card) ─────────────────
def render_styled_answer_cards(answer_text: str, question: str = "") -> str:
    """
    Renders a single clean answer card without multiple paragraphs, question numbers,
    or answer numbers. Contains strictly the direct question-related answer.
    """
    cleaned = clean_simple_direct_answer(answer_text, question)
    return f'<div class="answer-card-box">{cleaned}</div>'

# ── Chat Area Rendering ──────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    # Render messages (only latest pair)
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                clean_content = clean_simple_direct_answer(message["content"])
                ans_html = render_styled_answer_cards(clean_content)
                st.markdown(ans_html, unsafe_allow_html=True)

                # Multilingual Voice Output (Listen to Answer button)
                msg_lang = message.get("lang", selected_language)
                col_hl, _ = st.columns([0.45, 0.55])
                with col_hl:
                    if st.button(f"🔊 Listen to Answer ({msg_lang})", key=f"btn_listen_hist_{idx}"):
                        with st.spinner(f"🔊 Generating speech in {msg_lang}..."):
                            h_audio = text_to_speech_audio(clean_content, lang=msg_lang)
                        if h_audio:
                            st.audio(h_audio, format="audio/mp3", autoplay=True)
            else:
                st.markdown(message["content"], unsafe_allow_html=True)

    # Trigger generation when user enters a prompt
    if len(st.session_state.messages) == 1 and st.session_state.messages[-1]["role"] == "user":
        prompt = st.session_state.messages[-1]["content"]

        with st.chat_message("assistant"):
            # 1. Smart Language Detection
            detected_lang = detect_language(prompt)

            # 2. Query Translation for RAG (reports are in English)
            if detected_lang != "English":
                rag_query = translate_text(prompt, target_lang="English", source_lang=detected_lang)
            elif selected_language != "English":
                rag_query = translate_text(prompt, target_lang="English", source_lang=selected_language)
            else:
                rag_query = prompt

            if detected_lang != "English":
                st.markdown(
                    f'<div style="margin-bottom: 0.65rem;">'
                    f'<span class="pill-badge pill-green">🌐 Detected Language: <b>{detected_lang}</b></span> '
                    f'<span class="status-caption" style="margin-left: 0.4rem;">(Interpreted clinical query: <i>"{rag_query}"</i>)</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # Small animated typing indicator
            with st.status("🔍 Analyzing Patient Report...", expanded=False) as status:
                # UNCHANGED RAG CALL
                answer, retrieved_patient_ui, evidence_list = generate_answer(rag_query, model_name=selected_model)
                status.update(label="Assessment Complete ✅", state="complete")

            # 3. Translation Mode
            target_lang = "English"
            if translate_responses:
                target_lang = selected_language
            elif detected_lang != "English":
                target_lang = detected_lang

            clean_ans = clean_simple_direct_answer(answer, prompt)
            if target_lang != "English":
                with st.spinner(f"🌐 Translating response to {target_lang}..."):
                    display_answer = clean_simple_direct_answer(
                        translate_text(clean_ans, target_lang=target_lang, source_lang="English"),
                        prompt
                    )
            else:
                display_answer = clean_ans

            # Update Current Patient dynamically based on retrieved record
            if evidence_list and isinstance(evidence_list, dict):
                ret_pname = evidence_list.get("patient_name")
                ret_pid = evidence_list.get("patient_id")
                if ret_pname and ret_pname not in ["Not specified", "Unknown"]:
                    if ret_pid and ret_pid not in ["Not specified", "Extracted", "Unknown"]:
                        st.session_state.active_patient_display = f"{ret_pname} ({ret_pid})"
                    else:
                        st.session_state.active_patient_display = ret_pname

            # Show Answer Section in modern redesigned cards
            answer_cards_html = render_styled_answer_cards(display_answer, prompt)
            st.markdown(answer_cards_html, unsafe_allow_html=True)

            # Show Action Buttons (Listen to Answer, Download PDF)
            col_act1, col_act2 = st.columns([0.45, 0.55])
            with col_act1:
                listen_now_clicked = st.button(f"🔊 Listen to Answer ({target_lang})", key=f"btn_listen_current_{len(st.session_state.messages)}")
            with col_act2:
                target_pname = st.session_state.active_patient_display
                if evidence_list and isinstance(evidence_list, dict):
                    ep = evidence_list.get("patient_name")
                    if ep and ep != "Not specified":
                        target_pname = ep

                pdf_bytes = generate_answer_pdf(
                    question=prompt,
                    answer=display_answer,
                    patient_name=target_pname
                )

                st.download_button(
                    label="📥 Download Answer as PDF",
                    data=pdf_bytes,
                    file_name=f"Clinical_Answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=False
                )

            # Auto-speak if Voice Assistant Mode is active, or user clicked Listen
            is_auto_speak = st.session_state.get("auto_speak_answer", False) or voice_assistant_mode
            if listen_now_clicked or is_auto_speak:
                with st.spinner(f"🔊 Synthesizing voice response in {target_lang}..."):
                    speech_audio = text_to_speech_audio(display_answer, lang=target_lang)
                if speech_audio:
                    st.audio(speech_audio, format="audio/mp3", autoplay=True)
                st.session_state.auto_speak_answer = False

            # Show Clean Clinical Evidence Card (Zero Technical RAG Details)
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
                    st.markdown(clinical_card_html, unsafe_allow_html=True)

            # Save latest state with language metadata
            st.session_state.messages.append({
                "role": "assistant",
                "content": display_answer,
                "lang": target_lang
            })

# ── Tiny Footer ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Powered by: Amazon S3 • Amazon Bedrock • Knowledge Base • RAG
</div>
""", unsafe_allow_html=True)
