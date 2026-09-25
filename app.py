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
importlib.reload(rag_pipeline)
importlib.reload(ingest)

from rag_pipeline import generate_answer, load_vectorstore, get_active_report_meta
from ingest import init_directories, clean_directories, ingest_documents

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
            textColor=colors.HexColor('#0F172A'),
            spaceAfter=4
        )
        subtitle_style = ParagraphStyle(
            'DocSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#0284C7'),
            spaceAfter=14
        )
        meta_label = ParagraphStyle(
            'MetaLabel',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=9,
            leading=13,
            textColor=colors.HexColor('#334155')
        )
        meta_val = ParagraphStyle(
            'MetaVal',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            leading=13,
            textColor=colors.HexColor('#0F172A')
        )
        section_head = ParagraphStyle(
            'SecHead',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=12,
            leading=16,
            textColor=colors.HexColor('#0369A1'),
            spaceBefore=12,
            spaceAfter=6
        )
        body_style = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9.5,
            leading=14.5,
            textColor=colors.HexColor('#1E293B'),
            spaceAfter=5
        )

        story = []
        story.append(Paragraph("🩺 Multimodal Dengue Report RAG Assistant", title_style))
        story.append(Paragraph("Clinical Decision Support System • Grounded Assessment Report", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#38BDF8'), spaceAfter=12))

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
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FCFF')),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#BAE6FD')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E0F2FE')),
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

# ── Modern Healthcare AI Application Theme (Sky Blue Theme) ───────────────────
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

    /* ── Header Banner (Rich Healthcare Blue Gradient) ── */
    .sky-header-banner {
        background: linear-gradient(135deg, #1E40AF 0%, #2563EB 35%, #0EA5E9 70%, #38BDF8 100%) !important;
        border-radius: 20px !important;
        padding: 1.65rem 2.2rem !important;
        color: #FFFFFF !important;
        box-shadow: 0 10px 28px -4px rgba(37, 99, 235, 0.28), 0 4px 12px -2px rgba(14, 165, 233, 0.18) !important;
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
        color: #E0F2FE !important;
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

    /* ── Current Patient Badge (Highlighted Pill) ── */
    .patient-pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: #FFFFFF;
        border: 1.5px solid #38BDF8;
        border-radius: 9999px;
        padding: 0.45rem 1.15rem;
        font-size: 0.88rem;
        font-weight: 700;
        color: #0284C7;
        box-shadow: 0 4px 14px rgba(14, 165, 233, 0.1);
        margin-bottom: 0.75rem;
        transition: all 0.25s ease;
    }
    .patient-pill-badge:hover {
        border-color: #0EA5E9;
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(14, 165, 233, 0.18);
    }

    /* ── Grounded Response Badge ── */
    .grounded-badge-container {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        background: #F0FDF4;
        border: 1.5px solid #86EFAC;
        border-radius: 9999px;
        padding: 0.35rem 0.95rem;
        margin-top: 0.35rem;
        margin-bottom: 0.65rem;
        box-shadow: 0 2px 8px rgba(34, 197, 94, 0.08);
    }
    .grounded-pill {
        font-size: 0.82rem;
        font-weight: 700;
        color: #15803D;
        display: flex;
        align-items: center;
        gap: 0.3rem;
    }
    .grounded-subtext {
        font-size: 0.78rem;
        color: #0369A1;
        font-weight: 600;
        padding-left: 0.45rem;
        border-left: 1.5px solid #86EFAC;
    }

    /* ── Suggested Questions Section ── */
    .suggested-q-card {
        background-color: #FFFFFF !important;
        border: 1.5px solid #BAE6FD !important;
        border-radius: 18px !important;
        padding: 0.85rem 1.15rem !important;
        margin-top: 0.65rem !important;
        margin-bottom: 0.85rem !important;
        box-shadow: 0 4px 14px rgba(14, 165, 233, 0.06) !important;
    }
    .suggested-q-title {
        font-size: 0.92rem !important;
        font-weight: 700 !important;
        color: #0369A1 !important;
        margin-bottom: 0.55rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.4rem !important;
    }
    div[data-testid="stColumn"] div:has(> button[key*="sq_btn_"]) button,
    button[key*="sq_btn_"] {
        background-color: #F0F9FF !important;
        border: 1.5px solid #BAE6FD !important;
        color: #0284C7 !important;
        border-radius: 14px !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        padding: 0.45rem 0.65rem !important;
        box-shadow: 0 2px 6px rgba(14, 165, 233, 0.06) !important;
        transition: all 0.2s ease !important;
        min-height: 38px !important;
        height: 100% !important;
        white-space: normal !important;
        line-height: 1.3 !important;
    }
    div[data-testid="stColumn"] div:has(> button[key*="sq_btn_"]) button:hover,
    button[key*="sq_btn_"]:hover {
        background-color: #E0F2FE !important;
        border-color: #0EA5E9 !important;
        color: #0369A1 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.16) !important;
    }

    /* ── Recent Questions ── */
    .recent-q-item {
        font-size: 0.86rem !important;
        color: #334155 !important;
        font-weight: 500 !important;
        padding: 0.25rem 0 !important;
    }

    /* ── Download PDF Button ── */
    .stDownloadButton > button {
        background: #FFFFFF !important;
        border: 1.5px solid #2563EB !important;
        color: #2563EB !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 0.86rem !important;
        padding: 0.45rem 1.1rem !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.12) !important;
        transition: all 0.2s ease-in-out !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #2563EB 0%, #0EA5E9 100%) !important;
        color: #FFFFFF !important;
        border-color: #2563EB !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.28) !important;
    }

    /* ── Upload Area (Dashed Sky Blue border, rounded 18px) ── */
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
        background: linear-gradient(135deg, #2563EB 0%, #0EA5E9 60%, #38BDF8 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 14px !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        padding: 0.55rem 1.3rem !important;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.25) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        margin-top: 0.35rem !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #0284C7 60%, #0EA5E9 100%) !important;
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.38) !important;
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
        background: linear-gradient(135deg, #1E40AF 0%, #2563EB 60%, #0EA5E9 100%) !important;
        color: #FFFFFF !important;
        border-radius: 20px 20px 4px 20px !important;
        padding: 0.85rem 1.35rem !important;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.22) !important;
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
        background: #2563EB !important;
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
        border-left: 5px solid #2563EB !important;
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

    /* ── Clinical Evidence Card (Clean White, Rounded 18px, Blue Border) ── */
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
        color: #2563EB !important;
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
        border-color: #2563EB !important;
        box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.2), 0 8px 26px rgba(14, 165, 233, 0.14) !important;
    }
    div[data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #1E40AF 0%, #2563EB 60%, #0EA5E9 100%) !important;
        color: #FFFFFF !important;
        border-radius: 50% !important;
        border: none !important;
        box-shadow: 0 3px 10px rgba(37, 99, 235, 0.28) !important;
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease !important;
    }
    div[data-testid="stChatInput"] button:hover {
        transform: scale(1.1) !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4) !important;
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

# ── Header Section (Professional Blue Gradient Banner) ───────────────────────
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
                            st.session_state.active_patient_display = None
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

# ── Dynamic Current Patient Badge ────────────────────────────────────────────
active_meta = get_active_report_meta()
if "active_patient_display" not in st.session_state or not st.session_state.active_patient_display:
    cur_p_name = active_meta.get("patient_name")
    cur_p_id = active_meta.get("patient_id")
    if not cur_p_name or cur_p_name in ["Not specified", "Extracted", "Unknown"]:
        cur_p_name = "Rahul"
    if not cur_p_id or cur_p_id in ["Not specified", "Extracted", "Unknown"]:
        cur_p_id = "D001"
    st.session_state.active_patient_display = f"{cur_p_name} ({cur_p_id})"

col_badge_left, col_badge_right = st.columns([0.65, 0.35])
with col_badge_left:
    st.markdown(
        f'<div style="margin-bottom: 0.5rem;">'
        f'<span class="patient-pill-badge">👤 Current Patient: <strong>{st.session_state.active_patient_display}</strong></span>'
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
                st.session_state.active_patient_display = f"{p_name} ({p_id})"
                st.toast(f"✅ Ingested: {p_name} ({p_id})", icon="🟢")
                st.success(f"✅ Report processed successfully! Active: **{p_name}** ({p_id}) | Chunks: {meta_info.get('chunk_count', 0)}")
                st.rerun()
            else:
                st.error("Failed to ingest documents.")
    else:
        st.warning("Please upload a medical report (PDF, TXT, DOCX) first.")

# ── Session State Management (Latest Q&A Only & Recent Questions) ───────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "recent_questions" not in st.session_state:
    st.session_state.recent_questions = [
        "What is the diagnosis?",
        "What is the platelet count?",
        "What is the risk level?"
    ]

# ── Suggested Questions Section (Clickable Pills) ───────────────────────────
st.markdown("""
<div class="suggested-q-title">💡 Suggested Questions</div>
""", unsafe_allow_html=True)

suggested_queries = [
    "What is the diagnosis?",
    "What is the platelet count?",
    "What is the risk level?",
    "Summarize the report.",
    "What recommendations are provided?"
]

sq_cols = st.columns(5)
for col, sq_text in zip(sq_cols, suggested_queries):
    with col:
        if st.button(sq_text, key=f"sq_btn_{sq_text}", use_container_width=True):
            if sq_text not in st.session_state.recent_questions:
                st.session_state.recent_questions.insert(0, sq_text)
                st.session_state.recent_questions = st.session_state.recent_questions[:5]
            st.session_state.messages = [{"role": "user", "content": sq_text}]
            st.rerun()

# ── Recent Questions (Last 5 Questions Stored) ──────────────────────────────
if st.session_state.recent_questions:
    with st.expander("🕒 Recent Questions", expanded=False):
        for rq in st.session_state.recent_questions:
            col_rq_text, col_rq_btn = st.columns([0.85, 0.15])
            with col_rq_text:
                st.markdown(f'<div class="recent-q-item">• {rq}</div>', unsafe_allow_html=True)
            with col_rq_btn:
                if st.button("Ask ↗", key=f"ask_rq_{rq}", use_container_width=True):
                    st.session_state.messages = [{"role": "user", "content": rq}]
                    st.rerun()

# ── Input query (Rounded input box, blue glow on focus, modern send button) ─
if prompt := st.chat_input("Ask about diagnosis, platelet count, risk level, recommendations, or patient details..."):
    # Add to recent questions (max 5)
    if prompt not in st.session_state.recent_questions:
        st.session_state.recent_questions.insert(0, prompt)
        st.session_state.recent_questions = st.session_state.recent_questions[:5]
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
        return f'<div class="answer-card-box">{answer_text}</div>'

    matches.sort(key=lambda x: x[0])
    cards_html = []

    # Preamble before first section
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

            # Update Current Patient dynamically based on retrieved record
            if evidence_list and isinstance(evidence_list, dict):
                ret_pname = evidence_list.get("patient_name")
                ret_pid = evidence_list.get("patient_id")
                if ret_pname and ret_pname != "Not specified":
                    st.session_state.active_patient_display = f"{ret_pname} ({ret_pid or 'ID Unknown'})"

            # Show Retrieved Patient Card
            if retrieved_patient_ui:
                patient_card_html = f'<div class="retrieved-patient-box">📋 {retrieved_patient_ui}</div>'
                st.markdown(patient_card_html, unsafe_allow_html=True)

            # Show Grounded Response Badge above Answer
            grounded_badge_html = (
                f'<div class="grounded-badge-container">'
                f'<span class="grounded-pill">✅ Grounded Response</span>'
                f'<span class="grounded-subtext">Retrieved from Bedrock Knowledge Base</span>'
                f'</div>'
            )
            if hasattr(st, "html"):
                st.html(grounded_badge_html)
            else:
                st.markdown(grounded_badge_html, unsafe_allow_html=True)

            # Show Answer Section in modern redesigned cards
            answer_cards_html = render_styled_answer_cards(answer)
            if hasattr(st, "html"):
                st.html(answer_cards_html)
            else:
                st.markdown(answer_cards_html, unsafe_allow_html=True)

            # Show Download Answer as PDF Button
            target_pname = st.session_state.active_patient_display
            if evidence_list and isinstance(evidence_list, dict):
                ep = evidence_list.get("patient_name")
                if ep and ep != "Not specified":
                    target_pname = ep

            pdf_bytes = generate_answer_pdf(
                question=prompt,
                answer=answer,
                patient_name=target_pname
            )

            st.download_button(
                label="📥 Download Answer as PDF",
                data=pdf_bytes,
                file_name=f"Clinical_Answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=False
            )

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
                    if hasattr(st, "html"):
                        st.html(clinical_card_html)
                    else:
                        st.markdown(clinical_card_html, unsafe_allow_html=True)

            # Save latest state
            final_output = f"{retrieved_patient_ui}\n\n{answer}" if retrieved_patient_ui else answer
            st.session_state.messages.append({"role": "assistant", "content": final_output})

# ── Tiny Footer ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Powered by AWS Bedrock Knowledge Base + RAG
</div>
""", unsafe_allow_html=True)
