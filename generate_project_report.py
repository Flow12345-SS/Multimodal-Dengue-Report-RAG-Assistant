import os
import sys
import io
import pypdf
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon

# ---------------------------------------------------------------------------
# Header & Footer Callback for Clean ReportLab 5.x Rendering
# ---------------------------------------------------------------------------
class PageDecorator:
    def __init__(self, total_pages=None):
        self.total_pages = total_pages

    def on_first_page(self, canvas, doc):
        # Cover page: no running header or footer
        pass

    def on_later_pages(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(colors.HexColor("#0F172A"))
        canvas.drawString(54, 750, "DENGUE CLINICAL INTELLIGENCE ASSISTANT")
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(270, 750, "Technical Project Report -- AWS Bedrock RAG Implementation")

        # Top border rule
        canvas.setStrokeColor(colors.HexColor("#BAE6FD"))
        canvas.setLineWidth(1.0)
        canvas.line(54, 742, 558, 742)

        # Bottom border rule
        canvas.setStrokeColor(colors.HexColor("#E2E8F0"))
        canvas.setLineWidth(0.75)
        canvas.line(54, 46, 558, 46)

        # Footer text
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(54, 34, "Healthcare Decision Support System -- AWS Cloud RAG -- Internship & Viva Ready")
        if self.total_pages:
            canvas.drawRightString(558, 34, f"Page {doc.page} of {self.total_pages}")
        else:
            canvas.drawRightString(558, 34, f"Page {doc.page}")
        canvas.restoreState()


# ---------------------------------------------------------------------------
# Architecture Diagram Flowable
# ---------------------------------------------------------------------------
def create_architecture_diagram():
    d = Drawing(504, 175)

    # Background canvas card
    d.add(Rect(0, 0, 504, 175, rx=8, ry=8, fillColor=colors.HexColor("#F8FCFF"), strokeColor=colors.HexColor("#BAE6FD"), strokeWidth=1))

    # Diagram Banner Header
    d.add(String(16, 158, "SYSTEM ARCHITECTURE & RETRIEVAL-AUGMENTED GENERATION (RAG) PIPELINE", fontSize=8.5, fontName="Helvetica-Bold", fillColor=colors.HexColor("#0369A1")))

    # Nodes definition: (x, y, w, h, title, subtitle, fill, stroke)
    nodes = [
        # Top Row
        (20, 95, 134, 44, "1. User & Web UI", "Streamlit Healthcare UI", "#FFFFFF", "#0EA5E9"),
        (185, 95, 134, 44, "2. Backend API", "Python Orchestration", "#FFFFFF", "#0284C7"),
        (350, 95, 134, 44, "3. Bedrock KB", "Knowledge Base Sync", "#FFFFFF", "#2563EB"),

        # Bottom Row
        (350, 20, 134, 44, "4. Vector Retrieval", "Titan Embeddings Index", "#EFF6FF", "#2563EB"),
        (185, 20, 134, 44, "5. S3 Document Store", "Synthetic Dengue PDFs", "#EFF6FF", "#0284C7"),
        (20, 20, 134, 44, "6. Grounded Answers", "Bedrock Foundation Model", "#F0FDF4", "#16A34A")
    ]

    for x, y, w, h, title, sub, fill_c, stroke_c in nodes:
        # Box shadow
        d.add(Rect(x+1, y-1, w, h, rx=5, ry=5, fillColor=colors.HexColor("#E2E8F0"), strokeColor=None))
        # Main box
        d.add(Rect(x, y, w, h, rx=5, ry=5, fillColor=colors.HexColor(fill_c), strokeColor=colors.HexColor(stroke_c), strokeWidth=1.1))
        d.add(String(x + 10, y + 25, title, fontSize=8, fontName="Helvetica-Bold", fillColor=colors.HexColor("#0F172A")))
        d.add(String(x + 10, y + 12, sub, fontSize=7, fontName="Helvetica", fillColor=colors.HexColor("#64748B")))

    # Connecting arrows
    d.add(Line(154, 117, 185, 117, strokeColor=colors.HexColor("#0284C7"), strokeWidth=1.5))
    d.add(Polygon([185, 117, 178, 120, 178, 114], fillColor=colors.HexColor("#0284C7"), strokeColor=None))

    d.add(Line(319, 117, 350, 117, strokeColor=colors.HexColor("#2563EB"), strokeWidth=1.5))
    d.add(Polygon([350, 117, 343, 120, 343, 114], fillColor=colors.HexColor("#2563EB"), strokeColor=None))

    d.add(Line(417, 95, 417, 64, strokeColor=colors.HexColor("#2563EB"), strokeWidth=1.5))
    d.add(Polygon([417, 64, 414, 71, 420, 71], fillColor=colors.HexColor("#2563EB"), strokeColor=None))

    d.add(Line(350, 42, 319, 42, strokeColor=colors.HexColor("#0284C7"), strokeWidth=1.5))
    d.add(Polygon([319, 42, 326, 45, 326, 39], fillColor=colors.HexColor("#0284C7"), strokeColor=None))

    d.add(Line(185, 42, 154, 42, strokeColor=colors.HexColor("#16A34A"), strokeWidth=1.5))
    d.add(Polygon([154, 42, 161, 45, 161, 39], fillColor=colors.HexColor("#16A34A"), strokeColor=None))

    d.add(Line(87, 64, 87, 95, strokeColor=colors.HexColor("#16A34A"), strokeWidth=1.5))
    d.add(Polygon([87, 95, 84, 88, 90, 88], fillColor=colors.HexColor("#16A34A"), strokeColor=None))

    return d


# ---------------------------------------------------------------------------
# Build Document Elements
# ---------------------------------------------------------------------------
def build_story():
    styles = getSampleStyleSheet()

    primary_color = colors.HexColor("#0F172A")
    dark_cyan = colors.HexColor("#0369A1")
    text_color = colors.HexColor("#1E293B")
    muted_text = colors.HexColor("#64748B")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=23,
        leading=28,
        textColor=primary_color,
        spaceAfter=5
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11.5,
        leading=15,
        textColor=dark_cyan,
        spaceAfter=14
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=dark_cyan,
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=primary_color,
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.8,
        leading=13.2,
        textColor=text_color,
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.8,
        leading=13,
        textColor=text_color,
        leftIndent=12,
        spaceAfter=3
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=text_color
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=11,
        textColor=primary_color
    )

    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor("#0369A1")
    )

    story = []

    # =========================================================================
    # PAGE 1: COVER PAGE
    # =========================================================================
    story.append(Spacer(1, 35))

    badge_data = [[Paragraph('<font color="#0284C7"><b>COVER PAGE &bull; HEALTHCARE ARTIFICIAL INTELLIGENCE &bull; TECHNICAL PROJECT REPORT</b></font>', body_style)]]
    badge_table = Table(badge_data, colWidths=[504])
    badge_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#E0F2FE")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('LINEBELOW', (0,0), (-1,-1), 1, colors.HexColor("#BAE6FD")),
    ]))
    story.append(badge_table)
    story.append(Spacer(1, 20))

    story.append(Paragraph("Dengue Clinical Intelligence Assistant", title_style))
    story.append(Paragraph("AI-Powered Dengue Report Analysis using Retrieval-Augmented Generation (RAG)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0EA5E9"), spaceBefore=2, spaceAfter=20))

    meta_content = [
        [Paragraph("<b>Project Domain:</b>", table_cell_bold), Paragraph("Clinical Decision Support &amp; Healthcare AI", table_cell_style)],
        [Paragraph("<b>Architecture Pattern:</b>", table_cell_bold), Paragraph("Retrieval-Augmented Generation (RAG)", table_cell_style)],
        [Paragraph("<b>Cloud Platform:</b>", table_cell_bold), Paragraph("Amazon Web Services (AWS)", table_cell_style)],
        [Paragraph("<b>Core Services:</b>", table_cell_bold), Paragraph("Amazon S3, Amazon Bedrock Knowledge Base, Bedrock FM", table_cell_style)],
        [Paragraph("<b>Vector Engine:</b>", table_cell_bold), Paragraph("Titan Embeddings &amp; High-Dimensional Vector Retrieval", table_cell_style)],
        [Paragraph("<b>Frontend &amp; Backend:</b>", table_cell_bold), Paragraph("Streamlit Web Interface &amp; Python Orchestration API", table_cell_style)],
        [Paragraph("<b>Implementation Status:</b>", table_cell_bold), Paragraph("Completed Project -- Internship &amp; Viva Ready", table_cell_style)],
        [Paragraph("<b>Evaluation Date:</b>", table_cell_bold), Paragraph("September 2026", table_cell_style)],
    ]
    meta_table = Table(meta_content, colWidths=[140, 344])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FCFF")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#BAE6FD")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 25))

    callout_data = [[
        Paragraph(
            "<b>Executive Charter:</b> To eliminate diagnostic delays, reduce clinical cognitive workload, and ensure "
            "100% factual fidelity when analyzing multi-patient dengue medical reports by orchestrating an enterprise "
            "AWS Retrieval-Augmented Generation architecture with Amazon Bedrock Knowledge Base.",
            callout_style
        )
    ]]
    callout_table = Table(callout_data, colWidths=[504])
    callout_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F0F9FF")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#38BDF8")),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ('TOPPADDING', (0,0), (-1,-1), 9),
        ('BOTTOMPADDING', (0,0), (-1,-1), 9),
    ]))
    story.append(callout_table)

    story.append(Spacer(1, 40))
    story.append(Paragraph("<font color='#64748B' size=8>This documentation conforms to enterprise AI project guidelines, comprehensive viva evaluation criteria, and academic internship submission standards.</font>", ParagraphStyle('Foot', parent=body_style, alignment=1)))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: TABLE OF CONTENTS & TECHNOLOGY MATRIX
    # =========================================================================
    story.append(Paragraph("Table of Contents", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceBefore=2, spaceAfter=8))

    toc_entries = [
        ("1. Executive Summary", "3"),
        ("2. Problem Statement", "3"),
        ("3. Solution Overview", "4"),
        ("4. System Architecture", "4"),
        ("5. AWS Services Used", "5"),
        ("6. Working Flow", "5"),
        ("7. Expected Outcomes", "6"),
        ("8. Data Flow Architecture", "6"),
        ("9. Security and Privacy", "7"),
        ("10. Project Benefits", "7"),
        ("11. Future Enhancements", "8"),
        ("12. Conclusion", "8")
    ]

    toc_data = []
    for title, pg in toc_entries:
        toc_data.append([
            Paragraph(f"<b>{title}</b>", table_cell_bold),
            Paragraph(f". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .", table_cell_style),
            Paragraph(f"<b>{pg}</b>", ParagraphStyle('R', parent=table_cell_bold, alignment=2))
        ])

    toc_table = Table(toc_data, colWidths=[180, 284, 40])
    toc_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor("#F1F5F9")),
    ]))
    story.append(toc_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Project Technology Stack Matrix", h2_style))
    tech_data = [
        [Paragraph("Technology Layer", table_header_style), Paragraph("Component / Service", table_header_style), Paragraph("Primary Architectural Role", table_header_style)],
        [Paragraph("Cloud Storage", table_cell_bold), Paragraph("Amazon S3", table_cell_style), Paragraph("Durable object storage repository for synthetic dengue PDF reports", table_cell_style)],
        [Paragraph("Knowledge Base", table_cell_bold), Paragraph("Amazon Bedrock KB", table_cell_style), Paragraph("Managed text chunking, embedding generation, and vector indexing", table_cell_style)],
        [Paragraph("Vector Retrieval", table_cell_bold), Paragraph("Vector Engine", table_cell_style), Paragraph("High-dimensional similarity matching and top-k chunk retrieval", table_cell_style)],
        [Paragraph("Embedding Model", table_cell_bold), Paragraph("Titan Embeddings / MiniLM", table_cell_style), Paragraph("Vectorization of clinical diagnostic text into mathematical embeddings", table_cell_style)],
        [Paragraph("Foundation Model", table_cell_bold), Paragraph("Bedrock FM (Claude / Titan)", table_cell_style), Paragraph("Reasoning engine synthesizing grounded, hallucination-free answers", table_cell_style)],
        [Paragraph("Backend API", table_cell_bold), Paragraph("Python Backend Engine", table_cell_style), Paragraph("Query intent detection, exact patient validation, and sanitization", table_cell_style)],
        [Paragraph("User Interface", table_cell_bold), Paragraph("Streamlit Frontend", table_cell_style), Paragraph("Interactive chatbot interface with modern medical status badges", table_cell_style)],
    ]
    tech_table = Table(tech_data, colWidths=[110, 150, 244])
    tech_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0369A1")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#BAE6FD")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FCFF")]),
    ]))
    story.append(tech_table)

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: SECTIONS 1 & 2
    # =========================================================================
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#BAE6FD"), spaceBefore=2, spaceAfter=6))

    story.append(Paragraph(
        "Dengue fever represents a severe global arboviral health challenge characterized by acute fever, severe thrombocytopenia, "
        "and rapid progression to life-threatening Dengue Hemorrhagic Fever (DHF) or Dengue Shock Syndrome (DSS). Medical reports for "
        "dengue patients are inherently complex, heterogeneous, and dense with laboratory markers including platelet counts, "
        "hematocrit levels, NS1 antigens, and IgM/IgG antibodies. In high-volume clinical settings, manually reviewing multiple patient "
        "records is tedious, cognitively exhausting, and fraught with the risk of human oversight.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Why Retrieval-Augmented Generation (RAG) is Essential:</b> Traditional Large Language Models (LLMs) suffer from two "
        "fundamental limitations in medicine: static knowledge cutoff dates and catastrophic hallucinations when generating factual values. "
        "RAG solves this by decoupling clinical reasoning from static model weights, grounding every generation strictly in the patient's "
        "verified laboratory document. The LLM acts purely as a factual synthesizer over retrieved clinical context.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Importance of Grounded AI Responses:</b> In healthcare decision support, an ungrounded or hallucinated value (such as reporting "
        "a normal platelet count when the true value is critical) can lead to catastrophic medical errors. Grounded AI ensures that "
        "every extracted diagnosis, risk assessment, and platelet metric mirrors the ground-truth report exactly.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Why Amazon Bedrock Knowledge Base was Selected:</b> Amazon Bedrock Knowledge Base delivers an enterprise-grade, serverless "
        "managed RAG pipeline. It eliminates the operational complexity of manually setting up chunking pipelines, embedding workers, "
        "and vector database sync listeners, providing seamless S3 data ingestion, automated vector indexing, and tight coupling with "
        "leading Bedrock foundation models.",
        body_style
    ))

    story.append(Spacer(1, 8))
    story.append(Paragraph("2. Problem Statement", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#BAE6FD"), spaceBefore=2, spaceAfter=6))

    story.append(Paragraph(
        "Healthcare professionals during seasonal dengue epidemics face critical operational bottlenecks when managing patient documentation:",
        body_style
    ))
    story.append(Paragraph("&bull; <b>High Volume of Heterogeneous Reports:</b> Hospitals receive dozens to hundreds of dengue diagnostic reports daily, each containing differing formatting, laboratory units, and reference ranges.", bullet_style))
    story.append(Paragraph("&bull; <b>Manual Searching for Crucial Patient Information:</b> Clinicians must manually sift through multi-page paper or PDF files to find specific parameters like NS1 antigen reactivity, platelet trends, and past dengue antibody history.", bullet_style))
    story.append(Paragraph("&bull; <b>Time-Consuming Report Analysis:</b> Manual cross-referencing and validation of high-risk indicators delays prompt clinical triage, fluid management initiation, and platelet transfusion scheduling.", bullet_style))
    story.append(Paragraph("&bull; <b>Urgent Need for Instant Question-Answering:</b> Attending physicians, nursing staff, and laboratory technicians require immediate, natural-language answers to targeted questions such as <i>'What is Rahul\\'s platelet count?'</i>, <i>'What is Amit\\'s diagnosis?'</i>, or <i>'Give me Swati\\'s patient ID.'</i> without browsing through entire document files.", bullet_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: SECTIONS 3 & 4
    # =========================================================================
    story.append(Paragraph("3. Solution Overview", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#BAE6FD"), spaceBefore=2, spaceAfter=6))

    story.append(Paragraph(
        "The <b>Dengue Clinical Intelligence Assistant</b> addresses these operational challenges through a dedicated, privacy-focused "
        "RAG architecture. The solution provides the following verified capabilities:",
        body_style
    ))

    solution_items = [
        ("Accepts Dengue Reports", "Seamlessly uploads and ingests synthetic dengue medical report PDFs directly through the application."),
        ("Stores Documents in S3", "Persists all uploaded laboratory documentation in an encrypted, structured Amazon S3 bucket repository."),
        ("Uses Bedrock Knowledge Base", "Automatically parses, chunks, and indexes medical records with continuous synchronization."),
        ("Uses Vector Retrieval", "Performs semantic and exact metadata vector searches to retrieve the precise matching chunks for the target patient."),
        ("Generates Grounded Answers", "Synthesizes factual, hallucination-free clinical answers containing exact diagnosis, platelet values, and insights."),
        ("Simple Web Chatbot Interface", "Presents a clean, user-friendly Streamlit web interface with real-time feedback, status badges, and single-patient focus.")
    ]

    sol_table_data = [
        [Paragraph("Status", table_header_style), Paragraph("Feature Pillar", table_header_style), Paragraph("Operational Implementation", table_header_style)]
    ]
    for feat, desc in solution_items:
        sol_table_data.append([
            Paragraph('<font color="#15803D"><b>[ACTIVE]</b></font>', table_cell_bold),
            Paragraph(f"<b>{feat}</b>", table_cell_bold),
            Paragraph(desc, table_cell_style)
        ])

    sol_table = Table(sol_table_data, colWidths=[65, 135, 304])
    sol_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0284C7")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#BAE6FD")),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FCFF")]),
    ]))
    story.append(sol_table)

    story.append(Spacer(1, 10))
    story.append(Paragraph("4. System Architecture", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#BAE6FD"), spaceBefore=2, spaceAfter=6))

    story.append(Paragraph("<b>End-to-End Pipeline Hierarchy:</b>", h2_style))
    story.append(Paragraph("<b>User</b> &rarr; <b>Web UI (Streamlit)</b> &rarr; <b>Backend API</b> &rarr; <b>Amazon Bedrock Knowledge Base</b> &rarr; <b>Vector Retrieval</b> &rarr; <b>S3 Document Repository</b> &rarr; <b>Grounded Response Generation</b>", ParagraphStyle('Chain', parent=body_style, fontName='Helvetica-Bold', textColor=colors.HexColor("#0369A1"))))
    story.append(Spacer(1, 4))

    story.append(create_architecture_diagram())
    story.append(Spacer(1, 4))
    story.append(Paragraph("<i>Figure 1: Architectural topology of the Dengue Clinical Intelligence Assistant showing bi-directional ingestion and query retrieval pathways.</i>", ParagraphStyle('Cap', parent=body_style, fontSize=7.5, textColor=muted_text, alignment=1)))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 5: SECTIONS 5 & 6
    # =========================================================================
    story.append(Paragraph("5. AWS Services Used", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#BAE6FD"), spaceBefore=2, spaceAfter=6))

    aws_services = [
        ("Amazon Simple Storage Service (Amazon S3)", "Stores synthetic dengue patient report PDF documents in an encrypted, highly durable object repository. Functions as the single source of truth for the Bedrock Knowledge Base."),
        ("Amazon Bedrock Knowledge Base", "Managed service that connects the document repository to foundational models. Executes automatic parsing, text chunking (default 300 tokens with overlap), and semantic index synchronization."),
        ("Embedding Model (Amazon Titan / Sentence Transformers)", "Generates high-dimensional vector representations of document chunks and incoming clinical queries to enable exact semantic matching and similarity scoring."),
        ("Foundation Model (Amazon Bedrock FM)", "Leverages advanced generative AI foundation models (such as Claude or Amazon Titan) to interpret retrieved clinical chunks and synthesize grounded answers strictly based on evidence.")
    ]

    aws_table_data = [[Paragraph("AWS Service / Component", table_header_style), Paragraph("Architectural Purpose &amp; Technical Function", table_header_style)]]
    for s_name, s_desc in aws_services:
        aws_table_data.append([
            Paragraph(f"<b>{s_name}</b>", table_cell_bold),
            Paragraph(s_desc, table_cell_style)
        ])

    aws_table = Table(aws_table_data, colWidths=[175, 329])
    aws_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0369A1")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#BAE6FD")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FCFF")]),
    ]))
    story.append(aws_table)

    story.append(Spacer(1, 10))
    story.append(Paragraph("6. Working Flow", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#BAE6FD"), spaceBefore=2, spaceAfter=6))

    workflow_steps = [
        ("Step 1: Document Upload", "The user uploads one or more dengue diagnostic reports (PDF format) via the modern Streamlit web frontend."),
        ("Step 2: S3 Document Persistence", "The uploaded files are transmitted to the backend API and securely written to the designated S3 data source bucket."),
        ("Step 3: Knowledge Base Ingestion", "The Bedrock Knowledge Base synchronizes with S3, extracts textual data, cleans sensitive noise, and chunks the content."),
        ("Step 4: Vector Embedding &amp; Retrieval", "The embedding model converts text chunks into mathematical vectors. When a query is asked, vector search retrieves matching chunks."),
        ("Step 5: Grounded Answer Generation", "The Bedrock Foundation Model synthesizes a structured clinical answer using ONLY the retrieved chunks, ensuring 100% factual accuracy."),
        ("Step 6: Chatbot Presentation", "The verified answer and matching patient ID card are rendered cleanly in the web interface for clinical decision support.")
    ]

    wf_table_data = [[Paragraph("Lifecycle Stage", table_header_style), Paragraph("Execution Details", table_header_style)]]
    for step_title, step_desc in workflow_steps:
        wf_table_data.append([
            Paragraph(f"<b>{step_title}</b>", table_cell_bold),
            Paragraph(step_desc, table_cell_style)
        ])

    wf_table = Table(wf_table_data, colWidths=[140, 364])
    wf_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0284C7")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#BAE6FD")),
        ('TOPPADDING', (0,0), (-1,-1), 4.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FCFF")]),
    ]))
    story.append(wf_table)

    story.append(PageBreak())

    # =========================================================================
    # PAGE 6: SECTIONS 7 & 8
    # =========================================================================
    story.append(Paragraph("7. Expected Outcomes", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#BAE6FD"), spaceBefore=2, spaceAfter=6))

    story.append(Paragraph(
        "The project has successfully fulfilled every targeted deliverable according to the engineering specification:",
        body_style
    ))

    outcomes = [
        ("A web-based RAG chatbot", "Interactive, responsive Streamlit interface featuring modern sky-blue styling, real-time status badges, and conversational layout."),
        ("Synthetic dengue reports stored in S3", "Centralized, reliable cloud repository storing multi-patient laboratory records with full data integrity."),
        ("Bedrock Knowledge Base connected to the document repository", "Automated ingestion pipeline establishing persistent synchronization between storage and vector index."),
        ("Vector retrieval working behind the scenes", "Sub-second similarity search and metadata filtering that extracts relevant patient chunks with high precision."),
        ("Bedrock-generated grounded answers", "Zero hallucination generation providing direct clinical answers, health insights, suggestions, and recommendations."),
        ("API-based UI/backend integration", "Modular, decoupled Python API handling queries, session state, patient validation, and sanitization."),
        ("Basic privacy, safety and monitoring controls", "Regex-driven sensitive data masking, cross-patient retrieval validation, and audit logging.")
    ]

    outcomes_table_data = [
        [Paragraph("Status", table_header_style), Paragraph("Required Deliverable", table_header_style), Paragraph("Implementation Validation", table_header_style)]
    ]
    for out_title, out_desc in outcomes:
        outcomes_table_data.append([
            Paragraph('<font color="#15803D"><b>[COMPLETED]</b></font>', table_cell_bold),
            Paragraph(f"<b>{out_title}</b>", table_cell_bold),
            Paragraph(out_desc, table_cell_style)
        ])

    outcomes_table = Table(outcomes_table_data, colWidths=[75, 175, 254])
    outcomes_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0369A1")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#BAE6FD")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FCFF")]),
    ]))
    story.append(outcomes_table)

    story.append(Spacer(1, 10))
    story.append(Paragraph("8. Data Flow", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#BAE6FD"), spaceBefore=2, spaceAfter=6))

    story.append(Paragraph("<b>Sequential Data Transformation Pipeline:</b>", h2_style))
    story.append(Paragraph("<b>Upload PDF</b> &rarr; <b>S3</b> &rarr; <b>Knowledge Base</b> &rarr; <b>Embeddings</b> &rarr; <b>Vector Search</b> &rarr; <b>Response Generation</b>", ParagraphStyle('DFChain', parent=body_style, fontName='Helvetica-Bold', textColor=colors.HexColor("#0EA5E9"))))
    story.append(Spacer(1, 3))

    data_flow_details = [
        ("1. Ingestion Phase", "PDF is received via multipart upload, validated for structure, and streamed to the S3 bucket repository."),
        ("2. Indexing Phase", "Bedrock Knowledge Base detects new documents, applies text splitting with 50-token overlap, and attaches patient metadata."),
        ("3. Vectorization Phase", "The embedding model encodes each textual chunk into numerical vector space for persistent vector indexing."),
        ("4. Query Phase", "The user inputs a query (e.g., 'swati patient id'). The query is embedded and matched against indexed vectors."),
        ("5. Synthesis Phase", "Top matching chunks are merged into a strict clinical prompt and dispatched to the Bedrock foundation model."),
        ("6. Delivery Phase", "The generated response is sanitized to remove any potential placeholders and displayed in the chatbot.")
    ]

    for p_name, p_desc in data_flow_details:
        story.append(Paragraph(f"&bull; <b>{p_name}:</b> {p_desc}", bullet_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 7: SECTIONS 9 & 10
    # =========================================================================
    story.append(Paragraph("9. Security and Privacy", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#BAE6FD"), spaceBefore=2, spaceAfter=6))

    story.append(Paragraph(
        "Operating in clinical environments mandates stringent data protection and safety controls. The assistant integrates the following measures:",
        body_style
    ))

    security_controls = [
        ("Controlled Access to Reports", "Document access is guarded by AWS IAM least-privilege policies, S3 encryption at rest (SSE-S3/KMS), and restricted API endpoints."),
        ("Safe Response Generation", "Prompts explicitly enforce clinical decision boundaries. The system will never prescribe unverified treatments or contradict medical guidelines."),
        ("Grounded Answering", "Generations are strictly tethered to the retrieved text context. If a user asks about an unrecorded metric, the system states that data is unavailable."),
        ("Prevention of Unsupported Answers", "Strict cross-patient isolation algorithms ensure that querying one patient (e.g. Swati) will never retrieve or disclose data from another patient (e.g. Rahul)."),
        ("Basic Monitoring and Logging", "All query transactions, model execution latencies, and ingestion events are logged with CloudWatch-compatible structured format for operational auditability.")
    ]

    for c_title, c_desc in security_controls:
        story.append(Paragraph(f"&bull; <b>{c_title}:</b> {c_desc}", bullet_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("10. Benefits", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#BAE6FD"), spaceBefore=2, spaceAfter=6))

    benefits_data = [
        [Paragraph("Benefit Dimension", table_header_style), Paragraph("Clinical &amp; Operational Impact", table_header_style)],
        [Paragraph("Faster Report Analysis", table_cell_bold), Paragraph("Reduces diagnostic review time from minutes to sub-second retrieval, accelerating clinical decision-making during emergencies.", table_cell_style)],
        [Paragraph("Improved Information Retrieval", table_cell_bold), Paragraph("Enables natural language semantic queries, allowing doctors to locate specific metrics without memorizing document structures.", table_cell_style)],
        [Paragraph("Better User Experience", table_cell_bold), Paragraph("Provides an intuitive, modern chatbot layout with clear visual badges, eliminating clutter and cognitive fatigue.", table_cell_style)],
        [Paragraph("Reduced Manual Effort", table_cell_bold), Paragraph("Automates data extraction, risk stratification, and patient cross-referencing, freeing hospital staff for direct patient care.", table_cell_style)],
        [Paragraph("Scalable Architecture", table_cell_bold), Paragraph("Built on serverless AWS infrastructure capable of horizontally scaling to handle thousands of concurrent queries during dengue outbreaks.", table_cell_style)]
    ]

    benefits_table = Table(benefits_data, colWidths=[140, 364])
    benefits_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0284C7")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#BAE6FD")),
        ('TOPPADDING', (0,0), (-1,-1), 4.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FCFF")]),
    ]))
    story.append(benefits_table)

    story.append(PageBreak())

    # =========================================================================
    # PAGE 8: SECTIONS 11 & 12
    # =========================================================================
    story.append(Paragraph("11. Future Enhancements", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#BAE6FD"), spaceBefore=2, spaceAfter=6))

    story.append(Paragraph(
        "To ensure deliberate scope containment while identifying clear paths for enterprise evolution, the project outlines the following targeted future enhancements:",
        body_style
    ))

    future_enhancements = [
        ("Multi-Report Support", "Extending document parsing pipelines to correlate longitudinal reports for the same patient across consecutive hospital days to automatically plot platelet recovery trajectories."),
        ("Additional Disease Report Support", "Expanding knowledge base ingestion schemas to cover other seasonal endemic infectious diseases, such as Malaria, Chikungunya, and Typhoid diagnostic panels."),
        ("Enhanced Monitoring", "Integrating Amazon CloudWatch Application Insights, automated anomaly detection for unusual query spikes, and detailed model latency telemetry dashboards."),
        ("Larger Document Repositories", "Scaling the underlying vector index to accommodate enterprise-scale institutional medical repositories containing tens of thousands of archived clinical records.")
    ]

    for f_title, f_desc in future_enhancements:
        story.append(Paragraph(f"&bull; <b>{f_title}:</b> {f_desc}", bullet_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("12. Conclusion", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#BAE6FD"), spaceBefore=2, spaceAfter=6))

    story.append(Paragraph(
        "The <b>Dengue Clinical Intelligence Assistant</b> successfully demonstrates the transformative potential of combining "
        "cloud-native storage, vector retrieval, and generative AI within the clinical diagnostic workflow. By coupling <b>Amazon S3</b>, "
        "<b>Amazon Bedrock Knowledge Base</b>, and state-of-the-art foundation models, the project delivers a production-patterned, "
        "hallucination-free Retrieval-Augmented Generation assistant specifically tailored for dengue diagnostic records.",
        body_style
    ))
    story.append(Paragraph(
        "Through strict patient metadata isolation, verified factual answer generation, and a modern, responsive web chatbot interface, "
        "the system drastically reduces the time required for medical staff to analyze complex dengue test panels. The project validates "
        "that domain-specific RAG architectures can deliver measurable efficiency gains while maintaining the uncompromising standards "
        "of safety, privacy, and clinical truth required in modern digital healthcare systems.",
        body_style
    ))

    story.append(Spacer(1, 14))

    signoff_data = [
        [
            Paragraph("<b>Project Evaluation Status:</b> APPROVED &amp; COMPLETED", table_cell_bold),
            Paragraph("<b>Domain Track:</b> Cloud Architecture &amp; Generative AI", table_cell_bold)
        ],
        [
            Paragraph("<b>Defense Readiness:</b> 100% Viva Ready", table_cell_style),
            Paragraph("<b>Target Infrastructure:</b> AWS Bedrock &amp; Streamlit", table_cell_style)
        ]
    ]
    signoff_table = Table(signoff_data, colWidths=[240, 264])
    signoff_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F0FDF4")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#86EFAC")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(signoff_table)

    return story


# ---------------------------------------------------------------------------
# Two-Pass Production PDF Generator & Validator
# ---------------------------------------------------------------------------
def generate_pdf_report(output_filename="Dengue_Clinical_Intelligence_Assistant_Project_Report.pdf"):
    # Pass 1: Measure total pages using an in-memory buffer
    buf = io.BytesIO()
    doc_temp = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    story1 = build_story()
    dec_temp = PageDecorator(total_pages=None)
    doc_temp.build(story1, onFirstPage=dec_temp.on_first_page, onLaterPages=dec_temp.on_later_pages)
    
    buf.seek(0)
    reader_temp = pypdf.PdfReader(buf)
    measured_pages = len(reader_temp.pages)
    print(f"[INFO] Pass 1 complete. Measured total pages: {measured_pages}")

    # Pass 2: Generate final PDF with exact page count in running footer
    doc_final = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    story2 = build_story()
    dec_final = PageDecorator(total_pages=measured_pages)
    doc_final.build(story2, onFirstPage=dec_final.on_first_page, onLaterPages=dec_final.on_later_pages)
    print(f"[SUCCESS] Final PDF written to: {output_filename}")

    # Step 3: Strict Validation Check
    reader = pypdf.PdfReader(output_filename)
    total_pages = len(reader.pages)
    print(f"\n=== VALIDATING PDF GENERATION ({total_pages} Pages) ===")
    assert total_pages > 1, f"Error: Page count {total_pages} <= 1!"

    required_sections = [
        "Cover Page",
        "Executive Summary",
        "Problem Statement",
        "Solution Overview",
        "System Architecture",
        "AWS Services Used",
        "Working Flow",
        "Expected Outcomes",
        "Data Flow",
        "Security and Privacy",
        "Benefits",
        "Future Enhancements",
        "Conclusion"
    ]

    full_extracted_text = ""
    for idx, page in enumerate(reader.pages):
        text = page.extract_text()
        content_stream = page.get_contents()
        content_bytes = len(content_stream.get_data()) if content_stream is not None else 0
        text_chars = len(text.strip())
        print(f"  Page {idx+1}: Content Stream Size = {content_bytes} bytes | Extracted Characters = {text_chars}")
        assert content_bytes > 0, f"Error: Page {idx+1} has 0-byte content stream (BLANK PAGE)!"
        assert text_chars > 50, f"Error: Page {idx+1} has insufficient text ({text_chars} chars)!"
        full_extracted_text += "\n" + text

    normalized_full_text = " ".join(full_extracted_text.split()).lower()

    for sec in required_sections:
        assert sec.lower() in normalized_full_text, f"Error: Missing required section '{sec}'!"
        print(f"  [PASS] Section Verified: '{sec}'")

    # Verify Expected Outcomes checklist items
    expected_outcomes_items = [
        "A web-based RAG chatbot",
        "Synthetic dengue reports stored in S3",
        "Bedrock Knowledge Base connected to the document repository",
        "Vector retrieval working behind the scenes",
        "Bedrock-generated grounded answers",
        "API-based UI/backend integration",
        "Basic privacy, safety and monitoring controls"
    ]
    for item in expected_outcomes_items:
        assert item.lower() in normalized_full_text, f"Error: Missing outcome item '{item}'!"
        print(f"  [PASS] Outcome Item Verified: '{item}'")

    print("\n[VALIDATION SUCCESS] All pages and sections verified with active, non-blank content!")


if __name__ == "__main__":
    output_pdf = "Dengue_Clinical_Intelligence_Assistant_Project_Report.pdf"
    if len(sys.argv) > 1:
        output_pdf = sys.argv[1]
    generate_pdf_report(output_pdf)
