#!/usr/bin/env python3
"""
Build the final Manipal University Jaipur PBL-3 report from the official
template cover (logo + title block), filled with project content.

Outputs:
  reports/PBL3_Project_Report.docx
  reports/PBL3_Project_Report.pdf
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, Twips
from fpdf import FPDF
from lxml import etree

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
TEMPLATE = Path(
    "/Users/gagan/Downloads/PBL-3 Report Format Nov 2026 (CyberSecurity).doc.docx"
)
DOCX_OUT = ROOT / "PBL3_Project_Report.docx"
PDF_OUT = ROOT / "PBL3_Project_Report.pdf"
LOGO = ASSETS / "muj_logo.png"
MARGIN = 0.7

STUDENT = "Gagneet Singh"
REG_NO = "2430040049"
TITLE = "Multi-Factor Financial Decision System"
GUIDE = "Narendra Singh Yadav"
DEPT = "Department of Information Technology"
UNIV = "MANIPAL UNIVERSITY JAIPUR"
PLACE = "RAJASTHAN, INDIA"
DATE = "Nov 2026"
DEGREE_LINE = "Computer Science and Engineering (Cybersecurity)"
COURSE = "PBL-3 (ICS 3170)"
VERCEL = "https://multi-factor-decision-system.vercel.app"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


# ---------------------------------------------------------------------------
# Diagrams
# ---------------------------------------------------------------------------

def _save_fig(fig, name: str) -> Path:
    path = ASSETS / name
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def ensure_diagrams() -> dict:
    ASSETS.mkdir(parents=True, exist_ok=True)

    # Architecture
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("System Architecture – Multi-Factor Financial Decision System", fontsize=12)

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=color, edgecolor="#333", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)

    box(0.4, 5.4, 4.2, 1.2, "React Terminal (Vercel SPA)\nLive Analysis | Backtesting", "#DDE7F8")
    box(5.4, 5.4, 4.2, 1.2, "FastAPI 0.9.0\n/live /backtest /research /paper /explain", "#F8E8DD")
    box(0.4, 3.5, 3.0, 1.4, "Market Data\nYahoo history\nOptional broker adapter\nSQLite candles", "#E8F5E9")
    box(3.6, 3.5, 3.0, 1.4, "Research Engine\nFeatures + GB model\nSignals + risk\nNext-open backtest", "#FFF3E0")
    box(6.8, 3.5, 2.8, 1.4, "Support\nPaper ledger\nExplainability\nLive scanner", "#F3E5F5")
    box(0.4, 1.4, 3.0, 1.4, "market.sqlite", "#ECEFF1")
    box(3.6, 1.4, 3.0, 1.4, "paper.sqlite", "#ECEFF1")
    box(6.8, 1.4, 2.8, 1.4, "scans.sqlite", "#ECEFF1")
    ax.text(5, 0.55, "Hard invariant: broker_execution = false (no live order routing)",
            ha="center", fontsize=8, style="italic")
    arch = _save_fig(fig, "fig_architecture.png")

    # Dataflow
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    ax.set_title("Research Decision Data Flow (TECHNICAL_MODE)", fontsize=12)
    steps = [
        (0.3, "OHLCV\n(Yahoo/store)"),
        (2.4, "Feature &\nLabel build"),
        (4.5, "GB Classifier\n+ walk-forward"),
        (6.6, "BUY/SELL/HOLD\n+ risk gate"),
        (8.7, "Next-open\nbacktest"),
        (10.5, "Explain /\nAPI / UI"),
    ]
    for i, (x, label) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((x, 1.6), 1.6, 1.5, boxstyle="round,pad=0.03,rounding_size=0.08",
                                    facecolor="#E3F2FD", edgecolor="#1565C0", lw=1.2))
        ax.text(x + 0.8, 2.35, label, ha="center", va="center", fontsize=8)
        if i < len(steps) - 1:
            ax.annotate("", xy=(steps[i + 1][0], 2.35), xytext=(x + 1.6, 2.35),
                        arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.2))
    ax.text(6, 0.7, "Label: 7-day forward return > 1%. Fill at t+1 Open. Verified live tape not available in PBL-3.",
            ha="center", fontsize=8)
    flow = _save_fig(fig, "fig_dataflow.png")

    # Modules
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Module Interaction Overview", fontsize=12)
    modules = [
        (0.4, 4.2, "frontend/\nApp, pages, charts"),
        (3.5, 4.2, "api/\nrouters + settings"),
        (6.6, 4.2, "four_model_pipeline"),
        (0.4, 2.2, "market_data/"),
        (3.5, 2.2, "strategy + risk +\nbacktest + costs"),
        (6.6, 2.2, "paper_trading /\nreasoning /\nlive_analysis"),
        (3.5, 0.4, "SQLite stores\nmarket | paper | scans"),
    ]
    for x, y, t in modules:
        ax.add_patch(FancyBboxPatch((x, y), 2.8, 1.3, boxstyle="round,pad=0.03,rounding_size=0.08",
                                    facecolor="#FFF8E1", edgecolor="#6D4C41", lw=1.1))
        ax.text(x + 1.4, y + 0.65, t, ha="center", va="center", fontsize=8)
    mods = _save_fig(fig, "fig_modules.png")

    # Gantt
    tasks = [
        ("PBL-3: Core ML + signals + costs", 0, 3, "done"),
        ("PBL-3: Walk-forward reliability", 2, 4, "done"),
        ("PBL-3: Market SQLite + freshness labels", 3, 5, "done"),
        ("PBL-3: Paper trading isolation", 4, 6, "done"),
        ("PBL-3: FastAPI + React terminal", 5, 8, "done"),
        ("PBL-3: Explainability + analytics APIs", 7, 9, "done"),
        ("PBL-3: Live scanner + backtest APIs", 8, 10, "done"),
        ("PBL-3: Deployment configs + report", 9, 11, "done"),
        ("PBL-4: Wire UI to real backtest API", 11, 13, "plan"),
        ("PBL-4: Fix weekly-feature causality", 11, 13, "plan"),
        ("PBL-4: Deploy API + VITE_API_BASE_URL", 12, 14, "plan"),
        ("PBL-4: Live feed verification (if licensed)", 13, 16, "plan"),
        ("PBL-4: Restore paper/research UI pages", 14, 16, "plan"),
        ("PBL-4: Auth, CI frontend, holiday calendar", 15, 17, "plan"),
    ]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    colors = {"done": "#2E7D32", "plan": "#F9A825"}
    for i, (name, start, end, kind) in enumerate(tasks):
        ax.barh(i, end - start, left=start, height=0.55, color=colors[kind], edgecolor="black", lw=0.5)
        ax.text(-0.2, i, name, ha="right", va="center", fontsize=7.5)
    ax.set_yticks([])
    ax.set_xlabel("Project weeks (relative index)", fontsize=9)
    ax.set_xlim(-0.5, 18)
    ax.set_ylim(-1, len(tasks))
    ax.set_title("Gantt Chart – PBL-3 Completed vs PBL-4 Planned", fontsize=12)
    ax.axvline(11, color="#1565C0", linestyle="--", lw=1.2)
    ax.text(11.1, len(tasks) - 0.3, "PBL-3 → PBL-4", fontsize=8, color="#1565C0")
    ax.legend(handles=[
        mpatches.Patch(color=colors["done"], label="Completed in PBL-3"),
        mpatches.Patch(color=colors["plan"], label="Planned for PBL-4"),
    ], loc="lower right", fontsize=8)
    ax.invert_yaxis()
    gantt = _save_fig(fig, "fig_gantt.png")

    # Ensure logo extracted
    if not LOGO.exists() and (ASSETS / "template_image1.png").exists():
        from PIL import Image
        Image.open(ASSETS / "template_image1.png").convert("RGBA").save(LOGO)

    return {"architecture": arch, "dataflow": flow, "modules": mods, "gantt": gantt}


# ---------------------------------------------------------------------------
# DOCX helpers
# ---------------------------------------------------------------------------

def set_run_font(run, *, size=12, bold=False, italic=False):
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic


def fmt_para(p, *, align="justify", before=0, after=6, first_line=False):
    pf = p.paragraph_format
    p.alignment = {
        "justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
        "center": WD_ALIGN_PARAGRAPH.CENTER,
        "left": WD_ALIGN_PARAGRAPH.LEFT,
        "right": WD_ALIGN_PARAGRAPH.RIGHT,
    }[align]
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = 1.15
    pf.first_line_indent = Inches(0.25) if first_line and align == "justify" else Pt(0)


def add_body(doc, text, *, align="justify", first_line=True):
    p = doc.add_paragraph()
    fmt_para(p, align=align, first_line=first_line and align == "justify")
    run = p.add_run(text)
    set_run_font(run, size=12)
    return p


def add_h1(doc, text):
    p = doc.add_paragraph()
    fmt_para(p, align="left", before=12, after=8)
    run = p.add_run(text)
    set_run_font(run, size=14, bold=True)
    return p


def add_h2(doc, text):
    p = doc.add_paragraph()
    fmt_para(p, align="left", before=10, after=6)
    run = p.add_run(text)
    set_run_font(run, size=12, italic=True)
    return p


def add_bullet(doc, text):
    p = doc.add_paragraph()
    fmt_para(p, align="justify", after=3)
    p.paragraph_format.left_indent = Inches(0.25)
    run = p.add_run(f"• {text}")
    set_run_font(run, size=12)
    return p


def add_center_heading(doc, text, size=16):
    p = doc.add_paragraph()
    fmt_para(p, align="center", before=6, after=10)
    run = p.add_run(text)
    set_run_font(run, size=size, bold=True)
    return p


def add_caption(doc, text):
    p = doc.add_paragraph()
    fmt_para(p, align="center", before=4, after=10)
    run = p.add_run(text)
    set_run_font(run, size=11, italic=True)
    return p


def _set_cell_border(cell):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        element = OxmlElement(f"w:{edge}")
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), "000000")
        tc_borders.append(element)
    tc_pr.append(tc_borders)


def add_table(doc, caption, rows):
    add_caption(doc, caption)
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    try:
        table.style = "Table Grid"
    except KeyError:
        pass
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.rows[i].cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            run = p.add_run(str(val))
            set_run_font(run, size=10, bold=(i == 0))
            _set_cell_border(cell)
    doc.add_paragraph()


def add_figure(doc, path: Path, caption: str, width=6.2):
    if path.exists():
        doc.add_picture(str(path), width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_caption(doc, caption)


def set_margins(section):
    section.page_width = Inches(8.27)
    section.page_height = Inches(11.69)
    section.left_margin = Inches(MARGIN)
    section.right_margin = Inches(MARGIN)
    section.top_margin = Inches(MARGIN)
    section.bottom_margin = Inches(MARGIN)


def add_page_number(paragraph):
    run = paragraph.add_run()
    set_run_font(run, size=10)
    for kind, extra in [("begin", None), ("instr", " PAGE "), ("separate", None), ("end", None)]:
        if kind == "instr":
            node = OxmlElement("w:instrText")
            node.set(qn("xml:space"), "preserve")
            node.text = extra
            run._r.append(node)
        else:
            node = OxmlElement("w:fldChar")
            node.set(qn("w:fldCharType"), kind)
            run._r.append(node)


def setup_footer(section, start=1):
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p.clear()
    add_page_number(p)
    sect_pr = section._sectPr
    # remove old pgNumType if any
    for child in list(sect_pr):
        if child.tag == qn("w:pgNumType"):
            sect_pr.remove(child)
    pg = OxmlElement("w:pgNumType")
    pg.set(qn("w:start"), str(start))
    sect_pr.append(pg)


def patch_cover_text(doc: Document):
    """Replace placeholders inside the template cover VML/textbox paragraph."""
    cover = doc.paragraphs[1]._p
    nodes = cover.findall(f".//{{{W_NS}}}t")
    texts = [n.text or "" for n in nodes]

    def join_slice(a, b):
        return "".join(texts[a:b + 1])

    # TITLE_OF_PROJECT
    for i in range(len(texts) - 1):
        if texts[i] == "TITLE" and texts[i + 1] == "_OF_PROJECT":
            nodes[i].text = TITLE
            nodes[i + 1].text = ""
            break

    # <Student Name>
    for i in range(len(texts) - 4):
        if join_slice(i, i + 4) == "<Student Name>":
            nodes[i].text = STUDENT
            for j in range(i + 1, i + 5):
                nodes[j].text = ""
            break

    # <Registration Number>
    for i, t in enumerate(texts):
        if t == "<Registration Number>":
            nodes[i].text = REG_NO
            break

    # Guide Name -> keep template wording unless already fine
    for i in range(len(texts) - 2):
        if texts[i] == "Guide " and texts[i + 1] == "N" and texts[i + 2] == "ame":
            nodes[i].text = GUIDE
            nodes[i + 1].text = ""
            nodes[i + 2].text = ""
            break


def clear_after_cover(doc: Document):
    """Keep cover page elements; remove template guideline/body placeholders."""
    body = doc.element.body
    children = list(body)
    # Keep paragraphs until and including "Nov 2026" paragraph, then drop the rest
    # except sectPr at end.
    keep_until = None
    paras = [c for c in children if c.tag == qn("w:p")]
    # Map paragraph element to index in body
    for idx, child in enumerate(children):
        if child.tag != qn("w:p"):
            continue
        text = "".join(child.itertext())
        if "Nov" in text and "2026" in text:
            keep_until = idx
            break
    if keep_until is None:
        raise RuntimeError("Could not locate cover date paragraph in template")

    for child in children[keep_until + 1 :]:
        if child.tag == qn("w:sectPr"):
            continue
        body.remove(child)


def add_page_break(doc):
    p = doc.add_paragraph()
    run = p.add_run()
    run.add_break(WD_BREAK.PAGE)


ABSTRACT_P1 = (
    "This PBL-3 report presents the Multi-Factor Financial Decision System, a research and simulation "
    "software stack for Indian equities. The implemented system combines technical feature engineering, "
    "a gradient-boosting classifier, purged walk-forward evaluation, BUY/SELL/HOLD signal rules, risk gating, "
    "next-open backtesting with Indian equity cost assumptions, isolated paper trading, optional AI "
    "explainability, FastAPI services, and a React operator terminal. Live broker order execution is disabled "
    "by design. Verified live NSE market-tape operation was not achieved during PBL-3; research paths use "
    "delayed or historical data sources."
)
ABSTRACT_P2 = (
    "Work completed in PBL-3 includes the core Python research engine, SQLite persistence for market, paper "
    "and scan data, FastAPI routers for live scanning and backtesting, and a two-tab React interface (Live "
    "Analysis and Backtesting). Automated validation recorded 81 passing pytest cases and a successful "
    f"frontend production build. The frontend is reachable at {VERCEL}. The Backtesting tab currently renders "
    "client-side simulated strategy paths for demonstration and must not be interpreted as verified historical "
    "profitability. Planned PBL-4 work focuses on connecting the UI to the real backtest API, fixing identified "
    "data-leakage risks, and completing backend deployment wiring."
)


def build_docx(diags: dict):
    if not TEMPLATE.exists():
        raise FileNotFoundError(TEMPLATE)

    shutil.copy2(TEMPLATE, DOCX_OUT)
    doc = Document(str(DOCX_OUT))

    # Normalize section margins to 0.7"
    for section in doc.sections:
        set_margins(section)
        # clear footers on front matter
        section.footer.is_linked_to_previous = False
        if section.footer.paragraphs:
            section.footer.paragraphs[0].clear()

    patch_cover_text(doc)
    clear_after_cover(doc)

    # Front matter (no page numbers)
    add_page_break(doc)
    add_center_heading(doc, "ABSTRACT", 16)
    add_body(doc, ABSTRACT_P1)
    add_body(doc, ABSTRACT_P2)

    add_page_break(doc)
    add_center_heading(doc, "LIST OF TABLES", 16)
    for item in [
        "Table 3.1  Hardware and software requirements",
        "Table 3.2  Default TECHNICAL_MODE parameters",
        "Table 3.3  Signal generation rules by strategy mode",
        "Table 4.1  PBL-3 milestone status summary",
        "Table 4.2  Distinction of data/operation modes",
    ]:
        add_body(doc, item, align="left", first_line=False)

    add_page_break(doc)
    add_center_heading(doc, "LIST OF FIGURES", 16)
    for item in [
        "Figure 3.1  System architecture block diagram",
        "Figure 3.2  Research decision data-flow diagram",
        "Figure 3.3  Module interaction overview",
        "Figure 4.1  Deployed Live Analysis interface",
        "Figure 4.2  Deployed Backtesting interface",
        "Figure 4.3  Gantt chart – PBL-3 completed and PBL-4 planned work",
    ]:
        add_body(doc, item, align="left", first_line=False)

    add_page_break(doc)
    add_center_heading(doc, "TABLE OF CONTENTS", 16)
    for item in [
        "Abstract",
        "List of Tables",
        "List of Figures",
        "Chapter 1 – Introduction",
        "    1.1 Introduction",
        "    1.2 Problem Statement",
        "    1.3 Objectives",
        "    1.4 Scope of Project",
        "Chapter 3 – Proposed System Design & Methodology",
        "    3.1 System Architecture",
        "    3.2 Development Environment",
        "    3.3 Methodology",
        "Chapter 4 – Work Done in PBL-3",
        "    4.1 Tasks Achieved So Far",
        "    4.2 Screenshots, sketches and flow diagrams",
        "    4.3 Timeline/Gantt Chart",
        "Chapter 5 – Conclusion and Future Plan",
        "    5.1 Summary of PBL-3 Outcomes",
        "    5.2 Tasks Planned for PBL-4",
        "References",
        "Appendix A – Evidence Notes",
    ]:
        add_body(doc, item, align="left", first_line=False)

    # Body section with page numbers starting at 1
    body_section = doc.add_section()
    set_margins(body_section)
    setup_footer(body_section, start=1)

    # Chapter 1
    add_h1(doc, "Chapter 1 – Introduction")
    add_h2(doc, "1.1 Introduction")
    add_body(
        doc,
        "Financial decision support for equities is frequently built from informal indicator rules or opaque "
        "black-box models. In academic and early-stage engineering settings, such systems often omit realistic "
        "execution assumptions, transaction costs, chronological validation and clear labelling of data freshness. "
        "The Multi-Factor Financial Decision System was developed to address these gaps for Indian National Stock "
        "Exchange (NSE) cash equities in a controlled, non-brokerage research environment.",
    )
    add_body(
        doc,
        "Motivation for the project comes from the need to practise responsible quantitative software engineering: "
        "separating model probability from probability of profit, preventing trivial look-ahead where possible, "
        "and refusing to route live broker orders. Applications include classroom demonstration of ML-assisted "
        "signal research, paper-trading education, and structured comparison of technical strategy modes. Advantages "
        "of the implemented approach include inspectable Python modules, SQLite persistence, FastAPI access for "
        "integration, and a React terminal with explicit simulation/safety badges.",
    )
    add_h2(doc, "1.2 Problem Statement")
    add_body(
        doc,
        "The problem addressed in PBL-3 is to design and implement an end-to-end software system that generates "
        "BUY, SELL and HOLD decisions for selected NSE equities using technical features and a supervised classifier, "
        "evaluates those decisions with purged walk-forward methods and cost-aware next-open backtests, persists "
        "market and paper-trading state, exposes services through an API and operator UI, and clearly distinguishes "
        "simulated, delayed and historical operation from verified live market operation—without claiming live "
        "brokerage readiness or guaranteed profitability.",
    )
    add_h2(doc, "1.3 Objectives")
    for b in [
        "Implement a technical-mode machine-learning pipeline with documented features, labels and evaluation.",
        "Generate BUY/SELL/HOLD signals with probability thresholds and trend confirmation rules.",
        "Apply volatility and drawdown risk gates that can block unsuitable BUY entries.",
        "Backtest long/cash strategies using next-open fills and Indian equity cost assumptions.",
        "Provide isolated paper/demo trading ledgers with no live order routing.",
        "Expose research, market, paper, explainability, live-scan and backtest capabilities via FastAPI.",
        "Deliver a React terminal for Live Analysis and Backtesting demonstration.",
        "Validate core behaviour with automated tests and document deployment/safety constraints.",
    ]:
        add_bullet(doc, b)
    add_h2(doc, "1.4 Scope of Project")
    add_body(
        doc,
        "The PBL-3 scope covers research software for selected NSE symbols using delayed/historical price data "
        "(primarily Yahoo Finance via yfinance), offline and API-driven evaluation, paper trading, and a "
        "demonstration UI. Verified live NSE quote ingestion, holiday-complete session calendars, real-money "
        "execution, and production-grade multi-tenant hosting are outside the completed PBL-3 scope. The current "
        "React Backtesting view uses client-side simulated equity paths for showcase and is therefore marked as "
        "simulated rather than as a verified historical engine result.",
    )

    # Chapter 3
    add_h1(doc, "Chapter 3 – Proposed System Design & Methodology")
    add_h2(doc, "3.1 System Architecture")
    add_body(
        doc,
        "The architecture is layered. A React single-page application provides Live Analysis and Backtesting "
        "views. A FastAPI service (version 0.9.0) wraps Python research modules and starts an optional background "
        "market scanner. Persistent state is stored in three SQLite databases: market.sqlite (candles/quotes), "
        "paper.sqlite (accounts/orders/positions) and scans.sqlite (scan runs and saved backtest summaries). "
        "Figure 3.1 shows the block diagram; Figure 3.2 shows the research decision data flow; Figure 3.3 summarises "
        "module interactions.",
    )
    add_figure(doc, diags["architecture"], "Figure 3.1  System architecture block diagram")
    add_figure(doc, diags["dataflow"], "Figure 3.2  Research decision data-flow diagram")
    add_figure(doc, diags["modules"], "Figure 3.3  Module interaction overview")
    add_body(
        doc,
        "Data freshness is labelled in code and UI where implemented: delayed/historical for Yahoo research bars; "
        "demo/simulated for paper demo marks and the React showcase board; live only if a broker quote payload "
        "explicitly declares live freshness and live authorisation succeeds. During PBL-3, verified live operation "
        "was not demonstrated. Broker execution remains hard-disabled in public runtime configuration.",
    )

    add_h2(doc, "3.2 Development Environment")
    add_table(
        doc,
        "Table 3.1  Hardware and software requirements",
        [
            ["Category", "Requirement (as used in PBL-3)"],
            ["Hardware", "Standard development laptop/desktop; no GPU required"],
            ["OS", "macOS / Linux / Windows with Python and Node.js toolchains"],
            ["Language", "Python 3.9+ (Docker image 3.11); TypeScript/React"],
            ["ML / data", "pandas, NumPy, scikit-learn, yfinance"],
            ["API", "FastAPI, Uvicorn, httpx, pydantic"],
            ["Frontend", "React, Vite, React Router, lightweight-charts"],
            ["Storage", "SQLite files under data/ (or DATA_DIR in deployment)"],
            ["Tests / CI", "pytest; GitHub Actions runs pytest"],
            ["Deploy configs", "Vercel (frontend), Docker, Render, Railway templates"],
        ],
    )

    add_h2(doc, "3.3 Methodology")
    add_body(
        doc,
        "Default runtime uses TECHNICAL_MODE and CORE_MODE. Daily OHLCV bars are enriched with moving averages "
        "(MA20, MA50), RSI14, rate-of-change, returns and short-horizon volatility, plus optional weekly context "
        "features. The binary label marks whether the forward return over seven trading days exceeds one percent. "
        "A HistGradientBoostingClassifier estimates class probability. Evaluation uses chronological purged splits "
        "and expanding walk-forward folds with an embargo equal to the label horizon. Calibration may be applied "
        "on a pre-test slice when sample-size gates are met. Model probability is not treated as probability of profit.",
    )
    add_table(
        doc,
        "Table 3.2  Default TECHNICAL_MODE parameters",
        [
            ["Parameter", "Value in core_config.py"],
            ["Base interval", "1d"],
            ["Context interval", "1wk (optional features)"],
            ["Horizon bars", "7"],
            ["Min forward return (label)", "0.01 (1%)"],
            ["Model", "gb (HistGradientBoosting)"],
            ["Buy / sell thresholds", "0.6 / 0.4"],
            ["Hard vol / drawdown stops", "0.07 / 0.20"],
        ],
    )
    add_body(
        doc,
        "Signal generation applies strategy rules after scoring. In technical trend mode, BUY requires probability "
        "above the buy threshold and Close above MA50; SELL requires probability below the sell threshold and Close "
        "below MA50; otherwise HOLD. Momentum and mean-reversion modes add trend/RSI or z-score conditions. Risk "
        "management computes a RiskScore from realised volatility and rolling drawdown, shrinks position size, and "
        "can convert BUY to HOLD when hard stops trigger (exits are not blocked by research NoTrade).",
    )
    add_table(
        doc,
        "Table 3.3  Signal generation rules by strategy mode",
        [
            ["Mode", "BUY condition (summary)", "SELL condition (summary)"],
            ["technical_only", "P > 0.6 and Close > MA50", "P < 0.4 and Close < MA50"],
            ["momentum", "P + MA20>MA50 + RSI band", "P + MA20<MA50 trend down"],
            ["mean_reversion", "z-score < -2", "z-score > +2"],
            ["multi_factor", "P threshold rule", "P threshold rule"],
        ],
    )
    add_body(
        doc,
        "Backtesting assumes a long/cash book. A signal on bar t fills at the open of bar t+1. Slippage and "
        "IndiaEquityCostConfig charges (brokerage, STT, exchange, SEBI, stamp, GST assumptions) are applied. "
        "Paper trading uses a separate ledger and rejects live broker order paths. AI explainability builds "
        "deterministic narratives from structured fields; optional LLM enrichment cannot override the locked signal. "
        "The frontend Live tab consumes live APIs when a backend is configured; the Backtesting tab in the current "
        "release generates simulated sleeve paths locally for side-by-side demonstration.",
    )
    add_body(
        doc,
        "Working procedure for a typical research run is: select a symbol and date bounds; load or download "
        "candles; build features and labels; fit and score under purged walk-forward settings; attach signals; "
        "apply risk gating; simulate fills and costs; optionally request an explanation. For Live Analysis, the "
        "API scanner can iterate a watchlist on a timer when the backend process is running.",
    )

    # Chapter 4
    add_h1(doc, "Chapter 4 – Work Done in PBL-3")
    add_h2(doc, "4.1 Tasks Achieved So Far")
    add_body(doc, "The following tasks were completed and evidenced during PBL-3 development:")
    for b in [
        "Leakage-aware labelling helpers, next-open execution logic and Indian cost modelling.",
        "Purged walk-forward evaluation utilities and reliability reporting modules.",
        "Market-data SQLite store, quality checks and session gating (weekday cash hours).",
        "Isolated paper and demo trading engine with risk limits.",
        "FastAPI surface including research, market, paper, explain, analytics, live and backtest routers.",
        "React terminal with Live Analysis and Backtesting tabs, TradingView lightweight charts.",
        "Deterministic explainability with signal lock; optional LLM provider hooks.",
        "Deployment templates (Vercel/Docker/Render/Railway) and automated pytest suite (81 passed).",
    ]:
        add_bullet(doc, b)
    add_table(
        doc,
        "Table 4.1  PBL-3 milestone status summary",
        [
            ["Focus area", "Status in PBL-3"],
            ["Technical ML + signals + costs", "Implemented"],
            ["Walk-forward reliability tooling", "Implemented"],
            ["Market store + delayed/historical labelling", "Implemented"],
            ["Verified live NSE tape", "Not achieved"],
            ["Paper trading isolation", "Implemented"],
            ["FastAPI + React terminal", "Implemented (2 tabs)"],
            ["AI explainability lock", "Implemented"],
            ["UI connected to real backtest engine", "Not achieved (simulated showcase)"],
            ["Production API attached to Vercel Live", "Not verified in audit"],
        ],
    )
    add_table(
        doc,
        "Table 4.2  Distinction of data/operation modes",
        [
            ["Mode", "Meaning in this project"],
            ["Historical / delayed", "Yahoo/store research bars; not verified live tape"],
            ["Simulated / demo", "Paper demo marks or React showcase equity paths"],
            ["LIVE LOCKED", "UI state when verified live feed is unavailable"],
            ["Verified live", "Not demonstrated in PBL-3"],
        ],
    )

    add_h2(doc, "4.2 Screenshots, sketches and flow diagrams")
    add_body(
        doc,
        "Figures 4.1 and 4.2 are screenshots of the deployed React terminal. Live Analysis shows the locked live "
        "state and requires a configured API host for scanner results. Backtesting shows four strategy sleeves and "
        "a combined ensemble board used for demonstration; these values are simulated client-side and are not "
        "claimed as verified historical profits.",
    )
    add_figure(doc, ASSETS / "live-analysis-vercel.png", "Figure 4.1  Deployed Live Analysis interface")
    add_figure(doc, ASSETS / "backtesting-vercel.png", "Figure 4.2  Deployed Backtesting interface")

    add_h2(doc, "4.3 Timeline/Gantt Chart covering PBL-3 and planned PBL-4 phases")
    add_body(
        doc,
        "Figure 4.3 presents a relative-week Gantt chart. Green bars denote work completed in PBL-3. Amber bars "
        "denote work planned for PBL-4. The vertical divider marks the PBL-3 to PBL-4 transition.",
    )
    add_figure(doc, diags["gantt"], "Figure 4.3  Gantt chart – PBL-3 completed and PBL-4 planned work")

    # Chapter 5
    add_h1(doc, "Chapter 5 – Conclusion and Future Plan")
    add_h2(doc, "5.1 Summary of PBL-3 Outcomes")
    add_body(
        doc,
        "PBL-3 delivered a working research-oriented multi-factor decision codebase with clear safety defaults: "
        "no live broker orders, paper/demo isolation, and explainability that cannot override signals. The student "
        "implemented modelling, risk, backtesting, persistence, API and UI components sufficient for academic "
        "demonstration of the methodology. Limitations remaining at the end of PBL-3 include absence of verified "
        "live market operation, a Backtesting UI that still uses simulated showcase data, incomplete production "
        "API wiring for the Live tab on Vercel, and a known weekly-feature causality risk requiring remediation.",
    )
    add_h2(doc, "5.2 Tasks Planned for PBL-4")
    for b in [
        "Connect the Backtesting UI to FastAPI /api/backtest/simulate-strategies (or restore explicit DEMO labelling until then).",
        "Repair weekly feature alignment/look-ahead and add regression tests.",
        "Deploy the API with persistent disk and set VITE_API_BASE_URL plus CORS for the frontend.",
        "Pursue licensed live quote verification only if credentials and compliance requirements are met.",
        "Restore or redesign paper/research/analytics pages against existing APIs.",
        "Add API authentication, frontend CI build, and an NSE holiday calendar.",
    ]:
        add_bullet(doc, b)

    add_h1(doc, "References")
    for r in [
        '[1] M. López de Prado, Advances in Financial Machine Learning. Hoboken, NJ, USA: Wiley, 2018.',
        '[2] D. H. Bailey and M. López de Prado, “The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting, and non-normality,” J. Portfolio Manage., vol. 40, no. 5, pp. 94–107, 2014.',
        '[3] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in Proc. ICML, 2017, pp. 1321–1330.',
        '[4] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” J. Mach. Learn. Res., vol. 12, pp. 2825–2830, 2011.',
        '[5] FastAPI Developers, “FastAPI documentation,” [Online]. Available: https://fastapi.tiangolo.com/',
        '[6] TradingView, “Lightweight Charts documentation,” [Online]. Available: https://tradingview.github.io/lightweight-charts/',
        '[7] R. Aroussi et al., “yfinance: Yahoo! Finance market data downloader,” [Online]. Available: https://github.com/ranaroussi/yfinance',
        '[8] National Stock Exchange of India, “Equity market timings (cash market session 09:15–15:30 IST),” [Online]. Available: https://www.nseindia.com/',
        '[9] Project repository documentation: README.md, ARCHITECTURE.md, API.md and DEPLOYMENT.md.',
    ]:
        add_body(doc, r, align="left", first_line=False)

    add_h1(doc, "Appendix A – Evidence Notes")
    add_body(
        doc,
        "Automated checks recorded during report preparation: pytest reported 81 passed tests; the frontend "
        f"production build completed successfully; the deployed frontend at {VERCEL} responded over HTTPS. "
        "Internal audit markdown files under reports/ are companions for review and are not required sections of "
        "this university submission report.",
    )

    # Ensure Normal style
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.font.size = Pt(12)

    doc.save(DOCX_OUT)

    # Verify cover filled
    d2 = Document(str(DOCX_OUT))
    cover_text = "".join(d2.paragraphs[1]._p.itertext())
    assert STUDENT in cover_text, cover_text
    assert REG_NO in cover_text, cover_text
    assert "Multi-Factor" in cover_text or "Financial Decision" in cover_text, cover_text
    return DOCX_OUT


# ---------------------------------------------------------------------------
# PDF with same content + MUJ logo cover
# ---------------------------------------------------------------------------

class UniPDF(FPDF):
    def __init__(self):
        super().__init__(format="A4", unit="in")
        self.numbering = False
        self.body_page_offset = 0

    def footer(self):
        if not self.numbering:
            return
        self.set_y(-0.55)
        self.set_font("Times", "", 10)
        page = self.page_no() - self.body_page_offset
        if page >= 1:
            self.cell(0, 0.25, str(page), align="R")


def _safe(text: str) -> str:
    repl = {
        "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2022": "-", "\u2192": "->",
        "\u20b9": "INR ", "\u00a0": " ",
    }
    for a, b in repl.items():
        text = text.replace(a, b)
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf(diags: dict):
    pdf = UniPDF()
    pdf.set_margins(MARGIN, MARGIN, MARGIN)
    pdf.set_auto_page_break(True, margin=MARGIN + 0.15)

    def h1(text):
        pdf.set_font("Times", "B", 14)
        pdf.multi_cell(0, 0.28, _safe(text), align="L")
        pdf.ln(0.06)

    def h2(text):
        pdf.set_font("Times", "I", 12)
        pdf.multi_cell(0, 0.24, _safe(text), align="L")
        pdf.ln(0.04)

    def body(text, align="J"):
        pdf.set_font("Times", "", 12)
        pdf.multi_cell(0, 0.22 * 1.15, _safe(text), align=align)
        pdf.ln(0.05)

    def bullet(text):
        pdf.set_font("Times", "", 12)
        pdf.multi_cell(0, 0.22 * 1.15, _safe(f"- {text}"), align="J")
        pdf.ln(0.02)

    def center(text, size=12, bold=False):
        pdf.set_font("Times", "B" if bold else "", size)
        pdf.multi_cell(0, 0.25, _safe(text), align="C")
        pdf.ln(0.02)

    def caption(text):
        pdf.set_font("Times", "I", 11)
        pdf.multi_cell(0, 0.2, _safe(text), align="C")
        pdf.ln(0.08)

    def ensure_space(min_h=0.8):
        if pdf.get_y() > pdf.h - MARGIN - min_h:
            pdf.add_page()

    def table(caption_text, rows):
        ensure_space(1.2)
        caption(caption_text)
        for i, row in enumerate(rows):
            ensure_space(0.5)
            pdf.set_font("Times", "B" if i == 0 else "", 9)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin, 0.18,
                           _safe(" | ".join(str(c) for c in row)), align="L")
        pdf.ln(0.08)

    def figure(img: Path, caption_text: str, w=6.5):
        if img.exists():
            max_w = pdf.w - pdf.l_margin - pdf.r_margin
            w = min(w, max_w)
            ensure_space(3.4)
            pdf.image(str(img), x=pdf.l_margin, w=w)
            pdf.ln(0.05)
        caption(caption_text)

    # ---- Cover matching MUJ template ----
    pdf.add_page()
    if LOGO.exists():
        # logo centered near top like template
        logo_w = 3.2
        pdf.image(str(LOGO), x=(pdf.w - logo_w) / 2, y=0.85, w=logo_w)
        pdf.set_y(2.1)
    else:
        pdf.set_y(1.0)

    center("A Project Report on", 14, True)
    pdf.ln(0.08)
    center(TITLE, 16, True)
    pdf.ln(0.12)
    center(f"carried out as part of the {COURSE}", 12)
    center("Submitted by", 12)
    pdf.ln(0.06)
    center(STUDENT, 13, True)
    center(REG_NO, 12, True)
    pdf.ln(0.08)
    center("in partial fulfilment for the award of the degree of", 12)
    center("Bachelor of Technology", 13, True)
    center("in", 12)
    center(DEGREE_LINE, 12, True)
    pdf.ln(0.12)
    center("Under the Guidance of", 12)
    center(GUIDE, 12, True)
    pdf.ln(0.35)
    center(DEPT, 14, True)
    center(UNIV, 14, True)
    center(PLACE, 14, True)
    pdf.ln(0.15)
    center(DATE, 14, True)

    # Abstract
    pdf.add_page()
    center("ABSTRACT", 16, True)
    pdf.ln(0.1)
    body(ABSTRACT_P1)
    body(ABSTRACT_P2)

    pdf.add_page()
    center("LIST OF TABLES", 16, True)
    for item in [
        "Table 3.1  Hardware and software requirements",
        "Table 3.2  Default TECHNICAL_MODE parameters",
        "Table 3.3  Signal generation rules by strategy mode",
        "Table 4.1  PBL-3 milestone status summary",
        "Table 4.2  Distinction of data/operation modes",
    ]:
        body(item, align="L")

    pdf.add_page()
    center("LIST OF FIGURES", 16, True)
    for item in [
        "Figure 3.1  System architecture block diagram",
        "Figure 3.2  Research decision data-flow diagram",
        "Figure 3.3  Module interaction overview",
        "Figure 4.1  Deployed Live Analysis interface",
        "Figure 4.2  Deployed Backtesting interface",
        "Figure 4.3  Gantt chart - PBL-3 completed and PBL-4 planned work",
    ]:
        body(item, align="L")

    pdf.add_page()
    center("TABLE OF CONTENTS", 16, True)
    for item in [
        "Abstract",
        "List of Tables",
        "List of Figures",
        "Chapter 1 - Introduction",
        "    1.1 Introduction",
        "    1.2 Problem Statement",
        "    1.3 Objectives",
        "    1.4 Scope of Project",
        "Chapter 3 - Proposed System Design & Methodology",
        "    3.1 System Architecture",
        "    3.2 Development Environment",
        "    3.3 Methodology",
        "Chapter 4 - Work Done in PBL-3",
        "    4.1 Tasks Achieved So Far",
        "    4.2 Screenshots, sketches and flow diagrams",
        "    4.3 Timeline/Gantt Chart",
        "Chapter 5 - Conclusion and Future Plan",
        "    5.1 Summary of PBL-3 Outcomes",
        "    5.2 Tasks Planned for PBL-4",
        "References",
        "Appendix A - Evidence Notes",
    ]:
        body(item, align="L")

    # Body
    pdf.add_page()
    pdf.body_page_offset = pdf.page_no() - 1
    pdf.numbering = True

    h1("Chapter 1 - Introduction")
    h2("1.1 Introduction")
    body(
        "Financial decision support for equities is frequently built from informal indicator rules or opaque "
        "black-box models. In academic and early-stage engineering settings, such systems often omit realistic "
        "execution assumptions, transaction costs, chronological validation and clear labelling of data freshness. "
        "The Multi-Factor Financial Decision System was developed to address these gaps for Indian National Stock "
        "Exchange (NSE) cash equities in a controlled, non-brokerage research environment."
    )
    body(
        "Motivation for the project comes from the need to practise responsible quantitative software engineering: "
        "separating model probability from probability of profit, preventing trivial look-ahead where possible, "
        "and refusing to route live broker orders. Applications include classroom demonstration of ML-assisted "
        "signal research, paper-trading education, and structured comparison of technical strategy modes. Advantages "
        "of the implemented approach include inspectable Python modules, SQLite persistence, FastAPI access for "
        "integration, and a React terminal with explicit simulation/safety badges."
    )
    h2("1.2 Problem Statement")
    body(
        "The problem addressed in PBL-3 is to design and implement an end-to-end software system that generates "
        "BUY, SELL and HOLD decisions for selected NSE equities using technical features and a supervised classifier, "
        "evaluates those decisions with purged walk-forward methods and cost-aware next-open backtests, persists "
        "market and paper-trading state, exposes services through an API and operator UI, and clearly distinguishes "
        "simulated, delayed and historical operation from verified live market operation-without claiming live "
        "brokerage readiness or guaranteed profitability."
    )
    h2("1.3 Objectives")
    for b in [
        "Implement a technical-mode machine-learning pipeline with documented features, labels and evaluation.",
        "Generate BUY/SELL/HOLD signals with probability thresholds and trend confirmation rules.",
        "Apply volatility and drawdown risk gates that can block unsuitable BUY entries.",
        "Backtest long/cash strategies using next-open fills and Indian equity cost assumptions.",
        "Provide isolated paper/demo trading ledgers with no live order routing.",
        "Expose research, market, paper, explainability, live-scan and backtest capabilities via FastAPI.",
        "Deliver a React terminal for Live Analysis and Backtesting demonstration.",
        "Validate core behaviour with automated tests and document deployment/safety constraints.",
    ]:
        bullet(b)
    h2("1.4 Scope of Project")
    body(
        "The PBL-3 scope covers research software for selected NSE symbols using delayed/historical price data "
        "(primarily Yahoo Finance via yfinance), offline and API-driven evaluation, paper trading, and a "
        "demonstration UI. Verified live NSE quote ingestion, holiday-complete session calendars, real-money "
        "execution, and production-grade multi-tenant hosting are outside the completed PBL-3 scope. The current "
        "React Backtesting view uses client-side simulated equity paths for showcase and is therefore marked as "
        "simulated rather than as a verified historical engine result."
    )

    h1("Chapter 3 - Proposed System Design & Methodology")
    h2("3.1 System Architecture")
    body(
        "The architecture is layered. A React single-page application provides Live Analysis and Backtesting "
        "views. A FastAPI service (version 0.9.0) wraps Python research modules and starts an optional background "
        "market scanner. Persistent state is stored in three SQLite databases: market.sqlite, paper.sqlite and "
        "scans.sqlite. Figure 3.1 shows the block diagram; Figure 3.2 shows the research decision data flow; "
        "Figure 3.3 summarises module interactions."
    )
    figure(diags["architecture"], "Figure 3.1  System architecture block diagram")
    figure(diags["dataflow"], "Figure 3.2  Research decision data-flow diagram")
    figure(diags["modules"], "Figure 3.3  Module interaction overview")
    body(
        "Data freshness is labelled where implemented: delayed/historical for Yahoo research bars; demo/simulated "
        "for paper demo marks and the React showcase board; live only if a broker quote payload explicitly declares "
        "live freshness and authorisation succeeds. Verified live operation was not demonstrated in PBL-3. Broker "
        "execution remains hard-disabled."
    )
    h2("3.2 Development Environment")
    table("Table 3.1  Hardware and software requirements", [
        ["Category", "Requirement (as used in PBL-3)"],
        ["Hardware", "Standard development laptop/desktop; no GPU required"],
        ["OS", "macOS / Linux / Windows with Python and Node.js"],
        ["Language", "Python 3.9+ (Docker 3.11); TypeScript/React"],
        ["ML / data", "pandas, NumPy, scikit-learn, yfinance"],
        ["API", "FastAPI, Uvicorn, httpx, pydantic"],
        ["Frontend", "React, Vite, React Router, lightweight-charts"],
        ["Storage", "SQLite under data/ or DATA_DIR"],
        ["Tests / CI", "pytest; GitHub Actions pytest job"],
        ["Deploy configs", "Vercel, Docker, Render, Railway templates"],
    ])
    h2("3.3 Methodology")
    body(
        "Default runtime uses TECHNICAL_MODE and CORE_MODE. Daily OHLCV bars are enriched with MA20, MA50, RSI14, "
        "ROC, returns and short-horizon volatility, plus optional weekly context features. The binary label marks "
        "whether the forward return over seven trading days exceeds one percent. A HistGradientBoostingClassifier "
        "estimates class probability. Evaluation uses chronological purged splits and expanding walk-forward folds "
        "with an embargo equal to the label horizon. Model probability is not treated as probability of profit."
    )
    table("Table 3.2  Default TECHNICAL_MODE parameters", [
        ["Parameter", "Value in core_config.py"],
        ["Base interval", "1d"],
        ["Context interval", "1wk (optional features)"],
        ["Horizon bars", "7"],
        ["Min forward return (label)", "0.01 (1%)"],
        ["Model", "gb (HistGradientBoosting)"],
        ["Buy / sell thresholds", "0.6 / 0.4"],
        ["Hard vol / drawdown stops", "0.07 / 0.20"],
    ])
    body(
        "Signal generation applies strategy rules after scoring. In technical trend mode, BUY requires probability "
        "above the buy threshold and Close above MA50; SELL requires probability below the sell threshold and Close "
        "below MA50; otherwise HOLD. Risk management computes RiskScore from volatility and drawdown, shrinks "
        "position size, and can convert BUY to HOLD when hard stops trigger."
    )
    table("Table 3.3  Signal generation rules by strategy mode", [
        ["Mode", "BUY (summary)", "SELL (summary)"],
        ["technical_only", "P>0.6 and Close>MA50", "P<0.4 and Close<MA50"],
        ["momentum", "P + MA20>MA50 + RSI band", "P + downtrend filter"],
        ["mean_reversion", "z-score < -2", "z-score > +2"],
        ["multi_factor", "P threshold rule", "P threshold rule"],
    ])
    body(
        "Backtesting assumes a long/cash book with fills at the next open and IndiaEquityCostConfig charges. "
        "Paper trading uses a separate ledger. AI explainability cannot override locked signals. The Live tab needs "
        "a configured API; the Backtesting tab currently generates simulated sleeve paths locally for demonstration."
    )
    body(
        "Working procedure for a typical research run is: select a symbol and date bounds; load or download "
        "candles; build features and labels; fit and score under purged walk-forward settings; attach signals; "
        "apply risk gating; simulate fills and costs; optionally request an explanation."
    )

    h1("Chapter 4 - Work Done in PBL-3")
    h2("4.1 Tasks Achieved So Far")
    body("The following tasks were completed and evidenced during PBL-3:")
    for b in [
        "Leakage-aware labelling helpers, next-open execution logic and Indian cost modelling.",
        "Purged walk-forward evaluation utilities and reliability reporting modules.",
        "Market-data SQLite store, quality checks and session gating.",
        "Isolated paper and demo trading engine with risk limits.",
        "FastAPI routers for research, market, paper, explain, analytics, live and backtest.",
        "React terminal with Live Analysis and Backtesting tabs.",
        "Deterministic explainability with signal lock.",
        "Deployment templates and automated pytest suite (81 passed).",
    ]:
        bullet(b)
    table("Table 4.1  PBL-3 milestone status summary", [
        ["Focus area", "Status in PBL-3"],
        ["Technical ML + signals + costs", "Implemented"],
        ["Walk-forward reliability tooling", "Implemented"],
        ["Market store + delayed/historical labelling", "Implemented"],
        ["Verified live NSE tape", "Not achieved"],
        ["Paper trading isolation", "Implemented"],
        ["FastAPI + React terminal", "Implemented (2 tabs)"],
        ["AI explainability lock", "Implemented"],
        ["UI connected to real backtest engine", "Not achieved (simulated showcase)"],
        ["Production API attached to Vercel Live", "Not verified"],
    ])
    table("Table 4.2  Distinction of data/operation modes", [
        ["Mode", "Meaning in this project"],
        ["Historical / delayed", "Yahoo/store research bars; not verified live tape"],
        ["Simulated / demo", "Paper demo marks or React showcase equity paths"],
        ["LIVE LOCKED", "UI state when verified live feed is unavailable"],
        ["Verified live", "Not demonstrated in PBL-3"],
    ])
    h2("4.2 Screenshots, sketches and flow diagrams")
    body(
        "Figures 4.1 and 4.2 are screenshots of the deployed React terminal. Live Analysis shows the locked live "
        "state. Backtesting shows simulated strategy sleeves for demonstration and must not be read as verified "
        "historical profitability."
    )
    figure(ASSETS / "live-analysis-vercel.png", "Figure 4.1  Deployed Live Analysis interface")
    figure(ASSETS / "backtesting-vercel.png", "Figure 4.2  Deployed Backtesting interface")
    h2("4.3 Timeline/Gantt Chart covering PBL-3 and planned PBL-4 phases")
    body("Figure 4.3 presents a relative-week Gantt chart. Green bars denote completed PBL-3 work. Amber bars denote planned PBL-4 work.")
    figure(diags["gantt"], "Figure 4.3  Gantt chart - PBL-3 completed and PBL-4 planned work")

    h1("Chapter 5 - Conclusion and Future Plan")
    h2("5.1 Summary of PBL-3 Outcomes")
    body(
        "PBL-3 delivered a working research-oriented multi-factor decision codebase with clear safety defaults: "
        "no live broker orders, paper/demo isolation, and explainability that cannot override signals. Remaining "
        "limitations include absence of verified live market operation, a simulated Backtesting showcase UI, "
        "incomplete production API wiring for Live on Vercel, and a weekly-feature causality risk requiring remediation."
    )
    h2("5.2 Tasks Planned for PBL-4")
    for b in [
        "Connect Backtesting UI to FastAPI backtest routes (or keep explicit DEMO labelling).",
        "Repair weekly feature alignment/look-ahead and add regression tests.",
        "Deploy API with persistent disk and configure VITE_API_BASE_URL plus CORS.",
        "Pursue licensed live quote verification only if requirements are met.",
        "Restore or redesign paper/research/analytics pages against existing APIs.",
        "Add API authentication, frontend CI build, and an NSE holiday calendar.",
    ]:
        bullet(b)

    h1("References")
    for r in [
        '[1] M. Lopez de Prado, Advances in Financial Machine Learning. Hoboken, NJ, USA: Wiley, 2018.',
        '[2] D. H. Bailey and M. Lopez de Prado, "The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting, and non-normality," J. Portfolio Manage., vol. 40, no. 5, pp. 94-107, 2014.',
        '[3] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in Proc. ICML, 2017, pp. 1321-1330.',
        '[4] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," J. Mach. Learn. Res., vol. 12, pp. 2825-2830, 2011.',
        '[5] FastAPI Developers, "FastAPI documentation," [Online]. Available: https://fastapi.tiangolo.com/',
        '[6] TradingView, "Lightweight Charts documentation," [Online]. Available: https://tradingview.github.io/lightweight-charts/',
        '[7] R. Aroussi et al., "yfinance: Yahoo! Finance market data downloader," [Online]. Available: https://github.com/ranaroussi/yfinance',
        '[8] National Stock Exchange of India, "Equity market timings (cash market session 09:15-15:30 IST)," [Online]. Available: https://www.nseindia.com/',
        '[9] Project repository documentation: README.md, ARCHITECTURE.md, API.md and DEPLOYMENT.md.',
    ]:
        body(r, align="L")

    h1("Appendix A - Evidence Notes")
    body(
        "Automated checks recorded during report preparation: pytest reported 81 passed tests; the frontend "
        f"production build completed successfully; the deployed frontend at {VERCEL} responded over HTTPS."
    )

    pdf.output(str(PDF_OUT))
    return pdf.page_no(), pdf.body_page_offset


def main():
    if not TEMPLATE.exists():
        raise SystemExit(f"Template not found: {TEMPLATE}")
    # Ensure logo from template
    if not LOGO.exists():
        import zipfile
        from PIL import Image
        with zipfile.ZipFile(TEMPLATE) as z:
            raw = z.read("word/media/image1.png")
            tmp = ASSETS / "template_image1.png"
            tmp.write_bytes(raw)
            Image.open(tmp).convert("RGBA").save(LOGO)

    diags = ensure_diagrams()
    build_docx(diags)
    total, offset = build_pdf(diags)
    body_pages = total - offset
    print(f"DOCX: {DOCX_OUT} ({DOCX_OUT.stat().st_size} bytes)")
    print(f"PDF:  {PDF_OUT} ({PDF_OUT.stat().st_size} bytes)")
    print(f"PDF pages: {total} (body from Introduction: {body_pages})")
    if total < 12:
        raise SystemExit("PDF below 12 pages")
    # verify identity in docx cover
    doc = Document(str(DOCX_OUT))
    cover = "".join(doc.paragraphs[1]._p.itertext())
    print("Cover contains student:", STUDENT in cover)
    print("Cover contains reg:", REG_NO in cover)
    print("Cover contains title fragment:", "Financial Decision" in cover or "Multi-Factor" in cover)
    print("Verification OK")


if __name__ == "__main__":
    main()
