#!/usr/bin/env python3
"""
Generate university-guideline PBL-3 report:
  reports/PBL3_Project_Report.docx
  reports/PBL3_Project_Report.pdf

Uses only verified project facts. Placeholders for institutional identity.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, Twips
from fpdf import FPDF

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

VERCEL = "https://multi-factor-decision-system.vercel.app"
REPO_COMMIT = "fe1619a"
MARGIN_IN = 0.7

P = {
    "university": "[University Name]",
    "department": "[Department Name]",
    "programme": "Bachelor of Technology (B.Tech)",
    "course": "Project Based Learning - III (PBL-3)",
    "title": "Multi-Factor Financial Decision System",
    "students": "[Student Name(s) and Registration Number(s)]",
    "guide": "[Faculty Guide Name, Designation]",
    "year": "[Academic Year]",
    "submission": "[Submission Date]",
}


# ---------------------------------------------------------------------------
# Diagrams
# ---------------------------------------------------------------------------

def _save_fig(fig, name: str) -> Path:
    path = ASSETS / name
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_architecture_diagram() -> Path:
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("System Architecture – Multi-Factor Financial Decision System", fontsize=12, pad=8)

    def box(x, y, w, h, text, color="#E8EEF7"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=color, edgecolor="#333333", linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, wrap=True)

    box(0.4, 5.4, 4.2, 1.2, "React Terminal (Vercel SPA)\nLive Analysis | Backtesting\nlightweight-charts", "#DDE7F8")
    box(5.4, 5.4, 4.2, 1.2, "FastAPI 0.9.0 (Uvicorn)\n/api/live | /api/backtest | /api/research\n/api/paper | /api/explain | /api/market", "#F8E8DD")
    box(0.4, 3.5, 3.0, 1.4, "Market Data Layer\nYahoo history\nBroker adapter (optional)\nSQLite candles/quotes", "#E8F5E9")
    box(3.6, 3.5, 3.0, 1.4, "Research Engine\nFeatures + GB model\nSignals + risk gating\nNext-open backtest", "#FFF3E0")
    box(6.8, 3.5, 2.8, 1.4, "Support Services\nPaper ledger\nExplainability\nLive scanner thread", "#F3E5F5")
    box(0.4, 1.4, 3.0, 1.4, "market.sqlite", "#ECEFF1")
    box(3.6, 1.4, 3.0, 1.4, "paper.sqlite", "#ECEFF1")
    box(6.8, 1.4, 2.8, 1.4, "scans.sqlite", "#ECEFF1")

    for x1, y1, x2, y2 in [
        (2.5, 5.4, 2.5, 4.9),
        (7.5, 5.4, 7.5, 4.9),
        (5.4, 6.0, 4.6, 6.0),
        (1.9, 3.5, 1.9, 2.8),
        (5.1, 3.5, 5.1, 2.8),
        (8.2, 3.5, 8.2, 2.8),
    ]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.1))

    ax.text(5, 0.55, "Hard invariant: broker_execution = false (no live order routing)",
            ha="center", fontsize=8, style="italic")
    return _save_fig(fig, "fig_architecture.png")


def make_dataflow_diagram() -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Research Decision Data Flow (TECHNICAL_MODE)", fontsize=12, pad=8)

    steps = [
        (0.3, "OHLCV\n(Yahoo / store)"),
        (2.4, "Feature &\nLabel build"),
        (4.5, "GB Classifier\n+ walk-forward"),
        (6.6, "BUY/SELL/HOLD\n+ risk gate"),
        (8.7, "Next-open\nbacktest"),
        (10.5, "Explain /\nAPI / UI"),
    ]
    for i, (x, label) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((x, 1.8), 1.6, 1.5, boxstyle="round,pad=0.03,rounding_size=0.08",
                                    facecolor="#E3F2FD", edgecolor="#1565C0", linewidth=1.2))
        ax.text(x + 0.8, 2.55, label, ha="center", va="center", fontsize=8)
        if i < len(steps) - 1:
            ax.annotate("", xy=(steps[i + 1][0], 2.55), xytext=(x + 1.6, 2.55),
                        arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.2))

    ax.text(6, 0.9,
            "Label: 7-day forward return > 1%. Fill at t+1 Open. Costs: IndiaEquityCostConfig assumptions.",
            ha="center", fontsize=8)
    ax.text(6, 0.4,
            "Freshness: delayed/historical research data by default; verified live tape not available in PBL-3.",
            ha="center", fontsize=8, style="italic")
    return _save_fig(fig, "fig_dataflow.png")


def make_module_diagram() -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Module Interaction Overview", fontsize=12, pad=8)

    modules = [
        (0.4, 4.2, "frontend/\nApp, pages, charts"),
        (3.5, 4.2, "api/\nrouters + settings"),
        (6.6, 4.2, "four_model_pipeline"),
        (0.4, 2.2, "market_data/"),
        (3.5, 2.2, "strategy + risk +\nbacktest + costs"),
        (6.6, 2.2, "paper_trading/\nreasoning/\nlive_analysis/"),
        (3.5, 0.4, "SQLite stores\nmarket | paper | scans"),
    ]
    for x, y, t in modules:
        ax.add_patch(FancyBboxPatch((x, y), 2.8, 1.3, boxstyle="round,pad=0.03,rounding_size=0.08",
                                    facecolor="#FFF8E1", edgecolor="#6D4C41", linewidth=1.1))
        ax.text(x + 1.4, y + 0.65, t, ha="center", va="center", fontsize=8)

    for a, b in [((1.8, 4.2), (1.8, 3.5)), ((4.9, 4.2), (4.9, 3.5)), ((8.0, 4.2), (8.0, 3.5)),
                 ((4.9, 2.2), (4.9, 1.7)), ((1.8, 2.2), (4.0, 1.5)), ((8.0, 2.2), (6.0, 1.5))]:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color="#6D4C41", lw=1.0))
    return _save_fig(fig, "fig_modules.png")


def make_gantt_chart() -> Path:
    # Completed PBL-3 vs planned PBL-4
    tasks = [
        ("PBL-3: Core ML + signals + costs", 0, 3, "done"),
        ("PBL-3: Walk-forward reliability", 2, 4, "done"),
        ("PBL-3: Market SQLite + freshness labels", 3, 5, "done"),
        ("PBL-3: Paper trading isolation", 4, 6, "done"),
        ("PBL-3: FastAPI + React terminal", 5, 8, "done"),
        ("PBL-3: Explainability + analytics APIs", 7, 9, "done"),
        ("PBL-3: Live scanner + backtest APIs", 8, 10, "done"),
        ("PBL-3: Deployment configs + audit report", 9, 11, "done"),
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
        ax.barh(i, end - start, left=start, height=0.55, color=colors[kind], edgecolor="black", linewidth=0.5)
        ax.text(-0.2, i, name, ha="right", va="center", fontsize=7.5)
    ax.set_yticks([])
    ax.set_xlabel("Project weeks (relative index)", fontsize=9)
    ax.set_xlim(-0.5, 18)
    ax.set_ylim(-1, len(tasks))
    ax.set_title("Gantt Chart – PBL-3 Completed vs PBL-4 Planned", fontsize=12)
    ax.axvline(11, color="#1565C0", linestyle="--", linewidth=1.2)
    ax.text(11.1, len(tasks) - 0.3, "PBL-3 → PBL-4", fontsize=8, color="#1565C0")
    done_patch = mpatches.Patch(color=colors["done"], label="Completed in PBL-3")
    plan_patch = mpatches.Patch(color=colors["plan"], label="Planned for PBL-4")
    ax.legend(handles=[done_patch, plan_patch], loc="lower right", fontsize=8)
    ax.invert_yaxis()
    return _save_fig(fig, "fig_gantt.png")


def ensure_diagrams():
    return {
        "architecture": make_architecture_diagram(),
        "dataflow": make_dataflow_diagram(),
        "modules": make_module_diagram(),
        "gantt": make_gantt_chart(),
    }


# ---------------------------------------------------------------------------
# Shared narrative (identical for DOCX + PDF)
# ---------------------------------------------------------------------------

CONTENT = {
    "abstract_p1": (
        "This PBL-3 report presents the Multi-Factor Financial Decision System, a research and simulation "
        "software stack for Indian equities. The implemented system combines technical feature engineering, "
        "a gradient-boosting classifier, purged walk-forward evaluation, BUY/SELL/HOLD signal rules, risk gating, "
        "next-open backtesting with Indian equity cost assumptions, isolated paper trading, optional AI "
        "explainability, FastAPI services, and a React operator terminal. Live broker order execution is disabled "
        "by design. Verified live NSE market-tape operation was not achieved during PBL-3; research paths use "
        "delayed or historical data sources."
    ),
    "abstract_p2": (
        "Work completed in PBL-3 includes the core Python research engine, SQLite persistence for market, paper "
        "and scan data, FastAPI routers for live scanning and backtesting, and a two-tab React interface (Live "
        "Analysis and Backtesting). Automated validation recorded 81 passing pytest cases and a successful "
        f"frontend production build. The frontend is reachable at {VERCEL}. The Backtesting tab currently renders "
        "client-side simulated strategy paths for demonstration and must not be interpreted as verified historical "
        "profitability. Planned PBL-4 work focuses on connecting the UI to the real backtest API, fixing identified "
        "data-leakage risks, and completing backend deployment wiring."
    ),
}


# ---------------------------------------------------------------------------
# DOCX helpers
# ---------------------------------------------------------------------------

def set_run_font(run, *, size=12, bold=False, italic=False, name="Times New Roman"):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic


def set_paragraph_format(p, *, align="justify", space_after=6, space_before=0, first_line=False):
    pf = p.paragraph_format
    if align == "justify":
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    elif align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "right":
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    elif align == "left":
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(space_before)
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = 1.15
    if first_line:
        pf.first_line_indent = Inches(0.25)


def add_body(doc, text, *, align="justify", first_line=True):
    p = doc.add_paragraph()
    set_paragraph_format(p, align=align, first_line=first_line and align == "justify")
    run = p.add_run(text)
    set_run_font(run, size=12)
    return p


def add_main_heading(doc, text):
    p = doc.add_paragraph()
    set_paragraph_format(p, align="left", space_before=12, space_after=8, first_line=False)
    run = p.add_run(text)
    set_run_font(run, size=14, bold=True)
    return p


def add_subheading(doc, text):
    p = doc.add_paragraph()
    set_paragraph_format(p, align="left", space_before=10, space_after=6, first_line=False)
    run = p.add_run(text)
    set_run_font(run, size=12, italic=True)
    return p


def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    set_paragraph_format(p, align="justify", space_after=3, first_line=False)
    # clear default run and set font
    if p.runs:
        for r in p.runs:
            set_run_font(r, size=12)
    else:
        run = p.add_run(text)
        set_run_font(run, size=12)
        return p
    p.clear()
    run = p.add_run(text)
    set_run_font(run, size=12)
    return p


def add_caption(doc, text, *, kind="figure"):
    p = doc.add_paragraph()
    set_paragraph_format(p, align="center", space_before=4, space_after=10, first_line=False)
    run = p.add_run(text)
    set_run_font(run, size=11, italic=True)
    return p


def add_table(doc, caption, rows):
    add_caption(doc, caption, kind="table")
    # move caption above: already called before table
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.rows[i].cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            run = p.add_run(str(val))
            set_run_font(run, size=10, bold=(i == 0))
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    doc.add_paragraph()


def add_picture_with_caption(doc, path: Path, caption: str, width=6.2):
    if path.exists():
        doc.add_picture(str(path), width=Inches(width))
        last = doc.paragraphs[-1]
        last.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_caption(doc, caption)


def configure_sections(doc: Document):
    for section in doc.sections:
        section.page_width = Inches(8.27)
        section.page_height = Inches(11.69)
        section.left_margin = Inches(MARGIN_IN)
        section.right_margin = Inches(MARGIN_IN)
        section.top_margin = Inches(MARGIN_IN)
        section.bottom_margin = Inches(MARGIN_IN)


def setup_footer_page_numbers(section, *, start_number=None):
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    # clear
    p.clear()
    add_page_number_field(p)
    if start_number is not None:
        sectPr = section._sectPr
        pgNumType = OxmlElement("w:pgNumType")
        pgNumType.set(qn("w:start"), str(start_number))
        sectPr.append(pgNumType)


def add_page_number_field(paragraph):
    run = paragraph.add_run()
    set_run_font(run, size=10)
    fld_char_begin = OxmlElement("w:fldChar")
    fld_char_begin.set(qn("w:fldCharType"), "begin")

    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "

    fld_char_separate = OxmlElement("w:fldChar")
    fld_char_separate.set(qn("w:fldCharType"), "separate")

    fld_char_end = OxmlElement("w:fldChar")
    fld_char_end.set(qn("w:fldCharType"), "end")

    run._r.append(fld_char_begin)
    run._r.append(instr)
    run._r.append(fld_char_separate)
    run._r.append(fld_char_end)


def build_docx(diags: dict, path: Path):
    doc = Document()
    configure_sections(doc)

    # Normal style
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.font.size = Pt(12)

    # ---- Front matter section: no page numbers / restart later ----
    front = doc.sections[0]
    front.footer.is_linked_to_previous = False
    # empty footer for front matter
    if front.footer.paragraphs:
        front.footer.paragraphs[0].clear()

    # 1. Cover Page
    for _ in range(2):
        doc.add_paragraph()
    for line, size, bold in [
        (P["university"], 16, True),
        (P["department"], 12, False),
        (P["programme"], 12, False),
        ("", 12, False),
        ("A Project Report on", 12, False),
        (P["title"], 18, True),
        (f"Submitted in partial fulfilment of the requirements of\n{P['course']}", 12, False),
        ("", 12, False),
        ("Submitted by", 12, False),
        (P["students"], 12, False),
        ("", 12, False),
        ("Under the guidance of", 12, False),
        (P["guide"], 12, False),
        ("", 12, False),
        (f"Academic Year: {P['year']}", 12, False),
        (f"Submission Date: {P['submission']}", 12, False),
    ]:
        p = doc.add_paragraph()
        set_paragraph_format(p, align="center", space_after=4, first_line=False)
        if line:
            for part in line.split("\n"):
                if p.runs:
                    p = doc.add_paragraph()
                    set_paragraph_format(p, align="center", space_after=4, first_line=False)
                run = p.add_run(part)
                set_run_font(run, size=size, bold=bold)

    doc.add_page_break()

    # 2. Abstract
    add_main_heading(doc, "Abstract")
    add_body(doc, CONTENT["abstract_p1"])
    add_body(doc, CONTENT["abstract_p2"])
    doc.add_page_break()

    # 3. List of Tables
    add_main_heading(doc, "List of Tables")
    for item in [
        "Table 3.1  Hardware and software requirements",
        "Table 3.2  Default TECHNICAL_MODE parameters",
        "Table 3.3  Signal generation rules by strategy mode",
        "Table 4.1  PBL-3 milestone status summary",
        "Table 4.2  Distinction of data/operation modes",
    ]:
        add_body(doc, item, align="left", first_line=False)
    doc.add_page_break()

    # 4. List of Figures
    add_main_heading(doc, "List of Figures")
    for item in [
        "Figure 3.1  System architecture block diagram",
        "Figure 3.2  Research decision data-flow diagram",
        "Figure 3.3  Module interaction overview",
        "Figure 4.1  Deployed Live Analysis interface",
        "Figure 4.2  Deployed Backtesting interface",
        "Figure 4.3  Gantt chart – PBL-3 completed and PBL-4 planned work",
    ]:
        add_body(doc, item, align="left", first_line=False)
    doc.add_page_break()

    # 5. TOC
    add_main_heading(doc, "Table of Contents")
    toc = [
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
    ]
    for item in toc:
        add_body(doc, item, align="left", first_line=False)
    doc.add_page_break()

    # ---- Body section: page numbers start at 1 ----
    new_section = doc.add_section()
    new_section.page_width = Inches(8.27)
    new_section.page_height = Inches(11.69)
    new_section.left_margin = Inches(MARGIN_IN)
    new_section.right_margin = Inches(MARGIN_IN)
    new_section.top_margin = Inches(MARGIN_IN)
    new_section.bottom_margin = Inches(MARGIN_IN)
    setup_footer_page_numbers(new_section, start_number=1)

    # Chapter 1
    add_main_heading(doc, "Chapter 1 – Introduction")

    add_subheading(doc, "1.1 Introduction")
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

    add_subheading(doc, "1.2 Problem Statement")
    add_body(
        doc,
        "The problem addressed in PBL-3 is to design and implement an end-to-end software system that generates "
        "BUY, SELL and HOLD decisions for selected NSE equities using technical features and a supervised classifier, "
        "evaluates those decisions with purged walk-forward methods and cost-aware next-open backtests, persists "
        "market and paper-trading state, exposes services through an API and operator UI, and clearly distinguishes "
        "simulated, delayed and historical operation from verified live market operation—without claiming live "
        "brokerage readiness or guaranteed profitability.",
    )

    add_subheading(doc, "1.3 Objectives")
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

    add_subheading(doc, "1.4 Scope of Project")
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
    add_main_heading(doc, "Chapter 3 – Proposed System Design & Methodology")

    add_subheading(doc, "3.1 System Architecture")
    add_body(
        doc,
        "The architecture is layered. A React single-page application provides Live Analysis and Backtesting "
        "views. A FastAPI service (version 0.9.0) wraps Python research modules and starts an optional background "
        "market scanner. Persistent state is stored in three SQLite databases: market.sqlite (candles/quotes), "
        "paper.sqlite (accounts/orders/positions) and scans.sqlite (scan runs and saved backtest summaries). "
        "Figure 3.1 shows the block diagram; Figure 3.2 shows the research decision data flow; Figure 3.3 summarises "
        "module interactions.",
    )
    add_picture_with_caption(doc, diags["architecture"], "Figure 3.1  System architecture block diagram")
    add_picture_with_caption(doc, diags["dataflow"], "Figure 3.2  Research decision data-flow diagram")
    add_picture_with_caption(doc, diags["modules"], "Figure 3.3  Module interaction overview")
    add_body(
        doc,
        "Data freshness is labelled in code and UI where implemented: delayed/historical for Yahoo research bars; "
        "demo/simulated for paper demo marks and the React showcase board; live only if a broker quote payload "
        "explicitly declares live freshness and live authorisation succeeds. During PBL-3, verified live operation "
        "was not demonstrated. Broker execution remains hard-disabled in public runtime configuration.",
    )

    add_subheading(doc, "3.2 Development Environment")
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

    add_subheading(doc, "3.3 Methodology")
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
        "Reported metrics include net/gross return, maximum drawdown, Sharpe/Sortino, CAGR and trade statistics. "
        "Paper trading uses a separate ledger and rejects live broker order paths. AI explainability builds "
        "deterministic narratives from structured fields; optional LLM enrichment cannot override the locked signal. "
        "The frontend Live tab consumes live/status APIs when a backend is configured; the Backtesting tab in the "
        "current release generates simulated sleeve paths locally for side-by-side demonstration.",
    )
    add_body(
        doc,
        "Working procedure for a typical research run is: select a symbol and date bounds; load or download "
        "candles; build features and labels; fit and score under purged walk-forward settings; attach signals; "
        "apply risk gating; simulate fills and costs; optionally request an explanation. For Live Analysis, the "
        "API scanner can iterate a watchlist on a timer when the backend process is running. For the current "
        "React Backtesting demonstration, the browser regenerates four sleeves plus a combined ensemble from a "
        "deterministic seed of ticker, dates and capital without calling the historical backtest engine.",
    )

    # Chapter 4
    add_main_heading(doc, "Chapter 4 – Work Done in PBL-3")

    add_subheading(doc, "4.1 Tasks Achieved So Far")
    add_body(
        doc,
        "The following tasks were completed and evidenced during PBL-3 development and the repository audit:",
    )
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

    add_subheading(doc, "4.2 Screenshots, sketches and flow diagrams")
    add_body(
        doc,
        "Figures 4.1 and 4.2 are screenshots of the deployed React terminal. Live Analysis shows the locked live "
        "state and requires a configured API host for scanner results. Backtesting shows four strategy sleeves and "
        "a combined ensemble board used for demonstration; these values are simulated client-side and are not "
        "claimed as verified historical profits.",
    )
    live_shot = ASSETS / "live-analysis-vercel.png"
    bt_shot = ASSETS / "backtesting-vercel.png"
    add_picture_with_caption(doc, live_shot, "Figure 4.1  Deployed Live Analysis interface")
    add_picture_with_caption(doc, bt_shot, "Figure 4.2  Deployed Backtesting interface")
    add_body(
        doc,
        "Architecture and data-flow sketches used in Chapter 3 (Figures 3.1–3.3) were generated from the "
        "implemented module boundaries to support design documentation for PBL-3 evaluation.",
    )

    add_subheading(doc, "4.3 Timeline/Gantt Chart covering PBL-3 and planned PBL-4 phases")
    add_body(
        doc,
        "Figure 4.3 presents a relative-week Gantt chart. Green bars denote work completed in PBL-3. Amber bars "
        "denote work planned for PBL-4. The vertical divider marks the PBL-3 to PBL-4 transition.",
    )
    add_picture_with_caption(doc, diags["gantt"], "Figure 4.3  Gantt chart – PBL-3 completed and PBL-4 planned work")

    # Chapter 5
    add_main_heading(doc, "Chapter 5 – Conclusion and Future Plan")

    add_subheading(doc, "5.1 Summary of PBL-3 Outcomes")
    add_body(
        doc,
        "PBL-3 delivered a working research-oriented multi-factor decision codebase with clear safety defaults: "
        "no live broker orders, paper/demo isolation, and explainability that cannot override signals. The team "
        "implemented modelling, risk, backtesting, persistence, API and UI components sufficient for academic "
        "demonstration of the methodology. Limitations remaining at the end of PBL-3 include absence of verified "
        "live market operation, a Backtesting UI that still uses simulated showcase data, incomplete production "
        "API wiring for the Live tab on Vercel, and a known weekly-feature causality risk requiring remediation.",
    )

    add_subheading(doc, "5.2 Tasks Planned for PBL-4")
    for b in [
        "Connect the Backtesting UI to FastAPI /api/backtest/simulate-strategies (or restore explicit DEMO labelling until then).",
        "Repair weekly feature alignment/look-ahead and add regression tests.",
        "Deploy the API with persistent disk and set VITE_API_BASE_URL plus CORS for the frontend.",
        "Pursue licensed live quote verification only if credentials and compliance requirements are met.",
        "Restore or redesign paper/research/analytics pages against existing APIs.",
        "Add API authentication, frontend CI build, and an NSE holiday calendar.",
    ]:
        add_bullet(doc, b)

    # References IEEE
    add_main_heading(doc, "References")
    refs = [
        '[1] M. López de Prado, Advances in Financial Machine Learning. Hoboken, NJ, USA: Wiley, 2018.',
        '[2] D. H. Bailey and M. López de Prado, “The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting, and non-normality,” J. Portfolio Manage., vol. 40, no. 5, pp. 94–107, 2014.',
        '[3] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in Proc. ICML, 2017, pp. 1321–1330.',
        '[4] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” J. Mach. Learn. Res., vol. 12, pp. 2825–2830, 2011.',
        '[5] FastAPI Developers, “FastAPI documentation,” [Online]. Available: https://fastapi.tiangolo.com/',
        '[6] TradingView, “Lightweight Charts documentation,” [Online]. Available: https://tradingview.github.io/lightweight-charts/',
        '[7] Ran Aroussi et al., “yfinance: Yahoo! Finance market data downloader,” [Online]. Available: https://github.com/ranaroussi/yfinance',
        '[8] National Stock Exchange of India, “Equity market timings (cash market session 09:15–15:30 IST),” [Online]. Available: https://www.nseindia.com/',
        '[9] Project repository documentation: README.md, ARCHITECTURE.md, API.md, DEPLOYMENT.md, and reports/Technical_Audit.md (internal audit companion, not part of this submission body).',
    ]
    for r in refs:
        add_body(doc, r, align="left", first_line=False)

    # Appendix
    add_main_heading(doc, "Appendix A – Evidence Notes")
    add_body(
        doc,
        "Automated checks recorded during report preparation: pytest reported 81 passed tests; the frontend "
        f"production build completed successfully; the deployed frontend at {VERCEL} responded over HTTPS. "
        "Local API availability may vary by machine. Companion internal files Technical_Audit.md, "
        "Feature_Verification_Matrix.md and Architecture_Documentation.md support faculty review but are not "
        "required sections of this university submission report.",
    )

    doc.save(path)


# ---------------------------------------------------------------------------
# PDF (same content, guideline formatting)
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
        # bottom-right continuous numbering from Introduction
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


def build_pdf(diags: dict, path: Path):
    pdf = UniPDF()
    pdf.set_margins(MARGIN_IN, MARGIN_IN, MARGIN_IN)
    pdf.set_auto_page_break(True, margin=MARGIN_IN + 0.15)

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
        if pdf.get_y() > pdf.h - MARGIN_IN - min_h:
            pdf.add_page()

    def table(caption_text, rows):
        ensure_space(1.2)
        caption(caption_text)
        for i, row in enumerate(rows):
            ensure_space(0.5)
            pdf.set_font("Times", "B" if i == 0 else "", 9)
            line = " | ".join(str(c) for c in row)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin, 0.18, _safe(line), align="L")
        pdf.ln(0.08)

    def figure(img: Path, caption_text: str, w=6.5):
        if img.exists():
            max_w = pdf.w - pdf.l_margin - pdf.r_margin
            w = min(w, max_w)
            ensure_space(3.4)
            pdf.set_x(pdf.l_margin)
            pdf.image(str(img), x=pdf.l_margin, w=w)
            pdf.ln(0.05)
        caption(caption_text)

    # Cover
    pdf.add_page()
    pdf.ln(1.0)
    center(P["university"], 16, True)
    center(P["department"], 12)
    center(P["programme"], 12)
    pdf.ln(0.3)
    center("A Project Report on", 12)
    center(P["title"], 16, True)
    center(f"Submitted in partial fulfilment of the requirements of {P['course']}", 12)
    pdf.ln(0.3)
    center("Submitted by", 12)
    center(P["students"], 12)
    pdf.ln(0.2)
    center("Under the guidance of", 12)
    center(P["guide"], 12)
    pdf.ln(0.3)
    center(f"Academic Year: {P['year']}", 12)
    center(f"Submission Date: {P['submission']}", 12)

    # Abstract
    pdf.add_page()
    h1("Abstract")
    body(CONTENT["abstract_p1"])
    body(CONTENT["abstract_p2"])

    # List of Tables
    pdf.add_page()
    h1("List of Tables")
    for item in [
        "Table 3.1  Hardware and software requirements",
        "Table 3.2  Default TECHNICAL_MODE parameters",
        "Table 3.3  Signal generation rules by strategy mode",
        "Table 4.1  PBL-3 milestone status summary",
        "Table 4.2  Distinction of data/operation modes",
    ]:
        body(item, align="L")

    # List of Figures
    pdf.add_page()
    h1("List of Figures")
    for item in [
        "Figure 3.1  System architecture block diagram",
        "Figure 3.2  Research decision data-flow diagram",
        "Figure 3.3  Module interaction overview",
        "Figure 4.1  Deployed Live Analysis interface",
        "Figure 4.2  Deployed Backtesting interface",
        "Figure 4.3  Gantt chart - PBL-3 completed and PBL-4 planned work",
    ]:
        body(item, align="L")

    # TOC
    pdf.add_page()
    h1("Table of Contents")
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

    # Body with numbering
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
    table(
        "Table 3.1  Hardware and software requirements",
        [
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
        ],
    )

    h2("3.3 Methodology")
    body(
        "Default runtime uses TECHNICAL_MODE and CORE_MODE. Daily OHLCV bars are enriched with MA20, MA50, RSI14, "
        "ROC, returns and short-horizon volatility, plus optional weekly context features. The binary label marks "
        "whether the forward return over seven trading days exceeds one percent. A HistGradientBoostingClassifier "
        "estimates class probability. Evaluation uses chronological purged splits and expanding walk-forward folds "
        "with an embargo equal to the label horizon. Model probability is not treated as probability of profit."
    )
    table(
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
    body(
        "Signal generation applies strategy rules after scoring. In technical trend mode, BUY requires probability "
        "above the buy threshold and Close above MA50; SELL requires probability below the sell threshold and Close "
        "below MA50; otherwise HOLD. Risk management computes RiskScore from volatility and drawdown, shrinks "
        "position size, and can convert BUY to HOLD when hard stops trigger."
    )
    table(
        "Table 3.3  Signal generation rules by strategy mode",
        [
            ["Mode", "BUY (summary)", "SELL (summary)"],
            ["technical_only", "P>0.6 and Close>MA50", "P<0.4 and Close<MA50"],
            ["momentum", "P + MA20>MA50 + RSI band", "P + downtrend filter"],
            ["mean_reversion", "z-score < -2", "z-score > +2"],
            ["multi_factor", "P threshold rule", "P threshold rule"],
        ],
    )
    body(
        "Backtesting assumes a long/cash book with fills at the next open and IndiaEquityCostConfig charges. "
        "Paper trading uses a separate ledger. AI explainability cannot override locked signals. The Live tab needs "
        "a configured API; the Backtesting tab currently generates simulated sleeve paths locally for demonstration."
    )
    body(
        "Working procedure for a typical research run is: select a symbol and date bounds; load or download "
        "candles; build features and labels; fit and score under purged walk-forward settings; attach signals; "
        "apply risk gating; simulate fills and costs; optionally request an explanation. For Live Analysis, the "
        "API scanner can iterate a watchlist on a timer when the backend process is running. For the current "
        "React Backtesting demonstration, the browser regenerates four sleeves plus a combined ensemble from a "
        "deterministic seed of ticker, dates and capital without calling the historical backtest engine."
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
    table(
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
            ["Production API attached to Vercel Live", "Not verified"],
        ],
    )
    table(
        "Table 4.2  Distinction of data/operation modes",
        [
            ["Mode", "Meaning in this project"],
            ["Historical / delayed", "Yahoo/store research bars; not verified live tape"],
            ["Simulated / demo", "Paper demo marks or React showcase equity paths"],
            ["LIVE LOCKED", "UI state when verified live feed is unavailable"],
            ["Verified live", "Not demonstrated in PBL-3"],
        ],
    )

    h2("4.2 Screenshots, sketches and flow diagrams")
    body(
        "Figures 4.1 and 4.2 are screenshots of the deployed React terminal. Live Analysis shows the locked live "
        "state. Backtesting shows simulated strategy sleeves for demonstration and must not be read as verified "
        "historical profitability."
    )
    figure(ASSETS / "live-analysis-vercel.png", "Figure 4.1  Deployed Live Analysis interface")
    figure(ASSETS / "backtesting-vercel.png", "Figure 4.2  Deployed Backtesting interface")
    body("Architecture and data-flow sketches appear as Figures 3.1-3.3 in Chapter 3.")

    h2("4.3 Timeline/Gantt Chart covering PBL-3 and planned PBL-4 phases")
    body(
        "Figure 4.3 presents a relative-week Gantt chart. Green bars denote completed PBL-3 work. Amber bars "
        "denote planned PBL-4 work."
    )
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
        '[9] Project repository documentation: README.md, ARCHITECTURE.md, API.md, DEPLOYMENT.md (and internal audit companions kept outside this submission body).',
    ]:
        body(r, align="L")

    h1("Appendix A - Evidence Notes")
    body(
        "Automated checks recorded during report preparation: pytest reported 81 passed tests; the frontend "
        f"production build completed successfully; the deployed frontend at {VERCEL} responded over HTTPS. "
        "Companion internal audit markdown files are not part of this university submission body."
    )

    pdf.output(str(path))
    return pdf.page_no(), pdf.body_page_offset


def verify_docx_structure(path: Path) -> dict:
    doc = Document(str(path))
    texts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    joined = "\n".join(texts)
    required = [
        "Abstract",
        "List of Tables",
        "List of Figures",
        "Table of Contents",
        "Chapter 1",
        "1.1 Introduction",
        "1.2 Problem Statement",
        "1.3 Objectives",
        "1.4 Scope of Project",
        "Chapter 3",
        "3.1 System Architecture",
        "3.2 Development Environment",
        "3.3 Methodology",
        "Chapter 4",
        "4.1 Tasks Achieved So Far",
        "4.2 Screenshots",
        "4.3 Timeline",
        "Chapter 5",
        "5.1 Summary",
        "5.2 Tasks Planned",
        "References",
    ]
    missing = [r for r in required if r not in joined]
    # crude page estimate: sections + breaks
    breaks = sum(1 for p in doc.paragraphs if "w:br" in p._p.xml and "type=\"page\"" in p._p.xml)
    return {"paragraphs": len(texts), "missing": missing, "page_breaks": breaks, "sections": len(doc.sections)}


def main():
    diags = ensure_diagrams()
    docx_path = ROOT / "PBL3_Project_Report.docx"
    pdf_path = ROOT / "PBL3_Project_Report.pdf"
    build_docx(diags, docx_path)
    total_pages, offset = build_pdf(diags, pdf_path)
    body_pages = total_pages - offset
    checks = verify_docx_structure(docx_path)

    print(f"Wrote {docx_path} ({docx_path.stat().st_size} bytes)")
    print(f"Wrote {pdf_path} ({pdf_path.stat().st_size} bytes)")
    print(f"PDF total pages: {total_pages}; body pages from Introduction: {body_pages}")
    print(f"DOCX sections: {checks['sections']}; missing headings: {checks['missing'] or 'none'}")
    if total_pages < 12:
        raise SystemExit(f"ERROR: PDF has only {total_pages} pages (<12 required)")
    if checks["missing"]:
        raise SystemExit(f"ERROR: missing required headings: {checks['missing']}")
    print("Verification OK")


if __name__ == "__main__":
    main()
