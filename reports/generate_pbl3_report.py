#!/usr/bin/env python3
"""Generate PBL3_Project_Report.docx and .pdf from audited repository facts only."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from fpdf import FPDF

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
HEAD = "fe1619ad6683acdaadaecebbd25a7c0f0c24038b"
AUDIT_DATE = "16 November 2026"
VERCEL = "https://multi-factor-decision-system.vercel.app"


def _placeholders():
    return {
        "university": "[University Name]",
        "department": "[Department of Computer Science / Information Technology]",
        "programme": "Bachelor of Technology (B.Tech)",
        "course": "Project Based Learning – III (PBL-3)",
        "title": "Multi-Factor Financial Decision System",
        "students": "[Student Name(s) and Registration Number(s)]",
        "guide": "[Faculty Guide Name, Designation]",
        "hod": "[Head of Department Name]",
        "year": "[Academic Year]",
        "submission": "[Submission Date]",
    }


P = _placeholders()


# ---------------------------------------------------------------------------
# Content blocks shared by DOCX + PDF
# ---------------------------------------------------------------------------

SECTIONS = []


def add_section(number: str, title: str, paragraphs: list[str], bullets: list[str] | None = None, tables=None, figures=None):
    SECTIONS.append(
        {
            "number": number,
            "title": title,
            "paragraphs": paragraphs,
            "bullets": bullets or [],
            "tables": tables or [],
            "figures": figures or [],
        }
    )


add_section(
    "1",
    "Introduction",
    [
        f"This report documents the Multi-Factor Financial Decision System as implemented in the project repository at Git commit {HEAD[:12]}. "
        "The system is an academic research and simulation terminal for Indian equities. It combines technical feature engineering, "
        "supervised classification, purged walk-forward evaluation, risk gating, transaction-cost-aware backtesting, isolated paper trading, "
        "optional AI explainability, and a React operator interface.",
        "The project explicitly disables live broker order execution. Live NSE quote verification remains pending. "
        f"A static React frontend is publicly reachable at {VERCEL}; a production FastAPI backend URL was not verified during this audit.",
        "Unless otherwise stated, claims in this report are grounded in source inspection, automated tests (81 passed), a successful frontend production build, "
        "SQLite inspection, and screenshots of the deployed frontend.",
    ],
)

add_section(
    "2",
    "Problem Statement",
    [
        "Retail and academic traders often rely on ad-hoc technical indicators without chronological validation, without realistic costs, "
        "and without clear separation between delayed research data and verified live market data. Commercial platforms may obscure "
        "execution assumptions and encourage overconfidence in model probabilities.",
        "The problem addressed by this project is to design and implement a leakage-aware, inspectable multi-factor decision pipeline for NSE equities that: "
        "(i) produces BUY / SELL / HOLD decisions with documented rules; (ii) evaluates models with purged walk-forward methods; "
        "(iii) simulates fills at the next open with Indian cost assumptions; (iv) isolates paper trading from brokerage; and "
        "(v) exposes research capabilities through an API and a terminal UI—without claiming live brokerage readiness.",
    ],
)

add_section(
    "3",
    "Objectives and Scope",
    [
        "Objectives realised in code (to varying verification levels):",
    ],
    bullets=[
        "Build a technical-mode ML pipeline with daily features, optional weekly context, and a gradient-boosting classifier.",
        "Generate BUY / SELL / HOLD signals using probability thresholds and trend filters (MA50).",
        "Apply volatility and drawdown risk gates that can convert unsuitable BUY signals to HOLD.",
        "Backtest long/cash strategies with next-open fills and configurable Indian equity costs.",
        "Persist market candles, paper ledgers, and scan/backtest summaries in SQLite.",
        "Provide FastAPI endpoints and a React terminal for Live Analysis and Backtesting.",
        "Provide deterministic (and optional LLM) explanations that cannot override signals.",
    ],
)

add_section(
    "",
    "Scope exclusions (explicit)",
    [
        "The following are out of scope or not verified as production capabilities:",
    ],
    bullets=[
        "Live broker order routing or real-money execution.",
        "Verified live NSE tape during market hours (adapter present; feed not verified).",
        "Guaranteed profitability or investment advice.",
        "Fully featured three-tab React product (Portfolio tab removed; redirects only).",
        "Empirical interpretation of the current React Backtesting profit board (client-side synthetic data).",
    ],
)

add_section(
    "4",
    "Existing System and Literature / Technology Review",
    [
        "Traditional academic trading projects often use a single in-sample train/test split on price features, execute at the same-bar close, "
        "and omit costs. Such designs are vulnerable to look-ahead bias and optimistic Sharpe ratios (Bailey & López de Prado, 2014; López de Prado, 2018).",
        "Industry practice for supervised trading research emphasises purged cross-validation and embargo periods when labels depend on future returns "
        "(López de Prado, Advances in Financial Machine Learning, 2018). Calibration of probabilistic classifiers (isotonic / Platt scaling) is standard "
        "when thresholds are applied to predicted probabilities (Guo et al., 2017).",
        "Technology review of components used in this repository:",
    ],
    bullets=[
        "Python scientific stack: pandas, NumPy, scikit-learn (HistGradientBoostingClassifier).",
        "Market data: Yahoo Finance via yfinance for historical research (delayed/unofficial for live).",
        "Persistence: SQLite for market, paper, and scan databases.",
        "API: FastAPI + Uvicorn.",
        "UI: React + Vite + React Router; charts via TradingView lightweight-charts.",
        "Optional explainability: OpenAI-compatible chat completions with JSON responses.",
        "Deployment configs: Vercel (SPA), Render/Railway Docker templates.",
    ],
)

add_section(
    "5",
    "Proposed System",
    [
        "The proposed system is a layered architecture: (1) market-data ingestion and storage; (2) feature/label construction; "
        "(3) model training and walk-forward evaluation; (4) signal and risk layers; (5) backtest and paper engines; "
        "(6) FastAPI façade; (7) React terminal; (8) optional Streamlit academic UI.",
        "Default runtime mode sets TECHNICAL_MODE and CORE_MODE to True in core_config.py, collapsing the multi-model research stack "
        "to a single technical classifier for evaluation clarity. Full macro/micro/fusion paths remain in the repository but are not invoked on the default path.",
    ],
)

add_section(
    "6",
    "System Requirements",
    [
        "Software requirements observed in the repository:",
    ],
    bullets=[
        "Python 3.9+ (Dockerfile uses 3.11; CI uses 3.12).",
        "Node.js for frontend (Vite 8 / React).",
        "Dependencies: requirements.txt (FastAPI, scikit-learn, yfinance, pytest, …).",
        "Optional GPU: not required.",
        "Persistent disk recommended for production SQLite (Render disk / Railway volume).",
        "Environment variables as listed in .env.example (no secrets committed).",
    ],
    tables=[
        (
            "Table 1. Representative environment variables (names only)",
            [
                ["Variable", "Purpose"],
                ["MARKET_DATA_DB", "Path to market SQLite"],
                ["PAPER_TRADING_DB", "Path to paper SQLite"],
                ["SCAN_DB", "Path to scanner SQLite"],
                ["BROKER_LIVE_*", "Optional live quote adapter"],
                ["API_CORS_ORIGINS", "CORS allow-list"],
                ["AI_REASONING_*", "Optional LLM explainability"],
                ["VITE_API_BASE_URL", "Frontend → API base URL"],
                ["LIVE_SCANNER_ENABLED", "Background scanner on/off"],
            ],
        )
    ],
)

add_section(
    "7",
    "Overall Architecture",
    [
        "Figure 1 conceptually: React SPA ↔ (optional HTTPS) FastAPI ↔ Python modules ↔ SQLite stores. "
        "Detailed diagrams and data flows are provided in Architecture_Documentation.md.",
        "API version 0.9.0 starts a background MarketScanner on lifespan. broker_execution is hard-coded false in public runtime config.",
        "Observed deployment split: frontend on Vercel; backend intended for Render/Railway. Live Analysis on Vercel currently fails API calls when VITE_API_BASE_URL is unset.",
    ],
    figures=[("Figure 1", "assets/live-analysis-vercel.png", "Deployed Live Analysis page showing LIVE LOCKED and API connectivity errors without a backend.")],
)

add_section(
    "8",
    "Technology Stack",
    [],
    tables=[
        (
            "Table 2. Technology stack (as implemented)",
            [
                ["Layer", "Technology"],
                ["Language", "Python 3, TypeScript/React"],
                ["ML", "scikit-learn HistGradientBoostingClassifier"],
                ["API", "FastAPI 0.9.0 app, Uvicorn"],
                ["UI", "React Router, Vite, lightweight-charts"],
                ["Data", "yfinance, SQLite"],
                ["Tests", "pytest"],
                ["CI", "GitHub Actions (pytest only)"],
                ["Deploy configs", "Vercel, Docker, Render, Railway"],
            ],
        )
    ],
)

add_section(
    "9",
    "Module-wise Design and Implementation",
    [
        "Major Python packages and modules:",
    ],
    bullets=[
        "api/* — HTTP routers and settings.",
        "market_data/* — providers, ingest, store, session, quality.",
        "feature_engineering.py, model.py, model_evaluation.py — features, training, walk-forward.",
        "strategy.py, strategy_modes.py — signal rules.",
        "risk_model.py, transaction_costs.py — risk and costs.",
        "backtest.py — long/cash simulation.",
        "four_model_pipeline.py — orchestration (technical mode default).",
        "paper_trading/* — isolated ledger engine.",
        "reasoning/* — explainability.",
        "live_analysis/* — persistent scanner.",
        "strategy_research/* — regimes, portfolio, robustness.",
        "frontend/src/* — React terminal.",
    ],
)

add_section(
    "10",
    "Data Pipeline and Database Design",
    [
        "Historical candles are downloaded via yfinance and upserted into market.sqlite after OHLC quality checks. "
        "Research loading can merge stored bars with Yahoo backfill and apply as-of cutoffs. "
        "Paper and scan databases are separate files to isolate ledgers from market history.",
        "At audit time, market.sqlite contained candles for RELIANCE.NS, TCS.NS, HDFCBANK.NS and INFY.NS; the quotes table was empty; "
        "scans.sqlite contained zero completed runs.",
    ],
    tables=[
        (
            "Table 3. SQLite databases",
            [
                ["Database", "Primary tables"],
                ["market.sqlite", "candles, quotes, instruments, providers, quality_events, provider_events"],
                ["paper.sqlite", "accounts, orders, positions, ledger, closed_trades, equity_curve"],
                ["scans.sqlite", "scan_runs, scan_results, backtest_runs"],
            ],
        )
    ],
)

add_section(
    "11",
    "Machine Learning Methodology",
    [
        "Under TECHNICAL_MODE, features include MA20, MA50, Price_vs_MA50, RSI14, ROC_10, Daily_Return, Rolling_Volatility_10, "
        "and weekly context features TF_1wk_*. The binary target marks whether the forward return over CORE_HORIZON_BARS (7) exceeds TECHNICAL_MIN_FORWARD_RETURN (1%).",
        "The classifier is HistGradientBoosting (learning rate 0.05, max depth 4, max_iter 250). Evaluation uses purged chronological splits and expanding walk-forward folds with embargo equal to the label horizon. "
        "Calibration (isotonic or sigmoid) is fitted only on a pre-test slice when sample-size gates are met.",
        "Critical caveat: joining Yahoo weekly bars with merge_asof(direction='backward') without an additional causal shift can attach week-ending information to earlier weekdays. "
        "This look-ahead risk means default walk-forward metrics must be interpreted cautiously until fixed and regression-tested.",
        "Model probability is P(Target=1), not probability of net profit after costs.",
    ],
)

add_section(
    "12",
    "Financial Signal Generation",
    [
        "Default technical rule (strategy.attach_technical_trend_signals): BUY if probability exceeds 0.6 and Close > MA50; "
        "SELL if probability is below 0.4 and Close < MA50; otherwise HOLD. Momentum and mean-reversion modes add trend/RSI or z-score rules. "
        "Cross-sectional optional top-N BUY concentration is available (TECHNICAL_PORTFOLIO_TOP_N=3).",
        "Risk gating can rewrite BUY→HOLD when NoTrade triggers; SELL exits are not blocked by research NoTrade.",
    ],
)

add_section(
    "13",
    "Risk Management",
    [
        "Research RiskConfig (CORE_RISK_CONFIG): 20-day vol window, 60-day drawdown window, hard vol stop 0.07, hard drawdown stop 0.20, "
        "base position size 1.0, minimum 0.18. RiskScore blends scaled volatility and drawdown; PositionSize shrinks as RiskScore rises.",
        "PaperRiskConfig enforces max position value, order value, portfolio exposure, and daily loss, and disallows shorts by default. "
        "Audit note: paper daily-loss validation currently applies to SELL as well, which can block exits—recommended correction.",
    ],
)

add_section(
    "14",
    "Backtesting and Strategy Research",
    [
        "backtest.backtest_long_cash assumes a single long/cash book, fills at the next open, applies slippage and IndiaEquityCostConfig charges, "
        "and reports net/gross return, max drawdown, Sharpe, Sortino, CAGR, win rate, turnover, and related metrics.",
        "FastAPI exposes /api/backtest/run and /api/backtest/simulate-strategies with five presets. Audit found equity_curve serialization and window metric clipping weaknesses.",
        "The React Backtesting page does not call these endpoints. It renders buildShowcaseDemo synthetic sleeves. Therefore UI profit numbers are not empirical backtests. "
        "Strategy research modules (regimes, portfolio HHI, robustness) exist behind /api/analytics/* and are covered by tests, but lack React pages in the current UI.",
    ],
    figures=[("Figure 2", "assets/backtesting-vercel.png", "Deployed Backtesting UI (client-side strategy board).")],
)

add_section(
    "15",
    "Paper Trading",
    [
        "PaperTradingEngine maintains accounts paper:default and demo:* in paper.sqlite. Orders require confirmation flags and validate risk. "
        "Prices come from stored quotes/candles or fixed demo marks—not from a live order API. Broker execution remains disabled.",
        "React currently has no Paper Trading page; API wrappers exist unused in api.ts.",
    ],
)

add_section(
    "16",
    "AI Explainability",
    [
        "reasoning.service builds deterministic explanations from structured signal/risk/metric fields. Optional LLM enrichment uses an OpenAI-compatible endpoint when AI_REASONING_BASE_URL and AI_REASONING_API_KEY are set. "
        "validate_explanation enforces signal_lock and matching decision.signal. No path feeds explanations into trading actions.",
        "On the Live page, AiInsight may 404 until /api/research/run has populated an in-process cache—there is no UI control to run research.",
    ],
)

add_section(
    "17",
    "Frontend and User Experience",
    [
        "The React terminal provides two primary tabs: Live Analysis and Backtesting, plus a Settings drawer. "
        "Charts use TradingView lightweight-charts. Branding emphasises SIMULATED and NO LIVE ORDERS.",
        "Live Analysis correctly surfaces LIVE LOCKED when verified live tape is unavailable. Without a configured API base URL on Vercel, Live endpoints return 404.",
        "Backtesting UX allows selecting NSE tickers, dates, and capital, then regenerates four sleeves plus a combined ensemble. "
        "Presentation wording was adjusted (commit fe1619a) to remove DEMO headlines while retaining synthetic generation—an integrity risk for academic demonstration.",
    ],
)

add_section(
    "18",
    "Testing and Validation",
    [
        "Automated validation executed during this audit:",
    ],
    bullets=[
        "Command: .venv/bin/python -m pytest -q tests → 81 passed, 15 warnings, 5.02s.",
        "Command: cd frontend && npm run build → TypeScript + Vite production build succeeded.",
        "CI workflow runs pytest on Python 3.12; it does not build the frontend.",
    ],
    tables=[
        (
            "Table 4. Selected test modules",
            [
                ["Module", "Focus"],
                ["test_api.py / test_deployment_smoke.py", "API smoke"],
                ["test_live_backtest_api.py", "Live/backtest routes (mocked pipeline)"],
                ["test_reliability.py", "Walk-forward / reliability"],
                ["test_execution_risk_costs.py", "Costs and risk"],
                ["test_paper_trading.py", "Paper isolation"],
                ["test_reasoning.py", "Explainability lock"],
                ["test_market_*.py", "Store/ingest/research"],
                ["test_strategy_research.py", "Regimes/portfolio/robustness"],
            ],
        )
    ],
)

add_section(
    "19",
    "Results and Discussion",
    [
        "Empirical model metrics from a specific production ticker run are not reproduced in this report because the audit did not execute a fresh Yahoo-dependent research job under controlled leakage-fixed features. "
        "Offline synthetic tests pass and demonstrate pipeline mechanics.",
        "Deployed UI results: Backtesting displays deterministic synthetic profits for the selected window (example observation on 2026-11-16: Combined Ensemble profit figures on the order of tens of thousands of rupees on ₹100,000 capital). "
        "These figures must not be cited as market-validated performance.",
        "Live Analysis on Vercel demonstrates correct locking UX but cannot show ranked candidates without a backend.",
        "Discussion: the project successfully encodes many best-practice research controls (next-open fills, costs, purged WF, explainability lock, paper isolation). "
        "The principal threats to academic integrity are (1) UI synthetic backtests without clear labelling, (2) weekly feature causality, and (3) documentation drift.",
    ],
)

add_section(
    "20",
    "Limitations and Challenges",
    [],
    bullets=[
        "No verified live NSE feed; holiday calendar absent.",
        "Weekly feature look-ahead risk on default TECHNICAL_MODE path.",
        "React Backtesting not connected to historical engine.",
        "Three-tab product incomplete; paper/research/analytics UI removed.",
        "API unauthenticated; SQLite concurrency limited to single worker.",
        "Survivorship and index-membership drift not controlled for static universes.",
        "Yahoo data quality and corporate actions depend on third-party adjustments.",
        "In-process caches lost on restart.",
    ],
)

add_section(
    "21",
    "Security and Deployment",
    [
        "Secrets are expected via environment variables; .env is gitignored; diagnostics redact KEY/TOKEN values. "
        "Deployment artefacts include Dockerfile (one worker), render.yaml, railway.toml, Procfile, and frontend/vercel.json. "
        "DEPLOYMENT.md correctly states that live deploy success should not be claimed without checklist completion. "
        "This audit verified the Vercel frontend URL only; backend production health was not verified.",
    ],
)

add_section(
    "22",
    "Future Scope",
    [],
    bullets=[
        "Reconnect Backtesting UI to FastAPI simulate-strategies with honest historical metrics.",
        "Fix weekly feature causality and expand leakage tests.",
        "Complete live quote provider verification and holiday calendar.",
        "Restore or redesign Portfolio/Paper/Research pages against existing APIs.",
        "Add API authentication and frontend CI.",
        "Consider PostgreSQL for multi-instance deployments.",
        "Publish only leakage-safe, cost-aware walk-forward results in academic appendices.",
    ],
)

add_section(
    "23",
    "Conclusion",
    [
        "The Multi-Factor Financial Decision System is a substantial B.Tech PBL-3 implementation that demonstrates end-to-end research engineering for Indian equity decision support under explicit non-broker constraints. "
        "Backend modules for modelling, risk, costs, paper trading, and explainability are largely implemented and covered by an 81-test suite. "
        "The current React product focuses on Live Analysis and Backtesting; Live requires a deployed API, and Backtesting presently showcases synthetic strategy paths. "
        "With the corrective actions in the technical audit—especially UI honesty, feature causality, and API deployment—the project can serve as a strong academic demonstration of responsible quantitative software engineering.",
    ],
)

add_section(
    "24",
    "References",
    [
        "Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio. Journal of Portfolio Management.",
        "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.",
        "Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. ICML.",
        "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR.",
        "FastAPI documentation. https://fastapi.tiangolo.com/",
        "TradingView Lightweight Charts. https://tradingview.github.io/lightweight-charts/",
        "Yahoo Finance / yfinance library documentation (third-party historical data; not an exchange official feed).",
        "SEBI / NSE publicly documented cash-market session hours (09:15–15:30 IST) — used for session gating; holidays not encoded.",
        "Project internal documents: README.md, ARCHITECTURE.md, API.md, DEPLOYMENT.md (note: partially stale vs UI).",
        "Repository commit history through fe1619a (2026-11-16).",
    ],
)


APPENDIX_TEXT = [
    (
        "Appendix A — Feature Verification Summary",
        "See Feature_Verification_Matrix.md for the full matrix. Headline: Live NSE verified tape UNAVAILABLE; "
        "React Backtesting synthetic; backend research path IMPLEMENTED BUT NOT FULLY VERIFIED due to weekly leakage risk; "
        "paper isolation IMPLEMENTED AND VERIFIED; tests 81 passed.",
    ),
    (
        "Appendix B — Critical Findings",
        "See Technical_Audit.md severity register (A1–A14). Critical items: synthetic Backtesting presentation; weekly look-ahead.",
    ),
    (
        "Appendix C — Evidence Commands",
        "pytest -q tests → 81 passed; npm run build → success; curl Vercel → 200; local API :8000 unavailable at audit time.",
    ),
    (
        "Appendix D — Screenshots",
        "assets/live-analysis-vercel.png; assets/backtesting-vercel.png.",
    ),
]


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------

def set_doc_defaults(doc: Document):
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    style.font.size = Pt(12)


def add_centered(doc, text, *, bold=False, size=12, space_after=6):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    run.font.name = "Times New Roman"
    p.paragraph_format.space_after = Pt(space_after)
    return p


def add_heading_numbered(doc, number: str, title: str, level=1):
    text = f"{number} {title}".strip() if number else title
    return doc.add_heading(text, level=level)


def build_docx(path: Path):
    doc = Document()
    set_doc_defaults(doc)

    # Title page
    for _ in range(2):
        doc.add_paragraph()
    add_centered(doc, P["university"], bold=True, size=16)
    add_centered(doc, P["department"], size=12)
    add_centered(doc, P["programme"], size=12)
    doc.add_paragraph()
    add_centered(doc, "A Project Report on", size=12)
    add_centered(doc, P["title"], bold=True, size=18)
    add_centered(doc, f"Submitted in partial fulfilment of the requirements of {P['course']}", size=11)
    doc.add_paragraph()
    add_centered(doc, "Submitted by", size=12)
    add_centered(doc, P["students"], size=12)
    doc.add_paragraph()
    add_centered(doc, "Under the guidance of", size=12)
    add_centered(doc, P["guide"], size=12)
    doc.add_paragraph()
    add_centered(doc, f"Academic Year: {P['year']}", size=12)
    add_centered(doc, f"Submission Date: {P['submission']}", size=12)
    add_centered(doc, f"Repository audit commit: {HEAD[:12]}", size=10)
    add_centered(doc, f"Report generated: {AUDIT_DATE}", size=10)
    doc.add_page_break()

    # Certificate
    doc.add_heading("Certificate", level=1)
    doc.add_paragraph(
        f"This is to certify that the project entitled “{P['title']}” submitted by {P['students']} "
        f"in partial fulfilment of the requirements for the award of {P['programme']} "
        f"in {P['department']}, {P['university']}, is a bona fide record of work carried out under the guidance of {P['guide']}."
    )
    doc.add_paragraph()
    doc.add_paragraph(f"Guide: {P['guide']}                    Date: {P['submission']}")
    doc.add_paragraph(f"Head of Department: {P['hod']}")
    doc.add_paragraph("[College Seal / Official Stamp Placeholder]")
    doc.add_page_break()

    # Declaration
    doc.add_heading("Declaration", level=1)
    doc.add_paragraph(
        f"We hereby declare that the project work entitled “{P['title']}” submitted to {P['university']} "
        "is a record of original work done by us under the supervision of the faculty guide named above. "
        "The contents of this report are based on inspection of the implemented software repository and measured evidence. "
        "We have not fabricated experimental metrics. We understand that the system does not provide investment advice "
        "and does not execute live broker orders."
    )
    doc.add_paragraph()
    doc.add_paragraph(f"Name(s) & Signature(s): {P['students']}")
    doc.add_paragraph(f"Date: {P['submission']}")
    doc.add_page_break()

    # Acknowledgement
    doc.add_heading("Acknowledgement", level=1)
    doc.add_paragraph(
        f"We thank {P['guide']} for guidance, {P['hod']} and the faculty of {P['department']} for academic support, "
        "and open-source communities behind FastAPI, scikit-learn, React, and TradingView Lightweight Charts. "
        "Any remaining errors are our own. Placeholders above should be replaced with institutional details before binding."
    )
    doc.add_page_break()

    # Abstract
    doc.add_heading("Abstract", level=1)
    doc.add_paragraph(
        "This PBL-3 project implements a multi-factor financial decision system for Indian equities with emphasis on "
        "chronological integrity, cost-aware simulation, and operational safety. The software stack comprises a Python "
        "research engine, SQLite persistence, FastAPI services, and a React terminal. Default TECHNICAL_MODE trains a "
        "gradient-boosting classifier on technical features with a seven-day forward-return label, applies probability "
        "and moving-average signal rules, gates entries by volatility/drawdown risk, and backtests fills at the next open "
        "using Indian equity cost assumptions. Paper trading is ledger-isolated; AI explanations cannot override signals. "
        f"Automated tests report 81 passed. The deployed frontend ({VERCEL}) exposes Live Analysis and Backtesting tabs. "
        "Audit findings show that verified live NSE data is unavailable, the Live tab requires a configured API host, "
        "and the Backtesting tab currently renders client-side synthetic strategy paths rather than historical engine output. "
        "A weekly-feature look-ahead risk is identified in the default ML path. The report documents architecture, methodology, "
        "verification status, limitations, and recommended corrective actions."
    )
    doc.add_paragraph()
    p = doc.add_paragraph()
    r = p.add_run("Keywords: ")
    r.bold = True
    p.add_run(
        "algorithmic trading research; walk-forward validation; NSE equities; FastAPI; React; risk gating; paper trading; explainability"
    )
    doc.add_page_break()

    # TOC placeholder
    doc.add_heading("Table of Contents", level=1)
    doc.add_paragraph(
        "[In Microsoft Word: References → Table of Contents → Automatic Table. "
        "Section headings in this document are styled for TOC generation.]"
    )
    toc_items = [
        "Certificate",
        "Declaration",
        "Acknowledgement",
        "Abstract",
        "List of Figures",
        "List of Tables",
    ] + [f"{s['number']} {s['title']}".strip() for s in SECTIONS if s["number"]] + [
        "Appendix A — Feature Verification Summary",
        "Appendix B — Critical Findings",
        "Appendix C — Evidence Commands",
        "Appendix D — Screenshots",
    ]
    for item in toc_items:
        doc.add_paragraph(item)
    doc.add_page_break()

    doc.add_heading("List of Figures", level=1)
    doc.add_paragraph("Figure 1. Deployed Live Analysis page (Vercel) — LIVE LOCKED / API errors without backend.")
    doc.add_paragraph("Figure 2. Deployed Backtesting page (Vercel) — strategy sleeves and profit board.")
    doc.add_page_break()

    doc.add_heading("List of Tables", level=1)
    doc.add_paragraph("Table 1. Representative environment variables.")
    doc.add_paragraph("Table 2. Technology stack.")
    doc.add_paragraph("Table 3. SQLite databases.")
    doc.add_paragraph("Table 4. Selected test modules.")
    doc.add_paragraph("Table 5. Feature verification extract (see also Feature_Verification_Matrix.md).")
    doc.add_page_break()

    # Body
    for sec in SECTIONS:
        add_heading_numbered(doc, sec["number"], sec["title"], level=1)
        for para in sec["paragraphs"]:
            doc.add_paragraph(para)
        for b in sec["bullets"]:
            doc.add_paragraph(b, style="List Bullet")
        for caption, rows in sec["tables"]:
            doc.add_paragraph(caption).runs[0].bold = True
            table = doc.add_table(rows=len(rows), cols=len(rows[0]))
            table.style = "Table Grid"
            for i, row in enumerate(rows):
                for j, cell in enumerate(row):
                    table.rows[i].cells[j].text = str(cell)
            doc.add_paragraph()
        for fig_caption, rel, desc in sec["figures"]:
            img = ROOT / rel
            if img.exists():
                doc.add_picture(str(img), width=Inches(5.8))
            p = doc.add_paragraph(f"{fig_caption}. {desc}")
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Extra verification table
    doc.add_heading("Table 5. Feature verification extract", level=2)
    rows = [
        ["Feature", "Status"],
        ["FastAPI research/paper/explain APIs", "IMPLEMENTED AND VERIFIED (tests)"],
        ["Live NSE verified tape", "UNAVAILABLE"],
        ["React Live tab", "IMPLEMENTED; needs API host"],
        ["React Backtesting historical engine", "UNAVAILABLE (synthetic showcase)"],
        ["Risk blocks unsuitable BUY", "IMPLEMENTED AND VERIFIED"],
        ["Paper isolation", "IMPLEMENTED AND VERIFIED"],
        ["AI cannot override signals", "IMPLEMENTED AND VERIFIED"],
        ["Three-tab UI", "NOT IMPLEMENTED (two tabs)"],
        ["Frontend↔production backend", "UNAVAILABLE (404 on Live APIs)"],
    ]
    table = doc.add_table(rows=len(rows), cols=2)
    table.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            table.rows[i].cells[j].text = cell

    doc.add_page_break()
    doc.add_heading("Appendices", level=1)
    for title, body in APPENDIX_TEXT:
        doc.add_heading(title, level=2)
        doc.add_paragraph(body)

    doc.add_paragraph()
    note = doc.add_paragraph()
    r = note.add_run(
        "Companion artefacts: Technical_Audit.md, Feature_Verification_Matrix.md, Architecture_Documentation.md."
    )
    r.italic = True

    doc.save(path)


# ---------------------------------------------------------------------------
# PDF (fpdf2 — compatible with Python 3.9 used by this project's venv)
# ---------------------------------------------------------------------------


class PBL3PDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Times", "I", 9)
        self.cell(0, 10, f"PBL-3 | Multi-Factor Financial Decision System | Page {self.page_no()}", align="C")


def _ascii_safe(text: str) -> str:
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "-",
        "\u2192": "->",
        "\u2194": "<->",
        "\u00d7": "x",
        "\u20b9": "INR ",
        "\u2248": "~",
        "\u2260": "!=",
        "\u2713": "OK",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _pdf_p(pdf: FPDF, text: str, *, bold: bool = False, size: int = 11, align: str = "J", after: float = 3):
    pdf.set_font("Times", "B" if bold else "", size)
    pdf.multi_cell(0, 5.5, _ascii_safe(text), align=align)
    pdf.ln(after)


def build_pdf(path: Path):
    pdf = PBL3PDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    _pdf_p(pdf, P["university"], bold=True, size=16, align="C")
    _pdf_p(pdf, P["department"], size=12, align="C")
    _pdf_p(pdf, P["programme"], size=12, align="C")
    pdf.ln(6)
    _pdf_p(pdf, "A Project Report on", size=12, align="C")
    _pdf_p(pdf, P["title"], bold=True, size=18, align="C")
    _pdf_p(pdf, f"Submitted in partial fulfilment of {P['course']}", size=11, align="C")
    pdf.ln(6)
    _pdf_p(pdf, "Submitted by", size=12, align="C")
    _pdf_p(pdf, P["students"], size=12, align="C")
    _pdf_p(pdf, "Under the guidance of", size=12, align="C")
    _pdf_p(pdf, P["guide"], size=12, align="C")
    pdf.ln(6)
    _pdf_p(pdf, f"Academic Year: {P['year']}", size=11, align="C")
    _pdf_p(pdf, f"Submission Date: {P['submission']}", size=11, align="C")
    _pdf_p(pdf, f"Audit commit: {HEAD[:12]} | Generated: {AUDIT_DATE}", size=10, align="C")

    pdf.add_page()
    _pdf_p(pdf, "Certificate", bold=True, size=14, align="L")
    _pdf_p(
        pdf,
        f'This is to certify that the project entitled "{P["title"]}" submitted by {P["students"]} '
        f"is a bona fide record of work under the guidance of {P['guide']} at {P['university']}.",
    )
    _pdf_p(pdf, f"Guide: {P['guide']}")
    _pdf_p(pdf, f"HoD: {P['hod']}")
    _pdf_p(pdf, "[Seal Placeholder]")

    pdf.add_page()
    _pdf_p(pdf, "Declaration", bold=True, size=14, align="L")
    _pdf_p(
        pdf,
        f'We declare that "{P["title"]}" is based on the implemented repository and measured evidence. '
        "We have not fabricated metrics. The system does not execute live broker orders and is not investment advice.",
    )
    _pdf_p(pdf, f"Signatories: {P['students']}")

    pdf.add_page()
    _pdf_p(pdf, "Acknowledgement", bold=True, size=14, align="L")
    _pdf_p(
        pdf,
        f"We thank {P['guide']}, {P['hod']}, and open-source communities for FastAPI, scikit-learn, React, and Lightweight Charts.",
    )

    pdf.add_page()
    _pdf_p(pdf, "Abstract", bold=True, size=14, align="L")
    _pdf_p(
        pdf,
        "This PBL-3 project implements a multi-factor financial decision system for Indian equities emphasising chronological integrity, "
        "cost-aware simulation, and safety. A Python research engine, SQLite stores, FastAPI, and a React terminal are provided. "
        f"Tests: 81 passed. Frontend: {VERCEL}. Audit: live NSE tape unavailable; Live needs API; Backtesting UI is synthetic; "
        "weekly feature look-ahead risk identified. Companion markdown files detail verification and architecture.",
    )
    _pdf_p(
        pdf,
        "Keywords: walk-forward validation; NSE; FastAPI; React; risk gating; paper trading; explainability",
        bold=True,
    )

    pdf.add_page()
    _pdf_p(pdf, "Table of Contents", bold=True, size=14, align="L")
    for s in SECTIONS:
        label = f"{s['number']}. {s['title']}" if s["number"] else s["title"]
        _pdf_p(pdf, label, size=11, align="L", after=1)
    _pdf_p(pdf, "Appendices A-D", size=11, align="L")

    pdf.add_page()
    _pdf_p(pdf, "List of Figures and Tables", bold=True, size=14, align="L")
    _pdf_p(pdf, "Figure 1. Live Analysis (Vercel).", align="L")
    _pdf_p(pdf, "Figure 2. Backtesting (Vercel).", align="L")
    _pdf_p(pdf, "Tables 1-5 as in the DOCX body / companion matrix.", align="L")

    for sec in SECTIONS:
        pdf.add_page()
        heading = f"{sec['number']}. {sec['title']}" if sec["number"] else sec["title"]
        _pdf_p(pdf, heading, bold=True, size=13, align="L")
        for para in sec["paragraphs"]:
            _pdf_p(pdf, para)
        for b in sec["bullets"]:
            _pdf_p(pdf, f"- {b}", align="L", after=1)
        for caption, rows in sec["tables"]:
            _pdf_p(pdf, caption, bold=True, align="L")
            for row in rows:
                _pdf_p(pdf, " | ".join(str(c) for c in row), size=9, align="L", after=1)
            pdf.ln(2)
        for fig_caption, rel, desc in sec["figures"]:
            img_path = ROOT / rel
            if img_path.exists():
                # Keep image within page width
                pdf.image(str(img_path), w=170)
                pdf.ln(2)
            _pdf_p(pdf, f"{fig_caption}. {desc}", size=9, align="C")

    pdf.add_page()
    _pdf_p(pdf, "Appendices", bold=True, size=14, align="L")
    for title, body in APPENDIX_TEXT:
        _pdf_p(pdf, title, bold=True, size=12, align="L")
        _pdf_p(pdf, body)

    pdf.output(str(path))


def main():
    docx_path = ROOT / "PBL3_Project_Report.docx"
    pdf_path = ROOT / "PBL3_Project_Report.pdf"
    build_docx(docx_path)
    build_pdf(pdf_path)
    print(f"Wrote {docx_path}")
    print(f"Wrote {pdf_path}")
    for p in [
        docx_path,
        pdf_path,
        ROOT / "Technical_Audit.md",
        ROOT / "Feature_Verification_Matrix.md",
        ROOT / "Architecture_Documentation.md",
    ]:
        assert p.exists(), p
        print(f"OK {p.name} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
