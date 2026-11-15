"""Restrained financial-terminal styling for Streamlit."""

from __future__ import annotations

import streamlit as st

NAV_SECTIONS = [
    "Overview",
    "Market Data",
    "Research & Predictions",
    "Backtesting / Reliability",
    "Paper Trading",
    "Settings / Diagnostics",
]


def apply_terminal_theme() -> None:
    st.markdown(
        """
<style>
:root {
  --mf-ink: #14213d;
  --mf-muted: #5c6b7a;
  --mf-panel: #f3f6f8;
  --mf-line: #d5dee7;
  --mf-accent: #0f6e56;
  --mf-warn: #9a3412;
  --mf-live: #047857;
  --mf-delayed: #b45309;
  --mf-demo: #1d4ed8;
}
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }
h1, h2, h3 { color: var(--mf-ink); letter-spacing: -0.02em; }
div[data-testid="stMetric"] {
  background: linear-gradient(180deg, #ffffff 0%, var(--mf-panel) 100%);
  border: 1px solid var(--mf-line);
  padding: 0.75rem 0.9rem;
  border-radius: 8px;
}
div[data-testid="stMetric"] label { color: var(--mf-muted) !important; }
.mf-banner {
  border: 1px solid var(--mf-line);
  background:
    radial-gradient(1200px 240px at 10% -20%, rgba(15,110,86,0.10), transparent 55%),
    linear-gradient(180deg, #ffffff, var(--mf-panel));
  border-radius: 12px;
  padding: 1.1rem 1.25rem;
  margin-bottom: 1rem;
}
.mf-banner h2 { margin: 0 0 0.35rem 0; font-size: 1.55rem; }
.mf-banner p { margin: 0; color: var(--mf-muted); }
.mf-badge {
  display: inline-block;
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  padding: 0.18rem 0.55rem;
  border-radius: 999px;
  border: 1px solid var(--mf-line);
  background: #fff;
  color: var(--mf-ink);
}
.mf-badge.live { color: var(--mf-live); border-color: #a7f3d0; background: #ecfdf5; }
.mf-badge.delayed, .mf-badge.historical, .mf-badge.cached { color: var(--mf-delayed); border-color: #fde68a; background: #fffbeb; }
.mf-badge.demo { color: var(--mf-demo); border-color: #bfdbfe; background: #eff6ff; }
.mf-badge.unconfigured, .mf-badge.unavailable, .mf-badge.closed { color: var(--mf-warn); border-color: #fed7aa; background: #fff7ed; }
.mf-disclaimer {
  color: var(--mf-muted);
  font-size: 0.85rem;
  border-top: 1px solid var(--mf-line);
  margin-top: 1.5rem;
  padding-top: 0.75rem;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def freshness_badge(label: str) -> str:
    key = str(label).strip().lower().replace(" ", "_")
    text = str(label).strip().upper()
    return f'<span class="mf-badge {key}">{text}</span>'


def section_nav(default: str = "Overview") -> str:
    with st.sidebar:
        st.markdown("### Navigation")
        choice = st.radio("Section", NAV_SECTIONS, index=NAV_SECTIONS.index(default) if default in NAV_SECTIONS else 0)
        st.caption("Research and paper trading are separate. Paper mode never sends broker orders.")
    return choice
