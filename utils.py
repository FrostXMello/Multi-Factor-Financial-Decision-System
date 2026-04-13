from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import pandas as pd

# Best-effort default universe (Yahoo Finance tickers, NSE suffix).
# Note: NIFTY constituents can change over time; the app will skip tickers
# that fail to download so you can edit the list in the UI if needed.
NIFTY50_TICKERS: list[str] = [
    "ADANIPORTS.NS",
    "ASIANPAINT.NS",
    "AXISBANK.NS",
    "BAJAJ-AUTO.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "BPCL.NS",
    "BHARTIARTL.NS",
    "BRITANNIA.NS",
    "CIPLA.NS",
    "COALINDIA.NS",
    "DIVISLAB.NS",
    "DRREDDY.NS",
    "EICHERMOT.NS",
    "GRASIM.NS",
    "HCLTECH.NS",
    "HDFCBANK.NS",
    "HDFCLIFE.NS",
    "HEROMOTOCO.NS",
    "HINDALCO.NS",
    "HINDUNILVR.NS",
    "ICICIBANK.NS",
    "INDUSINDBK.NS",
    "INFY.NS",
    "ITC.NS",
    "JSWSTEEL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "M&M.NS",
    "MARUTI.NS",
    "NESTLEIND.NS",
    "NTPC.NS",
    "ONGC.NS",
    "POWERGRID.NS",
    "RELIANCE.NS",
    "SBILIFE.NS",
    "SBIN.NS",
    "SHREECEM.NS",
    "SUNPHARMA.NS",
    "TATACONSUM.NS",
    "TATAMOTORS.NS",
    "TATASTEEL.NS",
    "TATAPOWER.NS",
    "TECHM.NS",
    "TCS.NS",
    "TITAN.NS",
    "ULTRACEMCO.NS",
    "UPL.NS",
    "VEDL.NS",
    "WIPRO.NS",
]

# NIFTY Midcap 150 (NSE composition; rebalance periodically — refresh from NSE / `scripts/fetch_nifty_midcap150.py`).
NIFTY_MIDCAP150_TICKERS: list[str] = [
    "360ONE.NS",
    "3MINDIA.NS",
    "ABBOTINDIA.NS",
    "ABCAPITAL.NS",
    "ACC.NS",
    "AIAENG.NS",
    "AJANTPHARM.NS",
    "ALKEM.NS",
    "APARINDS.NS",
    "APLAPOLLO.NS",
    "APOLLOTYRE.NS",
    "ASHOKLEY.NS",
    "ASTRAL.NS",
    "ATGL.NS",
    "AUBANK.NS",
    "AUROPHARMA.NS",
    "AWL.NS",
    "BALKRISIND.NS",
    "BANKINDIA.NS",
    "BDL.NS",
    "BERGEPAINT.NS",
    "BHARATFORG.NS",
    "BHARTIHEXA.NS",
    "BHEL.NS",
    "BIOCON.NS",
    "BLUESTARCO.NS",
    "BSE.NS",
    "COCHINSHIP.NS",
    "COFORGE.NS",
    "COLPAL.NS",
    "CONCOR.NS",
    "COROMANDEL.NS",
    "CRISIL.NS",
    "CUMMINSIND.NS",
    "DABUR.NS",
    "DALBHARAT.NS",
    "DEEPAKNTR.NS",
    "DIXON.NS",
    "ENDURANCE.NS",
    "ESCORTS.NS",
    "EXIDEIND.NS",
    "FACT.NS",
    "FEDERALBNK.NS",
    "FLUOROCHEM.NS",
    "FORTIS.NS",
    "GICRE.NS",
    "GLAXO.NS",
    "GLENMARK.NS",
    "GMRAIRPORT.NS",
    "GODFRYPHLP.NS",
    "GODREJIND.NS",
    "GODREJPROP.NS",
    "GUJGASLTD.NS",
    "GVT&D.NS",
    "HDFCAMC.NS",
    "HEROMOTOCO.NS",
    "HEXT.NS",
    "HINDPETRO.NS",
    "HONAUT.NS",
    "HUDCO.NS",
    "ICICIPRULI.NS",
    "IDBI.NS",
    "IDEA.NS",
    "IDFCFIRSTB.NS",
    "IGL.NS",
    "INDIANB.NS",
    "INDUSINDBK.NS",
    "INDUSTOWER.NS",
    "IOB.NS",
    "IPCALAB.NS",
    "IRB.NS",
    "IRCTC.NS",
    "IREDA.NS",
    "ITCHOTELS.NS",
    "JKCEMENT.NS",
    "JSL.NS",
    "JSWINFRA.NS",
    "JUBLFOOD.NS",
    "KALYANKJIL.NS",
    "KEI.NS",
    "KPITTECH.NS",
    "KPRMILL.NS",
    "LICHSGFIN.NS",
    "LINDEINDIA.NS",
    "LLOYDSME.NS",
    "LTF.NS",
    "LTTS.NS",
    "LUPIN.NS",
    "M&MFIN.NS",
    "MAHABANK.NS",
    "MANKIND.NS",
    "MARICO.NS",
    "MEDANTA.NS",
    "MFSL.NS",
    "MOTILALOFS.NS",
    "MPHASIS.NS",
    "MRF.NS",
    "MUTHOOTFIN.NS",
    "NAM-INDIA.NS",
    "NATIONALUM.NS",
    "NHPC.NS",
    "NIACL.NS",
    "NLCINDIA.NS",
    "NMDC.NS",
    "NTPCGREEN.NS",
    "NYKAA.NS",
    "OBEROIRLTY.NS",
    "OFSS.NS",
    "OIL.NS",
    "PAGEIND.NS",
    "PATANJALI.NS",
    "PAYTM.NS",
    "PERSISTENT.NS",
    "PETRONET.NS",
    "PGHH.NS",
    "PHOENIXLTD.NS",
    "PIIND.NS",
    "POLICYBZR.NS",
    "POLYCAB.NS",
    "POWERINDIA.NS",
    "PREMIERENE.NS",
    "PRESTIGE.NS",
    "RVNL.NS",
    "SAIL.NS",
    "SBICARD.NS",
    "SCHAEFFLER.NS",
    "SJVN.NS",
    "SONACOMS.NS",
    "SRF.NS",
    "SUNDARMFIN.NS",
    "SUPREMEIND.NS",
    "SUZLON.NS",
    "SWIGGY.NS",
    "SYNGENE.NS",
    "TATACOMM.NS",
    "TATAELXSI.NS",
    "TATAINVEST.NS",
    "TATATECH.NS",
    "THERMAX.NS",
    "TIINDIA.NS",
    "TORNTPOWER.NS",
    "UBL.NS",
    "UCOBANK.NS",
    "UNIONBANK.NS",
    "UNOMINDA.NS",
    "UPL.NS",
    "VMM.NS",
    "VOLTAS.NS",
    "WAAREEENER.NS",
    "YESBANK.NS",
]


def parse_tickers_text(tickers_text: str) -> list[str]:
    """
    Parse tickers from a comma/newline/space-separated text input.
    """
    if not tickers_text or not tickers_text.strip():
        return []
    raw = (
        tickers_text.replace("\n", ",")
        .replace("\t", ",")
        .replace(" ", ",")
        .split(",")
    )
    tickers = [t.strip().upper() for t in raw if t and t.strip()]
    # Deduplicate while preserving order.
    seen = set()
    out = []
    for t in tickers:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility (best-effort)."""
    random.seed(seed)
    np.random.seed(seed)


def validate_columns(df: pd.DataFrame, required: Iterable[str], *, df_name: str = "DataFrame") -> None:
    """Raise a helpful error if expected columns are missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing columns: {missing}")

