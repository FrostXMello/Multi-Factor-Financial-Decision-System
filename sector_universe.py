"""
Sector indices (Yahoo) and liquid NSE stock buckets per sector for rotation scans.
"""

from __future__ import annotations

# Nifty sector indices on Yahoo Finance (verified download).
SECTOR_INDEX_YAHOO: dict[str, str] = {
    "BANK": "^NSEBANK",
    "IT": "^CNXIT",
    "FMCG": "^CNXFMCG",
    "AUTO": "^CNXAUTO",
    "PHARMA": "^CNXPHARMA",
    "METAL": "^CNXMETAL",
    "ENERGY": "^CNXENERGY",
}

SECTOR_DISPLAY_NAME: dict[str, str] = {
    "BANK": "NIFTY Bank",
    "IT": "NIFTY IT",
    "FMCG": "NIFTY FMCG",
    "AUTO": "NIFTY Auto",
    "PHARMA": "NIFTY Pharma",
    "METAL": "NIFTY Metal",
    "ENERGY": "NIFTY Energy",
}

# Curated liquid names per sector (Yahoo .NS). Keep 5–10; expand as needed.
STOCKS_BY_SECTOR: dict[str, list[str]] = {
    "BANK": [
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "KOTAKBANK.NS",
        "AXISBANK.NS",
        "INDUSINDBK.NS",
        "FEDERALBNK.NS",
        "BANKBARODA.NS",
        "BANDHANBNK.NS",
        "IDFCFIRSTB.NS",
    ],
    "IT": [
        "TCS.NS",
        "INFY.NS",
        "HCLTECH.NS",
        "WIPRO.NS",
        "TECHM.NS",
        "LTIM.NS",
        "PERSISTENT.NS",
        "COFORGE.NS",
        "MPHASIS.NS",
        "LTTS.NS",
    ],
    "FMCG": [
        "ITC.NS",
        "HINDUNILVR.NS",
        "NESTLEIND.NS",
        "BRITANNIA.NS",
        "DABUR.NS",
        "MARICO.NS",
        "GODREJCP.NS",
        "TATACONSUM.NS",
        "COLPAL.NS",
        "VBL.NS",
    ],
    "AUTO": [
        "MARUTI.NS",
        "M&M.NS",
        "TATAMOTORS.NS",
        "EICHERMOT.NS",
        "BAJAJ-AUTO.NS",
        "HEROMOTOCO.NS",
        "BOSCHLTD.NS",
        "UNOMINDA.NS",
        "MRF.NS",
        "TIINDIA.NS",
    ],
    "PHARMA": [
        "SUNPHARMA.NS",
        "DRREDDY.NS",
        "CIPLA.NS",
        "DIVISLAB.NS",
        "BIOCON.NS",
        "LUPIN.NS",
        "AUROPHARMA.NS",
        "TORNTPHARM.NS",
        "GLENMARK.NS",
        "ALKEM.NS",
    ],
    "METAL": [
        "TATASTEEL.NS",
        "HINDALCO.NS",
        "JSWSTEEL.NS",
        "VEDL.NS",
        "SAIL.NS",
        "NMDC.NS",
        "HINDZINC.NS",
        "NATIONALUM.NS",
        "JINDALSTEL.NS",
        "APLAPOLLO.NS",
    ],
    "ENERGY": [
        "RELIANCE.NS",
        "ONGC.NS",
        "OIL.NS",
        "BPCL.NS",
        "IOC.NS",
        "GAIL.NS",
        "PETRONET.NS",
        "HINDPETRO.NS",
        "ATGL.NS",
        "TATAPOWER.NS",
    ],
}


def all_sector_rotation_tickers() -> list[str]:
    syms: list[str] = []
    for lst in STOCKS_BY_SECTOR.values():
        syms.extend(lst)
    return list(dict.fromkeys(syms))
