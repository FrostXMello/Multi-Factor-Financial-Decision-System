"""
Fetch NIFTY Midcap 150 from NSE's public API and print a Python list (Yahoo .NS symbols).

Usage (from repo root):
    python scripts/fetch_nifty_midcap150.py

Paste the printed list into `utils.py`, replacing the existing `NIFTY_MIDCAP150_TICKERS` assignment.
"""
from __future__ import annotations

import json
import ssl
import urllib.request

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
URL = "https://www.nseindia.com/api/equity-stockindices?index=NIFTY%20MIDCAP%20150"


def main() -> None:
    ctx = ssl.create_default_context()
    req0 = urllib.request.Request("https://www.nseindia.com", headers=HEADERS, method="GET")
    req1 = urllib.request.Request(URL, headers=HEADERS, method="GET")
    with urllib.request.urlopen(req0, context=ctx, timeout=25) as _:
        pass
    with urllib.request.urlopen(req1, context=ctx, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8")).get("data") or []
    rows = [
        x
        for x in data
        if x.get("symbol") and "NIFTY" not in str(x.get("symbol", "")).upper()
    ]
    syms = sorted({str(x["symbol"]).strip().upper() + ".NS" for x in rows})
    print(f"# count = {len(syms)}")
    print("NIFTY_MIDCAP150_TICKERS: list[str] = [")
    for sym in syms:
        print(f'    "{sym}",')
    print("]")


if __name__ == "__main__":
    main()
