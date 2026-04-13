"""
Backward-compatible alias: the canonical list is ``utils.NIFTY_MIDCAP150_TICKERS``.

To refresh constituents, run ``python scripts/fetch_nifty_midcap150.py`` and replace the
``NIFTY_MIDCAP150_TICKERS`` block in ``utils.py``.
"""

from __future__ import annotations

from utils import NIFTY_MIDCAP150_TICKERS

__all__ = ["NIFTY_MIDCAP150_TICKERS"]
