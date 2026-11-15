"""
FastAPI entrypoint.

Run:
  uvicorn api.main:app --reload --port 8000

This API wraps existing Python modules. It does not place real broker orders.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.deps import cors_origins
from api.routes_market import router as market_router
from api.routes_overview import router as overview_router
from api.routes_paper import router as paper_router
from api.routes_research import router as research_router

app = FastAPI(
    title="Multi-Factor Financial Decision System API",
    version="0.5.0",
    description=(
        "Research, market-data, and paper-trading API. "
        "Demo mode is explicitly labelled. Real broker execution is disabled."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(overview_router)
app.include_router(market_router)
app.include_router(paper_router)
app.include_router(research_router)


@app.get("/")
def root() -> dict:
    return {
        "service": "multi-factor-financial-decision-system",
        "docs": "/docs",
        "health": "/api/health",
        "broker_execution": False,
    }
