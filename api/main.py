"""
FastAPI entrypoint.

Local:
  uvicorn api.main:app --reload --port 8000

Production (Render/Railway):
  uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 1

This API wraps existing Python modules. It does not place real broker orders.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.deps import clear_store_caches, cors_origins, get_market_store, get_paper_store
from api.routes_analytics import router as analytics_router
from api.routes_explain import router as explain_router
from api.routes_market import router as market_router
from api.routes_overview import router as overview_router
from api.routes_paper import router as paper_router
from api.routes_research import router as research_router
from api.settings import ensure_persistent_paths, public_runtime_config


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_persistent_paths()
    clear_store_caches()
    # Open stores once so schema migrations run at boot.
    get_market_store()
    get_paper_store()
    yield


app = FastAPI(
    title="Multi-Factor Financial Decision System API",
    version="0.8.0",
    description=(
        "Research, market-data, paper-trading, explainability, and strategy analytics API. "
        "Demo mode is explicitly labelled. Real broker execution is disabled. "
        "Explainability cannot override signals or risk decisions."
    ),
    lifespan=lifespan,
)

_origins = cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins if _origins != ["*"] else ["*"],
    allow_credentials=_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(overview_router)
app.include_router(market_router)
app.include_router(paper_router)
app.include_router(research_router)
app.include_router(explain_router)
app.include_router(analytics_router)


@app.get("/")
def root() -> dict:
    return {
        "service": "multi-factor-financial-decision-system",
        "docs": "/docs",
        "health": "/api/health",
        "ready": "/api/health/ready",
        "broker_execution": False,
        "version": app.version,
    }
