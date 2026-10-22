"""FastAPI server. Start (from backend/):  uvicorn app.main:app --reload"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .llm import get_filters
from .recommender import Recommender
from .schemas import (
    ChatRequest,
    ChatResponse,
    RecommendRequest,
    RecommendResponse,
)


def _load_env() -> None:
    """Read backend/.env for local development (hosts set real env vars instead)."""
    path = Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env()

ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", Path(__file__).resolve().parent.parent / "artifacts"))
rec = Recommender(ARTIFACTS)

app = FastAPI(title="Conversational Hybrid Recommender")

origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _ratings_dict(items) -> dict[int, float]:
    return {r.movie_id: r.rating for r in items}


@app.get("/health")
def health():
    return {"status": "ok", "movies": len(rec.movies)}


@app.get("/onboarding")
def onboarding(count: int = 12):
    return {"movies": rec.onboarding(max(1, min(count, 30)))}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    cards, applied, notes = rec.recommend(_ratings_dict(req.ratings), req.filters, req.limit)
    return RecommendResponse(results=cards, filters=applied, notice=" ".join(notes) or None)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    filters, llm_ok = get_filters(req.message, rec.all_genres)
    cards, applied, notes = rec.recommend(_ratings_dict(req.ratings), filters, req.limit)
    if not llm_ok:
        notes.insert(0, "Smart search is busy, so we used a simpler reading of your request.")
    return ChatResponse(
        results=cards, filters=applied, notice=" ".join(notes) or None, llm_ok=llm_ok
    )
