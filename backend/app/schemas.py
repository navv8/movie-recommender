"""Request/response models and the filter schema shared by the LLM layer and recommender."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Filters(BaseModel):
    """Structured intent extracted from a user's sentence (all fields optional)."""

    genres: list[str] = Field(default_factory=list)
    max_runtime: int | None = Field(default=None, ge=30, le=400)  # minutes
    min_year: int | None = Field(default=None, ge=1900, le=2100)
    max_year: int | None = Field(default=None, ge=1900, le=2100)
    similar_to: str | None = Field(default=None, max_length=200)  # a movie title


class RatingIn(BaseModel):
    movie_id: int  # the "id" field from movies.json
    rating: float = Field(ge=0.5, le=5.0)


class RecommendRequest(BaseModel):
    ratings: list[RatingIn] = Field(default_factory=list, max_length=500)
    filters: Filters | None = None
    limit: int = Field(default=20, ge=1, le=50)


class ChatRequest(BaseModel):
    ratings: list[RatingIn] = Field(default_factory=list, max_length=500)
    message: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=20, ge=1, le=50)


class RecommendResponse(BaseModel):
    results: list[dict]
    filters: dict
    notice: str | None = None


class ChatResponse(RecommendResponse):
    llm_ok: bool = True
