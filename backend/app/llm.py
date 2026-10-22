"""
Gemini layer: turns a sentence into structured filters via function calling.

The LLM never chooses movies. It only fills in the `set_filters` arguments, which are
then validated by Pydantic. Any failure returns None so the caller can fall back to
plain, unfiltered recommendations.
"""

from __future__ import annotations

import logging
import os
import re

from .schemas import Filters

log = logging.getLogger("llm")

MODELS = [m.strip() for m in os.environ.get("GEMINI_MODELS", "gemini-flash-lite-latest,gemini-flash-latest").split(",") if m.strip()]
TIMEOUT_MS = 12_000

SYSTEM_PROMPT = (
    "You convert a movie request into search filters by calling set_filters exactly once. "
    "Only include a filter if the user states it or clearly implies it. "
    "Never recommend or name movies other than a reference title the user mentions. "
    "Convert durations to minutes (2 hours = 120). "
    "Decades become year ranges (90s = 1990 to 1999). "
    "Use only the allowed genre names."
)

_client = None

_BOUNDS = {"max_runtime": (30, 400), "min_year": (1900, 2100), "max_year": (1900, 2100)}


def _clean(args: dict, allowed_genres: list[str]) -> dict:
    """Drop empty/zero values and clamp numbers, so a sloppy model reply doesn't fail validation."""
    out: dict = {}
    genres = [g for g in (args.get("genres") or []) if g in allowed_genres]
    if genres:
        out["genres"] = genres
    for key, (lo, hi) in _BOUNDS.items():
        try:
            v = int(args.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if v > 0:
            out[key] = min(max(v, lo), hi)
    title = str(args.get("similar_to") or "").strip()
    if title:
        out["similar_to"] = title
    return out


def _get_client():
    global _client
    if _client is None:
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        _client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=TIMEOUT_MS))
    return _client


def extract_filters(message: str, allowed_genres: list[str]) -> Filters | None:
    """Return validated Filters, or None if the LLM is unavailable or output is invalid."""
    try:
        from google.genai import types

        T = types.Type
        declaration = types.FunctionDeclaration(
            name="set_filters",
            description="Set the search filters extracted from the user's movie request.",
            parameters=types.Schema(
                type=T.OBJECT,
                properties={
                    "genres": types.Schema(
                        type=T.ARRAY,
                        items=types.Schema(type=T.STRING, enum=allowed_genres),
                        description="Genres the user asked for.",
                    ),
                    "max_runtime": types.Schema(
                        type=T.INTEGER, description="Maximum runtime in minutes."
                    ),
                    "min_year": types.Schema(type=T.INTEGER, description="Earliest release year."),
                    "max_year": types.Schema(type=T.INTEGER, description="Latest release year."),
                    "similar_to": types.Schema(
                        type=T.STRING,
                        description="Title of a movie the user wants something similar to.",
                    ),
                },
            ),
        )
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[types.Tool(function_declarations=[declaration])],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY", allowed_function_names=["set_filters"]
                )
            ),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=0.0,
        )
        response = None
        for model in MODELS:
            try:
                response = _get_client().models.generate_content(
                    model=model, contents=message, config=config
                )
                break
            except Exception as exc:
                log.warning("Gemini model %s failed: %r", model, exc)
        if response is None:
            return None

        for part in response.candidates[0].content.parts:
            call = getattr(part, "function_call", None)
            if call and call.name == "set_filters":
                return Filters.model_validate(_clean(dict(call.args or {}), allowed_genres))
        log.warning("Gemini returned no set_filters call")
    except Exception as exc:  # network, quota, bad output, validation...
        log.warning("Filter extraction failed: %r", exc)
    return None


_GENRE_WORDS = {
    "comed": "Comedy", "funny": "Comedy", "feel-good": "Comedy", "thriller": "Thriller",
    "sci-fi": "Science Fiction", "scifi": "Science Fiction", "science fiction": "Science Fiction",
    "animat": "Animation", "cartoon": "Animation", "family": "Family", "horror": "Horror",
    "romanc": "Romance", "romantic": "Romance", "action": "Action", "drama": "Drama",
    "crime": "Crime", "mystery": "Mystery", "fantasy": "Fantasy", "adventure": "Adventure",
    "war": "War", "documentar": "Documentary",
}


def rule_based_filters(message: str, allowed_genres: list[str]) -> Filters:
    """Simple keyword reading of the request, used only when Gemini is unavailable."""
    m = message.lower()
    out: dict = {}
    genres = []
    for word, genre in _GENRE_WORDS.items():
        if genre in allowed_genres and genre not in genres and re.search(rf"\b{re.escape(word)}", m):
            genres.append(genre)
    if genres:
        out["genres"] = genres
    t = re.search(r"(?:under|less than|below|within|max)\s+(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?)?", m)
    if t:
        value = float(t.group(1))
        mins = value * 60 if (t.group(2) or "").startswith("h") else value
        if 30 <= mins <= 400:
            out["max_runtime"] = int(mins)
    ref = re.search(r"\blike\s+(.+?)(?:\s+(?:under|with|from|before|after|and|for)\b|[,.]|$)", message, re.I)
    if ref:
        out["similar_to"] = ref.group(1).strip()
    return Filters.model_validate(out)


def get_filters(message: str, allowed_genres: list[str]) -> tuple[Filters, bool]:
    """Returns (filters, used_gemini). Falls back to the keyword reader if Gemini fails."""
    filters = extract_filters(message, allowed_genres)
    if filters is not None:
        return filters, True
    try:
        return rule_based_filters(message, allowed_genres), False
    except Exception:
        return Filters(), False
