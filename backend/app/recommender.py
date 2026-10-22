"""
Hybrid recommender: deterministic scoring from the artifacts saved by train.py.

  collaborative : fold-in of the visitor's ratings onto the saved SVD item factors
  content       : cosine similarity to a profile built from liked movies (+ reference title)
  hybrid        : both min-max normalized, blended with alpha = 1 / (1 + n_ratings / k)

Filters is duck-typed (genres, max_runtime, min_year, max_year, similar_to), so this
module needs only numpy.
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

import numpy as np

LIKE_THRESHOLD = 4.0
REFERENCE_MIN_ALPHA = 0.7  # "like X" queries lean on content similarity

GENRE_ALIASES = {
    "sci-fi": "science fiction",
    "scifi": "science fiction",
    "sci fi": "science fiction",
    "science-fiction": "science fiction",
    "sf": "science fiction",
    "animated": "animation",
    "cartoon": "animation",
    "kids": "family",
    "comedies": "comedy",
    "thrillers": "thriller",
    "romcom": "romance",
}


def _norm_title(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", text.lower())).strip()


def fmt_runtime(minutes: int) -> str:
    h, m = divmod(int(minutes), 60)
    return f"{h}h {m}m" if h else f"{m}m"


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / (hi - lo)


class Recommender:
    def __init__(self, artifacts_dir: str | Path):
        d = Path(artifacts_dir)
        self.movies: list[dict] = json.loads((d / "movies.json").read_text(encoding="utf-8"))
        self.emb = np.load(d / "embeddings.npy").astype(np.float32)
        self.V = np.load(d / "svd_components.npy").astype(np.float32)
        n = len(self.movies)
        if self.emb.shape[0] != n or self.V.shape[0] != n:
            raise ValueError("Artifacts are out of sync - rerun train.py")

        metrics_path = d / "metrics.json"
        self.k = 10
        if metrics_path.exists():
            self.k = int(json.loads(metrics_path.read_text(encoding="utf-8")).get("best_k", 10))

        self.years = np.array([m["year"] or 0 for m in self.movies])
        self.runtimes = np.array([m["runtime"] for m in self.movies])
        self.genre_sets = [{g.lower() for g in m["genres"]} for m in self.movies]
        self.all_genres = sorted({g for m in self.movies for g in m["genres"]})
        self._genre_lookup = {g.lower(): g for g in self.all_genres}
        self._titles = [(_norm_title(m["title"]), i) for i, m in enumerate(self.movies)]

        # Bayesian popularity (used when we know nothing about the visitor)
        n_r = np.array([m["n_ratings"] for m in self.movies], dtype=np.float64)
        avg = np.array([m["avg_rating"] for m in self.movies], dtype=np.float64)
        prior_w = 50.0
        self.popularity = (n_r * avg + prior_w * avg.mean()) / (n_r + prior_w)

    # ------------------------------------------------------------------ lookups
    def canonical_genre(self, name: str) -> str | None:
        key = name.strip().lower()
        key = GENRE_ALIASES.get(key, key)
        return self._genre_lookup.get(key)

    def find_title(self, query: str) -> int | None:
        q = _norm_title(query)
        if not q:
            return None
        for t, i in self._titles:
            if t == q:
                return i
        starts = [(t, i) for t, i in self._titles if t.startswith(q) or q.startswith(t)]
        if starts:
            return max(starts, key=lambda p: self.movies[p[1]]["n_ratings"])[1]
        close = difflib.get_close_matches(q, [t for t, _ in self._titles], n=1, cutoff=0.75)
        if close:
            return next(i for t, i in self._titles if t == close[0])
        return None

    @staticmethod
    def card(m: dict, reason: str | None = None) -> dict:
        return {
            "id": m["id"],
            "title": m["title"],
            "year": m["year"],
            "runtime": m["runtime"],
            "genres": m["genres"],
            "overview": m["overview"],
            "poster_path": m["poster_path"],
            "reason": reason,
        }

    # ---------------------------------------------------------------- onboarding
    def onboarding(self, count: int = 12) -> list[dict]:
        """Popular but varied: one movie per primary genre first, then fill by popularity."""
        pool = [m for m in self.movies if m["poster_path"]]
        pool.sort(key=lambda m: m["n_ratings"], reverse=True)
        picked, seen = [], set()
        for m in pool:
            g = m["genres"][0] if m["genres"] else ""
            if g not in seen:
                seen.add(g)
                picked.append(m)
            if len(picked) >= count:
                break
        for m in pool:
            if len(picked) >= count:
                break
            if m not in picked:
                picked.append(m)
        return [self.card(m) for m in picked[:count]]

    # ------------------------------------------------------------------- scoring
    def _content_scores(self, ratings: dict[int, float], ref_idx: int | None):
        vec = None
        liked = [(i, r) for i, r in ratings.items() if r >= LIKE_THRESHOLD]
        if liked:
            idx = np.array([i for i, _ in liked])
            w = np.array([r for _, r in liked], dtype=np.float32)
            vec = (w[:, None] * self.emb[idx]).sum(axis=0)
            vec /= max(np.linalg.norm(vec), 1e-12)
        if ref_idx is not None:
            ref = self.emb[ref_idx]
            vec = ref if vec is None else 0.35 * vec + 0.65 * ref
            vec = vec / max(np.linalg.norm(vec), 1e-12)
        return None if vec is None else self.emb @ vec

    def _collab_scores(self, ratings: dict[int, float]):
        if not ratings:
            return None
        idx = np.array(list(ratings.keys()))
        r = np.array(list(ratings.values()), dtype=np.float32)
        centered = r - r.mean()
        if np.allclose(centered, 0):
            return None  # all ratings equal: nothing to learn about preferences
        user_vec = centered @ self.V[idx]  # fold-in
        return self.V @ user_vec

    # ----------------------------------------------------------------- recommend
    def recommend(self, ratings: dict[int, float], filters=None, limit: int = 20):
        """Returns (cards, applied_filters, notes)."""
        n = len(self.movies)
        valid = {int(i): float(r) for i, r in ratings.items() if 0 <= int(i) < n}
        mask = np.ones(n, dtype=bool)
        for i in valid:
            mask[i] = False

        applied: dict = {"genres": [], "max_runtime": None, "min_year": None,
                         "max_year": None, "similar_to": None}
        notes: list[str] = []
        ref_idx = None

        if filters is not None:
            if getattr(filters, "similar_to", None):
                ref_idx = self.find_title(filters.similar_to)
                if ref_idx is None:
                    notes.append(f"Couldn't find \u201c{filters.similar_to}\u201d in our catalog.")
                else:
                    mask[ref_idx] = False
                    applied["similar_to"] = self.movies[ref_idx]["title"]
            for g in getattr(filters, "genres", None) or []:
                canon = self.canonical_genre(g)
                if canon and canon not in applied["genres"]:
                    applied["genres"].append(canon)
                    mask &= np.array([canon.lower() in s for s in self.genre_sets])
            if getattr(filters, "max_runtime", None):
                applied["max_runtime"] = int(filters.max_runtime)
                mask &= self.runtimes <= filters.max_runtime
            if getattr(filters, "min_year", None):
                applied["min_year"] = int(filters.min_year)
                mask &= self.years >= filters.min_year
            if getattr(filters, "max_year", None):
                applied["max_year"] = int(filters.max_year)
                mask &= self.years <= filters.max_year

        cand = np.where(mask)[0]
        if cand.size == 0:
            return [], applied, notes

        content = self._content_scores(valid, ref_idx)
        collab = self._collab_scores(valid)
        pop = _minmax(self.popularity[cand])

        if content is None and collab is None:
            score = pop
        elif content is None:
            score = _minmax(collab[cand])
        elif collab is None:
            score = _minmax(content[cand])
        else:
            alpha = 1.0 / (1.0 + len(valid) / self.k)
            if ref_idx is not None:
                alpha = max(alpha, REFERENCE_MIN_ALPHA)
            score = alpha * _minmax(content[cand]) + (1 - alpha) * _minmax(collab[cand])
        score = score + 1e-3 * pop  # deterministic tie-break

        top = cand[np.argsort(-score, kind="stable")[:limit]]

        # reasons
        liked_genres: dict[str, int] = {}
        for i, r in valid.items():
            if r >= LIKE_THRESHOLD:
                for g in self.movies[i]["genres"]:
                    liked_genres[g] = liked_genres.get(g, 0) + 1
        ranked_liked = sorted(liked_genres, key=lambda g: -liked_genres[g])

        cards = []
        for i in top:
            m = self.movies[int(i)]
            rt = fmt_runtime(m["runtime"])
            shared = [g for g in ranked_liked if g in m["genres"]]
            if applied["similar_to"]:
                reason = f"Similar to {applied['similar_to']} \u00b7 {rt}"
            elif shared:
                reason = f"Matches your taste in {shared[0].lower()} \u00b7 {rt}"
            elif applied["genres"]:
                reason = f"A {applied['genres'][0].lower()} pick \u00b7 {rt}"
            else:
                reason = f"Popular with viewers \u00b7 {rt}"
            cards.append(self.card(m, reason))
        return cards, applied, notes
