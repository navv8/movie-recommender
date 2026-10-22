"""
train.py - offline data prep, training and evaluation.

Run once on your laptop:

    cd backend
    pip install -r requirements-train.txt
    export TMDB_API_KEY=your_key        # or put TMDB_API_KEY=... in backend/.env
    python train.py

What it does
  1. Downloads MovieLens ml-latest-small into backend/data/ (not committed).
  2. Fetches overview, runtime, genres, year and poster from TMDB (cached).
  3. Keeps movies with >= 20 ratings, a non-empty overview and a runtime.
  4. Embeds each movie with all-MiniLM-L6-v2 (normalized vectors).
  5. Evaluates: RMSE (SVD vs global-mean baseline) and NDCG@10
     (content-only, SVD-only, hybrid for several k) on a held-out 20%.
  6. Refits SVD on ALL ratings and saves artifacts to backend/artifacts/:
       movies.json, embeddings.npy, svd_components.npy, metrics.json

Row i of embeddings.npy and svd_components.npy is movies.json[i].
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
SEED = 42
MIN_RATINGS = 20
N_FACTORS = 30
TEST_FRACTION = 0.2
LIKE_THRESHOLD = 4.0          # rating >= 4 counts as "liked" / relevant
TOP_N = 10                    # NDCG@10
K_VALUES = [2, 5, 10, 20, 50]  # candidates for the alpha constant k
EMBED_MODEL = "all-MiniLM-L6-v2"

ML_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
TMDB_URL = "https://api.themoviedb.org/3/movie/{}"

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
ARTIFACT_DIR = BASE / "artifacts"
TMDB_CACHE = DATA_DIR / "tmdb_cache.json"


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def load_dotenv_simple(path: Path) -> None:
    """Tiny .env reader so we don't need python-dotenv."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def log(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------------------------------------------------------
# 1. MovieLens
# ----------------------------------------------------------------------------
def download_movielens() -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    target = DATA_DIR / "ml-latest-small"
    if target.exists():
        return target
    log("Downloading MovieLens ml-latest-small ...")
    resp = requests.get(ML_URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(DATA_DIR)
    return target


# ----------------------------------------------------------------------------
# 2. TMDB
# ----------------------------------------------------------------------------
def fetch_one_tmdb(tmdb_id: int, api_key: str) -> dict | None:
    for attempt in range(4):
        try:
            r = requests.get(
                TMDB_URL.format(tmdb_id),
                params={"api_key": api_key, "language": "en-US"},
                timeout=15,
            )
            if r.status_code == 404:
                return {"missing": True}
            if r.status_code == 429:
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            d = r.json()
            return {
                "title": d.get("title") or "",
                "overview": (d.get("overview") or "").strip(),
                "runtime": d.get("runtime") or 0,
                "genres": [g["name"] for g in d.get("genres", [])],
                "release_date": d.get("release_date") or "",
                "poster_path": d.get("poster_path") or "",
            }
        except requests.RequestException:
            time.sleep(1.0 * (attempt + 1))
    return None


def fetch_tmdb(tmdb_ids: list[int], api_key: str) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if TMDB_CACHE.exists():
        cache = json.loads(TMDB_CACHE.read_text(encoding="utf-8"))
    todo = [i for i in tmdb_ids if str(i) not in cache]
    log(f"TMDB: {len(cache)} cached, {len(todo)} to fetch")

    if todo:
        done = 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            for tid, result in zip(todo, pool.map(lambda i: fetch_one_tmdb(i, api_key), todo)):
                if result is not None:  # None = network failure, retry on next run
                    cache[str(tid)] = result
                done += 1
                if done % 100 == 0:
                    log(f"  fetched {done}/{len(todo)}")
                    TMDB_CACHE.write_text(json.dumps(cache), encoding="utf-8")
        TMDB_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return cache


# ----------------------------------------------------------------------------
# 3. Build the movie table
# ----------------------------------------------------------------------------
def build_movies(ml_dir: Path, api_key: str):
    ratings = pd.read_csv(ml_dir / "ratings.csv")
    movies = pd.read_csv(ml_dir / "movies.csv")
    links = pd.read_csv(ml_dir / "links.csv").dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype(int)

    stats = ratings.groupby("movieId")["rating"].agg(n_ratings="count", avg_rating="mean")
    stats = stats[stats["n_ratings"] >= MIN_RATINGS].reset_index()
    df = stats.merge(movies, on="movieId").merge(links[["movieId", "tmdbId"]], on="movieId")
    log(f"{len(df)} movies have >= {MIN_RATINGS} ratings and a TMDB id")

    cache = fetch_tmdb(df["tmdbId"].tolist(), api_key)

    records = []
    for row in df.itertuples():
        meta = cache.get(str(row.tmdbId))
        if not meta or meta.get("missing"):
            continue
        if not meta["overview"] or not meta["runtime"]:
            continue
        year = int(meta["release_date"][:4]) if meta["release_date"][:4].isdigit() else None
        records.append(
            {
                "movieId": int(row.movieId),
                "tmdbId": int(row.tmdbId),
                "title": meta["title"] or row.title,
                "year": year,
                "runtime": int(meta["runtime"]),
                "genres": meta["genres"],
                "overview": meta["overview"],
                "poster_path": meta["poster_path"],
                "n_ratings": int(row.n_ratings),
                "avg_rating": round(float(row.avg_rating), 3),
            }
        )
    log(f"{len(records)} movies kept after overview/runtime filter")
    if not records:
        sys.exit("No movies kept - check your TMDB key.")

    kept_ids = {r["movieId"] for r in records}
    ratings = ratings[ratings["movieId"].isin(kept_ids)].copy()
    return records, ratings


# ----------------------------------------------------------------------------
# 4. Embeddings
# ----------------------------------------------------------------------------
def embed_movies(records: list[dict]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    log(f"Embedding {len(records)} movies with {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)
    texts = [f"{r['title']}. Genres: {', '.join(r['genres'])}. {r['overview']}" for r in records]
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    return emb.astype(np.float32)


# ----------------------------------------------------------------------------
# 5. SVD + evaluation
# ----------------------------------------------------------------------------
def build_matrix(df: pd.DataFrame, n_users: int, n_items: int) -> np.ndarray:
    R = np.zeros((n_users, n_items), dtype=np.float32)
    R[df["u"].to_numpy(), df["i"].to_numpy()] = df["rating"].to_numpy(dtype=np.float32)
    return R


def center(R: np.ndarray):
    """Subtract each user's mean (over rated items only). Unrated stay 0."""
    mask = R > 0
    counts = mask.sum(axis=1)
    means = np.where(counts > 0, R.sum(axis=1) / np.maximum(counts, 1), 0.0).astype(np.float32)
    C = np.where(mask, R - means[:, None], 0.0).astype(np.float32)
    return C, means, mask


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def ndcg_at_n(ranked_rel: np.ndarray, n_relevant: int, n: int = TOP_N) -> float:
    gains = ranked_rel[:n]
    dcg = float(np.sum(gains / np.log2(np.arange(2, len(gains) + 2))))
    ideal_len = min(n_relevant, n)
    idcg = float(np.sum(1.0 / np.log2(np.arange(2, ideal_len + 2))))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(ratings: pd.DataFrame, embeddings: np.ndarray, n_users: int, n_items: int):
    rng = np.random.default_rng(SEED)
    test_mask = rng.random(len(ratings)) < TEST_FRACTION
    train, test = ratings[~test_mask], ratings[test_mask]
    log(f"Split: {len(train)} train / {len(test)} test ratings")

    R_train = build_matrix(train, n_users, n_items)
    C_train, user_means, train_mask = center(R_train)

    svd = TruncatedSVD(n_components=N_FACTORS, random_state=SEED)
    svd.fit(C_train)
    V = svd.components_.T                      # (n_items, factors) item factors
    pred_centered = (C_train @ V) @ V.T        # fold-in then project back

    # ---- RMSE ---------------------------------------------------------------
    u, i = test["u"].to_numpy(), test["i"].to_numpy()
    truth = test["rating"].to_numpy()
    pred = np.clip(pred_centered[u, i] + user_means[u], 0.5, 5.0)
    rmse_svd = float(np.sqrt(np.mean((pred - truth) ** 2)))
    global_mean = float(train["rating"].mean())
    rmse_base = float(np.sqrt(np.mean((global_mean - truth) ** 2)))
    log(f"RMSE  SVD={rmse_svd:.4f}  global-mean baseline={rmse_base:.4f}")

    # ---- NDCG@10 ------------------------------------------------------------
    # content profile: rating-weighted average of liked train movies' embeddings
    liked_w = np.where(R_train >= LIKE_THRESHOLD, R_train, 0.0).astype(np.float32)
    has_liked = liked_w.sum(axis=1) > 0
    profile = liked_w @ embeddings
    norms = np.linalg.norm(profile, axis=1, keepdims=True)
    profile = profile / np.maximum(norms, 1e-12)
    content_all = profile @ embeddings.T       # cosine (embeddings are unit length)

    n_train = train_mask.sum(axis=1)
    test_by_user = test.groupby("u")

    variants = ["content", "svd"] + [f"hybrid_k{k}" for k in K_VALUES]
    scores_by_variant = {v: [] for v in variants}

    for user, group in test_by_user:
        relevant_items = set(group.loc[group["rating"] >= LIKE_THRESHOLD, "i"])
        if not relevant_items or not has_liked[user]:
            continue
        cand = np.where(~train_mask[user])[0]  # everything not rated in train
        rel = np.isin(cand, list(relevant_items)).astype(np.float64)

        s_content = minmax(content_all[user, cand])
        s_svd = minmax(pred_centered[user, cand])

        def score(s: np.ndarray) -> float:
            order = np.argsort(-s, kind="stable")
            return ndcg_at_n(rel[order], len(relevant_items))

        scores_by_variant["content"].append(score(s_content))
        scores_by_variant["svd"].append(score(s_svd))
        for k in K_VALUES:
            alpha = 1.0 / (1.0 + n_train[user] / k)
            scores_by_variant[f"hybrid_k{k}"].append(score(alpha * s_content + (1 - alpha) * s_svd))

    ndcg = {v: float(np.mean(s)) for v, s in scores_by_variant.items()}
    n_eval_users = len(scores_by_variant["content"])
    best_k = max(K_VALUES, key=lambda k: ndcg[f"hybrid_k{k}"])

    log(f"NDCG@{TOP_N} over {n_eval_users} users:")
    for v in variants:
        flag = "  <-- best k" if v == f"hybrid_k{best_k}" else ""
        log(f"  {v:<12} {ndcg[v]:.4f}{flag}")

    return {
        "rmse": {"svd": rmse_svd, "global_mean_baseline": rmse_base},
        "ndcg_at_10": ndcg,
        "best_k": best_k,
        "n_eval_users": n_eval_users,
        "n_train_ratings": int(len(train)),
        "n_test_ratings": int(len(test)),
    }


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main() -> None:
    load_dotenv_simple(BASE / ".env")
    api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        sys.exit("Set TMDB_API_KEY (environment variable or backend/.env).")

    np.random.seed(SEED)
    ml_dir = download_movielens()
    records, ratings = build_movies(ml_dir, api_key)

    # contiguous indices: item i == records[i]
    item_index = {r["movieId"]: i for i, r in enumerate(records)}
    user_ids = sorted(ratings["userId"].unique())
    user_index = {uid: u for u, uid in enumerate(user_ids)}
    ratings["i"] = ratings["movieId"].map(item_index)
    ratings["u"] = ratings["userId"].map(user_index)
    n_users, n_items = len(user_ids), len(records)
    log(f"{n_users} users x {n_items} movies, {len(ratings)} ratings")

    embeddings = embed_movies(records)

    metrics = evaluate(ratings, embeddings, n_users, n_items)

    # ---- final SVD on ALL ratings ------------------------------------------
    R = build_matrix(ratings, n_users, n_items)
    C, _, _ = center(R)
    svd = TruncatedSVD(n_components=N_FACTORS, random_state=SEED).fit(C)
    item_factors = svd.components_.T.astype(np.float32)   # (n_items, factors)

    metrics.update(
        {
            "n_movies": n_items,
            "n_users": n_users,
            "n_factors": N_FACTORS,
            "like_threshold": LIKE_THRESHOLD,
            "embedding_model": EMBED_MODEL,
            "explained_variance": float(svd.explained_variance_ratio_.sum()),
        }
    )

    ARTIFACT_DIR.mkdir(exist_ok=True)
    movies_out = [{"id": i, **r} for i, r in enumerate(records)]
    (ARTIFACT_DIR / "movies.json").write_text(json.dumps(movies_out, ensure_ascii=False), encoding="utf-8")
    np.save(ARTIFACT_DIR / "embeddings.npy", embeddings)
    np.save(ARTIFACT_DIR / "svd_components.npy", item_factors)
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    log(f"\nSaved artifacts to {ARTIFACT_DIR}")
    log("  movies.json, embeddings.npy, svd_components.npy, metrics.json")
    log(f"Set k = {metrics['best_k']} in the backend's alpha formula.")


if __name__ == "__main__":
    main()
