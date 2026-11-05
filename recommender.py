import pandas as pd
import numpy as np
import ast
import re
import requests
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()
TMDB_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE = "https://api.themoviedb.org/3"
POSTER_BASE = "https://image.tmdb.org/t/p/w500"

# ── HELPERS ──────────────────────────────────────────────────────────────────

def parse_list_col(text, key="name", limit=3):
    try:
        items = ast.literal_eval(text)
        return [i[key] for i in items[:limit]]
    except:
        return []

def get_director(crew_text):
    try:
        crew = ast.literal_eval(crew_text)
        for c in crew:
            if c.get("job") == "Director":
                return c.get("name", "")
    except:
        pass
    return ""

# ── LOAD & PROCESS DATA ───────────────────────────────────────────────────────

def load_and_process():
    movies  = pd.read_csv("data/tmdb_5000_movies.csv")
    credits = pd.read_csv("data/tmdb_5000_credits.csv")

    # credits has 'movie_id' or 'id' depending on version
    if "movie_id" in credits.columns:
        credits.rename(columns={"movie_id": "id"}, inplace=True)

    df = movies.merge(credits, on="id")

    # use title_x if merge created duplicate title columns
    if "title_x" in df.columns:
        df.rename(columns={"title_x": "title"}, inplace=True)
    if "title_y" in df.columns:
        df.drop(columns=["title_y"], inplace=True)

    df = df[df["overview"].notna()].reset_index(drop=True)

    df["genres_list"]   = df["genres"].apply(lambda x: parse_list_col(x))
    df["cast_list"]     = df["cast"].apply(lambda x: parse_list_col(x, limit=4))
    df["keywords_list"] = df["keywords"].apply(lambda x: parse_list_col(x, limit=8))
    df["director"]      = df["crew"].apply(get_director)

    # build soup for TF-IDF
    def make_soup(row):
        genres   = " ".join(row["genres_list"])
        cast     = " ".join([n.replace(" ","") for n in row["cast_list"]])
        director = row["director"].replace(" ","")
        keywords = " ".join(row["keywords_list"])
        overview = row["overview"] if isinstance(row["overview"], str) else ""
        # director repeated to give it more weight
        return f"{genres} {cast} {director} {director} {keywords} {overview}"

    df["soup"] = df.apply(make_soup, axis=1)

    return df

# ── BUILD MODELS ──────────────────────────────────────────────────────────────

def build_tfidf_model(df):
    tfidf  = TfidfVectorizer(stop_words="english", max_features=8000)
    matrix = tfidf.fit_transform(df["soup"])
    sim    = cosine_similarity(matrix, matrix)
    idx    = pd.Series(df.index, index=df["title"].str.lower()).drop_duplicates()
    return sim, idx

def build_nlp_model(df):
    """TF-IDF on overviews as lightweight NLP replacement."""
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = tfidf.fit_transform(df["overview"].fillna(""))
    nlp_sim = cosine_similarity(matrix, matrix)
    return nlp_sim

# ── TASTE PROFILE ─────────────────────────────────────────────────────────────

def build_taste_vector(liked_titles, disliked_titles, df, tfidf_sim, idx_map):
    """
    Build a score adjustment vector based on user ratings.
    Returns a per-movie boost/penalty array.
    """
    n = len(df)
    boost = np.zeros(n)

    for title in liked_titles:
        key = title.strip().lower()
        if key in idx_map:
            i = idx_map[key]
            boost += tfidf_sim[i] * 1.5   # boost similar movies

    for title in disliked_titles:
        key = title.strip().lower()
        if key in idx_map:
            i = idx_map[key]
            boost -= tfidf_sim[i] * 1.0   # penalize similar movies

    return boost

# ── CORE RECOMMEND ────────────────────────────────────────────────────────────
# cache to avoid repeated API calls
_poster_cache = {}

def fetch_poster_by_id(tmdb_id: int) -> str:
    """Fetch poster URL from TMDB API using movie ID."""
    if tmdb_id in _poster_cache:
        return _poster_cache[tmdb_id]
    try:
        url = f"{TMDB_BASE}/movie/{tmdb_id}"
        params = {"api_key": TMDB_KEY}
        r = requests.get(url, params=params, timeout=5)
        data = r.json()
        path = data.get("poster_path", None)
        if path:
            result = POSTER_BASE + path
        else:
            result = None
        _poster_cache[tmdb_id] = result
        return result
    except:
        return None
def recommend(
    seed_title: str,
    df: pd.DataFrame,
    tfidf_sim,
    nlp_sim,
    idx_map,
    liked: list = [],
    disliked: list = [],
    top_n: int = 10,
    tfidf_weight: float = 0.4,
    nlp_weight: float = 0.4,
    taste_weight: float = 0.2,
):
    key = seed_title.strip().lower()
    if key not in idx_map:
        return []

    idx = idx_map[key]

    # combine tfidf + nlp scores
    combined = tfidf_weight * tfidf_sim[idx] + nlp_weight * nlp_sim[idx]

    # add taste vector if user has rated movies
    if liked or disliked:
        taste = build_taste_vector(liked, disliked, df, tfidf_sim, idx_map)
        combined = combined + taste_weight * taste

    # sort and exclude seed itself
    scores = list(enumerate(combined))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx]

    results = []
    for i, score in scores[:top_n]:
        row = df.iloc[i]
        # NEW — fetch from TMDB API using movie ID
        poster_url = fetch_poster_by_id(int(row["id"]))
        # decide reason
        tfidf_s = tfidf_sim[idx][i]
        nlp_s   = nlp_sim[idx][i]
        if tfidf_s > 0.3 and nlp_s > 0.3:
            reason = "Similar plot, genre & cast"
        elif nlp_s > tfidf_s:
            reason = "Very similar storyline"
        else:
            reason = "Same genre & cast"

        results.append({
            "id":        int(row["id"]),
            "title":     row["title"],
            "overview":  row["overview"] if isinstance(row["overview"], str) else "",
            "poster":    poster_url,
            "rating":    round(float(row.get("vote_average", 0)), 1),
            "year":      str(row.get("release_date", ""))[:4],
            "genres":    row["genres_list"],
            "cast":      row["cast_list"],
            "director":  row["director"],
            "runtime":   int(row.get("runtime", 0) or 0),
            "budget":    int(row.get("budget", 0) or 0),
            "revenue":   int(row.get("revenue", 0) or 0),
            "score":     round(float(score), 4),
            "reason":    reason,
        })

    return results

# ── TRENDING FROM TMDB API ────────────────────────────────────────────────────

def get_trending():
    try:
        url = f"{TMDB_BASE}/trending/movie/week"
        r = requests.get(url, params={"api_key": TMDB_KEY}, timeout=6)
        items = r.json().get("results", [])[:10]
        return [{
            "id":       m["id"],
            "title":    m["title"],
            "overview": m.get("overview",""),
            "poster":   POSTER_BASE + m["poster_path"] if m.get("poster_path") else None,
            "rating":   m.get("vote_average", 0),
            "year":     str(m.get("release_date",""))[:4],
        } for m in items]
    except:
        return []

# ── SEARCH ────────────────────────────────────────────────────────────────────

def search_movies(query: str, df: pd.DataFrame, limit=8):
    q = query.lower().strip()
    mask = df["title"].str.lower().str.contains(q, na=False)
    results = df[mask].head(limit)
    out = []
    for _, row in results.iterrows():
        poster_path = row.get("poster_path", None)
        out.append({
            "id":      int(row["id"]),
            "title":   row["title"],
            "year":    str(row.get("release_date",""))[:4],
            "poster": fetch_poster_by_id(int(row["id"])),
            "rating":  round(float(row.get("vote_average", 0)), 1),
            "genres":  row["genres_list"],
        })
    return out
