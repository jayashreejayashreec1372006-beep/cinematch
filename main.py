import pandas as pd
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
from recommender import (
    load_and_process, build_tfidf_model,
    build_nlp_model, recommend,
    get_trending, search_movies
)

app = FastAPI(title="CineMatch API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── STARTUP: load everything once ────────────────────────────────────────────
print("Loading dataset...")
df = load_and_process()

print("Building TF-IDF model...")
tfidf_sim, idx_map = build_tfidf_model(df)

print("Building NLP model (first run downloads ~80MB)...")
nlp_sim = build_nlp_model(df)

print("✅ CineMatch ready!")

# ── REQUEST MODELS ────────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    seed_title:    str
    liked:         List[str] = []
    disliked:      List[str] = []
    top_n:         int = 10
    tfidf_weight:  float = 0.4
    nlp_weight:    float = 0.4
    taste_weight:  float = 0.2

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "CineMatch API running"}

@app.post("/recommend")
def get_recommendations(req: RecommendRequest):
    results = recommend(
        seed_title    = req.seed_title,
        df            = df,
        tfidf_sim     = tfidf_sim,
        nlp_sim       = nlp_sim,
        idx_map       = idx_map,
        liked         = req.liked,
        disliked      = req.disliked,
        top_n         = req.top_n,
        tfidf_weight  = req.tfidf_weight,
        nlp_weight    = req.nlp_weight,
        taste_weight  = req.taste_weight,
    )
    return {"results": results}

@app.get("/search")
def search(q: str):
    return {"results": search_movies(q, df)}

@app.get("/trending")
def trending():
    return {"results": get_trending()}

@app.get("/movie/{title:path}")
def movie_detail(title: str):
    title = title.strip()
    key   = title.lower()

    # exact match
    if key in idx_map:
        row = df.iloc[idx_map[key]]
    else:
        # partial match
        matches = df[df["title"].str.lower().str.contains(
            re.escape(key[:15]), na=False, regex=True)]
        if matches.empty:
            # last resort — find closest by first word
            first_word = key.split()[0] if key.split() else key
            matches = df[df["title"].str.lower().str.startswith(first_word, na=False)]
        if matches.empty:
            return {"error": f"Not found: {title}"}
        row = matches.iloc[0]

    from recommender import fetch_poster_by_id
    poster_url = fetch_poster_by_id(int(row["id"]))

    return {
        "id":       int(row["id"]),
        "title":    str(row["title"]),
        "overview": str(row["overview"]) if pd.notna(row.get("overview")) else "No summary available.",
        "poster":   poster_url,
        "rating":   round(float(row.get("vote_average", 0) or 0), 1),
        "year":     str(row.get("release_date", ""))[:4],
        "genres":   row["genres_list"] if isinstance(row["genres_list"], list) else [],
        "cast":     row["cast_list"]   if isinstance(row["cast_list"],   list) else [],
        "director": str(row["director"]) if row.get("director") else "",
        "runtime":  int(row.get("runtime", 0) or 0),
        "budget":   int(row.get("budget",  0) or 0),
        "revenue":  int(row.get("revenue", 0) or 0),
    }
@app.get("/genre/{genre_name}")
def genre_movies(genre_name: str, limit: int = 15):
    """Return top rated movies of a specific genre."""
    mask    = df["genres_list"].apply(
        lambda x: any(genre_name.lower() in g.lower() for g in x) if isinstance(x, list) else False
    )
    results = df[mask].sort_values("vote_average", ascending=False).head(limit)
    out = []
    for _, row in results.iterrows():
        poster_url = fetch_poster_by_id(int(row["id"]))
        out.append({
            "id":      int(row["id"]),
            "title":   str(row["title"]),
            "overview": str(row["overview"]) if pd.notna(row.get("overview")) else "",
            "poster":  poster_url,
            "rating":  round(float(row.get("vote_average", 0) or 0), 1),
            "year":    str(row.get("release_date", ""))[:4],
            "genres":  row["genres_list"] if isinstance(row["genres_list"], list) else [],
            "cast":    row["cast_list"]   if isinstance(row["cast_list"],   list) else [],
            "director": str(row["director"]) if row.get("director") else "",
            "score":   0.9,
            "reason":  f"Top rated {genre_name} film",
        })
    return {"results": out}
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)