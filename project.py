"""
Full-featured AI Movie Recommender (OMDb-powered)
- Uses OMDb API for poster/rating/metadata (apikey provided)
- Content-based recommendations using local dataset (TF-IDF)
- Features: Search, Trending (from dataset), Similar movies, Trailers (YouTube search),
  Filters (min rating, year range, include adult), Favorites, Personalized picks,
  Upload dataset support, and polished UI with CSS.
"""

import os
import time
import urllib.parse
from typing import List, Dict, Optional

import requests
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from dateutil import parser as dateparser

# ------------------------
# Configuration / Constants
# ------------------------
st.set_page_config(page_title="CineMate — AI Movie Recommender", layout="wide", page_icon="🎬")
OMDB_API_KEY = "d0dea6bc"           # provided by you
OMDB_BASE = "http://www.omdbapi.com/"
LOCAL_DEFAULT_CSV = "movies.csv"    # optional local dataset
FALLBACK_DATA = [
    {"title": "Harry Potter and the Sorcerer's Stone", "genre": "Fantasy", "plot": "Wizard boy goes to magic school"},
    {"title": "The Lord of the Rings: The Fellowship of the Ring", "genre": "Fantasy", "plot": "Hobbits try to destroy a ring"},
    {"title": "Inception", "genre": "Sci-Fi", "plot": "A thief enters dreams to steal secrets"},
    {"title": "Avengers: Endgame", "genre": "Action", "plot": "Superheroes fight cosmic threat"},
    {"title": "Titanic", "genre": "Romance", "plot": "A love story on a sinking ship"},
]

# ------------------------
# Styling (keeps your Netflix-like look)
# ------------------------
st.markdown(
    """
    <style>
    :root { --radius: 14px; }
    .stApp { background: #000; color: #e6eef8; }
    .title { text-align:center; color:#E50914; font-size:48px; font-weight:800; margin:10px 0 6px; }
    .sub { text-align:center; color:#cbd5e1; margin-bottom:18px; }
    .hero { border-radius:12px; background: linear-gradient(90deg, rgba(0,0,0,0.9), rgba(0,0,0,0.4)), url('https://wallpapercave.com/wp/wp4056410.jpg'); height:260px; background-size:cover; display:flex; align-items:center; justify-content:center; margin-bottom:22px;}
    .hero-text { color:white; font-size:34px; font-weight:700; text-shadow: 2px 2px 8px #000; }
    .movie-card { background:#0f1724; border-radius:12px; padding:10px; overflow:hidden; border:1px solid #213043; }
    .poster { width:100%; border-radius:8px; object-fit:cover; }
    .movie-title { font-weight:700; font-size:1.05rem; color:#f1f5f9; }
    .meta { color:#94a3b8; font-size:0.9rem; margin-bottom:6px;}
    .chip { display:inline-block; margin-right:6px; padding:4px 8px; border-radius:999px; background:#0b1220; color:#cbd5e1; border:1px solid #263844; font-size:0.8rem;}
    .btn { padding:8px 12px; background:#0ea5e9; color:#fff; border-radius:8px; text-decoration:none; font-weight:600; }
    .badge-adult { background:#521212; color:#fecaca; padding:4px 8px; border-radius:8px; font-weight:700; }
    .grid { display:grid; grid-template-columns: repeat(12, 1fr); gap:16px; }
    .col-3 { grid-column: span 3; }
    @media (max-width: 1000px) { .col-3 { grid-column: span 6; } }
    @media (max-width: 700px) { .col-3 { grid-column: span 12; } }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------
# Helper: OMDb API calls
# ------------------------
def omdb_request(params: Dict) -> Optional[Dict]:
    params = params.copy()
    params["apikey"] = OMDB_API_KEY
    try:
        r = requests.get(OMDB_BASE, params=params, timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

def fetch_by_title(title: str) -> Optional[Dict]:
    if not title:
        return None
    data = omdb_request({"t": title, "plot": "short", "r": "json"})
    if data and data.get("Response") == "True":
        return data
    return None

def fetch_by_imdb_id(imdb_id: str) -> Optional[Dict]:
    if not imdb_id:
        return None
    data = omdb_request({"i": imdb_id, "plot": "short", "r": "json"})
    if data and data.get("Response") == "True":
        return data
    return None

def trailer_search_link(title: str) -> str:
    q = urllib.parse.quote_plus(f"{title} trailer")
    return f"https://www.youtube.com/results?search_query={q}"

# ------------------------
# Dataset loading + TF-IDF similarity
# ------------------------
@st.cache_data(show_spinner=False)
def load_local_movies(path: str = LOCAL_DEFAULT_CSV) -> pd.DataFrame:
    # Accept either movies.csv with columns title, genre, plot OR fallback to sample
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            # normalize column names
            lower_cols = {c.lower(): c for c in df.columns}
            if "series_title" in lower_cols and "title" not in df.columns:
                df = df.rename(columns={lower_cols["series_title"]: "title"})
            if "title" not in df.columns:
                # try first text-like column
                possible = [c for c in df.columns if "title" in c.lower()]
                if possible:
                    df = df.rename(columns={possible[0]: "title"})
            # ensure columns exist
            for c in ["title", "genre", "plot"]:
                if c not in df.columns:
                    df[c] = ""
            df["title"] = df["title"].astype(str)
            df["genre"] = df["genre"].astype(str)
            df["plot"] = df["plot"].astype(str)
            return df[["title", "genre", "plot"]].drop_duplicates().reset_index(drop=True)
        except Exception:
            pass
    # fallback
    return pd.DataFrame(FALLBACK_DATA)

@st.cache_resource(show_spinner=False)
def build_tfidf_model(df: pd.DataFrame):
    # combine textual fields
    df = df.copy()
    df["combined"] = (df["title"].fillna("") + " " + df["genre"].fillna("") + " " + df["plot"].fillna("")).astype(str)
    tfidf = TfidfVectorizer(stop_words="english", max_features=15000)
    mat = tfidf.fit_transform(df["combined"].values)
    sim = linear_kernel(mat, mat)  # cosine similarities
    index_map = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()
    return df, sim, index_map

# load dataset + model
local_df = load_local_movies()
local_df, similarity_matrix, local_indices = build_tfidf_model(local_df)

def recommend_from_dataset(seed_title: str, top_n: int = 8) -> List[str]:
    if not seed_title:
        return []
    idx = local_indices.get(seed_title.lower())
    if idx is None:
        # fuzzy fallback: try exact case-insensitive match search
        matches = local_df[local_df["title"].str.lower().str.contains(seed_title.lower())]
        if not matches.empty:
            idx = matches.index[0]
        else:
            return []
    sims = list(enumerate(similarity_matrix[idx]))
    sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)
    rec_indices = [i for i, s in sims_sorted[1:top_n+1]]
    return local_df.iloc[rec_indices]["title"].tolist()

# ------------------------
# Utilities: UI render helpers
# ------------------------
def compact_movie_card_from_omdb(omdb: Dict) -> str:
    # returns a chunk of HTML (safe) representing the movie card using OMDb data
    poster = omdb.get("Poster") if omdb.get("Poster") and omdb.get("Poster") != "N/A" else ""
    title = omdb.get("Title", "Untitled")
    year = omdb.get("Year", "")
    rated = omdb.get("Rated", "N/A")
    imdb_rating = omdb.get("imdbRating", "N/A")
    genre = omdb.get("Genre", "")
    plot = omdb.get("Plot", "")[:220] + ("…" if len(omdb.get("Plot", "")) > 220 else "")
    adult_flag = "🔞 18+" if rated in ("R", "NC-17") else "✅"
    trailer = trailer_search_link(title)
    html = f"""
    <div class="movie-card">
      <img class="poster" src="{poster}" alt="poster"/>
      <div style="padding:8px;">
        <div class="movie-title">{title}</div>
        <div class="meta">{year} • ⭐ {imdb_rating} • {adult_flag}</div>
        <div style="margin:6px 0;" class="small">{genre}</div>
        <div style="color:#cdd6e6; margin-bottom:8px;">{plot}</div>
        <div style="display:flex; gap:8px;">
          <a class="btn" href="{trailer}" target="_blank">▶ Trailer</a>
          <a class="btn" href="https://www.imdb.com/title/{omdb.get('imdbID')}" target="_blank">ℹ️ IMDb</a>
        </div>
      </div>
    </div>
    """
    return html

# ------------------------
# Sidebar Controls (filters, upload, favorites)
# ------------------------
st.sidebar.header("CineMate Controls")

# API key (kept hidden but show the OMDb key currently in use)
st.sidebar.caption("Using OMDb API (no config required).")

st.sidebar.markdown("### Filters")
min_imdb = st.sidebar.slider("Min IMDb rating", 0.0, 10.0, 5.5, 0.1)
include_adult = st.sidebar.checkbox("Include adult/18+ (R/NC-17)", value=False)
year_min, year_max = st.sidebar.slider("Year range", 1900, 2030, (1970, 2026))
genre_select = st.sidebar.text_input("Filter genre (comma separated, optional)", "")

st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset")
uploaded = st.sidebar.file_uploader("Upload CSV (title,genre,plot) to use as dataset", type=["csv"])
if uploaded:
    # save to local file and rebuild model
    try:
        uploaded_path = "uploaded_movies.csv"
        with open(uploaded_path, "wb") as f:
            f.write(uploaded.getbuffer())
        # reload dataset & rebuild model
        local_df = load_local_movies(uploaded_path)
        local_df, similarity_matrix, local_indices = build_tfidf_model(local_df)
        st.sidebar.success("Dataset uploaded and model rebuilt.")
    except Exception as e:
        st.sidebar.error(f"Failed to load dataset: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("### Favorites")
# session favorites: store imdbIDs where possible
if "favorites" not in st.session_state:
    st.session_state["favorites"] = []  # list of dicts: {"title":..., "imdbID":...}

def add_favorite(omdb_data: Dict):
    # if movie has imdbID, use that to avoid duplicates
    movie_id = omdb_data.get("imdbID") or omdb_data.get("Title")
    existing_ids = [m.get("id") for m in st.session_state["favorites"]]
    if movie_id not in existing_ids:
        st.session_state["favorites"].append({"id": movie_id, "title": omdb_data.get("Title"), "imdbID": omdb_data.get("imdbID")})
        st.toast(f"Added to favorites: {omdb_data.get('Title')}")

def remove_favorite(movie_id):
    st.session_state["favorites"] = [m for m in st.session_state["favorites"] if m.get("id") != movie_id]
    st.toast("Removed from favorites")

# display favorites list
if st.session_state["favorites"]:
    for fav in st.session_state["favorites"]:
        colf1, colf2 = st.sidebar.columns([3,1])
        with colf1:
            st.sidebar.markdown(f"• {fav.get('title')}")
        with colf2:
            if st.sidebar.button("✖", key=f"rm_{fav.get('id')}"):
                remove_favorite(fav.get("id"))
else:
    st.sidebar.info("No favorites yet — add from movie cards.")

# ------------------------
# Main layout: title + hero
# ------------------------
st.markdown('<div class="title">🍿 CineMate</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">AI Movie Recommender — OMDb + Local dataset</div>', unsafe_allow_html=True)
st.markdown('<div class="hero"><div class="hero-text">Find, Explore & Get Recommendations</div></div>', unsafe_allow_html=True)

# ------------------------
# Tabs (Trending, Search, Because you liked..., Favorites, Personalized, Dataset)
# ------------------------
tabs = st.tabs(["🔥 Trending", "🔎 Search", "🎯 Because you liked…", "⭐ Favorites", "👤 Personalized", "📂 My Dataset"])

# ------------------------
# Helper: pass filters applied by sidebar
# ------------------------
def passes_filters_omdb(omdb: Dict) -> bool:
    # imdb rating
    try:
        ir = float(omdb.get("imdbRating") or 0)
    except Exception:
        ir = 0.0
    if ir < min_imdb:
        return False
    # year
    try:
        y = int((omdb.get("Year") or "")[:4])
        if y < year_min or y > year_max:
            return False
    except Exception:
        pass
    # adult
    rated = omdb.get("Rated", "")
    if (rated in ("R", "NC-17")) and (not include_adult):
        return False
    # genre filter
    if genre_select:
        wanted = [g.strip().lower() for g in genre_select.split(",") if g.strip()]
        all_genres = [(omdb.get("Genre") or "").lower()]
        if not any(w in g for w in wanted for g in all_genres):
            return False
    return True

# ------------------------
# Tab 0: Trending (from dataset or fallback)
# ------------------------
with tabs[0]:
    st.subheader("🔥 Trending (from dataset sampling & OMDb enrichment)")
    # We'll pick top N titles from dataset by some heuristic: use those with longer plot (proxy for content) or random
    # Simpler: sample most frequent top titles, but we don't have popularity — so sample and then show OMDb info
    sample_titles = list(local_df["title"].sample(n=min(12, len(local_df)), random_state=42))
    cards_per_row = 4
    rows = [sample_titles[i:i+cards_per_row] for i in range(0, len(sample_titles), cards_per_row)]
    for row in rows:
        cols = st.columns(cards_per_row)
        for i, title in enumerate(row):
            with cols[i]:
                om = fetch_by_title(title)
                if om:
                    if passes_filters_omdb(om):
                        st.markdown(compact_movie_card_from_omdb(om), unsafe_allow_html=True)
                        if st.button("❤️ Add to Favorites", key=f"trend_{om.get('imdbID') or title}"):
                            add_favorite(om)
                else:
                    st.info(f"{title} (details not found on OMDb)")

# ------------------------
# Tab 1: Search
# ------------------------
with tabs[1]:
    st.subheader("🔎 Search Movies")
    q = st.text_input("Enter movie title to search (exact or partial):", value="")
    search_btn = st.button("Search")
    if q and (search_btn or q):
        # try OMDb exact search by title first, then fallback to dataset fuzzy list
        om = fetch_by_title(q)
        if om:
            # Show main movie card
            st.markdown("### Result")
            st.markdown(compact_movie_card_from_omdb(om), unsafe_allow_html=True)
            # Add favorites button
            if st.button(f"❤️ Add {om.get('Title')} to Favorites"):
                add_favorite(om)
            # Show dataset-based recommendations
            recs = recommend_from_dataset(om.get("Title") or q, top_n=8)
            if recs:
                st.markdown("### 🎯 Similar (from dataset)")
                rec_cols = st.columns(4)
                for idx, rec_title in enumerate(recs):
                    with rec_cols[idx % 4]:
                        rec_om = fetch_by_title(rec_title)
                        if rec_om and passes_filters_omdb(rec_om):
                            st.markdown(compact_movie_card_from_omdb(rec_om), unsafe_allow_html=True)
                            if st.button(f"Add {rec_om.get('Title')}", key=f"recadd_{rec_om.get('imdbID') or rec_title}"):
                                add_favorite(rec_om)
                        else:
                            st.info(f"{rec_title} (no OMDb info)")
        else:
            # no exact OMDb result — suggest dataset matches
            matches = local_df[local_df["title"].str.lower().str.contains(q.lower())]
            if matches.empty:
                st.warning("No matches found in OMDb or dataset.")
            else:
                st.markdown(f"Found {len(matches)} dataset matches — showing top 12 enriched by OMDb")
                matches = matches.head(12)
                cols = st.columns(4)
                for idx, row in matches.iterrows():
                    with cols[idx % 4]:
                        rec_title = row["title"]
                        rec_om = fetch_by_title(rec_title)
                        if rec_om and passes_filters_omdb(rec_om):
                            st.markdown(compact_movie_card_from_omdb(rec_om), unsafe_allow_html=True)
                            if st.button("❤️ Add to Favorites", key=f"match_{rec_om.get('imdbID') or rec_title}"):
                                add_favorite(rec_om)
                        else:
                            st.info(f"{rec_title} (no OMDb info)")

# ------------------------
# Tab 2: Because you liked...
# ------------------------
with tabs[2]:
    st.subheader("🎯 Because you liked… (seed from input)")
    seed = st.text_input("Seed movie title for personalized recommendations:", value="")
    seed_btn = st.button("Get suggestions")
    if seed and (seed_btn or seed):
        # Try OMDb fetch to get canonical title if possible
        seed_om = fetch_by_title(seed)
        canonical = seed_om.get("Title") if seed_om else seed
        recs = recommend_from_dataset(canonical, top_n=12)
        if recs:
            st.markdown(f"Recommendations based on **{canonical}**:")
            cols = st.columns(4)
            for idx, rt in enumerate(recs):
                with cols[idx % 4]:
                    ro = fetch_by_title(rt)
                    if ro and passes_filters_omdb(ro):
                        st.markdown(compact_movie_card_from_omdb(ro), unsafe_allow_html=True)
                        if st.button(f"❤️ Add {ro.get('Title')}", key=f"seed_{ro.get('imdbID') or rt}"):
                            add_favorite(ro)
                    else:
                        st.info(f"{rt} (no OMDb info)")

# ------------------------
# Tab 3: Favorites
# ------------------------
with tabs[3]:
    st.subheader("⭐ Your Favorites")
    if not st.session_state["favorites"]:
        st.info("No favorites yet — add movies from the cards using 'Add to Favorites'.")
    else:
        # Display favorites with options to remove
        favs = st.session_state["favorites"]
        # layout in rows of 3
        n = len(favs)
        cols = st.columns(3)
        for i, fav in enumerate(favs):
            with cols[i % 3]:
                # Try to fetch by imdbID if present
                fav_imdb = fav.get("imdbID")
                data = None
                if fav_imdb:
                    data = fetch_by_imdb_id(fav_imdb)
                if not data:
                    data = fetch_by_title(fav.get("title"))
                if data:
                    st.markdown(compact_movie_card_from_omdb(data), unsafe_allow_html=True)
                    if st.button("Remove", key=f"rmfav_{fav.get('id')}"):
                        remove_favorite(fav.get("id"))
                else:
                    st.write(fav.get("title"))

# ------------------------
# Tab 4: Personalized (union of favorites)
# ------------------------
with tabs[4]:
    st.subheader("👤 Personalized Picks (from your Favorites)")
    if not st.session_state["favorites"]:
        st.info("Add favorites to get personalized picks here.")
    else:
        # collect recs for each favorite and combine
        aggregated = []
        for fav in st.session_state["favorites"][:10]:  # limit to first 10 for speed
            title = fav.get("title")
            if title:
                recs = recommend_from_dataset(title, top_n=8)
                aggregated.extend(recs)
        # dedupe and exclude already favorited
        aggregated = [r for r in list(dict.fromkeys(aggregated)) if r not in [f.get("title") for f in st.session_state["favorites"]]]
        if not aggregated:
            st.info("No personalized picks found (dataset too small or no matches).")
        else:
            cols = st.columns(4)
            for idx, ag in enumerate(aggregated[:24]):
                with cols[idx % 4]:
                    aom = fetch_by_title(ag)
                    if aom and passes_filters_omdb(aom):
                        st.markdown(compact_movie_card_from_omdb(aom), unsafe_allow_html=True)
                        if st.button("❤️ Add to Favorites", key=f"pers_{aom.get('imdbID') or ag}"):
                            add_favorite(aom)
                    else:
                        st.info(ag)

# ------------------------
# Tab 5: My Dataset (uploaded or default)
# ------------------------
with tabs[5]:
    st.subheader("📂 My Dataset")
    st.markdown(f"Dataset rows loaded: **{len(local_df)}** (source: `{LOCAL_DEFAULT_CSV}` or uploaded file)")
    st.markdown("You can upload a CSV in the sidebar at any time.")
    # show first rows
    if not local_df.empty:
        st.dataframe(local_df.head(50))
        st.markdown("Enrich a sample of dataset movies via OMDb:")
        sample = local_df["title"].sample(n=min(12, len(local_df)), random_state=1).tolist()
        cols = st.columns(4)
        for idx, t in enumerate(sample):
            with cols[idx % 4]:
                om = fetch_by_title(t)
                if om and passes_filters_omdb(om):
                    st.markdown(compact_movie_card_from_omdb(om), unsafe_allow_html=True)
                    if st.button("❤️ Add to Favorites", key=f"ds_{om.get('imdbID') or t}"):
                        add_favorite(om)
                else:
                    st.info(f"{t} (OMDb not found)")

# ------------------------
# Footer / Credits
# ------------------------
st.markdown("---")
st.caption("Data & images © OMDb & IMDb. This app uses the OMDb API (http://www.omdbapi.com). Built with Streamlit.")
