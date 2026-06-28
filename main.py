"""
FastAPI backend for Patent Quality Intelligence.
Serves pre-computed patent quality scores as JSON.
"""
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Optional

app = FastAPI(title="Patent Quality API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(__file__).parent.parent

# Load data once on startup
print("Loading data...", flush=True)
index_df  = pd.read_csv(DATA_DIR / "company_index.csv")
scores_df = pd.read_csv(DATA_DIR / "company_search_scores.csv")
ticker_df = pd.read_csv(DATA_DIR / "company_scores_full.csv")
names_df  = pd.read_csv(DATA_DIR / "ticker_names.csv")
names_map = dict(zip(names_df["ticker"], names_df["name"]))
print(f"Loaded: {len(index_df):,} companies, {len(scores_df):,} data points", flush=True)


def clean_name(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\b(inc|corp|corporation|ltd|limited|llc|co|company|plc|"
               r"holdings|group|technologies|technology|systems|"
               r"international|the|and|&|sa|ag|gmbh|bv|nv|sas)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


@app.get("/")
def root():
    return {"status": "ok", "companies": len(index_df), "data_range": "2000-2025"}


@app.get("/search")
def search(q: str = Query(..., min_length=1), limit: int = 8):
    """Search companies by name or ticker."""
    q = q.strip()

    # 1. Try as ticker first
    q_upper = q.upper()
    if q_upper in ticker_df["tk"].values:
        company_name = names_map.get(q_upper, q_upper)
        row = index_df[index_df["company"].str.contains(
            company_name.split()[0], case=False, na=False
        )].head(1)
        if len(row):
            r = row.iloc[0]
            return {"results": [{
                "company": r["company"],
                "sector": r["sector"],
                "n_total": int(r["n_total"]),
                "quality_avg": round(float(r["quality_avg"]), 3),
                "quality_norm_avg": round(float(r["quality_norm_avg"]), 3),
                "year_min": int(r.get("year_min", 2000)),
                "year_max": int(r.get("year_max", 2025)),
                "ticker": q_upper,
                "display_name": company_name,
            }]}

    # 2. Fuzzy name search
    qc = clean_name(q)
    mask = index_df["name_clean"].str.contains(qc, na=False)
    results = index_df[mask].sort_values("n_total", ascending=False).head(limit)

    if len(results) == 0 and qc:
        words = qc.split()
        if words:
            mask2 = index_df["name_clean"].str.contains(words[0], na=False)
            results = index_df[mask2].sort_values("n_total", ascending=False).head(limit)

    out = []
    for _, r in results.iterrows():
        out.append({
            "company": r["company"],
            "sector": r["sector"],
            "n_total": int(r["n_total"]),
            "quality_avg": round(float(r["quality_avg"]), 3),
            "quality_norm_avg": round(float(r["quality_norm_avg"]), 3),
            "year_min": int(r.get("year_min", 2000)),
            "year_max": int(r.get("year_max", 2025)),
            "ticker": None,
            "display_name": r["company"],
        })
    return {"results": out, "total": len(out)}


@app.get("/company/{company_name}")
def get_company(
    company_name: str,
    year_start: int = 2010,
    year_end: int = 2025,
):
    """Get full time-series data for a company."""
    # Try scores_df first (full history)
    data = scores_df[
        (scores_df["company"] == company_name) &
        (scores_df["grant_year"].between(year_start, year_end))
    ].copy()

    if data.empty:
        return {"error": f"No data for {company_name}"}

    sector = data["sector"].mode().iloc[0] if "sector" in data.columns else "Unknown"
    q_col   = "mean_quality"
    n_col   = "quality_norm" if "quality_norm" in data.columns else None

    # Sector peers for ranking
    peers = scores_df[
        (scores_df["sector"] == sector) &
        (scores_df["grant_year"].between(year_start, year_end))
    ]
    ranking = (peers.groupby("company")["quality_norm" if n_col else q_col]
               .mean().sort_values(ascending=False).reset_index())
    ranking.columns = ["company", "score"]
    rank_pos = int(ranking[ranking["company"] == company_name].index[0]) + 1 \
               if company_name in ranking["company"].values else None

    # Time series
    ts = data.groupby("grant_year").agg(
        mean_quality=(q_col, "mean"),
        quality_norm=(n_col, "mean") if n_col else (q_col, "mean"),
        n_patents=("n_patents", "sum"),
    ).reset_index()

    # Sector average time series
    sec_ts = peers.groupby("grant_year")["quality_norm" if n_col else q_col].mean().reset_index()
    sec_ts.columns = ["grant_year", "sector_avg"]

    return {
        "company": company_name,
        "sector": sector,
        "rank": rank_pos,
        "total_in_sector": len(ranking),
        "total_patents": int(data["n_patents"].sum()),
        "quality_avg": round(float(data[q_col].mean()), 3),
        "quality_norm_avg": round(float(data[n_col].mean()), 3) if n_col else None,
        "year_min": int(data["grant_year"].min()),
        "year_max": int(data["grant_year"].max()),
        "timeseries": ts.to_dict("records"),
        "sector_timeseries": sec_ts.to_dict("records"),
        "sector_ranking": ranking.head(30).to_dict("records"),
    }


@app.get("/sector/{sector_name}")
def get_sector(sector_name: str, year_start: int = 2015, year_end: int = 2025, limit: int = 30):
    """Top companies in a sector."""
    data = scores_df[
        (scores_df["sector"] == sector_name) &
        (scores_df["grant_year"].between(year_start, year_end))
    ]
    if data.empty:
        return {"error": f"No data for sector {sector_name}"}

    ranking = (data.groupby("company")
               .agg(quality_norm=("quality_norm", "mean"),
                    quality_avg=("mean_quality", "mean"),
                    n_patents=("n_patents", "sum"))
               .sort_values("quality_norm", ascending=False)
               .head(limit).reset_index())

    return {
        "sector": sector_name,
        "companies": ranking.to_dict("records"),
        "total": len(ranking),
    }
