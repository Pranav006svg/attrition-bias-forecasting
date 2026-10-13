"""Trial dates used ONLY to order evidence in time (never as model features).

Reads the raw API pages and writes data/interim/trial_dates.parquet with
  nct_id, start_date, results_first_post_date
Partial dates are resolved conservatively: a month-precision start date
("YYYY-MM") becomes the first day of that month and a year-only date becomes
January 1, so "evidence posted before start" is never satisfied by rounding.

Usage:
    python -m src.data.dates
"""
import gzip
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def parse_date(d):
    if not d:
        return pd.NaT
    parts = d.split("-")
    if len(parts) == 1:
        d = f"{parts[0]}-01-01"
    elif len(parts) == 2:
        d = f"{parts[0]}-{parts[1]}-01"
    return pd.to_datetime(d, errors="coerce")


def main():
    rows = []
    for p in tqdm(sorted(Path("data/raw").glob("page_*.json.gz")), desc="pages"):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for s in json.load(f).get("studies", []):
                sm = s.get("protocolSection", {}).get("statusModule", {})
                rows.append({
                    "nct_id": s["protocolSection"]["identificationModule"]["nctId"],
                    "start_date": parse_date((sm.get("startDateStruct") or {}).get("date")),
                    "results_first_post_date": parse_date((sm.get("resultsFirstPostDateStruct") or {}).get("date")),
                })
    df = pd.DataFrame(rows).drop_duplicates("nct_id")
    df.to_parquet("data/interim/trial_dates.parquet", index=False)
    print(f"{len(df):,} trials; missing start {df.start_date.isna().sum()}, "
          f"missing results post {df.results_first_post_date.isna().sum()}; "
          f"results posted before start: {(df.results_first_post_date < df.start_date).sum()}")


if __name__ == "__main__":
    main()
