"""Download all interventional studies with results from ClinicalTrials.gov API v2.

Each API page is stored unmodified as gzipped JSON in data/raw/page_NNNN.json.gz.
The script is resume-safe: on restart it reads the nextPageToken from the last
complete page on disk and continues from there. Pages are written atomically
(temp file + rename), so an interrupted write never leaves a partial page.

Usage:
    python -m src.data.download [--out data/raw]
"""
import argparse
import gzip
import json
import logging
import os
import time
from pathlib import Path

import requests

API_URL = "https://clinicaltrials.gov/api/v2/studies"
QUERY = {
    "filter.advanced": "AREA[StudyType]INTERVENTIONAL AND AREA[HasResults]true",
    "pageSize": 1000,
    "format": "json",
}
DELAY_S = 0.5
MAX_RETRIES = 8
TIMEOUT_S = 180
DONE_MARKER = "_COMPLETE"

log = logging.getLogger("download")


def page_path(out_dir: Path, idx: int) -> Path:
    return out_dir / f"page_{idx:04d}.json.gz"


def read_page(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def write_page_atomic(path: Path, raw_text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        f.write(raw_text)
    os.replace(tmp, path)


def resume_point(out_dir: Path):
    """Return (next_page_index, next_page_token) from pages already on disk."""
    idx = 0
    token = None
    while page_path(out_dir, idx).exists():
        try:
            page = read_page(page_path(out_dir, idx))
        except (OSError, json.JSONDecodeError, EOFError):
            log.warning("Corrupt page %d; will re-download it", idx)
            page_path(out_dir, idx).unlink()
            break
        token = page.get("nextPageToken")
        idx += 1
        if token is None:
            break
    return idx, token


def fetch(session: requests.Session, params: dict) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(API_URL, params=params, timeout=TIMEOUT_S)
            if r.status_code == 200:
                json.loads(r.text)  # validate before saving
                return r.text
            log.warning("HTTP %s (attempt %d): %s", r.status_code, attempt, r.text[:200])
        except (requests.RequestException, json.JSONDecodeError) as e:
            log.warning("Request error (attempt %d): %s", attempt, e)
        time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {params}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / DONE_MARKER).exists():
        log.info("Download already complete (%s exists).", DONE_MARKER)
        return

    idx, token = resume_point(out_dir)
    if idx > 0 and token is None:
        (out_dir / DONE_MARKER).touch()
        log.info("All %d pages already on disk.", idx)
        return
    log.info("Starting at page %d%s", idx, " (resumed)" if idx else "")

    session = requests.Session()
    while True:
        params = dict(QUERY)
        if token:
            params["pageToken"] = token
        if idx == 0:
            params["countTotal"] = "true"
        raw = fetch(session, params)
        page = json.loads(raw)
        write_page_atomic(page_path(out_dir, idx), raw)
        if idx == 0:
            log.info("totalCount reported by API: %s", page.get("totalCount"))
        log.info("Saved page %d (%d studies)", idx, len(page.get("studies", [])))
        token = page.get("nextPageToken")
        idx += 1
        if not token:
            break
        time.sleep(DELAY_S)

    (out_dir / DONE_MARKER).touch()
    log.info("Done: %d pages.", idx)


if __name__ == "__main__":
    main()
