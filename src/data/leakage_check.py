"""Assert that no forbidden (post-enrollment / results) field is a feature.

Feature columns are all columns except identifiers (nct_id, arm_idx, split),
labels (y_*) and results bookkeeping (meta_*). Checks, for every processed
arms*/trials* table:
  1. every feature column is on the pre-enrollment allowlist;
  2. no feature column name matches a forbidden pattern;
  3. no column anywhere is named like a forbidden raw field (enrollment,
     overallStatus, whyStopped, completion dates), even as y_/meta_;
  4. split assignment follows the temporal rule and trial sets are disjoint.

Usage:
    python -m src.data.leakage_check
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

from src.data.build_benchmark import ARM_FEATURES, TRIAL_FEATURES, split_of

PROCESSED = Path("data/processed")
ID_COLS = {"nct_id", "arm_idx", "split"}
ALLOWED = {"trials": set(TRIAL_FEATURES), "arms": set(ARM_FEATURES)}
FORBIDDEN_FEATURE = re.compile(
    r"enroll|overall_?status|why_?stopped|completion|result|started|withdraw|drop|flow|milestone"
    r"|attrition|status|stopped|primary_completion|last_update|results_first", re.IGNORECASE)
FORBIDDEN_ANYWHERE = re.compile(r"enrollment|overall_?status|why_?stopped|completion_?date", re.IGNORECASE)


def feature_cols(df: pd.DataFrame):
    return [c for c in df.columns if c not in ID_COLS and not c.startswith(("y_", "meta_"))]


def main():
    errors = []
    files = sorted(PROCESSED.glob("arms*.parquet")) + sorted(PROCESSED.glob("trials*.parquet"))
    if not files:
        sys.exit("No processed tables found")
    for f in files:
        kind = "arms" if f.name.startswith("arms") else "trials"
        df = pd.read_parquet(f)
        feats = feature_cols(df)
        extra = set(feats) - ALLOWED[kind]
        if extra:
            errors.append(f"{f.name}: non-allowlisted feature columns {sorted(extra)}")
        bad = [c for c in feats if FORBIDDEN_FEATURE.search(c)]
        if bad:
            errors.append(f"{f.name}: forbidden-pattern feature columns {bad}")
        anywhere = [c for c in df.columns if FORBIDDEN_ANYWHERE.search(c)]
        if anywhere:
            errors.append(f"{f.name}: forbidden raw fields present {anywhere}")
        if kind == "trials":
            wrong = df[df.split != df.start_year.map(split_of)]
            if len(wrong):
                errors.append(f"{f.name}: {len(wrong)} trials violate the temporal split rule")
        print(f"{f.name}: {len(df):,} rows, {len(feats)} feature columns checked")

    splits = json.loads((PROCESSED / "splits.json").read_text(encoding="utf-8"))
    for name, s in splits["sets"].items():
        tr, va, te = (set(s[k]) for k in ["train", "val", "test"])
        if tr & va or tr & te or va & te:
            errors.append(f"splits.json[{name}]: overlapping splits")

    if errors:
        print("LEAKAGE CHECK FAILED")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print("LEAKAGE CHECK PASSED")


if __name__ == "__main__":
    main()
