"""
Task 4 — Expert: Multi-Source Schema Alignment + Merge Pipeline

Two independent data sources (CRM + Marketing) have been exported with
misaligned column names and must be aligned to a canonical schema,
merged into one DataFrame, and then cleaned.

Grader sub-scores (equal weight):
  0.30 × schema_score    — correct columns present after align + merge
  0.25 × null_score      — missing values filled
  0.20 × country_score   — country capitalisation fixed
  0.15 × date_score      — signup_date format standardised
  0.10 × dupe_score      — duplicate rows removed

Inspired by:
  - CleanAgent (Qi & Wang, 2024) — declarative schema standardisation
  - Meta DataSchema system — column-level semantic annotation at scale
"""

import re
import pandas as pd
from server.data_generator import generate_task4_datasets

TASK_ID   = 4
MAX_STEPS = 50

DESCRIPTION = (
    "Task 4 (Expert) — Multi-Source Schema Alignment + Merge Pipeline\n"
    "You have TWO source DataFrames with misaligned schemas:\n\n"
    "  Source A (CRM, 150 rows) columns:\n"
    "    cust_id, full_name, Age, purchase_amt, Country, signup, email\n\n"
    "  Source B (Marketing, 100 rows) columns:\n"
    "    customer_id, name, age_years, spend, country_name, registration_date, email\n\n"
    "Target canonical schema (250 rows after merge):\n"
    "    customer_id, name, age, purchase_amount, country, signup_date, email\n\n"
    "Step 1 — align_schema: rename Source A columns to match target.\n"
    "Step 2 — merge_sources: concatenate Source A + Source B.\n"
    "Step 3 — Clean the merged dataset:\n"
    "  • fill_missing   — age, purchase_amount, country (~10% nulls each)\n"
    "  • fix_format     — country (mixed case), signup_date (mixed formats)\n"
    "  • drop_duplicates — ~10 duplicate rows\n\n"
    "Available operations:\n"
    "  align_schema    — no column needed; renames Source A to canonical schema\n"
    "  merge_sources   — no column needed; concatenates aligned A + B\n"
    "  fill_missing    — column + params.strategy\n"
    "  fix_format      — column: 'country' | 'signup_date'\n"
    "  drop_duplicates — no column needed\n\n"
    "Example actions:\n"
    '  {"operation": "align_schema"}\n'
    '  {"operation": "merge_sources"}\n'
    '  {"operation": "fill_missing", "column": "age", "params": {"strategy": "median"}}\n'
    '  {"operation": "fix_format", "column": "country"}\n'
    '  {"operation": "fix_format", "column": "signup_date"}\n'
    '  {"operation": "drop_duplicates"}'
)

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
VALID_COUNTRIES = {"USA", "UK", "Canada", "Australia", "Germany"}
TARGET_COLUMNS  = ["customer_id", "name", "age", "purchase_amount",
                   "country", "signup_date", "email"]

# Column mapping: Source A dirty names → canonical target names
SOURCE_A_RENAME = {
    "cust_id":      "customer_id",
    "full_name":    "name",
    "Age":          "age",
    "purchase_amt": "purchase_amount",
    "Country":      "country",
    "signup":       "signup_date",
    # "email" already matches
}


# ---------------------------------------------------------------------------
# Cache at module load
# ---------------------------------------------------------------------------

def _build_meta(source_a, source_b, clean_merged):
    import numpy as np

    # Align source_a and source_b to canonical schema before merging
    aligned_a = source_a.rename(columns=SOURCE_A_RENAME)
    source_b_rename = {
        "age_years":         "age",
        "spend":             "purchase_amount",
        "country_name":      "country",
        "registration_date": "signup_date",
    }
    aligned_b = source_b.rename(columns=source_b_rename)

    merged = pd.concat(
        [aligned_a[TARGET_COLUMNS], aligned_b[TARGET_COLUMNS]],
        ignore_index=True
    ).reset_index(drop=True)

    # Inject dirty issues deterministically
    import numpy as np
    rng = np.random.default_rng(42 + 4)

    n = len(merged)
    # Missing values
    for col, frac in [("age", 0.10), ("purchase_amount", 0.10), ("country", 0.08)]:
        idx = rng.choice(n, size=int(n * frac), replace=False)
        merged.loc[idx, col] = None

    # Mixed country case
    case_idx = rng.choice(n, size=int(n * 0.30), replace=False)
    merged.loc[case_idx, "country"] = merged.loc[case_idx, "country"].str.lower()

    # Mixed date formats
    import random as _random
    _random.seed(42 + 4)
    date_idx = rng.choice(n, size=int(n * 0.40), replace=False)
    for i in date_idx:
        val = merged.loc[i, "signup_date"]
        if pd.notna(val):
            try:
                dt = pd.to_datetime(str(val))
                fmt = rng.integers(0, 3)
                if fmt == 1:
                    merged.loc[i, "signup_date"] = dt.strftime("%b %d %Y")
                elif fmt == 2:
                    merged.loc[i, "signup_date"] = dt.strftime("%d/%m/%Y")
            except Exception:
                pass

    # Duplicates
    dup_idx = rng.choice(n, size=10, replace=False)
    dup_rows = merged.iloc[dup_idx].copy()
    merged = pd.concat([merged, dup_rows], ignore_index=True)

    orig_nulls = int(merged.isnull().sum().sum())
    orig_dupes = len(merged) - len(merged.drop_duplicates())
    orig_country_issues = int(
        (~merged["country"].isin(VALID_COUNTRIES) & merged["country"].notna()).sum()
    )
    orig_date_issues = int(
        (~merged["signup_date"].apply(
            lambda x: bool(DATE_RE.match(str(x))) if pd.notna(x) else False
        )).sum()
    )

    return {
        "orig_nulls":          max(orig_nulls, 1),
        "orig_dupes":          max(orig_dupes, 1),
        "orig_country_issues": max(orig_country_issues, 1),
        "orig_date_issues":    max(orig_date_issues, 1),
        "dirty_merged":        merged,   # stored for environment to use post-merge
    }


_SOURCE_A, _SOURCE_B, _CLEAN_MERGED = generate_task4_datasets()
_META_TEMPLATE = _build_meta(_SOURCE_A, _SOURCE_B, _CLEAN_MERGED)


def load():
    """
    Returns (source_a, source_b, clean_merged, meta).
    source_a is the initial active DataFrame (pre-alignment).
    source_b is held separately until merge_sources is called.
    """
    import copy
    meta = {k: v for k, v in _META_TEMPLATE.items() if k != "dirty_merged"}
    meta["dirty_merged"] = _META_TEMPLATE["dirty_merged"].copy()
    return _SOURCE_A.copy(), _SOURCE_B.copy(), _CLEAN_MERGED.copy(), meta


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

def score(current_df, meta: dict) -> float:
    """
    Weighted score across 5 sub-dimensions:
      0.30 schema_score  — all target columns present, no extra columns
      0.25 null_score    — missing values filled
      0.20 country_score — country capitalisation correct
      0.15 date_score    — signup_date in YYYY-MM-DD
      0.10 dupe_score    — no duplicate rows
    """
    # Schema score: are all target columns present?
    present = sum(1 for c in TARGET_COLUMNS if c in current_df.columns)
    schema_score = present / len(TARGET_COLUMNS)

    # Can only score the rest if schema is aligned AND merged
    if not all(c in current_df.columns for c in TARGET_COLUMNS):
        # Partial credit: schema only
        return round(max(0.01, min(0.99, 0.30 * schema_score)), 4)

    remaining_nulls = int(current_df.isnull().sum().sum())
    remaining_dupes = len(current_df) - len(current_df.drop_duplicates())
    remaining_country = int(
        (~current_df["country"].isin(VALID_COUNTRIES) & current_df["country"].notna()).sum()
    )
    remaining_dates = int(
        (~current_df["signup_date"].apply(
            lambda x: bool(DATE_RE.match(str(x))) if pd.notna(x) else False
        )).sum()
    )

    null_score    = 1.0 - remaining_nulls   / meta["orig_nulls"]
    dupe_score    = 1.0 - remaining_dupes   / meta["orig_dupes"]
    country_score = 1.0 - remaining_country / meta["orig_country_issues"]
    date_score    = 1.0 - remaining_dates   / meta["orig_date_issues"]

    combined = (0.30 * schema_score  +
                0.25 * null_score    +
                0.20 * country_score +
                0.15 * date_score    +
                0.10 * dupe_score)

    return round(max(0.01, min(0.99, combined)), 4)


def count_errors(current_df, meta: dict) -> int:
    errors = 0
    missing_cols = sum(1 for c in TARGET_COLUMNS if c not in current_df.columns)
    errors += missing_cols * 10   # heavy penalty for schema misalignment

    if all(c in current_df.columns for c in TARGET_COLUMNS):
        errors += int(current_df.isnull().sum().sum())
        errors += len(current_df) - len(current_df.drop_duplicates())
        errors += int(
            (~current_df["country"].isin(VALID_COUNTRIES) & current_df["country"].notna()).sum()
        )
        errors += int(
            (~current_df["signup_date"].apply(
                lambda x: bool(DATE_RE.match(str(x))) if pd.notna(x) else False
            )).sum()
        )
    return errors