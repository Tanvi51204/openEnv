"""
Synthetic dataset generation with a fixed seed for full reproducibility.
All datasets are generated purely from numpy/random — no external downloads.
"""

import random
import numpy as np
import pandas as pd

SEED = 42


# ---------------------------------------------------------------------------
# Task 1 — Employee records with missing values
# ---------------------------------------------------------------------------

def generate_task1_datasets():
    """Returns (dirty_df, clean_df) for Task 1."""
    rng = np.random.default_rng(SEED)
    random.seed(SEED)

    n = 100
    departments = ["Engineering", "Marketing", "Sales", "HR", "Finance"]
    first_names = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace",
                   "Heidi", "Ivan", "Judy", "Karl", "Laura", "Mallory", "Niaj",
                   "Oscar", "Peggy", "Quinn", "Romeo", "Sybil", "Trent"]
    last_names  = ["Smith", "Jones", "Brown", "Taylor", "Wilson", "Davis",
                   "Miller", "Anderson", "Thomas", "Jackson"]

    names       = [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(n)]
    ages        = rng.integers(22, 60, size=n).astype(float)
    salaries    = rng.integers(40_000, 120_000, size=n).astype(float)
    depts       = rng.choice(departments, size=n)
    experience  = rng.integers(0, 30, size=n).astype(float)

    clean_df = pd.DataFrame({
        "name":       names,
        "age":        ages,
        "salary":     salaries,
        "department": depts,
        "experience": experience,
    })

    dirty_df = clean_df.copy()

    # Inject ~20 % NaN into age, salary, department
    for col, frac in [("age", 0.20), ("salary", 0.20), ("department", 0.10)]:
        idx = rng.choice(n, size=int(n * frac), replace=False)
        dirty_df.loc[idx, col] = np.nan

    return dirty_df.reset_index(drop=True), clean_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Task 2 — Product catalog with format & duplicate issues
# ---------------------------------------------------------------------------

def _scramble_phone(phone: str, rng) -> str:
    digits = phone.replace("-", "")
    fmt = rng.integers(0, 3)
    if fmt == 0:
        return digits                          # 5551234567
    elif fmt == 1:
        return f"({digits[:3]}){digits[3:]}"   # (555)1234567
    else:
        return phone                           # 555-123-4567  (canonical)


def _scramble_date(date_str: str, rng) -> str:
    dt = pd.to_datetime(date_str)
    fmt = rng.integers(0, 3)
    if fmt == 0:
        return dt.strftime("%Y-%m-%d")
    elif fmt == 1:
        return dt.strftime("%b %d %Y")
    else:
        return dt.strftime("%d/%m/%Y")


def generate_task2_datasets():
    """Returns (dirty_df, clean_df) for Task 2."""
    rng = np.random.default_rng(SEED)
    random.seed(SEED)

    n = 200
    categories = ["Electronics", "Clothing", "Food", "Books", "Toys"]

    product_ids   = [f"P{str(i).zfill(4)}" for i in range(1, n + 1)]
    product_names = [f"Product_{i}" for i in range(1, n + 1)]
    prices        = np.round(rng.uniform(5.0, 500.0, size=n), 2)
    categories_col = rng.choice(categories, size=n)
    phones        = [
        f"{rng.integers(100,999)}-{rng.integers(100,999)}-{rng.integers(1000,9999)}"
        for _ in range(n)
    ]
    days_offset   = rng.integers(0, 1000, size=n)
    dates         = [
        (pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
        for d in days_offset
    ]

    clean_df = pd.DataFrame({
        "product_id":   product_ids,
        "product_name": product_names,
        "price":        prices,
        "category":     categories_col,
        "phone":        phones,
        "listed_date":  dates,
    })

    dirty_df = clean_df.copy()

    # Scramble ~60 % of phone formats
    phone_idx = rng.choice(n, size=int(n * 0.6), replace=False)
    dirty_df.loc[phone_idx, "phone"] = [
        _scramble_phone(dirty_df.loc[i, "phone"], rng) for i in phone_idx
    ]

    # Scramble ~60 % of date formats
    date_idx = rng.choice(n, size=int(n * 0.6), replace=False)
    dirty_df.loc[date_idx, "listed_date"] = [
        _scramble_date(dirty_df.loc[i, "listed_date"], rng) for i in date_idx
    ]

    # Add 15 duplicate rows
    dup_idx  = rng.choice(n, size=15, replace=False)
    dup_rows = dirty_df.iloc[dup_idx].copy()
    dirty_df = pd.concat([dirty_df, dup_rows], ignore_index=True)

    return dirty_df.reset_index(drop=True), clean_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Task 3 — Customer database: full pipeline
# ---------------------------------------------------------------------------

def generate_task3_datasets():
    """Returns (dirty_df, clean_df) for Task 3."""
    rng = np.random.default_rng(SEED)
    random.seed(SEED)

    n = 300
    countries  = ["USA", "UK", "Canada", "Australia", "Germany"]
    first_names = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace",
                   "Heidi", "Ivan", "Judy"]
    last_names  = ["Smith", "Jones", "Brown", "Taylor", "Wilson"]

    names             = [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(n)]
    ages              = rng.integers(18, 75, size=n).astype(float)
    purchase_amounts  = np.round(rng.uniform(10.0, 500.0, size=n), 2)
    countries_col     = rng.choice(countries, size=n)
    emails            = [f"user{i}@example.com" for i in range(1, n + 1)]
    days_offset       = rng.integers(0, 730, size=n)
    signup_dates      = [
        (pd.Timestamp("2022-01-01") + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
        for d in days_offset
    ]

    clean_df = pd.DataFrame({
        "name":            names,
        "age":             ages,
        "purchase_amount": purchase_amounts,
        "country":         countries_col,
        "email":           emails,
        "signup_date":     signup_dates,
    })

    dirty_df = clean_df.copy()

    # Missing values (~15 % in age, purchase_amount, country, signup_date)
    for col, frac in [("age", 0.15), ("purchase_amount", 0.15),
                      ("country", 0.10), ("signup_date", 0.10)]:
        idx = rng.choice(n, size=int(n * frac), replace=False)
        dirty_df.loc[idx, col] = np.nan

    # Outliers in purchase_amount (~3 %)
    out_idx = rng.choice(n, size=int(n * 0.03), replace=False)
    dirty_df.loc[out_idx, "purchase_amount"] = (
        dirty_df.loc[out_idx, "purchase_amount"] * 10
    )

    # Mixed case in country (~40 %)
    case_idx = rng.choice(n, size=int(n * 0.40), replace=False)
    dirty_df.loc[case_idx, "country"] = dirty_df.loc[case_idx, "country"].str.lower()

    # Mixed date formats (~50 %) — only scramble non-null entries
    date_idx = rng.choice(n, size=int(n * 0.50), replace=False)
    valid_date_idx = [i for i in date_idx if pd.notna(dirty_df.loc[i, "signup_date"])]
    for i in valid_date_idx:
        dirty_df.loc[i, "signup_date"] = _scramble_date(dirty_df.loc[i, "signup_date"], rng)

    # 20 duplicate rows
    dup_idx  = rng.choice(n, size=20, replace=False)
    dup_rows = dirty_df.iloc[dup_idx].copy()
    dirty_df = pd.concat([dirty_df, dup_rows], ignore_index=True)

    return dirty_df.reset_index(drop=True), clean_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Task 4 — Multi-source merge pipeline (Expert)
# ---------------------------------------------------------------------------
# Two independently generated "source" DataFrames with misaligned schemas
# that must be aligned and merged before the standard cleaning pipeline.
#
# Source A — CRM export (150 rows):
#   cust_id, full_name, Age, purchase_amt, Country, signup
#
# Source B — Marketing export (100 rows):
#   customer_id, name, age_years, spend, country_name, registration_date, email
#
# Target schema after align_schema + merge_sources (250 rows):
#   customer_id, name, age, purchase_amount, country, signup_date, email
#
# Additional dirty issues injected after merge:
#   - Missing values in age, purchase_amount, country (~10%)
#   - Mixed country capitalisation (~30%)
#   - Mixed date formats in signup_date (~40%)
#   - 10 duplicate rows

def generate_task4_datasets():
    """
    Returns (source_a, source_b, clean_merged_df).
    source_a and source_b have misaligned schemas.
    clean_merged_df is the ground-truth after alignment + merge + cleaning.
    """
    rng = np.random.default_rng(SEED + 4)   # distinct seed offset
    random.seed(SEED + 4)

    countries   = ["USA", "UK", "Canada", "Australia", "Germany"]
    first_names = ["Alice", "Bob", "Carol", "David", "Eve", "Frank",
                   "Grace", "Heidi", "Ivan", "Judy", "Karl", "Laura"]
    last_names  = ["Smith", "Jones", "Brown", "Taylor", "Wilson",
                   "Davis", "Miller", "Anderson", "Thomas", "Jackson"]

    # ---- Source A — CRM (150 rows) ----
    n_a = 150
    names_a   = [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(n_a)]
    ages_a    = rng.integers(18, 75, size=n_a).astype(float)
    amounts_a = np.round(rng.uniform(10.0, 500.0, size=n_a), 2)
    countries_a = rng.choice(countries, size=n_a)
    days_a    = rng.integers(0, 730, size=n_a)
    dates_a   = [(pd.Timestamp("2022-01-01") + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
                 for d in days_a]
    emails_a  = [f"crm_{i}@example.com" for i in range(1, n_a + 1)]

    source_a = pd.DataFrame({
        "cust_id":      [f"A{str(i).zfill(4)}" for i in range(1, n_a + 1)],
        "full_name":    names_a,           # → name
        "Age":          ages_a,            # → age  (capital A — schema mismatch)
        "purchase_amt": amounts_a,         # → purchase_amount (truncated name)
        "Country":      countries_a,       # → country (capital C)
        "signup":       dates_a,           # → signup_date (truncated name)
        "email":        emails_a,
    })

    # ---- Source B — Marketing (100 rows) ----
    n_b = 100
    names_b   = [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(n_b)]
    ages_b    = rng.integers(18, 75, size=n_b).astype(float)
    amounts_b = np.round(rng.uniform(10.0, 500.0, size=n_b), 2)
    countries_b = rng.choice(countries, size=n_b)
    days_b    = rng.integers(0, 730, size=n_b)
    dates_b   = [(pd.Timestamp("2022-01-01") + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
                 for d in days_b]
    emails_b  = [f"mkt_{i}@example.com" for i in range(1, n_b + 1)]

    source_b = pd.DataFrame({
        "customer_id":        [f"B{str(i).zfill(4)}" for i in range(1, n_b + 1)],
        "name":               names_b,
        "age_years":          ages_b,      # → age  (suffix mismatch)
        "spend":              amounts_b,   # → purchase_amount (synonym)
        "country_name":       countries_b, # → country (suffix mismatch)
        "registration_date":  dates_b,     # → signup_date (synonym)
        "email":              emails_b,
    })

    # ---- Ground-truth clean merged DataFrame ----
    clean_a = pd.DataFrame({
        "customer_id":    source_a["cust_id"],
        "name":           source_a["full_name"],
        "age":            source_a["Age"],
        "purchase_amount":source_a["purchase_amt"],
        "country":        source_a["Country"],
        "signup_date":    source_a["signup"],
        "email":          source_a["email"],
    })
    clean_b = pd.DataFrame({
        "customer_id":    source_b["customer_id"],
        "name":           source_b["name"],
        "age":            source_b["age_years"],
        "purchase_amount":source_b["spend"],
        "country":        source_b["country_name"],
        "signup_date":    source_b["registration_date"],
        "email":          source_b["email"],
    })
    clean_merged = pd.concat([clean_a, clean_b], ignore_index=True).reset_index(drop=True)

    return source_a.copy(), source_b.copy(), clean_merged