"""
Core environment implementing reset / step / state.
Each call to reset() picks a task (round-robin: 1 → 2 → 3 → 1 …)
or a specific task_id can be forced via reset(task_id=N).
"""

import re
import uuid
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple

from models import DataCleaningAction, DataCleaningObservation, DataCleaningState
import server.tasks.task1_missing  as t1
import server.tasks.task2_format   as t2
import server.tasks.task3_pipeline as t3

TASK_MODULES = {1: t1, 2: t2, 3: t3}

PHONE_RE = re.compile(r"^\d{3}-\d{3}-\d{4}$")
DATE_RE  = re.compile(r"^\d{4}-\d{2}-\d{2}$")
VALID_COUNTRIES = {"USA", "UK", "Canada", "Australia", "Germany"}


class DataCleaningEnvironment:

    def __init__(self):
        self._df: Optional[pd.DataFrame]    = None
        self._clean_df: Optional[pd.DataFrame] = None
        self._meta: Any                     = None   # task-specific metadata
        self._task_id: int                  = 1
        self._episode_id: str               = ""
        self._step_count: int               = 0
        self._max_steps: int                = 20
        self._total_errors: int             = 0
        self._last_score: float             = 0.0
        self._task_cycle: int               = 0      # for round-robin default

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: Optional[int] = None) -> DataCleaningObservation:
        if task_id is None:
            self._task_cycle = (self._task_cycle % 3) + 1
            task_id = self._task_cycle

        if task_id not in TASK_MODULES:
            raise ValueError(f"task_id must be 1, 2, or 3 — got {task_id}")

        mod = TASK_MODULES[task_id]
        self._task_id   = task_id
        self._episode_id = str(uuid.uuid4())
        self._step_count = 0
        self._max_steps  = mod.MAX_STEPS

        if task_id == 1:
            self._df, self._clean_df, self._meta = mod.load()
        else:
            self._df, self._clean_df, self._meta = mod.load()

        self._last_score   = self._compute_score()
        self._total_errors = self._count_errors()

        return self._build_obs(self._last_score, False, "Episode started. Begin cleaning.")

    def step(self, action: DataCleaningAction) -> DataCleaningObservation:
        if self._df is None:
            raise RuntimeError("Call reset() before step().")

        self._step_count += 1
        score_before = self._last_score

        message, applied = self._apply_action(action)

        score_after    = self._compute_score()
        self._last_score = score_after

        delta   = score_after - score_before
        if not applied:
            reward = -0.05
        elif delta <= 0:
            reward = -0.01
        else:
            reward = round(delta, 4)

        done = (score_after >= 0.95) or (self._step_count >= self._max_steps)
        if done and score_after >= 0.95:
            reward = round(reward + 0.2, 4)

        return self._build_obs(reward, done, message)

    def state(self) -> DataCleaningState:
        if self._df is None:
            return DataCleaningState(
                episode_id="", task_id=0, step_count=0,
                max_steps=0, total_errors=0, errors_remaining=0,
            )
        return DataCleaningState(
            episode_id    = self._episode_id,
            task_id       = self._task_id,
            step_count    = self._step_count,
            max_steps     = self._max_steps,
            total_errors  = self._total_errors,
            errors_remaining = self._count_errors(),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_score(self) -> float:
        if self._task_id == 1:
            raw = t1.score(self._df, self._meta)
        elif self._task_id == 2:
            raw = t2.score(self._df, self._meta)
        else:
            raw = t3.score(self._df, self._meta)

        EPS = 1e-4

        # First round safely
        raw = float(raw)

        # HARD clamp AFTER rounding risk
        if raw >= 1.0:
            raw = 1.0 - EPS
        elif raw <= 0.0:
            raw = EPS

        return round(raw, 4)

    def _count_errors(self) -> int:
        if self._task_id == 1:
            return t1.count_errors(self._df)
        elif self._task_id == 2:
            return t2.count_errors(self._df, self._meta)
        else:
            return t3.count_errors(self._df, self._meta)

    def _build_obs(self, reward: float, done: bool, message: str) -> DataCleaningObservation:
        mod = TASK_MODULES[self._task_id]
        missing = {col: int(n) for col, n in self._df.isnull().sum().items() if n > 0}
        dupes   = len(self._df) - len(self._df.drop_duplicates())
        dtype_issues = self._detect_dtype_issues()
        preview = self._df.head(10).to_csv(index=False)

        return DataCleaningObservation(
            done             = done,
            reward           = reward,
            data_preview     = preview,
            data_shape       = list(self._df.shape),
            missing_counts   = missing,
            duplicate_count  = dupes,
            dtype_issues     = dtype_issues,
            task_description = mod.DESCRIPTION,
            message          = message,
            step_count       = self._step_count,
            current_score    = self._last_score,
        )

    def _detect_dtype_issues(self) -> Dict[str, str]:
        issues: Dict[str, str] = {}
        for col in self._df.columns:
            series = self._df[col].dropna()
            if series.empty:
                continue
            if self._df[col].dtype == object:
                numeric_count = pd.to_numeric(series, errors="coerce").notna().sum()
                if numeric_count / len(series) > 0.8:
                    issues[col] = "stored as string but appears numeric"
        return issues

    # ------------------------------------------------------------------
    # Action dispatcher
    # ------------------------------------------------------------------

    def _apply_action(self, action: DataCleaningAction) -> Tuple[str, bool]:
        op  = action.operation.strip().lower()
        col = action.column
        p   = action.params or {}

        try:
            if op == "fill_missing":
                return self._fill_missing(col, p)
            elif op == "drop_duplicates":
                return self._drop_duplicates()
            elif op == "fix_format":
                return self._fix_format(col)
            elif op == "replace_value":
                return self._replace_value(col, p)
            elif op == "drop_outliers":
                return self._drop_outliers(col)
            elif op == "fix_dtype":
                return self._fix_dtype(col, p)
            else:
                return f"Unknown operation '{op}'. Choose from: fill_missing, drop_duplicates, fix_format, replace_value, drop_outliers, fix_dtype.", False
        except Exception as exc:
            return f"Operation failed: {exc}", False

    def _fill_missing(self, col, p) -> Tuple[str, bool]:
        if col is None or col not in self._df.columns:
            return f"Column '{col}' not found.", False
        n_before = int(self._df[col].isnull().sum())
        if n_before == 0:
            return f"No missing values in '{col}'.", False

        strategy = str(p.get("strategy", "median")).lower()
        if strategy == "median":
            fill_val = self._df[col].median(skipna=True)
        elif strategy == "mean":
            fill_val = self._df[col].mean(skipna=True)
        elif strategy == "mode":
            mode = self._df[col].mode(dropna=True)
            fill_val = mode.iloc[0] if not mode.empty else None
        elif strategy == "constant":
            fill_val = p.get("value")
        else:
            return f"Unknown strategy '{strategy}'.", False

        if fill_val is None:
            return "Could not determine fill value.", False

        self._df[col] = self._df[col].fillna(fill_val)
        n_after = int(self._df[col].isnull().sum())
        return f"Filled {n_before - n_after} missing values in '{col}' using {strategy}.", True

    def _drop_duplicates(self) -> Tuple[str, bool]:
        n_before = len(self._df)
        self._df = self._df.drop_duplicates().reset_index(drop=True)
        n_after  = len(self._df)
        removed  = n_before - n_after
        if removed == 0:
            return "No duplicate rows found.", False
        return f"Dropped {removed} duplicate rows.", True

    def _fix_format(self, col) -> Tuple[str, bool]:
        if col is None or col not in self._df.columns:
            return f"Column '{col}' not found.", False

        if col == "phone":
            return self._fix_phone(col)
        elif col in ("listed_date", "signup_date"):
            return self._fix_date(col)
        elif col == "country":
            return self._fix_country(col)
        else:
            return f"No format rule defined for column '{col}'.", False

    def _fix_phone(self, col) -> Tuple[str, bool]:
        def normalise(val):
            if pd.isna(val):
                return val
            digits = re.sub(r"\D", "", str(val))
            if len(digits) == 10:
                return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
            return val

        before = (~self._df[col].str.match(PHONE_RE, na=False)).sum()
        self._df[col] = self._df[col].apply(normalise)
        after  = (~self._df[col].str.match(PHONE_RE, na=False)).sum()
        fixed  = int(before - after)
        if fixed == 0:
            return f"No phone format issues found in '{col}'.", False
        return f"Fixed {fixed} phone numbers in '{col}' to NNN-NNN-NNNN format.", True

    def _fix_date(self, col) -> Tuple[str, bool]:
        _DATE_FORMATS = ["%Y-%m-%d", "%b %d %Y", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"]

        def normalise(val):
            if pd.isna(val):
                return val
            s = str(val).strip()
            for fmt in _DATE_FORMATS:
                try:
                    return pd.to_datetime(s, format=fmt).strftime("%Y-%m-%d")
                except Exception:
                    pass
            # last-resort flexible parse
            try:
                return pd.to_datetime(s).strftime("%Y-%m-%d")
            except Exception:
                return val

        before = (~self._df[col].apply(
            lambda x: bool(DATE_RE.match(str(x))) if pd.notna(x) else False
        )).sum()
        self._df[col] = self._df[col].apply(normalise)
        after  = (~self._df[col].apply(
            lambda x: bool(DATE_RE.match(str(x))) if pd.notna(x) else False
        )).sum()
        fixed  = int(before - after)
        if fixed == 0:
            return f"No date format issues found in '{col}'.", False
        return f"Fixed {fixed} dates in '{col}' to YYYY-MM-DD format.", True

    def _fix_country(self, col) -> Tuple[str, bool]:
        def normalise(val):
            if pd.isna(val):
                return val
            mapping = {
                "usa": "USA", "uk": "UK", "canada": "Canada",
                "australia": "Australia", "germany": "Germany",
            }
            return mapping.get(str(val).strip().lower(), val)

        before = (~self._df[col].isin(VALID_COUNTRIES) & self._df[col].notna()).sum()
        self._df[col] = self._df[col].apply(normalise)
        after  = (~self._df[col].isin(VALID_COUNTRIES) & self._df[col].notna()).sum()
        fixed  = int(before - after)
        if fixed == 0:
            return f"No country capitalisation issues found.", False
        return f"Fixed {fixed} country values to correct capitalisation.", True

    def _replace_value(self, col, p) -> Tuple[str, bool]:
        if col is None or col not in self._df.columns:
            return f"Column '{col}' not found.", False
        old = p.get("old")
        new = p.get("new")
        if old is None:
            return "params.old is required for replace_value.", False
        count = int((self._df[col] == old).sum())
        if count == 0:
            return f"Value '{old}' not found in '{col}'.", False
        self._df[col] = self._df[col].replace(old, new)
        return f"Replaced {count} occurrences of '{old}' with '{new}' in '{col}'.", True

    def _drop_outliers(self, col) -> Tuple[str, bool]:
        if col is None or col not in self._df.columns:
            return f"Column '{col}' not found.", False
        if not pd.api.types.is_numeric_dtype(self._df[col]):
            return f"'{col}' is not numeric.", False
        q1  = self._df[col].quantile(0.25)
        q3  = self._df[col].quantile(0.75)
        iqr = q3 - q1
        mask     = (self._df[col] >= q1 - 3 * iqr) & (self._df[col] <= q3 + 3 * iqr)
        n_before = len(self._df)
        self._df = self._df[mask | self._df[col].isna()].reset_index(drop=True)
        removed  = n_before - len(self._df)
        if removed == 0:
            return f"No outliers found in '{col}'.", False
        return f"Removed {removed} outlier rows from '{col}' using IQR method.", True

    def _fix_dtype(self, col, p) -> Tuple[str, bool]:
        if col is None or col not in self._df.columns:
            return f"Column '{col}' not found.", False
        dtype = str(p.get("dtype", "float")).lower()
        try:
            if dtype == "float":
                self._df[col] = pd.to_numeric(self._df[col], errors="coerce").astype(float)
            elif dtype == "int":
                self._df[col] = pd.to_numeric(self._df[col], errors="coerce")
            elif dtype == "str":
                self._df[col] = self._df[col].astype(str)
            else:
                return f"Unknown dtype '{dtype}'.", False
            return f"Converted '{col}' to {dtype}.", True
        except Exception as exc:
            return f"dtype conversion failed: {exc}", False
