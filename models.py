from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class DataCleaningAction(BaseModel):
    """
    Action to apply to the current dirty DataFrame.

    operation choices:
        fill_missing    – fill NaN values in a column
        drop_duplicates – remove duplicate rows
        fix_format      – standardise string formats (phone, date, country)
        replace_value   – replace a specific value with another
        drop_outliers   – remove rows where column value is a statistical outlier
        fix_dtype       – cast a column to the correct dtype
        align_schema    – rename / reorder columns to match target schema (Task 4)
        merge_sources   – merge the two aligned source DataFrames (Task 4)
    """
    operation: str
    column: Optional[str] = None
    params: Dict[str, Any] = {}


class DataQualityMetrics(BaseModel):
    """Standard DQ dimensions — populated by /profile and embedded in every observation."""
    completeness_pct: float      # % non-null cells across whole DataFrame
    uniqueness_pct: float        # % rows that are not duplicates
    validity_pct: float          # % cells passing format / dtype / range constraints
    total_cells: int
    null_cells: int
    duplicate_rows: int
    invalid_cells: int           # format violations + dtype issues + out-of-range values


class DataCleaningObservation(BaseModel):
    done: bool
    reward: float
    data_preview: str                       # First 10 rows as CSV string
    data_shape: List[int]                   # [rows, cols]
    missing_counts: Dict[str, int]
    duplicate_count: int
    dtype_issues: Dict[str, str]
    task_description: str
    message: str
    step_count: int
    current_score: float                    # Running grader score 0.0-1.0

    # --- Phase 2 additions ---
    dq_metrics: DataQualityMetrics          # Live data quality vitals
    tried_operations: List[str]             # e.g. ["fill_missing:age", "drop_duplicates"]
    plan: List[str]                         # Agent-facing recommended next 1-3 actions


class DataCleaningState(BaseModel):
    episode_id: str
    task_id: int
    step_count: int
    max_steps: int
    total_errors: int
    errors_remaining: int


class EpisodeReport(BaseModel):
    """Returned by GET /report — full cleaning episode summary."""
    episode_id: str
    task_id: int
    task_name: str
    initial_score: float
    final_score: float
    score_improvement: float
    steps_taken: int
    max_steps: int
    step_efficiency_pct: float              # How few steps used vs max (higher = better)
    operations_applied: List[str]           # Ordered list of what was done
    issues_fixed: Dict[str, int]            # e.g. {"nulls_filled": 40, "dupes_removed": 15}
    final_dq_metrics: DataQualityMetrics
    completed: bool                         # True if score >= 0.95