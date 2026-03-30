from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class DataCleaningAction(BaseModel):
    """
    Action to apply to the current dirty DataFrame.

    operation choices:
        fill_missing    – fill NaN values in a column
        drop_duplicates – remove duplicate rows
        fix_format      – standardise string formats (phone, date, text)
        replace_value   – replace a specific value with another
        drop_outliers   – remove rows where column value is a statistical outlier
        fix_dtype       – cast a column to the correct dtype
    """
    operation: str
    column: Optional[str] = None
    params: Dict[str, Any] = {}


class DataCleaningObservation(BaseModel):
    done: bool
    reward: float
    data_preview: str           # First 10 rows as CSV string
    data_shape: List[int]       # [rows, cols]
    missing_counts: Dict[str, int]
    duplicate_count: int
    dtype_issues: Dict[str, str]
    task_description: str
    message: str
    step_count: int
    current_score: float        # Running grader score 0.0–1.0


class DataCleaningState(BaseModel):
    episode_id: str
    task_id: int
    step_count: int
    max_steps: int
    total_errors: int
    errors_remaining: int
