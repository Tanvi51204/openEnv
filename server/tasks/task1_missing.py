"""
Task 1 — Easy: Fill Missing Values
Objective: Fill all NaN values in the employee records DataFrame.
Score: 1.0 - (remaining_nulls / original_nulls)
"""

from server.data_generator import generate_task1_datasets

TASK_ID = 1
MAX_STEPS = 20
DESCRIPTION = (
    "Task 1 (Easy) — Fill Missing Values\n"
    "You have an employee records dataset with missing values (NaN) in "
    "'age', 'salary', and 'department' columns. "
    "Your goal is to fill all missing values so the dataset is complete.\n\n"
    "Available operation: fill_missing\n"
    "  params.strategy: 'median' | 'mean' | 'mode' | 'constant'\n"
    "  params.value: (required when strategy='constant') the fill value\n"
    "Example action: {\"operation\": \"fill_missing\", \"column\": \"age\", \"params\": {\"strategy\": \"median\"}}"
)

# Cache at module load — seed=42 makes output identical every time
_DIRTY_TEMPLATE, _CLEAN_DF = generate_task1_datasets()
_ORIGINAL_NULLS = int(_DIRTY_TEMPLATE.isnull().sum().sum())


def load():
    """Return (dirty_df, clean_df, original_null_count) — uses cached template."""
    return _DIRTY_TEMPLATE.copy(), _CLEAN_DF, _ORIGINAL_NULLS


def score(current_df, original_nulls: int) -> float:
    """Score in [0, 1]: fraction of nulls filled."""
    if original_nulls == 0:
        return 0.99
    remaining = int(current_df.isnull().sum().sum())
    return round(max(0.01, min(0.99, 1.0 - remaining / original_nulls)), 4)


def count_errors(current_df) -> int:
    return int(current_df.isnull().sum().sum())