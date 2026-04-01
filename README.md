---
title: Data Cleaning Environment
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - rl
  - data-cleaning
---

# Data Cleaning OpenEnv

A **real-world data cleaning environment** for AI agent training, built for the Scaler × OpenEnv hackathon.

An agent interacts with a dirty DataFrame through a simple `reset() / step() / state()` API, learning to fix common data quality issues: missing values, duplicate rows, format inconsistencies, outliers, and dtype errors.

---

## Environment Description

Real-world datasets are rarely clean. Data engineers spend a significant fraction of their time:
- Filling missing values with appropriate strategies (median/mean/mode)
- Removing duplicate records
- Standardising inconsistent formats (phone numbers, dates, country names)
- Detecting and removing statistical outliers

This environment turns those tasks into a reinforcement learning challenge with deterministic, programmatic graders and a meaningful partial-progress reward signal.

---

## Action Space

Actions are JSON objects sent to `POST /step`:

| `operation`      | `column`   | `params`                                         | Description                         |
|------------------|------------|--------------------------------------------------|-------------------------------------|
| `fill_missing`   | required   | `{"strategy": "median\|mean\|mode\|constant", "value": ...}` | Fill NaN values             |
| `drop_duplicates`| —          | —                                                | Remove duplicate rows               |
| `fix_format`     | required   | —                                                | Standardise phone/date/country col  |
| `replace_value`  | required   | `{"old": ..., "new": ...}`                       | Replace a specific value            |
| `drop_outliers`  | required   | —                                                | Remove IQR outliers in numeric col  |
| `fix_dtype`      | required   | `{"dtype": "float\|int\|str"}`                   | Cast column to correct dtype        |

**Example:**
```json
{"operation": "fill_missing", "column": "salary", "params": {"strategy": "median"}}
{"operation": "drop_duplicates"}
{"operation": "fix_format", "column": "signup_date"}
```

---

## Observation Space

The `POST /step` and `POST /reset` responses return:

```json
{
  "observation": {
    "done":             false,
    "reward":           0.05,
    "data_preview":     "name,age,salary,...\n...",
    "data_shape":       [100, 5],
    "missing_counts":   {"salary": 18, "age": 20},
    "duplicate_count":  0,
    "dtype_issues":     {},
    "task_description": "Task 1 (Easy) — Fill Missing Values\n...",
    "message":          "Filled 20 missing values in 'age' using median.",
    "step_count":       1,
    "current_score":    0.25
  },
  "reward": 0.05,
  "done": false,
  "info": {}
}
```

---

## Tasks

### Task 1 — Fill Missing Values (Easy)
- **Dataset:** 100-row employee records (name, age, salary, department, experience)
- **Issues:** ~20 % NaN in `age`, `salary`, `department`
- **Goal:** Fill all missing values
- **Grader:** `1.0 - remaining_nulls / original_nulls`
- **Max steps:** 20
- **Expected baseline score:** ~0.95

### Task 2 — Fix Formats + Remove Duplicates (Medium)
- **Dataset:** 200-row product catalog (product_id, price, phone, listed_date, …)
- **Issues:** Mixed phone formats, mixed date formats, 15 duplicate rows
- **Goal:** Standardise all formats and remove duplicates
- **Grader:** `0.35 × phone_score + 0.35 × date_score + 0.30 × dupe_score`
- **Max steps:** 30
- **Expected baseline score:** ~0.80

### Task 3 — Full Cleaning Pipeline (Hard)
- **Dataset:** 300-row customer database (name, age, purchase_amount, country, email, signup_date)
- **Issues:** Missing values (4 cols), 20 duplicates, outliers in `purchase_amount`, mixed country case, mixed date formats
- **Goal:** Clean all issues end-to-end
- **Grader:** `0.25 × null + 0.20 × dupe + 0.20 × outlier + 0.175 × country + 0.175 × date`
- **Max steps:** 40
- **Expected baseline score:** ~0.70

---

## Reward Function

| Scenario                   | Reward                             |
|----------------------------|------------------------------------|
| Progress (score improves)  | `new_score - old_score` (≥ 0)      |
| No effect                  | `-0.01`                            |
| Invalid operation          | `-0.05`                            |
| Episode completion (≥0.95) | `delta + 0.20` terminal bonus      |

Rewards are bounded to `[-0.05, 1.2]`. Partial rewards are emitted every step.

---

## API Endpoints

| Method | Path      | Description                       |
|--------|-----------|-----------------------------------|
| GET    | `/health` | Health check → `{"status":"ok"}`  |
| POST   | `/reset`  | Start episode. Body: `{"task_id": 1\|2\|3}` (optional; default: round-robin) |
| POST   | `/step`   | Execute action. Body: action JSON |
| POST   | `/state`  | Get episode state                 |
| GET    | `/docs`   | Interactive Swagger UI            |

---

## Setup & Usage

### Local (Python)
```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Docker
```bash
docker build -t data-cleaning-env .
docker run -p 8000:8000 data-cleaning-env
```

### Run Baseline Inference
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key"
export ENV_URL="http://localhost:8000"

python inference.py
```

---

## Baseline Scores

| Task | Difficulty | Score  |
|------|------------|--------|
| 1    | Easy       | 1.000  |
| 2    | Medium     | 1.000  |
| 3    | Hard       | 1.000  |
| avg  | —          | 1.000  |

*(Scores produced by `google/gemma-3-27b-it` via NVIDIA NIM, temperature=0)*

> Full agent step-by-step logs available in `inference_log.txt`
---

## Project Structure

```
openenv-data-cleaning/
├── server/
│   ├── environment.py        # Core env: reset/step/state + action dispatcher
│   ├── app.py                # FastAPI HTTP API
│   ├── data_generator.py     # Synthetic dataset generation (fixed seed=42)
│   └── tasks/
│       ├── task1_missing.py  # Task 1: missing values dataset + grader
│       ├── task2_format.py   # Task 2: format + duplicates dataset + grader
│       └── task3_pipeline.py # Task 3: full pipeline dataset + grader
├── models.py                 # Pydantic models (Action, Observation, State)
├── inference.py              # Baseline inference script
├── openenv.yaml              # OpenEnv manifest
├── Dockerfile
├── requirements.txt
└── README.md
```
