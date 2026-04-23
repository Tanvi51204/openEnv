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
  - multi-agent
  - data-quality
---

# DataMedic — AI Data Cleaning OpenEnv

An **agentic data quality environment** for training and evaluating AI agents on real-world data cleaning tasks.

An agent interacts with dirty pandas DataFrames through a standard `reset() / step() / state()` HTTP API, learning to fix missing values, duplicate rows, inconsistent formats, statistical outliers, and dtype errors — across **four progressively harder tasks** including a novel multi-source schema alignment challenge.

🤗 **Live HuggingFace Space:** https://srishtichugh-openenv-hack.hf.space
🖥️ **Live DataMedic UI:** https://srishtichugh-openenv-hack.hf.space
📖 **Interactive API docs:** https://srishtichugh-openenv-hack.hf.space/docs
✅ **Health check:** https://srishtichugh-openenv-hack.hf.space/health

---

## What Makes This Different

Most data cleaning tools are one-shot. DataMedic is an **RL training environment** where:

- The agent **diagnoses** a dirty dataset via `/profile` (completeness, uniqueness, validity %)
- It **plans** a treatment — every observation includes a `plan` field with the next recommended actions
- It **executes** cleaning operations step by step with dense per-step rewards
- It **receives a health certificate** via `/report` summarising what was fixed and how efficiently
- It **exports** the cleaned result via `/export`

Grounded in peer-reviewed research:
- **Bendinelli et al. 2025** — LLM Agents for Cleaning Tabular ML Datasets (arXiv:2503.06664)
- **CleanAgent** — Qi & Wang 2024 (arXiv:2403.08291)
- **AutoDCWorkflow** — EMNLP 2025 Findings
- **HoloClean** — Rekatsinas et al. 2017

---

## Environment Description & Motivation

Real-world datasets are almost never clean. Data engineers routinely spend 60–80% of their time on data cleaning. This environment turns that into an RL challenge with:

- **Deterministic, programmatic graders** — ground-truth DataFrames generated with `seed=42`; every reward is reproducible
- **Meaningful partial rewards** — dense delta reward every step, not just at episode end
- **Four difficulty levels** — easy → medium → hard → expert (multi-source merge)
- **Live DQ metrics** — completeness %, uniqueness %, validity % in every observation
- **Agentic planning** — `plan` field recommends next actions; `tried_operations` prevents loops
- **No external data downloads** — all datasets generated synthetically via `numpy` + `Faker`

---

## DataMedic UI

Open `https://srishtichugh-openenv-hack.hf.space` in your browser to see the live monitoring dashboard:

- **Health Score Ring** — animated score gauge, color-coded by severity (green/amber/red)
- **DQ Dimension Bars** — live completeness, uniqueness, validity bars updating each step
- **Score Trajectory Chart** — real-time line chart of score vs steps
- **Agent Treatment Plan** — next recommended actions shown before each step
- **Operation Log** — every action taken, result, and reward delta streamed live
- **Dataset Preview** — first 10 rows with NULL values highlighted in red
- **Export CSV** — download the cleaned DataFrame at any point

Click any task button — the dataset loads automatically and the demo agent runs end-to-end.

---

## Action Space

Actions are JSON objects sent to `POST /step`.

| `operation` | Required `column` | `params` | Description |
|---|---|---|---|
| `fill_missing` | ✅ | `{"strategy": "median\|mean\|mode\|constant", "value": ...}` | Fill NaN values in a column |
| `drop_duplicates` | ❌ | — | Remove all duplicate rows |
| `fix_format` | ✅ | — | Standardise phone/date/country format |
| `replace_value` | ✅ | `{"old": ..., "new": ...}` | Replace a specific value |
| `drop_outliers` | ✅ | — | Remove IQR outliers from a numeric column |
| `fix_dtype` | ✅ | `{"dtype": "float\|int\|str"}` | Cast column to correct dtype |
| `align_schema` | ❌ | — | Rename Source A columns to canonical schema *(Task 4 only)* |
| `merge_sources` | ❌ | — | Concatenate aligned Source A + Source B *(Task 4 only)* |

**Format rules enforced by `fix_format`:**

| Column | Target format |
|---|---|
| `phone` | `NNN-NNN-NNNN` |
| `listed_date` / `signup_date` | `YYYY-MM-DD` |
| `country` | Canonical name (`USA`, `UK`, `Canada`, `Australia`, `Germany`) |

---

## Observation Space

Every `POST /reset` and `POST /step` returns:

```json
{
  "observation": {
    "done":             false,
    "reward":           0.40,
    "data_preview":     "name,age,salary,...\n...",
    "data_shape":       [100, 5],
    "missing_counts":   {"age": 20, "salary": 20, "department": 10},
    "duplicate_count":  0,
    "dtype_issues":     {},
    "task_description": "Task 1 (Easy) — Fill Missing Values\n...",
    "message":          "Filled 20 missing values in 'age' using median.",
    "step_count":       1,
    "current_score":    0.4000,
    "dq_metrics": {
      "completeness_pct": 86.67,
      "uniqueness_pct":   100.0,
      "validity_pct":     94.5,
      "total_cells":      500,
      "null_cells":       50,
      "duplicate_rows":   0,
      "invalid_cells":    12
    },
    "tried_operations": ["fill_missing:age"],
    "plan": [
      "fill_missing on \"salary\" (20 nulls) using median",
      "fill_missing on \"department\" (10 nulls) using mode"
    ]
  },
  "reward": 0.40,
  "done":   false,
  "info":   {}
}
```

| Field | Type | Description |
|---|---|---|
| `done` | bool | Episode finished (score ≥ 0.95 or max steps reached) |
| `reward` | float | Per-step delta reward |
| `data_preview` | string | First 10 rows as CSV |
| `data_shape` | [int, int] | Current `[rows, cols]` |
| `missing_counts` | object | `{column: null_count}` for columns with NaN |
| `duplicate_count` | int | Number of duplicate rows |
| `dtype_issues` | object | `{column: issue_description}` |
| `task_description` | string | Full task instructions |
| `message` | string | Human-readable result of last action |
| `step_count` | int | Steps taken this episode |
| `current_score` | float | Running grader score 0.0–1.0 |
| `dq_metrics` | object | Completeness / uniqueness / validity % + raw counts |
| `tried_operations` | array | Operations already applied — prevents agent loops |
| `plan` | array | Up to 3 recommended next actions (rule-based planning engine) |

---

## Tasks

### Task 1 — Fill Missing Values *(Easy)*

| Property | Value |
|---|---|
| Dataset | 100-row employee records (name, age, salary, department, experience) |
| Issues | ~20% NaN in `age`, `salary`; ~10% NaN in `department` |
| Goal | Fill all missing values |
| Valid operations | `fill_missing` |
| Grader | `1.0 − remaining_nulls / original_nulls` |
| Max steps | 20 |
| Optimal steps | 3 |

### Task 2 — Fix Formats + Remove Duplicates *(Medium)*

| Property | Value |
|---|---|
| Dataset | 215-row product catalog (product_id, price, category, phone, listed_date) |
| Issues | ~60% phone numbers in mixed formats, ~60% dates in mixed formats, 15 duplicate rows |
| Goal | Standardise all phone/date formats and remove duplicates |
| Valid operations | `fix_format`, `drop_duplicates` |
| Grader | `0.35 × phone_score + 0.35 × date_score + 0.30 × dupe_score` |
| Max steps | 30 |
| Optimal steps | 3 |

### Task 3 — Full Cleaning Pipeline *(Hard)*

| Property | Value |
|---|---|
| Dataset | 320-row customer database (name, age, purchase_amount, country, email, signup_date) |
| Issues | Missing values (4 cols), 20 duplicate rows, outliers in `purchase_amount`, mixed country case, mixed date formats |
| Goal | Fix all issues end-to-end |
| Valid operations | All 6 operations |
| Grader | `0.25×null + 0.20×dupe + 0.20×outlier + 0.175×country + 0.175×date` |
| Max steps | 40 |
| Optimal steps | 8 |

### Task 4 — Multi-Source Schema Alignment + Merge *(Expert)*

| Property | Value |
|---|---|
| Source A | 150-row CRM export: `cust_id, full_name, Age, purchase_amt, Country, signup, email` |
| Source B | 100-row Marketing export: `customer_id, name, age_years, spend, country_name, registration_date, email` |
| Issues | Misaligned schemas, missing values, mixed country case, mixed date formats, 10 duplicate rows |
| Goal | Align schemas → merge → clean |
| Valid operations | `align_schema`, `merge_sources`, `fill_missing`, `fix_format`, `drop_duplicates` |
| Grader | `0.30×schema + 0.25×null + 0.20×country + 0.15×date + 0.10×dupe` |
| Max steps | 50 |
| Optimal steps | 8 |

*Inspired by Meta's DataSchema system — column-level semantic annotation across misaligned sources.*

---

## Reward Function

| Scenario | Reward |
|---|---|
| Score improves (delta > 0) | `new_score − old_score` (positive) |
| Operation had no effect | `−0.01` |
| Invalid operation / bad column | `−0.05` |

Rewards are bounded to **[−0.05, 0.99]**. Dense signal every step.

---

## Intelligence Endpoints (Phase 2)

| Method | Path | Description |
|---|---|---|
| `GET` | `/profile` | Rich per-column DQ profile — null %, unique %, min/max/mean, top values |
| `GET` | `/report` | Full episode cleaning summary — score improvement, efficiency, issues fixed |
| `GET` | `/export` | Download current cleaned DataFrame as CSV |

### `/profile` response example
```json
{
  "dq_metrics": {
    "completeness_pct": 90.0,
    "uniqueness_pct": 100.0,
    "validity_pct": 88.5
  },
  "columns": {
    "age": {"null_count": 20, "null_pct": 20.0, "min": 22, "max": 59, "mean": 40.3}
  }
}
```

### `/report` response example
```json
{
  "initial_score": 0.01,
  "final_score": 0.99,
  "score_improvement": 0.98,
  "steps_taken": 3,
  "step_efficiency_pct": 85.0,
  "issues_fixed": {"nulls_filled": 50, "dupes_removed": 15, "formats_fixed": 168},
  "completed": true
}
```

---

## All API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | DataMedic live monitoring UI |
| `GET` | `/health` | Health check → `{"status": "healthy"}` |
| `POST` | `/reset` | Start episode. Body: `{"task_id": 1\|2\|3\|4}` |
| `POST` | `/step` | Execute action. Body: action JSON |
| `GET` | `/state` | Episode metadata |
| `GET` | `/metadata` | Environment info + paper citations |
| `GET` | `/schema` | Full action/observation/state JSON schemas |
| `GET` | `/profile` | Rich data quality profile of current DataFrame |
| `GET` | `/report` | Full episode cleaning summary |
| `GET` | `/export` | Download cleaned DataFrame as CSV |
| `GET` | `/docs` | Interactive Swagger UI |

---

## Baseline Scores

| Task | Difficulty | Score |
|---|---|---|
| 1 — Fill Missing Values | Easy | 0.999 |
| 2 — Fix Formats + Duplicates | Medium | 0.999 |
| 3 — Full Cleaning Pipeline | Hard | 0.999 |
| 4 — Multi-Source Merge | Expert | 0.990 |
| **Average** | — | **0.997** |

---

## Setup & Usage

### Prerequisites
- Python 3.11+
- Docker (for containerised deployment)

### Local — Python
```bash
git clone https://github.com/Tanvi51204/openEnv.git
cd openEnv
pip install -r requirements.txt
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Then open:
- UI: http://localhost:8000
- Docs: http://localhost:8000/docs

### Local — Docker
```bash
docker build -t data-cleaning-env .
docker run -p 8000:8000 data-cleaning-env
```

### Run baseline inference
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-..."
export ENV_URL="http://localhost:8000"

python inference.py
```

Produces `[START]` / `[STEP]` / `[END]` lines to stdout and `baseline_scores.json`.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://api.openai.com/v1` | LLM API endpoint (OpenAI-compatible) |
| `MODEL_NAME` | `gpt-4o-mini` | Model identifier |
| `HF_TOKEN` | — | API key for LLM calls |
| `ENV_URL` | `http://localhost:8000` | Environment server URL |

---

## Project Structure

```
openenv-data-cleaning/
├── models.py              Pydantic contracts — Action / Observation / State / DQMetrics / Report
├── client.py              Sync HTTP client (reset / step / state / health)
├── inference.py           Baseline LLM agent with [START]/[STEP]/[END] logging
├── Dockerfile             python:3.11-slim, non-root user, HEALTHCHECK
├── requirements.txt       pip dependencies
└── server/
    ├── app.py             FastAPI routes + /profile + /report + /export + UI
    ├── environment.py     reset / step / state + 8 operations + planning engine + DQ metrics
    ├── data_generator.py  Synthetic dataset generation (seed=42, reproducible)
    ├── ui.html            DataMedic live monitoring dashboard
    └── tasks/
        ├── task1_missing.py    Easy   — fill NaN grader
        ├── task2_format.py     Medium — format + duplicates grader
        ├── task3_pipeline.py   Hard   — full pipeline grader
        └── task4_merge.py      Expert — multi-source schema alignment + merge grader
```

---

## Live Demo

🤗 **HuggingFace Space:** https://srishtichugh-openenv-hack.hf.space

- UI:     https://srishtichugh-openenv-hack.hf.space
- Health: https://srishtichugh-openenv-hack.hf.space/health
- Docs:   https://srishtichugh-openenv-hack.hf.space/docs
- Profile: https://srishtichugh-openenv-hack.hf.space/profile
- Report: https://srishtichugh-openenv-hack.hf.space/report