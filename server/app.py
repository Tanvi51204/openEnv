"""
FastAPI application exposing the OpenEnv-compatible HTTP API.

Endpoints:
  GET  /health       Health check
  GET  /metadata     Environment info
  GET  /schema       Action / observation / state schemas
  POST /reset        Start new episode
  POST /step         Execute cleaning action (with 30s timeout)
  GET  /state        Episode metadata
  POST /state        Episode metadata (backward compat)
  GET  /profile      Rich data quality profile of current DataFrame
  GET  /report       Full episode cleaning summary (health certificate)
  GET  /export       Download current cleaned DataFrame as CSV
"""

import asyncio
import os
from typing import Any, Dict, Optional
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn

from models import DataCleaningAction, DataCleaningObservation, DataCleaningState, EpisodeReport
from server.environment import DataCleaningEnvironment

app = FastAPI(
    title="Data Cleaning OpenEnv",
    description=(
        "A real-world data cleaning environment for AI agent training and evaluation. "
        "An agent interacts with dirty pandas DataFrames through a standard reset/step/state API, "
        "learning to fix missing values, duplicates, format inconsistencies, outliers, and dtype errors. "
        "Grounded in CleanAgent (2024), AutoDCWorkflow (EMNLP 2025), and Meta-scale data quality principles."
    ),
    version="0.2.0",
)

# Single shared environment instance
env = DataCleaningEnvironment()

STEP_TIMEOUT_SECONDS = 30


class ResetRequest(BaseModel):
    task_id: Optional[int] = None


class StepResponse(BaseModel):
    observation: DataCleaningObservation
    reward: float
    done: bool
    info: dict = {}


# ------------------------------------------------------------------
# Core OpenEnv routes
# ------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui():
    """DataMedic — live agent monitoring dashboard."""
    ui_path = os.path.join(os.path.dirname(__file__), "ui.html")
    with open(ui_path, "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    return {
        "name": "data-cleaning-env",
        "description": (
            "A real-world data cleaning RL environment. The agent diagnoses dirty datasets, "
            "plans a treatment, executes cleaning operations step-by-step, and produces a "
            "health certificate — grounded in AutoDCWorkflow, CleanAgent, and HoloClean research."
        ),
        "version": "0.2.0",
        "tags": ["openenv", "data-cleaning", "rl", "real-world", "agentic"],
        "tasks": [
            {"id": "task1", "name": "Fill Missing Values",               "difficulty": "easy"},
            {"id": "task2", "name": "Fix Formats and Remove Duplicates", "difficulty": "medium"},
            {"id": "task3", "name": "Full Cleaning Pipeline",            "difficulty": "hard"},
            {"id": "task4", "name": "Multi-Source Schema Alignment + Merge", "difficulty": "expert"},
        ],
        "observation_extras": ["dq_metrics", "tried_operations", "plan"],
        "papers": [
            "Bendinelli et al. 2025 — LLM Agents for Cleaning Tabular ML Datasets (arXiv:2503.06664)",
            "CleanAgent — Qi & Wang 2024 (arXiv:2403.08291)",
            "AutoDCWorkflow — EMNLP 2025 Findings",
            "HoloClean — Rekatsinas et al. 2017",
        ],
    }


@app.get("/schema")
def schema():
    return {
        "action": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "fill_missing", "drop_duplicates", "fix_format",
                        "replace_value", "drop_outliers", "fix_dtype",
                        "align_schema", "merge_sources",
                    ],
                },
                "column": {"type": "string", "nullable": True},
                "params":  {"type": "object", "nullable": True},
            },
            "required": ["operation"],
        },
        "observation": {
            "type": "object",
            "properties": {
                "done":             {"type": "boolean"},
                "reward":           {"type": "number"},
                "data_preview":     {"type": "string"},
                "data_shape":       {"type": "array", "items": {"type": "integer"}},
                "missing_counts":   {"type": "object"},
                "duplicate_count":  {"type": "integer"},
                "dtype_issues":     {"type": "object"},
                "task_description": {"type": "string"},
                "message":          {"type": "string"},
                "step_count":       {"type": "integer"},
                "current_score":    {"type": "number"},
                "dq_metrics":       {"type": "object", "description": "Completeness/uniqueness/validity %"},
                "tried_operations": {"type": "array",  "description": "Operations already applied"},
                "plan":             {"type": "array",  "description": "Agent recommended next actions"},
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "episode_id":       {"type": "string"},
                "task_id":          {"type": "integer"},
                "step_count":       {"type": "integer"},
                "max_steps":        {"type": "integer"},
                "total_errors":     {"type": "integer"},
                "errors_remaining": {"type": "integer"},
            },
        },
    }


@app.post("/reset", response_model=StepResponse)
def reset(req: ResetRequest = ResetRequest()):
    try:
        obs = env.reset(task_id=req.task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StepResponse(observation=obs, reward=obs.reward, done=False)


@app.post("/step", response_model=StepResponse)
async def step(body: Dict[str, Any] = Body(...)):
    """
    Accept both openenv-core wrapped format:
        {"action": {"operation": "...", ...}, "timeout_s": 15}
    and direct format:
        {"operation": "...", "column": "...", "params": {...}}
    Times out after 30 seconds to prevent hanging during evaluation.
    """
    action_data = body.get("action", body)
    try:
        action = DataCleaningAction(**action_data)
        loop = asyncio.get_event_loop()
        obs = await asyncio.wait_for(
            loop.run_in_executor(None, env.step, action),
            timeout=STEP_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Step timed out after {STEP_TIMEOUT_SECONDS}s")
    except (TypeError, KeyError, Exception) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StepResponse(observation=obs, reward=obs.reward, done=obs.done)


@app.get("/state", response_model=DataCleaningState)
def state_get():
    return env.state()


@app.post("/state", response_model=DataCleaningState)
def state_post():
    return env.state()


# ------------------------------------------------------------------
# Phase 2: Intelligence endpoints
# ------------------------------------------------------------------

@app.get("/profile")
def profile():
    """
    Rich data quality profile of the current DataFrame.

    Returns per-column statistics (null %, unique %, min/max/mean for numerics,
    top values for categoricals) plus dataset-level DQ metrics:
    completeness %, uniqueness %, validity %.

    Inspired by standard Data Quality dimensions (ISO 8000) and
    Meta's data schematization approach.
    """
    try:
        return env.get_profile()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/report", response_model=EpisodeReport)
def report():
    """
    Full episode cleaning summary — the 'health certificate'.

    Returns: initial vs final score, score improvement, step efficiency,
    ordered list of operations applied, issues fixed by category,
    and final DQ metrics. Call after the episode completes for best results.
    """
    try:
        return env.get_report()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/export")
def export():
    """
    Download the current (cleaned) DataFrame as a CSV file.

    Returns the live state of the DataFrame — call after the agent
    finishes cleaning to get the cleaned output.
    """
    try:
        csv_data = env.get_export()
        return PlainTextResponse(
            content=csv_data,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=cleaned_data.csv"},
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, workers=1)


if __name__ == "__main__":
    main()