"""
Baseline inference script for the Data Cleaning OpenEnv environment.
Uses the OpenAI client against all 3 tasks and reports scores.

Required environment variables:
    API_BASE_URL   — LLM API endpoint (OpenAI-compatible)
    MODEL_NAME     — model identifier
    HF_TOKEN       — API key
    ENV_URL        — environment server URL (default: http://localhost:8000)
"""

import json
import os
import sys
import time
import httpx
from openai import OpenAI

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_URL      = os.environ.get("ENV_URL",      "http://localhost:8000")

if not HF_TOKEN:
    print("[WARNING] HF_TOKEN is not set — LLM calls may fail.", file=sys.stderr)

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are a data cleaning agent. You control a data cleaning environment
through JSON actions. Each turn you receive an observation JSON describing the current state
of a dataset (preview, missing counts, duplicate count, dtype issues, current score, etc.)
and a task description.

Your job is to pick the single best action to improve the dataset quality.

Respond ONLY with a valid JSON object — no markdown, no explanation, just the JSON.

Available operations and their required parameters:

1. fill_missing
   {"operation": "fill_missing", "column": "<col>", "params": {"strategy": "median|mean|mode|constant", "value": <only if constant>}}

2. drop_duplicates
   {"operation": "drop_duplicates"}

3. fix_format
   {"operation": "fix_format", "column": "phone|listed_date|signup_date|country"}

4. replace_value
   {"operation": "replace_value", "column": "<col>", "params": {"old": "<val>", "new": "<val>"}}

5. drop_outliers
   {"operation": "drop_outliers", "column": "<numeric_col>"}

6. fix_dtype
   {"operation": "fix_dtype", "column": "<col>", "params": {"dtype": "float|int|str"}}

Rules:
- Address the highest-impact issues first (missing values > duplicates > outliers > format).
- Do not repeat an operation that returned no effect (watch the 'message' field).
- Stop when current_score >= 0.95.
"""


# ------------------------------------------------------------------
# HTTP helpers
# ------------------------------------------------------------------

def api_post(path: str, payload: dict = None) -> dict:
    url = ENV_URL.rstrip("/") + path
    resp = httpx.post(url, json=payload or {}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def api_get(path: str) -> dict:
    url = ENV_URL.rstrip("/") + path
    resp = httpx.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ------------------------------------------------------------------
# Agent loop
# ------------------------------------------------------------------

def obs_to_text(obs: dict) -> str:
    lines = [
        f"current_score: {obs['current_score']}",
        f"step_count:    {obs['step_count']}",
        f"data_shape:    {obs['data_shape']}",
        f"duplicate_count: {obs['duplicate_count']}",
        f"missing_counts: {json.dumps(obs['missing_counts'])}",
        f"dtype_issues:   {json.dumps(obs['dtype_issues'])}",
        f"message:        {obs['message']}",
        "",
        "=== DATA PREVIEW (first 10 rows) ===",
        obs["data_preview"],
        "",
        "=== TASK DESCRIPTION ===",
        obs["task_description"],
    ]
    return "\n".join(lines)


def run_task(task_id: int) -> float:
    print(f"\n{'='*60}")
    print(f"  Running Task {task_id}")
    print(f"{'='*60}")

    result  = api_post("/reset", {"task_id": task_id})
    obs     = result["observation"]
    history = []

    for step_num in range(1, 50):
        if obs["done"]:
            break

        obs_text = obs_to_text(obs)
        history.append({"role": "user", "content": obs_text})

        response = client.chat.completions.create(
            model    = MODEL_NAME,
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history,
            temperature = 0.0,
            max_tokens  = 256,
        )
        action_str = response.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": action_str})

        # Parse action
        try:
            action = json.loads(action_str)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code fence
            import re
            m = re.search(r"\{.*\}", action_str, re.DOTALL)
            if m:
                try:
                    action = json.loads(m.group())
                except Exception:
                    print(f"  Step {step_num}: Could not parse action JSON, skipping.")
                    break
            else:
                print(f"  Step {step_num}: No JSON found in response, skipping.")
                break

        print(f"  Step {step_num:2d} | score={obs['current_score']:.4f} | action={json.dumps(action)}")

        result = api_post("/step", action)
        obs    = result["observation"]
        print(f"           → {obs['message']}")

        # Slight delay to stay within rate limits on free-tier endpoints
        time.sleep(0.3)

    final_score = obs["current_score"]
    print(f"\n  Task {task_id} final score: {final_score:.4f}  (steps used: {obs['step_count']})")
    return final_score


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    print("Data Cleaning OpenEnv — Baseline Inference")
    print(f"Model : {MODEL_NAME}")
    print(f"Env   : {ENV_URL}")

    # Smoke-test health endpoint
    health = api_get("/health")
    assert health.get("status") == "ok", f"Health check failed: {health}"
    print("Health check: OK\n")

    scores = {}
    for task_id in [1, 2, 3]:
        scores[f"task{task_id}"] = run_task(task_id)

    print("\n" + "="*60)
    print("  BASELINE RESULTS")
    print("="*60)
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}")
    avg = sum(scores.values()) / len(scores)
    print(f"  average: {avg:.4f}")
    print("="*60)

    # Write scores to file for automated validators
    with open("baseline_scores.json", "w") as f:
        json.dump({"scores": scores, "average": avg}, f, indent=2)
    print("\nScores written to baseline_scores.json")


if __name__ == "__main__":
    main()
