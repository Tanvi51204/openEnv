"""
FastAPI application exposing the OpenEnv-compatible HTTP API.
Endpoints: GET /health, POST /reset, POST /step, POST /state, GET /docs
"""

from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from models import DataCleaningAction, DataCleaningObservation, DataCleaningState
from server.environment import DataCleaningEnvironment

app = FastAPI(
    title="Data Cleaning OpenEnv",
    description="A real-world data cleaning environment for AI agent training.",
    version="0.1.0",
)

# Single shared environment instance (stateful server)
env = DataCleaningEnvironment()


class ResetRequest(BaseModel):
    task_id: Optional[int] = None


class StepResponse(BaseModel):
    observation: DataCleaningObservation
    reward: float
    done: bool
    info: dict = {}


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset", response_model=StepResponse)
def reset(req: ResetRequest = ResetRequest()):
    try:
        obs = env.reset(task_id=req.task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StepResponse(observation=obs, reward=0.0, done=False)


@app.post("/step", response_model=StepResponse)
def step(action: DataCleaningAction):
    try:
        obs = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StepResponse(observation=obs, reward=obs.reward, done=obs.done)


@app.post("/state", response_model=DataCleaningState)
def state():
    return env.state()


# ------------------------------------------------------------------
# Entry point (required by openenv-core and [project.scripts])
# ------------------------------------------------------------------

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()