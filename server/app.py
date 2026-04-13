"""
server/app.py — FastAPI server exposing the GC-MARL OpenEnv.

Endpoints:
  POST /reset  — start a new episode (accepts task_id + optional seed)
  POST /step   — advance one timestep with joint actions
  GET  /state  — current network snapshot (no side effects)
  GET  /health — liveness probe for Docker and orchestration
"""

from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException

from env import IoTNetworkEnv
from models import (
    EPSILON,
    NetworkState,
    NodeState,
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResponse,
    _clamp,
)

app = FastAPI(
    title="GC-MARL IoT Self-Healing - OpenEnv API",
    description=(
        "Three-task MARL benchmark for decentralised IoT network recovery. "
        "Task 0=Easy (random failures), Task 1=Medium (cascades), "
        "Task 2=Hard (Byzantine adversaries)."
    ),
    version="1.0.0",
)

# Single shared environment instance for sequential evaluation workflows.
_env: IoTNetworkEnv | None = None
_last_obs: list[list[float]] | None = None


def _sanitise_info_payload(info: dict) -> dict:
    payload = dict(info)

    fiedler = payload.get("fiedler")
    if fiedler is None:
        payload["fiedler"] = EPSILON
    else:
        fv = float(fiedler)
        payload["fiedler"] = _clamp(fv) if fv > 0.0 else EPSILON

    episode_score = payload.get("episode_score")
    payload["episode_score"] = _clamp(float(episode_score)) if episode_score is not None else EPSILON

    reward_score = payload.get("reward_score")
    payload["reward_score"] = _clamp(float(reward_score)) if reward_score is not None else EPSILON

    return payload


def _require_env() -> IoTNetworkEnv:
    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialised. Call POST /reset first.",
        )
    return _env


@app.post("/reset", response_model=ResetResponse)
def reset(body: ResetRequest | None = None) -> ResetResponse:
    global _env, _last_obs

    body = body or ResetRequest()

    if _env is None:
        _env = IoTNetworkEnv(task_id=body.task_id)

    obs, info = _env.reset(task_id=body.task_id, seed=body.seed)
    _last_obs = obs.tolist()
    safe_info = _sanitise_info_payload(info)

    return ResetResponse(
        observations=_last_obs,
        task_id=body.task_id,
        info=safe_info,
    )


@app.post("/step", response_model=StepResponse)
def step(body: StepRequest) -> StepResponse:
    env = _require_env()

    obs, rewards, terminated, truncated, info = env.step(
        np.asarray(body.actions, dtype=np.int32)
    )

    global _last_obs
    _last_obs = obs.tolist()
    safe_info = _sanitise_info_payload(info)

    return StepResponse(
        observations=_last_obs,
        rewards=[round(float(r), 6) for r in rewards],
        terminated=[bool(t) for t in terminated],
        truncated=bool(truncated),
        info=safe_info,
    )


@app.get("/state", response_model=NetworkState)
def state() -> NetworkState:
    env = _require_env()
    snapshot = env.get_state_snapshot()

    return NetworkState(
        step=snapshot["step"],
        task_id=snapshot["task_id"],
        fiedler_value=snapshot["fiedler_value"],
        alive_count=snapshot["alive_count"],
        total_nodes=snapshot["total_nodes"],
        nodes=[NodeState(**node) for node in snapshot["nodes"]],
        episode_score=snapshot["episode_score"],
        reward_score=snapshot.get("reward_score"),
    )


@app.get("/health")
def health() -> dict[str, bool | str]:
    return {"status": "ok", "env_ready": _env is not None}


def main() -> None:
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
