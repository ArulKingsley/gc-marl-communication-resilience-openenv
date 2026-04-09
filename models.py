"""
models.py — Pydantic type schemas for GC-MARL OpenEnv API.

Observation / Action / Reward / State schemas used by every endpoint.
Per-step rewards and episode reward summaries are normalised to [0.0, 1.0].
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
import math


# ── Primitive dims ────────────────────────────────────────────────────────────
OBS_DIM    = 8    # per-agent observation vector length
N_AGENTS   = 30   # matches CFG['N_NODES']
N_ACTIONS  = 4    # Discrete(4)


# ── Reset ─────────────────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    task_id: int = Field(
        default=0,
        ge=0, le=3,
        description="0=Easy, 1=Medium, 2=Hard (Byzantine), 3=Combined Siege",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility. None = random.",
    )

class ResetResponse(BaseModel):
    observations: List[List[float]] = Field(
        description=f"Per-agent observations; shape ({N_AGENTS}, {OBS_DIM}).",
    )
    task_id: int
    info: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("observations")
    @classmethod
    def check_shape(cls, v):
        if len(v) != N_AGENTS:
            raise ValueError(f"Expected {N_AGENTS} agents, got {len(v)}")
        for row in v:
            if len(row) != OBS_DIM:
                raise ValueError(f"Obs dim must be {OBS_DIM}, got {len(row)}")
        return v


# ── Step ──────────────────────────────────────────────────────────────────────
class StepRequest(BaseModel):
    actions: List[int] = Field(
        description=f"One discrete action per agent; length {N_AGENTS}, values in [0,{N_ACTIONS-1}].",
    )

    @field_validator("actions")
    @classmethod
    def check_actions(cls, v):
        if len(v) != N_AGENTS:
            raise ValueError(f"Expected {N_AGENTS} actions, got {len(v)}")
        for a in v:
            if not (0 <= a < N_ACTIONS):
                raise ValueError(f"Action {a} out of range [0,{N_ACTIONS-1}]")
        return v

class StepResponse(BaseModel):
    observations:  List[List[float]]
    rewards:       List[float] = Field(
        description="Per-agent rewards, normalised to [0.0, 1.0].",
    )
    terminated:    List[bool]
    truncated:     bool
    info:          Dict[str, Any]


# ── State (GET /state) ────────────────────────────────────────────────────────
class NodeState(BaseModel):
    node_id:   int
    alive:     bool
    byzantine: bool
    energy:    float
    load:      float
    node_type: str   # "gateway" | "mobile" | "constrained"

class NetworkState(BaseModel):
    step:            int
    task_id:         int
    fiedler_value:   float = Field(description="Current λ₂ of alive subgraph.")
    alive_count:     int
    total_nodes:     int
    nodes:           List[NodeState]
    episode_score:   Optional[float] = Field(
        default=None,
        description="Running normalised score in [0,1]. None until episode ends.",
    )
    reward_score:    Optional[float] = Field(
        default=None,
        description="Sigmoid-normalised cumulative team reward in [0,1].",
    )


# ── Reward normalisation utility ──────────────────────────────────────────────
def sigmoid_normalise(total_reward: float, n_nodes: int = N_AGENTS) -> float:
    """
    Map unbounded cumulative reward → [0.0, 1.0].
    score = 1 / (1 + exp(-total_reward / n_nodes))
    """
    x = total_reward / max(n_nodes, 1)
    # Numerically stable sigmoid for large-magnitude rewards.
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# ── Per-task grader functions ─────────────────────────────────────────────────
def grade_easy(fiedler_final: float, fiedler_initial: float) -> float:
    """Task 0: final λ₂ normalised by initial λ₂, clamped to [0,1]."""
    if fiedler_initial <= 0:
        return 0.0
    score = fiedler_final / fiedler_initial
    return float(min(1.0, max(0.0, score)))


def grade_medium(
    alive_count: int,
    n_nodes: int,
    fiedler_final: float,
    fiedler_initial: float,
) -> float:
    """Task 1: average of alive_ratio and fiedler_ratio, clamped to [0,1]."""
    alive_ratio = alive_count / max(n_nodes, 1)
    fiedler_ratio = (fiedler_final / fiedler_initial) if fiedler_initial > 0 else 0.0
    score = (alive_ratio + fiedler_ratio) / 2.0
    return float(min(1.0, max(0.0, score)))


def grade_hard(connected_steps: int, total_steps: int) -> float:
    """Task 2: fraction of steps where alive subgraph stayed connected."""
    if total_steps == 0:
        return 0.0
    score = connected_steps / total_steps
    return float(min(1.0, max(0.0, score)))


def grade_combined(
    alive_count: int,
    n_nodes: int,
    fiedler_final: float,
    fiedler_initial: float,
    connected_steps: int,
    total_steps: int,
) -> float:
    """Task 3: weighted blend of medium and hard graders."""
    medium = grade_medium(alive_count, n_nodes, fiedler_final, fiedler_initial)
    hard = grade_hard(connected_steps, total_steps)
    return float(round(0.4 * medium + 0.6 * hard, 6))
