"""OpenEnv client for the GC-MARL communication resilience environment."""

from __future__ import annotations

from typing import Dict, List

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class IoTAction(Action):
    """Joint action for all agents in one environment step."""

    actions: List[int] = Field(default_factory=list)


class IoTObservation(Observation):
    """Observation payload returned by the environment."""

    observations: List[List[float]] = Field(default_factory=list)


class GCMarlClient(EnvClient[IoTAction, IoTObservation, State]):
    """WebSocket-capable client for interacting with the GC-MARL environment."""

    def _step_payload(self, action: IoTAction) -> Dict:
        return {"actions": action.actions}

    def _parse_result(self, payload: Dict) -> StepResult[IoTObservation]:
        obs = payload.get("observations", [])
        done = bool(payload.get("truncated", False))
        info = payload.get("info", {})
        reward = info.get("reward_score")

        observation = IoTObservation(
            observations=obs,
            done=done,
            reward=reward,
            metadata=info,
        )
        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=None,
            step_count=int(payload.get("step", 0)),
        )
