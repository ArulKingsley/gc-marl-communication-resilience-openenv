"""
smoke_test.py — lightweight OpenEnv API smoke test.

Checks endpoint flow in one command:
  GET /health -> POST /reset -> GET /state -> POST /step -> GET /state

Examples:
  python smoke_test.py
  python smoke_test.py --host http://localhost:8000 --task 2 --seed 7
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--task", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=10)
    return parser.parse_args()


def request_json(host: str, method: str, path: str, timeout: int, payload: dict | None = None) -> dict:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        url=f"{host}{path}",
        method=method,
        data=data,
        headers=headers,
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def fail(message: str) -> None:
    print(f"SMOKE TEST FAILED: {message}")
    sys.exit(1)


def validate_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        fail(f"{field_name} must be bool, got {type(value).__name__}")
    return value


def main() -> None:
    args = parse_args()
    host = args.host.rstrip("/")

    try:
        health = request_json(host, "GET", "/health", args.timeout)
        if health.get("status") != "ok":
            fail(f"Unexpected /health payload: {health}")
        print(f"Health ok: {health}")

        reset_resp = request_json(
            host,
            "POST",
            "/reset",
            args.timeout,
            payload={"task_id": args.task, "seed": args.seed},
        )

        observations = reset_resp.get("observations")
        if not isinstance(observations, list) or not observations:
            fail("/reset did not return a non-empty observations list")
        if reset_resp.get("task_id") != args.task:
            fail(f"/reset task_id mismatch: expected {args.task}, got {reset_resp.get('task_id')}")

        n_agents = len(observations)
        first_obs = observations[0]
        if not isinstance(first_obs, list) or len(first_obs) != 8:
            fail("/reset observation shape mismatch; expected per-agent dimension 8")

        print(f"Reset ok: task={args.task}, agents={n_agents}, obs_dim={len(first_obs)}")

        state_before = request_json(host, "GET", "/state", args.timeout)
        if state_before.get("task_id") != args.task:
            fail("/state task_id does not match current reset task")
        if state_before.get("total_nodes") != n_agents:
            fail("/state total_nodes does not match observation count")

        print(
            "State before step: "
            f"step={state_before.get('step')}, alive={state_before.get('alive_count')}, "
            f"fiedler={state_before.get('fiedler_value')}"
        )

        actions = [0] * n_agents
        step_resp = request_json(
            host,
            "POST",
            "/step",
            args.timeout,
            payload={"actions": actions},
        )

        rewards = step_resp.get("rewards")
        if not isinstance(rewards, list) or len(rewards) != n_agents:
            fail("/step rewards length mismatch")

        for idx, reward in enumerate(rewards):
            if not isinstance(reward, (int, float)):
                fail(f"/step reward at index {idx} is not numeric")
            if reward < 0.0 or reward > 1.0:
                fail(f"/step reward at index {idx} out of [0,1]: {reward}")

        terminated = step_resp.get("terminated")
        if not isinstance(terminated, list) or len(terminated) != n_agents:
            fail("/step terminated length mismatch")
        for idx, value in enumerate(terminated):
            validate_bool(value, f"terminated[{idx}]")

        validate_bool(step_resp.get("truncated"), "truncated")

        info = step_resp.get("info")
        if not isinstance(info, dict):
            fail("/step info is missing or invalid")

        print(
            "Step ok: "
            f"mean_reward={sum(rewards)/len(rewards):.4f}, "
            f"alive={info.get('alive_count')}, fiedler={info.get('fiedler')}"
        )

        state_after = request_json(host, "GET", "/state", args.timeout)
        before_step = int(state_before.get("step", 0))
        after_step = int(state_after.get("step", -1))
        if after_step < before_step + 1:
            fail(f"/state step did not advance (before={before_step}, after={after_step})")

        print(
            "State after step: "
            f"step={after_step}, alive={state_after.get('alive_count')}, "
            f"episode_score={state_after.get('episode_score')}, "
            f"reward_score={state_after.get('reward_score')}"
        )

        print("SMOKE TEST PASSED")

    except urllib.error.URLError as exc:
        fail(f"Could not reach {host}: {exc}")
    except json.JSONDecodeError as exc:
        fail(f"Received non-JSON response: {exc}")


if __name__ == "__main__":
    main()
