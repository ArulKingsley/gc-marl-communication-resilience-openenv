"""
smoke_test.py — robust OpenEnv API smoke test with Pydantic validation.

Checks endpoint flow:
  GET /health -> POST /reset -> GET /state -> (POST /step -> GET /state) x3
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

from pydantic import ValidationError

from models import NetworkState, ResetResponse, StepResponse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--task", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=10)
    return parser.parse_args()


def request_json(
    host: str, method: str, path: str, timeout: int, payload: dict | None = None
) -> tuple[int, dict]:
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

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return response.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read().decode("utf-8")
            body = json.loads(raw) if raw else {}
        except Exception:
            body = {}
        return exc.code, body


def fail(message: str) -> None:
    print(f"SMOKE TEST FAILED: {message}", file=sys.stderr)
    sys.exit(1)


def expect_200(endpoint: str, status: int) -> None:
    if status != 200:
        fail(f"{endpoint} returned {status}, expected 200")


def main() -> None:
    args = parse_args()
    host = args.host.rstrip("/")

    try:
        status, health = request_json(host, "GET", "/health", args.timeout)
        expect_200("GET /health", status)
        if health.get("status") != "ok":
            fail(f"/health payload unexpected: {health}")
        print("PASS /health")

        status, reset_raw = request_json(
            host,
            "POST",
            "/reset",
            args.timeout,
            payload={"task_id": args.task, "seed": args.seed},
        )
        expect_200("POST /reset", status)
        try:
            reset_resp = ResetResponse.model_validate(reset_raw)
        except ValidationError as exc:
            fail(f"/reset schema validation failed: {exc}")
        n_agents = len(reset_resp.observations)
        print("PASS /reset")

        status, state_raw = request_json(host, "GET", "/state", args.timeout)
        expect_200("GET /state after reset", status)
        try:
            state = NetworkState.model_validate(state_raw)
        except ValidationError as exc:
            fail(f"/state schema validation failed: {exc}")
        if state.task_id != args.task:
            fail(f"/state task_id mismatch: {state.task_id} != {args.task}")
        if state.total_nodes != n_agents:
            fail(f"/state total_nodes mismatch: {state.total_nodes} != {n_agents}")
        print("PASS /state after reset")

        actions = [0] * n_agents
        for idx in range(1, 4):
            status, step_raw = request_json(
                host,
                "POST",
                "/step",
                args.timeout,
                payload={"actions": actions},
            )
            expect_200(f"POST /step #{idx}", status)
            try:
                step = StepResponse.model_validate(step_raw)
            except ValidationError as exc:
                fail(f"/step #{idx} schema validation failed: {exc}")

            if len(step.rewards) != n_agents:
                fail(f"/step #{idx} rewards length mismatch")
            if len(step.terminated) != n_agents:
                fail(f"/step #{idx} terminated length mismatch")
            print(f"PASS /step #{idx}")

            status, state_raw = request_json(host, "GET", "/state", args.timeout)
            expect_200(f"GET /state after step #{idx}", status)
            try:
                NetworkState.model_validate(state_raw)
            except ValidationError as exc:
                fail(f"/state after step #{idx} schema validation failed: {exc}")
            print(f"PASS /state after step #{idx}")

        print("SMOKE TEST PASSED")
    except urllib.error.URLError as exc:
        fail(f"Network error reaching {host}: {exc}")
    except json.JSONDecodeError as exc:
        fail(f"Non-JSON response: {exc}")


if __name__ == "__main__":
    main()
