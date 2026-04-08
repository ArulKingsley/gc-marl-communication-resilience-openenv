"""
inference.py — Evaluation harness for OpenEnv GC-MARL tasks.

Default behavior:
  - 3 tasks (0/1/2)
  - 5 seeds per task
  - 3 policies (random, greedy, heuristic)

Outputs:
  - One JSON log per (agent, task, seed)
  - One aggregate summary JSON file
    - Structured stdout logs using [START], [STEP], [END] tags
    - Console summary table
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Callable
import urllib.request

import numpy as np
from openai import OpenAI

N_AGENTS = 30
TASK_NAMES = {
    0: "Easy",
    1: "Medium",
    2: "Hard (Byzantine)",
}
ALL_TASKS = [0, 1, 2]
ALL_AGENTS = ["random", "greedy", "heuristic"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument(
        "--api-base-url",
        default=None,
        help="LLM API base URL. Default uses API_BASE_URL env var.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="LLM model identifier. Default uses MODEL_NAME env var.",
    )
    parser.add_argument(
        "--task",
        type=int,
        choices=ALL_TASKS,
        default=None,
        help="Run only one task (0/1/2). Default: run all tasks.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of seeds per task. Default: 5.",
    )
    parser.add_argument(
        "--agents",
        default="all",
        help="Comma-separated policies to run (random,greedy,heuristic) or 'all'.",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory for per-run and summary JSON logs.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional override for HF token. Default uses HF_TOKEN env var.",
    )
    return parser.parse_args()


def resolve_agents(agents_arg: str) -> list[str]:
    if agents_arg.strip().lower() == "all":
        return list(ALL_AGENTS)

    chosen = [part.strip().lower() for part in agents_arg.split(",") if part.strip()]
    invalid = [name for name in chosen if name not in ALL_AGENTS]
    if invalid:
        raise ValueError(f"Unknown agent(s): {invalid}. Valid options: {ALL_AGENTS} or 'all'.")
    if not chosen:
        raise ValueError("No valid agents provided.")
    return chosen


def resolve_required_config(args: argparse.Namespace) -> tuple[str, str, str]:
    api_base_url = (args.api_base_url or os.environ.get("API_BASE_URL") or "").strip()
    model_name = (args.model_name or os.environ.get("MODEL_NAME") or "").strip()
    hf_token = (args.hf_token or os.environ.get("HF_TOKEN") or "").strip()

    missing: list[str] = []
    if not api_base_url:
        missing.append("API_BASE_URL")
    if not model_name:
        missing.append("MODEL_NAME")
    if not hf_token:
        missing.append("HF_TOKEN")

    if missing:
        raise RuntimeError(
            "Missing required environment configuration: "
            + ", ".join(missing)
            + ". Define API_BASE_URL, MODEL_NAME, and HF_TOKEN before running inference."
        )

    return api_base_url, model_name, hf_token


def build_openai_client(api_base_url: str, hf_token: str) -> OpenAI:
    return OpenAI(base_url=api_base_url, api_key=hf_token)


def emit_structured(tag: str, fields: dict) -> None:
    payload = json.dumps(fields, separators=(",", ":"), ensure_ascii=True)
    print(f"[{tag}] {payload}", flush=True)


def _auth_headers(token: str | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def post_json(host: str, path: str, body: dict, token: str | None = None) -> dict:
    payload = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json", **_auth_headers(token)}
    request = urllib.request.Request(
        url=f"{host}{path}",
        data=payload,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def get_json(host: str, path: str, timeout: int = 10, token: str | None = None) -> dict:
    request = urllib.request.Request(
        url=f"{host}{path}",
        headers=_auth_headers(token),
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def random_agent(obs: list[list[float]]) -> list[int]:
    del obs
    return [random.randint(0, 3) for _ in range(N_AGENTS)]


def greedy_agent(obs: list[list[float]]) -> list[int]:
    actions: list[int] = []
    for row in obs:
        load = row[2]
        actions.append(1 if load > 0.3 else 0)
    return actions


def heuristic_agent(obs: list[list[float]]) -> list[int]:
    actions: list[int] = []
    for row in obs:
        energy = row[1]
        load = row[2]
        alive_ratio = row[3]

        if load > 0.35 and energy > 0.1:
            actions.append(1)
        elif load > 0.25 and energy <= 0.1:
            actions.append(2)
        elif alive_ratio < 0.5:
            actions.append(3)
        else:
            actions.append(0)
    return actions


AGENT_FNS: dict[str, Callable[[list[list[float]]], list[int]]] = {
    "random": random_agent,
    "greedy": greedy_agent,
    "heuristic": heuristic_agent,
}


def wait_for_server(host: str, token: str | None, retries: int = 15, delay_seconds: int = 2) -> bool:
    for attempt in range(1, retries + 1):
        try:
            get_json(host, "/health", token=token)
            return True
        except Exception:
            emit_structured(
                "STEP",
                {
                    "phase": "wait_for_server",
                    "attempt": attempt,
                    "status": "retry",
                },
            )
            time.sleep(delay_seconds)
    return False


def run_episode(
    host: str,
    task_id: int,
    seed: int,
    agent_name: str,
    token: str | None,
    model_name: str,
) -> dict:
    agent_fn = AGENT_FNS[agent_name]

    emit_structured(
        "START",
        {
            "agent": agent_name,
            "task_id": task_id,
            "task_name": TASK_NAMES[task_id],
            "seed": seed,
            "model_name": model_name,
        },
    )

    reset_resp = post_json(host, "/reset", {"task_id": task_id, "seed": seed}, token=token)
    obs = reset_resp["observations"]

    reward_series: list[float] = []
    fiedler_series: list[float] = []
    alive_series: list[int] = []
    connected_steps = 0
    total_steps = 0
    episode_score = 0.0
    reward_score = 0.0

    while True:
        actions = agent_fn(obs)
        step_resp = post_json(host, "/step", {"actions": actions}, token=token)

        obs = step_resp["observations"]
        rewards = step_resp["rewards"]
        info = step_resp["info"]
        truncated = bool(step_resp["truncated"])

        reward_series.append(float(np.mean(rewards)))
        fiedler_series.append(float(info.get("fiedler", 0.0)))
        alive_series.append(int(info.get("alive_count", 0)))
        connected_steps = int(info.get("connected_steps", connected_steps))
        total_steps += 1

        emit_structured(
            "STEP",
            {
                "agent": agent_name,
                "task_id": task_id,
                "seed": seed,
                "step": total_steps,
                "mean_reward": round(float(np.mean(rewards)), 6),
                "alive_count": int(info.get("alive_count", 0)),
                "fiedler": round(float(info.get("fiedler", 0.0)), 6),
                "truncated": truncated,
            },
        )

        if truncated:
            episode_score = float(info.get("episode_score") or 0.0)
            reward_score = float(info.get("reward_score") or 0.0)
            emit_structured(
                "END",
                {
                    "agent": agent_name,
                    "task_id": task_id,
                    "seed": seed,
                    "total_steps": total_steps,
                    "episode_score": round(episode_score, 6),
                    "reward_score": round(reward_score, 6),
                    "connected_steps": connected_steps,
                },
            )
            break

    return {
        "task_id": task_id,
        "task_name": TASK_NAMES[task_id],
        "seed": seed,
        "agent": agent_name,
        "episode_score": episode_score,
        "reward_score": reward_score,
        "mean_reward": float(np.mean(reward_series)) if reward_series else 0.0,
        "final_fiedler": fiedler_series[-1] if fiedler_series else 0.0,
        "final_alive": alive_series[-1] if alive_series else 0,
        "connected_steps": connected_steps,
        "total_steps": total_steps,
        "reward_series": reward_series,
        "fiedler_series": fiedler_series,
        "alive_series": alive_series,
    }


def summarise_scores(scores: list[float]) -> dict:
    if not scores:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    array = np.asarray(scores, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def main() -> None:
    args = parse_args()
    host = args.host.rstrip("/")
    tasks = [args.task] if args.task is not None else list(ALL_TASKS)
    agents = resolve_agents(args.agents)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    try:
        api_base_url, model_name, token = resolve_required_config(args)
    except RuntimeError as exc:
        emit_structured(
            "END",
            {
                "phase": "config",
                "status": "error",
                "error": str(exc),
            },
        )
        raise SystemExit(2) from exc

    # Required by submission instructions for any LLM-backed policy calls.
    _openai_client = build_openai_client(api_base_url, token)
    if _openai_client is None:
        raise RuntimeError("Failed to initialise OpenAI client.")

    emit_structured(
        "START",
        {
            "phase": "run",
            "host": host,
            "api_base_url": api_base_url,
            "model_name": model_name,
            "tasks": tasks,
            "agents": agents,
            "seeds": args.seeds,
        },
    )

    if not wait_for_server(host, token=token):
        emit_structured(
            "END",
            {
                "phase": "wait_for_server",
                "status": "error",
                "error": "Server not reachable",
            },
        )
        return

    summary_rows: list[dict] = []

    for agent_name in agents:
        for task_id in tasks:
            task_scores: list[float] = []

            for seed in range(args.seeds):
                result = run_episode(
                    host,
                    task_id,
                    seed,
                    agent_name,
                    token=token,
                    model_name=model_name,
                )
                task_scores.append(result["episode_score"])

                run_path = log_dir / f"task{task_id}_seed{seed}_{agent_name}.json"
                with run_path.open("w", encoding="utf-8") as handle:
                    json.dump(result, handle, indent=2)

            stats = summarise_scores(task_scores)
            summary_rows.append(
                {
                    "agent": agent_name,
                    "task_id": task_id,
                    "task_name": TASK_NAMES[task_id],
                    "mean_score": stats["mean"],
                    "std_score": stats["std"],
                    "min_score": stats["min"],
                    "max_score": stats["max"],
                }
            )

    summary_payload = {
        "host": host,
        "api_base_url": api_base_url,
        "model_name": model_name,
        "tasks": tasks,
        "agents": agents,
        "seeds": args.seeds,
        "rows": summary_rows,
    }
    summary_path = log_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    emit_structured(
        "END",
        {
            "phase": "run",
            "status": "ok",
            "summary_path": str(summary_path),
            "rows": summary_rows,
        },
    )


if __name__ == "__main__":
    main()
