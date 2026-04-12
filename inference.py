"""
inference.py — Evaluation harness for OpenEnv GC-MARL tasks.

Default behavior:
    - 4 tasks (0/1/2/3)
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
import sys
import time
import traceback
from pathlib import Path
from typing import Callable
import urllib.error
import urllib.request

import numpy as np
from openai import OpenAI

N_AGENTS = 30
TASK_NAMES = {
    0: "Easy",
    1: "Medium",
    2: "Hard (Byzantine)",
    3: "Combined Siege",
}
ALL_TASKS = [0, 1, 2, 3]
ALL_AGENTS = ["random", "greedy", "heuristic"]
SCORE_EPSILON = 1e-6


def clip_open_unit_interval(value: float) -> float:
    return float(min(1.0 - SCORE_EPSILON, max(SCORE_EPSILON, value)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument(
        "--api-key",
        default=None,
        help="LLM API key. Default uses API_KEY or OPENAI_API_KEY env vars.",
    )
    parser.add_argument(
        "--api-base-url",
        default=None,
        help="LLM API base URL. Default uses API_BASE_URL env var.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="LLM model identifier. Default uses MODEL_NAME env var, then gpt-4o-mini.",
    )
    parser.add_argument(
        "--task",
        type=int,
        choices=ALL_TASKS,
        default=None,
        help="Run only one task (0/1/2/3). Default: run all tasks.",
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
        help="Legacy alias for API key/token. Prefer --api-key and API_KEY.",
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
    api_key = (
        args.api_key
        or os.environ.get("API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or args.hf_token
        or os.environ.get("HF_TOKEN")
        or ""
    ).strip()

    if not model_name:
        model_name = "gpt-4o-mini"

    missing: list[str] = []
    if not api_base_url:
        missing.append("API_BASE_URL")
    if not api_key:
        missing.append("API_KEY")

    if missing:
        raise RuntimeError(
            "Missing required environment configuration: "
            + ", ".join(missing)
            + ". Define API_BASE_URL and one of API_KEY / OPENAI_API_KEY / HF_TOKEN before running inference. "
            + "MODEL_NAME is optional and defaults to gpt-4o-mini."
        )

    return api_base_url, model_name, api_key


def build_openai_client(api_base_url: str, api_key: str) -> OpenAI:
    try:
        return OpenAI(base_url=api_base_url, api_key=api_key)
    except Exception as e:
        emit_structured(
            "END",
            {
                "phase": "openai_client_init",
                "status": "error",
                "error": f"Failed to build OpenAI client: {str(e)}",
            },
        )
        raise RuntimeError(f"OpenAI client initialization failed: {e}") from e


def verify_llm_proxy_call(client: OpenAI, model_name: str) -> bool:
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": "Reply with a single token: OK",
                }
            ],
            max_tokens=1,
            temperature=0,
            timeout=20,
        )
        emit_structured(
            "STEP",
            {
                "phase": "llm_proxy_check",
                "status": "success",
                "model_name": model_name,
                "response_id": getattr(response, "id", None),
            },
        )
        return True
    except Exception as e:
        emit_structured(
            "STEP",
            {
                "phase": "llm_proxy_check",
                "status": "warning",
                "error": str(e),
            },
        )
        print(
            "[WARNING] LLM proxy preflight failed; continuing episode execution. "
            "Ensure API_BASE_URL / MODEL_NAME / HF_TOKEN are the injected validator values.",
            file=sys.stderr,
            flush=True,
        )
        return False


def emit_structured(tag: str, fields: dict) -> None:
    try:
        payload = json.dumps(fields, separators=(",", ":"), ensure_ascii=True)
        print(f"[{tag}] {payload}", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to emit structured log: {e}", file=sys.stderr, flush=True)


def _auth_headers(token: str | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def post_json(host: str, path: str, body: dict, token: str | None = None) -> dict:
    try:
        payload = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json", **_auth_headers(token)}
        request = urllib.request.Request(
            url=f"{host}{path}",
            data=payload,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8', errors='replace') if e.fp else "No error body"
        raise RuntimeError(
            f"HTTP {e.code} error for POST {host}{path}: {e.reason}. Body: {error_body}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"URL error for POST {host}{path}: {e.reason}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON response from POST {host}{path}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error in POST {host}{path}: {e}") from e


def get_json(host: str, path: str, timeout: int = 10, token: str | None = None) -> dict:
    try:
        request = urllib.request.Request(
            url=f"{host}{path}",
            headers=_auth_headers(token),
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8', errors='replace') if e.fp else "No error body"
        raise RuntimeError(
            f"HTTP {e.code} error for GET {host}{path}: {e.reason}. Body: {error_body}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"URL error for GET {host}{path}: {e.reason}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON response from GET {host}{path}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error in GET {host}{path}: {e}") from e


def random_agent(obs: list[list[float]]) -> list[int]:
    del obs
    return [random.randint(0, 3) for _ in range(N_AGENTS)]


def _greedy_action_score(row: list[float], action: int) -> float:
    """Proxy objective: maximise alive-neighbour potential minus load increase risk."""
    try:
        energy = float(row[1])
        load = float(row[2])
        alive_ratio = float(row[3])
    except (IndexError, ValueError, TypeError) as e:
        # Fallback for malformed observation
        print(f"[WARNING] Invalid observation row: {row}, error: {e}", file=sys.stderr, flush=True)
        return 0.0

    alive_gain = 0.0
    load_increase = 0.0

    if action == 0:  # hold/no-op
        alive_gain = 0.05 * alive_ratio
        load_increase = max(0.0, load - 0.25) * 0.6
    elif action == 1:  # reroute
        if 0.30 < load < 0.55 and energy > 0.20:
            alive_gain = 0.30
            load_increase = max(0.0, load - 0.35) * 0.55
        else:
            # Penalise aggressive reroutes that are likely to trigger cascades.
            alive_gain = 0.05
            load_increase = max(0.0, load - 0.30) * 0.95 + (0.2 if load > 0.6 else 0.0)
    elif action == 2:  # reduce local processing
        alive_gain = 0.20 + 0.25 * load
        load_increase = -0.35 * load
    else:  # action == 3, beacon
        if energy < 0.20:
            alive_gain = -0.40
            load_increase = 0.05
        else:
            alive_gain = 0.30 if alive_ratio < 0.50 else 0.10
            load_increase = 0.02

    return alive_gain - load_increase + 0.05 * alive_ratio


def greedy_agent(obs: list[list[float]]) -> list[int]:
    try:
        actions: list[int] = []
        for row in obs:
            scores = [_greedy_action_score(row, action) for action in range(4)]
            best_action = int(np.argmax(scores))
            actions.append(best_action)
        return actions
    except Exception as e:
        print(f"[WARNING] Greedy agent error: {e}, using random fallback", file=sys.stderr, flush=True)
        return random_agent(obs)


def heuristic_agent(obs: list[list[float]]) -> list[int]:
    try:
        actions: list[int] = []
        for row in obs:
            energy = row[1]
            load = row[2]
            alive_ratio = row[3]

            # Never beacon below 0.2 energy to avoid low-energy misuse penalties.
            if energy < 0.20:
                if load > 0.25:
                    actions.append(2)
                else:
                    actions.append(0)
            elif load > 0.45:
                actions.append(2)
            elif load > 0.30 and alive_ratio >= 0.50:
                actions.append(1)
            elif alive_ratio < 0.50:
                actions.append(3)
            elif load > 0.20:
                actions.append(2)
            else:
                actions.append(0)
        return actions
    except Exception as e:
        print(f"[WARNING] Heuristic agent error: {e}, using random fallback", file=sys.stderr, flush=True)
        return random_agent(obs)


AGENT_FNS: dict[str, Callable[[list[list[float]]], list[int]]] = {
    "random": random_agent,
    "greedy": greedy_agent,
    "heuristic": heuristic_agent,
}


def wait_for_server(host: str, token: str | None, retries: int = 15, delay_seconds: int = 2) -> bool:
    for attempt in range(1, retries + 1):
        try:
            get_json(host, "/health", token=token)
            emit_structured(
                "STEP",
                {
                    "phase": "wait_for_server",
                    "attempt": attempt,
                    "status": "success",
                },
            )
            return True
        except Exception as e:
            emit_structured(
                "STEP",
                {
                    "phase": "wait_for_server",
                    "attempt": attempt,
                    "status": "retry",
                    "error": str(e),
                },
            )
            if attempt < retries:
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
    try:
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
        obs = reset_resp.get("observations", [])
        
        if not obs:
            raise ValueError("Empty observations from /reset endpoint")

        reward_series: list[float] = []
        fiedler_series: list[float] = []
        alive_series: list[int] = []
        connected_steps = 0
        total_steps = 0
        episode_score = SCORE_EPSILON
        reward_score = SCORE_EPSILON
        
        max_steps = 1000  # Safety limit to prevent infinite loops

        while total_steps < max_steps:
            actions = agent_fn(obs)
            
            if len(actions) != N_AGENTS:
                raise ValueError(f"Agent returned {len(actions)} actions, expected {N_AGENTS}")
            
            step_resp = post_json(host, "/step", {"actions": actions}, token=token)

            obs = step_resp.get("observations", [])
            rewards = step_resp.get("rewards", [])
            info = step_resp.get("info", {})
            truncated = bool(step_resp.get("truncated", False))

            reward_series.append(float(np.mean(rewards)) if rewards else 0.0)
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
                    "mean_reward": round(float(np.mean(rewards)) if rewards else 0.0, 6),
                    "alive_count": int(info.get("alive_count", 0)),
                    "fiedler": round(float(info.get("fiedler", 0.0)), 6),
                    "truncated": truncated,
                },
            )

            if truncated:
                raw_episode_score = float(info.get("episode_score") or SCORE_EPSILON)
                raw_reward_score = float(info.get("reward_score") or SCORE_EPSILON)
                episode_score = clip_open_unit_interval(raw_episode_score)
                reward_score = clip_open_unit_interval(raw_reward_score)
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
        
        if total_steps >= max_steps:
            print(f"[WARNING] Episode reached max steps ({max_steps})", file=sys.stderr, flush=True)

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
    except Exception as e:
        emit_structured(
            "END",
            {
                "agent": agent_name,
                "task_id": task_id,
                "seed": seed,
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def summarise_scores(scores: list[float]) -> dict:
    if not scores:
        return {
            "mean": SCORE_EPSILON,
            "std": 0.0,
            "min": SCORE_EPSILON,
            "max": SCORE_EPSILON,
        }
    array = np.asarray([clip_open_unit_interval(score) for score in scores], dtype=np.float64)
    return {
        "mean": clip_open_unit_interval(float(np.mean(array))),
        "std": float(np.std(array)),
        "min": clip_open_unit_interval(float(np.min(array))),
        "max": clip_open_unit_interval(float(np.max(array))),
    }


def main() -> None:
    try:
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
            print(f"[ERROR] Configuration error: {exc}", file=sys.stderr, flush=True)
            sys.exit(2)

        # Required by submission instructions for any LLM-backed policy calls.
        _openai_client = build_openai_client(api_base_url, token)
        if _openai_client is None:
            raise RuntimeError("Failed to initialise OpenAI client.")

        # Attempt a proxy preflight request, but do not crash if it fails.
        preflight_ok = verify_llm_proxy_call(_openai_client, model_name)
        if not preflight_ok:
            emit_structured(
                "STEP",
                {
                    "phase": "llm_proxy_check",
                    "status": "continuing_without_preflight",
                },
            )

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
                    "error": "Server not reachable after retries",
                },
            )
            print(f"[ERROR] Server at {host} not reachable", file=sys.stderr, flush=True)
            sys.exit(1)

        summary_rows: list[dict] = []

        for agent_name in agents:
            for task_id in tasks:
                task_scores: list[float] = []

                for seed in range(args.seeds):
                    try:
                        result = run_episode(
                            host,
                            task_id,
                            seed,
                            agent_name,
                            token=token,
                            model_name=model_name,
                        )
                        result["episode_score"] = clip_open_unit_interval(
                            float(result["episode_score"])
                        )
                        task_scores.append(result["episode_score"])

                        run_path = log_dir / f"task{task_id}_seed{seed}_{agent_name}.json"
                        with run_path.open("w", encoding="utf-8") as handle:
                            json.dump(result, handle, indent=2)
                    except Exception as e:
                        print(
                            f"[ERROR] Episode failed for agent={agent_name}, task={task_id}, seed={seed}: {e}",
                            file=sys.stderr,
                            flush=True,
                        )
                        traceback.print_exc(file=sys.stderr)
                        # Continue with next episode instead of crashing

                if task_scores:
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
        
        sys.exit(0)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user", file=sys.stderr, flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"[FATAL] Unhandled exception in main: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        emit_structured(
            "END",
            {
                "phase": "main",
                "status": "fatal_error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
