---
title: GC-MARL Communication Resilience
author: GC-MARL Research
emoji: "🛰"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - network-resilience
---

# 1. Environment Overview and Motivation

This environment simulates a real-world communication infrastructure resilience problem: a city-scale IoT and edge communication network under outage and attack conditions. Each agent is a device or gateway operator that must make local decisions to keep emergency communication service available. The benchmark is designed for network infrastructure repair prioritization and incident response, not for game scoring.

Why this is a real operator task:
- Utilities, transport systems, hospitals, and emergency services depend on resilient communication backbones.
- During failures, operators must reroute traffic, reduce local processing load, and maintain connectivity despite partial compromise.
- Better policies directly map to better service continuity during incidents.

# 2. Observation and Action Space Definitions

Observation space:
- Type: Box
- Shape: [8]
- Dtype: float32
- Per-agent features:
  - degree_norm
  - energy
  - load
  - alive_ratio_khop
  - avg_link_quality
  - one-hot node type: gateway, mobile, constrained

Action space:
- Type: Discrete(4)
- Actions:
  - 0: hold/no-op
  - 1: reroute to lower-load neighbor
  - 2: reduce local processing (offload)
  - 3: broadcast connectivity beacon

Undesirable behavior penalties:
- Repeated no-op loop penalty: -0.5 when an agent executes action 0 for 3 or more consecutive steps.
- Destructive reroute penalty: -0.3 when reroute would push a neighbor above cascade threshold.
- Destructive beacon penalty: -0.3 when beacon is sent at very low energy.

Per-step API rewards are sigmoid-normalized into [0, 1], and an episode-level reward_score is produced by sigmoid(total_reward / N_NODES).

# 3. Task Descriptions and Difficulty

Task 0: Easy - Random Failure Recovery
- Config: N_BYZANTINE=0, FAILURE_RATE=0.02, REPAIR_RATE=0.20, CASCADE_THRESH=0.4
- Grader: final_lambda2 / initial_lambda2 (clamped to [0,1])
- Plain-English explanation:
  1. Lambda2 measures how well the remaining network stays tied together through alternate paths.
  2. The ratio compares end-of-episode resilience to the healthy starting condition.
  3. A high score means the operator policy preserved backbone connectivity after random failures.

Task 1: Medium - Cascading Failure Survival
- Config: N_BYZANTINE=0, FAILURE_RATE=0.04, REPAIR_RATE=0.15, CASCADE_THRESH=0.3
- Grader: ((alive_count / N_NODES) + (final_lambda2 / initial_lambda2)) / 2
- Plain-English explanation:
  1. Alive ratio measures service footprint: how many devices are still functioning.
  2. Lambda2 ratio measures structural quality of the remaining network.
  3. Averaging both terms gives partial credit when one signal degrades, preserving task sensitivity.

Task 2: Hard - Byzantine Adversaries
- Config: N_BYZANTINE=3, Byzantine injection at reset
- Grader: connected_steps / total_steps
- Plain-English explanation:
  1. This score tracks how often the alive network stayed fully connected over time.
  2. It captures sustained continuity, not just end-state luck.
  3. A high score means the system stayed operational through adversarial disruption.

Task 3: Combined Siege (Expert)
- Config: N_BYZANTINE=3, FAILURE_RATE=0.06, REPAIR_RATE=0.12, CASCADE_THRESH=0.32
- Grader: 0.4 * medium_score + 0.6 * hard_score
- Plain-English explanation:
  1. This task combines final-state survivability and sustained connectivity under attack.
  2. The medium component rewards remaining alive nodes and end-episode structural quality.
  3. The hard component rewards staying connected throughout the episode timeline.

# 4. Reward Design

The reward design is intentionally shaped around operational resilience instead of raw action count. Each step reward combines local service health, network connectivity, and safety penalties, then normalizes the result into a stable [0, 1] range so policies can be compared across tasks. The episode-level `reward_score` is also sigmoid-normalized from total accumulated reward, which keeps long episodes from dominating the scale and makes the metric easy to read during evaluation. In practice, this means the model is rewarded for keeping more nodes alive, preserving connectivity, and avoiding destructive actions rather than for maximizing one-step gains.

Standard RL environments for network control use binary connectivity rewards: the network is either connected or it is not. GC-MARL replaces this sparse signal with a dense, continuous penalty based on the Frobenius norm of the local Laplacian difference (||L_i,t - L_i,t-1||_F), computed per agent over the k-hop neighborhood. This penalty fires whenever local topology changes, even before global disconnection occurs, enabling agents to detect instability early. Agents therefore learn to maintain network health continuously rather than reacting only after connectivity is lost. This reward component, derived from spectral graph theory, is the core novel contribution of the GC-MARL framework.

# 5. Setup and Usage Instructions

Install dependencies:
```bash
pip install -r requirements.txt
```

Start API server:
```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Run smoke test:
```bash
python smoke_test.py --host http://127.0.0.1:8000 --task 2 --seed 7
```

Run baseline inference:
```bash
python inference.py --host http://127.0.0.1:8000 --log-dir logs_latest
```

Mandatory environment configuration:
- `API_BASE_URL`: API endpoint for OpenAI-compatible LLM inference.
- `MODEL_NAME`: Model identifier used for LLM inference calls.
- `HF_TOKEN`: Hugging Face/API key used for authenticated requests.

The `inference.py` script now validates these variables at startup and exits with code `2` if any are missing.

PowerShell example:
```bash
$env:API_BASE_URL="https://api.openai.com/v1"
$env:MODEL_NAME="gpt-4o-mini"
$env:HF_TOKEN="hf_xxx"
python inference.py --host https://<space-url> --log-dir logs_latest
```

Structured stdout logs (mandatory):
- `[START]` is emitted once per episode with fields in order:
  `agent`, `task_id`, `task_name`, `seed`, `model_name`
- `[STEP]` is emitted every environment step with fields in order:
  `agent`, `task_id`, `seed`, `step`, `mean_reward`, `alive_count`, `fiedler`, `truncated`
- `[END]` is emitted once per episode with fields in order:
  `agent`, `task_id`, `seed`, `total_steps`, `episode_score`, `reward_score`, `connected_steps`

Example:
```text
[START] {"agent":"random","task_id":0,"task_name":"Easy","seed":0,"model_name":"gpt-4o-mini"}
[STEP] {"agent":"random","task_id":0,"seed":0,"step":1,"mean_reward":0.713845,"alive_count":29,"fiedler":0.705499,"truncated":false}
[END] {"agent":"random","task_id":0,"seed":0,"total_steps":25,"episode_score":0.354373,"reward_score":0.999957,"connected_steps":15}
```

OpenEnv validator command (spec requirement):
```bash
python -m openenv.cli validate .
```

Validation output and endpoint checks are listed in Section 6.

Hugging Face Space deployment checklist:
- This README frontmatter already includes tag openenv.
- Create a Docker Space and push this repository.
- Verify Space metadata shows tag openenv before submission.
- Ensure the deployed Space health endpoint responds at /health.

Docker:
```bash
docker build -t gc-marl-openenv .
docker run --rm -p 8000:8000 gc-marl-openenv
```

# 6. Validation

OpenEnv validation (final pre-submit check):
```bash
python -m openenv.cli validate .
```

Exact output:
```text
[OK] : Ready for multi-mode deployment
```

Live Space endpoint checks:
- `GET /health` -> 200
- `POST /reset` -> 200
- `GET /state` -> 200

# 7. Baseline Performance Scores

Run metadata:
- Date: 2026-04-09
- Command: python inference.py --host http://127.0.0.1:8000 --seeds 5 --agents all --log-dir logs_p1_policyfix_20260409
- Seeds: 5
- Agents: random, greedy, heuristic
- Source: logs_p1_policyfix_20260409/summary.json

| Agent | Task | Mean | Std | Min | Max |
|---|---|---:|---:|---:|---:|
| random | Easy | 0.9583 | 0.0381 | 0.9004 | 1.0000 |
| random | Medium | 0.1945 | 0.0896 | 0.1333 | 0.3723 |
| random | Hard (Byzantine) | 0.8160 | 0.3680 | 0.0800 | 1.0000 |
| greedy | Easy | 0.7583 | 0.3805 | 0.0000 | 1.0000 |
| greedy | Medium | 0.7544 | 0.1275 | 0.5525 | 0.9008 |
| greedy | Hard (Byzantine) | 0.8080 | 0.3454 | 0.1200 | 1.0000 |
| heuristic | Easy | 0.7583 | 0.3805 | 0.0000 | 1.0000 |
| heuristic | Medium | 0.6536 | 0.1499 | 0.4651 | 0.8829 |
| heuristic | Hard (Byzantine) | 0.8080 | 0.3454 | 0.1200 | 1.0000 |

Note: After the P1 policy update, greedy and heuristic avoid low-energy beacon misuse and
reduce cascade-triggering reroutes. This substantially improves Medium-task performance for
non-learning baselines while keeping all scores within the required [0,1] range.

Interpretation:
- Easy remains high across policies, with random currently strongest on this fixed-seed benchmark.
- Medium now shows a clear spread and no near-zero collapse after policy and grader fixes.
- Hard remains adversarial and variable, with all baselines still bounded in [0,1].

Task 3 benchmark (Combined Siege):
- Date: 2026-04-09
- Command: python inference.py --host http://127.0.0.1:8000 --task 3 --seeds 5 --agents all --log-dir logs_p3_task3
- Seeds: 5
- Agents: random, greedy, heuristic
- Source: logs_p3_task3/summary.json

| Agent | Task | Mean | Std | Min | Max |
|---|---|---:|---:|---:|---:|
| random | Combined Siege | 0.3929 | 0.2218 | 0.1773 | 0.7216 |
| greedy | Combined Siege | 0.8601 | 0.1037 | 0.6864 | 1.0000 |
| heuristic | Combined Siege | 0.6892 | 0.1885 | 0.4107 | 0.9040 |
