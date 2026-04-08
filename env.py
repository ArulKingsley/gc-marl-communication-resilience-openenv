"""
env.py — OpenEnv wrapper around your existing IoTNetworkEnv.

Changes vs. your notebook code:
  1. reset() accepts task_id (0/1/2) and seeds the RNG.
  2. CFG is dynamically patched per task — no duplicate class needed.
  3. Byzantine injection happens inside reset() for task_id=2.
  4. Per-step rewards are sigmoid-normalised for API responses.
      Episode-end cumulative team reward is also normalised to [0,1].
  5. info dict always carries fiedler_value, alive_count, connected_steps.

Everything else (IoTTopology, FailureEngine, IoTNetworkEnv) is your
original notebook code, copied verbatim below so this file is self-contained.
"""

from __future__ import annotations
import numpy as np
import networkx as nx
import random, warnings
warnings.filterwarnings("ignore")

import gymnasium as gym
from gymnasium import spaces

from models import (
    sigmoid_normalise,
    grade_easy, grade_medium, grade_hard,
    OBS_DIM, N_ACTIONS,
)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Base config (task 0 defaults) ─────────────────────────────────────────────
BASE_CFG = dict(
    N_NODES        = 30,
    N_GATEWAYS     = 3,
    K_HOP          = 2,
    EDGE_PROB      = 0.25,
    FAILURE_RATE   = 0.05,
    REPAIR_RATE    = 0.10,
    CASCADE_THRESH = 0.4,
    N_EPISODES     = 150,
    MAX_STEPS      = 25,
    GAMMA          = 0.99,
    LR_ACTOR       = 3e-4,
    LR_CRITIC      = 1e-3,
    CLIP_EPS       = 0.2,
    PPO_EPOCHS     = 4,
    BATCH_SIZE     = 64,
    LAMBDA_STAB    = 0.5,
    BETA_CONSENSUS = 0.3,
    GAMMA_DECAY    = 0.1,
    N_BYZANTINE    = 0,
    TRUST_INIT     = 1.0,
)

# ── Per-task config overrides ─────────────────────────────────────────────────
TASK_OVERRIDES = {
    0: dict(N_BYZANTINE=0, FAILURE_RATE=0.02, REPAIR_RATE=0.20, CASCADE_THRESH=0.4),
    1: dict(N_BYZANTINE=0, FAILURE_RATE=0.04, REPAIR_RATE=0.15, CASCADE_THRESH=0.3),
    2: dict(N_BYZANTINE=3, FAILURE_RATE=0.05, CASCADE_THRESH=0.4),
}


# ═══════════════════════════════════════════════════════════════════════════════
# IoTTopology  (your original class, verbatim)
# ═══════════════════════════════════════════════════════════════════════════════
class IoTTopology:
    def __init__(self, cfg=BASE_CFG, seed=SEED):
        self.cfg  = cfg
        self.n    = cfg["N_NODES"]
        self.seed = seed
        self.graph = None
        self.pos   = None
        self._build()

    def _build(self):
        g = nx.erdos_renyi_graph(self.n, self.cfg["EDGE_PROB"], seed=self.seed)
        while not nx.is_connected(g):
            g = nx.erdos_renyi_graph(
                self.n, self.cfg["EDGE_PROB"] + 0.02, seed=self.seed + 1
            )

        gateway_ids = random.sample(range(self.n), self.cfg["N_GATEWAYS"])
        mobile_ids  = random.sample(
            [i for i in range(self.n) if i not in gateway_ids], 4
        )

        for node in g.nodes():
            if node in gateway_ids:
                ntype, energy = "gateway", 1.0
            elif node in mobile_ids:
                ntype  = "mobile"
                energy = np.random.uniform(0.3, 0.7)
            else:
                ntype  = "constrained"
                energy = np.random.uniform(0.2, 0.9)

            g.nodes[node].update(
                dict(type=ntype, energy=energy,
                     load=np.random.uniform(0.1, 0.4),
                     alive=True, byzantine=False)
            )

        for u, v in g.edges():
            g[u][v]["weight"]  = np.random.uniform(0.5, 1.0)
            g[u][v]["latency"] = np.random.uniform(1, 10)

        self.graph = g
        self.pos   = nx.spring_layout(g, seed=self.seed)

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        self._build()

    def fiedler_value(self) -> float:
        alive = [n for n, d in self.graph.nodes(data=True) if d["alive"]]
        sub   = self.graph.subgraph(alive)
        if not nx.is_connected(sub) or len(alive) < 2:
            return 0.0
        L  = nx.laplacian_matrix(sub).toarray().astype(float)
        ev = np.sort(np.linalg.eigvalsh(L))
        return float(ev[1]) if len(ev) > 1 else 0.0

    def local_laplacian(self, node, k=2):
        nbrs = {node}
        frontier = {node}
        for _ in range(k):
            new_frontier = set()
            for n in frontier:
                new_frontier |= set(self.graph.neighbors(n))
            nbrs |= new_frontier
            frontier = new_frontier
        alive_nbrs = [n for n in nbrs if self.graph.nodes[n]["alive"]]
        sub = self.graph.subgraph(alive_nbrs)
        return (
            nx.laplacian_matrix(sub).toarray().astype(float)
            if len(alive_nbrs) > 1
            else np.zeros((1, 1))
        )

    def is_alive_subgraph_connected(self) -> bool:
        alive = [n for n, d in self.graph.nodes(data=True) if d["alive"]]
        if len(alive) < 2:
            return False
        return nx.is_connected(self.graph.subgraph(alive))


# ═══════════════════════════════════════════════════════════════════════════════
# FailureEngine  (your original class, verbatim)
# ═══════════════════════════════════════════════════════════════════════════════
class FailureEngine:
    def __init__(self, topo: IoTTopology, cfg=BASE_CFG):
        self.topo = topo
        self.cfg  = cfg

    def step(self):
        g = self.topo.graph
        failed, repaired = [], []
        for node in list(g.nodes()):
            d = g.nodes[node]
            if d["type"] == "gateway":
                continue
            if d["alive"]:
                if np.random.rand() < self.cfg["FAILURE_RATE"]:
                    d["alive"] = False
                    failed.append(node)
                    self._redistribute_load(node)
                elif d["load"] > self.cfg["CASCADE_THRESH"]:
                    d["alive"] = False
                    failed.append(node)
                    self._redistribute_load(node)
            else:
                if np.random.rand() < self.cfg["REPAIR_RATE"]:
                    d["alive"] = True
                    d["load"]  = np.random.uniform(0.1, 0.3)
                    repaired.append(node)
        return failed, repaired

    def _redistribute_load(self, failed_node):
        g    = self.topo.graph
        load = g.nodes[failed_node]["load"]
        alive_nbrs = [
            n for n in g.neighbors(failed_node) if g.nodes[n]["alive"]
        ]
        if alive_nbrs:
            share = load / len(alive_nbrs)
            for nb in alive_nbrs:
                g.nodes[nb]["load"] = min(1.0, g.nodes[nb]["load"] + share)

    def inject_byzantine(self, n_byzantine=None):
        n = n_byzantine or self.cfg["N_BYZANTINE"]
        candidates = [
            nd for nd, d in self.topo.graph.nodes(data=True)
            if d["type"] == "constrained" and d["alive"]
        ]
        chosen = random.sample(candidates, min(n, len(candidates)))
        for nd in chosen:
            self.topo.graph.nodes[nd]["byzantine"] = True
        return chosen


# ═══════════════════════════════════════════════════════════════════════════════
# IoTNetworkEnv  (your original class + OpenEnv additions)
# ═══════════════════════════════════════════════════════════════════════════════
class IoTNetworkEnv(gym.Env):
    """
    OpenEnv-wrapped MARL environment.

    New vs. notebook original:
      • reset(task_id, seed) patches CFG per task and injects Byzantine nodes.
            • _normalise_step_rewards() applies sigmoid to keep per-step rewards in [0,1].
            • repeated no-op loops and destructive actions incur explicit penalties.
      • info dict carries fiedler, alive_count, connected_steps, episode_score.
    """

    metadata = {"render_modes": []}

    def __init__(self, task_id: int = 0, gc_marl: bool = False):
        super().__init__()
        self.task_id = task_id
        self.gc_marl = gc_marl

        self.cfg = {**BASE_CFG, **TASK_OVERRIDES[task_id]}

        self.topo    = IoTTopology(self.cfg)
        self.engine  = FailureEngine(self.topo, self.cfg)

        self.n_agents  = self.cfg["N_NODES"]
        self.obs_dim   = OBS_DIM
        self.n_actions = N_ACTIONS

        self.observation_space = spaces.Box(
            -1.0, 2.0, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_actions)

        self._prev_laplacians : dict = {}
        self._step_count      : int  = 0
        self._ep_raw_rewards  : np.ndarray = np.zeros(self.n_agents)
        self._fiedler_initial : float = 0.0
        self._connected_steps : int   = 0
        self._byzantine_ids   : list  = []
        self._episode_score   : float | None = None
        self._reward_score    : float | None = None
        self._noop_streak     : np.ndarray = np.zeros(self.n_agents, dtype=np.int32)

    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self, task_id: int | None = None, seed: int | None = None,
              options: dict | None = None):
        if task_id is not None:
            if task_id not in TASK_OVERRIDES:
                raise ValueError(f"Unsupported task_id={task_id}. Expected one of {list(TASK_OVERRIDES)}")
            self.task_id = task_id
            self.cfg = {**BASE_CFG, **TASK_OVERRIDES[task_id]}
            self.engine.cfg = self.cfg

        rng_seed = seed if seed is not None else SEED
        random.seed(rng_seed)
        np.random.seed(rng_seed)

        self.topo.reset(seed=rng_seed)
        self.engine = FailureEngine(self.topo, self.cfg)

        # Inject Byzantine nodes for task 2
        self._byzantine_ids = []
        if self.cfg["N_BYZANTINE"] > 0:
            self._byzantine_ids = self.engine.inject_byzantine()

        self._prev_laplacians = {
            nd: self.topo.local_laplacian(nd, self.cfg["K_HOP"])
            for nd in self.topo.graph.nodes()
        }
        self._step_count      = 0
        self._ep_raw_rewards  = np.zeros(self.n_agents)
        self._fiedler_initial = self.topo.fiedler_value()
        self._connected_steps = 0
        self._episode_score   = None
        self._reward_score    = None
        self._noop_streak     = np.zeros(self.n_agents, dtype=np.int32)

        obs = self._get_obs()
        info = {
            "task_id"       : self.task_id,
            "fiedler"       : self._fiedler_initial,
            "alive_count"   : self._alive_count(),
            "byzantine_ids" : self._byzantine_ids,
        }
        return obs, info

    # ── step ──────────────────────────────────────────────────────────────────
    def step(self, actions):
        actions_arr = np.asarray(actions, dtype=np.int32)
        if len(actions_arr) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} actions, got {len(actions_arr)}")
        if np.any(actions_arr < 0) or np.any(actions_arr >= self.n_actions):
            raise ValueError(f"Actions must be in [0, {self.n_actions - 1}]")

        g       = self.topo.graph
        rewards = np.zeros(self.n_agents)
        behaviour_penalties = np.zeros(self.n_agents)

        # Apply actions
        for agent_id, action in enumerate(actions_arr):
            if action == 0:
                self._noop_streak[agent_id] += 1
            else:
                self._noop_streak[agent_id] = 0

            if not g.nodes[agent_id]["alive"]:
                continue
            nd = g.nodes[agent_id]

            if action == 1:  # Reroute to lowest-load neighbour
                alive_nbrs = [n for n in g.neighbors(agent_id) if g.nodes[n]["alive"]]
                if alive_nbrs:
                    target   = min(alive_nbrs, key=lambda n: g.nodes[n]["load"])
                    overflow = max(0, nd["load"] - 0.3)
                    projected_target_load = g.nodes[target]["load"] + overflow * 0.5
                    if projected_target_load > self.cfg["CASCADE_THRESH"]:
                        behaviour_penalties[agent_id] -= 0.3
                    g.nodes[target]["load"] = min(1.0, projected_target_load)
                    nd["load"]   = max(0.1, nd["load"] - overflow * 0.5)
                    nd["energy"] = max(0.0, nd["energy"] - 0.02)

            elif action == 2:  # Reduce local processing (offload)
                nd["load"]   = max(0.05, nd["load"] * 0.8)
                nd["energy"] = max(0.0,  nd["energy"] - 0.01)

            elif action == 3:  # Broadcast connectivity beacon
                if nd["energy"] < 0.12:
                    behaviour_penalties[agent_id] -= 0.3
                nd["energy"] = max(0.0, nd["energy"] - 0.03)

        # Failure dynamics
        self.engine.step()

        # Base rewards
        for agent_id in range(self.n_agents):
            nd = g.nodes[agent_id]
            if nd["alive"]:
                if nd["load"] < self.cfg["CASCADE_THRESH"]:
                    rewards[agent_id] += 1.0
                failed_nbrs = sum(
                    1 for n in g.neighbors(agent_id)
                    if not g.nodes[n]["alive"]
                )
                rewards[agent_id] -= 0.2 * failed_nbrs
            else:
                rewards[agent_id] -= 0.5

            # Explicit anti-loop penalty for repeated no-op behavior.
            if self._noop_streak[agent_id] >= 3:
                behaviour_penalties[agent_id] -= 0.5

        rewards += behaviour_penalties

        # GC-MARL Frobenius stability penalty
        if self.gc_marl:
            curr_laps = {
                nd: self.topo.local_laplacian(nd, self.cfg["K_HOP"])
                for nd in g.nodes()
            }
            for agent_id in range(self.n_agents):
                if g.nodes[agent_id]["alive"]:
                    prev = self._prev_laplacians.get(agent_id,
                           np.zeros((1, 1)))
                    curr = curr_laps[agent_id]
                    min_sz = min(prev.shape[0], curr.shape[0])
                    frob   = np.linalg.norm(
                        curr[:min_sz, :min_sz] - prev[:min_sz, :min_sz], "fro"
                    )
                    rewards[agent_id] -= self.cfg["LAMBDA_STAB"] * frob
            self._prev_laplacians = curr_laps

        # Track connectivity for hard-task grader
        if self.topo.is_alive_subgraph_connected():
            self._connected_steps += 1

        self._ep_raw_rewards += rewards
        self._step_count     += 1

        truncated = self._step_count >= self.cfg["MAX_STEPS"]
        terminated = [False] * self.n_agents

        fiedler_now  = self.topo.fiedler_value()
        alive_count  = self._alive_count()
        episode_score = None
        reward_score = None

        if truncated:
            episode_score = self._compute_episode_score(
                fiedler_now, alive_count
            )
            reward_score = self._compute_reward_score()
            self._episode_score = episode_score
            self._reward_score = reward_score

        info = {
            "fiedler"        : fiedler_now,
            "alive_count"    : alive_count,
            "connected_steps": self._connected_steps,
            "step"           : self._step_count,
            "raw_reward_mean": float(np.mean(rewards)),
            "loop_penalty_count": int(np.sum(self._noop_streak >= 3)),
            "episode_score"  : episode_score,
            "reward_score"   : reward_score,
        }

        # Normalise per-step rewards to [0,1]
        norm_rewards = self._normalise_step_rewards(rewards)

        return self._get_obs(), norm_rewards.tolist(), terminated, truncated, info

    # ── helpers ───────────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        g   = self.topo.graph
        obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        max_deg = max(dict(g.degree()).values()) or 1

        for agent_id in range(self.n_agents):
            nd    = g.nodes[agent_id]
            alive_nbrs = [n for n in g.neighbors(agent_id) if g.nodes[n]["alive"]]
            k_hop_nodes = list(nx.single_source_shortest_path_length(
                g, agent_id, cutoff=self.cfg["K_HOP"]
            ).keys())
            alive_in_khop = sum(1 for n in k_hop_nodes if g.nodes[n]["alive"])
            alive_ratio   = alive_in_khop / max(len(k_hop_nodes), 1)

            link_qualities = [
                g[agent_id][nb].get("weight", 0.5)
                for nb in g.neighbors(agent_id)
            ]
            avg_lq = float(np.mean(link_qualities)) if link_qualities else 0.0

            type_onehot = [
                float(nd["type"] == "gateway"),
                float(nd["type"] == "mobile"),
                float(nd["type"] == "constrained"),
            ]

            obs[agent_id] = [
                g.degree(agent_id) / max_deg,   # degree_norm
                nd["energy"],                    # energy
                nd["load"],                      # load
                alive_ratio,                     # alive_ratio_khop
                avg_lq,                          # avg_link_quality
            ] + type_onehot

        return obs

    def _alive_count(self) -> int:
        return sum(1 for _, d in self.topo.graph.nodes(data=True) if d["alive"])

    def _normalise_step_rewards(self, rewards: np.ndarray) -> np.ndarray:
        """Squash per-step rewards to (0,1) via sigmoid."""
        return 1.0 / (1.0 + np.exp(-rewards))

    def _compute_episode_score(self, fiedler_now: float, alive_count: int) -> float:
        if self.task_id == 0:
            return grade_easy(fiedler_now, self._fiedler_initial)
        elif self.task_id == 1:
            return grade_medium(
                alive_count, self.n_agents, fiedler_now, self._fiedler_initial
            )
        else:
            return grade_hard(self._connected_steps, self._step_count)

    def _compute_reward_score(self) -> float:
        team_total_reward = float(np.sum(self._ep_raw_rewards))
        return sigmoid_normalise(team_total_reward, self.n_agents)

    # ── state snapshot (used by GET /state) ───────────────────────────────────
    def get_state_snapshot(self) -> dict:
        g = self.topo.graph
        nodes = []
        for nd_id, d in g.nodes(data=True):
            nodes.append({
                "node_id"  : nd_id,
                "alive"    : d["alive"],
                "byzantine": d["byzantine"],
                "energy"   : round(float(d["energy"]), 4),
                "load"     : round(float(d["load"]), 4),
                "node_type": d["type"],
            })
        return {
            "step"          : self._step_count,
            "task_id"       : self.task_id,
            "fiedler_value" : round(self.topo.fiedler_value(), 6),
            "alive_count"   : self._alive_count(),
            "total_nodes"   : self.n_agents,
            "nodes"         : nodes,
            "episode_score" : self._episode_score,
            "reward_score"  : self._reward_score,
        }
