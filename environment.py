"""
CloudHealRL Environment
=======================
Gymnasium-compatible RL environment for autonomous cloud cluster healing.

Services topology:
    [Auth] --> [API Gateway] --> [Payment] --> [Database]
                             --> [Notification]
"""

import gymnasium as gym
import numpy as np
import json
import random
from gymnasium import spaces
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

SERVICES = ["auth", "api_gateway", "payment", "database", "notification"]

DEPENDENCIES = {
    "database":     [],
    "auth":         [],
    "api_gateway":  ["auth"],
    "payment":      ["api_gateway", "database"],
    "notification": ["api_gateway"],
}

STATUS_HEALTHY   = 0
STATUS_DEGRADED  = 1
STATUS_CRASHED   = 2

FAILURE_NONE          = 0
FAILURE_CPU_SPIKE     = 1
FAILURE_MEMORY_LEAK   = 2
FAILURE_BAD_DEPLOY    = 3
FAILURE_NETWORK_SPLIT = 4
FAILURE_HARD_CRASH    = 5

FAILURE_NAMES = {
    FAILURE_NONE: "none", FAILURE_CPU_SPIKE: "cpu_spike",
    FAILURE_MEMORY_LEAK: "memory_leak", FAILURE_BAD_DEPLOY: "bad_deploy",
    FAILURE_NETWORK_SPLIT: "network_split", FAILURE_HARD_CRASH: "hard_crash",
}

ACTION_DO_NOTHING = 0
N_ACTIONS = 21

ACTION_NAMES = {0: "do_nothing"}
for _i, _svc in enumerate(SERVICES):
    ACTION_NAMES[1  + _i] = f"scale_up_{_svc}"
    ACTION_NAMES[6  + _i] = f"restart_{_svc}"
    ACTION_NAMES[11 + _i] = f"rollback_{_svc}"
    ACTION_NAMES[16 + _i] = f"reroute_{_svc}"


@dataclass
class Service:
    name: str
    cpu: float          = 20.0
    memory: float       = 30.0
    error_rate: float   = 0.0
    latency: float      = 50.0
    status: int         = STATUS_HEALTHY
    active_failure: int = FAILURE_NONE
    heal_cooldown: int  = 0

    def to_obs(self) -> List[float]:
        return [
            self.cpu / 100.0,
            self.memory / 100.0,
            float(self.error_rate),
            min(self.latency, 2000.0) / 2000.0,
            self.status / 2.0,
            self.active_failure / 5.0,
        ]

    def is_healthy(self) -> bool: return self.status == STATUS_HEALTHY
    def is_crashed(self) -> bool: return self.status == STATUS_CRASHED


SCENARIOS = {
    "database_crash": {
        "description": "Database hard crash — Payment and API Gateway will cascade",
        "injections": {"database": {"status": STATUS_CRASHED, "error_rate": 1.0, "cpu": 100.0, "active_failure": FAILURE_HARD_CRASH}}
    },
    "cpu_storm": {
        "description": "CPU storm across Auth and API Gateway",
        "injections": {
            "auth":        {"cpu": 95.0, "latency": 800.0,  "status": STATUS_DEGRADED, "active_failure": FAILURE_CPU_SPIKE},
            "api_gateway": {"cpu": 92.0, "latency": 1200.0, "status": STATUS_DEGRADED, "active_failure": FAILURE_CPU_SPIKE},
        }
    },
    "bad_deployment": {
        "description": "Bad deploy pushed to Payment service",
        "injections": {"payment": {"error_rate": 0.85, "latency": 900.0, "status": STATUS_DEGRADED, "active_failure": FAILURE_BAD_DEPLOY}}
    },
    "network_split": {
        "description": "Network partition — Notification service isolated",
        "injections": {"notification": {"error_rate": 0.9, "latency": 1800.0, "status": STATUS_DEGRADED, "active_failure": FAILURE_NETWORK_SPLIT}}
    },
    "full_meltdown": {
        "description": "FULL MELTDOWN — 3 services failing simultaneously",
        "injections": {
            "database": {"status": STATUS_CRASHED,  "error_rate": 1.0, "cpu": 100.0, "active_failure": FAILURE_HARD_CRASH},
            "payment":  {"status": STATUS_DEGRADED, "error_rate": 0.7, "cpu": 85.0,  "active_failure": FAILURE_BAD_DEPLOY},
            "auth":     {"status": STATUS_DEGRADED, "cpu": 91.0, "latency": 700.0,   "active_failure": FAILURE_CPU_SPIKE},
        }
    },
}


class CloudHealEnv(gym.Env):
    """
    CloudHealRL — Autonomous cloud cluster healing.
    Observation: 30-dim float32 (5 services × 6 metrics)
    Action:      Discrete(21)
    Reward:      Always 0.0–1.0 (cluster health fraction)
    """
    metadata = {"render_modes": ["human", "json"]}

    def __init__(self, task: int = 1, max_steps: int = 200):
        super().__init__()
        assert task in [1, 2, 3]
        self.task      = task
        self.max_steps = max_steps
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(30,), dtype=np.float32)
        self.action_space      = spaces.Discrete(N_ACTIONS)
        self.failure_interval  = {1: 20, 2: 12, 3: 7}[task]
        self.cascade_enabled   = task >= 2
        self.multi_fail        = task == 3
        self.services: Dict[str, Service] = {}
        self.step_count     = 0
        self.episode_reward = 0.0
        self.action_log: List[dict] = []
        self._next_failure_step = 0
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.services = {name: Service(name=name) for name in SERVICES}
        self.step_count     = 0
        self.episode_reward = 0.0
        self.action_log     = []
        self._next_failure_step = self.failure_interval + random.randint(-3, 3)
        return self._get_obs(), {}

    def step(self, action: int):
        self.step_count += 1
        failure_event = None
        if self.step_count >= self._next_failure_step:
            failure_event = self._inject_random_failure()
            self._next_failure_step = self.step_count + self.failure_interval + random.randint(-2, 4)
        action_name, action_result = self._apply_action(int(action))
        if self.cascade_enabled:
            self._propagate_cascades()
        self._tick_services()
        reward = self._compute_reward()
        self.episode_reward += reward
        self.action_log.append({
            "step": self.step_count, "action": action_name, "result": action_result,
            "reward": round(reward, 4), "failure": failure_event,
            "cluster_ok": round(self._cluster_health_pct(), 3),
        })
        done = self.step_count >= self.max_steps
        info = {
            "step": self.step_count, "action_name": action_name,
            "action_result": action_result, "failure_event": failure_event,
            "cluster_health": self._cluster_health_pct(),
            "episode_reward": round(self.episode_reward, 3),
            "services": self.get_state_dict(),
        }
        return self._get_obs(), reward, done, False, info

    def inject_scenario(self, scenario_name: str) -> dict:
        if scenario_name not in SCENARIOS:
            return {"error": f"Unknown scenario: {scenario_name}"}
        for svc_name, overrides in SCENARIOS[scenario_name]["injections"].items():
            svc = self.services[svc_name]
            for attr, val in overrides.items():
                setattr(svc, attr, val)
        return {"injected": scenario_name, "description": SCENARIOS[scenario_name]["description"]}

    def _apply_action(self, action: int) -> Tuple[str, str]:
        name = ACTION_NAMES.get(action, "unknown")
        if action == ACTION_DO_NOTHING:
            return name, "no_op"
        if   1  <= action <= 5:  atype, sidx = "scale_up", action - 1
        elif 6  <= action <= 10: atype, sidx = "restart",  action - 6
        elif 11 <= action <= 15: atype, sidx = "rollback", action - 11
        elif 16 <= action <= 20: atype, sidx = "reroute",  action - 16
        else: return name, "invalid"
        svc = self.services[SERVICES[sidx]]

        if atype == "scale_up":
            if svc.cpu > 60:
                svc.cpu     = max(svc.cpu - 35.0, 20.0)
                svc.latency = max(svc.latency - 200.0, 50.0)
                if svc.status == STATUS_DEGRADED and svc.active_failure == FAILURE_CPU_SPIKE:
                    svc.status = STATUS_HEALTHY; svc.active_failure = FAILURE_NONE
                return name, "cpu_reduced"
            return name, "not_needed"

        elif atype == "restart":
            if svc.status in [STATUS_DEGRADED, STATUS_CRASHED]:
                svc.cpu = 25.0; svc.memory = 30.0; svc.error_rate = 0.05
                svc.latency = 80.0; svc.status = STATUS_HEALTHY
                svc.active_failure = FAILURE_NONE; svc.heal_cooldown = 1
                return name, "restarted"
            return name, "already_healthy"

        elif atype == "rollback":
            if svc.active_failure == FAILURE_BAD_DEPLOY:
                svc.error_rate = 0.02; svc.latency = 60.0
                svc.status = STATUS_HEALTHY; svc.active_failure = FAILURE_NONE
                return name, "rolled_back"
            return name, "no_bad_deploy"

        elif atype == "reroute":
            if svc.active_failure == FAILURE_NETWORK_SPLIT:
                svc.error_rate = 0.03; svc.latency = 100.0
                svc.status = STATUS_HEALTHY; svc.active_failure = FAILURE_NONE
                return name, "rerouted"
            if svc.status == STATUS_CRASHED:
                for dep_name, deps in DEPENDENCIES.items():
                    if svc.name in deps:
                        self.services[dep_name].latency = max(self.services[dep_name].latency - 100, 50)
                return name, "partial_reroute"
            return name, "not_needed"

        return name, "no_effect"

    def _inject_random_failure(self) -> Optional[str]:
        targets = random.sample(SERVICES, 2 if self.multi_fail else 1)
        failure_types = [FAILURE_CPU_SPIKE, FAILURE_MEMORY_LEAK, FAILURE_BAD_DEPLOY, FAILURE_NETWORK_SPLIT, FAILURE_HARD_CRASH]
        events = []
        for svc_name in targets:
            svc = self.services[svc_name]
            if svc.is_crashed(): continue
            ftype = random.choice(failure_types)
            svc.active_failure = ftype
            if ftype == FAILURE_CPU_SPIKE:
                svc.cpu = min(svc.cpu + random.uniform(40, 65), 100); svc.latency = min(svc.latency + random.uniform(200, 600), 2000); svc.status = STATUS_DEGRADED
            elif ftype == FAILURE_MEMORY_LEAK:
                svc.memory = min(svc.memory + random.uniform(35, 55), 100); svc.error_rate = min(svc.error_rate + 0.2, 1.0); svc.status = STATUS_DEGRADED
            elif ftype == FAILURE_BAD_DEPLOY:
                svc.error_rate = min(svc.error_rate + random.uniform(0.4, 0.8), 1.0); svc.latency = min(svc.latency + random.uniform(300, 700), 2000); svc.status = STATUS_DEGRADED
            elif ftype == FAILURE_NETWORK_SPLIT:
                svc.error_rate = min(svc.error_rate + random.uniform(0.5, 0.9), 1.0); svc.latency = min(svc.latency + random.uniform(500, 1500), 2000); svc.status = STATUS_DEGRADED
            elif ftype == FAILURE_HARD_CRASH:
                svc.cpu = 100.0; svc.memory = 95.0; svc.error_rate = 1.0; svc.latency = 2000.0; svc.status = STATUS_CRASHED
            events.append(f"{svc_name}:{FAILURE_NAMES[ftype]}")
        return ", ".join(events) if events else None

    def _propagate_cascades(self):
        for svc_name, deps in DEPENDENCIES.items():
            svc = self.services[svc_name]
            if svc.is_crashed(): continue
            for dep_name in deps:
                dep = self.services[dep_name]
                if dep.is_crashed():
                    svc.error_rate = min(svc.error_rate + 0.08, 1.0)
                    svc.latency    = min(svc.latency + 80.0, 2000.0)
                    if svc.error_rate > 0.85: svc.status = STATUS_CRASHED
                    elif svc.error_rate > 0.25: svc.status = STATUS_DEGRADED
                elif dep.status == STATUS_DEGRADED:
                    svc.latency    = min(svc.latency + 30.0, 2000.0)
                    svc.error_rate = min(svc.error_rate + 0.02, 1.0)

    def _tick_services(self):
        for svc in self.services.values():
            if svc.heal_cooldown > 0: svc.heal_cooldown -= 1; continue
            if svc.is_healthy() and svc.active_failure == FAILURE_NONE:
                svc.cpu        = max(svc.cpu - 1.0, 15.0)
                svc.memory     = max(svc.memory - 0.5, 25.0)
                svc.latency    = max(svc.latency - 5.0, 40.0)
                svc.error_rate = max(svc.error_rate - 0.01, 0.0)
            if svc.active_failure == FAILURE_MEMORY_LEAK:
                svc.memory = min(svc.memory + 3.0, 100.0)
                if svc.memory > 90: svc.status = STATUS_CRASHED; svc.error_rate = 1.0

    def _compute_reward(self) -> float:
        total = 0.0
        for svc in self.services.values():
            if svc.status == STATUS_HEALTHY:   total += max(0.0, 1.0 - svc.error_rate * 0.3)
            elif svc.status == STATUS_DEGRADED: total += max(0.0, 0.4 - svc.error_rate * 0.4)
        return round(float(np.clip(total / len(SERVICES), 0.0, 1.0)), 4)

    def _get_obs(self) -> np.ndarray:
        obs = []
        for name in SERVICES: obs.extend(self.services[name].to_obs())
        return np.array(obs, dtype=np.float32)

    def _cluster_health_pct(self) -> float:
        return round(sum(1 for s in self.services.values() if s.is_healthy()) / len(SERVICES), 3)

    def get_state_dict(self) -> dict:
        return {name: asdict(svc) for name, svc in self.services.items()}

    def get_action_log(self) -> List[dict]:
        return self.action_log

    def render(self):
        print(f"\n=== Step {self.step_count} | Health: {self._cluster_health_pct()*100:.0f}% ===")
        for name, svc in self.services.items():
            print(f"  {name:15s} {['✅','⚠️ ','💀'][svc.status]} | CPU:{svc.cpu:5.1f}% MEM:{svc.memory:5.1f}% ERR:{svc.error_rate:.2f} LAT:{svc.latency:.0f}ms")


def grade(env: CloudHealEnv) -> float:
    if not env.action_log: return 0.0
    avg = sum(e["reward"] for e in env.action_log) / len(env.action_log)
    return round(float(np.clip(avg, 0.0, 1.0)), 4)


if __name__ == "__main__":
    print("=== CloudHealRL Smoke Test ===")
    for task in [1, 2, 3]:
        env = CloudHealEnv(task=task, max_steps=30)
        obs, _ = env.reset()
        for _ in range(30):
            obs, reward, done, _, info = env.step(env.action_space.sample())
            assert 0.0 <= reward <= 1.0
        print(f"Task {task}: grade={grade(env):.4f} health={info['cluster_health']*100:.0f}% ✅")
    print("All passed.")
