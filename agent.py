"""
CloudHealRL Agent
=================
PPO agent with heuristic fallback.
Heal order: root services first to prevent cascade amplification.
"""
import os
import numpy as np
from environment import (
    CloudHealEnv, SERVICES,
    STATUS_DEGRADED, STATUS_CRASHED, STATUS_HEALTHY,
    FAILURE_CPU_SPIKE, FAILURE_MEMORY_LEAK,
    FAILURE_BAD_DEPLOY, FAILURE_NETWORK_SPLIT, FAILURE_HARD_CRASH,
)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "cloudheal_ppo")
HEAL_ORDER = ["database", "auth", "api_gateway", "payment", "notification"]


class HeuristicAgent:
    """
    Priority rule agent:
    1. Restart crashed services (root deps first)
    2. Preemptive restart: memory leak > 70%
    3. Fix degraded by correct action per failure type
    4. Fix cascade victims (degraded, no tag)
    5. Preemptive CPU scale > 75%
    6. Do nothing
    """
    def _decode(self, obs):
        svcs = {}
        for i, name in enumerate(SERVICES):
            b = i * 6
            svcs[name] = {
                "idx": i, "name": name,
                "cpu":        float(obs[b])   * 100,
                "memory":     float(obs[b+1]) * 100,
                "error_rate": float(obs[b+2]),
                "latency":    float(obs[b+3]) * 2000,
                "status":     round(float(obs[b+4]) * 2),
                "failure":    round(float(obs[b+5]) * 5),
            }
        return svcs

    def predict(self, obs, deterministic=True, state=None, episode_start=None):
        svcs = self._decode(obs)
        # 1. Restart crashed (root first)
        for name in HEAL_ORDER:
            if svcs[name]["status"] == STATUS_CRASHED:
                return 6 + svcs[name]["idx"], state
        # 2. Preemptive memory leak
        for name in HEAL_ORDER:
            s = svcs[name]
            if s["failure"] == FAILURE_MEMORY_LEAK and s["memory"] > 70:
                return 6 + s["idx"], state
        # 3. Fix degraded by type
        for name in HEAL_ORDER:
            s = svcs[name]
            if s["status"] != STATUS_DEGRADED: continue
            f = s["failure"]
            if f == FAILURE_BAD_DEPLOY:    return 11 + s["idx"], state
            if f == FAILURE_NETWORK_SPLIT: return 16 + s["idx"], state
            if f == FAILURE_CPU_SPIKE:     return  1 + s["idx"], state
            if f in [FAILURE_MEMORY_LEAK, FAILURE_HARD_CRASH]: return 6 + s["idx"], state
            return 6 + s["idx"], state  # unknown failure → restart
        # 4. Cascade victims
        for name in HEAL_ORDER:
            s = svcs[name]
            if s["status"] == STATUS_DEGRADED:
                if s["error_rate"] > 0.4: return 6 + s["idx"], state
                if s["cpu"] > 75:         return 1 + s["idx"], state
        # 5. Preemptive CPU
        for name in HEAL_ORDER:
            if svcs[name]["cpu"] > 80: return 1 + svcs[name]["idx"], state
        return 0, state

    def is_trained(self): return False


class PPOAgent:
    def __init__(self):
        self.model = None
        self._heuristic = HeuristicAgent()
        self._model_mtime = None
        self._try_load()

    def _try_load(self):
        path = MODEL_PATH + ".zip"
        try:
            from stable_baselines3 import PPO
            if os.path.exists(path):
                self.model = PPO.load(MODEL_PATH)
                self._model_mtime = os.path.getmtime(path)
                print(f"[PPOAgent] Loaded model from {path}")
            else:
                print(f"[PPOAgent] No model at {path} — using heuristic.")
        except ImportError:
            print("[PPOAgent] stable-baselines3 not installed — using heuristic.")
        except Exception as e:
            print(f"[PPOAgent] Load error: {e} — using heuristic.")

    def _maybe_reload(self):
        path = MODEL_PATH + ".zip"
        if os.path.exists(path):
            try:
                mtime = os.path.getmtime(path)
                if mtime != self._model_mtime:
                    print("[PPOAgent] Model updated — reloading...")
                    self._try_load()
            except Exception: pass

    def predict(self, obs, deterministic=True, state=None, episode_start=None):
        self._maybe_reload()
        if self.model is not None:
            try:
                action, state = self.model.predict(obs, deterministic=True)
                return int(action), state
            except Exception as e:
                print(f"[PPOAgent] predict error: {e} — using heuristic")
        return self._heuristic.predict(obs, state=state)

    def is_trained(self): return self.model is not None


_agent_instance = None

def get_agent():
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = PPOAgent()
    else:
        _agent_instance._maybe_reload()
    return _agent_instance
