---
title: CloudHealRL
emoji: ⚡
colorFrom: cyan
colorTo: green
sdk: docker
pinned: true
license: MIT
---

# ⚡ CloudHealRL — Autonomous Cloud Cluster Healing
---
title: CloudHealthRL
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
app_file: server/app.py
---

# 🛡️ CloudHealthRL — Autonomous Cloud Cluster Healing via Reinforcement Learning

> **A PPO-trained RL agent that watches your microservice cluster in real time and heals it before cascading failures bring everything down.**

[![HF Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/vasvas23/CloudHealthRL)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Stable Baselines3](https://img.shields.io/badge/SB3-PPO-orange)](https://stable-baselines3.readthedocs.io/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-compatible-purple)](https://gymnasium.farama.org/)

---

## 🚨 The Problem

In production cloud systems, failures don't happen in isolation. A database crash cascades into API Gateway degradation, which spirals into payment failures, which triggers a full meltdown — all within seconds. Traditional alerting + manual remediation is too slow.

**CloudHealthRL solves this with a reinforcement learning agent that acts autonomously, selecting the right healing action at the right time.**

---

## 🧠 How It Works

The system models a realistic 5-service microservice cluster:

```
[Auth] ──► [API Gateway] ──► [Payment]  ──► [Database]
                         └──► [Notification]
```

Each service emits 6 real-time metrics (CPU, memory, error rate, latency, status, failure type). The RL agent observes all 30 values and selects from **21 discrete healing actions**:

| Action Category | Actions | What it does |
|---|---|---|
| `scale_up` | × 5 services | Reduces CPU load, clears CPU spike failures |
| `restart` | × 5 services | Full service reset, clears most failure types |
| `rollback` | × 5 services | Reverts a bad deployment specifically |
| `reroute` | × 5 services | Fixes network partitions, reduces cascade impact |
| `do_nothing` | × 1 | When the cluster is healthy, stay out of the way |

---

## ⚙️ Failure Scenarios

The environment injects 5 types of realistic cloud failures:

| Failure | Effect | Correct Response |
|---|---|---|
| **CPU Spike** | CPU → 95%, latency → 800ms+ | `scale_up` |
| **Memory Leak** | Memory creeps to 100%, service crashes | `restart` (early) |
| **Bad Deploy** | Error rate → 85%, latency spikes | `rollback` |
| **Network Split** | Error rate → 90%, packet loss | `reroute` |
| **Hard Crash** | CPU/MEM/ERR all max, service down | `restart` |

Plus **cascade propagation**: a crashed database degrades every service that depends on it.

### 🔥 Pre-built Stress Scenarios

```python
env.inject_scenario("database_crash")   # DB hard crash → cascade to Payment & API Gateway
env.inject_scenario("cpu_storm")        # Auth + API Gateway CPU storm simultaneously
env.inject_scenario("bad_deployment")  # Payment service bad deploy
env.inject_scenario("network_split")   # Notification service isolated
env.inject_scenario("full_meltdown")   # 3 services fail simultaneously
```

---

## 🏋️ Training Difficulty Levels

| Task | Failure Interval | Cascades | Multi-Failure |
|---|---|---|---|
| **Task 1** (Easy) | Every 20 steps | ❌ | ❌ |
| **Task 2** (Medium) | Every 12 steps | ✅ | ❌ |
| **Task 3** (Hard) | Every 7 steps | ✅ | ✅ (2 at once) |

---

## 🤖 Agent Architecture

The agent uses a **dual-mode design**:

1. **PPO Agent** (primary): Loaded from `models/cloudheal_ppo.zip` via Stable Baselines3. Hot-reloads automatically if the model file is updated.
2. **Heuristic Fallback**: Deterministic rule-based agent activates if no trained model is found. Prioritizes root dependencies first to prevent cascade amplification.

The heuristic heal order: `database → auth → api_gateway → payment → notification`

---

## 📊 Observation & Reward Space

**Observation**: `Box(30,)` — 5 services × 6 normalized metrics each (all in `[0.0, 1.0]`)

**Reward**: Cluster health fraction at each step
- Healthy service with low error rate → up to `1.0`
- Degraded service → up to `0.4`
- Crashed service → `0.0`
- Episode reward = average over all steps (`grade()` function)

---

## 🚀 Quick Start

### Run the Live Demo
➡️ **[Open the Space App](https://huggingface.co/spaces/vasvas23/CloudHealthRL)**

### Use the Environment Locally

```bash
git clone https://huggingface.co/spaces/vasvas23/CloudHealthRL
cd CloudHealthRL
pip install -r requirements.txt
```

```python
from environment import CloudHealEnv

env = CloudHealEnv(task=2, max_steps=200)
obs, _ = env.reset()

# Inject a scenario
env.inject_scenario("database_crash")

for _ in range(200):
    action = env.action_space.sample()  # replace with your agent
    obs, reward, done, _, info = env.step(action)
    if done:
        break

print(f"Final cluster health: {info['cluster_health'] * 100:.0f}%")
```

### Train with PPO

```bash
python train.py
```

The trained model is saved to `models/cloudheal_ppo.zip` and auto-loaded by the agent.

---

## 📁 Project Structure

```
CloudHealthRL/
├── environment.py      # Gymnasium-compatible RL environment (5 services, 21 actions)
├── agent.py            # PPO agent + heuristic fallback
├── train.py            # Training script (Stable Baselines3 PPO)
├── inference.py        # Inference utilities
├── server/
│   └── app.py          # FastAPI backend for the HF Space UI
├── models/
│   └── cloudheal_ppo.zip   # Trained PPO model weights
├── openenv.yaml        # Environment config
└── requirements.txt
```

---

## 🔬 Technical Details

- **RL Algorithm**: Proximal Policy Optimization (PPO) via Stable Baselines3
- **Environment**: Custom `gymnasium.Env` (Gymnasium-compatible)
- **Observation space**: `Box(low=0, high=1, shape=(30,), dtype=float32)`
- **Action space**: `Discrete(21)`
- **Reward range**: `[0.0, 1.0]` (cluster health fraction, clipped)
- **Backend**: FastAPI (Docker-based HF Space)
- **Cascade simulation**: Dependency-aware propagation with configurable intensity

---

## 📈 Results

The PPO agent achieves significantly higher average cluster health compared to random baseline, particularly on Task 3 (hard difficulty with multi-failures and fast cascades).

| Agent | Task 1 Grade | Task 2 Grade | Task 3 Grade |
|---|---|---|---|
| Random | ~0.55 | ~0.42 | ~0.31 |
| Heuristic | ~0.82 | ~0.74 | ~0.65 |
| **PPO (ours)** | **~0.91** | **~0.85** | **~0.78** |

---

## 🛣️ Roadmap

- [ ] Multi-agent healing (one agent per service)
- [ ] Real Kubernetes metrics integration (Prometheus adapter)
- [ ] SAC / TD3 agent comparison
- [ ] Longer context via recurrent policies (LSTM-PPO)
- [ ] Add anomaly detection pre-stage before RL decision

---

## 📜 License

MIT — free to use, modify, and deploy.

---

*Built with ❤️ using [Gymnasium](https://gymnasium.farama.org/), [Stable Baselines3](https://stable-baselines3.readthedocs.io/), and [Hugging Face Spaces](https://huggingface.co/spaces).*
