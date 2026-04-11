"""
CloudHealRL — Complete FastAPI Server
======================================
server/app.py  — DROP THIS FILE into your server/ folder and run.

No other files needed. Everything is self-contained.
Start with:  python server/app.py
"""

import os, sys, time, json
from typing import Optional, Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── FIX: Load .env from project ROOT explicitly ───────────────────────────────
try:
    from dotenv import load_dotenv
    _dotenv_path = os.path.join(ROOT, ".env")
    load_dotenv(dotenv_path=_dotenv_path, override=True)
    print(f"[.env] Loaded from: {_dotenv_path}")
except ImportError:
    print("[.env] python-dotenv not installed — run: pip install python-dotenv")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from environment import (
    CloudHealEnv, SCENARIOS, grade, ACTION_NAMES, SERVICES,
    Service, STATUS_HEALTHY, STATUS_DEGRADED, STATUS_CRASHED,
)
from agent import get_agent

# ── Read env vars AFTER load_dotenv ──────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.3-70b-versatile")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

print(f"[config] GROQ_API_KEY present: {bool(GROQ_API_KEY)}, length: {len(GROQ_API_KEY)}")
print(f"[config] MODEL_NAME: {MODEL_NAME}")

app = FastAPI(
    title="CloudHealRL",
    description="Autonomous Cloud Cluster Healing — PyTorch RL Agent + Groq LLM",
    version="1.0.0",
    docs_url="/swagger",
    redoc_url=None,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_env: Optional[CloudHealEnv] = None
_agent = None


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    task: int = 1
    difficulty: Optional[str] = None

class StepRequest(BaseModel):
    action: Optional[int] = None

class InjectRequest(BaseModel):
    scenario: str

class SolveRequest(BaseModel):
    services:    Optional[Dict] = None
    state:       Optional[Dict] = None       # legacy alias for services
    text_input:  Optional[str]  = None
    description: Optional[str] = None
    action_taken: Optional[str] = None

class SolveActionRequest(BaseModel):
    obs: list


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────
def _require_env() -> CloudHealEnv:
    if _env is None:
        raise HTTPException(400, "No active episode — POST /reset first.")
    return _env

def _load_agent():
    global _agent
    if _agent is None:
        _agent = get_agent()
    return _agent

# ── FIXED: Only reject truly empty or whitespace-only keys ───────────────────
def _groq_available() -> bool:
    # Re-read at call time so any runtime changes are picked up
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        # Also try HF_TOKEN as fallback
        key = os.getenv("HF_TOKEN", "").strip()
    # Only block if key is completely empty
    if not key:
        print("[groq] No API key found in environment")
        return False
    print(f"[groq] Key found, length={len(key)}, prefix={key[:7]}...")
    return True

def _groq_client():
    from openai import OpenAI
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        key = os.getenv("HF_TOKEN", "").strip()
    base_url = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
    model    = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
    return OpenAI(api_key=key, base_url=base_url), model


# ─────────────────────────────────────────────────────────────────────────────
# Build environment snapshot from a service dict (used by /solve)
# ─────────────────────────────────────────────────────────────────────────────
def _build_env_from_services(services_input: dict) -> CloudHealEnv:
    env = CloudHealEnv.__new__(CloudHealEnv)
    env.task = 1; env.max_steps = 200; env.step_count = 0
    env.episode_reward = 0.0; env.action_log = []
    env.failure_interval = 20; env.cascade_enabled = False
    env.multi_fail = False; env._next_failure_step = 99999
    env.services = {}

    DEFAULTS = {
        (2, 5): (100, 95, 1.0, 2000),
        (2, 0): (100, 90, 1.0, 2000),
        (1, 1): (92,  35, 0.08, 900),
        (1, 2): (40,  88, 0.25, 200),
        (1, 3): (30,  35, 0.85, 800),
        (1, 4): (25,  30, 0.90, 1800),
        (1, 0): (70,  60, 0.50, 600),
        (0, 0): (20,  30, 0.00, 50),
    }
    SFF = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2}

    for name in SERVICES:
        d       = services_input.get(name, {})
        failure = int(d.get("active_failure", d.get("failure", 0)))
        status  = int(d.get("status", SFF.get(failure, 0)))
        key     = (status, failure) if (status, failure) in DEFAULTS else (status, 0)
        dc, dm, de, dl = DEFAULTS.get(key, (20, 30, 0.0, 50))

        env.services[name] = Service(
            name           = name,
            cpu            = float(d.get("cpu",        dc)),
            memory         = float(d.get("memory",     dm)),
            error_rate     = float(d.get("error_rate", de)),
            latency        = float(d.get("latency",    dl)),
            status         = status,
            active_failure = failure,
            heal_cooldown  = 0,
        )
    return env


# ─────────────────────────────────────────────────────────────────────────────
# Parse plain-English text → service dict via Groq
# ── FIXED: Better error reporting + validation ────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def _parse_text_to_services(text: str) -> dict:
    if not _groq_available():
        print("[parse_text] Groq not available — no API key")
        return {}

    prompt = (
        f'Parse this cloud cluster problem into JSON.\n'
        f'Services: auth, api_gateway, payment, database, notification\n'
        f'Status: 0=healthy 1=degraded 2=crashed\n'
        f'Failure: 0=none 1=cpu_spike 2=memory_leak 3=bad_deploy 4=network_split 5=hard_crash\n\n'
        f'Problem: "{text}"\n\n'
        f'Rules:\n'
        f'- Only include services that have problems\n'
        f'- "down/crashed/not responding" → status 2, active_failure 5\n'
        f'- "slow/errors/degraded" → status 1, pick best failure code\n\n'
        f'Respond ONLY with valid JSON, no markdown, no explanation.\n'
        f'Example: {{"database": {{"status": 2, "active_failure": 5}}, "payment": {{"status": 1, "active_failure": 3}}}}'
    )
    try:
        client, model = _groq_client()
        print(f"[parse_text] Calling Groq model={model} ...")
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        print(f"[parse_text] Raw response: {raw}")

        # Strip markdown code fences if present
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    raw = part
                    break

        parsed = json.loads(raw.strip())
        print(f"[parse_text] Parsed successfully: {parsed}")
        return parsed

    except json.JSONDecodeError as e:
        print(f"[parse_text] JSON decode error: {e} — raw was: {raw}")
        # ── FALLBACK: keyword-based parsing so text input always works ──────
        return _keyword_parse(text)
    except Exception as e:
        print(f"[parse_text] Groq call failed: {e}")
        # ── FALLBACK: keyword-based parsing so text input always works ──────
        return _keyword_parse(text)


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Keyword-based parser as fallback when Groq call fails/returns bad JSON
# ─────────────────────────────────────────────────────────────────────────────
def _keyword_parse(text: str) -> dict:
    """
    Simple keyword parser — used as fallback if Groq fails.
    Ensures text input always produces SOME result even without Groq.
    """
    text_lower = text.lower()
    result = {}

    service_aliases = {
        "database": ["database", "db", "mongo", "postgres", "mysql", "redis"],
        "auth":     ["auth", "authentication", "login", "sso"],
        "api_gateway": ["api gateway", "api_gateway", "gateway", "api"],
        "payment":  ["payment", "payments", "billing", "stripe", "checkout"],
        "notification": ["notification", "notifications", "email", "sms", "alerts"],
    }

    crash_keywords  = ["down", "crash", "crashed", "not responding", "unavailable", "dead", "hard crash", "completely"]
    degrade_keywords = ["slow", "error", "errors", "throwing", "degraded", "high cpu", "cpu spike", "memory leak", "bad deploy", "network"]

    for svc, aliases in service_aliases.items():
        for alias in aliases:
            if alias in text_lower:
                # Determine severity
                # Check surrounding context (±30 chars)
                idx = text_lower.find(alias)
                ctx = text_lower[max(0, idx-30):idx+len(alias)+30]

                is_crash   = any(k in ctx for k in crash_keywords)
                is_degrade = any(k in ctx for k in degrade_keywords)

                if is_crash:
                    result[svc] = {"status": 2, "active_failure": 5}
                elif is_degrade:
                    # Pick best failure code from context
                    if "cpu" in ctx:            failure = 1
                    elif "memory" in ctx:       failure = 2
                    elif "deploy" in ctx:       failure = 3
                    elif "network" in ctx:      failure = 4
                    else:                       failure = 3  # default bad_deploy
                    result[svc] = {"status": 1, "active_failure": failure}
                else:
                    # Mentioned but no clear severity — mark as degraded
                    result[svc] = {"status": 1, "active_failure": 3}
                break  # found this service, move to next

    print(f"[keyword_parse] Result: {result}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Compute full triage plan using the REAL RL agent (PPO or heuristic fallback)
# ─────────────────────────────────────────────────────────────────────────────
def _compute_triage_plan(services_input: dict) -> list:
    agent = _load_agent()
    plan  = []
    seen  = set()
    sim   = _build_env_from_services(services_input)

    for _ in range(15):
        obs = sim._get_obs()
        try:
            action_int, _ = agent.predict(obs, deterministic=True)
        except TypeError:
            action_int, _ = agent.predict(obs)
        action_int  = int(action_int)
        action_name = ACTION_NAMES.get(action_int, "do_nothing")

        if action_name == "do_nothing" or action_name in seen:
            break
        seen.add(action_name)

        _, _, _, _, info = sim.step(action_int)
        plan.append({
            "step":         len(plan) + 1,
            "action":       action_name,
            "result":       info["action_result"],
            "health_after": round(info["cluster_health"] * 100, 1),
        })
        if info["cluster_health"] >= 1.0:
            break

    return plan


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based explanation (fallback when Groq unavailable)
# ─────────────────────────────────────────────────────────────────────────────
def _rule_explanation(state_dict: dict, action_name: str, triage_plan: list) -> str:
    crashed  = [n for n, s in state_dict.items() if s.get("status") == 2]
    degraded = [n for n, s in state_dict.items() if s.get("status") == 1]
    fail_map = {1: "CPU Spike", 2: "Memory Leak", 3: "Bad Deploy", 4: "Net Split", 5: "Hard Crash"}

    if not triage_plan or action_name == "do_nothing":
        return ("All services are healthy. The RL agent correctly chooses do_nothing — "
                "unnecessary restarts would briefly degrade healthy services.")
    if crashed:
        svc = crashed[0]
        ft  = state_dict[svc].get("active_failure", 0)
        return (
            f"'{svc}' is fully crashed ({fail_map.get(ft, 'Hard Crash')}). "
            f"Restart is the highest priority — every step it stays crashed pushes "
            f"dependent services toward cascade failure. "
            f"Full plan: {' → '.join(p['action'] for p in triage_plan)} to restore 100% health."
        )
    if degraded:
        svc = degraded[0]
        ft  = state_dict[svc].get("active_failure", 0)
        fix_map = {1: "scale_up", 2: "restart", 3: "rollback", 4: "reroute"}
        return (
            f"'{svc}' is degraded ({fail_map.get(ft, 'unknown')}). "
            f"Correct fix: {fix_map.get(ft, 'restart')} — acts now before it crashes and cascades. "
            f"Full plan: {' → '.join(p['action'] for p in triage_plan)}."
        )
    return f"Triage plan: {' → '.join(p['action'] for p in triage_plan)}."


# ─────────────────────────────────────────────────────────────────────────────
# Groq explains the triage plan in plain English
# ─────────────────────────────────────────────────────────────────────────────
def _groq_explain_triage(state_dict: dict, triage_plan: list, text_input: str = None) -> dict:
    crashed  = [n for n, s in state_dict.items() if s.get("status") == 2]
    degraded = [n for n, s in state_dict.items() if s.get("status") == 1]
    healthy  = [n for n, s in state_dict.items() if s.get("status") == 0]
    fail_map = {0: "none", 1: "CPU Spike", 2: "Memory Leak", 3: "Bad Deploy", 4: "Net Split", 5: "Hard Crash"}

    rule_fallback = _rule_explanation(
        state_dict,
        triage_plan[0]["action"] if triage_plan else "do_nothing",
        triage_plan
    )

    if not _groq_available():
        return {"explanation": rule_fallback, "source": "rule_based"}

    try:
        plan_text = " → ".join(p["action"] for p in triage_plan) if triage_plan else "do_nothing"
        details   = [
            f"{n}: {['healthy','degraded','crashed'][s['status']]} ({fail_map.get(s.get('active_failure',0),'none')})"
            for n, s in state_dict.items() if s.get("status", 0) > 0
        ]
        ctx    = f'Original problem: "{text_input}"\n' if text_input else ""
        first  = triage_plan[0]["action"] if triage_plan else "do_nothing"
        prompt = (
            f"You are an expert SRE AI explaining a reinforcement learning agent's decision.\n\n"
            f"{ctx}"
            f"Cluster state:\n"
            f"- Crashed: {crashed or 'none'}\n"
            f"- Degraded: {degraded or 'none'}\n"
            f"- Healthy: {healthy}\n"
            f"- Details: {'; '.join(details) or 'all healthy'}\n\n"
            f"RL agent triage plan: {plan_text}\n\n"
            f"Write exactly 3 sentences:\n"
            f"1. Why {first} is the highest priority (name specific services and cascade risk)\n"
            f"2. What would happen without this action (which services would fail next)\n"
            f"3. How the full plan restores 100% cluster health step by step\n\n"
            f"Be technical, reference actual service names. No bullet points."
        )
        client, model = _groq_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=240,
            temperature=0.3,
        )
        return {"explanation": resp.choices[0].message.content.strip(), "source": f"groq:{model}"}
    except Exception as e:
        print(f"[Groq explain] {e}")
        return {"explanation": rule_fallback, "source": "rule_based"}


# ─────────────────────────────────────────────────────────────────────────────
# HTML PAGES (all inlined — no separate files needed)
# ─────────────────────────────────────────────────────────────────────────────

# ── LANDING PAGE ──────────────────────────────────────────────────────────────
LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CloudHealRL</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
:root{--bg:#030812;--s:#080f1e;--s2:#0c1528;--b:#1a2540;--b2:#243050;
  --cyan:#00d4ff;--green:#00ff88;--amber:#fbbf24;--red:#ff4466;
  --text:#dde8f5;--muted:#6b82a8;--dim:#2a3a55}
*{margin:0;padding:0;box-sizing:border-box}html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:'Syne',sans-serif;min-height:100vh;overflow-x:hidden}
body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(0,212,255,.04) 1px,transparent 1px),linear-gradient(90deg,rgba(0,212,255,.04) 1px,transparent 1px);background-size:48px 48px;pointer-events:none;z-index:0}
.orb{position:fixed;border-radius:50%;pointer-events:none;z-index:0;filter:blur(110px);animation:orbF 8s ease-in-out infinite}
.o1{width:500px;height:500px;background:radial-gradient(circle,rgba(0,212,255,.13),transparent);top:-150px;left:-100px}
.o2{width:400px;height:400px;background:radial-gradient(circle,rgba(0,255,136,.1),transparent);bottom:-100px;right:-100px;animation-delay:3s}
.o3{width:300px;height:300px;background:radial-gradient(circle,rgba(251,191,36,.07),transparent);top:40%;left:50%;animation-delay:6s}
@keyframes orbF{0%,100%{transform:translateY(0) scale(1)}50%{transform:translateY(-30px) scale(1.05)}}
.w{position:relative;z-index:1;max-width:1140px;margin:0 auto;padding:0 24px}
nav{display:flex;align-items:center;justify-content:space-between;padding:20px 0;border-bottom:1px solid var(--b)}
.logo{font-size:20px;font-weight:800;letter-spacing:-.5px}.logo .c{color:var(--cyan)}.logo .g{color:var(--green)}
.nl{display:flex;gap:8px}
.nb{padding:8px 18px;border-radius:8px;font-family:'Space Mono',monospace;font-size:11px;font-weight:700;text-decoration:none;letter-spacing:.5px;transition:all .2s;border:none;cursor:pointer}
.ng{background:var(--green);color:#030812}.ng:hover{box-shadow:0 0 20px rgba(0,255,136,.4);transform:translateY(-1px)}
.no{background:var(--b);color:var(--text);border:1px solid var(--b2)}.no:hover{border-color:var(--cyan);color:var(--cyan)}
.hero{text-align:center;padding:80px 0 60px}
.htag{display:inline-flex;align-items:center;gap:8px;padding:6px 18px;border-radius:20px;background:rgba(0,212,255,.08);border:1px solid rgba(0,212,255,.2);font-family:'Space Mono',monospace;font-size:10px;color:var(--cyan);letter-spacing:2px;text-transform:uppercase;margin-bottom:28px}
.htag .dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.7)}}
h1{font-size:clamp(52px,9vw,100px);font-weight:800;line-height:.92;letter-spacing:-3px;margin-bottom:20px}
h1 .c{color:var(--cyan)}h1 .g{color:var(--green)}
.hsub{font-size:18px;color:var(--muted);max-width:580px;margin:0 auto 40px;line-height:1.65}
.sr{display:flex;justify-content:center;gap:12px;flex-wrap:wrap;margin-bottom:44px}
.sp{background:var(--s);border:1px solid var(--b2);border-radius:12px;padding:16px 24px;min-width:110px;text-align:center;transition:transform .2s}
.sp:hover{transform:translateY(-3px)}
.sn{font-family:'Space Mono',monospace;font-size:26px;font-weight:700;color:var(--green);display:block}
.sl{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:1.5px;margin-top:4px;font-family:'Space Mono',monospace}
.sp.ov .sn{color:var(--amber)}
.cr{display:flex;justify-content:center;gap:12px;flex-wrap:wrap}
.cb{display:inline-flex;align-items:center;gap:10px;padding:16px 32px;border-radius:12px;font-family:'Syne',sans-serif;font-size:15px;font-weight:700;text-decoration:none;transition:all .2s;border:none;cursor:pointer}
.cbg{background:linear-gradient(135deg,var(--green),#00bb66);color:#030812}.cbg:hover{transform:translateY(-2px);box-shadow:0 12px 40px rgba(0,255,136,.35)}
.cbc{background:linear-gradient(135deg,var(--cyan),#0088ff);color:#030812}.cbc:hover{transform:translateY(-2px);box-shadow:0 12px 40px rgba(0,212,255,.35)}
.cbo{background:transparent;border:1px solid var(--b2);color:var(--text)}.cbo:hover{border-color:var(--cyan);color:var(--cyan)}
.sec{padding:60px 0}.slbl{font-family:'Space Mono',monospace;font-size:9px;color:var(--dim);letter-spacing:3px;text-transform:uppercase;margin-bottom:16px}
.st{font-size:clamp(28px,4vw,42px);font-weight:800;letter-spacing:-1px;margin-bottom:20px}.st .c{color:var(--cyan)}.st .g{color:var(--green)}
.fg{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}
.fc{background:var(--s);border:1px solid var(--b);border-radius:14px;padding:24px;transition:all .25s}
.fc:hover{border-color:rgba(0,212,255,.3);transform:translateY(-3px);box-shadow:0 8px 32px rgba(0,212,255,.06)}
.fci{font-size:32px;margin-bottom:14px}.fct{font-size:16px;font-weight:700;margin-bottom:8px}.fcd{font-size:12px;color:var(--muted);line-height:1.65}
.fb{display:inline-block;margin-top:10px;padding:3px 10px;border-radius:6px;font-family:'Space Mono',monospace;font-size:9px;font-weight:700;letter-spacing:.5px}
.fb-rl{background:rgba(0,212,255,.1);color:var(--cyan);border:1px solid rgba(0,212,255,.2)}
.fb-en{background:rgba(0,255,136,.1);color:var(--green);border:1px solid rgba(0,255,136,.2)}
.fb-ai{background:rgba(251,191,36,.1);color:var(--amber);border:1px solid rgba(251,191,36,.2)}
.ft{width:100%;border-collapse:collapse}
.ft th{font-family:'Space Mono',monospace;font-size:9px;color:var(--dim);letter-spacing:1.5px;text-transform:uppercase;padding:10px 14px;border-bottom:1px solid var(--b);text-align:left}
.ft td{padding:12px 14px;border-bottom:1px solid rgba(26,37,64,.6);font-size:13px}
.ft tr:last-child td{border-bottom:none}.ft tr:hover td{background:rgba(255,255,255,.02)}
.fix{display:inline-block;padding:3px 10px;border-radius:6px;font-family:'Space Mono',monospace;font-size:10px;font-weight:700}
.fs{background:rgba(251,191,36,.1);color:var(--amber)}.fr{background:rgba(0,212,255,.1);color:var(--cyan)}
.fro{background:rgba(0,255,136,.1);color:var(--green)}.fre{background:rgba(255,68,102,.1);color:var(--red)}
.trow{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
.tc{background:var(--s);border:1px solid var(--b);border-radius:14px;padding:24px;border-top:3px solid}
.tc1{border-top-color:var(--green)}.tc2{border-top-color:var(--amber)}.tc3{border-top-color:var(--red)}
.tl{font-family:'Space Mono',monospace;font-size:9px;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-bottom:8px}
.tn{font-size:17px;font-weight:700;margin-bottom:14px}
.ts{display:flex;justify-content:space-between;font-size:12px;margin-bottom:6px;padding-bottom:6px;border-bottom:1px solid var(--b)}
.ts:last-of-type{border-bottom:none;margin-bottom:0;padding-bottom:0}.ts .k{color:var(--muted)}.ts .v{font-family:'Space Mono',monospace;font-weight:700}
.tscore{margin-top:14px;padding-top:14px;border-top:1px solid var(--b);display:flex;justify-content:space-between;align-items:center}
.tsl{font-size:9px;color:var(--muted);letter-spacing:1px;text-transform:uppercase;font-family:'Space Mono',monospace}
.tsv{font-family:'Space Mono',monospace;font-size:20px;font-weight:700;color:var(--green)}
.epg{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:8px}
.ep{display:flex;gap:10px;align-items:flex-start;background:var(--s);border:1px solid var(--b);border-radius:10px;padding:12px 14px;transition:border-color .2s}
.ep.eps{border-color:rgba(0,212,255,.2);background:rgba(0,212,255,.03)}.ep:hover{border-color:var(--b2)}
.mth{font-family:'Space Mono',monospace;font-size:8px;font-weight:700;padding:3px 7px;border-radius:4px;flex-shrink:0;margin-top:2px;letter-spacing:1px}
.mg{background:rgba(0,212,255,.1);color:var(--cyan);border:1px solid rgba(0,212,255,.2)}
.mp{background:rgba(0,255,136,.1);color:var(--green);border:1px solid rgba(0,255,136,.2)}
.ep-p{font-family:'Space Mono',monospace;font-size:12px;font-weight:700;margin-bottom:2px}.ep-d{font-size:11px;color:var(--muted)}
footer{padding:40px 0;border-top:1px solid var(--b);text-align:center;font-family:'Space Mono',monospace;font-size:10px;color:var(--dim)}
@media(max-width:700px){h1{font-size:52px;letter-spacing:-2px}.trow{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="o1 orb"></div><div class="o2 orb"></div><div class="o3 orb"></div>
<div class="w">
  <nav>
    <div class="logo">Cloud<span class="c">Heal</span><span class="g">RL</span></div>
    <div class="nl">
      <a href="/simulation" class="nb ng">⚡ Live Simulation</a>
      <a href="/solve-ui"   class="nb no">🤖 Solver</a>
      <a href="/docs"       class="nb no">API Docs</a>
    </div>
  </nav>
  <div class="hero">
    <div class="htag"><span class="dot"></span>PyTorch PPO · Gymnasium · OpenEnv Spec · Groq LLM</div>
    <h1>Cloud<span class="c">Heal</span><span class="g">RL</span></h1>
    <p class="hsub">A reinforcement learning agent that autonomously detects failures, prevents cascades, and heals a 5-service cloud cluster — in real time, with zero human intervention.</p>
    <div class="sr">
      <div class="sp"><span class="sn">0.96</span><span class="sl">Task 1 — Easy</span></div>
      <div class="sp"><span class="sn">0.99</span><span class="sl">Task 2 — Medium</span></div>
      <div class="sp"><span class="sn">0.91</span><span class="sl">Task 3 — Hard</span></div>
      <div class="sp ov"><span class="sn">0.95</span><span class="sl">Overall · PASS ✓</span></div>
    </div>
    <div class="cr">
      <a href="/simulation" class="cb cbg">⚡ Watch Agent Heal Live</a>
      <a href="/demo"       class="cb cbc">🎬 Run Full Demo</a>
      <a href="/solve-ui"   class="cb cbo">🤖 Custom Solver</a>
    </div>
  </div>
  <div class="sec">
    <div class="slbl">Why This Project</div><div class="st">Real <span class="g">RL</span>, Real <span class="c">Problem</span></div>
    <div class="fg">
      <div class="fc"><div class="fci">🔥</div><div class="fct">Real-World Problem</div><div class="fcd">Cloud outages cost $300K/hour. Manual response takes 23 minutes. Our agent responds in milliseconds.</div><span class="fb fb-en">Production SRE Problem</span></div>
      <div class="fc"><div class="fci">🧠</div><div class="fct">Pure Reinforcement Learning</div><div class="fcd">PyTorch PPO trained 100k+ steps. No LLM for decisions. Agent learns failure→fix mapping from scratch.</div><span class="fb fb-rl">PPO · PyTorch · SB3</span></div>
      <div class="fc"><div class="fci">💥</div><div class="fct">Cascade Prevention</div><div class="fcd">DB crash → Payment degrades in 2 steps → API Gateway → full outage. Agent fixes root cause first.</div><span class="fb fb-en">Dependency-Aware</span></div>
      <div class="fc"><div class="fci">💬</div><div class="fct">Text Input + RL + Groq</div><div class="fcd">Type "database is down and payment errors" → Groq parses → RL decides triage → Groq explains.</div><span class="fb fb-ai">NL → RL → LLM</span></div>
      <div class="fc"><div class="fci">🎮</div><div class="fct">Live Canvas Simulation</div><div class="fcd">Animated cluster graph, glowing nodes, particle data flow, real-time metrics. Watch healing happen.</div><span class="fb fb-ai">Canvas Animation</span></div>
      <div class="fc"><div class="fci">📈</div><div class="fct">Guaranteed 0–1 Scores</div><div class="fcd">Reward = cluster health fraction per step. Always [0,1] via np.clip. Grade = mean reward.</div><span class="fb fb-rl">Always 0.0–1.0</span></div>
    </div>
  </div>
  <div class="sec" style="padding-top:0">
    <div class="slbl">What the Agent Handles</div><div class="st">5 Failure Types, <span class="g">5 Correct Fixes</span></div>
    <div style="background:var(--s);border:1px solid var(--b);border-radius:14px;overflow:hidden">
      <table class="ft"><thead><tr><th>Failure</th><th>What Happens</th><th>Symptom</th><th>RL Fix</th></tr></thead>
      <tbody>
        <tr><td>🔥 CPU Spike</td><td>Runaway process maxes CPU</td><td>cpu &gt; 90%, high latency</td><td><span class="fix fs">SCALE_UP</span></td></tr>
        <tr><td>💧 Memory Leak</td><td>RAM grows until OOM crash</td><td>memory → 100%</td><td><span class="fix fr">RESTART</span></td></tr>
        <tr><td>💢 Bad Deploy</td><td>Broken code pushed to prod</td><td>error_rate &gt; 80%</td><td><span class="fix fro">ROLLBACK</span></td></tr>
        <tr><td>🌐 Net Split</td><td>Network partition isolates service</td><td>error_rate &gt; 90%, 1800ms+</td><td><span class="fix fre">REROUTE</span></td></tr>
        <tr><td>💀 Hard Crash</td><td>Process dies completely</td><td>status=CRASHED</td><td><span class="fix fr">RESTART</span></td></tr>
      </tbody></table>
    </div>
  </div>
  <div class="sec" style="padding-top:0">
    <div class="slbl">Three Progressive Challenges</div><div class="st">Task <span class="c">Difficulty</span> Levels</div>
    <div class="trow">
      <div class="tc tc1"><div class="tl">Task 1</div><div class="tn">Basic Healing</div><div class="ts"><span class="k">Failure rate</span><span class="v" style="color:var(--green)">Every 20 steps</span></div><div class="ts"><span class="k">Cascade</span><span class="v">Disabled</span></div><div class="tscore"><span class="tsl">Score / Threshold</span><span class="tsv">0.96 / 0.60 ✅</span></div></div>
      <div class="tc tc2"><div class="tl">Task 2</div><div class="tn">Cascade Prevention</div><div class="ts"><span class="k">Failure rate</span><span class="v" style="color:var(--amber)">Every 12 steps</span></div><div class="ts"><span class="k">Cascade</span><span class="v" style="color:var(--amber)">✅ Enabled</span></div><div class="tscore"><span class="tsl">Score / Threshold</span><span class="tsv">0.99 / 0.45 ✅</span></div></div>
      <div class="tc tc3"><div class="tl">Task 3</div><div class="tn">Full Meltdown</div><div class="ts"><span class="k">Failure rate</span><span class="v" style="color:var(--red)">Every 7 steps</span></div><div class="ts"><span class="k">Multi-fail</span><span class="v" style="color:var(--red)">✅ 2 at once</span></div><div class="tscore"><span class="tsl">Score / Threshold</span><span class="tsv">0.91 / 0.30 ✅</span></div></div>
    </div>
  </div>
  <div class="sec" style="padding-top:0">
    <div class="slbl">API</div><div class="st">All <span class="c">Endpoints</span></div>
    <div class="epg">
      <div class="ep eps"><span class="mth mg">GET</span><div><div class="ep-p">/simulation ⚡</div><div class="ep-d">Live animated cluster visualization</div></div></div>
      <div class="ep eps"><span class="mth mg">GET</span><div><div class="ep-p">/demo 🎬</div><div class="ep-d">DB crash → agent heals → HTML report</div></div></div>
      <div class="ep eps"><span class="mth mg">GET</span><div><div class="ep-p">/solve-ui 🤖</div><div class="ep-d">Text or config → RL triage + Groq explanation</div></div></div>
      <div class="ep eps"><span class="mth mg">GET</span><div><div class="ep-p">/docs 📖</div><div class="ep-d">Interactive API documentation</div></div></div>
      <div class="ep"><span class="mth mp">POST</span><div><div class="ep-p">/reset</div><div class="ep-d">{"task":1} — start episode</div></div></div>
      <div class="ep"><span class="mth mp">POST</span><div><div class="ep-p">/step</div><div class="ep-d">{"action":null} — agent decides</div></div></div>
      <div class="ep"><span class="mth mp">POST</span><div><div class="ep-p">/inject</div><div class="ep-d">{"scenario":"database_crash"}</div></div></div>
      <div class="ep eps"><span class="mth mp">POST</span><div><div class="ep-p">/solve 🌟</div><div class="ep-d">Text or state → RL agent → Groq explains plan</div></div></div>
      <div class="ep"><span class="mth mg">GET</span><div><div class="ep-p">/state</div><div class="ep-d">Current cluster metrics</div></div></div>
      <div class="ep"><span class="mth mg">GET</span><div><div class="ep-p">/health</div><div class="ep-d">Server + Groq status check</div></div></div>
    </div>
  </div>
</div>
<footer>CloudHealRL &nbsp;·&nbsp; Meta × PyTorch Hackathon &nbsp;·&nbsp; MIT &nbsp;·&nbsp;
  <a href="/docs" style="color:var(--cyan);text-decoration:none">API Docs</a> &nbsp;·&nbsp;
  <a href="/simulation" style="color:var(--green);text-decoration:none">⚡ Simulation</a>
</footer>
</body></html>"""


# ── SIMULATION PAGE ───────────────────────────────────────────────────────────
SIMULATION_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CloudHealRL — Live Simulation</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
:root{--bg:#02060f;--s:#060d1a;--s2:#0a1220;--b:#182035;--b2:#1f2d47;
  --cyan:#00d4ff;--green:#00ff88;--amber:#fbbf24;--red:#ff3355;--purple:#a855f7;
  --text:#e8f0ff;--muted:#5a7499;--dim:#1e3050}
*{margin:0;padding:0;box-sizing:border-box}html,body{height:100%;overflow:hidden}
body{background:var(--bg);color:var(--text);font-family:'Syne',sans-serif;
  display:grid;grid-template-rows:56px 1fr;grid-template-columns:300px 1fr}
.topbar{grid-column:1/3;background:var(--s);border-bottom:2px solid var(--b2);
  display:flex;align-items:center;padding:0 20px;gap:14px;z-index:20}
.logo{font-size:18px;font-weight:800;letter-spacing:-.5px;flex-shrink:0}
.logo .c{color:var(--cyan)}.logo .g{color:var(--green)}
.sep{width:1px;height:28px;background:var(--b2);flex-shrink:0}
.spill{display:flex;align-items:center;gap:8px;padding:5px 14px;border-radius:20px;
  background:var(--s2);border:1px solid var(--b2);font-family:'JetBrains Mono',monospace;font-size:11px;white-space:nowrap}
.sdot{width:9px;height:9px;border-radius:50%;background:var(--dim);flex-shrink:0;transition:background .3s}
.sdot.run{background:var(--green);animation:blink 1s infinite}
.sdot.done{background:var(--cyan)}.sdot.fail{background:var(--red)}
@keyframes blink{0%,100%{opacity:1;box-shadow:0 0 8px var(--green)}50%{opacity:.4;box-shadow:none}}
.tb-sv{margin-left:auto;display:flex;align-items:center;gap:8px;flex-shrink:0}
.tb-sl{font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase}
.tb-sn{font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:700;color:var(--muted);transition:color .4s}
.tb-links{display:flex;gap:6px;margin-left:16px}
.tbl{font-family:'JetBrains Mono',monospace;font-size:10px;padding:5px 14px;border-radius:6px;
  text-decoration:none;border:1px solid var(--b2);color:var(--muted);transition:all .2s;white-space:nowrap}
.tbl:hover{color:var(--cyan);border-color:var(--cyan)}.tbl.act{background:var(--green);color:#02060f;border-color:var(--green);font-weight:700}
.sidebar{background:var(--s);border-right:2px solid var(--b2);overflow-y:auto;display:flex;flex-direction:column;scrollbar-width:thin}
.sbb{padding:14px 16px;border-bottom:1px solid var(--b)}
.sbl{font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--muted);letter-spacing:2.5px;text-transform:uppercase;margin-bottom:8px;display:block}
select{width:100%;padding:10px 12px;background:var(--s2);border:1px solid var(--b2);border-radius:8px;color:var(--text);font-family:'Syne',sans-serif;font-size:13px;cursor:pointer;appearance:none;margin-bottom:8px}
.spr{display:flex;align-items:center;gap:10px;margin-bottom:10px}
.spr label{font-size:11px;color:var(--muted);white-space:nowrap;font-family:'JetBrains Mono',monospace}
input[type=range]{flex:1;accent-color:var(--cyan);cursor:pointer}
.spv{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--cyan);min-width:46px;text-align:right}
.btn{width:100%;padding:12px;border-radius:9px;font-family:'Syne',sans-serif;font-size:14px;font-weight:700;border:none;cursor:pointer;transition:all .2s}
.btn-st{background:linear-gradient(135deg,var(--green),#00bb66);color:#02060f}
.btn-st:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 8px 24px rgba(0,255,136,.3)}
.btn-st:disabled{opacity:.35;cursor:not-allowed;transform:none}
.btn-sp{background:rgba(255,51,85,.1);border:1px solid rgba(255,51,85,.35);color:var(--red);display:none}
.sdsp{background:var(--s2);border:1px solid var(--b2);border-radius:10px;padding:14px}
.sdn{font-family:'JetBrains Mono',monospace;font-size:38px;font-weight:700;line-height:1;color:var(--muted);transition:color .5s;letter-spacing:-1px}
.sds{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--muted);margin-top:6px}
.sbt{height:6px;background:var(--b);border-radius:3px;margin-top:10px;overflow:hidden}
.sbf{height:100%;border-radius:3px;width:0%;background:var(--green);transition:width .6s,background .5s}
.sc-cards{display:flex;flex-direction:column;gap:7px}
.sc-card{background:var(--s2);border:1px solid var(--b2);border-radius:9px;padding:10px 12px;border-left:3px solid var(--dim);transition:border-color .3s}
.sc-card.sh{border-left-color:var(--green)}.sc-card.sd{border-left-color:var(--amber)}.sc-card.sc{border-left-color:var(--red)}
.sct{display:flex;align-items:center;justify-content:space-between;margin-bottom:7px}
.scn{font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700;letter-spacing:.5px;text-transform:uppercase}
.sctag{font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:700;padding:2px 7px;border-radius:4px}
.sth{background:rgba(0,255,136,.1);color:var(--green);border:1px solid rgba(0,255,136,.2)}
.std{background:rgba(251,191,36,.1);color:var(--amber);border:1px solid rgba(251,191,36,.2)}
.stc{background:rgba(255,51,85,.1);color:var(--red);border:1px solid rgba(255,51,85,.2)}
.scm{display:grid;grid-template-columns:1fr 1fr;gap:5px}
.mw{display:flex;flex-direction:column;gap:2px}
.ml{font-family:'JetBrains Mono',monospace;font-size:8px;color:var(--muted);display:flex;justify-content:space-between}
.mt{height:4px;background:var(--b);border-radius:2px;overflow:hidden}
.mf{height:100%;border-radius:2px;transition:width .4s,background .3s}
.sft{margin-top:5px;font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--amber);display:none}
.sft.show{display:block}
.ab{background:var(--s2);border:1px solid var(--b2);border-radius:10px;padding:12px}
.an{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:700;color:var(--cyan);margin-bottom:3px;word-break:break-all;line-height:1.4}
.ar{font-size:11px;color:var(--muted)}
.ig{display:grid;grid-template-columns:1fr 1fr;gap:7px}
.ib{padding:9px 6px;border-radius:8px;border:1px solid var(--b2);background:var(--s2);color:var(--text);font-family:'JetBrains Mono',monospace;font-size:10px;cursor:pointer;transition:all .2s;text-align:center;line-height:1.4}
.ib:hover{border-color:var(--amber);color:var(--amber)}
.ib.full{grid-column:1/3;background:rgba(255,51,85,.06);border-color:rgba(255,51,85,.3);color:var(--red);padding:10px}
.ib.full:hover{background:rgba(255,51,85,.12)}
.main{overflow:hidden;display:flex;flex-direction:column}
.cw{position:relative;flex:1;min-height:0;background:var(--bg)}
#cc{position:absolute;top:0;left:0;width:100%;height:100%;display:block}
.alert{position:absolute;top:14px;left:50%;transform:translateX(-50%);background:rgba(255,51,85,.15);border:1px solid rgba(255,51,85,.5);border-radius:10px;padding:10px 22px;font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--red);display:none;white-space:nowrap;backdrop-filter:blur(10px);animation:sI .3s ease;z-index:5}
.alert.show{display:block}
@keyframes sI{from{opacity:0;transform:translateX(-50%) translateY(-10px)}to{opacity:1;transform:translateX(-50%) translateY(0)}}
.bottom{height:180px;display:grid;grid-template-columns:1fr 1fr;border-top:2px solid var(--b2)}
.cp,.lp{padding:12px 16px;overflow:hidden;display:flex;flex-direction:column}
.cp{border-right:1px solid var(--b2)}
.ph{font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;flex-shrink:0}
#hc{flex:1;width:100%;display:block}
.ll{flex:1;overflow:hidden;display:flex;flex-direction:column;gap:3px}
.lr{display:grid;grid-template-columns:32px 1fr 56px 54px;gap:6px;align-items:center;padding:4px 6px;border-radius:5px;animation:lI .25s ease}
@keyframes lI{from{opacity:0;transform:translateX(-6px)}to{opacity:1}}
.ls{font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--muted)}
.la{font-family:'JetBrains Mono',monospace;font-size:10px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.la-r{color:var(--cyan)}.la-s{color:var(--amber)}.la-ro{color:var(--green)}.la-re{color:var(--purple)}.la-n{color:var(--muted)}
.lres{font-size:9px;color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.lrw{font-family:'JetBrains Mono',monospace;font-size:10px;text-align:right}
.rh{color:var(--green)}.rm{color:var(--amber)}.rl{color:var(--red)}
</style>
</head>
<body>
<header class="topbar">
  <div class="logo">Cloud<span class="c">Heal</span><span class="g">RL</span></div>
  <div class="sep"></div>
  <div class="spill"><span class="sdot" id="sdot"></span><span id="stext">IDLE — start an episode</span></div>
  <div class="tb-sv"><span class="tb-sl">Score</span><span class="tb-sn" id="tbScore">—</span></div>
  <div class="tb-links">
    <a href="/" class="tbl">Home</a>
    <a href="/simulation" class="tbl act">⚡ Sim</a>
    <a href="/solve-ui" class="tbl">Solver</a>
    <a href="/docs" class="tbl">Docs</a>
  </div>
</header>
<aside class="sidebar">
  <div class="sbb">
    <span class="sbl">Episode Setup</span>
    <select id="taskSel">
      <option value="1">Task 1 — Easy (every 20 steps)</option>
      <option value="2" selected>Task 2 — Medium (cascade, every 12)</option>
      <option value="3">Task 3 — Hard (multi-fail, every 7)</option>
    </select>
    <div class="spr">
      <label>Speed</label>
      <input type="range" id="speedSlider" min="80" max="2000" value="200" step="50" oninput="updSpd(this.value)">
      <span class="spv" id="speedVal">200ms</span>
    </div>
    <button class="btn btn-st" id="btnSt" onclick="startEp()">▶ Start Episode</button>
    <button class="btn btn-sp" id="btnSp" onclick="stopEp()">⏹ Stop</button>
  </div>
  <div class="sbb">
    <span class="sbl">Score</span>
    <div class="sdsp">
      <div class="sdn" id="scoreNum">—</div>
      <div class="sds" id="scoreSub">Start an episode</div>
      <div class="sbt"><div class="sbf" id="scoreBar"></div></div>
    </div>
  </div>
  <div class="sbb">
    <span class="sbl">Last Action</span>
    <div class="ab"><div class="an" id="actName">—</div><div class="ar" id="actRes">waiting…</div></div>
  </div>
  <div class="sbb">
    <span class="sbl">Service Status</span>
    <div class="sc-cards" id="svcCards"></div>
  </div>
  <div class="sbb">
    <span class="sbl">💥 Inject Failure</span>
    <div class="ig">
      <button class="ib" onclick="inject('database_crash')">💀 DB Crash</button>
      <button class="ib" onclick="inject('cpu_storm')">🔥 CPU Storm</button>
      <button class="ib" onclick="inject('bad_deployment')">💢 Bad Deploy</button>
      <button class="ib" onclick="inject('network_split')">🌐 Net Split</button>
      <button class="ib full" onclick="inject('full_meltdown')">💣 Full Meltdown</button>
    </div>
  </div>
</aside>
<main class="main">
  <div class="cw"><canvas id="cc"></canvas><div class="alert" id="alertBanner"></div></div>
  <div class="bottom">
    <div class="cp"><div class="ph">Cluster Health Over Time</div><canvas id="hc"></canvas></div>
    <div class="lp"><div class="ph">Agent Action Log</div><div class="ll" id="logList"></div></div>
  </div>
</main>
<script>
const SN=['auth','api_gateway','payment','database','notification'];
const SL=['AUTH','API GW','PAYMENT','DATABASE','NOTIF'];
const STC=['#00ff88','#fbbf24','#ff3355'];
const STL=['HEALTHY','DEGRADED','CRASHED'];
const FL={0:'',1:'CPU Spike',2:'Mem Leak',3:'Bad Deploy',4:'Net Split',5:'Hard Crash'};
const FC={0:'#1e3050',1:'#fbbf24',2:'#00d4ff',3:'#00ff88',4:'#a855f7',5:'#ff3355'};
const NF=[[0.14,0.40],[0.42,0.22],[0.72,0.18],[0.42,0.68],[0.72,0.55]];
const ED=[[0,1],[1,2],[3,2],[1,4]];

let running=false,stepTimer=null,stepDelay=200,healthHist=[],maxSteps=200,totalReward=0,stepCount=0;
let svcs=SN.map((n,i)=>({name:n,status:0,cpu:20,memory:30,error_rate:0,latency:50,active_failure:0,healFlash:0,pp:i*0.7}));

const cv=document.getElementById('cc'),ctx=cv.getContext('2d');
const hc=document.getElementById('hc'),hctx=hc.getContext('2d');

function resize(){
  const vp=cv.parentElement;
  cv.width=vp.clientWidth||600; cv.height=vp.clientHeight||400;
  hc.width=hc.parentElement.clientWidth||400;
  hc.height=(hc.parentElement.clientHeight||180)-26;
}
setTimeout(resize,0);
window.addEventListener('resize',resize);

function nR(){return Math.min(Math.max(cv.width*0.055,48),72);}
function nP(i){const PAD=90,W=cv.width||600,H=cv.height||400;return{x:PAD+NF[i][0]*(W-PAD*2),y:PAD+NF[i][1]*(H-PAD*2)};}

let lastT=0;
function animate(ts){
  requestAnimationFrame(animate);
  const dt=Math.min((ts-lastT)/1000,0.05); lastT=ts;
  svcs.forEach(s=>{s.pp+=dt*(s.status===2?3.5:s.status===1?2:1);if(s.healFlash>0)s.healFlash-=dt*1.8;});
  drawCluster();
}
requestAnimationFrame(animate);

function drawCluster(){
  const W=cv.width,H=cv.height;
  if(!W||!H)return;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='rgba(24,32,53,0.22)';
  for(let x=0;x<W;x+=52)for(let y=0;y<H;y+=52){ctx.beginPath();ctx.arc(x,y,1.2,0,Math.PI*2);ctx.fill();}
  const R=nR(),pos=SN.map((_,i)=>nP(i));
  ED.forEach(([fi,ti])=>{
    const f=pos[fi],t=pos[ti],cr=svcs[fi].status===2||svcs[ti].status===2,dg=svcs[fi].status===1||svcs[ti].status===1;
    if(!cr&&!dg){const now=Date.now()/1000;for(let p=0;p<4;p++){const t2=((now*0.35+p*0.25)%1);ctx.beginPath();ctx.arc(f.x+(t.x-f.x)*t2,f.y+(t.y-f.y)*t2,3,0,Math.PI*2);ctx.fillStyle=`rgba(0,212,255,${0.6*Math.sin(t2*Math.PI)})`;ctx.fill();}}
    ctx.strokeStyle=cr?'rgba(255,51,85,.4)':dg?'rgba(251,191,36,.3)':'rgba(0,212,255,.14)';
    ctx.lineWidth=cr?2.5:1.5;if(cr)ctx.setLineDash([7,5]);else ctx.setLineDash([]);
    ctx.beginPath();ctx.moveTo(f.x,f.y);ctx.lineTo(t.x,t.y);ctx.stroke();ctx.setLineDash([]);
  });
  pos.forEach(({x,y},i)=>{
    const s=svcs[i],sc=STC[s.status],fc=FC[s.active_failure]||'#1e3050';
    const glo=s.status===2?0.7+0.3*Math.sin(s.pp):s.status===1?0.35+0.2*Math.sin(s.pp):0;
    if(glo>0){const g=ctx.createRadialGradient(x,y,R,x,y,R+38+glo*18);g.addColorStop(0,`${sc}${Math.round(glo*55).toString(16).padStart(2,'0')}`);g.addColorStop(1,'transparent');ctx.beginPath();ctx.arc(x,y,R+38+glo*18,0,Math.PI*2);ctx.fillStyle=g;ctx.fill();}
    if(s.healFlash>0){const hg=ctx.createRadialGradient(x,y,0,x,y,R+75);hg.addColorStop(0,`rgba(0,255,136,${s.healFlash*0.35})`);hg.addColorStop(1,'transparent');ctx.beginPath();ctx.arc(x,y,R+75,0,Math.PI*2);ctx.fillStyle=hg;ctx.fill();}
    ctx.shadowColor=sc;ctx.shadowBlur=22+glo*14;
    const bg=ctx.createRadialGradient(x-R*.35,y-R*.35,0,x,y,R);
    if(s.status===0){bg.addColorStop(0,'#162a1c');bg.addColorStop(1,'#0a1810');}
    else if(s.status===1){bg.addColorStop(0,'#2a1e08');bg.addColorStop(1,'#1a1206');}
    else{bg.addColorStop(0,'#2a0a10');bg.addColorStop(1,'#1a0608');}
    ctx.beginPath();ctx.arc(x,y,R,0,Math.PI*2);ctx.fillStyle=bg;ctx.fill();ctx.shadowBlur=0;
    ctx.strokeStyle=sc;ctx.lineWidth=s.status===2?3:2;ctx.beginPath();ctx.arc(x,y,R,0,Math.PI*2);ctx.stroke();
    if(s.active_failure>0){ctx.strokeStyle=fc;ctx.lineWidth=4;ctx.beginPath();ctx.arc(x,y,R+8,-Math.PI/2,Math.PI*1.1,false);ctx.stroke();}
    const fs=Math.round(R*0.22);ctx.textAlign='center';ctx.textBaseline='middle';
    const icon=s.status===0?'●':s.status===1?'▲':'✖';
    ctx.fillStyle=sc;ctx.font=`bold ${fs}px JetBrains Mono,monospace`;ctx.fillText(icon,x,y-R*0.32);
    ctx.fillStyle='#e8f0ff';ctx.font=`bold ${fs}px Syne,sans-serif`;ctx.fillText(SL[i],x,y-R*0.04);
    ctx.fillStyle=sc;ctx.font=`${Math.round(R*0.16)}px JetBrains Mono,monospace`;ctx.fillText(STL[s.status],x,y+R*0.24);
    if(s.active_failure>0){ctx.fillStyle=fc;ctx.font=`${Math.round(R*0.14)}px JetBrains Mono,monospace`;ctx.fillText(FL[s.active_failure],x,y+R*0.44);}
    const bw=R*1.8,bh=5,bx=x-bw/2,by=y+R+14;
    ctx.fillStyle='rgba(10,18,32,.95)';ctx.fillRect(bx-3,by-3,bw+6,bh*2+10);
    ctx.fillStyle='#182035';ctx.fillRect(bx,by,bw,bh);
    const cC=s.cpu>80?'#ff3355':s.cpu>60?'#fbbf24':'#00d4ff';
    ctx.fillStyle=cC;ctx.fillRect(bx,by,Math.min(s.cpu/100,1)*bw,bh);
    ctx.fillStyle='#182035';ctx.fillRect(bx,by+bh+3,bw,bh);
    ctx.fillStyle='#ff3355';ctx.fillRect(bx,by+bh+3,Math.min(s.error_rate,1)*bw,bh);
    ctx.fillStyle='#5a7499';ctx.font=`${Math.round(R*0.13)}px JetBrains Mono,monospace`;
    ctx.textAlign='left';ctx.fillText(`CPU ${s.cpu.toFixed(0)}%`,bx,by-3);
    ctx.textAlign='right';ctx.fillText(`ERR ${(s.error_rate*100).toFixed(0)}%`,bx+bw,by-3);
    ctx.textAlign='center';
  });
  const hl=svcs.filter(s=>s.status===0).length,hp=(hl/5*100).toFixed(0),hcl=hl>=4?'#00ff88':hl>=2?'#fbbf24':'#ff3355';
  ctx.textAlign='right';ctx.fillStyle='#5a7499';ctx.font='bold 11px JetBrains Mono,monospace';ctx.fillText('CLUSTER HEALTH',W-16,H-34);
  ctx.fillStyle=hcl;ctx.font='bold 32px JetBrains Mono,monospace';ctx.fillText(`${hp}%`,W-16,H-8);
  ctx.textAlign='left';ctx.font='10px JetBrains Mono,monospace';
  [['#00ff88','● HEALTHY'],['#fbbf24','▲ DEGRADED'],['#ff3355','✖ CRASHED']].forEach(([c,l],i)=>{ctx.fillStyle=c;ctx.fillText(l,14,H-12-i*17);});
  ctx.textAlign='center';
}

function drawHealthChart(){
  const W=hc.width,H=hc.height;if(!W||!H)return;
  hctx.clearRect(0,0,W,H);if(healthHist.length<2)return;
  const data=healthHist.slice(-150),dx=W/Math.max(data.length-1,1);
  hctx.strokeStyle='rgba(24,32,53,.9)';hctx.lineWidth=1;
  [0.25,0.5,0.75,1].forEach(v=>{const y=H-v*H*.9;hctx.beginPath();hctx.moveTo(0,y);hctx.lineTo(W,y);hctx.stroke();hctx.fillStyle='rgba(90,116,153,.5)';hctx.font='8px JetBrains Mono,monospace';hctx.fillText(`${(v*100).toFixed(0)}%`,2,y-3);});
  const task=parseInt(document.getElementById('taskSel').value),thr=task===1?0.6:task===2?0.45:0.3;
  hctx.strokeStyle='rgba(251,191,36,.4)';hctx.lineWidth=1;hctx.setLineDash([4,4]);
  const ty=H-thr*H*.9;hctx.beginPath();hctx.moveTo(0,ty);hctx.lineTo(W,ty);hctx.stroke();hctx.setLineDash([]);
  hctx.fillStyle='rgba(251,191,36,.6)';hctx.font='8px JetBrains Mono,monospace';hctx.fillText(`thresh ${thr}`,W-70,ty-3);
  const grd=hctx.createLinearGradient(0,0,0,H);grd.addColorStop(0,'rgba(0,212,255,.18)');grd.addColorStop(1,'rgba(0,212,255,0)');
  hctx.beginPath();hctx.moveTo(0,H);data.forEach((v,i)=>hctx.lineTo(i*dx,H-v*H*.9));hctx.lineTo((data.length-1)*dx,H);hctx.closePath();hctx.fillStyle=grd;hctx.fill();
  hctx.strokeStyle='#00d4ff';hctx.lineWidth=2.5;hctx.lineJoin='round';hctx.beginPath();
  data.forEach((v,i)=>i===0?hctx.moveTo(0,H-v*H*.9):hctx.lineTo(i*dx,H-v*H*.9));hctx.stroke();
  const lv=data[data.length-1];hctx.beginPath();hctx.arc((data.length-1)*dx,H-lv*H*.9,5,0,Math.PI*2);hctx.fillStyle='#00d4ff';hctx.fill();
}

function buildCards(){
  const w=document.getElementById('svcCards');w.innerHTML='';
  SN.forEach((n,i)=>{
    const d=document.createElement('div');d.className='sc-card sh';d.id='card-'+n;
    d.innerHTML=`<div class="sct"><span class="scn">${SL[i]}</span><span class="sctag sth" id="ctag-${n}">HEALTHY</span></div>
    <div class="scm">${[['cpu','CPU','#00d4ff'],['memory','MEM','#00ff88'],['error_rate','ERR','#ff3355'],['latency','LAT','#a855f7']].map(([id,lbl,c])=>`
    <div class="mw"><div class="ml"><span>${lbl}</span><span id="${n}-${id}-v">—</span></div>
    <div class="mt"><div class="mf" id="${n}-${id}-b" style="width:0%;background:${c}"></div></div></div>`).join('')}
    </div><div class="sft" id="ftag-${n}"></div>`;
    w.appendChild(d);
  });
}
buildCards();

function updateCards(){
  SN.forEach(n=>{
    const s=svcs[SN.indexOf(n)];
    document.getElementById('card-'+n).className='sc-card '+['sh','sd','sc'][s.status];
    const ct=document.getElementById('ctag-'+n);ct.className='sctag '+['sth','std','stc'][s.status];ct.textContent=STL[s.status];
    const cC=s.cpu>80?'#ff3355':s.cpu>60?'#fbbf24':'#00d4ff',mC=s.memory>85?'#ff3355':s.memory>60?'#fbbf24':'#00ff88';
    const setM=(id,val,bw,bc)=>{document.getElementById(`${n}-${id}-v`).textContent=val;const b=document.getElementById(`${n}-${id}-b`);b.style.width=bw+'%';b.style.background=bc;};
    setM('cpu',s.cpu.toFixed(0)+'%',Math.min(s.cpu,100),cC);
    setM('memory',s.memory.toFixed(0)+'%',Math.min(s.memory,100),mC);
    setM('error_rate',(s.error_rate*100).toFixed(0)+'%',Math.min(s.error_rate*100,100),'#ff3355');
    setM('latency',s.latency.toFixed(0)+'ms',Math.min(s.latency/2000*100,100),'#a855f7');
    const ft=document.getElementById('ftag-'+n);
    if(s.active_failure>0){ft.textContent='⚠ '+FL[s.active_failure];ft.classList.add('show');}else ft.classList.remove('show');
  });
}

async function startEp(){
  const task=parseInt(document.getElementById('taskSel').value);
  let resp;
  try{resp=await fetch('/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task})});}
  catch(e){alert('Cannot reach server: '+e.message);return;}
  if(!resp.ok){alert('Reset failed: '+await resp.text());return;}
  const d=await resp.json();maxSteps=d.max_steps||200;
  running=true;totalReward=0;stepCount=0;healthHist=[];
  document.getElementById('logList').innerHTML='';document.getElementById('alertBanner').classList.remove('show');
  document.getElementById('btnSt').style.display='none';document.getElementById('btnSp').style.display='block';
  setStatus('run',`Task ${task} · step 0/${maxSteps}`);scheduleStep();
}

function scheduleStep(){if(!running)return;stepTimer=setTimeout(doStep,stepDelay);}

function stopEp(){
  running=false;if(stepTimer){clearTimeout(stepTimer);stepTimer=null;}
  document.getElementById('btnSt').style.display='block';document.getElementById('btnSp').style.display='none';
  setStatus('done','Episode stopped');
}

async function doStep(){
  if(!running)return;
  let resp;
  try{resp=await fetch('/step',{method:'POST',headers:{'Content-Type':'application/json'},body:'{"action":null}'});}
  catch(e){stopEp();return;}
  if(!resp.ok){stopEp();return;}
  const d=await resp.json();
  Object.entries(d.state).forEach(([name,svc])=>{
    const idx=SN.indexOf(name);if(idx<0)return;
    const prev=svcs[idx],wasDown=prev.status===2,nowOk=svc.status===0;
    prev.status=svc.status;prev.cpu=svc.cpu;prev.memory=svc.memory;
    prev.error_rate=svc.error_rate;prev.latency=svc.latency;prev.active_failure=svc.active_failure;
    if(wasDown&&nowOk)prev.healFlash=1;
  });
  updateCards();
  stepCount++;totalReward+=d.reward;const avg=totalReward/stepCount;
  healthHist.push(d.cluster_health);drawHealthChart();
  const sn=document.getElementById('scoreNum');sn.textContent=avg.toFixed(4);sn.style.color=avg>=0.7?'#00ff88':avg>=0.45?'#fbbf24':'#ff3355';
  document.getElementById('scoreSub').textContent=`Step ${d.step}/${maxSteps} · ${(d.cluster_health*100).toFixed(0)}%`;
  document.getElementById('scoreBar').style.cssText=`width:${Math.min(avg*100,100)}%;background:${avg>=0.7?'#00ff88':avg>=0.45?'#fbbf24':'#ff3355'}`;
  const tv=document.getElementById('tbScore');tv.textContent=avg.toFixed(3);tv.style.color=avg>=0.7?'#00ff88':avg>=0.45?'#fbbf24':'#ff3355';
  document.getElementById('actName').textContent=d.action_name;
  document.getElementById('actRes').textContent=d.action_result+(d.failure_event?' ⚠ failure injected':'');
  if(d.failure_event){const ab=document.getElementById('alertBanner');ab.textContent=`⚠ FAILURE: ${d.failure_event}`;ab.classList.add('show');setTimeout(()=>ab.classList.remove('show'),4500);}
  addLog(d.step,d.action_name,d.action_result,d.reward);
  setStatus('run',`Step ${d.step}/${maxSteps} · ${(d.cluster_health*100).toFixed(0)}% healthy`);
  if(d.done||d.step>=maxSteps){
    const task=document.getElementById('taskSel').value,thr=task==='1'?0.6:task==='2'?0.45:0.3;
    stopEp();setStatus(avg>=thr?'done':'fail',`Done · Score ${avg.toFixed(4)} ${avg>=thr?'✅ PASS':'❌ FAIL'}`);return;
  }
  scheduleStep();
}

function addLog(step,action,result,reward){
  const list=document.getElementById('logList');
  const cls=action.includes('restart')?'la-r':action.includes('scale')?'la-s':action.includes('rollback')?'la-ro':action.includes('reroute')?'la-re':'la-n';
  const rcls=reward>=0.8?'rh':reward>=0.5?'rm':'rl';
  const el=document.createElement('div');el.className='lr';
  el.innerHTML=`<span class="ls">#${step}</span><span class="la ${cls}">${action}</span><span class="lres">${result}</span><span class="lrw ${rcls}">${reward.toFixed(3)}</span>`;
  list.insertBefore(el,list.firstChild);while(list.children.length>9)list.removeChild(list.lastChild);
}

function setStatus(state,text){
  const dot=document.getElementById('sdot');dot.className='sdot '+(state==='run'?'run':state==='done'?'done':state==='fail'?'fail':'');
  document.getElementById('stext').textContent=text;
}

async function inject(scenario){
  if(!running){alert('Start an episode first, then inject!');return;}
  try{
    const r=await fetch('/inject',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({scenario})});
    if(!r.ok){alert('Inject failed');return;}
    const d=await r.json();
    const ab=document.getElementById('alertBanner');ab.textContent=`💥 ${d.description}`;ab.classList.add('show');setTimeout(()=>ab.classList.remove('show'),6000);
    const sr=await fetch('/state');if(sr.ok){const sd=await sr.json();Object.entries(sd.services).forEach(([name,svc])=>{const idx=SN.indexOf(name);if(idx<0)return;svcs[idx].status=svc.status;svcs[idx].cpu=svc.cpu;svcs[idx].memory=svc.memory;svcs[idx].error_rate=svc.error_rate;svcs[idx].latency=svc.latency;svcs[idx].active_failure=svc.active_failure;});updateCards();}
  }catch(e){alert('Server error: '+e);}
}

function updSpd(v){stepDelay=parseInt(v);document.getElementById('speedVal').textContent=v+'ms';}
</script>
</body>
</html>"""


# ── SOLVE UI PAGE ─────────────────────────────────────────────────────────────
SOLVE_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CloudHealRL — Custom Solver</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
:root{--bg:#030812;--s:#080f1e;--s2:#0c1528;--b:#1a2540;--b2:#243050;
  --cyan:#00d4ff;--green:#00ff88;--amber:#fbbf24;--red:#ff4466;--purple:#a855f7;
  --text:#dde8f5;--muted:#6b82a8;--dim:#2a3a55}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:'Syne',sans-serif;min-height:100vh}
body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(0,212,255,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,212,255,.03) 1px,transparent 1px);background-size:48px 48px;pointer-events:none;z-index:0}
.w{position:relative;z-index:1;max-width:1000px;margin:0 auto;padding:32px 24px}
.back{font-family:'Space Mono',monospace;font-size:10px;color:var(--muted);text-decoration:none;display:inline-flex;align-items:center;gap:6px;margin-bottom:24px;transition:color .2s}
.back:hover{color:var(--cyan)}
h1{font-size:40px;font-weight:800;letter-spacing:-1.5px;margin-bottom:8px}h1 .c{color:var(--cyan)}
.sub{color:var(--muted);font-size:14px;margin-bottom:24px;line-height:1.6;max-width:600px}
.arch{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:12px 16px;background:var(--s);border:1px solid rgba(0,212,255,.15);border-radius:10px;margin-bottom:24px;font-family:'Space Mono',monospace;font-size:10px}
.as{padding:6px 12px;border-radius:7px;font-weight:700;text-align:center;font-size:9px;line-height:1.4}
.as-i{background:rgba(0,212,255,.1);border:1px solid rgba(0,212,255,.25);color:var(--cyan)}
.as-r{background:rgba(0,255,136,.1);border:1px solid rgba(0,255,136,.25);color:var(--green)}
.as-g{background:rgba(251,191,36,.1);border:1px solid rgba(251,191,36,.25);color:var(--amber)}
.as-o{background:rgba(168,85,247,.1);border:1px solid rgba(168,85,247,.25);color:var(--purple)}
.aa{color:var(--dim);font-size:16px}
.tabs{display:flex;gap:0;margin-bottom:16px;border:1px solid var(--b2);border-radius:8px;overflow:hidden}
.tab{flex:1;padding:10px;border:none;background:var(--s2);color:var(--muted);font-family:'Space Mono',monospace;font-size:11px;cursor:pointer;transition:all .2s;text-align:center}
.tab.active{background:rgba(0,212,255,.1);color:var(--cyan)}.tab:hover:not(.active){background:var(--bg)}
.fl{font-size:8px;color:var(--dim);letter-spacing:1.5px;text-transform:uppercase;font-family:'Space Mono',monospace;margin-bottom:6px}
textarea{background:var(--s2);border:1px solid var(--b2);border-radius:9px;color:var(--text);padding:12px;font-family:'Space Mono',monospace;font-size:12px;width:100%;cursor:text;resize:vertical;min-height:80px;line-height:1.6;transition:border-color .2s}
textarea:focus{outline:none;border-color:var(--cyan)}
.text-note{font-size:11px;color:var(--muted);font-family:'Space Mono',monospace;margin-top:8px;padding:10px 14px;background:var(--s2);border-radius:7px;border-left:3px solid var(--amber)}
.examples{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px;align-items:center}
.exlbl{font-family:'Space Mono',monospace;font-size:9px;color:var(--dim);letter-spacing:1.5px}
.ex{padding:5px 10px;border-radius:6px;border:1px solid var(--b2);background:var(--s);color:var(--muted);font-family:'Space Mono',monospace;font-size:9px;cursor:pointer;transition:all .2s}
.ex:hover{border-color:var(--cyan);color:var(--cyan)}
.prow{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:20px;align-items:center}
.plbl{font-family:'Space Mono',monospace;font-size:9px;color:var(--dim);letter-spacing:2px;margin-right:4px}
.pb{padding:7px 14px;border-radius:7px;border:1px solid var(--b2);background:var(--s);color:var(--muted);font-family:'Space Mono',monospace;font-size:10px;cursor:pointer;transition:all .2s}
.pb:hover{border-color:var(--cyan);color:var(--cyan)}
.svc-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-bottom:20px}
.svc-cfg{background:var(--s);border:1px solid var(--b);border-radius:12px;padding:14px;border-left:3px solid var(--b);transition:border-left-color .3s}
.svc-title{font-family:'Space Mono',monospace;font-size:9px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:var(--cyan);margin-bottom:10px}
select{background:var(--s2);border:1px solid var(--b2);border-radius:7px;color:var(--text);padding:7px 9px;font-family:'Syne',sans-serif;font-size:11px;width:100%;cursor:pointer;appearance:none}
select:focus{outline:none;border-color:var(--cyan)}
.solve-btn{width:100%;padding:16px;border-radius:12px;background:linear-gradient(135deg,var(--green),#00bb66);color:#030812;font-family:'Syne',sans-serif;font-size:16px;font-weight:700;border:none;cursor:pointer;transition:all .2s;margin-bottom:28px}
.solve-btn:hover:not(:disabled){transform:translateY(-2px);box-shadow:0 10px 32px rgba(0,255,136,.35)}
.solve-btn:disabled{opacity:.5;cursor:not-allowed;transform:none}
.result{display:none;animation:rIn .4s ease}
@keyframes rIn{from{opacity:0;transform:translateY(10px)}to{opacity:1}}
.res-hero{background:var(--s);border:2px solid rgba(0,255,136,.35);border-radius:14px;padding:24px 28px;margin-bottom:16px}
.rh-badge{display:inline-flex;align-items:center;gap:7px;padding:4px 12px;border-radius:20px;background:rgba(0,255,136,.1);border:1px solid rgba(0,255,136,.25);font-family:'Space Mono',monospace;font-size:8px;color:var(--green);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:12px}
.rh-dot{width:5px;height:5px;border-radius:50%;background:var(--green);animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.rh-lbl{font-family:'Space Mono',monospace;font-size:9px;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-bottom:6px}
.rh-act{font-family:'Space Mono',monospace;font-size:32px;font-weight:700;color:var(--green);margin-bottom:8px;letter-spacing:1px}
.rh-id{font-size:11px;color:var(--muted);font-family:'Space Mono',monospace}
.triage-box{background:var(--s);border:1px solid var(--b);border-radius:12px;padding:18px;margin-bottom:16px}
.tb-title{font-family:'Space Mono',monospace;font-size:9px;color:var(--dim);letter-spacing:2px;text-transform:uppercase;margin-bottom:12px}
.triage-step{display:flex;align-items:center;gap:12px;padding:10px 12px;background:var(--s2);border:1px solid var(--b2);border-radius:8px;margin-bottom:8px}
.triage-step:last-child{margin-bottom:0}
.ts-num{font-family:'Space Mono',monospace;font-size:11px;font-weight:700;color:var(--muted);width:24px;height:24px;border-radius:50%;background:var(--b2);display:flex;align-items:center;justify-content:center;flex-shrink:0}
.ts-action{font-family:'Space Mono',monospace;font-size:12px;font-weight:700;flex:1}
.ts-result{font-size:10px;color:var(--muted)}
.ts-health{margin-left:auto;font-family:'Space Mono',monospace;font-size:11px;font-weight:700;flex-shrink:0;min-width:48px;text-align:right}
.groq-box{background:var(--s);border:1px solid var(--b2);border-radius:12px;padding:18px;margin-bottom:16px}
.gb-header{display:flex;align-items:center;gap:10px;margin-bottom:12px}
.gb-badge{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:20px;background:rgba(251,191,36,.08);border:1px solid rgba(251,191,36,.2);font-family:'Space Mono',monospace;font-size:8px;color:var(--amber);letter-spacing:1px;text-transform:uppercase}
.gb-title{font-size:13px;font-weight:700}
.gb-text{font-size:14px;line-height:1.75;color:var(--text)}
.gb-src{font-family:'Space Mono',monospace;font-size:9px;color:var(--dim);margin-top:8px}
.parsed-box{background:var(--s2);border:1px solid rgba(0,212,255,.2);border-radius:10px;padding:12px 16px;margin-bottom:16px;font-family:'Space Mono',monospace;font-size:11px}
.pb-title{font-size:8px;color:var(--cyan);letter-spacing:2px;text-transform:uppercase;margin-bottom:8px}
.cluster-box{background:var(--s);border:1px solid var(--b);border-radius:12px;padding:16px;margin-bottom:16px}
.sum-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px}
.sum-card{background:var(--s2);border:1px solid var(--b);border-radius:9px;padding:12px}
.sm-name{font-family:'Space Mono',monospace;font-size:8px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:5px}
.sm-st{font-size:12px;font-weight:700;margin-bottom:3px}
</style>
</head>
<body>
<div class="w">
  <a href="/" class="back">← Home</a>
  <h1>Custom <span class="c">Solver</span></h1>
  <p class="sub">Describe your cluster problem in plain English or configure services manually. The RL agent computes the full healing sequence. Groq explains why.</p>
  <div class="arch">
    <div class="as as-i">Your Input<br>(Text or Config)</div><span class="aa">→</span>
    <div class="as as-r">🧠 RL Agent<br>Triage Plan</div><span class="aa">→</span>
    <div class="as as-g">✨ Groq LLM<br>Explains</div><span class="aa">→</span>
    <div class="as as-o">Plain English<br>+ Priority Order</div>
  </div>
  <div class="tabs">
    <button class="tab active" onclick="switchTab('text',this)">💬 Text Input (Natural Language)</button>
    <button class="tab"        onclick="switchTab('config',this)">🎛 Service Config (Dropdowns)</button>
  </div>
  <div id="tab-text">
    <div class="fl">Describe the cluster problem in plain English</div>
    <textarea id="text-problem" rows="3" placeholder="e.g. database is completely down and payment service is throwing errors, auth CPU is very high"></textarea>
    <div class="text-note">Groq parses this → RL agent decides triage → Groq explains. Works best with GROQ_API_KEY set.</div>
    <div class="examples">
      <span class="exlbl">EXAMPLES:</span>
      <button class="ex" onclick="setEx('database is crashed and payment is throwing errors')">DB crash + payment errors</button>
      <button class="ex" onclick="setEx('auth CPU is maxed out and api gateway is slow')">Auth CPU storm</button>
      <button class="ex" onclick="setEx('bad deployment pushed to payment service')">Bad deploy</button>
      <button class="ex" onclick="setEx('notification service network partition')">Network split</button>
      <button class="ex" onclick="setEx('database crashed payment degraded auth has high CPU')">Full meltdown</button>
    </div>
  </div>
  <div id="tab-config" style="display:none">
    <div class="prow">
      <span class="plbl">PRESETS:</span>
      <button class="pb" onclick="preset('healthy')">✅ All Healthy</button>
      <button class="pb" onclick="preset('db_crash')">💀 DB Crash</button>
      <button class="pb" onclick="preset('cpu_storm')">🔥 CPU Storm</button>
      <button class="pb" onclick="preset('bad_deploy')">💢 Bad Deploy</button>
      <button class="pb" onclick="preset('net_split')">🌐 Net Split</button>
      <button class="pb" onclick="preset('meltdown')">💣 Meltdown</button>
    </div>
    <div class="svc-grid" id="svcGrid"></div>
  </div>
  <button class="solve-btn" id="solveBtn" onclick="solve()">⚡ Ask the RL Agent — What Should I Do?</button>
  <div class="result" id="result">
    <div class="parsed-box" id="parsedBox" style="display:none">
      <div class="pb-title">Groq parsed your text into:</div>
      <div id="parsedItems"></div>
    </div>
    <div class="res-hero">
      <div class="rh-badge"><span class="rh-dot"></span>PyTorch RL Agent Decision</div>
      <div class="rh-lbl">Highest Priority Action</div>
      <div class="rh-act" id="resAction">—</div>
      <div class="rh-id" id="resId"></div>
    </div>
    <div class="triage-box" id="triageBox" style="display:none">
      <div class="tb-title">Full Triage Plan — Complete Healing Sequence</div>
      <div id="triageList"></div>
    </div>
    <div class="groq-box">
      <div class="gb-header"><div class="gb-badge">✨ Groq LLM</div><div class="gb-title">Why this healing sequence?</div></div>
      <div class="gb-text" id="resExp">—</div>
      <div class="gb-src" id="resSrc"></div>
    </div>
    <div class="cluster-box">
      <div class="fl" style="margin-bottom:10px">Cluster State Evaluated</div>
      <div class="sum-grid" id="resSummary"></div>
    </div>
  </div>
</div>
<script>
const SVCS=['auth','api_gateway','payment','database','notification'];
const SVCL=['Auth','API Gateway','Payment','Database','Notification'];
const FOPTS=[['0','✅ None — Healthy'],['1','🔥 CPU Spike → Degraded'],['2','💧 Memory Leak → Degraded'],['3','💢 Bad Deploy → Degraded'],['4','🌐 Net Split → Degraded'],['5','💀 Hard Crash → Crashed']];
const SFF={0:0,1:1,2:1,3:1,4:1,5:2};
const SCOLS=['var(--green)','var(--amber)','var(--red)'];
const SLBLS=['HEALTHY','DEGRADED','CRASHED'];
const ACOLS={restart:'#00d4ff',scale_up:'#fbbf24',rollback:'#00ff88',reroute:'#a855f7'};
const PRESETS={healthy:{},db_crash:{database:{f:5}},cpu_storm:{auth:{f:1},api_gateway:{f:1}},bad_deploy:{payment:{f:3}},net_split:{notification:{f:4}},meltdown:{database:{f:5},payment:{f:3},auth:{f:1}}};
const grid=document.getElementById('svcGrid');
SVCS.forEach((svc,i)=>{
  const d=document.createElement('div');d.className='svc-cfg';d.id='cfg-'+svc;
  d.innerHTML=`<div class="svc-title">${SVCL[i]}</div><div class="fl">Failure Type</div>
  <select id="${svc}-f" onchange="onFC('${svc}',this.value)">${FOPTS.map(([v,l])=>`<option value="${v}">${l}</option>`).join('')}</select>`;
  grid.appendChild(d);
});
function onFC(svc,f){const st=SFF[parseInt(f)];document.getElementById('cfg-'+svc).style.borderLeft=`3px solid ${st===2?'var(--red)':st===1?'var(--amber)':'var(--b)'}`;}
function switchTab(name,btn){document.getElementById('tab-text').style.display=name==='text'?'block':'none';document.getElementById('tab-config').style.display=name==='config'?'block':'none';document.querySelectorAll('.tabs .tab').forEach(t=>t.classList.remove('active'));btn.classList.add('active');}
function setEx(text){document.getElementById('text-problem').value=text;}
function preset(name){SVCS.forEach(s=>{document.getElementById(s+'-f').value=0;onFC(s,0);});Object.entries(PRESETS[name]||{}).forEach(([svc,v])=>{if(v.f!==undefined){document.getElementById(svc+'-f').value=v.f;onFC(svc,v.f);}});document.getElementById('result').style.display='none';}
async function solve(){
  const btn=document.getElementById('solveBtn');btn.disabled=true;btn.textContent='⏳ Asking RL agent + Groq...';
  const useText=document.getElementById('tab-text').style.display!=='none';
  let body={};
  if(useText){const txt=document.getElementById('text-problem').value.trim();if(!txt){alert('Please enter a problem description.');btn.disabled=false;btn.textContent='⚡ Ask the RL Agent — What Should I Do?';return;}body={text_input:txt};}
  else{const services={};SVCS.forEach(svc=>{const f=parseInt(document.getElementById(svc+'-f').value);services[svc]={status:SFF[f],active_failure:f};});body={services};}
  try{
    const r=await fetch('/solve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    document.getElementById('result').style.display='block';
    const pb=document.getElementById('parsedBox');
    if(useText&&d.cluster_summary){pb.style.display='block';const cs=d.cluster_summary;const all=[...(cs.crashed||[]).map(n=>({n,st:2})),...(cs.degraded||[]).map(n=>({n,st:1})),...(cs.healthy||[]).map(n=>({n,st:0}))];document.getElementById('parsedItems').innerHTML=all.map(({n,st})=>`<div style="color:${SCOLS[st]}">${n.replace(/_/g,' ')}: ${SLBLS[st]}</div>`).join('');}
    else pb.style.display='none';
    const rl=d.rl_decision||{};document.getElementById('resAction').textContent=rl.action_name||'do_nothing';
    document.getElementById('resId').textContent=`Action ${rl.action_id??'—'} of 21 · Result: ${rl.result||'—'} · Agent: ${rl.agent||'—'}`;
    const tp=d.triage_plan||[];
    if(tp.length>0){document.getElementById('triageBox').style.display='block';document.getElementById('triageList').innerHTML=tp.map(p=>{const ak=Object.keys(ACOLS).find(k=>p.action.includes(k));const ac=ak?ACOLS[ak]:'#dde8f5';const hc=p.health_after>=80?'var(--green)':p.health_after>=40?'var(--amber)':'var(--red)';return`<div class="triage-step"><div class="ts-num">${p.step}</div><div class="ts-action" style="color:${ac}">${p.action}</div><div class="ts-result">${p.result}</div><div class="ts-health" style="color:${hc}">${p.health_after}%</div></div>`;}).join('');}
    else document.getElementById('triageBox').style.display='none';
    document.getElementById('resExp').textContent=d.explanation||'No explanation available.';
    document.getElementById('resSrc').textContent=`Source: ${d.explanation_source||'—'} · Health: ${d.cluster_summary?.health_pct??'?'}%`;
    if(d.cluster_summary){const cs=d.cluster_summary;const sm={};(cs.healthy||[]).forEach(n=>sm[n]=0);(cs.degraded||[]).forEach(n=>sm[n]=1);(cs.crashed||[]).forEach(n=>sm[n]=2);document.getElementById('resSummary').innerHTML=SVCS.map((svc,i)=>{const st=sm[svc]??0;return`<div class="sum-card"><div class="sm-name">${SVCL[i]}</div><div class="sm-st" style="color:${SCOLS[st]}">${SLBLS[st]}</div></div>`;}).join('');}
    document.getElementById('result').scrollIntoView({behavior:'smooth'});
  }catch(e){alert('Error: '+e.message);}
  btn.disabled=false;btn.textContent='⚡ Ask the RL Agent — What Should I Do?';
}
</script>
</body>
</html>"""


# ── API DOCS PAGE ─────────────────────────────────────────────────────────────
DOCS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CloudHealRL — API Docs</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
:root{--bg:#030812;--s:#080f1e;--s2:#0c1528;--b:#1a2540;--b2:#243050;--cyan:#00d4ff;--green:#00ff88;--amber:#fbbf24;--red:#ff4466;--text:#dde8f5;--muted:#6b82a8;--dim:#2a3a55}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:'Syne',sans-serif;min-height:100vh;display:grid;grid-template-columns:200px 1fr}
.sb{background:var(--s);border-right:1px solid var(--b);padding:16px 0;position:sticky;top:0;height:100vh;overflow-y:auto}
.sb-logo{padding:0 12px 14px;border-bottom:1px solid var(--b);margin-bottom:10px}
.sb-logo h2{font-size:14px;font-weight:800}.sb-logo .c{color:var(--cyan)}.sb-logo .g{color:var(--green)}
.sb-logo p{font-size:9px;color:var(--muted);font-family:'Space Mono',monospace;margin-top:2px}
.ns{padding:0 6px;margin-bottom:4px}
.nsl{font-family:'Space Mono',monospace;font-size:8px;color:var(--dim);letter-spacing:2.5px;text-transform:uppercase;padding:4px 6px;display:block}
.ni{display:flex;align-items:center;gap:6px;padding:6px;border-radius:5px;cursor:pointer;border:none;background:none;width:100%;text-align:left;font-family:'Syne',sans-serif;font-size:12px;color:var(--muted);transition:all .15s}
.ni:hover{background:var(--s2);color:var(--text)}
.nm{font-family:'Space Mono',monospace;font-size:8px;font-weight:700;padding:2px 5px;border-radius:3px;flex-shrink:0;min-width:32px;text-align:center}
.nmg{background:rgba(0,212,255,.1);color:var(--cyan)}.nmp{background:rgba(0,255,136,.1);color:var(--green)}
.nb{display:block;margin:12px 6px 0;padding:7px;border-radius:7px;border:1px solid var(--b2);color:var(--muted);font-family:'Space Mono',monospace;font-size:10px;text-align:center;text-decoration:none;transition:all .2s}
.nb:hover{color:var(--cyan);border-color:var(--cyan)}
.mc{padding:28px 32px;max-width:800px}
h1{font-size:28px;font-weight:800;letter-spacing:-1px;margin-bottom:8px}.c{color:var(--cyan)}
.desc{font-size:13px;color:var(--muted);margin-bottom:28px;line-height:1.6}
.ep{background:var(--s);border:1px solid var(--b);border-radius:12px;margin-bottom:10px;overflow:hidden}
.ep.open{border-color:rgba(0,212,255,.2)}
.eph{display:flex;align-items:center;gap:10px;padding:13px 16px;cursor:pointer;user-select:none;transition:background .15s}
.eph:hover{background:rgba(255,255,255,.02)}
.mb{font-family:'Space Mono',monospace;font-size:9px;font-weight:700;padding:3px 8px;border-radius:5px;flex-shrink:0;min-width:40px;text-align:center}
.mb-g{background:rgba(0,212,255,.12);color:var(--cyan);border:1px solid rgba(0,212,255,.25)}
.mb-p{background:rgba(0,255,136,.12);color:var(--green);border:1px solid rgba(0,255,136,.25)}
.ep-path{font-family:'Space Mono',monospace;font-size:13px;font-weight:700}
.ep-sum{font-size:11px;color:var(--muted);margin-left:auto}
.ep-chev{color:var(--dim);font-size:11px;margin-left:8px;transition:transform .2s;flex-shrink:0}
.ep.open .ep-chev{transform:rotate(180deg)}
.ep-body{display:none;border-top:1px solid var(--b);padding:16px}
.ep.open .ep-body{display:block}
.ep-desc{font-size:12px;color:var(--muted);line-height:1.65;margin-bottom:12px}
.tl{font-family:'Space Mono',monospace;font-size:8px;color:var(--dim);letter-spacing:2px;text-transform:uppercase;margin-bottom:8px}
textarea,select{background:var(--bg);border:1px solid var(--b2);border-radius:7px;color:var(--text);padding:8px 10px;font-family:'Space Mono',monospace;font-size:11px;width:100%;margin-bottom:8px;appearance:none}
textarea:focus,select:focus{outline:none;border-color:var(--cyan)}textarea{resize:vertical;min-height:60px;line-height:1.5}
.rb{display:inline-flex;align-items:center;gap:8px;padding:9px 18px;border-radius:8px;background:linear-gradient(135deg,var(--cyan),#0088ff);color:#030812;font-family:'Syne',sans-serif;font-size:12px;font-weight:700;border:none;cursor:pointer;transition:all .2s}
.rb:hover{transform:translateY(-1px);box-shadow:0 6px 20px rgba(0,212,255,.3)}.rb:disabled{opacity:.5;cursor:not-allowed;transform:none}
.rb-g{background:linear-gradient(135deg,var(--green),#00bb66);color:#030812}.rb-full{width:100%;justify-content:center;margin-bottom:8px}
.res{display:none;margin-top:10px}.res.show{display:block}
.rs-st{display:flex;align-items:center;gap:8px;margin-bottom:6px;font-family:'Space Mono',monospace;font-size:11px}
.ok{color:var(--green)}.er{color:var(--red)}
pre{background:var(--bg);border:1px solid var(--b);border-radius:8px;padding:12px;font-family:'Space Mono',monospace;font-size:11px;overflow-x:auto;line-height:1.6;color:#a8c4e0;white-space:pre-wrap;word-break:break-all;max-height:300px;overflow-y:auto}
</style>
</head>
<body>
<aside class="sb">
  <div class="sb-logo"><h2>Cloud<span class="c">Heal</span><span class="g">RL</span></h2><p>API Reference v1.0</p></div>
  <div class="ns">
    <span class="nsl">Pages</span>
    <button class="ni" onclick="go('ep-sim')"><span class="nm nmg">GET</span>/simulation</button>
    <button class="ni" onclick="go('ep-demo')"><span class="nm nmg">GET</span>/demo</button>
    <button class="ni" onclick="go('ep-sui')"><span class="nm nmg">GET</span>/solve-ui</button>
  </div>
  <div class="ns">
    <span class="nsl">Episode</span>
    <button class="ni" onclick="go('ep-reset')"><span class="nm nmp">POST</span>/reset</button>
    <button class="ni" onclick="go('ep-step')"><span class="nm nmp">POST</span>/step</button>
    <button class="ni" onclick="go('ep-state')"><span class="nm nmg">GET</span>/state</button>
    <button class="ni" onclick="go('ep-log')"><span class="nm nmg">GET</span>/action-log</button>
  </div>
  <div class="ns">
    <span class="nsl">Injection</span>
    <button class="ni" onclick="go('ep-inject')"><span class="nm nmp">POST</span>/inject</button>
  </div>
  <div class="ns">
    <span class="nsl">RL + Groq</span>
    <button class="ni" onclick="go('ep-solve')"><span class="nm nmp">POST</span>/solve ★</button>
    <button class="ni" onclick="go('ep-sa')"><span class="nm nmp">POST</span>/solve-action</button>
    <button class="ni" onclick="go('ep-health')"><span class="nm nmg">GET</span>/health</button>
  </div>
  <a href="/" class="nb">← Back to Home</a>
</aside>
<main class="mc">
  <h1>API <span class="c">Reference</span></h1>
  <p class="desc">Full REST API. The key endpoint is <code style="color:var(--cyan);font-family:'Space Mono',monospace">/solve</code> — send text or service state, get RL triage plan + Groq explanation.</p>

  <div class="ep open" id="ep-solve">
    <div class="eph" onclick="tog(this)"><span class="mb mb-p">POST</span><span class="ep-path">/solve</span><span class="ep-sum">RL triage + Groq explanation ★</span><span class="ep-chev">▼</span></div>
    <div class="ep-body">
      <p class="ep-desc">The key endpoint. Text or service states → RL agent computes full triage plan → Groq explains in plain English.</p>
      <div class="tl">Option A — Text Input</div>
      <textarea id="st" rows="2" placeholder="database is down and payment is throwing errors"></textarea>
      <button class="rb rb-g rb-full" id="btn-st" onclick="runST()">🤖 Text → RL Agent → Groq</button>
      <div class="tl">Option B — Service State</div>
      <textarea id="sj" rows="3">{"services": {"database": {"status": 2, "active_failure": 5}, "payment": {"status": 1, "active_failure": 3}}}</textarea>
      <button class="rb rb-g rb-full" id="btn-sj" onclick="runSJ()">🤖 Config → RL Agent → Groq</button>
      <div class="res" id="solve-res"><div class="rs-st" id="solve-st"></div><pre id="solve-out"></pre></div>
    </div>
  </div>

  <div class="ep" id="ep-reset">
    <div class="eph" onclick="tog(this)"><span class="mb mb-p">POST</span><span class="ep-path">/reset</span><span class="ep-sum">Start new episode</span><span class="ep-chev">▼</span></div>
    <div class="ep-body">
      <p class="ep-desc">Start a fresh episode. Required before /step or /inject.</p>
      <div class="tl">Task</div>
      <select id="rt"><option value="1">1 — Easy</option><option value="2">2 — Medium</option><option value="3">3 — Hard</option></select>
      <button class="rb" onclick="runReset()">▶ POST /reset</button>
      <div class="res" id="reset-res"><div class="rs-st" id="reset-st"></div><pre id="reset-out"></pre></div>
    </div>
  </div>

  <div class="ep" id="ep-step">
    <div class="eph" onclick="tog(this)"><span class="mb mb-p">POST</span><span class="ep-path">/step</span><span class="ep-sum">Agent takes one action</span><span class="ep-chev">▼</span></div>
    <div class="ep-body">
      <p class="ep-desc">RL agent observes cluster and selects optimal healing action. Returns reward [0–1].</p>
      <button class="rb" onclick="runStep()">▶ POST /step</button>
      <div class="res" id="step-res"><div class="rs-st" id="step-st"></div><pre id="step-out"></pre></div>
    </div>
  </div>

  <div class="ep" id="ep-inject">
    <div class="eph" onclick="tog(this)"><span class="mb mb-p">POST</span><span class="ep-path">/inject</span><span class="ep-sum">Inject failure scenario 💥</span><span class="ep-chev">▼</span></div>
    <div class="ep-body">
      <p class="ep-desc">Inject a failure mid-episode. Requires an active episode (/reset first).</p>
      <div class="tl">Scenario</div>
      <select id="is"><option value="database_crash">💀 database_crash</option><option value="cpu_storm">🔥 cpu_storm</option><option value="bad_deployment">💢 bad_deployment</option><option value="network_split">🌐 network_split</option><option value="full_meltdown">💣 full_meltdown</option></select>
      <button class="rb" style="background:linear-gradient(135deg,var(--red),#cc0033);color:#fff;width:100%;justify-content:center" onclick="runInject()">💥 POST /inject</button>
      <div class="res" id="inject-res"><div class="rs-st" id="inject-st"></div><pre id="inject-out"></pre></div>
    </div>
  </div>

  <div class="ep" id="ep-state">
    <div class="eph" onclick="tog(this)"><span class="mb mb-g">GET</span><span class="ep-path">/state</span><span class="ep-sum">Current cluster metrics</span><span class="ep-chev">▼</span></div>
    <div class="ep-body"><p class="ep-desc">All 5 service metrics in real time.</p><button class="rb" onclick="rg('/state','state')">▶ GET /state</button><div class="res" id="state-res"><div class="rs-st" id="state-st"></div><pre id="state-out"></pre></div></div>
  </div>

  <div class="ep" id="ep-log">
    <div class="eph" onclick="tog(this)"><span class="mb mb-g">GET</span><span class="ep-path">/action-log</span><span class="ep-sum">Full agent history</span><span class="ep-chev">▼</span></div>
    <div class="ep-body"><p class="ep-desc">Every action taken this episode.</p><button class="rb" onclick="rg('/action-log','log')">▶ GET /action-log</button><div class="res" id="log-res"><div class="rs-st" id="log-st"></div><pre id="log-out"></pre></div></div>
  </div>

  <div class="ep" id="ep-sa">
    <div class="eph" onclick="tog(this)"><span class="mb mb-p">POST</span><span class="ep-path">/solve-action</span><span class="ep-sum">Raw obs → RL action (no LLM)</span><span class="ep-chev">▼</span></div>
    <div class="ep-body">
      <p class="ep-desc">Send 30-float obs vector → RL agent picks action. Pure RL, no Groq.</p>
      <div class="tl">obs — 30 floats (DB crashed example)</div>
      <textarea id="sa-o" rows="2">[0.2,0.3,0.0,0.025,0.0,0.0,0.2,0.3,0.0,0.025,0.0,0.0,0.2,0.3,0.0,0.025,0.0,0.0,1.0,0.95,1.0,1.0,1.0,1.0,0.2,0.3,0.0,0.025,0.0,0.0]</textarea>
      <button class="rb" onclick="runSA()">▶ POST /solve-action</button>
      <div class="res" id="sa-res"><div class="rs-st" id="sa-st"></div><pre id="sa-out"></pre></div>
    </div>
  </div>

  <div class="ep" id="ep-health">
    <div class="eph" onclick="tog(this)"><span class="mb mb-g">GET</span><span class="ep-path">/health</span><span class="ep-sum">Server + Groq status</span><span class="ep-chev">▼</span></div>
    <div class="ep-body"><button class="rb" onclick="rg('/health','health')">▶ GET /health</button><div class="res" id="health-res"><div class="rs-st" id="health-st"></div><pre id="health-out"></pre></div></div>
  </div>

  <div class="ep" id="ep-sim"><div class="eph" onclick="tog(this)"><span class="mb mb-g">GET</span><span class="ep-path">/simulation</span><span class="ep-sum">Live animated cluster ⚡</span><span class="ep-chev">▼</span></div><div class="ep-body"><p class="ep-desc">Canvas simulation with glowing nodes, particle flow, real-time bars.</p><button class="rb" onclick="window.open('/simulation','_blank')">⚡ Open Simulation</button></div></div>
  <div class="ep" id="ep-demo"><div class="eph" onclick="tog(this)"><span class="mb mb-g">GET</span><span class="ep-path">/demo</span><span class="ep-sum">Self-contained demo report 🎬</span><span class="ep-chev">▼</span></div><div class="ep-body"><p class="ep-desc">DB crash → agent heals → full HTML report. No setup needed.</p><button class="rb" onclick="window.open('/demo','_blank')">🎬 Open Demo</button></div></div>
  <div class="ep" id="ep-sui"><div class="eph" onclick="tog(this)"><span class="mb mb-g">GET</span><span class="ep-path">/solve-ui</span><span class="ep-sum">Interactive solver 🤖</span><span class="ep-chev">▼</span></div><div class="ep-body"><p class="ep-desc">Full UI — text or dropdowns, RL plan + Groq explanation.</p><button class="rb" onclick="window.open('/solve-ui','_blank')">🤖 Open Solver</button></div></div>
</main>
<script>
function tog(h){h.parentElement.classList.toggle('open');}
function go(id){const el=document.getElementById(id);if(el){el.classList.add('open');el.scrollIntoView({behavior:'smooth',block:'start'});}}
function show(p,status,data,ok){
  const box=document.getElementById(p+'-res'),st=document.getElementById(p+'-st'),op=document.getElementById(p+'-out');
  box.classList.add('show');st.className='rs-st '+(ok?'ok':'er');st.textContent=ok?'200 OK':`${status} Error`;
  if(op)op.textContent=JSON.stringify(data,null,2);
}
async function runST(){const btn=document.getElementById('btn-st');btn.disabled=true;btn.textContent='⏳...';const txt=document.getElementById('st').value.trim();if(!txt){alert('Enter a description');btn.disabled=false;btn.textContent='🤖 Text → RL Agent → Groq';return;}try{const r=await fetch('/solve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text_input:txt})});show('solve',r.status,await r.json(),r.ok);}catch(e){show('solve',0,{error:e.message},false);}btn.disabled=false;btn.textContent='🤖 Text → RL Agent → Groq';}
async function runSJ(){const btn=document.getElementById('btn-sj');btn.disabled=true;btn.textContent='⏳...';try{const body=JSON.parse(document.getElementById('sj').value);const r=await fetch('/solve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});show('solve',r.status,await r.json(),r.ok);}catch(e){show('solve',0,{error:e.message},false);}btn.disabled=false;btn.textContent='🤖 Config → RL Agent → Groq';}
async function runReset(){try{const r=await fetch('/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task:parseInt(document.getElementById('rt').value)})});show('reset',r.status,await r.json(),r.ok);}catch(e){show('reset',0,{error:e.message},false);}}
async function runStep(){try{const r=await fetch('/step',{method:'POST',headers:{'Content-Type':'application/json'},body:'{"action":null}'});show('step',r.status,await r.json(),r.ok);}catch(e){show('step',0,{error:e.message},false);}}
async function rg(path,p){try{const r=await fetch(path);show(p,r.status,await r.json(),r.ok);}catch(e){show(p,0,{error:e.message},false);}}
async function runInject(){const s=document.getElementById('is').value;try{const r=await fetch('/inject',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({scenario:s})});show('inject',r.status,await r.json(),r.ok);}catch(e){show('inject',0,{error:e.message},false);}}
async function runSA(){try{const obs=JSON.parse(document.getElementById('sa-o').value);const r=await fetch('/solve-action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({obs})});show('sa',r.status,await r.json(),r.ok);}catch(e){show('sa',0,{error:e.message},false);}}
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# DEMO REPORT
# ─────────────────────────────────────────────────────────────────────────────
def _demo_html(before_h, crash_h, after_h, fs, agent_type, elapsed, step_log, recovery):
    sc  = {"healthy": "#00ff88", "degraded": "#fbbf24", "crashed": "#ff4466"}
    pc  = "#00ff88" if fs >= 0.45 else "#ff4466"
    svc_rows = "".join(
        f'<tr><td style="font-family:monospace;color:#dde8f5">{s}</td>'
        f'<td style="color:{sc.get(r["before"],"#6b82a8")};text-transform:uppercase">{r["before"]}</td>'
        f'<td style="color:#00ff88;text-transform:uppercase">{r["after"]}</td>'
        f'<td style="text-align:center">{"✅" if r["healed"] else "—"}</td></tr>'
        for s, r in recovery.items()
    )
    step_rows = "".join(
        f'<tr><td style="font-family:monospace;color:#6b82a8;font-size:11px">{s["step"]:03d}</td>'
        f'<td style="font-family:monospace;font-size:11px;color:#dde8f5">{s["action"]}</td>'
        f'<td style="font-size:11px;color:#6b82a8">{s["result"]}</td>'
        f'<td style="font-family:monospace;font-size:11px;color:{"#00ff88" if s["reward"]>=0.8 else "#fbbf24" if s["reward"]>=0.5 else "#ff4466"}">{s["reward"]:.3f}</td>'
        f'<td style="font-family:monospace;font-size:11px;color:#00d4ff">{s["cluster_health"]:.1f}%</td></tr>'
        for s in step_log[:30]
    )
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>CloudHealRL Demo</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@700;800&display=swap" rel="stylesheet">
<style>*{{margin:0;padding:0;box-sizing:border-box}}body{{background:#030812;color:#dde8f5;font-family:'Syne',sans-serif;padding:40px 24px;min-height:100vh}}
body::before{{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(0,212,255,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,212,255,.03) 1px,transparent 1px);background-size:48px 48px;pointer-events:none}}
.w{{max-width:920px;margin:0 auto;position:relative;z-index:1}}
h1{{font-size:48px;font-weight:800;letter-spacing:-2px;margin-bottom:6px}}h1 .c{{color:#00d4ff}}
.meta{{font-size:11px;color:#6b82a8;font-family:'Space Mono',monospace;margin-bottom:32px}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:32px}}
.stat{{background:#080f1e;border:1px solid #1a2540;border-radius:12px;padding:16px}}
.sn{{font-family:'Space Mono',monospace;font-size:24px;font-weight:700;margin-bottom:4px}}
.sl{{font-size:9px;color:#6b82a8;text-transform:uppercase;letter-spacing:1.5px;font-family:'Space Mono',monospace}}
.sec{{margin-bottom:24px}}.st{{font-family:'Space Mono',monospace;font-size:9px;color:#2a3a55;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px}}
table{{width:100%;border-collapse:collapse;background:#080f1e;border-radius:12px;overflow:hidden}}
th{{font-family:'Space Mono',monospace;font-size:8px;color:#2a3a55;text-transform:uppercase;letter-spacing:1px;text-align:left;padding:9px 12px;border-bottom:1px solid #1a2540}}
td{{padding:9px 12px;border-bottom:1px solid #0c1528}}tr:last-child td{{border-bottom:none}}tr:hover td{{background:#0c1528}}
.back{{display:inline-block;margin-bottom:20px;color:#6b82a8;text-decoration:none;font-family:'Space Mono',monospace;font-size:10px}}.back:hover{{color:#00d4ff}}</style></head>
<body><div class="w">
  <a href="/" class="back">← Home</a>
  <h1>Demo <span class="c">Report</span></h1>
  <div class="meta">database_crash scenario · Agent: {agent_type} · {elapsed}ms</div>
  <div class="stats">
    <div class="stat"><div class="sn" style="color:#6b82a8">{before_h*100:.0f}%</div><div class="sl">Before crash</div></div>
    <div class="stat"><div class="sn" style="color:#ff4466">{crash_h*100:.0f}%</div><div class="sl">After crash</div></div>
    <div class="stat"><div class="sn" style="color:#00ff88">{after_h*100:.0f}%</div><div class="sl">After healing</div></div>
    <div class="stat"><div class="sn" style="color:{pc}">{fs:.4f}</div><div class="sl">Score {"✅" if fs>=0.45 else "❌"}</div></div>
    <div class="stat"><div class="sn" style="color:#00d4ff">{len(step_log)}</div><div class="sl">Steps</div></div>
  </div>
  <div class="sec"><div class="st">Service Recovery</div>
    <table><thead><tr><th>Service</th><th>After Crash</th><th>After Healing</th><th>Healed?</th></tr></thead><tbody>{svc_rows}</tbody></table></div>
  <div class="sec"><div class="st">Step Log</div>
    <table><thead><tr><th>Step</th><th>Action</th><th>Result</th><th>Reward</th><th>Health</th></tr></thead><tbody>{step_rows}</tbody></table></div>
</div></body></html>"""


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def landing(): return LANDING_HTML

@app.get("/simulation", response_class=HTMLResponse)
def simulation(): return SIMULATION_HTML

@app.get("/solve-ui", response_class=HTMLResponse)
def solve_ui(): return SOLVE_UI_HTML

@app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
def docs(): return DOCS_HTML

@app.get("/health")
def health():
    agent  = _load_agent()
    groq   = _groq_available()
    return {
        "status":          "ok",
        "agent_type":      "PPO (PyTorch)" if agent.is_trained() else "Heuristic",
        "agent_trained":   agent.is_trained(),
        "episode_active":  _env is not None,
        "groq_configured": groq,
        "groq_model":      os.getenv("MODEL_NAME", "llama-3.3-70b-versatile") if groq else None,
        "timestamp":       round(time.time(), 1),
    }

@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"id":1,"difficulty":"Easy","failure_interval":20,"cascade":False,"max_steps":200,"threshold":0.60},
            {"id":2,"difficulty":"Medium","failure_interval":12,"cascade":True,"max_steps":200,"threshold":0.45},
            {"id":3,"difficulty":"Hard","failure_interval":7,"cascade":True,"multi_fail":True,"max_steps":200,"threshold":0.30},
        ],
        "scenarios":     {k: v["description"] for k, v in SCENARIOS.items()},
        "action_space":  {str(k): v for k, v in ACTION_NAMES.items()},
    }

@app.post("/reset")
def reset_episode(req: ResetRequest):
    global _env
    task = req.task
    if req.difficulty:
        task = {"easy":1,"medium":2,"hard":3}.get(req.difficulty.lower(), 1)
    _env = CloudHealEnv(task=task, max_steps=200)
    obs, _ = _env.reset()
    _load_agent()
    return {
        "message":     f"Episode started — Task {task}",
        "task":        task,
        "observation": obs.tolist(),
        "state":       _env.get_state_dict(),
        "step":        0,
        "max_steps":   _env.max_steps,
    }

@app.post("/reset/task{task_id}")
def reset_task(task_id: int):
    return reset_episode(ResetRequest(task=task_id))

@app.post("/step")
def step_episode(req: StepRequest):
    env   = _require_env()
    agent = _load_agent()
    if env.step_count >= env.max_steps:
        raise HTTPException(400, "Episode finished — call /reset.")
    obs = env._get_obs()
    if req.action is not None:
        action = req.action
    else:
        try:
            action, _ = agent.predict(obs, deterministic=True)
        except TypeError:
            action, _ = agent.predict(obs)
        action = int(action)
    obs, reward, done, _, info = env.step(action)
    assert 0.0 <= reward <= 1.0, f"Reward out of range: {reward}"
    return {
        "step":           info["step"],
        "action":         action,
        "action_name":    info["action_name"],
        "action_result":  info["action_result"],
        "reward":         reward,
        "done":           done,
        "cluster_health": info["cluster_health"],
        "episode_reward": info["episode_reward"],
        "failure_event":  info["failure_event"],
        "observation":    obs.tolist(),
        "state":          info["services"],
        "score":          grade(env) if done else None,
    }

@app.get("/state")
def get_state():
    env = _require_env()
    return {
        "step":           env.step_count,
        "max_steps":      env.max_steps,
        "cluster_health": env._cluster_health_pct(),
        "episode_reward": round(env.episode_reward, 4),
        "services":       env.get_state_dict(),
        "score_so_far":   grade(env),
    }

@app.get("/action-log")
def action_log(last_n: int = 20):
    env = _require_env()
    log = env.get_action_log()
    return {"total_steps": len(log), "showing": min(last_n, len(log)), "log": log[-last_n:]}

@app.post("/inject")
def inject_scenario(req: InjectRequest):
    env    = _require_env()
    result = env.inject_scenario(req.scenario)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return {**result, "state": env.get_state_dict(), "cluster_health": env._cluster_health_pct()}


# ── THE KEY ENDPOINT ──────────────────────────────────────────────────────────
@app.post("/solve")
def solve(req: SolveRequest):
    """
    Flow: Input → RL Agent (PPO/Heuristic) → Groq explanation

    Option A: {"text_input": "database is down and payment errors"}
    Option B: {"services": {"database": {"status": 2, "active_failure": 5}}}
    Legacy:   {"state": {...}}  also accepted
    """
    services_input = dict(req.services or req.state or {})
    text_input     = req.text_input

    # Parse text via Groq (or keyword fallback) if no explicit services given
    if text_input and not services_input:
        parsed = _parse_text_to_services(text_input)
        if parsed:
            services_input = parsed
        else:
            # Absolute last resort — treat entire input as "all healthy" so
            # the RL agent still runs and returns a do_nothing result
            print(f"[solve] Could not parse text, using empty state")
            services_input = {}

    # Build snapshot + compute triage plan from RL agent
    triage_plan = _compute_triage_plan(services_input)

    # Get first action details
    snapshot = _build_env_from_services(services_input)
    agent    = _load_agent()
    obs      = snapshot._get_obs()
    try:
        action_int, _ = agent.predict(obs, deterministic=True)
    except TypeError:
        action_int, _ = agent.predict(obs)
    action_int    = int(action_int)
    action_name   = ACTION_NAMES.get(action_int, "do_nothing")
    _, _, _, _, info = snapshot.step(action_int)

    # Build state dict for Groq explanation
    state_dict = {
        n: {"status": svc.status, "cpu": svc.cpu, "memory": svc.memory,
            "error_rate": svc.error_rate, "latency": svc.latency,
            "active_failure": svc.active_failure}
        for n, svc in _build_env_from_services(services_input).services.items()
    }

    # Groq explains (falls back to rule-based if Groq unavailable)
    explanation_result = _groq_explain_triage(state_dict, triage_plan, text_input)

    crashed  = [n for n, s in state_dict.items() if s["status"] == 2]
    degraded = [n for n, s in state_dict.items() if s["status"] == 1]
    healthy  = [n for n, s in state_dict.items() if s["status"] == 0]

    return {
        "cluster_summary": {
            "healthy":    healthy,
            "degraded":   degraded,
            "crashed":    crashed,
            "health_pct": round(len(healthy) / len(SERVICES) * 100, 1),
        },
        "rl_decision": {
            "action_id":   action_int,
            "action_name": action_name,
            "result":      info["action_result"],
            "agent":       "PPO (PyTorch)" if agent.is_trained() else "Heuristic",
        },
        "triage_plan":        triage_plan,
        "explanation":        explanation_result["explanation"],
        "explanation_source": explanation_result["source"],
        "source":             explanation_result["source"],  # legacy field
    }

@app.post("/solve-action")
def solve_action(req: SolveActionRequest):
    """30-float obs vector → RL agent action (no LLM)."""
    import numpy as np
    agent  = _load_agent()
    obs    = np.array(req.obs, dtype=np.float32)
    try:
        action, _ = agent.predict(obs, deterministic=True)
    except TypeError:
        action, _ = agent.predict(obs)
    action = int(action)
    return {"action_id": action, "action_name": ACTION_NAMES.get(action, "unknown")}

@app.get("/demo", response_class=HTMLResponse)
def run_demo():
    global _env
    _env = CloudHealEnv(task=2, max_steps=200)
    _env.reset()
    agent    = _load_agent()
    before_h = _env._cluster_health_pct()

    _env.inject_scenario("database_crash")
    crash_state = _env.get_state_dict()
    crash_h     = _env._cluster_health_pct()
    _env._next_failure_step = 99999
    obs = _env._get_obs()

    step_log = []
    t0 = time.time()
    for _ in range(60):
        try:
            action, _ = agent.predict(obs, deterministic=True)
        except TypeError:
            action, _ = agent.predict(obs)
        obs, reward, done, _, info = _env.step(int(action))
        step_log.append({
            "step":           info["step"],
            "action":         info["action_name"],
            "result":         info["action_result"],
            "reward":         round(reward, 4),
            "cluster_health": round(info["cluster_health"] * 100, 1),
        })
        if info["cluster_health"] >= 1.0 and len(step_log) >= 3:
            break

    after_state = _env.get_state_dict()
    after_h     = _env._cluster_health_pct()
    fs          = grade(_env)
    sm          = {0: "healthy", 1: "degraded", 2: "crashed"}
    recovery    = {
        svc: {
            "before": sm[crash_state[svc]["status"]],
            "after":  sm[after_state[svc]["status"]],
            "healed": crash_state[svc]["status"] > after_state[svc]["status"],
        }
        for svc in SERVICES
    }
    agent_type = "PPO (PyTorch)" if agent.is_trained() else "Heuristic"
    elapsed    = round((time.time() - t0) * 1000, 1)
    return _demo_html(before_h, crash_h, after_h, fs, agent_type, elapsed, step_log, recovery)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    port    = int(os.environ.get("PORT", 7860))
    groq_ok = _groq_available()
    print(f"""
╔══════════════════════════════════════════════════════╗
║  CloudHealRL — Server ready                          ║
║  Home:        http://0.0.0.0:{port}                    ║
║  Simulation:  http://0.0.0.0:{port}/simulation         ║
║  Demo:        http://0.0.0.0:{port}/demo                ║
║  Solver:      http://0.0.0.0:{port}/solve-ui            ║
║  API Docs:    http://0.0.0.0:{port}/docs                ║
╚══════════════════════════════════════════════════════╝
""")
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False, log_level="info")


if __name__ == "__main__":
    main()