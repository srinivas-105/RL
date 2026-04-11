"""
CloudHealRL Inference Script
=============================
Hackathon evaluator entry point. Runs all 3 tasks, prints parseable scores.

Output format:
    TASK_1_SCORE: 0.XXXX
    TASK_2_SCORE: 0.XXXX
    TASK_3_SCORE: 0.XXXX
    OVERALL_SCORE: 0.XXXX
    PASS: True/False

Usage:
    python inference.py
    python inference.py --task 2 --episodes 5 --verbose
    python inference.py --start-server
"""
import argparse, os, sys, json, subprocess, time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from environment import CloudHealEnv, grade, SCENARIOS
from agent import get_agent

TASK_THRESHOLDS = {1: 0.60, 2: 0.45, 3: 0.30}
TASK_NAMES      = {1: "Easy", 2: "Medium (cascade)", 3: "Hard (multi-fail)"}


def clamp(v): return round(float(max(0.0, min(1.0, v))), 4)


def run_episode(agent, task, max_steps=200, verbose=False):
    env = CloudHealEnv(task=task, max_steps=max_steps)
    obs, _ = env.reset()
    done = False; step = 0
    print(f"[START] task={task}", flush=True)
    while not done:
        action, _ = agent.predict(obs)
        obs, reward, done, _, info = env.step(int(action))
        assert 0.0 <= reward <= 1.0
        step += 1
        if verbose and step % 20 == 0:
            print(f"  [STEP] step={step:3d} health={info['cluster_health']*100:.0f}% action={info['action_name']:30s} reward={reward:.4f}", flush=True)
    score = clamp(grade(env))
    print(f"[END] task={task} score={score:.4f} steps={step} status={'PASS' if score >= TASK_THRESHOLDS[task] else 'FAIL'}", flush=True)
    return {"task": task, "score": score, "episode_reward": clamp(env.episode_reward / max(step, 1)),
            "steps": step, "final_health": clamp(info["cluster_health"])}


def run_task(agent, task, n_episodes=3, verbose=False):
    print(f"\n{'─'*60}\n  TASK {task} | {TASK_NAMES[task]} | Threshold: {TASK_THRESHOLDS[task]}\n{'─'*60}")
    results = []
    for ep in range(1, n_episodes + 1):
        r = run_episode(agent, task=task, verbose=verbose)
        results.append(r)
        status = "✅ PASS" if r["score"] >= TASK_THRESHOLDS[task] else "❌ FAIL"
        print(f"  Episode {ep}/{n_episodes}: score={r['score']:.4f} health={r['final_health']*100:.0f}% {status}")
    avg = clamp(sum(r["score"] for r in results) / len(results))
    passed = avg >= TASK_THRESHOLDS[task]
    print(f"\n  Avg: {avg:.4f} | Threshold: {TASK_THRESHOLDS[task]} | {'✅ PASSED' if passed else '❌ FAILED'}")
    return {"task": task, "avg_score": avg, "threshold": TASK_THRESHOLDS[task], "passed": passed, "episodes": results}


def print_summary(task_results, overall, passed):
    print(f"\n{'='*60}\n  CLOUDHEALRL — FINAL RESULTS\n{'='*60}")
    for tr in task_results:
        bar = "█" * int(tr["avg_score"] * 30)
        print(f"  Task {tr['task']}: {bar:<30s} {tr['avg_score']:.4f} {'✅' if tr['passed'] else '❌'}")
    print(f"{'─'*60}")
    print(f"  Overall:  {'█'*int(overall*30):<30s} {overall:.4f} {'✅ PASS' if passed else '❌ FAIL'}")
    print(f"{'='*60}\n")
    for tr in task_results:
        print(f"TASK_{tr['task']}_SCORE: {tr['avg_score']:.4f}")
    print(f"OVERALL_SCORE: {overall:.4f}")
    print(f"PASS: {passed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",         type=int, default=None, choices=[1,2,3])
    parser.add_argument("--episodes",     type=int, default=3)
    parser.add_argument("--verbose",      action="store_true")
    parser.add_argument("--output-json",  type=str, default=None)
    parser.add_argument("--start-server", action="store_true")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  CloudHealRL — Autonomous Cloud Healing Agent")
    print("  Meta × PyTorch Hackathon")
    print("="*60)

    if args.start_server:
        srv = os.path.join(ROOT, "server", "app.py")
        if os.path.exists(srv):
            subprocess.Popen([sys.executable, srv], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2)
            print(f"  Server started at http://127.0.0.1:7860")

    agent = get_agent()
    print(f"\n  Agent: {'PPO (trained)' if agent.is_trained() else 'Heuristic (rule-based)'}")

    tasks = [args.task] if args.task else [1, 2, 3]
    task_results = []
    for task in tasks:
        task_results.append(run_task(agent, task=task, n_episodes=args.episodes, verbose=args.verbose))

    overall = clamp(sum(tr["avg_score"] for tr in task_results) / len(task_results))
    passed  = all(tr["passed"] for tr in task_results)
    print_summary(task_results, overall, passed)

    for tr in task_results:
        assert 0.0 <= tr["avg_score"] <= 1.0
    assert 0.0 <= overall <= 1.0

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"agent": str(agent.is_trained()), "task_results": task_results,
                       "overall_score": overall, "passed": passed}, f, indent=2)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())