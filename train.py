"""
CloudHealRL Training Script
============================
Curriculum: Task1 → Task2 → Task3 → Mixed

Usage:
    python train.py                   # curriculum (recommended)
    python train.py --task 1
    python train.py --all --steps 150000
"""
import os, argparse, random
MODEL_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "cloudheal_ppo")
os.makedirs(MODEL_DIR, exist_ok=True)

def make_mixed_env():
    from environment import CloudHealEnv
    import gymnasium as gym
    class MixedEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self._env = CloudHealEnv(task=1)
            self.observation_space = self._env.observation_space
            self.action_space      = self._env.action_space
        def reset(self, seed=None, options=None):
            self._env = CloudHealEnv(task=random.choice([1,2,3]))
            return self._env.reset(seed=seed)
        def step(self, action): return self._env.step(action)
        def render(self): return self._env.render()
    return MixedEnv

def train(task=None, total_steps=100_000, verbose=1):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError:
        print("ERROR: pip install stable-baselines3"); return None
    from environment import CloudHealEnv
    print(f"\n{'='*55}\n  Training PPO — {'Mixed' if task is None else f'Task {task}'} | {total_steps:,} steps\n{'='*55}\n")
    env_fn = (lambda t=task: CloudHealEnv(task=t, max_steps=200)) if task else make_mixed_env()
    train_env = make_vec_env(env_fn, n_envs=8)
    if os.path.exists(MODEL_PATH + ".zip"):
        model = PPO.load(MODEL_PATH, env=train_env)
        print(f"  Continuing from checkpoint")
    else:
        model = PPO("MlpPolicy", train_env, learning_rate=3e-4, n_steps=1024, batch_size=256,
                    n_epochs=10, gamma=0.995, gae_lambda=0.95, clip_range=0.2, ent_coef=0.02,
                    vf_coef=0.5, max_grad_norm=0.5, verbose=verbose,
                    policy_kwargs=dict(net_arch=[dict(pi=[256,256], vf=[256,256])]))
    model.learn(total_timesteps=total_steps, progress_bar=True, reset_num_timesteps=False)
    model.save(MODEL_PATH)
    print(f"\n✅ Saved → {MODEL_PATH}.zip")
    train_env.close()
    return model

def curriculum_train(steps_per_stage=100_000):
    print("\n" + "="*55 + "\n  CURRICULUM TRAINING\n  Task1 → Task2 → Task3 → Mixed\n" + "="*55)
    for i, (task, label) in enumerate([(1,"Easy"),(2,"Cascade"),(3,"Hard"),(None,"Mixed")]):
        print(f"\n  Stage {i+1}/4: {label}")
        train(task=task, total_steps=steps_per_stage if task else steps_per_stage//2)
    print(f"\n✅ Curriculum complete → {MODEL_PATH}.zip")

def evaluate(task=1, n_episodes=5):
    from environment import CloudHealEnv, grade
    from agent import get_agent
    agent = get_agent()
    thresholds = {1:0.60, 2:0.45, 3:0.30}
    print(f"\nEvaluating {'PPO' if agent.is_trained() else 'Heuristic'} — Task {task}")
    scores = []
    for ep in range(n_episodes):
        env = CloudHealEnv(task=task, max_steps=200)
        obs, _ = env.reset(); done = False
        while not done:
            action, _ = agent.predict(obs)
            obs, reward, done, _, info = env.step(int(action))
            assert 0.0 <= reward <= 1.0
        s = grade(env); scores.append(s)
        print(f"  Ep {ep+1}: {s:.4f} {'✅' if s>=thresholds[task] else '❌'}")
    print(f"  Avg: {sum(scores)/len(scores):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",  type=int, default=None, choices=[1,2,3])
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--all",   action="store_true")
    parser.add_argument("--eval",  action="store_true")
    args = parser.parse_args()
    if args.all or not args.task:
        curriculum_train(steps_per_stage=args.steps)
        for t in [1,2,3]: evaluate(task=t, n_episodes=3)
    else:
        train(task=args.task, total_steps=args.steps)
        if args.eval: evaluate(task=args.task)