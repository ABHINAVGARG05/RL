import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from env.resource_env import ResourceAllocationEnv
from agents.dqn_agent import DQNAgent
from baselines.heuristics import FirstFitBaseline, BestFitBaseline, GreedyPriorityBaseline, RandomBaseline
from utils.logger import EpisodeLogger



CONFIG = {
    "n_machines": 4,
    "cpu_capacity": 16.0,
    "mem_capacity": 64.0,
    "max_jobs_per_ep": 200,

    "n_episodes": 1000,
    "eval_every": 100,
    "eval_episodes": 20,

    "lr": 1e-3,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 30_000,
    "batch_size": 64,
    "buffer_capacity": 50_000,
    "target_update_freq": 500,
    "hidden": 256,

    "save_path": "checkpoints/dqn_resource.pt",
}


def make_env(seed=None):
    return ResourceAllocationEnv(
        n_machines=CONFIG["n_machines"],
        cpu_capacity=CONFIG["cpu_capacity"],
        mem_capacity=CONFIG["mem_capacity"],
        max_jobs_per_ep=CONFIG["max_jobs_per_ep"],
        seed=seed,
    )


def evaluate_agent(agent: DQNAgent, n_episodes: int = 20) -> dict:
    env = make_env(seed=42)
    rewards, cpu_utils, mem_utils = [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(obs)
            obs, reward, done, _, _ = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)
        u = env.utilization()
        cpu_utils.append(u["cpu"])
        mem_utils.append(u["mem"])
    return {
        "reward_mean": np.mean(rewards),
        "reward_std": np.std(rewards),
        "cpu_util": np.mean(cpu_utils),
        "mem_util": np.mean(mem_utils),
    }


def evaluate_baseline(baseline, n_episodes: int = 50) -> dict:
    env = make_env(seed=42)
    rewards, cpu_utils, mem_utils = [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = baseline.select_action(obs, env)
            obs, reward, done, _, _ = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)
        u = env.utilization()
        cpu_utils.append(u["cpu"])
        mem_utils.append(u["mem"])
    return {
        "reward_mean": np.mean(rewards),
        "reward_std": np.std(rewards),
        "cpu_util": np.mean(cpu_utils),
        "mem_util": np.mean(mem_utils),
    }

def train():
    os.makedirs("checkpoints", exist_ok=True)
    env = make_env()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        lr=CONFIG["lr"],
        gamma=CONFIG["gamma"],
        epsilon_start=CONFIG["epsilon_start"],
        epsilon_end=CONFIG["epsilon_end"],
        epsilon_decay_steps=CONFIG["epsilon_decay_steps"],
        batch_size=CONFIG["batch_size"],
        buffer_capacity=CONFIG["buffer_capacity"],
        target_update_freq=CONFIG["target_update_freq"],
        hidden=CONFIG["hidden"],
    )

    logger = EpisodeLogger(print_every=50)
    ep_rewards = []
    eval_rewards = []
    eval_steps = []

    print(f"\n{'='*60}")
    print(f"  DQN Resource Allocation Agent")
    print(f"  Machines: {CONFIG['n_machines']}  |  Jobs/ep: {CONFIG['max_jobs_per_ep']}")
    print(f"  Obs dim: {obs_dim}  |  Actions: {n_actions}")
    print(f"  Device: {agent.device}")
    print(f"{'='*60}\n")

    for ep in range(1, CONFIG["n_episodes"] + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_loss = 0.0
        steps = 0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, _, info = env.step(action)
            agent.store(obs, action, reward, next_obs, done)
            loss = agent.update()
            obs = next_obs
            ep_reward += reward
            ep_loss += loss
            steps += 1

        avg_loss = ep_loss / steps
        ep_rewards.append(ep_reward)
        logger.log_episode(ep_reward, avg_loss, agent.epsilon)

        if ep % CONFIG["eval_every"] == 0:
            metrics = evaluate_agent(agent, CONFIG["eval_episodes"])
            eval_rewards.append(metrics["reward_mean"])
            eval_steps.append(ep)
            print(
                f"\n  [EVAL @ ep {ep}] "
                f"Reward={metrics['reward_mean']:.1f} ± {metrics['reward_std']:.1f}  "
                f"CPU util={metrics['cpu_util']:.1%}  MEM util={metrics['mem_util']:.1%}\n"
            )
            agent.save(CONFIG["save_path"])

    print("\n" + "="*60)
    print("  Final Benchmark vs Baselines")
    print("="*60)

    final_dqn = evaluate_agent(agent, 50)
    print(f"  DQN (ours)       : reward={final_dqn['reward_mean']:.2f}  cpu={final_dqn['cpu_util']:.1%}  mem={final_dqn['mem_util']:.1%}")

    baselines = [FirstFitBaseline(), BestFitBaseline(), GreedyPriorityBaseline(), RandomBaseline()]
    baseline_results = {}
    for bl in baselines:
        r = evaluate_baseline(bl)
        baseline_results[bl.name] = r
        print(f"  {bl.name:<16} : reward={r['reward_mean']:.2f}  cpu={r['cpu_util']:.1%}  mem={r['mem_util']:.1%}")

    _plot_results(ep_rewards, eval_steps, eval_rewards, final_dqn, baseline_results)
    print("\nTraining complete. Plots saved to training/plots/")


def _plot_results(ep_rewards, eval_steps, eval_rewards, dqn_result, baseline_results):
    os.makedirs("training/plots", exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("DQN Resource Allocation Agent", fontsize=14, fontweight="bold")

    ax = axes[0]
    window = 50
    smoothed = np.convolve(ep_rewards, np.ones(window) / window, mode="valid")
    ax.plot(ep_rewards, alpha=0.2, color="steelblue", label="Raw")
    ax.plot(range(window - 1, len(ep_rewards)), smoothed, color="steelblue", label=f"Smoothed ({window}ep)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Training Reward Curve")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(eval_steps, eval_rewards, marker="o", color="darkorange", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Eval Reward")
    ax.set_title("Evaluation Reward")
    ax.grid(alpha=0.3)

    ax = axes[2]
    names = ["DQN"] + list(baseline_results.keys())
    means = [dqn_result["reward_mean"]] + [v["reward_mean"] for v in baseline_results.values()]
    stds  = [dqn_result["reward_std"]]  + [v["reward_std"]  for v in baseline_results.values()]
    colors = ["#2196F3"] + ["#78909C"] * len(baseline_results)
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors, edgecolor="white")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("DQN vs Baselines")
    ax.grid(axis="y", alpha=0.3)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{mean:.0f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig("training/plots/results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → training/plots/results.png")


if __name__ == "__main__":
    train()