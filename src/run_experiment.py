"""
run_experiment.py

Runs the independent-vs-shared Q-table comparison across multiple random
seeds and episodes, and reports the collision-rate reduction from policy
coordination, the headline safety metric for this study.

Usage:
    python src/run_experiment.py --episodes 1000 --seeds 8
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import trange

from agents import IndependentQAgents, SharedQAgents
from environment import ACTIONS, ForagingEnv


def run_single(agent_cls, n_episodes, seed, max_steps=100):
    env = ForagingEnv(seed=seed)
    agents = agent_cls(n_agents=env.n_agents, seed=seed)

    collisions_per_episode = []
    for episode in trange(n_episodes, desc=f"{agent_cls.__name__} seed={seed}", leave=False):
        states = env.reset()
        for _ in range(max_steps):
            action_idxs = agents.act(states, episode)
            actions = [ACTIONS[i] for i in action_idxs]
            next_states, rewards, done, info = env.step(actions)
            agents.update(states, action_idxs, rewards, next_states)
            states = next_states
            if done:
                break
        collisions_per_episode.append(env.collisions_this_episode)

    return collisions_per_episode


def main():
    parser = argparse.ArgumentParser(description="Compare independent vs shared Q-learning agents.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    independent_results, shared_results = [], []

    for seed in range(args.seeds):
        independent_results.append(run_single(IndependentQAgents, args.episodes, seed))
        shared_results.append(run_single(SharedQAgents, args.episodes, seed))

    independent_mean = float(np.mean([np.mean(r) for r in independent_results]))
    shared_mean = float(np.mean([np.mean(r) for r in shared_results]))
    reduction_pct = (independent_mean - shared_mean) / independent_mean * 100

    summary = {
        "episodes_per_run": args.episodes,
        "n_seeds": args.seeds,
        "mean_collisions_independent": round(independent_mean, 3),
        "mean_collisions_shared": round(shared_mean, 3),
        "collision_reduction_pct": round(reduction_pct, 1),
    }
    print(json.dumps(summary, indent=2))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "independent_collisions.npy", np.array(independent_results))
    np.save(out_dir / "shared_collisions.npy", np.array(shared_results))
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nRaw results and summary saved to {out_dir}/")


if __name__ == "__main__":
    main()
