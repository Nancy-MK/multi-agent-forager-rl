"""
analysis.py

Loads the raw per-seed, per-episode collision counts produced by
run_experiment.py and reports the statistics behind the headline
collision-reduction finding, plus a simple behavioural-unpredictability
metric (episode-to-episode variance in collisions) for each policy type.

Usage:
    python src/analysis.py --results-dir results/
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def unpredictability_score(collisions: np.ndarray) -> float:
    """Higher variance in collisions across episodes = less predictable
    behaviour. Averaged across seeds."""
    per_seed_variance = collisions.var(axis=1)
    return float(per_seed_variance.mean())


def main():
    parser = argparse.ArgumentParser(description="Analyse independent vs shared Q-learning results.")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    independent = np.load(results_dir / "independent_collisions.npy")
    shared = np.load(results_dir / "shared_collisions.npy")

    independent_mean = independent.mean()
    shared_mean = shared.mean()
    reduction_pct = (independent_mean - shared_mean) / independent_mean * 100

    report = {
        "mean_collisions_independent": round(float(independent_mean), 3),
        "mean_collisions_shared": round(float(shared_mean), 3),
        "collision_reduction_pct": round(reduction_pct, 1),
        "unpredictability_independent": round(unpredictability_score(independent), 3),
        "unpredictability_shared": round(unpredictability_score(shared), 3),
    }
    print(json.dumps(report, indent=2))

    # Learning-curve style plot: rolling mean collisions per episode, both policies.
    window = 20
    ind_curve = np.convolve(independent.mean(axis=0), np.ones(window) / window, mode="valid")
    shared_curve = np.convolve(shared.mean(axis=0), np.ones(window) / window, mode="valid")

    plt.figure(figsize=(8, 5))
    plt.plot(ind_curve, label="Independent Q-tables")
    plt.plot(shared_curve, label="Shared Q-table")
    plt.xlabel("Episode")
    plt.ylabel(f"Collisions (rolling mean, window={window})")
    plt.title("Collision rate: independent vs shared policy")
    plt.legend()
    plt.tight_layout()

    out_path = results_dir / "collision_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")

    with open(results_dir / "analysis_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
