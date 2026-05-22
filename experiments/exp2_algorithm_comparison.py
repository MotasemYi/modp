"""
Experiment 2: NSGA-II vs SPEA2 on User 1.
Each algorithm is run 5 times with different seeds.
Reports mean ± std hypervolume. Seed 42 is the representative run for plots.
"""
import gc
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.makedirs("results", exist_ok=True)

from db_loader import load_all_for_user
from chromosome import Chromosome
from problem import MODPProblem
from algorithms.nsga2_runner import run_nsga2
from algorithms.spea2_runner import run_spea2
from metrics import get_reference_point, compute_hypervolume, hypervolume_convergence
from visualize import plot_pareto_front, plot_hypervolume_convergence

USER_ID = 1
SEEDS = [42, 123, 456, 789, 1000]
PLOT_SEED = 42
ALGO_RUNNERS = {"NSGA-II": run_nsga2, "SPEA2": run_spea2}


def run_experiment2(n_gen=200, pop_size=100):
    print("=" * 60)
    print("Experiment 2: Algorithm Comparison (NSGA-II vs SPEA2, User 1, 5 seeds)")
    print("=" * 60)

    data = load_all_for_user(USER_ID)
    ch = Chromosome(data["food_ids"], data["nutrient_matrix"],
                    data["dri"], data["food_groups"],
                    data["food_preferences"])
    problem = MODPProblem(ch, data["foods_df"])

    # ── Phase 1: run all seeds for both algorithms ─────────────────────
    runs = {}   # {(algo_name, seed): res}

    for algo_name, runner in ALGO_RUNNERS.items():
        print(f"\n--- {algo_name} ---")
        for seed in SEEDS:
            verbose = (seed == PLOT_SEED)
            save_history = (seed == PLOT_SEED)
            print(f"  seed={seed}" + ("" if verbose else " (silent)"))
            res = runner(problem, n_gen=n_gen, pop_size=pop_size,
                         seed=seed, verbose=verbose, save_history=save_history)
            runs[(algo_name, seed)] = res
            gc.collect()

    # ── Phase 2: shared reference point from ALL fronts ────────────────
    all_F = [runs[(algo, s)].F for algo in ALGO_RUNNERS for s in SEEDS]
    ref_point = get_reference_point(all_F)
    print(f"\n  Shared reference point (all runs): {ref_point}")

    # ── Phase 3: compute HV for every run ──────────────────────────────
    hv = {(algo, s): compute_hypervolume(runs[(algo, s)].F, ref_point)
          for algo in ALGO_RUNNERS for s in SEEDS}

    # ── Phase 4: save per-run detail CSV ───────────────────────────────
    detail_rows = []
    for algo in ALGO_RUNNERS:
        for seed in SEEDS:
            F = runs[(algo, seed)].F
            detail_rows.append({
                "algorithm":  algo,
                "seed":       seed,
                "front_size": len(F),
                "hypervolume": round(hv[(algo, seed)], 4),
                "avg_preference": round(float(-F[:, 0].mean()), 3),
                "avg_cost":       round(float(F[:, 1].mean()), 3),
                "avg_co2":        round(float(F[:, 2].mean()), 3),
            })
    pd.DataFrame(detail_rows).to_csv(
        os.path.join("results", "exp2_all_runs.csv"), index=False
    )
    print("  Saved: results/exp2_all_runs.csv")

    # ── Phase 5: save representative (seed=42) Pareto fronts ───────────
    for algo in ALGO_RUNNERS:
        F = runs[(algo, PLOT_SEED)].F
        df = pd.DataFrame({
            "preference": -F[:, 0],
            "cost":        F[:, 1],
            "co2":         F[:, 2],
        })
        path = os.path.join("results", f"exp2_{algo.lower().replace('-','')}_pareto.csv")
        df.to_csv(path, index=False)
        print(f"  Saved representative front: {path} ({len(F)} solutions)")

    # ── Phase 6: print per-algorithm HV stats ──────────────────────────
    print()
    for algo in ALGO_RUNNERS:
        hvs = [hv[(algo, s)] for s in SEEDS]
        print(f"  {algo} HV per seed: " +
              "  ".join(f"s{s}={v:.0f}" for s, v in zip(SEEDS, hvs)))
        print(f"    mean={np.mean(hvs):.2f}  std={np.std(hvs):.2f}  "
              f"min={np.min(hvs):.2f}  max={np.max(hvs):.2f}")

    # ── Phase 7: summary CSV ───────────────────────────────────────────
    summary = []
    for algo in ALGO_RUNNERS:
        hvs = [hv[(algo, s)] for s in SEEDS]
        F42 = runs[(algo, PLOT_SEED)].F
        summary.append({
            "algorithm":  algo,
            "hv_mean":    round(np.mean(hvs), 4),
            "hv_std":     round(np.std(hvs),  4),
            "hv_min":     round(np.min(hvs),  4),
            "hv_max":     round(np.max(hvs),  4),
            **{f"hv_seed{s}": round(hv[(algo, s)], 4) for s in SEEDS},
            "front_size_seed42":     len(F42),
            "avg_preference_seed42": round(float(-F42[:, 0].mean()), 3),
            "avg_cost_seed42":       round(float(F42[:, 1].mean()), 3),
            "avg_co2_seed42":        round(float(F42[:, 2].mean()), 3),
        })
    pd.DataFrame(summary).to_csv(
        os.path.join("results", "exp2_summary.csv"), index=False
    )
    print("  Saved: results/exp2_summary.csv")

    # ── Phase 8: plots using seed=42 ───────────────────────────────────
    hv_curve_nsga2 = hypervolume_convergence(
        runs[("NSGA-II", PLOT_SEED)].history, ref_point)
    hv_curve_spea2 = hypervolume_convergence(
        runs[("SPEA2",  PLOT_SEED)].history, ref_point)
    plot_hypervolume_convergence(
        [hv_curve_nsga2, hv_curve_spea2],
        ["NSGA-II", "SPEA2"],
        f"Experiment 2 — HV Convergence: NSGA-II vs SPEA2 (seed={PLOT_SEED})",
        "exp2_hv_convergence",
    )
    plot_pareto_front(
        [runs[("NSGA-II", PLOT_SEED)].F, runs[("SPEA2", PLOT_SEED)].F],
        ["NSGA-II", "SPEA2"],
        f"Experiment 2 — Pareto Fronts: NSGA-II vs SPEA2 (seed={PLOT_SEED})",
        "exp2_pareto_comparison",
    )

    print("\nExperiment 2 complete.")
    return runs, ref_point


if __name__ == "__main__":
    run_experiment2()
