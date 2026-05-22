"""
Experiment 1: Run NSGA-II for User 1 (Isla Morris) and User 2 (Zoely Butler).
Each user is run 5 times with different seeds. Reports mean ± std hypervolume.
Seed 42 is the representative run used for Pareto front plots.
"""
import gc
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.makedirs("results", exist_ok=True)
os.makedirs(os.path.join("results", "figures"), exist_ok=True)

from db_loader import load_all_for_user
from chromosome import Chromosome
from problem import MODPProblem
from algorithms.nsga2_runner import run_nsga2
from metrics import get_reference_point, compute_hypervolume, hypervolume_convergence
from visualize import plot_pareto_front, plot_hypervolume_convergence

SEEDS = [42, 123, 456, 789, 1000]
PLOT_SEED = 42
USER_IDS = [1, 2]
USER_LABELS = {1: "User 1 (Isla)", 2: "User 2 (Zoely)"}


def run_experiment1(n_gen=200, pop_size=100):
    print("=" * 60)
    print("Experiment 1: User Comparison (NSGA-II, 5 seeds)")
    print("=" * 60)

    # ── Phase 1: run all seeds for both users ──────────────────────────
    problems = {}
    runs = {}   # {(user_id, seed): res}

    for user_id in USER_IDS:
        data = load_all_for_user(user_id)
        ch = Chromosome(data["food_ids"], data["nutrient_matrix"],
                        data["dri"], data["food_groups"],
                        data["food_preferences"])
        problems[user_id] = (MODPProblem(ch, data["foods_df"]), data)

        print(f"\n--- User {user_id} ---")
        for seed in SEEDS:
            verbose = (seed == PLOT_SEED)
            save_history = (seed == PLOT_SEED)
            print(f"  seed={seed}" + ("" if verbose else " (silent)"))
            res = run_nsga2(problems[user_id][0], n_gen=n_gen,
                            pop_size=pop_size, seed=seed, verbose=verbose,
                            save_history=save_history)
            runs[(user_id, seed)] = res
            gc.collect()

    # ── Phase 2: shared reference point from ALL fronts ────────────────
    all_F = [runs[(uid, s)].F for uid in USER_IDS for s in SEEDS]
    ref_point = get_reference_point(all_F)
    print(f"\n  Shared reference point (all runs): {ref_point}")

    # ── Phase 3: compute HV for every run ──────────────────────────────
    hv = {(uid, s): compute_hypervolume(runs[(uid, s)].F, ref_point)
          for uid in USER_IDS for s in SEEDS}

    # ── Phase 4: save per-run detail CSV ───────────────────────────────
    detail_rows = []
    for uid in USER_IDS:
        for seed in SEEDS:
            F = runs[(uid, seed)].F
            detail_rows.append({
                "user_id":   uid,
                "seed":      seed,
                "front_size": len(F),
                "hypervolume": round(hv[(uid, seed)], 4),
                "avg_preference": round(float(-F[:, 0].mean()), 3),
                "avg_cost":       round(float(F[:, 1].mean()), 3),
                "avg_co2":        round(float(F[:, 2].mean()), 3),
            })
    pd.DataFrame(detail_rows).to_csv(
        os.path.join("results", "exp1_all_runs.csv"), index=False
    )
    print("  Saved: results/exp1_all_runs.csv")

    # ── Phase 5: save representative (seed=42) Pareto fronts ───────────
    for uid in USER_IDS:
        F = runs[(uid, PLOT_SEED)].F
        df = pd.DataFrame({
            "preference":    -F[:, 0],
            "cost":           F[:, 1],
            "co2":            F[:, 2],
            "neg_pref_raw":   F[:, 0],
        })
        csv_path = os.path.join("results", f"exp1_user{uid}_pareto.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Saved representative front: {csv_path} ({len(F)} solutions)")

    # ── Phase 6: print per-user HV stats ───────────────────────────────
    print()
    for uid in USER_IDS:
        hvs = [hv[(uid, s)] for s in SEEDS]
        print(f"  User {uid} HV per seed: " +
              "  ".join(f"s{s}={v:.0f}" for s, v in zip(SEEDS, hvs)))
        print(f"    mean={np.mean(hvs):.2f}  std={np.std(hvs):.2f}  "
              f"min={np.min(hvs):.2f}  max={np.max(hvs):.2f}")

    # ── Phase 7: summary CSV (mean/std + seed=42 objective averages) ───
    summary = []
    for uid in USER_IDS:
        hvs = [hv[(uid, s)] for s in SEEDS]
        F42 = runs[(uid, PLOT_SEED)].F
        summary.append({
            "user_id":       uid,
            "hv_mean":       round(np.mean(hvs), 4),
            "hv_std":        round(np.std(hvs),  4),
            "hv_min":        round(np.min(hvs),  4),
            "hv_max":        round(np.max(hvs),  4),
            **{f"hv_seed{s}": round(hv[(uid, s)], 4) for s in SEEDS},
            "front_size_seed42":    len(F42),
            "avg_preference_seed42": round(float(-F42[:, 0].mean()), 3),
            "avg_cost_seed42":       round(float(F42[:, 1].mean()), 3),
            "avg_co2_seed42":        round(float(F42[:, 2].mean()), 3),
        })
    pd.DataFrame(summary).to_csv(
        os.path.join("results", "exp1_summary.csv"), index=False
    )
    print("  Saved: results/exp1_summary.csv")

    # ── Phase 8: plots using seed=42 representative runs ───────────────
    hv1_curve = hypervolume_convergence(
        runs[(1, PLOT_SEED)].history, ref_point)
    hv2_curve = hypervolume_convergence(
        runs[(2, PLOT_SEED)].history, ref_point)
    plot_hypervolume_convergence(
        [hv1_curve, hv2_curve],
        [USER_LABELS[1], USER_LABELS[2]],
        f"Experiment 1 — HV Convergence: User Comparison (seed={PLOT_SEED})",
        "exp1_hv_convergence",
    )
    plot_pareto_front(
        [runs[(1, PLOT_SEED)].F, runs[(2, PLOT_SEED)].F],
        [USER_LABELS[1], USER_LABELS[2]],
        f"Experiment 1 — Pareto Fronts: User 1 vs User 2 (seed={PLOT_SEED})",
        "exp1_pareto_comparison",
    )

    print("\nExperiment 1 complete.")
    return runs, ref_point


if __name__ == "__main__":
    run_experiment1()
