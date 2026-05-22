"""
Experiment 3: Impact of the diversity mechanism on User 1.
Each configuration (with/without diversity) is run 5 times with different seeds.
Reports mean ± std hypervolume and average distinct food groups.
Seed 42 is the representative run for Pareto front plots.

Diversity is controlled via alpha in MODPProblem:
  alpha=0.5  → diversity penalty active
  alpha=0.0  → diversity penalty disabled
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
from metrics import get_reference_point, compute_hypervolume
from visualize import plot_pareto_front

USER_ID = 1
SEEDS = [42, 123, 456, 789, 1000]
PLOT_SEED = 42
CONFIGS = {
    "with_diversity":    {"alpha": 0.5},
    "without_diversity": {"alpha": 0.0},
}


def _decode_population(res, ch):
    """Re-decode Pareto-front genotypes to count distinct food groups."""
    return [ch.decode(x.astype(int))[3] for x in res.X]


def run_experiment3(n_gen=200, pop_size=100):
    print("=" * 60)
    print("Experiment 3: Diversity Mechanism Impact (User 1, 5 seeds)")
    print("=" * 60)

    data = load_all_for_user(USER_ID)
    ch = Chromosome(data["food_ids"], data["nutrient_matrix"],
                    data["dri"], data["food_groups"],
                    data["food_preferences"])

    # ── Phase 1: run all seeds for both configurations ─────────────────
    runs = {}   # {(config_name, seed): res}

    for cfg_name, cfg in CONFIGS.items():
        problem = MODPProblem(ch, data["foods_df"], alpha=cfg["alpha"])
        print(f"\n--- {cfg_name} (alpha={cfg['alpha']}) ---")
        for seed in SEEDS:
            verbose = (seed == PLOT_SEED)
            save_history = (seed == PLOT_SEED)
            print(f"  seed={seed}" + ("" if verbose else " (silent)"))
            res = run_nsga2(problem, n_gen=n_gen, pop_size=pop_size,
                            seed=seed, verbose=verbose, save_history=save_history)
            runs[(cfg_name, seed)] = res
            gc.collect()

    # ── Phase 2: shared reference point from ALL fronts ────────────────
    all_F = [runs[(cfg, s)].F for cfg in CONFIGS for s in SEEDS]
    ref_point = get_reference_point(all_F)
    print(f"\n  Shared reference point (all runs): {ref_point}")

    # ── Phase 3: compute HV for every run ──────────────────────────────
    hv = {(cfg, s): compute_hypervolume(runs[(cfg, s)].F, ref_point)
          for cfg in CONFIGS for s in SEEDS}

    # ── Phase 4: decode group diversity for every run ──────────────────
    groups = {(cfg, s): _decode_population(runs[(cfg, s)], ch)
              for cfg in CONFIGS for s in SEEDS}

    # ── Phase 5: save per-run detail CSV ───────────────────────────────
    detail_rows = []
    for cfg in CONFIGS:
        for seed in SEEDS:
            F = runs[(cfg, seed)].F
            g = groups[(cfg, seed)]
            detail_rows.append({
                "config":      cfg,
                "alpha":       CONFIGS[cfg]["alpha"],
                "seed":        seed,
                "front_size":  len(F),
                "hypervolume": round(hv[(cfg, seed)], 4),
                "avg_distinct_groups": round(float(np.mean(g)), 3),
                "avg_preference": round(float(-F[:, 0].mean()), 3),
                "avg_cost":       round(float(F[:, 1].mean()), 3),
                "avg_co2":        round(float(F[:, 2].mean()), 3),
            })
    pd.DataFrame(detail_rows).to_csv(
        os.path.join("results", "exp3_all_runs.csv"), index=False
    )
    print("  Saved: results/exp3_all_runs.csv")

    # ── Phase 6: save representative (seed=42) Pareto fronts ───────────
    for cfg in CONFIGS:
        F = runs[(cfg, PLOT_SEED)].F
        g = groups[(cfg, PLOT_SEED)]
        df = pd.DataFrame({
            "preference":      -F[:, 0],
            "cost":             F[:, 1],
            "co2":              F[:, 2],
            "distinct_groups":  g,
        })
        path = os.path.join("results", f"exp3_diversity_{cfg.split('_')[0]}.csv")
        df.to_csv(path, index=False)
        print(f"  Saved representative front: {path} ({len(F)} solutions)")

    # ── Phase 7: print per-config HV and group stats ───────────────────
    print()
    for cfg in CONFIGS:
        hvs  = [hv[(cfg, s)]                     for s in SEEDS]
        grps = [float(np.mean(groups[(cfg, s)])) for s in SEEDS]
        print(f"  {cfg}")
        print(f"    HV per seed: " +
              "  ".join(f"s{s}={v:.0f}" for s, v in zip(SEEDS, hvs)))
        print(f"    HV   mean={np.mean(hvs):.2f}  std={np.std(hvs):.2f}  "
              f"min={np.min(hvs):.2f}  max={np.max(hvs):.2f}")
        print(f"    Groups mean={np.mean(grps):.2f}  std={np.std(grps):.2f}")

    # ── Phase 8: summary CSV ───────────────────────────────────────────
    summary = []
    for cfg in CONFIGS:
        hvs  = [hv[(cfg, s)]                     for s in SEEDS]
        grps = [float(np.mean(groups[(cfg, s)])) for s in SEEDS]
        F42  = runs[(cfg, PLOT_SEED)].F
        summary.append({
            "config":    cfg,
            "alpha":     CONFIGS[cfg]["alpha"],
            "hv_mean":   round(np.mean(hvs), 4),
            "hv_std":    round(np.std(hvs),  4),
            "hv_min":    round(np.min(hvs),  4),
            "hv_max":    round(np.max(hvs),  4),
            **{f"hv_seed{s}": round(hv[(cfg, s)], 4) for s in SEEDS},
            "avg_groups_mean": round(np.mean(grps), 3),
            "avg_groups_std":  round(np.std(grps),  3),
            "front_size_seed42":     len(F42),
            "avg_preference_seed42": round(float(-F42[:, 0].mean()), 3),
            "avg_cost_seed42":       round(float(F42[:, 1].mean()), 3),
            "avg_co2_seed42":        round(float(F42[:, 2].mean()), 3),
        })
    pd.DataFrame(summary).to_csv(
        os.path.join("results", "exp3_summary.csv"), index=False
    )
    print("  Saved: results/exp3_summary.csv")

    # ── Phase 9: plot using seed=42 representative runs ────────────────
    plot_pareto_front(
        [runs[("with_diversity", PLOT_SEED)].F,
         runs[("without_diversity", PLOT_SEED)].F],
        ["With Diversity", "Without Diversity"],
        f"Experiment 3 — Diversity Mechanism Impact (seed={PLOT_SEED})",
        "exp3_diversity_comparison",
    )

    print("\nExperiment 3 complete.")
    return runs, ref_point


if __name__ == "__main__":
    run_experiment3()
