"""Visualisation utilities — Pareto front plots and HV convergence curves."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

os.makedirs(os.path.join("results", "figures"), exist_ok=True)


def _fig_path(filename):
    return os.path.join("results", "figures", f"{filename}.png")


def plot_pareto_front(F_list, labels, title, filename):
    """
    3-D scatter plot of Pareto fronts.
    F_list : list of (n, 3) arrays (pymoo minimise convention)
    Axes   : preference (negated back to positive), cost, co2.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    colors = ["blue", "red", "green", "orange"]

    for F, label, color in zip(F_list, labels, colors):
        if F is None or len(F) == 0:
            continue
        pref = -F[:, 0]   # un-negate
        cost = F[:, 1]
        co2  = F[:, 2]
        ax.scatter(pref, cost, co2, label=label, alpha=0.6, c=color, s=30)

    ax.set_xlabel("Preference (MAX)")
    ax.set_ylabel("Cost (MIN)")
    ax.set_zlabel("CO2 (MIN)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(_fig_path(filename), dpi=150)
    plt.close()
    print(f"  Saved: {_fig_path(filename)}")


def plot_hypervolume_convergence(hv_curves, labels, title, filename):
    """Line plot: HV vs generation."""
    plt.figure(figsize=(10, 5))
    for hv, label in zip(hv_curves, labels):
        plt.plot(hv, label=label, linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(_fig_path(filename), dpi=150)
    plt.close()
    print(f"  Saved: {_fig_path(filename)}")


def print_sample_menu(selected_food_ids, foods_df, nutrient_totals,
                      dri, objectives, solution_index=0):
    """Print a formatted menu table."""
    from config import NUTRIENT_IDS
    nutrient_names = {4: "Fiber(g)", 5: "Energy(kcal)", 8: "Carbs(g)",
                      15: "Protein(g)", 17: "Sodium(mg)"}

    pref, cost, co2 = objectives
    print(f"\n--- Solution {solution_index} ---")
    print(f"  Preference: {pref:.2f}  Cost: {cost:.2f}  CO2: {co2:.2f}")
    print(f"  Items ({len(selected_food_ids)}):")

    mask = foods_df["food_id"].isin(selected_food_ids)
    subset = foods_df[mask]
    for _, row in subset.iterrows():
        name = str(row["name"])[:40]
        print(f"    {row['food_id']:5d}  {name:<40s}  pref={row['preference']:.1f}"
              f"  cost={row['cost']:.2f}  co2={row['co2']:.2f}")

    print("  Nutrient totals:")
    for i, nid in enumerate(NUTRIENT_IDS):
        rll, rul = dri.get(nid, (0, 0))
        val = nutrient_totals[i]
        status = "OK" if rll <= val <= rul else "VIOL"
        print(f"    {nutrient_names.get(nid, nid):15s}: {val:8.1f}  "
              f"[{rll:.1f} - {rul:.1f}] {status}")
