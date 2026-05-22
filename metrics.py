"""Hypervolume and convergence metrics."""
import numpy as np
from pymoo.indicators.hv import HV


def compute_hypervolume(F, ref_point):
    """
    F         : (n_solutions, 3) array of objective values (all minimise)
    ref_point : (3,) array — worst-case reference point
    Returns scalar hypervolume.
    """
    if F is None or len(F) == 0:
        return 0.0
    ind = HV(ref_point=np.asarray(ref_point, dtype=float))
    return float(ind(F))


def get_reference_point(all_F_results, margin=0.10):
    """
    Compute a shared reference point from all combined results.
    Uses the worst (max) value per objective across all runs, plus margin.

    all_F_results : list of (n_solutions, 3) arrays
    """
    combined = np.vstack([F for F in all_F_results if F is not None and len(F) > 0])
    worst = np.max(combined, axis=0)
    # For objectives that could be negative (negated preference), ensure margin
    # is applied correctly: add margin * |worst| + small epsilon
    ref = worst + margin * np.abs(worst) + 1e-6
    return ref


def hypervolume_convergence(history, ref_point):
    """
    Extract hypervolume at each generation from a pymoo result history list.

    history   : res.history (list of Algorithm snapshots)
    ref_point : reference point (same for all comparisons)
    Returns list of HV values, one per generation.
    """
    hv_values = []
    for gen in history:
        try:
            F = gen.opt.get("F")
            hv_values.append(compute_hypervolume(F, ref_point))
        except Exception:
            hv_values.append(0.0)
    return hv_values
