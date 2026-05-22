"""
Penalty function for DRI constraint violations.

For each nutrient j:
  viol_low_j  = max(0, RLL_j - v_j)  / (RUL_j - RLL_j)
  viol_high_j = max(0, v_j - RUL_j)  / (RUL_j - RLL_j)

R = 0.7 * sum(viol_low_j) + 0.3 * sum(viol_high_j)

Under-nutrition (low) penalised more (0.7) than over-nutrition (0.3).

Diversity penalty added:
  R_total = R + alpha / max(n_groups, 1)

Tunable parameters: lambda_penalty (scale), alpha (diversity weight).
"""
import numpy as np
from config import NUTRIENT_IDS


def compute_penalty(nutrient_totals, dri, n_groups,
                    lambda_penalty=1.0, alpha=0.5):
    """
    Parameters
    ----------
    nutrient_totals : array-like of length 5, ordered as NUTRIENT_IDS
    dri             : dict {nutrient_id: (RLL, RUL)}
    n_groups        : int — distinct food groups in the menu
    lambda_penalty  : global penalty scale
    alpha           : diversity penalty weight

    Returns
    -------
    float — total scaled penalty R
    """
    totals = np.asarray(nutrient_totals, dtype=float)
    rll = np.array([dri.get(nid, (0, 0))[0] for nid in NUTRIENT_IDS])
    rul = np.array([dri.get(nid, (0, 0))[1] for nid in NUTRIENT_IDS])

    # Avoid division by zero if RLL == RUL
    spread = np.where(rul - rll > 0, rul - rll, 1.0)

    viol_low  = np.maximum(0.0, rll - totals) / spread
    viol_high = np.maximum(0.0, totals - rul)  / spread

    R_nutrition = 0.7 * viol_low.sum() + 0.3 * viol_high.sum()
    R_diversity = alpha / max(n_groups, 1)

    return lambda_penalty * (R_nutrition + R_diversity)


def penalize_objective(obj_value, nutrient_totals, dri,
                       n_groups, maximize=True,
                       lambda_penalty=1.0, alpha=0.5):
    """
    Apply penalty to a single objective value.

    For maximization objectives: penalized = obj_value - R
    For minimization objectives: penalized = obj_value + R

    Returns the penalized objective value.
    """
    R = compute_penalty(nutrient_totals, dri, n_groups, lambda_penalty, alpha)
    if maximize:
        return obj_value - R
    return obj_value + R
