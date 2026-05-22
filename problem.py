"""
pymoo Problem class for the MODP.

Decision variable: integer permutation of length TOTAL_FOODS.
Objectives (all MINIMISED by pymoo):
  F[0] = -preference + R   (we maximise preference, so negate)
  F[1] =  cost       + R
  F[2] =  co2        + R

Constraint violations are folded into the objectives via penalty.
"""
import numpy as np
from pymoo.core.problem import Problem

from chromosome import Chromosome, TOTAL_FOODS
from penalty import compute_penalty


class MODPProblem(Problem):
    def __init__(self, chromosome_handler: Chromosome, foods_df,
                 lambda_penalty=1.0, alpha=0.5, **kwargs):
        """
        chromosome_handler : Chromosome instance (already loaded with nutrients/DRI)
        foods_df           : DataFrame with food_id, preference, cost, co2
        lambda_penalty     : penalty scale factor
        alpha              : diversity penalty weight
        """
        super().__init__(
            n_var=TOTAL_FOODS,
            n_obj=3,
            n_constr=0,
            xl=0,
            xu=TOTAL_FOODS - 1,
            vtype=int,
            **kwargs,
        )
        self.ch = chromosome_handler
        self.foods_df = foods_df
        self.lambda_penalty = lambda_penalty
        self.alpha = alpha

    def _evaluate(self, X, out, *args, **kwargs):
        """
        X : population matrix (pop_size x TOTAL_FOODS), each row is a permutation.
        """
        F = np.empty((len(X), 3), dtype=float)

        for i, x in enumerate(X):
            genes = x.astype(int)
            selected, totals, feasible, n_groups = self.ch.decode(genes)
            pref, cost, co2 = self.ch.evaluate_objectives(selected, self.foods_df)
            R = compute_penalty(totals, self.ch.dri, n_groups,
                                self.lambda_penalty, self.alpha)
            F[i, 0] = -pref + R   # maximise → negate for pymoo
            F[i, 1] = cost  + R
            F[i, 2] = co2   + R

        out["F"] = F
