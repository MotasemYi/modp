"""
Chromosome representation for the MODP.

Genotype: permutation of indices 0..N-1 (where N = total foods for this user).
  - Positions 0..BREAKFAST_SIZE-1  → breakfast candidate pool
  - Positions BREAKFAST_SIZE..N-1  → lunch/dinner candidate pool

Greedy decoder maps the permutation to an actual menu (phenotype).
Food-group filtering and preference=-1 skipping are applied during decode.
"""
import numpy as np
from config import NUTRIENT_IDS

BREAKFAST_SIZE = 96
LUNCH_SIZE = 309
TOTAL_FOODS = 405

EPS_UPPER = 1.15
EPS_LOWER = 0.90
BREAKFAST_RATIO = 0.35
MIN_GROUPS = 4
SKIP_PREFERENCE = -1.0  # foods with this preference value are never selected

# Food groups the professor designated as breakfast-appropriate
# (Dairy, Chicken, Jams, Honey, Pancakes, Seed/Bean, Bakery, Sweet Things,
#  Beverages, Cereals, Fruits, Appetizer, Beverages2, Bakery2)
BREAKFAST_GROUPS = frozenset([1, 2, 4, 5, 7, 8, 11, 12, 13, 14, 20, 22, 26, 27])

# Subset of BREAKFAST_GROUPS that are breakfast-exclusive (never in lunch/dinner)
# Jams(4), Honey(5), Pancakes(7), Sweet Things(12), Cereals(14)
# Groups NOT listed here (Dairy, Chicken, Seed/Bean, Bakery, Beverages,
# Fruits, Appetizer, Bakery2) can appear in both meals.
EXCLUSIVE_BREAKFAST_GROUPS = frozenset([4, 5, 7, 12, 14])


class Chromosome:
    def __init__(self, food_ids, nutrient_matrix, dri, food_groups,
                 food_preferences=None):
        """
        food_ids        : list of food IDs in fixed order (length N)
        nutrient_matrix : pd.DataFrame indexed by food_id, cols = NUTRIENT_IDS
        dri             : dict {nutrient_id: (RLL, RUL)}
        food_groups     : dict {food_id: group_id}
        food_preferences: dict {food_id: preference_value} — used to skip
                          foods with preference == SKIP_PREFERENCE (-1)
        """
        self.food_ids = list(food_ids)
        self.n = len(self.food_ids)
        self.food_groups = food_groups
        self.dri = dri
        self._preferences = food_preferences or {}

        # Build fast numpy array: shape (N, 5) in the order of food_ids
        self._nut_array = np.zeros((self.n, len(NUTRIENT_IDS)), dtype=float)
        for i, fid in enumerate(self.food_ids):
            if fid in nutrient_matrix.index:
                self._nut_array[i] = nutrient_matrix.loc[fid, NUTRIENT_IDS].values

        # DRI vectors ordered as NUTRIENT_IDS
        self._rll = np.array([dri.get(nid, (0, 0))[0] for nid in NUTRIENT_IDS])
        self._rul = np.array([dri.get(nid, (0, 0))[1] for nid in NUTRIENT_IDS])

        # Breakfast DRI sub-targets (35% of daily)
        self._rll_b = self._rll * BREAKFAST_RATIO
        self._rul_b = self._rul * BREAKFAST_RATIO

        # Index positions within NUTRIENT_IDS for key nutrients
        self._energy_idx = NUTRIENT_IDS.index(5)   # Energy
        self._protein_idx = NUTRIENT_IDS.index(15)  # Protein

    # ------------------------------------------------------------------
    # Chromosome initialisation
    # ------------------------------------------------------------------

    def random_chromosome(self):
        """Return a random permutation as a 1-D integer numpy array."""
        return np.random.permutation(self.n)

    # ------------------------------------------------------------------
    # Greedy decoder
    # ------------------------------------------------------------------

    def decode(self, genes):
        """
        Decode a permutation array (length N) into a breakfast + lunch/dinner menu.

        Breakfast phase (genes[0:BREAKFAST_SIZE]):
          - Only foods from BREAKFAST_GROUPS are considered.
          - Foods with preference == -1 are skipped.
          - Add food if energy/protein won't exceed EPS_UPPER * breakfast RUL.
          - Stop when EPS_LOWER * breakfast RLL is met for energy & protein.

        Lunch/dinner phase (genes[BREAKFAST_SIZE:]):
          - Foods from EXCLUSIVE_BREAKFAST_GROUPS are skipped.
          - Foods with preference == -1 are skipped.
          - Add food if no nutrient exceeds EPS_UPPER * daily RUL.
          - Stop when all 5 nutrients meet EPS_LOWER * daily RLL.

        Returns
        -------
        selected_ids : list of food_ids chosen
        totals       : np.array shape (5,) — daily nutrient totals
        is_feasible  : bool
        n_groups     : int — distinct food-group IDs in the menu
        """
        breakfast_genes = genes[:BREAKFAST_SIZE]
        lunch_genes = genes[BREAKFAST_SIZE:]

        # ---- Breakfast phase ----
        breakfast_ids = []
        totals_b = np.zeros(len(NUTRIENT_IDS))

        for idx in breakfast_genes:
            fid = self.food_ids[idx]

            # Skip disliked foods
            if self._preferences.get(fid, 0.0) == SKIP_PREFERENCE:
                continue

            # Only breakfast-appropriate food groups
            if self.food_groups.get(fid) not in BREAKFAST_GROUPS:
                continue

            nutrients = self._nut_array[idx]
            new_totals = totals_b + nutrients

            # Hard cap: don't breach EPS_UPPER * breakfast RUL on energy/protein
            if (new_totals[self._energy_idx] > EPS_UPPER * self._rul_b[self._energy_idx] or
                    new_totals[self._protein_idx] > EPS_UPPER * self._rul_b[self._protein_idx]):
                continue

            breakfast_ids.append(fid)
            totals_b = new_totals

            # Early stop: energy & protein both reach EPS_LOWER * breakfast RLL
            if (totals_b[self._energy_idx] >= EPS_LOWER * self._rll_b[self._energy_idx] and
                    totals_b[self._protein_idx] >= EPS_LOWER * self._rll_b[self._protein_idx]):
                break

        # ---- Lunch/Dinner phase ----
        lunch_ids = []
        totals = totals_b.copy()

        for idx in lunch_genes:
            fid = self.food_ids[idx]

            # Skip disliked foods
            if self._preferences.get(fid, 0.0) == SKIP_PREFERENCE:
                continue

            # Exclude groups that belong exclusively to breakfast
            if self.food_groups.get(fid) in EXCLUSIVE_BREAKFAST_GROUPS:
                continue

            nutrients = self._nut_array[idx]
            new_totals = totals + nutrients

            # Don't breach EPS_UPPER * daily RUL for any nutrient
            if np.any(new_totals > EPS_UPPER * self._rul):
                continue

            lunch_ids.append(fid)
            totals = new_totals

            # Early stop: all 5 nutrients meet EPS_LOWER * daily RLL
            if np.all(totals >= EPS_LOWER * self._rll):
                break

        selected_ids = breakfast_ids + lunch_ids

        # ---- Diversity check ----
        groups = {self.food_groups.get(fid) for fid in selected_ids
                  if fid in self.food_groups}
        groups.discard(None)
        n_groups = len(groups)

        is_feasible = (
            np.all(totals >= self._rll) and
            np.all(totals <= self._rul) and
            n_groups >= MIN_GROUPS
        )

        return selected_ids, totals, is_feasible, n_groups

    # ------------------------------------------------------------------
    # Objective evaluation
    # ------------------------------------------------------------------

    def evaluate_objectives(self, selected_ids, foods_df):
        """
        Returns (preference_total, cost_total, co2_total).
        preference comes from user_foods.preference (foods_df column).
        Foods with preference == -1 are excluded from the preference sum
        (they should not have been selected, but as a safeguard we zero them).
        """
        if not selected_ids:
            return 0.0, 0.0, 0.0

        mask = foods_df["food_id"].isin(selected_ids)
        subset = foods_df[mask]

        # Treat -1 as 0 contribution to preference sum (safety net)
        pref_values = subset["preference"].where(subset["preference"] != SKIP_PREFERENCE, 0.0)
        preference = float(pref_values.sum())
        cost = float(subset["cost"].sum())
        co2 = float(subset["co2"].sum())
        return preference, cost, co2
