# Multi-Objective Diet Optimization Problem (MODP)

**Course:** Heuristic Optimization Algorithms — Term Project (30% of final grade)  
**Model:** Multi-Objective Multidimensional Knapsack Problem (MOMKP)  
**Algorithms:** NSGA-II and SPEA2 via [pymoo](https://pymoo.org/)

---

## Problem Description

Recommend a daily menu (breakfast + lunch/dinner) from a database of **405 prepared food items**, optimising simultaneously across three conflicting objectives:

| Objective | Direction | Source column |
|-----------|-----------|---------------|
| f1 — User preference | **MAXIMISE** | `user_foods.preference` |
| f2 — Cost | **MINIMISE** | `foods.cost` |
| f3 — CO₂ footprint | **MINIMISE** | `foods.co2` |

Subject to five daily nutritional constraints (DRI bounds per user):

| Constraint | Nutrient | nutrient_id |
|------------|----------|-------------|
| C1 | Energy | 5 (kcal) |
| C2 | Protein | 15 (g) |
| C3 | Carbohydrate | 8 (g) |
| C4 | Fiber, total dietary | 4 (g) |
| C5 | Sodium, Na | 17 (mg) |

### Users

| ID | Name | Diet |
|----|------|------|
| 1 | Isla Morris | Non-vegetarian |
| 2 | Zoely Butler | Vegetarian (98 foods excluded via preference = −1) |

---

## Project Structure

```
modp/
├── config.py                  # DB connection + nutrient IDs
├── db_loader.py               # All DB queries (no hardcoded values)
├── chromosome.py              # Permutation chromosome + greedy decoder
├── penalty.py                 # Penalty function for DRI violations
├── problem.py                 # pymoo Problem wrapper
├── metrics.py                 # Hypervolume calculation
├── visualize.py               # Pareto front plots + HV convergence curves
├── run_all.py                 # Main entry point (all 3 experiments)
├── show_results.py            # Print result summary tables
├── algorithms/
│   ├── nsga2_runner.py
│   └── spea2_runner.py
├── experiments/
│   ├── exp1_user_comparison.py
│   ├── exp2_algorithm_comparison.py
│   └── exp3_diversity_impact.py
└── results/
    ├── exp1_user1_pareto.csv      # Pareto front, seed=42
    ├── exp1_user2_pareto.csv
    ├── exp1_all_runs.csv          # HV for all 5 seeds, both users
    ├── exp1_summary.csv           # mean/std/min/max HV per user
    ├── exp2_nsgaii_pareto.csv     # Pareto front, seed=42
    ├── exp2_spea2_pareto.csv
    ├── exp2_all_runs.csv          # HV for all 5 seeds, both algorithms
    ├── exp2_summary.csv           # mean/std/min/max HV per algorithm
    ├── exp3_diversity_with.csv    # Pareto front, seed=42
    ├── exp3_diversity_without.csv
    ├── exp3_all_runs.csv          # HV for all 5 seeds, both configs
    ├── exp3_summary.csv           # mean/std/min/max HV per config
    └── figures/
        ├── exp1_pareto_comparison.png   # seed=42 representative
        ├── exp1_hv_convergence.png
        ├── exp2_pareto_comparison.png
        ├── exp2_hv_convergence.png
        └── exp3_diversity_comparison.png
```

---

## Installation

Requires Python 3.10+ and a running MySQL server with the `diet` database.

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
mysql-connector-python>=8.0.33
pymoo>=0.6.1
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scipy>=1.10.0
```

### Database Configuration

Edit `config.py` if your MySQL credentials differ from the defaults:

```python
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "",
    "database": "diet",
}
```

---

## Usage

```bash
# Full run — 200 generations, population 100 (≈ 5 minutes)
python run_all.py

# Quick smoke-test — 20 generations, population 30 (≈ 15 seconds)
python run_all.py --quick

# Print result tables only (no rerun)
python show_results.py

# Run individual experiments
python experiments/exp1_user_comparison.py
python experiments/exp2_algorithm_comparison.py
python experiments/exp3_diversity_impact.py
```

---

## Implementation Details

### Chromosome Representation

A **permutation of all 405 food indices** split positionally into two pools:

```
[ breakfast pool (96 genes) | lunch/dinner pool (309 genes) ]
```

The greedy decoder respects food-group constraints during phenotype construction:

**Breakfast phase** (`genes[0:96]`) — only foods from breakfast-appropriate groups:

| Group ID | Name |
|----------|------|
| 1 | Dairy Products |
| 2 | Chicken Products |
| 4 | Jams Syrups |
| 5 | Honey Products |
| 7 | Pancakes |
| 8 | Seed Bean Olive |
| 11 | Bakery |
| 12 | Sweet Things |
| 13 | Beverages |
| 14 | Cereals |
| 20 | Fruits |
| 22 | Appetizer |
| 26 | Beverages 2 |
| 27 | Bakery 2 |

**Lunch/dinner phase** (`genes[96:]`) — any group except breakfast-exclusive ones (Jams, Honey, Pancakes, Sweet Things, Cereals). Groups like Dairy, Chicken, Fruits, Beverages can appear in both meals.

**Foods with preference = −1 are skipped entirely** in both phases (vegetarian/dietary exclusion).

### Greedy Decoder Targets

| Phase | Nutrient targets | Early stop |
|-------|-----------------|------------|
| Breakfast | Energy + Protein ≤ 1.15 × 35% RUL | Energy + Protein ≥ 0.90 × 35% RLL |
| Lunch/dinner | All 5 nutrients ≤ 1.15 × daily RUL | All 5 nutrients ≥ 0.90 × daily RLL |

### Penalty Function

For each nutrient j, normalised violations are computed and combined:

```
viol_low_j  = max(0, RLL_j − v_j) / (RUL_j − RLL_j)
viol_high_j = max(0, v_j − RUL_j) / (RUL_j − RLL_j)

R_nutrition = 0.7 × Σ viol_low_j + 0.3 × Σ viol_high_j
R_diversity = α / max(distinct_groups, 1)
R_total     = λ × (R_nutrition + R_diversity)
```

Under-nutrition (0.7) is penalised more heavily than over-nutrition (0.3).  
Default: `λ = 1.0`, `α = 0.5`.

### pymoo Operators

| Component | Setting |
|-----------|---------|
| Sampling | `PermutationRandomSampling` |
| Crossover | `OrderCrossover` (OX), prob = 0.9 |
| Mutation | `InversionMutation`, prob = 0.2 |
| Duplicates | Eliminated |

### Statistical Evaluation Protocol

Each experiment runs **5 independent seeds** `[42, 123, 456, 789, 1000]` per configuration:

- The **shared reference point** for hypervolume is computed from all fronts combined across all seeds and configurations within each experiment, ensuring fair comparison.
- **Seed 42** is the representative run used for Pareto front plots and HV convergence curves.
- Silent seeds (`save_history=False`) reduce memory usage; history is only retained for seed 42.
- Summary CSVs report `hv_mean`, `hv_std`, `hv_min`, `hv_max`, and individual `hv_seed{N}` columns. Per-run detail is in `*_all_runs.csv`.

---

## Experimental Results (200 generations, population = 100, 5 seeds)

### Experiment 1 — User Comparison (NSGA-II)

| User | HV Mean | HV Std | HV Min | HV Max |
|------|---------|--------|--------|--------|
| User 1 — Isla (non-veg) | 476,306 | 9,900 | 465,514 | 494,102 |
| User 2 — Zoely (veg) | **752,848** | **6,777** | 741,855 | 760,243 |

Per-seed hypervolumes:

| User | seed=42 | seed=123 | seed=456 | seed=789 | seed=1000 |
|------|---------|----------|----------|----------|-----------|
| User 1 | 469,020 | 465,514 | 475,077 | 494,102 | 477,819 |
| User 2 | 749,361 | 753,488 | 760,243 | 759,292 | 741,855 |

User 2's mean HV is **58% higher** with lower variance (std 6,777 vs 9,900). Despite 98 foods excluded via preference = −1, the remaining vegetarian items carry higher personal ratings, consistently pushing the Pareto front to a more favourable region across all seeds.

Representative front (seed=42) objective ranges:

| User | Preference (max) | Cost (min–max) | CO₂ (min–max) |
|------|-----------------|----------------|----------------|
| User 1 | 120.69 | 4.95 – 37.35 | 2.03 – 86.55 |
| User 2 | 229.55 | 4.25 – 44.58 | 3.01 – 89.65 |

### Experiment 2 — Algorithm Comparison (User 1)

| Algorithm | HV Mean | HV Std | HV Min | HV Max |
|-----------|---------|--------|--------|--------|
| NSGA-II | 348,740 | **7,007** | 341,794 | 361,526 |
| SPEA2 | **361,055** | 15,761 | 339,561 | 379,907 |

Per-seed hypervolumes:

| Algorithm | seed=42 | seed=123 | seed=456 | seed=789 | seed=1000 |
|-----------|---------|----------|----------|----------|-----------|
| NSGA-II | 344,654 | 341,794 | 350,647 | 361,526 | 345,078 |
| SPEA2 | 379,907 | 352,522 | 339,561 | 354,611 | 378,672 |

SPEA2 has a higher mean HV (+3.5%) but **2.25× larger standard deviation** (15,761 vs 7,007). NSGA-II is the more consistent choice: its worst run (341,794) still outperforms SPEA2's worst run (339,561), and its variance is substantially lower. SPEA2's strength-based archive occasionally finds superior fronts (seed 42: 379,907) but can also underperform (seed 456: 339,561).

Representative front (seed=42) objective averages:

| Algorithm | Avg Preference | Avg Cost | Avg CO₂ |
|-----------|---------------|----------|---------|
| NSGA-II | 73.51 | 17.03 | 30.78 |
| SPEA2 | 68.94 | **13.17** | **22.63** |

### Experiment 3 — Diversity Mechanism Impact (User 1)

| Config | HV Mean | HV Std | Avg Groups (mean ± std) |
|--------|---------|--------|------------------------|
| With diversity (α = 0.5) | 362,010 | **7,250** | **7.664 ± 0.356** |
| Without diversity (α = 0.0) | 365,899 | 23,681 | 7.498 ± 0.565 |

Per-seed hypervolumes:

| Config | seed=42 | seed=123 | seed=456 | seed=789 | seed=1000 |
|--------|---------|----------|----------|----------|-----------|
| With diversity | 357,591 | 354,664 | 363,591 | 375,318 | 358,887 |
| Without diversity | 327,192 | 371,995 | 371,195 | 358,730 | 400,382 |

The without-diversity configuration has a marginally higher mean HV (+1.1%) but **3.3× larger standard deviation** (23,681 vs 7,250). Its seed-to-seed range spans 73,190 units vs only 20,654 for with-diversity — the diversity penalty acts as a stabiliser. With diversity also produces more varied menus (7.664 vs 7.498 avg distinct food groups) with lower variance in group counts (0.356 vs 0.565), indicating more reliable nutritional variety across runs.

---

## Database Schema (relevant tables)

| Table | Key columns |
|-------|-------------|
| `foods` | `id`, `name`, `foodGroupId`, `cost`, `co2`, `preference` |
| `user_foods` | `userId`, `foodId`, `preference` |
| `food_nutrients` | `foodId`, `nutrientId`, `quantity` |
| `dri` | `nutrient_id`, `low_age`, `up_age`, `gender`, `RLL`, `RUL` |
| `user` | `id`, `age`, `gender` |
| `food_group` | `id`, `name` |

All values — nutrient IDs, DRI bounds, food group IDs — are queried from the database at runtime. Nothing is hardcoded in the algorithm logic.
