"""
Main entry point — runs all three experiments sequentially.
Usage: python run_all.py [--quick]
  --quick : n_gen=20, pop_size=30 for rapid testing
"""
import sys
import os
import time

os.makedirs("results", exist_ok=True)
os.makedirs(os.path.join("results", "figures"), exist_ok=True)

quick = "--quick" in sys.argv
n_gen = 20 if quick else 200
pop_size = 30 if quick else 100

print(f"MODP — Running all experiments (n_gen={n_gen}, pop_size={pop_size})")
print("=" * 60)

t0 = time.time()

from experiments.exp1_user_comparison import run_experiment1
from experiments.exp2_algorithm_comparison import run_experiment2
from experiments.exp3_diversity_impact import run_experiment3

print("\n[1/3] Experiment 1: User Comparison")
run_experiment1(n_gen=n_gen, pop_size=pop_size)

print("\n[2/3] Experiment 2: Algorithm Comparison")
run_experiment2(n_gen=n_gen, pop_size=pop_size)

print("\n[3/3] Experiment 3: Diversity Impact")
run_experiment3(n_gen=n_gen, pop_size=pop_size)

elapsed = time.time() - t0
print(f"\nAll experiments done in {elapsed/60:.1f} minutes.")
print("Results saved to: results/")
print("Figures saved to: results/figures/")
