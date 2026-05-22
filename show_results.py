import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')

print('=== Experiment 1: User Comparison (NSGA-II, 200 gen, pop=100) ===')
print(pd.read_csv('results/exp1_summary.csv').to_string(index=False))

print()
print('=== Experiment 2: Algorithm Comparison (User 1, 200 gen, pop=100) ===')
print(pd.read_csv('results/exp2_summary.csv').to_string(index=False))

print()
print('=== Experiment 3: Diversity Mechanism Impact (User 1, 200 gen, pop=100) ===')
print(pd.read_csv('results/exp3_summary.csv').to_string(index=False))

print()
print('=== Pareto Front Details ===')
for label, f in [('User 1 (Isla, non-veg)', 'results/exp1_user1_pareto.csv'),
                  ('User 2 (Zoely, veg)',    'results/exp1_user2_pareto.csv')]:
    df = pd.read_csv(f)
    print(f'{label}: {len(df)} non-dominated solutions')
    print(f'  Preference: min={df.preference.min():.2f}  max={df.preference.max():.2f}  avg={df.preference.mean():.2f}')
    print(f'  Cost (EUR): min={df.cost.min():.2f}       max={df.cost.max():.2f}       avg={df.cost.mean():.2f}')
    print(f'  CO2 (g):    min={df.co2.min():.2f}        max={df.co2.max():.2f}        avg={df.co2.mean():.2f}')

print()
print('=== Generated Files ===')
import os
for root, dirs, files in os.walk('results'):
    dirs.sort()
    for fn in sorted(files):
        path = os.path.join(root, fn)
        size = os.path.getsize(path)
        print(f'  {size:8,d} bytes  {path}')
