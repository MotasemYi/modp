"""NSGA-II runner using pymoo with permutation operators."""
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination


def run_nsga2(problem, n_gen=200, pop_size=100, seed=42, verbose=True,
              save_history=True):
    """
    Run NSGA-II on a MODPProblem instance.

    save_history=False for silent seeds saves significant memory when running
    multiple seeds sequentially (history stores 200 full generation snapshots).
    """
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=PermutationRandomSampling(),
        crossover=OrderCrossover(prob=0.9),
        mutation=InversionMutation(prob=0.2),
        eliminate_duplicates=True,
    )
    termination = get_termination("n_gen", n_gen)
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        verbose=verbose,
        save_history=save_history,
    )
    return res
