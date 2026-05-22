"""SPEA2 runner using pymoo with permutation operators."""
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination


def run_spea2(problem, n_gen=200, pop_size=100, seed=42, verbose=True,
              save_history=True):
    """
    Run SPEA2 on a MODPProblem instance.

    Archive size is set equal to pop_size.
    save_history=False for silent seeds saves significant memory.
    """
    algorithm = SPEA2(
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
