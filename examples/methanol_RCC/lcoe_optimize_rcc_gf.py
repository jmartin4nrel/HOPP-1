import os
from pathlib import Path
from examples.methanol_RCC.rcc_setup_optimize import calculate_lcoe_rcc_optimizer
from examples.H2_Analysis.simple_dispatch import SimpleDispatch
from examples.H2_Analysis.gradient_free import GeneticAlgorithm
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def objective_function(x):
    """
    This is the objective function to be used in the gradient-free optimization,
    specifically with the genetic algorithm from gradient_free.py.
    Right now it is setup with global variables, which should be fixed soon.
    
    INPUTS
    x: array or list, the design variables considered in the optimization

    OUTPUTS:
    h_lcoe: float, the levelized cost of hydrogen that we want to minimize
    """

    solar_capacity_mw = x[0]

    lcoe = calculate_lcoe_rcc_optimizer(solar_capacity_mw)
    
    return lcoe


def optimize_gf(workdir=os.getcwd(), show_plot=False):
    """
    Run the plant optimization to minimize LCOH using gradient-free optimization,
    specifically with the genetic algorithm from gradient_free.py.
    Right now it is setup with global variables, which should be fixed soon.
    This function gives a template on how to set up and run the genetic algorithm.
    """

    h2_examples_path = Path(__file__).absolute().parent

    ga = GeneticAlgorithm()
    ga.objective_function = objective_function
    ga.bits = np.array([8])
    ga.bounds = np.array([(100,1000)])
    ga.variable_type = np.array(["float"])
    
    ga.max_generation = 5
    ga.population_size = 3
    ga.convergence_iters = 10
    ga.tol = 1E-6
    ga.crossover_rate = 0.1
    ga.mutation_rate = 0.01

    ga.optimize_ga(print_progress=True)

    solution_history = ga.solution_history
    opt_lcoe = ga.optimized_function_value
    opt_vars = ga.optimized_design_variables

    opt_solar_size_mw = opt_vars[0]

    if show_plot:
        import matplotlib.pyplot as plt
        plt.plot(solution_history)
        plt.show()

    return opt_lcoe, opt_solar_size_mw

if __name__=="__main__":

    import time

    start = time.time()
    opt_lcoe, opt_solar_size_mw = optimize_gf(show_plot=True)

    print("time to run: ", time.time()-start)
    print("opt_lcoe: ", opt_lcoe)
    print("opt_solar: ", opt_solar_size_mw)
