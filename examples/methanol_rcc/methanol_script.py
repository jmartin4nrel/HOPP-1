from hopp.simulation import HoppInterface
import numpy as np
import multiprocessing
import time
from examples.methanol_rcc.calculate_methanol_cost import calculate_methanol_cost

from typing import Tuple
import numpy as np
from collections import OrderedDict
from hopp.simulation.hybrid_simulation import HybridSimulation
from hopp.tools.optimization import DataRecorder
from hopp.tools.optimization.optimization_problem import OptimizationProblem
from hopp.tools.optimization.optimization_driver import OptimizationDriver

class HybridSizingProblem(OptimizationProblem):
    """
    Optimize the hybrid system sizing design variables
    """
    def __init__(self,
                # simulation: HybridSimulation
                ) -> None:
        """
        design_variables: nametuple of hybrid technologies each with a namedtuple of design variables
        """
        super().__init__()
        # self.simulation = simulation
        self.candidate_dict = OrderedDict({
        "pct_wind": {
            "type": float,
            "prior": {
                "mu": 50, "sigma": 100
            },
            "min": 0, "max": 100
        },
        "pct_overbuild": {
            "type": float,
            "prior": {
                "mu": 50, "sigma": 100
            },
            "min": 0, "max": 100
        }})

    def _set_simulation_to_candidate(self,
                                    candidate: np.ndarray,
                                    ):
        pct_wind = candidate[0]
        pct_overbuild = candidate[1]
    
        return pct_wind, pct_overbuild

    def objective(self,
                    candidate: object
                    ) -> Tuple:
        candidate_conforming, penalty_conforming = self.conform_candidate_and_get_penalty(candidate)
        pct_wind, pct_overbuild = self._set_simulation_to_candidate(candidate_conforming)
        lat = 32.34
        lon = -98.27
        evaluation = 10-calculate_methanol_cost(pct_wind, pct_overbuild, lat, lon)
        score = 10-evaluation
        return score, evaluation, candidate_conforming


if __name__ == '__main__':

    # Enter number of cores to use to process optimization
    num_cores = 12

    # TODO sweep through reactor parameters

    # TODO sweep through years

    # Either sweep through locations or pick a specific location
    lat = 32.34
    lon = -98.27

    # Set up to optimize the % wind and % overbuild
    arg_lists = []
    pcts_wind = np.arange(0,120,20)
    pcts_overbuild = np.arange(0,120,20)
    for i in range(len(pcts_overbuild)):
        for j in range(len(pcts_overbuild)):
            arg_list = [pcts_wind[i],pcts_overbuild[j],lat,lon]
            arg_lists.append(arg_list)

    
    # Run optimization on the methanol cost
    
    # calculate_methanol_cost(50,50,lat,lon)

    # for arg_list in arg_lists:
    #     calculate_methanol_cost(*arg_list)
    
    # start = time.time()
    # with multiprocessing.Pool(num_cores) as p:
    #     p.starmap(calculate_methanol_cost, arg_lists)
    # stop = time.time()
    # print("Elapsed Time: {:.1f} seconds".format(stop-start))
            
    # hi = HoppInterface("./08-wind-solar-electrolyzer-fuel.yaml")
    # hybrid_plant = hi.system

    max_iterations = 5
    optimizer_config = {
        'method':               'CMA-ES',
        'nprocs':               12,
        'generation_size':      10,
        'selection_proportion': .33,
        'prior_scale':          1.0,
        'prior_params':         {
            "grid_angle": {
                "mu": 0.1
                }
            }
        }
    
    start = time.time()
    problem = HybridSizingProblem() #hybrid_plant
    optimizer = OptimizationDriver(problem, recorder=DataRecorder.make_data_recorder("log"), **optimizer_config)
    
    while optimizer.num_iterations() < max_iterations:
        optimizer.step()
        best_score, best_evaluation, best_solution = optimizer.best_solution()
        stop = time.time()
        print(optimizer.num_iterations(), ' ', optimizer.num_evaluations(), best_score, best_evaluation)
        print("Elapsed Time: {:.1f} seconds".format(stop-start))