from hopp.simulation import HoppInterface
import numpy as np
import pandas as pd
import multiprocessing
import time
from examples.efuel.calculate_efuel_cost import calculate_efuel_cost
from examples.efuel.import_sites import import_sites
import matplotlib.pyplot as plt

from typing import Tuple
import numpy as np
from collections import OrderedDict
from hopp.simulation.hybrid_simulation import HybridSimulation
from hopp.tools.optimization import DataRecorder
from hopp.tools.optimization.optimization_problem import OptimizationProblem
from hopp.tools.optimization.optimization_driver import OptimizationDriver




class EfuelHybridProblem(OptimizationProblem):
    """
    Optimize the hybrid power system for fuel production

    Solar and wind electricity are being used to power electrolysis to supply electricity to fuel production.
    This optimizes 2 things: the ratio of solar to wind and the amount of electricity generated.
    Excess generation will be sold to the grid, displacing grid CO2e emissions.
    
    Right now, it is not being passed a HybridSimulation - need to figure out how to automatically
    size the full e-fuel hybrid within a HybridSimulation without using a wrapper function
    """
    
    def __init__(self,
                # simulation: HybridSimulation
                ) -> None:
        """
        Design variables:
            pct_wind (0 to 100) - percentage of wind in the overall electricity mix (gets rounded off by discrete turbine size)
            pct_overbuild (0 to 100) - percentage above the electrolysis requirements to building the renewable production  
        """
        super().__init__()
        # self.simulation = simulation
        self.candidate_dict = OrderedDict({
        "pct_wind": {
            "type": float,
            "prior": {
                "mu": 50, "sigma": 50
            },
            "min": 0, "max": 100
        },
        "pct_overbuild": {
            "type": float,
            "prior": {
                "mu": 50, "sigma": 50
            },
            "min": 0, "max": 100
        }
        })

    def _set_simulation_to_candidate(self,
                                    candidate: np.ndarray,
                                    ):
        pct_wind = candidate[0]
        pct_overbuild = candidate[1]
    
        return pct_wind, pct_overbuild

    def objective(self,
                    candidate: object,
                    candidate_index: int
                    ) -> Tuple:
        candidate_conforming, penalty_conforming = self.conform_candidate_and_get_penalty(candidate)
        pct_wind, pct_overbuild = self._set_simulation_to_candidate(candidate_conforming)
        lat = 32.34
        lon = -98.27
        evaluation = (-calculate_efuel_cost(pct_wind, pct_overbuild, lat, lon),candidate_index)
        score = evaluation[0]
        return score, evaluation, candidate_conforming


if __name__ == '__main__':

    # Enter number of cores in CPU
    num_cores = 14

    # TODO sweep through reactor parameters

    # TODO sweep through years

    # Either sweep through locations or pick a specific location
    lat = 32.337679
    lon = -98.26680948967483

    # Set up to optimize the % wind and % overbuild
    arg_lists = []
    pcts_wind = np.arange(90,101,1)
    pcts_overbuild = np.arange(0,20,20)
    for i in range(len(pcts_wind)):
        for j in range(len(pcts_overbuild)):
            arg_list = [pcts_wind[i],pcts_overbuild[j],lat,lon]
            arg_lists.append(arg_list)


    ### Run optimization on the e-fuel cost
    

    ## One instance
            
    # calculate_efuel_cost(100,0,lat,lon,True)


    ## Grid
        
    # start = time.time()
    # with multiprocessing.Pool(num_cores) as p:
    #     p.starmap(calculate_efuel_cost, arg_lists)
    # stop = time.time()
    # print("Elapsed Time: {:.1f} seconds".format(stop-start))
            

    ## All locations
            
    site_file = 'ngcc_sites_full.csv'
    lats, lons = import_sites(site_file)

    x, y = np.shape(lats)
    lcom_array = np.zeros((x,y))
    CI_array = np.zeros((x,y))
    WC_array = np.zeros((x,y))
    
    for i in range(x):
        lat_list = lats[i]
        lon_list = lons[i]
        arg_lists = []
        for j in range(y):
            arg_list = [100,0,lat_list[j],lon_list[j]]
            arg_lists.append(arg_list)
        
        start = time.time()
        with multiprocessing.Pool(num_cores) as p:
            results = p.starmap(calculate_efuel_cost, arg_lists)
        stop = time.time()
        
        result_array = np.array(results)
        lcom_array[i,:] = result_array[:,0]
        CI_array[i,:] = result_array[:,1]
        WC_array[i,:] = result_array[:,2]
        np.savetxt("lcom.csv",lcom_array,delimiter=',')
        np.savetxt("CI.csv",CI_array,delimiter=',')
        np.savetxt("WC.csv",WC_array,delimiter=',')
        write_time = time.time()
        
        print("Site #{} of {} complete, elapsed time: {:.1f} seconds ({:.1f} to write)".format(i+1,x,write_time-start,write_time-stop))

    ## With optimizer

    # max_iterations = 5
    # optimizer_config = {
    #     'method':               'CEM',
    #     'nprocs':               num_cores,
    #     'generation_size':      num_cores,
    #     'selection_proportion': .33,
    #     'prior_scale':          1.0
    #     }
    
    # start = time.time()
    # problem = EfuelHybridProblem() #hybrid_plant
    # optimizer = OptimizationDriver(problem, recorder=DataRecorder.make_data_recorder("log"), **optimizer_config)
    
    # plt.ion()
    # while optimizer.num_iterations() < max_iterations:
    #     stopped, candidates = optimizer.step()
    #     best_score, best_evaluation, best_solution = optimizer.best_solution()
    #     stop = time.time()
    #     print(optimizer.num_iterations(), ' ', optimizer.num_evaluations(), -best_score, best_solution)
    #     plt.plot(np.array(candidates)[:,0],np.array(candidates)[:,1],'.')
    # print("Elapsed Time: {:.1f} seconds".format(stop-start))

    # calculate_efuel_cost(best_solution[0],best_solution[1],lat,lon)