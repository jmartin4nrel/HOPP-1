from hopp.simulation import HoppInterface
import numpy as np
import pandas as pd
import multiprocessing
import time
from examples.efuel.calculate_efuel_cost import calculate_efuel_cost
from examples.efuel.import_sites import import_sites
import matplotlib.pyplot as plt
from pathlib import Path

from typing import Tuple
import numpy as np
from collections import OrderedDict
from hopp.simulation.hybrid_simulation import HybridSimulation
from hopp.tools.optimization import DataRecorder
from hopp.tools.optimization.optimization_problem import OptimizationProblem
from hopp.tools.optimization.optimization_driver import OptimizationDriver

main_path = Path("inputs/wind-solar-electrolyzer-fuel.yaml")
turndown_path = Path("inputs/methanol-battery.yaml")

dollar_year = 2020 

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
                startup_year = 2020,
                lat = 32.337679,
                lon = -98.26680948967483,
                state = 'TX'
                ) -> None:
        """
        Design variables:
            pct_wind (0 to 100) - percentage of wind in the overall electricity mix (gets rounded off by discrete turbine size)
            pct_overbuild (0 to 100) - percentage above the electrolysis requirements to building the renewable production  
        """
        super().__init__()
        self.startup_year = startup_year
        self.lat = lat
        self.lon = lon
        self.state = state
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
        fuel = 'methanol'
        reactor = 'CO2 hydrogenation'
        catalyst = 'None'
        lca, ci, wc = calculate_efuel_cost(main_path, turndown_path, fuel, reactor, catalyst, pct_wind, pct_overbuild,
                                            dollar_year, self.startup_year, self.lat, self.lon, self.state)
        evaluation = (-lca,candidate_index)
        score = evaluation[0]
        return score, evaluation, candidate_conforming


if __name__ == '__main__':

    resource_dir = Path(__file__).parent/'inputs'
    output_dir = Path(__file__).parent/'outputs'
    
    # Enter number of cores in CPU
    num_cores = 14

    # TODO sweep through reactor parameters
    fuel = 'methanol'
    reactor = 'CO2 hydrogenation'
    # reactor = 'RCC recycle'
    catalyst = 'CZA'

    # TODO sweep through years
    startup_year = 2020
    year_sweep = np.arange(2020,2055,5)

    # Either sweep through locations or pick a specific location
    lat = 32.337679
    lon = -98.26680948967483
    state = 'TX'

    # Set up to optimize the % wind and % overbuild
    arg_lists = []
    pct_overbuild = 0
    pcts_wind = np.arange(50,90.5,10)
    pcts_overbuild = np.arange(0,25,5)
    for i in range(len(pcts_wind)):
        for j in range(len(pcts_overbuild)):
            arg_list = [main_path, turndown_path, fuel, reactor, catalyst, pcts_wind[i], pcts_overbuild[j], dollar_year, startup_year, lat, lon, state]
            arg_lists.append(arg_list)


    ### Run optimization on the e-fuel cost
    

    ## One instance

    # calculate_efuel_cost(main_path, turndown_path, fuel, reactor, catalyst, 90, 80, dollar_year, startup_year, lat, lon, state, True)


    ## Grid
     
    # x = len(pcts_wind)
    # y = len(pcts_overbuild)

    # lcom_array = np.zeros((x,y))
    # CI_array = np.zeros((x,y))
    # WC_array = np.zeros((x,y))

    # start = time.time()
    # with multiprocessing.Pool(num_cores) as p:
    #     results = p.starmap(calculate_efuel_cost, arg_lists)
    # stop = time.time()
    # print("Elapsed Time: {:.1f} seconds".format(stop-start))
    
    # result_array = np.array(results)
    # for i in range(x):
    #     lcom_array[i,:] = result_array[i*y:(i+1)*y,0]
    #     CI_array[i,:] = result_array[i*y:(i+1)*y,1]
    #     WC_array[i,:] = result_array[i*y:(i+1)*y,2]
    # np.savetxt(output_dir/"lcom.csv",lcom_array,delimiter=',')
    # np.savetxt(output_dir/"CI.csv",CI_array,delimiter=',')
    # np.savetxt(output_dir/"WC.csv",WC_array,delimiter=',')

    # All locations
            
    
    lats, lons, states = import_sites(resource_dir/'ngcc_sites_full.csv')

    x, y = np.shape(lats)
    lcom_array = np.zeros((x,y))
    CI_array = np.zeros((x,y))
    WC_array = np.zeros((x,y))
    
    for i in range(x):
        
        # # Multiprocess site block
        # lat_list = lats[i]
        # lon_list = lons[i]
        # arg_lists = []
        # for j in range(y):
        #     arg_list = [main_path, turndown_path, fuel, reactor, catalyst, 100, 0, dollar_year, startup_year, lat_list[j], lon_list[j], states[i]]
        #     arg_lists.append(arg_list)
        
        # start = time.time()
        # with multiprocessing.Pool(num_cores) as p:
        #     results = p.starmap(calculate_efuel_cost, arg_lists)
        # stop = time.time()
        
        # Multiprocess optimizer block
        
        start = time.time()
            
        results = np.zeros((y,3))

        for j in range(y):
            
            lat = lats[i,j]
            lon = lons[i,j]
            state = states[i]
            
            arg_lists = []
            pcts_wind = np.arange(10,110,50)
            pcts_overbuild = np.arange(0,100,50)
            for k in range(len(pcts_wind)):
                for l in range(len(pcts_overbuild)):
                    arg_list = [main_path, turndown_path, fuel, reactor, catalyst, pcts_wind[k], pcts_overbuild[l], dollar_year, startup_year, lat, lon, state]
                    arg_lists.append(arg_list)
            
            with multiprocessing.Pool(num_cores) as p:
                pointresults = p.starmap(calculate_efuel_cost, arg_lists)
            
            pointresults = np.array(pointresults)
            
            lcom = pointresults[:,0]
            ci = pointresults[:,1]
            wc = pointresults[:,2]

            min_arg = np.argmin(lcom)

            results[j,0] = lcom[min_arg]
            results[j,1] = ci[min_arg]
            results[j,2] = wc[min_arg]

        
        stop = time.time() 

        # End of alternating blocks

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

    # for startup_year in year_sweep:

    #     max_iterations = 2
    #     optimizer_config = {
    #         'method':               'CEM',
    #         'nprocs':               num_cores,
    #         'generation_size':      num_cores,
    #         'selection_proportion': .33,
    #         'prior_scale':          1.0
    #         }
        
    #     start = time.time()
    #     problem = EfuelHybridProblem(int(startup_year), lat, lon) #hybrid_plant
    #     optimizer = OptimizationDriver(problem, recorder=DataRecorder.make_data_recorder("log"), **optimizer_config)
        
    #     # plt.ion()
    #     while optimizer.num_iterations() < max_iterations:
    #         stopped, candidates = optimizer.step()
    #         best_score, best_evaluation, best_solution = optimizer.best_solution()
    #         stop = time.time()
    #         print(optimizer.num_iterations(), ' ', optimizer.num_evaluations(), -best_score, best_solution)
    #         # plt.plot(np.array(candidates)[:,0],np.array(candidates)[:,1],'.')
    #     print("Elapsed Time: {:.1f} seconds".format(stop-start))

    #     calculate_efuel_cost(main_path, turndown_path, fuel, reactor, catalyst, best_solution[0], best_solution[1], dollar_year, startup_year, lat, lon, state, True)