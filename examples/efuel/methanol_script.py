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

from global_land_mask import globe

main_path = Path("inputs/wind-solar-electrolyzer-fuel.yaml")
turndown_path = Path("inputs/methanol-battery.yaml")

dollar_year = 2020

min_wind = 0
max_wind = 100
avg_wind = (min_wind+max_wind)/2
sig_wind = avg_wind-min_wind

min_bld = 0
max_bld = 50
avg_bld = (min_bld+max_bld)/2
sig_bld = avg_bld-min_bld

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
                startup_year = 2025,
                lat = 32.337679,
                lon = -98.26680948967483,
                state = 'TX',
                wind_cap = 42.4683794096513,
                pv_cap = 24.91935255018996
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
        self.wind_cap = wind_cap
        self.pv_cap = pv_cap
        self.candidate_dict = OrderedDict({
        "pct_wind": {
            "type": float,
            "prior": {
                "mu": avg_wind, "sigma": sig_wind
            },
            "min": min_wind, "max": max_wind
        },
        "pct_overbuild": {
            "type": float,
            "prior": {
                "mu": avg_bld, "sigma": sig_bld
            },
            "min": min_bld, "max": max_bld
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
        lcom, _, _, _, _ = calculate_efuel_cost(main_path, turndown_path, fuel, reactor, catalyst, pct_wind, pct_overbuild,
                                                dollar_year, self.startup_year, self.lat, self.lon, self.state, False, False, False,
                                                indep_hybrid=True)#, wind_ppa_lcoe_ratio=1, solar_ppa_lcoe_ratio=1)#, self.wind_cap, self.pv_cap)
        evaluation = (-lcom,candidate_index)
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
    # reactor = 'CO RCC'
    catalyst = 'K/ZA'

    # TODO sweep through years
    startup_year = 2023
    year_sweep = np.arange(2020,2055,5)

    # Either sweep through locations or pick a specific location
    lat = 32.337679
    lon = -98.26680948967483
    state = 'TX'
    wind_cap = 42.4683794096513
    pv_cap = 26.12#24.91935255018996
    # lat = 43.6575 
    # lon = -70.06423732247559
    # state = 'ME'
    # lat = 46.23308232155108
    # lon = -122.58574166664751
    # state = "WA"
    # lat = 40.403137
    # lon = -122.427904
    # state = "CA"


    # Set up to optimize the % wind and % overbuild
    arg_lists = []
    pct_overbuild = 0
    pcts_wind = np.arange(10,90.5,20)
    pcts_overbuild = np.arange(0,25,5)
    for i in range(len(pcts_wind)):
        for j in range(len(pcts_overbuild)):
            arg_list = [main_path, turndown_path, fuel, reactor, catalyst, pcts_wind[i], pcts_overbuild[j], dollar_year, startup_year, lat, lon, state,
                        False, False, False]
            arg_lists.append(arg_list)


    ### Run optimization on the e-fuel cost
    

    ## One instance

    start = time.time()
    calculate_efuel_cost(main_path, turndown_path, fuel, reactor, catalyst, 100, 0, dollar_year, startup_year, lat, lon, state,
                         True, False, False, indep_hybrid=True, wind_ppa_lcoe_ratio=1.1925, solar_ppa_lcoe_ratio=1.3725)#, 42.4683794096513, 24.91935255018996)
    stop = time.time()
    print("Elapsed Time: {:.1f} seconds".format(stop-start))
    
    ## Grid
     
    # x = len(pcts_wind)
    # y = len(pcts_overbuild)

    # lcom_array = np.zeros((x,y))
    # CI_array = np.zeros((x,y))
    # WC_array = np.zeros((x,y))
    # wind_cap_array = np.zeros((x,y))
    # pv_cap_array = np.zeros((x,y))

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
    #     wind_cap_array[i,:] = result_array[i*y:(i+1)*y,3]*100
    #     pv_cap_array[i,:] = result_array[i*y:(i+1)*y,4]*100
    # np.savetxt(output_dir/"lcom.csv",lcom_array,delimiter=',')
    # np.savetxt(output_dir/"CI.csv",CI_array,delimiter=',')
    # np.savetxt(output_dir/"WC.csv",WC_array,delimiter=',')
    # np.savetxt(output_dir/"wind_cap.csv",wind_cap_array,delimiter=',')
    # np.savetxt(output_dir/"pv_cap.csv",pv_cap_array,delimiter=',')

    ## All locations
            
    # load_caps = True

    # lats, lons, states = import_sites(resource_dir/'ngcc_sites_full.csv')

    # x, y = np.shape(lats)
    # lcom_array = np.zeros((x,y))
    # CI_array = np.zeros((x,y))
    # WC_array = np.zeros((x,y))
    # if load_caps:
    #     wind_cap_array = pd.read_csv("wind_cap.csv",header=None).values
    #     pv_cap_array = pd.read_csv("pv_cap.csv",header=None).values
    # else:
    #     wind_cap_array = np.zeros((x,y))
    #     pv_cap_array = np.zeros((x,y))
    
    # for i in range(x):
        
    #     # Multiprocess site block

    #     # lat_list = lats[i]
    #     # lon_list = lons[i]
    #     # arg_lists = []
    #     # for j in range(y):
    #     #     arg_list = [main_path, turndown_path, fuel, reactor, catalyst, 50, 50, dollar_year, startup_year, lat_list[j], lon_list[j], states[i]]
    #     #     arg_lists.append(arg_list)
        
    #     # start = time.time()
    #     # with multiprocessing.Pool(num_cores) as p:
    #     #     results = p.starmap(calculate_efuel_cost, arg_lists)
    #     # stop = time.time()
        
    #     # Multiprocess optimizer block
        
    #     start = time.time()
            
    #     results = np.zeros((y,3))

    #     for j in range(y):
            
    #         lat = lats[i,j]
    #         lon = lons[i,j]
    #         state = states[i]

    #         if load_caps:
    #             wind_cap = wind_cap_array[i,j]
    #             pv_cap = pv_cap_array[i,j]
    #         else:
    #             wind_cap = None
    #             pv_cap = None
            
    #         if not globe.is_land(lat,lon):
    #             arg_lists = []
    #             pcts_wind = np.arange(10,110,20)
    #             pcts_overbuild = np.arange(0,100,20)
    #             for k in range(len(pcts_wind)):
    #                 for l in range(len(pcts_overbuild)):
    #                     arg_list = [main_path, turndown_path, fuel, reactor, catalyst, pcts_wind[k], pcts_overbuild[l], dollar_year, startup_year, lat, lon, state,
    #                                 False, False, False, wind_cap, pv_cap]
    #                     arg_lists.append(arg_list)
                
    #             with multiprocessing.Pool(num_cores) as p:
    #                 pointresults = p.starmap(calculate_efuel_cost, arg_lists)
                
    #             pointresults = np.array(pointresults)
                
    #             lcom = pointresults[:,0]
    #             ci = pointresults[:,1]
    #             wc = pointresults[:,2]

    #             min_arg = np.argmin(lcom)

    #             results[j,0] = lcom[min_arg]
    #             results[j,1] = ci[min_arg]
    #             results[j,2] = wc[min_arg]
    #         else:
    #             results[j,0] = 0
    #             results[j,1] = 0
    #             results[j,2] = 0

        
    #     stop = time.time() 

    #     # End of alternating blocks

    #     result_array = np.array(results)
    #     lcom_array[i,:] = result_array[:,0]
    #     CI_array[i,:] = result_array[:,1]
    #     WC_array[i,:] = result_array[:,2]
    #     wind_cap_array[i,:] = result_array[:,3]
    #     pv_cap_array[i,:] = result_array[:,4]
    #     np.savetxt("lcom.csv",lcom_array,delimiter=',')
    #     np.savetxt("CI.csv",CI_array,delimiter=',')
    #     np.savetxt("WC.csv",WC_array,delimiter=',')
    #     if not load_caps:
    #         np.savetxt("wind_cap.csv",wind_cap_array,delimiter=',')
    #         np.savetxt("pv_cap.csv",pv_cap_array,delimiter=',')
    #     write_time = time.time()
        
    #     print("Site #{} of {} complete, elapsed time: {:.1f} seconds ({:.1f} to write)".format(i+1,x,write_time-start,write_time-stop))

    ## With optimizer

    # max_iterations = 3
    # optimizer_config = {
    #     'method':               'CMA-ES',
    #     'nprocs':               num_cores,
    #     'generation_size':      num_cores*4,
    #     'selection_proportion': .33,
    #     'prior_scale':          1.0
    #     }
    
    # start = time.time()
    # problem = EfuelHybridProblem(int(startup_year), lat, lon, state, wind_cap, pv_cap) #hybrid_plant
    # optimizer = OptimizationDriver(problem, recorder=DataRecorder.make_data_recorder("log"), **optimizer_config)
    

    # cm = plt.get_cmap("jet")
    # lcom_list = []
    # candidate_X_list = []
    # candidate_Y_list = []
    
    # plt.ioff()
    # while optimizer.num_iterations() < max_iterations:
    #     stopped, candidates = optimizer.step()
    #     best_score, best_evaluation, best_solution = optimizer.best_solution()
    #     stop = time.time()
    #     print(optimizer.num_iterations(), ' ', optimizer.num_evaluations(), -best_score, best_solution)
    #     # candidate_X = np.array(candidates)[:,0]
    #     # candidate_Y = np.array(candidates)[:,1]
    #     for record in optimizer.recorder._records[-1][0]:
    #         lcom_list.append(-record[0])
    #         candidate_X_list.append(record[2][0])
    #         candidate_Y_list.append(record[2][1])
    #     candidate_X = np.array(candidate_X_list)
    #     candidate_Y = np.array(candidate_Y_list)
    #     lcom_array = np.array(lcom_list)
    #     candidate_X = np.maximum(candidate_X,min_wind)
    #     candidate_Y = np.maximum(candidate_Y,min_bld)
    #     candidate_X = np.minimum(candidate_X,max_wind)
    #     candidate_Y = np.minimum(candidate_Y,max_bld)
        
    #     col = [cm((float(i)-np.min(lcom_array))/(np.max(lcom_array)-np.min(lcom_array))) for i in lcom_array]
    #     ax = plt.scatter(candidate_X,candidate_Y, s=10, c=col, marker='o')
    #     ax.set_cmap("jet")
    #     plt.colorbar(ax)
    #     print(min(lcom_array))
    #     print(max(lcom_array))
    #     plt.show()
    #     # plt.plot(candidate_X,candidate_Y,'.')
    # print("Elapsed Time: {:.1f} seconds".format(stop-start))

    # calculate_efuel_cost(main_path, turndown_path, fuel, reactor, catalyst, best_solution[0], best_solution[1], dollar_year, startup_year, lat, lon, state, True, False, False, wind_cap, pv_cap)