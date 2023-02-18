"""
This file shows an example of how to set up a gradinet-based optiization with
pyoptsparse. Currently, the LCOH with respect to wind capacity is discontinuous,
because HOPP rounds wind capacity to integer values of wind turbines. 
h2_setup_optimize currently fixes the wind energy production to be continuous, but
not the costs. So, this file will have trouble running for certain values, but
can still be used as a reference for setting up a gradient-based optimization.
"""

from numpy.lib.npyio import save
from examples.methanol_RCC.rcc_setup_optimize import calculate_lcoe_rcc_optimizer
import pyoptsparse
import pandas as pd
import warnings
import os
import numpy as np
import csv
warnings.filterwarnings("ignore")

def objective_function(x):
    """
    This is the onjective function to be used in the gradient-based optimization.
    Right now it is setup with global variables, which should be fixed soon.
    This objective function is setup to run with pyoptsparse, meaning that it takes
    a dictionary (x in this case) as the input, and returns funcs and fail. funcs is a
    dictionary with the objective, and constraints if there are any, and fail is the flag
    indicating if the function failed.
    """
    
    solar_capacity_mw = x["solar_capacity_mw"]    

    lcoe = calculate_lcoe_rcc_optimizer(solar_capacity_mw)

    funcs = {}
    fail = False
    funcs["lcoe"] = lcoe*1e4
    
    return funcs, fail


if __name__=="__main__":

    import time

    start_time = time.time()

    start_solar = 350
    
    x = {}
    x["solar_capacity_mw"] = start_solar
    
    funcs,_ = objective_function(x)
    start_lcoe = funcs["lcoe"]
    print("start_lcoe: ", start_lcoe)

    optProb = pyoptsparse.Optimization("optimize_sizing",objective_function)
    optProb.addVar("solar_capacity_mw",type="c",lower=100,upper=1000,value=start_solar)
    
    optProb.addObj("lcoe")
    optimize = pyoptsparse.SLSQP()
    optimize.setOption("MAXIT",value=5)
    # optimize.setOption("ACC",value=1E-6)
    # optimize = pyoptsparse.SNOPT()
    

    print("start GB optimization")
    solution = optimize(optProb,sens="FD")
    print("******************************************")
    print("finished optimization")

    opt_DVs = solution.getDVs()
    opt_solar = opt_DVs["solar_capacity_mw"]
    
    funcs,fail = objective_function(opt_DVs)
    opt_lcoe = funcs["lcoe"]/1e4
    time_to_run = time.time()-start_time

    print("opt_lcoe: ", opt_lcoe)
    print("opt_solar_mw: ", opt_solar)
    print("time_to_run: ", time_to_run)