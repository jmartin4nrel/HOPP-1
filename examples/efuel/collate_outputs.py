from pathlib import Path
import os
import pandas as pd
import numpy as np

location = 'lat_32.338_lon_-98.267'

output_list = os.listdir('outputs/')

reactor_df = pd.read_csv('inputs/Reactor_inputs_doe.csv')
    
reactors = list(reactor_df['reactor_tech'])#['RCC recycle','RCC recycle','RCC recycle']#,'CO RCC','CO RCC','CO RCC','CO RCC','CO RCC','CO RCC']#'SMR','CO2 hydrogenation','RCC recycle','CO RCC','CO RCC','CO RCC','CO RCC']
catalysts = list(reactor_df['catalyst'])#['CZA','Ca/CZA','K/CZA']#,'ZA','K/ZA','Na/ZA','Au-Na/ZA','Na/ZA 30','Au-Na/ZA 30']#,'ZA-Z 30','K/ZA-Z 30','ZA 30','K/ZA 30']

num_results = 0
result_reactors = []
result_catalysts = []
for idx, reactor in enumerate(reactors):
    catalyst = catalysts[idx]
    partial_fn = location+'_reactor_'+reactor+'_catalyst_'+catalyst+'_wind_'
    fns = [fn for fn in output_list if partial_fn in fn]
    if len(fns) == 1:
        results = pd.read_csv('outputs/'+fns[0],header=None)
        if num_results == 0:
            results_mat = results.values
        else:
            results_mat = np.hstack((results_mat,results.values))
        result_reactors.append(reactor)
        result_catalysts.append(catalyst)
        num_results += 1
    
result_cols = pd.MultiIndex.from_arrays((result_reactors,result_catalysts))
result_df = pd.DataFrame(results_mat,index=range(len(results.values)),columns=result_cols)
result_df.to_csv('outputs/'+location+'_results.txt')