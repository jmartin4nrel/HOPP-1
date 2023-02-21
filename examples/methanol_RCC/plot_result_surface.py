import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

current_dir = Path(__file__).parent.absolute()
years = np.arange(2020,2055,5)
    

site = 'IA01'#TX01'
n_years = len(years)

plant_size_pcts = [60,70,80,90,100]
wind_pcts = [10,30,50,70,90]
n_plants = len(plant_size_pcts)
n_winds = len(wind_pcts)


X, Y = np.meshgrid(plant_size_pcts,wind_pcts)

plt.clf()
for i, year in enumerate(years):
    results_dir = current_dir/'..'/'resource_files'/'methanol_RCC'/'HOPP_results'/str(year)
    orig_lcoe = np.zeros((n_winds,n_plants))
    lcoe = np.zeros((n_winds,n_plants))
    for j, plant_pct in enumerate(plant_size_pcts):
        for k, wind_pct in enumerate(wind_pcts):

            filename = '{}_plant{:03d}_wind{:02d}.txt'.format(site,plant_pct,wind_pct)
            
            read_dir = results_dir/'OrigLCOE'
            with open(Path(read_dir/filename),'r') as file:
                orig_lcoe[k][j] = np.loadtxt(file)
            read_dir = results_dir/'LCOE'
            with open(Path(read_dir/filename),'r') as file:
                lcoe[k][j] = np.loadtxt(file)
    
    plt.subplot(n_years,2,i*2+1)
    plt.contourf(X,Y,orig_lcoe)
    plt.xlabel('Plant size, % of original estimate')
    plt.ylabel('% wind')
    plt.colorbar(label='$/kWh')

    plt.subplot(n_years,2,i*2+2)
    plt.contourf(X,Y,lcoe)
    plt.xlabel('Plant size, % of original estimate')
    plt.ylabel('% wind')
    plt.colorbar(label='$/kWh')

plt.show()