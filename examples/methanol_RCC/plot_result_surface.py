import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

current_dir = Path(__file__).parent.absolute()
years = np.arange(2020,2055,5)#[2050,]#
years = [2050]

# sites = ['IA01','TX01']##
site = 'IA01'
# n_sites = len(sites)
n_years = len(years)

plant_size_pcts = np.arange(60,160,20)
wind_pcts = np.arange(10,110,20)
n_plants = len(plant_size_pcts)
n_winds = len(wind_pcts)


X, Y = np.meshgrid(plant_size_pcts,wind_pcts)

plt.clf()
# for i, site in enumerate(sites):
for i, year in enumerate(years):
    results_dir = current_dir/'..'/'resource_files'/'methanol_RCC'/'HOPP_results'/str(year)
    orig_lcoe = np.zeros((n_winds,n_plants))
    lcoe = np.zeros((n_winds,n_plants))
    orig_CI = np.zeros((n_winds,n_plants))
    CI = np.zeros((n_winds,n_plants))
    for j, plant_pct in enumerate(plant_size_pcts):
        for k, wind_pct in enumerate(wind_pcts):

            filename = '{}_plant{:03d}_wind{:02d}_HCO2.txt'.format(site,plant_pct,wind_pct)
            
            read_dir = results_dir/'OrigLCOE'
            with open(Path(read_dir/filename),'r') as file:
                orig_lcoe[k][j] = np.loadtxt(file)
            read_dir = results_dir/'LCOE'
            with open(Path(read_dir/filename),'r') as file:
                lcoe[k][j] = np.loadtxt(file)
            read_dir = results_dir/'OrigCI'
            with open(Path(read_dir/filename),'r') as file:
                orig_CI[k][j] = np.loadtxt(file)
            read_dir = results_dir/'CI'
            with open(Path(read_dir/filename),'r') as file:
                CI[k][j] = np.loadtxt(file)
    
    # plt.subplot(n_sites,2,i*2+1)
    plt.subplot(n_years*2,2,i*4+1)
    plt.contourf(X,Y,orig_lcoe, vmin=0.01, vmax=0.08)
    plt.xlabel('Plant size, % of original estimate')
    plt.ylabel('% wind')
    plt.colorbar(label='$/kWh')

    # lcoe = lcoe*0+.02
    # lcoe[3,3] = 0.0495
    # plt.subplot(n_sites,2,i*2+2)
    plt.subplot(n_years*2,2,i*4+2)
    plt.contourf(X,Y,lcoe, vmin=0.01, vmax=0.08) #
    plt.xlabel('Plant size, % of original estimate')
    plt.ylabel('% wind')
    plt.colorbar(label='$/kWh')

    plt.subplot(n_years*2,2,i*4+3)
    plt.contourf(X,Y,orig_CI, vmin=-30, vmax=100)
    plt.xlabel('Plant size, % of original estimate')
    plt.ylabel('% wind')
    plt.colorbar(label='gCO2e/kWh')

    plt.subplot(n_years*2,2,i*4+4)
    plt.contourf(X,Y,CI, vmin=-30, vmax=100)
    plt.xlabel('Plant size, % of original estimate')
    plt.ylabel('% wind')
    plt.colorbar(label='gCO2e/kWh')

plt.gcf().set_tight_layout(True)
plt.show()