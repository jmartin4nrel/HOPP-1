from hopp.simulation import HoppInterface
from hopp.simulation.technologies.hydrogen.electrolysis import run_h2_PEM
import numpy as np
import matplotlib.pyplot as plt

# hi = HoppInterface("./inputs/05-floris-wake-model.yaml")
hi = HoppInterface("./inputs/08-wind-solar-electrolyzer-fuel.yaml")

plant_life = 30 #years
hi.simulate(plant_life)

electrolyzer_size_mw = 100
simulation_length = 8760 #1 year
use_degradation_penalty=True
number_electrolyzer_stacks = 2
grid_connection_scenario = 'off-grid'
EOL_eff_drop = 10
pem_control_type = 'basic'
user_defined_pem_param_dictionary = {
    "Modify BOL Eff": False,
    "BOL Eff [kWh/kg-H2]": [],
    "Modify EOL Degradation Value": True,
    "EOL Rated Efficiency Drop": EOL_eff_drop,
    }

hybrid_plant = hi.system
solar_plant_power = np.array(hybrid_plant.pv.generation_profile[0:simulation_length])
wind_plant_power = np.array(hybrid_plant.wind.generation_profile[0:simulation_length])
hybrid_plant_generation_profile = solar_plant_power + wind_plant_power

h2_results, H2_Timeseries, H2_Summary,energy_input_to_electrolyzer =\
run_h2_PEM.run_h2_PEM(hybrid_plant_generation_profile,
electrolyzer_size_mw,
plant_life, number_electrolyzer_stacks,[],
pem_control_type,100,user_defined_pem_param_dictionary,
use_degradation_penalty,grid_connection_scenario,[])

# Total hydrogen output timeseries (kg-H2/hour)
hydrogen_production_kg_pr_hr = H2_Timeseries['hydrogen_hourly_production']
# Rated/maximum hydrogen production from electrolysis system
max_h2_pr_h2 = h2_results['new_H2_Results']['Rated BOL: H2 Production [kg/hr]']
#x-values as hours of year
hours_of_year = np.arange(0,len(hydrogen_production_kg_pr_hr),1)

hour_start = 2000
n_hours = 72
hour_end = hour_start + n_hours

fig,ax=plt.subplots(2,1,sharex=True)
fig.set_figwidth(8.0)
fig.set_figheight(6.0)

ax[0].plot(hours_of_year[hour_start:hour_end],hydrogen_production_kg_pr_hr[hour_start:hour_end],lw=2,c='green',label='H2 Produced')
ax[0].plot(hours_of_year[hour_start:hour_end],max_h2_pr_h2*np.ones(n_hours),lw=2,ls='--',c='red',label='Rated H2')
ax[0].legend(loc='center right')
ax[0].set_ylabel('Hydrogen [kg]',fontsize=14)
ax[0].set_xlim((hour_start,hour_end-1))

ax[1].plot(hours_of_year[hour_start:hour_end],solar_plant_power[hour_start:hour_end]/1e3,lw=2,ls='--',c='orange',label='Solar')
ax[1].plot(hours_of_year[hour_start:hour_end],wind_plant_power[hour_start:hour_end]/1e3,lw=2,ls=':',c='blue',label='Wind')
ax[1].plot(hours_of_year[hour_start:hour_end],hybrid_plant_generation_profile[hour_start:hour_end]/1e3,lw=2,alpha=0.5,c='green',label='Wind + Solar')
ax[1].set_ylabel('Renewable Energy [MWh]',fontsize=14)
ax[1].set_xlabel('Time [hour of year]',fontsize=14)
ax[1].set_xlim((hour_start,hour_end-1))
ax[1].legend()
fig.tight_layout()

plt.ioff()
plt.show()