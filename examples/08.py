# %% [markdown]
# # Simple Wind/Solar/H2/Methanol Hybrid Plant Example
# ---
# In this example, we will walk through the process of simulating a hybrid renewable energy system using the Hybrid Optimization Performance Platform ([HOPP](https://github.com/NREL/HOPP)) library. We will simulate a hybrid system at a given location consisting of both wind and solar electricity, sent to a hydrogen electrolyzer, whose hydrogen output is then sent to make methanol, and show how to access the simulation outputs.

# %% [markdown]
# ### Import Required Modules
# We start by importing the necessary modules and setting up our working environment.

# %%
from hopp.simulation import HoppInterface
from hopp.simulation.technologies.hydrogen.electrolysis import run_h2_PEM
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ### Create the HOPP Model
# To generate the HOPP Model, instantiate the `HoppInterface` class and supply the required YAML configuration. In this example, the yaml activates the "simple" financial model that has fewer, stripped-down inputs and outputs as compared to the PySAM model.
# 
# Within the YAML configuration, you have the flexibility to define the plant's location details and configure the associated technologies, in this case wind, solar pv, and a fuel plant that generates methanol from hydrogen and CO2.
# 
# In this example, we use the Flatirons site as a sample location and configure the wind and solar data for this particular site using pre-existing data files.

# %%
hi = HoppInterface("./inputs/08-wind-solar-electrolyzer-fuel.yaml")

# %% [markdown]
# ### Use the fuel plant to size the wind, solar, co2 source, and electrolyzer
# 
# Only the simple methanol reactor model will be run, to determine the hydrogen and co2 input needed, and in turn the electricity input needed to the electrolyzer, to size the wind, solar, and electrolyzer components.

# %%
# Run just the reactor model
hi.system.fuel.simulate_flow(1)
total_elec_kw = hi.system.fuel._system_model.input_streams_kw['electricity']

# Use the calculated co2 input flowrate to size the co2 source plant
co2_kg_s = hi.system.fuel._system_model.input_streams_kg_s['carbon dioxide']
getattr(hi.system,'co2').value('co2_kg_s',co2_kg_s)

# Calculate the (discrete) wind plant size needed based on an estimated capacity factor and the desired percentage of the total wind/pv output from wind
percent_wind = 90
wind_cap_factor = 0.185
wind_cap_kw = total_elec_kw*percent_wind/100/wind_cap_factor
turb_rating_kw = getattr(hi.system,'wind').value('turb_rating')
num_turbines = int(np.round(wind_cap_kw/turb_rating_kw,0))
getattr(hi.system,'wind').value('num_turbines',num_turbines)
wind_cap_kw = num_turbines*turb_rating_kw
percent_wind = wind_cap_kw*wind_cap_factor/total_elec_kw*100

# # Widen site to match number of turbines needed
# Site = hi.system.site
# # For site area: square with sides = sqrt of number of turbines times rotor diameter times ten
# d = hi.system.wind.rotor_diameter
# n = hi.system.wind.num_turbines
# side = 10*d*n**.5
# site = getattr(hi.system,'site')
# setattr(site,'vertices',np.array([[0,0],[0,side],[side,side],[side,0]]))
# hi.system.layout.wind.reset_grid(n)

# Calculate the (continuous) pv plant size needed based on an estimated capacity factor and the wind plant size
percent_pv = 100-percent_wind
pv_cap_factor = 0.288
pv_cap_kw = total_elec_kw*percent_pv/100/pv_cap_factor
getattr(hi.system,'pv').value('system_capacity_kw',pv_cap_kw)

# Calculate the electrolyzer and interconnect size needed based on an estimated capacity factor
electrolyzer_cap_factor = 0.97
electrolyzer_cap_kw = total_elec_kw/electrolyzer_cap_factor
getattr(hi.system,'grid').value('interconnect_kw',electrolyzer_cap_kw)


# %% [markdown]
# ### Run the Simulation and Set the Load
# Simulate the hybrid renewable energy system. First, simulate for 1 year to determine the generation schedule, and when grid electricity will be needed to keep the electrolyzer near full capacity. Use this to generate and set the load schedule. Then, simulate for a specified number of years (in this case, 30 years).

# %%
# Simulate plant for 1 year 
plant_life = 1 #years
hi.simulate(plant_life)

# Determine the capacity threshold at which grid electricity must be bought to meet electrolyzer demand
cap_thresh = electrolyzer_cap_factor
needed_kwh = electrolyzer_cap_factor*electrolyzer_cap_kw*8760
total_kwh = electrolyzer_cap_kw*8760
timestep_h = 8760/len(hi.system.generation_profile['hybrid'])
while total_kwh > needed_kwh:
    when_above = [i>=electrolyzer_cap_kw for i in hi.system.generation_profile['hybrid']]
    when_below = [i<=electrolyzer_cap_kw*cap_thresh for i in hi.system.generation_profile['hybrid']]
    when_in_between = list(np.logical_and(np.logical_not(when_above),np.logical_not(when_below)))
    total_kwh = sum(when_above)*electrolyzer_cap_kw*timestep_h + \
                sum(when_below)*electrolyzer_cap_kw*cap_thresh*timestep_h + \
                sum(np.multiply(hi.system.generation_profile['hybrid'],when_in_between))*timestep_h
    if total_kwh > needed_kwh:
        cap_thresh -= 0.001
    else:
        makeup_kwh = needed_kwh - sum(when_above)*electrolyzer_cap_kw*timestep_h - \
                                    sum(np.multiply(hi.system.generation_profile['hybrid'],when_in_between))*timestep_h
        makeup_kw = makeup_kwh/sum(when_below)/timestep_h
        cap_thresh = makeup_kw/electrolyzer_cap_kw

# Make load schedule (in MEGAwatts)
load_schedule = [i/1000 for i in hi.system.generation_profile['hybrid']]
for i, gen in enumerate(load_schedule):
    if when_above[i]:
        load_schedule[i] = electrolyzer_cap_kw/1000
    elif when_below[i]:
        load_schedule[i] = electrolyzer_cap_kw*cap_thresh/1000
hi.system.site.desired_schedule = load_schedule
hi.system.site.follow_desired_schedule = True

# Simulate plant for 30 years, getting curtailment (will be sold to grid) and missed load (will be purchased from grid)
plant_life = 1 #years
hi.simulate(plant_life)
sell_kw = hi.system.grid.schedule_curtailed
buy_kw = hi.system.grid.missed_load


# %% [markdown]
# ### Set electrolyzer and project parameters
# in this example, we're simulating an on-grid electrolyzer system. We define electrolyzer stack size as close to 50 MW as posssible to meet the total capacity. The other inputs are reasonable default values, and will be discussed further in future examples.

# %%
electrolyzer_size_mw = electrolyzer_cap_kw/1000
simulation_length = 8760 #1 year
use_degradation_penalty=True
number_electrolyzer_stacks = round(electrolyzer_size_mw/50)
grid_connection_scenario = 'off-grid'
EOL_eff_drop = 10
pem_control_type = 'basic'
user_defined_pem_param_dictionary = {
    "Modify BOL Eff": False,
    "BOL Eff [kWh/kg-H2]": [],
    "Modify EOL Degradation Value": True,
    "EOL Rated Efficiency Drop": EOL_eff_drop,
}

# %% [markdown]
# ### Retrieve power generation profile from wind and solar components
# 
# ``solar_plant_power`` is the solar generation profile, and ``wind_plant_power`` is the wind generation profile, which combine to ``renewable_generation_profile``. Then, power is bought (``bought_power``) and sold (``sold_power``) from the grid to send a relatively flat input profile (``electrolyzer_profile``) to the electrolyzer. These are in units of kWh.

# %%
hybrid_plant = hi.system
solar_plant_power = np.array(hybrid_plant.pv.generation_profile[0:simulation_length])
wind_plant_power = np.array(hybrid_plant.wind.generation_profile[0:simulation_length])
renewable_generation_profile = solar_plant_power + wind_plant_power
sold_power = np.array(sell_kw[0:simulation_length])
bought_power = np.array(buy_kw[0:simulation_length])
electrolyzer_profile = solar_plant_power + wind_plant_power - sold_power + bought_power

# %% [markdown]
# ### Run the electrolyzer
# 
# The key electrolyzer inputs are:
# - ``hybrid_plant_generation_profile``: energy input to the electrolyzer
# - ``electrolyzer_size_mw``: total installed electrolyzer capacity
# - ``number_electrolyzer_stacks``: how many individual stacks make up the electrolyzer system.
# 
# The outputs are:
# - ``h2_results``: aggregated performance information
# - ``H2_Timeseries``: hourly time-series of hydrogen production and other key parameters
# - ``H2_Summary``: averages or totals of performance data over the entire simulation
# - ``energy_input_to_electrolyzer``: for this example (off-grid scenario), this is the same as ``hybrid_plant_generation_profile``.

# %%
h2_results, H2_Timeseries, H2_Summary,energy_input_to_electrolyzer =\
run_h2_PEM.run_h2_PEM(electrolyzer_profile,
electrolyzer_size_mw,
plant_life, number_electrolyzer_stacks,[],
pem_control_type,100,user_defined_pem_param_dictionary,
use_degradation_penalty,grid_connection_scenario,[])

# %% [markdown]
# ### Get the time-series data and rated hydrogen production

# %%
# Total hydrogen output timeseries (kg-H2/hour)
hydrogen_production_kg_pr_hr = H2_Timeseries['hydrogen_hourly_production']
# Rated/maximum hydrogen production from electrolysis system
max_h2_pr_h2 = h2_results['new_H2_Results']['Rated BOL: H2 Production [kg/hr]']
#x-values as hours of year
hours_of_year = np.arange(0,len(hydrogen_production_kg_pr_hr),1)

# %% [markdown]
# ### Send hydrogen production to the methanol reactor
# 
# In this very simple version of the methanol reactor model (SimpleReactor), the hydrogen output is just linearly scaled to methanol output

# %%
# Get ratio of hydrogen to methanol in reactor
h2in = hi.system.fuel._system_model.input_streams_kg_s['hydrogen']
meoh_out = hi.system.fuel.fuel_prod_kg_s
meoh_h2_ratio = meoh_out/h2in

# Scale hydrogen production to methanol production
methanol_production_kg_pr_hr = [i*meoh_h2_ratio for i in hydrogen_production_kg_pr_hr]
max_meoh_pr_hr = max_h2_pr_h2*meoh_h2_ratio
avg_meoh_pr_hr = np.mean(methanol_production_kg_pr_hr)

# %% [markdown]
# ### Plot results (Optional)
# We're only going to look at 72 hour frame of the results, starting at hour 2000. We will also see the calculated levelized cost of methanol (LCOM).
# 
# The top plot shows the renewable energy produced (green solid line) and the individual wind (blue dotted line) and solar (orange dashed line) generation profiles.
# 
# The middle plot shows the energy bought (red dashed line) and sold (purple dotted line) from the grid to get to the total energy input to the electrolyzer (grey solid line).
# 
# The bottom plot shows the methanol produced (green solid line) with the average methanol production over the whole lifetime indicated by a red dashed line.
# 

# %%
hour_start = 0
n_hours = 8760
hour_end = hour_start + n_hours

fig,ax=plt.subplots(3,1,sharex=True)
fig.set_figwidth(8.0)
fig.set_figheight(9.0)

ax[0].plot(hours_of_year[hour_start:hour_end],solar_plant_power[hour_start:hour_end]/1e3,lw=2,ls='--',c='orange',label='Solar')
ax[0].plot(hours_of_year[hour_start:hour_end],wind_plant_power[hour_start:hour_end]/1e3,lw=2,ls=':',c='blue',label='Wind')
ax[0].plot(hours_of_year[hour_start:hour_end],renewable_generation_profile[hour_start:hour_end]/1e3,lw=2,alpha=0.5,c='green',label='Wind + Solar')
ax[0].set_ylabel('Renewable Energy [MWh]',fontsize=14)
ax[0].set_xlim((hour_start,hour_end-1))
ax[0].legend()

ax[1].plot(hours_of_year[hour_start:hour_end],renewable_generation_profile[hour_start:hour_end]/1e3,lw=2,alpha=0.5,c='green',label='Wind + Solar')
ax[1].plot(hours_of_year[hour_start:hour_end],bought_power[hour_start:hour_end]/1e3,lw=2,ls='--',c='red',label='Bought from Grid')
ax[1].plot(hours_of_year[hour_start:hour_end],sold_power[hour_start:hour_end]/1e3,lw=2,ls=':',c='purple',label='Sold to Grid')
ax[1].plot(hours_of_year[hour_start:hour_end],electrolyzer_profile[hour_start:hour_end]/1e3,lw=2,alpha=0.5,c='black',label='Electrolyzer')
ax[1].set_ylabel('Total Energy [MWh]',fontsize=14)
ax[1].set_xlim((hour_start,hour_end-1))
ax[1].legend()

ax[2].plot(hours_of_year[hour_start:hour_end],methanol_production_kg_pr_hr[hour_start:hour_end],lw=2,c='green',label='Methanol Produced')
ax[2].plot(hours_of_year[hour_start:hour_end],avg_meoh_pr_hr*np.ones(n_hours),lw=2,ls='--',c='red',label='Average methanol')
ax[2].legend(loc='center right')
ax[2].set_ylabel('Methanol [kg/hr]',fontsize=14)
ax[2].set_xlabel('Time [hour of year]',fontsize=14)
ax[2].set_xlim((hour_start,hour_end-1))

fig.tight_layout()

lb_kg = 2.208
MJ_kg = 20.1
MJ_MMBTU = 1055.

print("Annual methanol production, tonne/yr: {:f}".format(hi.system.fuel.annual_mass_kg/1000))
print("Levelized cost of methanol (LCOM), $/kg: {:.2f}".format(hi.system.fuel._financial_model.lc_kg))
# print("Levelized cost of methanol (LCOM), $/lb: {:.2f}".format(hi.system.fuel._financial_model.lc_kg/lb_kg))
# print("Levelized cost of methanol (LCOM), $/tonne: {:.2f}".format(hi.system.fuel._financial_model.lc_kg*1000))
# print("Levelized cost of methanol (LCOM), $/MJ: {:.2f}".format(hi.system.fuel._financial_model.lc_kg/MJ_kg))
# print("Levelized cost of methanol (LCOM), $/MMBTU: {:.2f}".format(hi.system.fuel._financial_model.lc_kg/MJ_kg*MJ_MMBTU))
# print("Levelized cost of methanol (LCOM), $/MMBTU: {:.2f}".format(hi.system.fuel._financial_model.lc_kg/MJ_kg*MJ_MMBTU))
print("Carbon Intensity (CI), kg/kg-MeOH: {:.2f}".format(hi.system.lca['co2_kg_kg']))


