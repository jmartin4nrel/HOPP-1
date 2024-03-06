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
import copy
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
total_elec_kw = np.mean((hi.system.fuel._system_model.input_streams_kw['electricity']))

# Use the calculated co2 input flowrate to size the co2 source plant, switch off costs if using raw flue gas
co2_kg_s = np.mean(hi.system.fuel._system_model.input_streams_kg_s['carbon dioxide'])
getattr(hi.system,'co2').value('co2_kg_s',co2_kg_s)
if hi.system.tech_config.co2.capture_model == 'None':
    hi.system.co2._financial_model.voc_kg = 0.
    hi.system.tech_config.co2.lca['co2_kg_kg'] = 0.
    hi.system.ng._system_model.annual_mass_kg = 0.
hi.system.co2.simulate_flow(1)
ng_kg_s = np.mean(hi.system.co2._system_model.input_streams_kg_s['natural gas'])
hi.system.ng._system_model.ng_kg_s = ng_kg_s
getattr(hi.system,'ng').value('ng_kg_s',ng_kg_s)
hi.system.tech_config.ng.ng_kg_s = ng_kg_s
hi.system.ng.config.ng_kg_s = ng_kg_s
hi.system.ng.ng_kg_s = ng_kg_s

# Calculate the (discrete) wind plant size needed based on an estimated capacity factor and the desired percentage of the total wind/pv output from wind
percent_wind = 90
percent_overbuild = 1
overbuild_elec_kw = total_elec_kw*(100+percent_overbuild)/100
wind_cap_factor = 0.42
wind_cap_kw = overbuild_elec_kw*percent_wind/100/wind_cap_factor
turb_rating_kw = getattr(hi.system,'wind').value('turb_rating')
num_turbines = int(np.round(wind_cap_kw/turb_rating_kw,0))
getattr(hi.system,'wind').value('num_turbines',num_turbines)
hi.system.wind._financial_model.system_capacity_kw = hi.system.wind._system_model.Farm.system_capacity
hi.system.wind.simulate_power(1)
wind_cap_factor = getattr(hi.system,'wind').value('capacity_factor')/100
wind_cap_kw = overbuild_elec_kw*percent_wind/100/wind_cap_factor
num_turbines = np.ceil(wind_cap_kw/turb_rating_kw)
getattr(hi.system,'wind').value('num_turbines',num_turbines)
wind_cap_kw = hi.system.wind._system_model.Farm.system_capacity
hi.system.wind._financial_model.system_capacity_kw = wind_cap_kw
percent_wind = wind_cap_kw*wind_cap_factor/overbuild_elec_kw*100

# # Widen site to match number of turbines needed
# Site = hi.system.site
# # For site area: square with sides = sqrt of number of turbines times rotor diameter times ten
# d = hi.system.wind.rotor_diameter
# n = hi.system.wind.num_turbines
# side = 10*d*n**.5
# getattr(hi.system.site,'vertices',np.array([[0,0],[0,side],[side,side],[side,0]]))

# Calculate the (continuous) pv plant size needed based on an estimated capacity factor and the wind plant size
percent_pv = 100-percent_wind
pv_cap_factor = 0.22
pv_cap_kw = overbuild_elec_kw*percent_pv/100/pv_cap_factor
getattr(hi.system,'pv').value('system_capacity_kw',pv_cap_kw)
hi.system.pv.simulate_power(1)
pv_cap_factor = hi.system.pv._system_model.Outputs.capacity_factor/100
pv_cap_kw = overbuild_elec_kw*percent_pv/100/pv_cap_factor
getattr(hi.system,'pv').value('system_capacity_kw',pv_cap_kw)

# Calculate the electrolyzer and interconnect size needed based on an estimated capacity factor
electrolyzer_cap_factor = 0.97
electrolyzer_cap_kw = total_elec_kw/electrolyzer_cap_factor
sales_cap_kw = wind_cap_kw+pv_cap_kw-electrolyzer_cap_kw
getattr(hi.system,'grid').value('interconnect_kw',wind_cap_kw+pv_cap_kw)
getattr(hi.system,'grid_sales').value('interconnect_kw',sales_cap_kw)
getattr(hi.system,'grid_purchase').value('interconnect_kw',electrolyzer_cap_kw)
getattr(hi.system,'electrolyzer').value('system_capacity_kw',electrolyzer_cap_kw)

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

# Make electrolyzer/sales/purchase profiles
sell_kw = [0.0]*8760
buy_kw = [0.0]*8760
load_schedule = list(hi.system.generation_profile['hybrid'])
for i, gen in enumerate(load_schedule):
    if when_above[i]:
        sell_kw[i] = electrolyzer_cap_kw-gen
        load_schedule[i] = electrolyzer_cap_kw
    elif when_below[i]:
        load_schedule[i] = electrolyzer_cap_kw*cap_thresh
        buy_kw[i] = electrolyzer_cap_kw*cap_thresh-gen
hi.system.electrolyzer.generation_profile = load_schedule
hi.system.grid_sales.generation_profile = sell_kw
hi.system.grid_purchase.generation_profile = buy_kw

# Simulate plant for 30 years, getting curtailment (will be sold to grid) and missed load (will be purchased from grid)
plant_life = 1 #years
hi.simulate(plant_life)


# %% [markdown]
# ### Retrieve power generation and flow profiles from components
# 
# ``solar_plant_power`` is the solar generation profile, and ``wind_plant_power`` is the wind generation profile, which combine to ``renewable_generation_profile``. Then, power is bought (``bought_power``) and sold (``sold_power``) from the grid to send a relatively flat input profile (``electrolyzer_profile``) to the electrolyzer. These are in units of kWh. Then, hydrogen and methanol profiles are in terms of kg/s by default in the hopp structure, and converted to kg/hr

# %%
hybrid_plant = hi.system
solar_plant_power = np.array(hybrid_plant.pv.generation_profile)
wind_plant_power = np.array(hybrid_plant.wind.generation_profile)
renewable_generation_profile = solar_plant_power + wind_plant_power
sold_power = np.array(hybrid_plant.grid_sales.generation_profile)
bought_power = np.array(hybrid_plant.grid_purchase.generation_profile)
electrolyzer_profile = np.array(hybrid_plant.electrolyzer.generation_profile)

# Total hydrogen output timeseries (kg-H2/hour)
hydrogen_production_kg_s = hybrid_plant.electrolyzer._system_model.output_streams_kg_s['hydrogen']
hydrogen_production_kg_pr_hr = hydrogen_production_kg_s*3600
# Rated/maximum hydrogen production from electrolysis system
max_h2_pr_h2 = np.max(hydrogen_production_kg_pr_hr)
avg_h2_pr_hr = np.mean(hydrogen_production_kg_pr_hr)
#x-values as hours of year
hours_of_year = np.arange(0,len(hydrogen_production_kg_pr_hr),1)
methanol_production_kg_s = hybrid_plant.fuel._system_model.output_streams_kg_s['methanol']
methanol_production_kg_pr_hr = methanol_production_kg_s*3600
max_meoh_pr_hr = np.max(methanol_production_kg_pr_hr)
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
hour_start = 2000
n_hours = 72
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

ax[2].plot(hours_of_year[hour_start:hour_end],hydrogen_production_kg_pr_hr[hour_start:hour_end],lw=2,c='green',label='Hydrogen Produced')
ax[2].plot(hours_of_year[hour_start:hour_end],avg_h2_pr_hr*np.ones(n_hours),lw=2,ls='--',c='red',label='Average hydrogen')
ax[2].legend(loc='center right')
ax[2].set_ylabel('Hydrogen [kg/hr]',fontsize=14)
ax[2].set_xlabel('Time [hour of year]',fontsize=14)
ax[2].set_xlim((hour_start,hour_end-1))

# ax[2].plot(hours_of_year[hour_start:hour_end],methanol_production_kg_pr_hr[hour_start:hour_end],lw=2,c='green',label='Methanol Produced')
# ax[2].plot(hours_of_year[hour_start:hour_end],avg_meoh_pr_hr*np.ones(n_hours),lw=2,ls='--',c='red',label='Average methanol')
# ax[2].legend(loc='center right')
# ax[2].set_ylabel('Methanol [kg/hr]',fontsize=14)
# ax[2].set_xlabel('Time [hour of year]',fontsize=14)
# ax[2].set_xlim((hour_start,hour_end-1))

fig.tight_layout()

lb_kg = 2.208
MJ_kg = 20.1
MJ_MMBTU = 1055.

print((np.sum(sum(sold_power)+sum(bought_power)))/np.sum(electrolyzer_profile))
print("Annual methanol production, tonne/yr: {:f}".format(hi.system.fuel.annual_mass_kg/1000))
print("Levelized cost of methanol (LCOM), $/kg: {:.3f}".format(hi.system.lc))
print(hi.system.lc_breakdown)
print("Carbon Intensity (CI), kg/kg-MeOH: {:.3f}".format(hi.system.lca['co2_kg_kg']))
print(hi.system.lca_breakdown)


