'''
weeklong_sim.py

Emulates sending a power demand signal from HOPP to IESS controller over a week
'''

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hybrid.sites import SiteInfo, flatirons_site
from hybrid.validation.wind.iessGE15.ge15_wind_forecast_parse import process_wind_forecast
from hybrid.validation.solar.iessFirstSolar.firstSolar_forecast_parse import process_solar_forecast
from hybrid.validation.examples.tune_iess import tune_iess
from hybrid.hybrid_simulation import HybridSimulation
from hybrid.dispatch.plot_tools import plot_battery_output, plot_battery_dispatch_error, plot_generation_profile

# Pick time period to simulate
sim_start = '07/28/22'
sim_end = '08/14/22'

# Set path to forecast files
current_dir = Path(__file__).parent.absolute()
validation_dir = current_dir/'..'/'..'
wind_dir = validation_dir/'wind'/'iessGE15'
solar_dir = validation_dir/'solar'/'iessFirstSolar'
wind_files =   {'speed_m_s':'Wind Speed-data-as-seriestocolumns-2023-02-13 20_22_45.csv',
                'gusts_m_s':'Gusts-data-as-seriestocolumns-2023-02-14 15_18_13.csv',
                'dir_deg':  'Wind Direction-data-as-seriestocolumns-2023-02-13 20_24_43.csv',
                'temp_C':   'Temperature-data-as-seriestocolumns-2023-02-13 16_23_32.csv',
                'pres_mbar':'Pressure-data-as-seriestocolumns-2023-02-13 16_19_59.csv'}
solar_files =  {'GHI_W_m2': 'GHI-data-as-seriestocolumns-2023-02-13 16_05_04.csv',
                'DHI_W_m2': 'DIFI-data-as-seriestocolumns-2023-02-13 16_06_43.csv',
                'DNI_W_m2': 'DNI-data-as-seriestocolumns-2023-02-13 16_06_07.csv'}
wind_fp = {}
for key, fn in wind_files.items(): wind_fp[key] = wind_dir/fn
solar_fp = {}
for key, fn in solar_files.items(): solar_fp[key] = solar_dir/fn

# Set hours ahead and hours offset in each column of forecast file
wind_hours_ahead = {'speed_m_s':np.arange(0,168),
                    'gusts_m_s':np.arange(0,168),
                    'dir_deg':  np.arange(0,168),
                    'temp_C':   np.concatenate((np.arange(24,216,24),np.arange(0,24))),
                    'pres_mbar':np.concatenate((np.arange(24,216,24),np.arange(0,24)))}
wind_hours_offset ={'speed_m_s':np.arange(1,169),
                    'gusts_m_s':np.arange(1,169),
                    'dir_deg':  np.arange(1,169),
                    'temp_C':   np.concatenate((np.arange(0,192,24),np.arange(0,24))),
                    'pres_mbar':np.concatenate((np.arange(0,192,24),np.arange(0,24)))}
solar_hours_ahead ={'GHI_W_m2': np.concatenate((np.arange(24,216,24),np.arange(0,24))),
                    'DHI_W_m2': np.concatenate((np.arange(24,216,24),np.arange(0,24))),
                    'DNI_W_m2': np.concatenate((np.arange(24,216,24),np.arange(0,24)))}
solar_hours_offset={'GHI_W_m2': np.concatenate((np.arange(0,192,24),np.arange(0,24))),
                    'DHI_W_m2': np.concatenate((np.arange(0,192,24),np.arange(0,24))),
                    'DNI_W_m2': np.concatenate((np.arange(0,192,24),np.arange(0,24)))}

# Set path to resource files
base_dir = validation_dir/'..'/'..'
wind_res_dir = base_dir/'resource_files'/'wind'
solar_res_dir = base_dir/'resource_files'/'solar'
wind_res_fp = wind_res_dir/'wind_m5_YYYY.srw'
solar_res_fp = solar_res_dir/'solar_m2_YYYY.csv'

# Set years and setup forecast
forecast_year = 2022
resource_years = [2019,2020,2021,2022]
wind_dict = process_wind_forecast(wind_fp,wind_hours_ahead,wind_hours_offset,forecast_year,wind_res_fp,resource_years)
solar_dict = process_solar_forecast(solar_fp,solar_hours_ahead,solar_hours_offset,forecast_year,solar_res_fp,resource_years)


year_hours = pd.date_range(str(forecast_year)+'-01-01', periods=8760, freq='H')
    
# Plot wind forecast to check
plt.subplot(2,1,1)
actual_speed = wind_dict['speed_m_s'][0]
forecast_speed = wind_dict['speed_m_s'][1]
forecast_gusts = wind_dict['gusts_m_s'][1]
plt.plot(year_hours,actual_speed,label='Actual Wind Speed, Met Tower')
plt.plot(year_hours,forecast_speed,label='Forecast Sustained Winds')
plt.plot(year_hours,forecast_gusts,label='Forecast Gusts')
plt.xlim(pd.DatetimeIndex((sim_start,sim_end)))
plt.legend()

# Plot solar forecast to check
plot_n = 3
for key, forecast in solar_dict.items():
    plot_n += 1
    plt.subplot(2,3,plot_n)
    actual_speed = forecast[0]
    forecast_speed = forecast[1]
    forecast_gusts = forecast[24]
    plt.plot(year_hours,actual_speed,label='Actual Irradiance, Met Tower')
    plt.plot(year_hours,forecast_speed,label='Forecast Irradiance, 1 hour ahead')
    plt.plot(year_hours,forecast_gusts,label='Forecast Irradiance, 24 hours ahead')
    plt.legend()
    plt.title(key)
    plt.xlim(pd.DatetimeIndex((sim_start,sim_end)))
plt.show()

# Set up and tune hybrid
period_file = "GE_FirstSolar_Periods_Recleaning_Weeklong.csv"
hybrid_plant, overshoots = tune_iess(period_file)

# Add battery to plant
solar_size_mw = 0.43
array_type = 0 # Fixed-angle
dc_degradation = 0

wind_size_mw = 1.5
hub_height = 80
rotor_diameter = 77
wind_dir = current_dir / ".." / "wind" / "iessGE15"
wind_power_curve = wind_dir/ "NREL_Reference_1.5MW_Turbine_Site_Level_Refactored_Hourly.csv"
wind_shear_exp = 0.15
wind_wake_model = 3 # constant wake loss, layout-independent

batt_time_h = 1
batt_cap_mw = 1

interconnection_size_mw = 2

technologies = {'pv': {
                    'system_capacity_kw': solar_size_mw * 1000
                },
                'wind': {
                    'num_turbines': 1,
                    'turbine_rating_kw': wind_size_mw * 1000,
                    'hub_height': hub_height,
                    'rotor_diameter': rotor_diameter
                },
                'battery': {
                    'system_capacity_kwh': batt_cap_mw * batt_time_h * 1000,
                    'system_capacity_kw': batt_cap_mw * 1000
                }}

# Get resource
flatirons_site['lat'] = 39.91
flatirons_site['lon'] = -105.22
flatirons_site['elev'] = 1835
flatirons_site['year'] = 2020
base_dir = current_dir/ ".." / ".." / ".." / ".."
resource_dir = base_dir/ "resource_files"
prices_file = resource_dir/ "grid" / "pricing-data-2015-IronMtn-002_factors.csv"
solar_file = resource_dir/ "solar" / 'solar_m2_2022.csv'
wind_file = resource_dir/ "wind" / 'wind_m5_2022.srw'
site = SiteInfo(flatirons_site, grid_resource_file=prices_file, solar_resource_file=solar_file, wind_resource_file=wind_file)

hybrid_plant = HybridSimulation(technologies, site, interconnect_kw=interconnection_size_mw * 1000)

# prices_file are unitless dispatch factors, so add $/kwh here
hybrid_plant.ppa_price = 0.04

# Get whole year simulated with resource files, as starting point
hybrid_plant.simulate(project_life=1)

print("output after losses over gross output",
      hybrid_plant.wind.value("annual_energy") / hybrid_plant.wind.value("annual_gross_energy"))

# Save the outputs
annual_energies = hybrid_plant.annual_energies
npvs = hybrid_plant.net_present_values
revs = hybrid_plant.total_revenues
print(annual_energies)
print(npvs)
print(revs)


file = 'figures/'
tag = 'simple2_'
'''
for d in range(0, 360, 5):
    plot_battery_output(hybrid_plant, start_day=d, plot_filename=file+tag+'day'+str(d)+'_battery_gen.png')
    plot_generation_profile(hybrid_plant, start_day=d, plot_filename=file+tag+'day'+str(d)+'_system_gen.png')
'''
plot_battery_dispatch_error(hybrid_plant)
plot_battery_output(hybrid_plant)
plot_generation_profile(hybrid_plant)
#plot_battery_dispatch_error(hybrid_plant, plot_filename=tag+'battery_dispatch_error.png')

#TODO: Save yearlong battery output from resource files, read it back in instead of re-doing

#TODO: Set load profile (flat 300 kW for now)

sim_times = pd.date_range(sim_start, sim_end, freq='H')
for time in sim_times:

    not_yet = 'implemented'

    #TODO: Feed forecast as resource files, as it would appear in each hour

    #TODO: Run battery dispatch optimization, only for coming week (treat past dispatch as fixed)

#TODO: Compare forecasted plant output with acutual output after each data