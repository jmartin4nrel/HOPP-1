'''
weeklong_sim.py

Emulates sending a power demand signal from HOPP to IESS controller over a week
'''

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
sys.path.append('/home/gstarke/Research_Programs/HOPP/HOPP/')

from hybrid.sites import SiteInfo, flatirons_site
from hybrid.validation.wind.iessGE15.ge15_wind_forecast_parse import process_wind_forecast, save_wind_forecast_SAM
from hybrid.validation.solar.iessFirstSolar.firstSolar_forecast_parse import process_solar_forecast, save_solar_forecast_SAM
from hybrid.validation.examples.tune_iess import tune_iess
from hybrid.validation.validate_hybrid import tune_manual
from hybrid.hybrid_simulation import HybridSimulation
from hybrid.dispatch.plot_tools import plot_battery_output, plot_battery_dispatch_error, plot_generation_profile
from hybrid.resource import (
    SolarResource,
    WindResource
    )

# Change run ID to start from scratch (leave the same to load previously generated results)
run_id = 'run010'

# Pick time period to simulate
sim_start = '07/28/22'
# sim_end = '08/14/22'
sim_end = '08/04/22'

# Set path to forecast files
current_dir = Path(__file__).parent.absolute()
validation_dir = current_dir/'..'/'..'
wind_dir = validation_dir/'wind'/'iessGE15'
solar_dir = validation_dir/'solar'/'iessFirstSolar'
wind_files =   {'speed_m_s':'Wind Speed-data-as-seriestocolumns-2023-02-17 09_57_04.csv',
                'gusts_m_s':'Gusts-data-as-seriestocolumns-2023-02-17 09_56_01.csv',
                'dir_deg':  'Wind Direction-data-as-seriestocolumns-2023-02-17 09_51_27.csv',
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
# status_fp = validation_dir/'wind'/'iessGE15'/'GE15_IEC_validity_hourly_2019_2022'
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
forecast_1hour = wind_dict['speed_m_s'][1]
forecast_1day = wind_dict['speed_m_s'][24]
plt.plot(year_hours,actual_speed,label='Actual Wind Speed, M5 Met Tower')
plt.plot(year_hours,forecast_1hour,label='Forecast Wind Speed, 1 hour ahead')
plt.plot(year_hours,forecast_1day,label='Forecast Wind Speed, 24 hours ahead')
plt.title('GE Wind Speed [m/s]')
plt.xlim(pd.DatetimeIndex((sim_start,sim_end)))
plt.legend()

# Plot solar forecast to check
plt.subplot(2,1,2)
forecast = solar_dict['GHI_W_m2']
actual_speed = forecast[0]
forecast_1hour = forecast[1]
forecast_1day = forecast[24]
plt.plot(year_hours,actual_speed,label='Actual GHI, M2 Met Tower')
plt.plot(year_hours,forecast_1hour,label='Forecast GHI, 1 hour ahead')
plt.plot(year_hours,forecast_1day,label='Forecast GHI, 24 hours ahead')
plt.legend()
plt.title('First Solar GHI [W/m^2]')
plt.xlim(pd.DatetimeIndex((sim_start,sim_end)))
# plt.show()

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

#Set load profile (flat 300 kW for now)
load_schedule = [0.3]*8760

site = SiteInfo(flatirons_site, grid_resource_file=prices_file,
                solar_resource_file=solar_file, wind_resource_file=wind_file,
                desired_schedule=load_schedule)

# Save forecasts
sim_times = pd.date_range(sim_start, sim_end, freq='H')
save_solar_forecast_SAM(solar_dict, sim_times[0], solar_file)
save_wind_forecast_SAM(wind_dict, sim_times[0], wind_file)

# Add battery to plant
solar_size_mw = 0.43
array_type = 0 # Fixed-angle
dc_degradation = 0

wind_size_mw = 1.5
hub_height = 80
rotor_diameter = 77
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
                }
        }

# Set up hybrid with dispatch solver options
solver = 'simple'
grid_charge_bool = False
dispatch_opt_dict = {'battery_dispatch':solver,'grid_charging':grid_charge_bool}
hybrid_plant = HybridSimulation(technologies, site, interconnect_kw=interconnection_size_mw * 1000, 
                                dispatch_options=dispatch_opt_dict)
hybrid_plant.grid._system_model.GridLimits.grid_interconnection_limit_kwac = interconnection_size_mw * 1000

# Enter turbine power curve
curve_data = pd.read_csv(wind_power_curve)
wind_speed = curve_data['Wind Speed [m/s]'].values.tolist() 
curve_power = curve_data['Power [kW]']
hybrid_plant.wind._system_model.Turbine.wind_turbine_powercurve_windspeeds = wind_speed
hybrid_plant.wind._system_model.Turbine.wind_turbine_powercurve_powerout = curve_power    

# # Set up and tune hybrid
# period_file = "GE_FirstSolar_Periods_Recleaning_Weeklong.csv"
# hybrid_plant, overshoots = tune_iess(period_file)
# getattr(hybrid_plant,'pv').value('losses',overshoots['pv'])
# getattr(hybrid_plant,'wind').value('turb_specific_loss',overshoots['wind'])

# Set tuning coefficients (flat losses for now)
hybrid_plant = tune_manual(hybrid_plant,base_dir/'hybrid'/'validation'/'results'/'IESS tune 2_16_23.csv')

# prices_file are unitless dispatch factors, so add $/kwh here
hybrid_plant.ppa_price = 0.04

# Get whole year of actual generation data
results_dir = validation_dir/'results'/'weeklong_sim'
wind_gen_fn = 'GE1pt5MW_2022.csv'
solar_gen_fn = 'FirstSolar_2022.csv'
orig_fn = 'orig_outputs_'+run_id
orig_df = pd.DataFrame(index=year_hours,columns=['PV [kW]','Wind [kW]','BESS P [kW]','BESS SOC [%]'])
orig_df['PV [kW]'] = pd.read_csv(solar_dir/solar_gen_fn).loc[:,'P [kW]'].values
orig_df['Wind [kW]'] = pd.read_csv(wind_dir/wind_gen_fn).loc[:,'P [kW]'].values
orig_df['BESS P [kW]'] = np.repeat(0,8760)
orig_df['BESS SOC [%]'] = np.repeat(90,8760)

# Sum generation
total_gen = []
pv_gen = orig_df['PV [kW]'].values
pv_gen[np.where(np.isnan(pv_gen))[0]] = 0
pv_gen[np.where(pv_gen<0)[0]] = 0
wind_gen = orig_df['Wind [kW]'].values
wind_gen[np.where(np.isnan(wind_gen))[0]] = 0
wind_gen[np.where(wind_gen<0)[0]] = 0
batt_gen = orig_df['BESS P [kW]'].values
batt_soc = orig_df['BESS SOC [%]'].values
for i in range(orig_df.shape[0]):
    total_gen.append(pv_gen[i]+wind_gen[i]+batt_gen[i])

# Build load schedule to guide battery dispatch optimizer
start_idx = list(year_hours).index(pd.Timestamp(sim_start))
load_schedule_mw = np.full(year_hours.shape, 0.3)
load_schedule_mw[:start_idx] = [i/1000 for i in total_gen[:start_idx]]
hybrid_plant.site.desired_schedule = load_schedule_mw

# Perform optimized "crystal ball" dispatch using wind data
if orig_fn not in os.listdir(results_dir):
    hybrid_plant.simulate(project_life=1)
    orig_df['BESS P [kW]'] = hybrid_plant.battery.Outputs.P
    orig_df['BESS SOC [%]'] = hybrid_plant.battery.Outputs.SOC
    orig_df.to_json(results_dir/orig_fn)
else:
    orig_df = pd.read_json(results_dir/orig_fn, convert_axes=True)
batt_gen[start_idx:] = orig_df['BESS P [kW]'].values[start_idx:]
batt_soc[start_idx:] = orig_df['BESS SOC [%]'].values[start_idx:]
for i in range(orig_df.shape[0]):
    total_gen[i] = pv_gen[i]+wind_gen[i]+batt_gen[i]

total_gen_0 = np.copy(total_gen)
week_time = int(7 * 24)
# Loop through sim times
prev_batt_power = []
prev_batt_soc = []
for i, time in enumerate(sim_times):
        # print('i', i, 'time', time)
        loop_indx = list(year_hours).index(pd.Timestamp(time))
        dispatch_indexs = {'start_indx': loop_indx,\
                        'end_indx': loop_indx + week_time,\
                        'initial_soc': batt_soc[loop_indx-1]}

        # print('battery soc', batt_soc[loop_indx-10:loop_indx+10])
        # print('bat 2', batt_soc[loop_indx-1:loop_indx+1])
        j = list(year_hours).index(time)
        print('index check', j, time, loop_indx)



        #Change load schedule to actual hybrid plant generation before current time
        hybrid_plant.site.desired_schedule[:i] = [i/10000 for i in total_gen[:i]]

        # Adjust forecast
        new_solar_fp = save_solar_forecast_SAM(solar_dict, time, solar_file)
        new_wind_fp = save_wind_forecast_SAM(wind_dict, time, wind_file)

        # Previously figured out how to change resource file without setting up HybridSimulation all over again:
        NewSolarRes = SolarResource(hybrid_plant.site.lat,hybrid_plant.site.lon,forecast_year,filepath=new_solar_fp)
        NewWindRes = WindResource(hybrid_plant.site.lat,hybrid_plant.site.lon,forecast_year,hub_height,filepath=new_wind_fp)
        # Have to change pressure to sea level!
        for k in range(len(NewWindRes.data['data'])):
                NewWindRes.data['data'][k][1] = 1
        hybrid_plant.pv._system_model.SolarResource.solar_resource_data = NewSolarRes.data
        hybrid_plant.wind._system_model.Resource.wind_resource_data = NewWindRes.data

        # Generate filename for this timestep
        time_fn = run_id+\
                '_{:02d}'.format(time.month)+\
                '_{:02d}'.format(time.day)+\
                '_{:02d}'.format(time.hour)

        # print('time_fn', time_fn, 'list', os.listdir(results_dir))
        #     jkjkjkj

        #TODO: tell batt dispatch optimizer not to optimize whole year, just coming week
        #     if time_fn not in os.listdir(results_dir):
        # hybrid_plant.simulate(project_life=1)
        hybrid_plant.simulate(project_life=1, finite_dispatch=dispatch_indexs)
        time_df = pd.DataFrame(index=year_hours,columns=['PV [kW]',
                                                        'Wind [kW]',
                                                        'BESS P [kW]',
                                                        'BESS SOC [%]',
                                                        'Plant [kW]'])
        time_df['PV [kW]'] = hybrid_plant.pv.generation_profile
        time_df['Wind [kW]'] = hybrid_plant.wind.generation_profile
        time_df['BESS P [kW]'] = hybrid_plant.battery.Outputs.P
        time_df['BESS SOC [%]'] = hybrid_plant.battery.Outputs.SOC
        time_df['Plant [kW]'] = np.sum(np.vstack((hybrid_plant.pv.generation_profile,
                                                hybrid_plant.wind.generation_profile,
                                                hybrid_plant.battery.Outputs.P)),axis=0)
        time_df.to_json(results_dir/time_fn)
        #     else:
        #         time_df = pd.read_json(results_dir/time_fn, convert_axes=True)

        # print('Battery Outputs', time_df['BESS P [kW]'][loop_indx-10:loop_indx+10], time_df['BESS SOC [%]'][loop_indx-10:loop_indx+10])
        # print('Ohter Outputs', time_df['PV [kW]'][loop_indx-10:loop_indx+10], time_df['Wind [kW]'][loop_indx-10:loop_indx+10], time_df['Plant [kW]'][loop_indx-10:loop_indx+10])
        # Update battery and total generation
        prev_batt_power.append(np.copy(batt_gen[start_idx:]))
        prev_batt_soc.append(np.copy(batt_soc[start_idx:]))
        batt_gen[j:] -= batt_gen[j:]
        batt_soc[j:] -= batt_soc[j:]
        batt_gen[j:] += time_df['BESS P [kW]'].iloc[j:]
        batt_soc[j:] += time_df['BESS SOC [%]'].iloc[j:]
        total_gen[j:] -= batt_gen[j:]
        total_gen[j:] += time_df['BESS P [kW]'].iloc[j:]

        # print(batt_gen[j:], batt_soc[j:], time_df['BESS P [kW]'].iloc[j:], time_df['BESS SOC [%]'].iloc[j:])
        # for ijk in range(len(prev_batt_power)):
        #         print('Prev batt power', ijk, prev_batt_power[ijk][0:24])
        
        #TODO: Compare forecasted plant output with acutual output after each data point

        # xmin = time-pd.Timedelta(3,unit='D')
        # xmax = time+pd.Timedelta(7,unit='D')
        print(time, sim_start)
        xmin = sim_times[0] - pd.Timedelta(1,unit='D')
        xmax = sim_times[-1] + pd.Timedelta(1,unit='D')     
        plt.clf()
        plt.subplot(5,1,1)
        plt.plot(year_hours[:start_idx],time_df['PV [kW]'].iloc[:start_idx],
                color=[0,0,0],label='Past generation')
        plt.plot(year_hours[start_idx:],time_df['PV [kW]'].iloc[start_idx:],
                color=[0,0,1],label='Upcoming generation, forecast')  
        plt.plot(year_hours[start_idx:],pv_gen[start_idx:],
                color=[.5,.5,1],alpha=.5,label='Upcoming generation, actual')
        plt.ylabel('PV [kW]')
        plt.xlim([xmin,xmax])
        plt.legend()
        plt.subplot(5,1,2)
        plt.plot(year_hours[:start_idx],time_df['Wind [kW]'].iloc[:start_idx],
                color=[0,0,0],label='Past generation')
        plt.plot(year_hours[start_idx:],time_df['Wind [kW]'].iloc[start_idx:],
                color=[0,0,1],label='Upcoming generation, forecast')  
        plt.plot(year_hours[start_idx:],wind_gen[start_idx:],
                color=[.5,.5,1],alpha=.5,label='Upcoming generation, actual')
        plt.ylabel('Wind [kW]')
        plt.xlim([xmin,xmax])
        plt.legend()
        plt.subplot(5,1,3)
        plt.plot(year_hours[:start_idx],batt_gen[:start_idx],
                color=[0,0,0],label='Past generation/charge')
        # for ijk in range(len(prev_batt_power)):
        #         plt.plot(year_hours[start_idx:], prev_batt_power[ijk], '--', alpha=.5)
        plt.plot(year_hours[start_idx:],time_df['BESS P [kW]'].iloc[start_idx:],
                color=[0,.5,0],label='BESS strategy, using forecasts')  
        # plt.plot(year_hours[start_idx:],batt_gen[start_idx:],
        #         color=[.5,1,.5],alpha=.5,label='BESS strategy, "crystal ball"') 
        plt.plot(year_hours[start_idx:],orig_df['BESS P [kW]'] [start_idx:],
                color=[.5,1,.5],alpha=.5,label='BESS strategy, "crystal ball"')       
     
        plt.ylabel('BESS P [kW]')
        plt.xlim([xmin,xmax])
        plt.legend()
        plt.subplot(5,1,4)
        plt.plot(year_hours[:start_idx],batt_soc[:start_idx],
                color=[0,0,0],label='Past SOC')
        # for ijk in range(len(prev_batt_soc)):
        #         plt.plot(year_hours[start_idx:], prev_batt_soc[ijk], '--', alpha=.5)   
        plt.plot(year_hours[start_idx:],time_df['BESS SOC [%]'].iloc[start_idx:],
                color=[0,.5,0],label='BESS strategy, using forecasts')  
        # plt.plot(year_hours[start_idx:],batt_soc[start_idx:],
        #         color=[.5,1,.5],alpha=.5,label='BESS strategy, "crystal ball"') 
        plt.plot(year_hours[start_idx:],orig_df['BESS SOC [%]'] [start_idx:],
                color=[.5,1,.5],alpha=.5,label='BESS strategy, "crystal ball"') 
              
        plt.ylabel('BESS SOC [%]')
        plt.xlim([xmin,xmax])
        plt.legend()
        plt.subplot(5,1,5)
        plt.plot(year_hours[:j],total_gen[:j],
                color=[0,0,0],label='Past generation')
        plt.plot(year_hours[j:],time_df['Plant [kW]'].iloc[j:],
                color=[0,0,1],label='Upcoming generation, forecast')  
        plt.plot(year_hours[start_idx:],total_gen_0[start_idx:],
                color=[.5,.5,1],alpha=.5,label='Upcoming generation, actual')  
        # plt.plot(year_hours[j:],total_gen[j:],
        #         color=[.5,.5,1],alpha=.5,label='Upcoming generation, actual')  
        plt.ylabel('Plant [kW]')
        plt.xlim([xmin,xmax])
        plt.legend()
        # plt.show()
        plt.savefig(results_dir/time_fn)
        #     plt.pause(1)

plt.show()

    