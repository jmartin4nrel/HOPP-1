'''
weeklong_sim.py

Emulates sending a power demand signal from HOPP to IESS controller over a week
'''

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import json
import time as stopwatch

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
run_id = 'run001'

target_firm_power_mw = 0.3

# Pick time period to simulate
sim_start = '07/28/22'
# sim_end = '08/14/22'
sim_end = '08/04/22'

# Get offset to current time for controller, put delay for reading battery
cur_time = pd.Timestamp.now()
cur_time = cur_time - pd.Timedelta(cur_time.microsecond*1000)
delay = pd.Timedelta(10, 'min')
offset = pd.Timedelta(cur_time+delay-pd.Timestamp(sim_start))

# Make figures fullscreen
dpi = plt.rcParams['figure.dpi']
width = 1920
height = 1080
plt.rcParams['figure.figsize'] = [width/dpi,height/dpi]

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
results_dir = validation_dir/'results'/'weeklong_sim'
forecast_year = 2022
resource_years = [2019,2020,2021,2022]
wind_forecast_fn = run_id + '_wind_forecast'
if wind_forecast_fn not in os.listdir(results_dir):
    wind_dict = process_wind_forecast(wind_fp,wind_hours_ahead,wind_hours_offset,forecast_year,wind_res_fp,resource_years)
    with open(results_dir/wind_forecast_fn, 'w') as fp:
        json.dump(list(wind_dict.keys()), fp)
    for key, df in wind_dict.items():
        df.to_csv(results_dir/(wind_forecast_fn+'_'+key))
else:
    wind_dict = {}
    with open(results_dir/wind_forecast_fn, 'r') as fp:
        wind_keys = json.load(fp)
        for key in wind_keys:
            forecast_item = pd.read_csv(results_dir/(wind_forecast_fn+'_'+key), index_col=0)
            forecast_item.index = pd.DatetimeIndex(forecast_item.index)
            forecast_item.columns = [int(i) for i in forecast_item.columns.values]
            wind_dict[key] = forecast_item
solar_forecast_fn = run_id + '_solar_forecast'
if solar_forecast_fn not in os.listdir(results_dir):
    solar_dict = process_solar_forecast(solar_fp,solar_hours_ahead,solar_hours_offset,forecast_year,solar_res_fp,resource_years)
    with open(results_dir/solar_forecast_fn, 'w') as fp:
        json.dump(list(solar_dict.keys()), fp)
    for key, df in solar_dict.items():
        df.to_csv(results_dir/(solar_forecast_fn+'_'+key))
else:
    solar_dict = {}
    with open(results_dir/solar_forecast_fn, 'r') as fp:
        solar_keys = json.load(fp)
        for key in solar_keys:
            forecast_item = pd.read_csv(results_dir/(solar_forecast_fn+'_'+key), index_col=0)
            forecast_item.index = pd.DatetimeIndex(forecast_item.index)
            forecast_item.columns = [int(i) for i in forecast_item.columns.values]
            solar_dict[key] = forecast_item

year_hours = pd.date_range(str(forecast_year)+'-01-01', periods=8760, freq='H')
sim_start_idx = list(year_hours).index(pd.Timestamp(sim_start))
    
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
orig_df['BESS P [kW]'] = np.repeat(0.0,8760)
start_soc = 50.0
dumb_batt_start_soc = [start_soc]
orig_df['BESS SOC [%]'] = np.repeat(start_soc,8760)

# Sum generation
total_gen = []
pv_gen = orig_df['PV [kW]'].values
pv_gen[np.where(np.isnan(pv_gen))[0]] = 0.0
pv_gen[np.where(pv_gen<0)[0]] = 0.0
wind_gen = orig_df['Wind [kW]'].values
wind_gen[np.where(np.isnan(wind_gen))[0]] = 0.0
wind_gen[np.where(wind_gen<0)[0]] = 0.0
batt_gen = orig_df['BESS P [kW]'].values
batt_soc = orig_df['BESS SOC [%]'].values
for i in range(orig_df.shape[0]):
    total_gen.append(pv_gen[i]+wind_gen[i]+batt_gen[i])

# Build load schedule to guide battery dispatch optimizer
start_idx = list(year_hours).index(pd.Timestamp(sim_start))
end_idx = list(year_hours).index(pd.Timestamp(sim_end))
load_schedule_mw = np.full(year_hours.shape, target_firm_power_mw)
load_schedule_mw[:start_idx] = [i/1000 for i in total_gen[:start_idx]]
hybrid_plant.site.desired_schedule = load_schedule_mw

# # Perform optimized "crystal ball" dispatch using wind data
# if orig_fn not in os.listdir(results_dir):
#     hybrid_plant.simulate(project_life=1)
#     orig_df['BESS P [kW]'] = hybrid_plant.battery.Outputs.P
#     orig_df['BESS SOC [%]'] = hybrid_plant.battery.Outputs.SOC
#     orig_df.to_json(results_dir/orig_fn)
# else:
#     orig_df = pd.read_json(results_dir/orig_fn, convert_axes=True)
# batt_gen[start_idx:] = orig_df['BESS P [kW]'].values[start_idx:]
# batt_soc[start_idx:] = orig_df['BESS SOC [%]'].values[start_idx:]
# for i in range(orig_df.shape[0]):
#     total_gen[i] = pv_gen[i]+wind_gen[i]+batt_gen[i]

total_gen_0 = np.copy(total_gen)
week_time = int(7 * 24)
# Loop through sim times
prev_batt_power = []
prev_batt_soc = []
total_gen_dumb = [total_gen[start_idx-1]]
min_soc = hybrid_plant.battery._system_model.ParamsCell.minimum_SOC
max_soc = hybrid_plant.battery._system_model.ParamsCell.maximum_SOC

for i, time in enumerate(sim_times):
    # print('i', i, 'time', time)
    
    tic = stopwatch.time()

    if i == 14:
        print('stop')
    
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

    # Tell batt dispatch optimizer not to optimize whole year, just coming week
    if time_fn not in os.listdir(results_dir):
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

        # If command for this period is within 20% of firm power requirement,
        # just change it to firm power requirement
        if abs(time_df.loc[time,'Plant [kW]']-target_firm_power_mw*1000)/target_firm_power_mw/1000 < 0.2:
            prev_time = sim_times[i-1]
            prev_soc = time_df.loc[prev_time,'BESS SOC [%]']
            new_bess_p = target_firm_power_mw*1000 - time_df.loc[time,'PV [kW]'] - time_df.loc[time,'Wind [kW]']
            new_soc = prev_soc-new_bess_p/(batt_cap_mw*batt_time_h*1000)*100
            if (new_soc >= min_soc) and (new_soc <= max_soc):
                time_df.loc[time,'Plant [kW]'] = target_firm_power_mw*1000
                time_df.loc[time,'BESS SOC [%]'] = new_soc
                time_df.loc[time,'BESS P [kW]'] = new_bess_p
                    
        time_df.to_json(results_dir/time_fn)
    else:
        time_df = pd.read_json(results_dir/time_fn, convert_axes=True)

    # print('Battery Outputs', time_df['BESS P [kW]'][loop_indx-10:loop_indx+10], time_df['BESS SOC [%]'][loop_indx-10:loop_indx+10])
    # print('Ohter Outputs', time_df['PV [kW]'][loop_indx-10:loop_indx+10], time_df['Wind [kW]'][loop_indx-10:loop_indx+10], time_df['Plant [kW]'][loop_indx-10:loop_indx+10])
    
    # Update battery and total generation
    prev_batt_power.append(np.copy(batt_gen[start_idx:]))
    prev_batt_soc.append(np.copy(batt_soc[start_idx:]))
    total_gen[j:] -= batt_gen[j:]
    batt_gen[j:] -= batt_gen[j:]
    batt_soc[j:] -= batt_soc[j:]
    batt_gen[j:] += time_df['BESS P [kW]'].iloc[j:]
    batt_soc[j:] += time_df['BESS SOC [%]'].iloc[j:]
    total_gen[j:] += time_df['BESS P [kW]'].iloc[j:]

    # Simulate what the controller will actually do to meet setpoint + "dumb" controller @ constant setpoint
    act_batt_gen = []
    act_batt_soc = [batt_soc[j-1]]
    dumb_batt_gen = []
    dumb_batt_soc = copy.deepcopy(dumb_batt_start_soc)
    total_gen_dumb_future = []
    for k in np.arange(j,len(batt_gen)):
        attempted_batt_gen = time_df['Plant [kW]'].iloc[k]-wind_gen[k]-pv_gen[k]
        att_dumb_batt_gen = target_firm_power_mw*1000-wind_gen[k]-pv_gen[k]
        # Reduce to battery capacity if above capacity
        if abs(attempted_batt_gen) > batt_cap_mw*1000:
            attempted_batt_gen -= (abs(attempted_batt_gen)-batt_cap_mw*1000)*np.sign(attempted_batt_gen)
        attempted_end_soc = act_batt_soc[-1]-attempted_batt_gen/1000/(batt_cap_mw*batt_time_h)*100
        dumb_end_soc = dumb_batt_soc[-1]-att_dumb_batt_gen/1000/(batt_cap_mw*batt_time_h)*100
        # Reduce to fully discharge if trying to more than fully discharge
        if attempted_end_soc < min_soc:
            attempted_batt_gen = (act_batt_soc[-1]-min_soc)/100*batt_cap_mw*batt_time_h*1000
        if dumb_end_soc < min_soc:
            att_dumb_batt_gen = (dumb_batt_soc[-1]-min_soc)/100*batt_cap_mw*batt_time_h*1000
        # Reduce to fully charge if trying to more than fully charge
        if attempted_end_soc > max_soc:
            attempted_batt_gen = (act_batt_soc[-1]-max_soc)/100*batt_cap_mw*batt_time_h*1000
        if dumb_end_soc > max_soc:
            att_dumb_batt_gen = (dumb_batt_soc[-1]-max_soc)/100*batt_cap_mw*batt_time_h*1000
        act_batt_gen.append(attempted_batt_gen)
        dumb_batt_gen.append(att_dumb_batt_gen)
        act_soc_change = -attempted_batt_gen/(batt_cap_mw*batt_time_h*1000)*100
        dumb_soc_change = -att_dumb_batt_gen/(batt_cap_mw*batt_time_h*1000)*100
        act_batt_soc.append(act_batt_soc[-1]+act_soc_change)
        dumb_batt_soc.append(dumb_batt_soc[-1]+dumb_soc_change)
        total_gen_0[k] = wind_gen[k] + pv_gen[k] + attempted_batt_gen
        total_gen_dumb_future.append(wind_gen[k] + pv_gen[k] + att_dumb_batt_gen)
        if k == j:
            dumb_batt_start_soc.append(dumb_batt_start_soc[-1]+dumb_soc_change)
            total_gen_dumb.append(wind_gen[k] + pv_gen[k] + att_dumb_batt_gen)
            # Update battery and total generation
            total_gen[j] -= batt_gen[j]
            batt_gen[j] -= batt_gen[j]
            batt_soc[j] -= batt_soc[j]
            batt_gen[j] += attempted_batt_gen
            batt_soc[j] += act_batt_soc[-1]
            total_gen[j] += attempted_batt_gen

    # print(batt_gen[j:], batt_soc[j:], time_df['BESS P [kW]'].iloc[j:], time_df['BESS SOC [%]'].iloc[j:])
    # for ijk in range(len(prev_batt_power)):
    #         print('Prev batt power', ijk, prev_batt_power[ijk][0:24])
    
    # Compare forecasted plant output with acutual output after each data point

    # xmin = time-pd.Timedelta(3,unit='D')
    # xmax = time+pd.Timedelta(7,unit='D')
    print(time, sim_start)
    xmin = sim_times[0] - pd.Timedelta(1,unit='D')
    xmax = sim_times[-1] + pd.Timedelta(1,unit='D')     
    plt.clf()

    plt.subplot(5,1,1)
    plt.plot(year_hours[:j],time_df['PV [kW]'].iloc[:j],
            color=[0,0,0],label='Past generation')
    plt.plot(year_hours[j:(j+168)],time_df['PV [kW]'].iloc[j:(j+168)],
            color=[0,0,1],label='Upcoming generation, forecast')  
    plt.plot(year_hours[j:(j+168)],pv_gen[j:(j+168)],
            color=[.5,.5,1],alpha=.5,label='Upcoming generation, actual')
    plt.plot([year_hours[j]-pd.Timedelta(30,'min'),year_hours[j]-pd.Timedelta(30,'min')],
             [min(pv_gen),max(pv_gen)],'k:',label=None)
    plt.ylabel('PV [kW]')
    plt.xlim([xmin,xmax])
    plt.legend()

    plt.subplot(5,1,2)
    plt.plot(year_hours[:j],time_df['Wind [kW]'].iloc[:j],
            color=[0,0,0],label='Past generation')
    plt.plot(year_hours[j:(j+168)],time_df['Wind [kW]'].iloc[j:(j+168)],
            color=[0,0,1],label='Upcoming generation, forecast')  
    plt.plot(year_hours[j:(j+168)],wind_gen[j:(j+168)],
            color=[.5,.5,1],alpha=.5,label='Upcoming generation, actual')
    plt.plot([year_hours[j]-pd.Timedelta(30,'min'),year_hours[j]-pd.Timedelta(30,'min')],
             [min(wind_gen),max(wind_gen)],'k:',label=None)
    plt.ylabel('Wind [kW]')
    plt.xlim([xmin,xmax])
    plt.legend()

    plt.subplot(5,1,3)
    plt.plot(year_hours[:j],batt_gen[:j],
            color=[0,0,0],label='Past generation/charge')
    # for ijk in range(len(prev_batt_power)):
    #         plt.plot(year_hours[i:], prev_batt_power[ijk], '--', alpha=.5)
    plt.plot(year_hours[j:(j+168)],time_df['BESS P [kW]'].iloc[j:(j+168)],
            color=[0,.5,0],label='BESS strategy, using forecasts')  
    # plt.plot(year_hours[i:],batt_gen[i:],
    #         color=[.5,1,.5],alpha=.5,label='BESS strategy, "crystal ball"') 
    # plt.plot(year_hours[i:],orig_df['BESS P [kW]'] [i:],
    #         color=[.5,1,.5],alpha=.5,label='BESS strategy, "crystal ball"')
    plt.plot(year_hours[j:(j+168)],act_batt_gen[:168],
            color=[.5,1,.5],alpha=.5,label='What controller will do to meet Plant kW setpoint')             
    plt.plot([year_hours[j]-pd.Timedelta(30,'min'),year_hours[j]-pd.Timedelta(30,'min')],
             [min(batt_gen),max(batt_gen)],'k:',label=None)
    plt.ylabel('BESS P [kW]')
    plt.xlim([xmin,xmax])
    plt.legend(ncol=3)

    plt.subplot(5,1,4)
    plt.plot(year_hours[:start_idx],batt_soc[:start_idx],
            color=[0,0,0],label='Past SOC')
    plt.plot(year_hours[start_idx-1:j],batt_soc[start_idx-1:j],
            color=[0,.5,0],label=None)
    # for ijk in range(len(prev_batt_soc)):
    #         plt.plot(year_hours[start_idx:], prev_batt_soc[ijk], '--', alpha=.5)   
    plt.plot(year_hours[j:(j+168)],time_df['BESS SOC [%]'].iloc[j:(j+168)],
            color=[0,.5,0],label='BESS strategy, using forecasts')  
    # plt.plot(year_hours[start_idx:],batt_soc[start_idx:],
    #         color=[.5,1,.5],alpha=.5,label='BESS strategy, "crystal ball"') 
    # plt.plot(year_hours[i:],orig_df['BESS SOC [%]'] [i:],
    #         color=[.5,1,.5],alpha=.5,label='BESS strategy, "crystal ball"') 
    plt.plot(year_hours[j:(j+168)],act_batt_soc[1:169],
            color=[.5,1,.5],alpha=.5,label='What controller will do to meet Plant kW setpoint')     
    plt.plot([year_hours[j]-pd.Timedelta(30,'min'),year_hours[j]-pd.Timedelta(30,'min')],
             [min(batt_soc),max(batt_soc)],'k:',label=None)
    plt.plot(year_hours[(start_idx-1):j],dumb_batt_start_soc[:-1],
             color=[1,0,0],label="What a 'non-strategizing' controller will actually to do")
    plt.ylabel('BESS SOC [%]')
    plt.xlim([xmin,xmax])
    plt.ylim([0,110])
    plt.legend(ncol=2)

    plt.subplot(5,1,5)
    plt.plot(year_hours[:start_idx],total_gen[:start_idx],
            color=[0,0,0],label='Past generation')
    plt.plot(year_hours[j:(j+168)],time_df['Plant [kW]'].iloc[j:(j+168)],
            color=[0,0,1],label='Controller setpoint, strategized 1 week out')  
    plt.plot(year_hours[j:(j+168)],total_gen_0[j:(j+168)],
            color=[.5,.5,1],alpha=.5,label='What the controller will actually be able to achieve')  
    plt.plot(year_hours[j:(j+168)],[l*0+target_firm_power_mw*1000 for l in act_batt_gen[:168]],
            ':',color=[1,0,0],alpha=.5,label="What a 'non-strategizing' controller will try to do")  
    plt.plot(year_hours[(start_idx-1):j],total_gen_dumb[0:-1],
            color=[1,0,0],label="What a 'non-strategizing' controller actually does")  
    plt.plot(year_hours[(start_idx-1):j],total_gen[(start_idx-1):j],
            color=[0,.5,0],label="What the 'look-ahead' controller actually does")  
    # plt.plot(year_hours[j:],total_gen[j:],
    #         color=[.5,.5,1],alpha=.5,label='Upcoming generation, actual')  
    plt.plot([year_hours[j]-pd.Timedelta(30,'min'),year_hours[j]-pd.Timedelta(30,'min')],
             [min(total_gen),max(total_gen)],'k:',label=None)
    plt.ylabel('Plant [kW]')
    plt.xlim([xmin,xmax])
    plt.legend(ncol=3)
    plt.savefig(results_dir/(time_fn+'_pretty'))
    # plt.show()
    # plt.pause(1)

    # # Export timeseries with current time for reference
    # sec_df = time_df.loc[time:time+pd.Timedelta('2H')].resample('1S').bfill()
    # sec_df['BESS SOC [%]'] = time_df.loc[time:time+pd.Timedelta('2H'),'BESS P [kW]'].resample('1S').interpolate()
    # sec_df['Plant [kW]'] = np.transpose(np.sum(np.vstack((sec_df['PV [kW]'].values,
    #                                                     sec_df['Wind [kW]'].values,
    #                                                     sec_df['BESS P [kW]'].values)), axis=0))
    # real_time = sec_df.index+offset
    # sec_df.reset_index(inplace=True)
    # sec_df = sec_df.rename(columns = {'index':'Data time'})
    # sec_df.index = real_time
    # sec_fn = '{}_sec_df'.format(run_id)+\
    #         '_{:02d}'.format(time.month)+\
    #         '_{:02d}'.format(time.day)+\
    #         '_{:02d}'.format(time.hour)+'.csv'
    # sec_df.to_csv(results_dir/sec_fn)

    toc = stopwatch.time()

    print('loop time {:.2f} sec'.format(toc-tic))

    if (j-start_idx) > 0:

        hours_firm_power_met_forecast = np.sum(np.greater(time_df['Plant [kW]'].iloc[start_idx:end_idx].values,
                                                        target_firm_power_mw*1000*.95))
        hours_firm_power_met_current_strategy = np.sum(np.greater(total_gen[start_idx:j],
                                                                target_firm_power_mw*1000*.95))
        hours_firm_power_met_dumb = np.sum(np.greater(total_gen_dumb[:-1],
                                                      target_firm_power_mw*1000*.95))

        pct_firm_power_met_forecast = hours_firm_power_met_forecast/(end_idx-start_idx)*100
        pct_firm_power_met_current_strategy = hours_firm_power_met_current_strategy/(j-start_idx)*100
        pct_firm_power_met_dumb = hours_firm_power_met_dumb/(j-start_idx)*100

        print('% time firm power met, forecast: {:.2f}%'.format(pct_firm_power_met_forecast))
        print('% time firm power met, actual strategy: {:.2f}%'.format(pct_firm_power_met_current_strategy))
        print('% time firm power met, non-strategizing: {:.2f}%'.format(pct_firm_power_met_dumb))

plt.show()

    