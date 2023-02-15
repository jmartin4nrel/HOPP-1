'''
weeklong_sim.py

Emulates sending a power demand signal from HOPP to IESS controller over a week
'''

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hybrid.validation.wind.iessGE15.ge15_wind_forecast_parse import process_wind_forecast
from hybrid.validation.solar.iessFirstSolar.firstSolar_forecast_parse import process_solar_forecast

# Pick week to simulate
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