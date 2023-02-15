from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import circmean

import matplotlib.pyplot as plt

def process_wind_forecast(forecast_files:dict, forecast_hours_ahead:dict, forecast_hours_offset:dict,
                            forecast_year:int, resource_fp, resource_years:list):

    # Set up forecasts as dict of dataframes
    max_hours_ahead = 0
    for value in forecast_hours_ahead.values():
        max_hours_ahead = max(max(value),max_hours_ahead)

    year_hours = pd.date_range(str(forecast_year)+'-01-01', periods=8760, freq='H')
    # TODO: Correct for leap year (know we're not using a leap year rn)
    forecast_dict = {}
    for key in forecast_files.keys():
        forecast_dict[key] = pd.DataFrame([], index=year_hours, columns=np.arange(0,max_hours_ahead+1))
    
    # Read in forecasts
    for key, forecast_fp in forecast_files.items():
        forecast_df = forecast_dict[key]
        file_df = pd.read_csv(forecast_fp, parse_dates=True, index_col=0)
        file_df.columns = forecast_hours_ahead[key]
        forecast_times = file_df.index
        # TODO: Correct for DST
        # For now: only have forecasts during DST, so just subtract 1 hour
        forecast_times = [i - pd.Timedelta(1,unit='H') for i in forecast_times]
        for i, hours_ahead in enumerate(forecast_hours_ahead[key]):
            hours_offset = pd.Timedelta(forecast_hours_offset[key][i],unit='H')
            offset_times = [i + hours_offset for i in forecast_times]
            forecast_df.loc[offset_times,hours_ahead] = file_df.loc[:,hours_ahead].values

    # Read in resource data
    resource_dict = {}
    for key in forecast_files.keys():
        resource_dict[key] = pd.DataFrame([], index=year_hours, columns=resource_years+['Avg.'])
    year_idx = str(resource_fp).find('YYYY')
    for year in resource_years:
        year_fp = Path(str(resource_fp)[:year_idx] + 
                        str(year) + 
                        str(resource_fp)[year_idx+4:])
        res_file_df = pd.read_csv(year_fp, skiprows=[0,1,3,4], header=0)
        res_file_df.index = year_hours
        res_file_df.columns = ['temp_C','pres_mbar','dir_deg','speed_m_s']
        for column in res_file_df.columns:
            resource_dict[column].loc[:,year] = res_file_df[column].values*(1+999*(column=='pres_mbar'))
        resource_dict['gusts_m_s'].loc[:,year] = res_file_df['speed_m_s'].values

    # Average years
    for key, resource in resource_dict.items():
        if key == 'dir_deg':
            resource['Avg.'] = circmean(resource.loc[:,resource_years],360,axis=1)
        else:
            resource['Avg.'] = np.mean(resource.loc[:,resource_years],axis=1)

    # Use average resource data as long-term forecast (>1 week out)
    for key, forecast in forecast_dict.items():
        forecast.iloc[:,-1] = resource_dict[key]['Avg.']

    # Plot correlation between sustained wind, gusts, and actual wind speed
    resource_speed = resource_dict['speed_m_s'][forecast_year].values
    forecast_speed = forecast_dict['speed_m_s'][1].values
    forecast_gusts = forecast_dict['gusts_m_s'][1].values
    forecast_idxs = np.where(forecast_speed>0)[0]
    no_gusts = np.where((forecast_speed>0) & (forecast_gusts<forecast_speed))[0]
    forecast_gusts[no_gusts] = forecast_speed[no_gusts]
    plt.errorbar(resource_speed[forecast_idxs],
                    forecast_speed[forecast_idxs],
                    fmt='.',
                    yerr=(np.vstack((np.zeros(len(forecast_idxs)),
                            np.subtract(forecast_gusts[forecast_idxs],
                                        forecast_speed[forecast_idxs])))))
    plt.xlabel('Actual wind speed [m/s]')
    plt.ylabel('Hour-ahead forecast wind speed [m/s] - sustained + gust')
    plt.ylim([0,25])
    plt.show()

    # Plot to check
    plot_n = 0
    plt.clf()
    for key, forecast in forecast_dict.items():
        plot_n += 1
        plt.subplot(5,1,plot_n)
        actual = resource_dict[key][forecast_year]
        plt.plot(year_hours,actual,label='Actual Data')
        for hours_ahead in [1,2,4,8,24,72,-1]:
            plt.plot(year_hours,forecast.iloc[:,hours_ahead],label='Forecast {:} hours ahead'.format(hours_ahead))
    plt.legend()
    hourly = np.mean(np.reshape(forecast_dict['speed_m_s'].iloc[:,hours_ahead].values,(24,365)),axis=1)
    plt.plot(year_hours[0:24],hourly)
    plt.show()

    return forecast_df