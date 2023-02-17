from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import circmean
import copy
import csv

import matplotlib.pyplot as plt

def process_solar_forecast(forecast_files:dict, forecast_hours_ahead:dict, forecast_hours_offset:dict,
                            forecast_year:int, resource_fp, resource_years:list, plot_corr=False):

    # Set up forecasts as dict of dataframes
    max_hours_ahead = 0
    for value in forecast_hours_ahead.values():
        max_hours_ahead = max(max(value),max_hours_ahead)

    year_hours = pd.date_range(str(forecast_year)+'-01-01', periods=8760, freq='H')
    # TODO: Correct for leap year (know we're not using a leap year rn)
    forecast_dict = {}
    for key in forecast_files.keys():
        forecast_dict[key] = pd.DataFrame([], index=year_hours, columns=np.arange(0,max_hours_ahead+2))
    
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
        res_file_df = pd.read_csv(year_fp, skiprows=[0,1], header=0, usecols=[5,6,7])
        res_file_df.index = year_hours
        res_file_df.columns = ['GHI_W_m2','DHI_W_m2','DNI_W_m2']
        for column in res_file_df.columns:
            resource_dict[column].loc[:,year] = res_file_df[column].values

    # Average years
    for key, resource in resource_dict.items():
        resource['Avg.'] = np.mean(resource.loc[:,resource_years],axis=1)

    # Use average resource data as long-term forecast (>1 week out)
    for key, forecast in forecast_dict.items():
        forecast.iloc[:,-1] = resource_dict[key]['Avg.']

    # Fill in nans and negatives
    for key, forecast in forecast_dict.items():
        col = forecast.shape[1]-1
        while col > 1:
            not_nan = np.where(np.invert(np.isnan(forecast.iloc[:,col].astype(float))))[0]
            prev_not_nan = copy.deepcopy(not_nan)
            col -= 1
            fill_values = np.full(forecast.shape[0],np.nan)
            fill_values[prev_not_nan] = forecast.iloc[prev_not_nan,col+1]
            not_nan = np.where(np.invert(np.isnan(forecast.iloc[:,col].astype(float))))[0]
            fill_values[not_nan] = forecast.iloc[not_nan,col]
            forecast.iloc[:,col] = fill_values
            negatives = np.where(forecast.iloc[:,col].values<0)[0]
            forecast.iloc[negatives,col] = 0

    if plot_corr:
        # Plot correlation between forecast and actual wind speed
        plot_n = 0
        for key, forecast in forecast_dict.items():
            plot_n += 1
            plt.subplot(1,3,plot_n)
            resource_W_m2 = resource_dict[key][forecast_year].values
            forecast_W_m2 = forecast[1].values
            forecast_idxs = np.where(forecast_W_m2>0)[0]
            plt.plot(resource_W_m2[forecast_idxs], forecast_W_m2[forecast_idxs],'.')
            plt.xlabel('Actual irradiance [W/m^2]')
            plt.ylabel('Hour-ahead forecast irradiance [W/m^2]')
        plt.show()

    # Use forecast year resource data as real-time data (0 hours advance)
    for key, forecast in forecast_dict.items():
        forecast.loc[:,0] = resource_dict[key][forecast_year]

    return forecast_dict


def save_solar_forecast_SAM(forecast_dict, time, orig_fp):

    with open(orig_fp) as fin:
        reader = csv.reader(fin)
        n_header_lines = 3
        lines = []
        for line in range(8760+n_header_lines):
            lines.append(reader.__next__())

    new_fp = Path(str(orig_fp)[:-4]+
                  '_{:02d}'.format(time.month)+
                  '_{:02d}'.format(time.day)+
                  '_{:02d}'.format(time.hour)+
                  '.csv')

    cols = ['GHI_W_m2','DHI_W_m2','DNI_W_m2']
    n_index_cols = 5
    current_idx = forecast_dict[cols[0]].index.to_list().index(time)
    
    for key, forecast in forecast_dict.items():
        if key in cols:
            col_idx = cols.index(key)
            for forecast_row in range(current_idx,8760):
                forecast_col = min(forecast.shape[1]-1,forecast_row-current_idx)
                forecast_val = forecast.iloc[forecast_row,forecast_col]
                if key == 'pres_mbar':
                    forecast_val /= 1000
                lines[forecast_row+n_header_lines][col_idx+n_index_cols] = forecast_val

    with open(new_fp, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerows(lines)

    return new_fp