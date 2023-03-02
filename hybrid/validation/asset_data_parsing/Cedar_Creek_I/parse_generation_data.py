import os
import numpy as np
import pandas as pd
from pytz import timezone
mdt = timezone('US/Mountain')

def parse_wind(gen_path, years, overwrite=False):

    filepaths = {}
    col_label = 'P [kW]'

    # Search through folder for unaggregated monthly minute-by-minute, turbine-by-turbine P
    raw_files = os.listdir(gen_path)
    # Check for already-processed yearly hour-by-hour aggregated plant P
    out_files = os.listdir(gen_path)
    for year in years:
        fn_sum = 'CC1_{}.csv'.format(year)
        fp_sum = gen_path/fn_sum
        if (fn_sum not in out_files) or overwrite:
            
            # Set up data frames to send hourly aggregated power to
            months = pd.date_range(str(year)+'-01-01', end=str(year)+'-12-01', freq='MS')
            hours = pd.date_range(str(year)+'-01-01 01:00', end=str(year+1), freq='H')
            # Get rid of leap day
            if len(hours) > 8760:
                hours = hours[:1416].union(hours[1440:])
            year_sum_df = pd.DataFrame(np.zeros(len(hours)),index=hours,columns=[col_label])
            year_std_df = pd.DataFrame(np.zeros(len(hours)),index=hours,columns=[col_label])
            
            # Open the files that are found and make second-by-second DataFrame
            months_str = months.strftime('%Y%b')
            for i, month_str in enumerate(months_str):
                month_fn_sfx = '_turbine_generation_1m_'+month_str
                month_fn = 'not found'
                for file in raw_files:
                    if month_fn_sfx in file:
                        month_fn = file
                if month_fn in raw_files:
                    print('Processing {} generation data...'.format(month_str))
                    month_df = pd.read_csv(gen_path/month_fn, skiprows=2, header=None, index_col=0)
                    month_end_str = str(months[i]+pd.tseries.offsets.MonthEnd(0))
                    mins = pd.date_range(month_str, end=month_end_str+' 23:59', freq='m')
                    
                    # Check for daylight savings time and adjust seconds index
                    month_start = pd.Timestamp(mins[0], tz=mdt)
                    month_end = pd.Timestamp(mins[-1], tz=mdt)
                    if bool(month_start.dst()):
                        if not bool(month_end.dst()):
                            month_end_str = str(months[i]+pd.tseries.offsets.MonthEnd(0)+pd.tseries.offsets.Hour(1))
                            mins = pd.date_range(month_str, end=month_end_str+' 23:59', freq='m')
                        mins = mins.shift(-60)
                    elif bool(month_end.dst()):
                        month_end_str = str(months[i]+pd.tseries.offsets.MonthEnd(0)-pd.tseries.offsets.Hour(1))
                        mins = pd.date_range(month_str, end=month_end_str+' 23:59', freq='m')
                    mins = mins.shift(1) # Move time index 1 min forward for convenient resampling
                    month_df.index = mins
                    #print('Started {}, ended {}'.format(secs[0],secs[-1]))
                    
                    # Resample day to hourly, sum inverters, and pass to yearly df
                    month_df = month_df.resample('1H', closed='right', label='right').mean()
                    month_df.columns = [col_label]*month_df.shape[1]
                    month_sum_df = month_df.groupby(month_df.columns, axis=1).sum()
                    month_std_df = month_df.groupby(month_df.columns, axis=1).std()
                    year_sum_df.loc[month_sum_df.index] = month_sum_df
                    year_std_df.loc[month_std_df.index] = month_std_df

            # Save yearly outputs
            year_sum_df.to_csv(fp_sum)
            fp_std = gen_path/'CC1_std_{}.csv'.format(year)
            year_std_df.to_csv(fp_std)
        filepaths[year] = fp_sum

    return filepaths