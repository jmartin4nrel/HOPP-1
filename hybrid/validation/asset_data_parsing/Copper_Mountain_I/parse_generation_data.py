import os
import numpy as np
import pandas as pd
from pytz import timezone
pdt = timezone('US/Pacific')

def parse_pv(gen_path, years, overwrite=False):

    filepaths = {}
    col_label = 'P [kW]'

    # Search through folder for unaggregated daily second-by-second, inverter-by-inverter P
    raw_files = os.listdir(gen_path/'Unaggregated')
    # Check for already-processed yearly hour-by-hour aggregated plant P
    out_files = os.listdir(gen_path)
    for year in years:
        fn_sum = 'CM1_{}.csv'.format(year)
        fp_sum = gen_path/fn_sum
        if (fn_sum not in out_files) or overwrite:
            
            # Set up data frames to send hourly aggregated power to
            days = pd.date_range(str(year)+'-01-01', end=str(year)+'-12-31', freq='D')
            hours = pd.date_range(str(year)+'-01-01 01:00', end=str(year+1), freq='H')
            # Get rid of leap day
            if len(days) > 365:
                days = days[:59].union(days[60:])
                hours = hours[:1416].union(hours[1440:])
            year_sum_df = pd.DataFrame(np.zeros(len(hours)),index=hours,columns=[col_label])
            year_std_df = pd.DataFrame(np.zeros(len(hours)),index=hours,columns=[col_label])
            
            # Open the files that are found and make second-by-second DataFrame
            days_str = days.strftime('%Y-%m-%d')
            for day_str in days_str:
                day_fn = day_str+'_CM_gen.csv'
                if day_fn in raw_files:
                    print('Processing {} generation data...'.format(day_str))
                    day_df = pd.read_csv(gen_path/'Unaggregated'/day_fn, header=None)
                    secs = pd.date_range(day_str, periods=24*60*60, freq='S')
                    
                    # Check for daylight savings time and adjust seconds index
                    day_start = pd.Timestamp(secs[0], tz=pdt)
                    day_end = pd.Timestamp(secs[-1], tz=pdt)
                    if bool(day_start.dst()):
                        if not bool(day_end.dst()):
                            secs = pd.date_range(day_str, periods=25*60*60, freq='S')
                        secs = secs.shift(-3600)
                    elif bool(day_end.dst()):
                        secs = pd.date_range(day_str, periods=23*60*60, freq='S')
                    secs = secs.shift(1) # Move time index 1 sec forward for convenient resampling
                    day_df.index = secs
                    #print('Started {}, ended {}'.format(secs[0],secs[-1]))
                    
                    # Resample day to hourly, sum inverters, and pass to yearly df
                    day_df = day_df.resample('1H', closed='right', label='right').mean()
                    day_df.columns = [col_label]*day_df.shape[1]
                    day_sum_df = day_df.groupby(day_df.columns, axis=1).sum()
                    day_std_df = day_df.groupby(day_df.columns, axis=1).std()
                    year_sum_df.loc[day_sum_df.index] = day_sum_df
                    year_std_df.loc[day_std_df.index] = day_std_df

            # Save yearly outputs
            year_sum_df.to_csv(fp_sum)
            fp_std = gen_path/'CM1_std_{}.csv'.format(year)
            year_std_df.to_csv(fp_std)
        filepaths[year] = fp_sum

    return filepaths