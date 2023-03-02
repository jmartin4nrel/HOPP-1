import os
import numpy as np
import pandas as pd
from scipy.stats import circmean

def parse_status_pv_manual(sts_subpath, manual_fn, years, overwrite=False):

    #Process manually-selected periods where pv array data is 'good'
    
    files = os.listdir(sts_subpath)
    new_fn = manual_fn[:-4]+'_{}_to_{}'.format(years[0],years[-1])
    if (new_fn not in files) or overwrite:

        manual = {}
        period_df = pd.read_csv(sts_subpath/('PV_Periods_'+manual_fn))

        starts = pd.DatetimeIndex(period_df.loc[:,'PV Starts'])
        stops = pd.DatetimeIndex(period_df.loc[:,'PV Stops'])
        
        for year in years:

            year_hours = pd.date_range(str(year)+'-01-01', end=str(year)+'-12-31 23:00', freq='H')
            if len(year_hours) != 8760:
                year_hours = year_hours[:1416].union(year_hours[1440:])
                
            year_status = np.full(len(year_hours),False)
            
            year_bools = [i == year for i in starts.year]
            year_starts = starts[year_bools]
            year_stops = stops[year_bools]
            
            for i, start in enumerate(year_starts):
                stop = year_stops[i]
                start_idx = list(year_hours).index(start)
                stop_idx = list(year_hours).index(stop)
                year_status[start_idx:(stop_idx+1)] = True

            manual[year] = pd.Series(year_status, year_hours, name='pv_manual')

        with open(sts_subpath/new_fn, 'wb') as fp:
            pd.to_pickle(manual, fp)

    return (sts_subpath/new_fn)