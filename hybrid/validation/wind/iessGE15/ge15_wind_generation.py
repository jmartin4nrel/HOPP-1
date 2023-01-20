import os
import numpy as np
import pandas as pd
from time import time, localtime
from pathlib import Path
from scipy.stats import circmean

def process_status():

    # Set initial directory
    current_dir = Path(__file__).parent.absolute()
    status_dir = current_dir / "status"
    files = os.listdir(status_dir)

    # Create hourly dataframe
    hours = pd.date_range('2019-01-01','2023-01-01',freq='H')
    hours = hours[:10176].union(hours[10200:]) # Take out leap day
    hour_df = pd.DataFrame(np.full((len(hours),1),False),index=hours)

    # Set valid status codes
    valid_codes = [1,2,3,7,16] # Source: Jason Roadman
    max_int = 5*60+10 # Maximum interval between status reports - 5 minutes = 5*60 seconds + 10 sec buffer

    # Load each secondly file
    for file in files:
        sec_df = pd.read_csv(status_dir/file, index_col=0, parse_dates=True)
        # Get indexes of the hours that are covered by this file, adjusting for DST in the timestamps
        first_sec_dst = True if sec_df.index[0].month > 4 else False
        last_sec_dst = True if sec_df.index[-1].month > 4 else False
        first_hour_idx = np.where(sec_df.index[0]==hours)[0] - 1*first_sec_dst + 1
        last_hour_idx = np.where(sec_df.index[-1]==hours)[0] - 1*last_sec_dst
        hour_idx = first_hour_idx
        # Check if this is the leap day file
        start_sec = 24*60*60 if sec_df.index[0] == '2020-03-01 00:00:00' else 0
        # Check for invalid status codes + 5 minute updates in each hour
        while start_sec+60*60 < len(sec_df):
            print("Processing {}...".format(hours[hour_idx]))
            codes = sec_df.iloc[start_sec:start_sec+60*60,0].values
            # Check max consecutive nans - must not go more than 5 min without reporting status
            nans = pd.Series(np.isnan(codes))
            temp = nans.ne(nans.shift()).cumsum()
            nan_counter = nans.groupby(temp).transform('size')*np.where(nans,1,-1)
            if np.max(nan_counter.values) <= max_int:
                # Check that all codes are valid and if so make this hour True in hourly dataframe
                if np.all(np.in1d(codes[np.invert(nans)],valid_codes)):
                    hour_df.iloc[hour_idx,0] = True
            start_sec += 60*60
            hour_idx +=1

        # Make sure the hours match up
        if hour_idx-1 != last_hour_idx:
            ValueError('Incorrect number of hours were processed')

    hour_df.to_json(current_dir / "GE15_IEC_validity_hourly_2019_2022")


def get_yaw_mismatch(yaw_file, tenmin_wind_file, years):

    yaw_mat = np.loadtxt(yaw_file)
    yaw_mat = np.where(yaw_mat>=0, yaw_mat, np.full(np.size(yaw_mat), np.nan))
    tenmin_wind_df = pd.read_json(tenmin_wind_file)
    tenmin_wind_df.index = tenmin_wind_df.index.shift(-7, freq='H')
    direction_label = 'Direction (sonic_74m)'
    yaw_label = 'Turbine Yaw [deg]'
    wind_label = 'Speed (cup_ 80 m)'
    used_points = 0
    year_hour_mismatches = np.zeros((len(years),8760))
    hour_mat = np.transpose(np.array([[],[],[],[]]))
    for i, year in enumerate(years):
        print('Calculating yaw mismatch for {:.0f}...'.format(year))
        year_start = pd.DatetimeIndex([str(year)])[0]
        year_end = pd.DatetimeIndex([str(year+1)])[0]
        times = pd.date_range(start=year_start, end=year_end, freq='10min')[:-1]
        hours = pd.date_range(start=year_start, end=year_end, freq='H')
        year_df = pd.DataFrame(yaw_mat[used_points:(used_points+len(times))], index=times, columns=[yaw_label])
        used_points += len(times)
        year_df = pd.concat((year_df,tenmin_wind_df.loc[(tenmin_wind_df.index>=year_start)&(tenmin_wind_df.index<year_end)]),axis=1)
        if year % 4 == 0:
            leap_start = pd.DatetimeIndex([str(year)+'-02-29'])[0]
            leap_end = pd.DatetimeIndex([str(year)+'-03-01'])[0]
            year_df = pd.concat((year_df.loc[(year_df.index<leap_start)],year_df.loc[(year_df.index>=leap_end)]))
            hours = hours[(hours<leap_start)|(hours>=leap_end)]
        for hour_idx, hour in enumerate(hours[:-1]):
            hour_df = year_df.loc[(year_df.index>hour)&(year_df.index<=hours[hour_idx+1])]
            mismatches = hour_df.loc[:,direction_label]+180-hour_df.loc[:,yaw_label]
            mismatches[mismatches>180] = mismatches[mismatches>180]-360
            year_hour_mismatches[i,hour_idx] = circmean(mismatches, 180, -180)
            mean_hour = np.array([[circmean(hour_df.loc[:,direction_label], 360),\
                                    circmean(hour_df.loc[:,yaw_label], 360),\
                                    circmean(mismatches, 180, -180),\
                                    np.mean(hour_df.loc[:,wind_label])]])
            hour_mat = np.vstack((hour_mat,mean_hour))
    hourly_df = pd.DataFrame(data=hour_mat, columns=['Wind Direction [deg]', yaw_label, 'Yaw Mismatch [deg]', 'Wind Speed [m/s]'])
    hourly_df.to_json(Path(str(yaw_file)[:-4]+' mismatch'))
    hourly_df.to_csv(Path(str(yaw_file)[:-4]+' mismatch.csv'))