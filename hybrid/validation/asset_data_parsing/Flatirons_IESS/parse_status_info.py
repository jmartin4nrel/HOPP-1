import os
import json
import numpy as np
import pandas as pd
from scipy.stats import circmean

def parse_status_dummy(sts_path, good_period_file, years):

    #Dummy function leading to already-generated files

    good_period_fp = sts_path/'hybrid'/good_period_file
    status_fp = sts_path/'wind'/'GE15_IEC_validity_hourly_2019_2022'
    yaw_mismatch_fp = sts_path/'wind'/'GE Turbine Yaw Dec 2019 to 2022 mismatch'

    return good_period_fp, status_fp, yaw_mismatch_fp

def parse_status_pv_manual(sts_subpath, manual_fn, years, overwrite=False):

    #Process manually-selected periods where wind turbine data is 'good'
    
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


def parse_status_wind_manual(sts_subpath, manual_fn, years, overwrite=False):

    #Process manually-selected periods where wind turbine data is 'good'
    
    files = os.listdir(sts_subpath)
    new_fn = manual_fn[:-4]+'_{}_to_{}'.format(years[0],years[-1])
    if (new_fn not in files) or overwrite:

        manual = {}
        period_df = pd.read_csv(sts_subpath/('Wind_Periods_'+manual_fn))

        starts = pd.DatetimeIndex(period_df.loc[:,'Wind Starts'])
        stops = pd.DatetimeIndex(period_df.loc[:,'Wind Stops'])
        
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

            manual[year] = pd.Series(year_status, year_hours, name='wind_manual')

        with open(sts_subpath/new_fn, 'wb') as fp:
            pd.to_pickle(manual, fp)

    return (sts_subpath/new_fn)


def parse_status_wind_status_bit(sts_subpath, years, overwrite=False):

    # Determine whether turbine status bit stayed valid for entire hour

    # Set initial directory
    filename = "GE15_IEC_validity_hourly_"+str(years[0])+"_"+str(years[-1])
    files = os.listdir(sts_subpath)
    
    if (filename not in files) or overwrite:
    
        status_bit = {}

        status_dir = sts_subpath / "raw_files"
        files = os.listdir(status_dir)

        for year in years:
            
            year_hours = pd.date_range(str(year)+'-01-01', end=str(year+1)+'-01-01' , freq='H')
            if len(year_hours) != 8761:
                year_hours = year_hours[:1416].union(year_hours[1440:])

            year_status = np.full(len(year_hours),False)
            
            # Set valid status codes
            valid_codes = [1,2,3,7,16] # Source: Jason Roadman
            max_int = 5*60+10 # Maximum interval between status reports - 5 minutes = 5*60 seconds + 10 sec buffer

            # Load each secondly file
            for file in files:
                if int(file[-14:-10]) == year:
                    sec_df = pd.read_csv(status_dir/file, index_col=0, parse_dates=True)
                    
                    # Get indexes of the hours that are covered by this file, adjusting for DST in the timestamps
                    first_sec_dst = True if sec_df.index[0].month > 4 else False
                    last_sec_dst = True if sec_df.index[-1].month > 4 else False
                    first_hour_idx = np.where(sec_df.index[0]==year_hours)[0] - 1*first_sec_dst + 1
                    last_hour_idx = np.where(sec_df.index[-1]==year_hours)[0] - 1*last_sec_dst
                    hour_idx = first_hour_idx
                    # Check if this is the leap day file
                    start_sec = 24*60*60 if sec_df.index[0] == '2020-03-01 00:00:00' else 0
                    # Check for invalid status codes + 5 minute updates in each hour
                    while start_sec+60*60 < len(sec_df):
                        if (hour_idx < 8760) & (hour_idx%24 == 0):
                            print("Processing {}...".format(year_hours[hour_idx]))
                        codes = sec_df.iloc[start_sec:start_sec+60*60,0].values
                        # Check max consecutive nans - must not go more than 5 min without reporting status
                        nans = pd.Series(np.isnan(codes))
                        temp = nans.ne(nans.shift()).cumsum()
                        nan_counter = nans.groupby(temp).transform('size')*np.where(nans,1,-1)
                        if np.max(nan_counter.values) <= max_int:
                            # Check that all codes are valid and if so make this hour True in hourly dataframe
                            if np.all(np.in1d(codes[np.invert(nans)],valid_codes)):
                                year_status[hour_idx] = True
                        start_sec += 60*60
                        hour_idx +=1

                    # Make sure the hours match up
                    if hour_idx-1 != last_hour_idx:
                        ValueError('Incorrect number of hours were processed')

            status_bit[year] = pd.Series(year_status[:-1], year_hours[:-1], name='wind_status_bit')

        with open(sts_subpath/filename, 'wb') as fp:
            pd.to_pickle(status_bit, fp)

    return sts_subpath/filename


def parse_status_wind_yaw(sts_subpath, years, overwrite=False):

    # Set initial directory
    yaw_file = 'GE Turbine Yaw Dec 2019 to 2022.csv'
    filename = yaw_file[:-21]+' {} to {}'.format(years[0],years[-1])
    files = os.listdir(sts_subpath)
    
    if (filename not in files) or overwrite:

        yaw = {}
        
        yaw_mat = np.loadtxt(sts_subpath/yaw_file)
        yaw_mat = np.where(yaw_mat>=0, yaw_mat, np.full(np.size(yaw_mat), np.nan))
        yaw_label = 'Turbine Yaw [deg]'
        used_points = 0
        for year in years:
            hour_mat = np.transpose(np.array([[],]))
            year_start = pd.DatetimeIndex([str(year)])[0]
            year_end = pd.DatetimeIndex([str(year+1)])[0]
            times = pd.date_range(start=year_start, end=year_end, freq='10min')[:-1]
            hours = pd.date_range(start=year_start, end=year_end, freq='H')
            year_df = pd.DataFrame(yaw_mat[used_points:(used_points+len(times))], index=times, columns=[yaw_label])
            used_points += len(times)
            if year % 4 == 0:
                leap_start = pd.DatetimeIndex([str(year)+'-02-29'])[0]
                leap_end = pd.DatetimeIndex([str(year)+'-03-01'])[0]
                year_df = pd.concat((year_df.loc[(year_df.index<leap_start)],year_df.loc[(year_df.index>=leap_end)]))
                hours = hours[(hours<leap_start)|(hours>=leap_end)]
            for hour_idx, hour in enumerate(hours[:-1]):
                hour_df = year_df.loc[(year_df.index>hour)&(year_df.index<=hours[hour_idx+1])]
                mean_hour = np.array([[circmean(hour_df.loc[:,yaw_label], 360)]])
                hour_mat = np.vstack((hour_mat,mean_hour))

            yaw[year] = pd.Series(hour_mat[:,0], hours[:-1], name='wind_yaw')

        with open(sts_subpath/filename, 'wb') as fp:
            pd.to_pickle(yaw, fp)
    
    return sts_subpath/filename


def parse_status_wind_yaw_mismatch(sts_subpath, years, overwrite=False):

    # Set initial directory
    yaw_file = 'GE Turbine Yaw Dec 2019 to 2022.csv'
    tenmin_wind_file = 'August 2012 to October 2022 M5 wind 10 min'
    res_subpath = sts_subpath/'..'/'..'/'..'/'resource'/'wind'
    filename = yaw_file[:-21]+'mismatch {} to {}'.format(years[0],years[-1])
    files = os.listdir(sts_subpath)
    
    if (filename not in files) or overwrite:

        yaw_mismatch = {}
    
        yaw_mat = np.loadtxt(sts_subpath/yaw_file)
        yaw_mat = np.where(yaw_mat>=0, yaw_mat, np.full(np.size(yaw_mat), np.nan))
        tenmin_wind_df = pd.read_json(res_subpath/tenmin_wind_file)
        tenmin_wind_df.index = tenmin_wind_df.index.shift(-7, freq='H')
        direction_label = 'Direction (sonic_74m)'
        yaw_label = 'Turbine Yaw [deg]'
        wind_label = 'Speed (cup_ 80 m)'
        used_points = 0
        for year in years:
            hour_mat = np.transpose(np.array([[],[],[],[]]))
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
                mean_hour = np.array([[circmean(hour_df.loc[:,direction_label], 360),\
                                        circmean(hour_df.loc[:,yaw_label], 360),\
                                        circmean(mismatches, 180, -180),\
                                        np.mean(hour_df.loc[:,wind_label])]])
                hour_mat = np.vstack((hour_mat,mean_hour))

            yaw_mismatch[year] = pd.Series(hour_mat[:,2], hours[:-1], name='wind_yaw_mismatch')

        with open(sts_subpath/filename, 'wb') as fp:
            pd.to_pickle(yaw_mismatch, fp)
    
    return sts_subpath/filename