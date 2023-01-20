import os
import numpy as np
import pandas as pd
from time import time, localtime
from pathlib import Path

# Set initial directory
examples_dir = Path(__file__).parent.absolute()
status_dir = examples_dir / "resource_files" / "Status Data Secondly"
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

hour_df.to_json(examples_dir / "resource_files" / "GE15_IEC_validity_hourly_2019_2022")