import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

'''
Imports m5 met tower wind data closest to GE 1.5MW turbine hub height (80m)
and formats it for SAM usage
'''

def import_m5_wind(start_month='August 2012', end_month='August 2012'):

    current_path = Path(__file__).parent.absolute()
    m5_path = Path('Y:/Wind/WindWeb/MetData/135mData/M5Twr/10min/txt')

    # Define time period and filenames
    months = pd.date_range(start=start_month,end=end_month,freq='MS')
    filenames = [str(i.year)+'_'+i.month_name()+'.txt' for i in months]

    # Select variables to import
    var_names = ['Air Temperature (87 m)','Air Pressure (87 m)','Direction (sonic_74m)','Speed (cup_ 80 m)']
    qc_var_names = [i+' QC' for i in var_names]

    # Loop through files with 10-minute averages on Y drive and import
    tenmin_df = pd.DataFrame()
    for fn in filenames:
        if Path.exists(m5_path / fn):
            print('Importing {}...'.format(fn))
            file_df = pd.read_csv(m5_path / fn, skiprows=[0,1,2,3,4,5,6,8,9], index_col='Date', parse_dates=True, header=0, usecols=var_names+['Date'], dayfirst=True)
            qc_df = pd.read_csv(m5_path / fn, skiprows=[0,1,2,3,4,5,6,8,9], index_col='Date', parse_dates=True, header=0, usecols=qc_var_names+['Date'], dayfirst=True)
            # Turn data values to NaN if quality control value is not 1 
            file_df.where(qc_df.values==1, other=float('NaN'), inplace=True)
            tenmin_df = pd.concat((tenmin_df,file_df), axis=0)

    # Convert temperature from K to C
    tenmin_df.loc[:,var_names[0]] = np.add(tenmin_df.loc[:,var_names[0]],-273.15)
    # Convert pressure from mbar to atm
    tenmin_df.loc[:,var_names[1]] = np.multiply(tenmin_df.loc[:,var_names[1]],.001)

    # Remove duplicate times
    tenmin_df = tenmin_df.loc[np.invert(tenmin_df.index.duplicated())]

    # Save to .json
    tenmin_df.to_json(current_path/(start_month+' to '+end_month+' M5 wind 10 min'))

## Turn 10 minute averages into hourly averages, binning the 10 min means w.r.t. hourly means
def downsample_m5_wind(bin_resolution=4, start_month='August 2012', end_month='August 2012'):

    # Load imported data
    current_path = Path(__file__).parent.absolute()
    tenmin_df = pd.read_json(current_path/(start_month+' to '+end_month+' M5 wind 10 min'))

    # Define time period and variable names
    months = pd.date_range(start=start_month,end=end_month,freq='MS')
    var_names = tenmin_df.columns.values
    dir_idx = np.argmax(var_names=='Direction (sonic_74m)')

    # Make wind speed bins (centered on .125, .375, etc.; edges at .25, .5, etc.; inclusive at left edge)
    bin_var = 'Speed (cup_ 80 m)'
    max_speed = 20
    tenmin_bins = np.arange(1/2/bin_resolution,max_speed+1/bin_resolution,1/bin_resolution)
    hour_bins = np.arange(1/2/bin_resolution,max_speed+1/bin_resolution,1/bin_resolution)
    bin_mat = np.zeros((len(hour_bins),len(tenmin_bins)))

    # Make hourly timeseries for each year
    start_year = months[0].year
    end_year = months[-1].year+1
    years = pd.date_range(start=str(start_year),end=str(end_year),freq='YS')
    for year_idx, year_start in enumerate(years[:-1]):
        hours = pd.date_range(start=year_start,end=years[year_idx+1],freq='H')
        
        # Get hourly averages
        hour_df = pd.DataFrame(index=hours[:-1], columns=var_names)
        for hour_idx, hour_start in enumerate(hours[:-1]):
            if hour_idx % 24 == 0:
                print('Downsampling {}...'.format(hour_start.strftime('%Y-%m-%d')))
            hour_end = hours[hour_idx+1]
            tenmin_values = tenmin_df.loc[(tenmin_df.index>hour_start)&(tenmin_df.index<=hour_end)]
            if tenmin_values.values.any() and ~np.any(np.isnan(tenmin_values)):
                hour_averages = np.mean(tenmin_values,axis=0)
                hour_averages[dir_idx] = stats.circmean(tenmin_values.iloc[:,dir_idx], high=360)
                hour_df.loc[hour_start,:] = hour_averages
                
                # Bin 10 min averages by hourly average
                hour_value = hour_df.loc[hour_start,bin_var]
                hour_idx = int(hour_value*bin_resolution)
                hour_idx = min(hour_idx,bin_resolution*max_speed)
                for tenmin_value in tenmin_values.loc[:,bin_var].values:
                    tenmin_idx = int(tenmin_value*bin_resolution)
                    tenmin_idx = min(tenmin_idx,bin_resolution*max_speed)
                    bin_mat[hour_idx,tenmin_idx] += 1
        
        # Save to .json
        export_name = 'M5 unformatted hourly wind data {}'.format(year_start.strftime('%Y'))
        if start_year == year_start.year and months[0].month != 1:
            export_name += ' ({} through '.format(months[0].month_name())
            if months[-1].year == start_year:
                export_name += '{})'.format(months[-1].month_name())
            else:
                export_name += 'December)'
        elif months[-1].year == year_start.year and months[-1].month != 12:
            export_name += ' (January through {})'.format(months[-1].month_name())
        hour_df.to_json(current_path/export_name)
    save_bins = pd.DataFrame(bin_mat, index=hour_bins, columns=tenmin_bins)
    save_bins.to_json(current_path/(start_month+' to '+end_month+' M5 wind {:.0f} bins 1 hr 10 min'.format(bin_resolution)))
    
    # # Get rid of leap day
    # hour_df = hour_df['02-29'!=hour_df.index.strftime('%m-%d')]

## Plot wind bins
def plot_m5_wind_bins(bin_resolution=4, start_month='August 2012', end_month='August 2012'):

    # Load binned data
    current_path = Path(__file__).parent.absolute()
    save_bins = pd.read_json(current_path/(start_month+' to '+end_month+' M5 wind {:.0f} bins 1 hr 10 min'.format(bin_resolution)), convert_axes=False)
    hour_bins = save_bins.index.values.astype(np.float)
    tenmin_bins = save_bins.index.values.astype(np.float)
    max_speed = np.mean(tenmin_bins[-2:])
    bin_mat = save_bins.values
    
    # Set offset for each bin
    offset = 5

    # Set minimum N for distribution to be valid
    min_N = 20

    # Set up axes
    plt.clf()
    ax1 = plt.gca()
    ax1.set_xlim([0,max_speed])
    ax1.set_xlabel('10 minute average wind speed bins, 0.25 intervals [m/s]')
    ax1.set_ylabel('Percentage of 10 minute periods average wind speed in this bin [m/s]\n(offset by {:.0f} for each hourly average)'.format(offset))
    ax1.set_ylim([-offset/2,max_speed*offset*bin_resolution-offset/2])
    ax2 = plt.twinx()
    ax2.set_ylabel('Hourly average wind speed [m/s]')
    ax2.set_ylim([0,max_speed])
    
    # Plot each bin, normalized
    for bin_idx, bin in enumerate(bin_mat):
        color = [1,0,0]
        if np.sum(bin) >= min_N:
            color[0] = 0
        if np.sum(bin) > 0:
            pct_bin = 100*bin/np.sum(bin)
            ax1.plot(tenmin_bins[:-1],pct_bin[:-1]+bin_idx*offset,'-',color=color)
        T = ax1.text(21,bin_idx*offset,'N = {:.0f}'.format(np.sum(bin)),color=color)
        T.set_fontsize(8)

    plt.show()

## Refactor 10 min power curve to 1 hr power curve
def refactor_power_curve_m5_wind(power_curve, rotor_diameter, air_density=1.225, bin_resolution=4, start_month='August 2012', end_month='August 2012'):

    # Load binned data
    current_path = Path(__file__).parent.absolute()
    save_bins = pd.read_json(current_path/(start_month+' to '+end_month+' M5 wind {:.0f} bins 1 hr 10 min'.format(bin_resolution)), convert_axes=False)
    save_bins2 = pd.read_json(current_path/(start_month+' to '+end_month+' M5 wind 4 bins 1 hr 10 min'), convert_axes=False)
    hour_bins = save_bins.index.values.astype(np.float)
    hour_bins2 = save_bins2.index.values.astype(np.float)
    tenmin_bins = save_bins.index.values.astype(np.float)
    tenmin_bins2 = save_bins2.index.values.astype(np.float)
    max_speed = np.mean(tenmin_bins[-2:])
    bin_mat = save_bins.values
    bin_mat2 = save_bins2.values
    
    # Load power curve
    curve_data = pd.read_csv(current_path/power_curve)
    wind_speed = np.hstack(([0],curve_data['Wind Speed [m/s]'].values))
    curve_power = np.hstack(([0],curve_data['Power [kW]'].values))

    # Interpolate bin speeds
    tenmin_bin_curve_idxs = []
    tenmin_interp_fracs = []
    for bin in tenmin_bins:
        tenmin_dists = wind_speed-bin
        bin_curve_idx = np.argmax(tenmin_dists>0)
        tenmin_bin_curve_idxs.append(bin_curve_idx)
        tenmin_interp_fracs.append(tenmin_dists[bin_curve_idx]/(tenmin_dists[bin_curve_idx]-tenmin_dists[bin_curve_idx-1]))
    interp_power = np.add(np.multiply(curve_power[np.subtract(tenmin_bin_curve_idxs,1)],tenmin_interp_fracs),\
                        np.multiply(curve_power[tenmin_bin_curve_idxs],np.subtract(1,tenmin_interp_fracs)))

    # Refactor the power curve to hourly based on the bins
    hour_power = []
    valid_bins = []
    for bin_idx, bin in enumerate(bin_mat[:-1]):
        if np.sum(bin) > 0:
            frac_bin = bin/np.sum(bin)
            hour_power.append(np.sum(np.multiply(frac_bin,interp_power)))
            valid_bins.append(hour_bins[bin_idx])

    # Interpolate bin speeds
    tenmin_bin_curve_idxs2 = []
    tenmin_interp_fracs2 = []
    for bin in tenmin_bins2:
        tenmin_dists = wind_speed-bin
        bin_curve_idx = np.argmax(tenmin_dists>0)
        tenmin_bin_curve_idxs2.append(bin_curve_idx)
        tenmin_interp_fracs2.append(tenmin_dists[bin_curve_idx]/(tenmin_dists[bin_curve_idx]-tenmin_dists[bin_curve_idx-1]))
    interp_power2 = np.add(np.multiply(curve_power[np.subtract(tenmin_bin_curve_idxs2,1)],tenmin_interp_fracs2),\
                        np.multiply(curve_power[tenmin_bin_curve_idxs2],np.subtract(1,tenmin_interp_fracs2)))

    # Refactor the power curve to hourly based on the bins
    hour_power2 = []
    valid_bins2 = []
    for bin_idx, bin in enumerate(bin_mat2[:-1]):
        if np.sum(bin) > 0:
            frac_bin2 = bin/np.sum(bin)
            hour_power2.append(np.sum(np.multiply(frac_bin2,interp_power2)))
            valid_bins2.append(hour_bins2[bin_idx])
            
    # Plot original and refactored curves to check
    plt.plot(wind_speed,curve_power,'.-')
    plt.plot(valid_bins,hour_power,'.-')
    plt.plot(valid_bins2,hour_power2,'.-')
    plt.show()

    # Make new powercurve file
    hour_power_mat = np.hstack((np.transpose(np.array([valid_bins])),np.transpose(np.array([hour_power]))))
    speed = hour_power_mat[:,0]
    power = hour_power_mat[:,1]
    available_power = 1/2*np.pi*(rotor_diameter/2)**2*air_density*np.power(speed,3)
    c_p = np.divide(power*1000,available_power)
    hour_power_mat = np.hstack((hour_power_mat,np.transpose(np.array([c_p]))))
    columns = ['Wind Speed [m/s]','Power [kW]','Cp [-]']
    refactored_power_curve = pd.DataFrame(hour_power_mat, columns=columns)
    refactored_power_curve.to_csv(current_path/(power_curve[:-4]+'_Refactored_Hourly.csv'),index=False)