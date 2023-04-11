import numpy as np
import pandas as pd
import copy
from datetime import datetime, timedelta
from time import time, sleep
from pymodbus.client.sync import ModbusTcpClient
from pathlib import Path
import keyboard
import matplotlib.pyplot as plt

def balance_power(target_kw,wind_kw,pv_kw,bess_dict):

    '''
    Balances the target plant output (produced by HOPP optimization) with the actual wind and PV generation
    Calculates the necessary BESS setpoint and checks that it will not exceed limits
    '''

    # Parse BESS dict
    bess_cap_kw = bess_dict['cap_kw']
    bess_kwh = bess_dict['cap_kwh']
    bess_kw = bess_dict['kw']
    bess_soc = bess_dict['soc']
    bess_min = bess_dict['min']
    bess_max = bess_dict['max']
    bess_max_kw_s = bess_dict['max_kw_s']

    # Calculate BESS power to match load
    new_bess_kw = target_kw - wind_kw - pv_kw

    # Make sure not exceeding capacity
    if abs(new_bess_kw) > bess_cap_kw:
        if new_bess_kw > bess_cap_kw:
            new_bess_kw = bess_cap_kw
        else:
            new_bess_kw = -bess_cap_kw

    # Make sure not exceeding ramp rate
    bess_kw_s = abs(new_bess_kw-bess_kw)
    if bess_kw_s > bess_max_kw_s:
        if new_bess_kw > bess_kw:
            new_bess_kw = bess_kw+bess_max_kw_s
        else:
            new_bess_kw = bess_kw-bess_max_kw_s

    # Make sure not exceeding min/max SOC
    new_soc = (bess_soc/100*bess_kwh-(bess_kw+new_bess_kw*2)/3600)/bess_kwh*100
    if new_soc < bess_min:
        new_bess_kw = (bess_soc-bess_kw/3600/bess_kwh*100-bess_min)/100*bess_kwh*1800
    elif new_soc > bess_max:
        new_bess_kw = (bess_soc-bess_kw/3600/bess_kwh*100-bess_max)/100*bess_kwh*1800

    return new_bess_kw


def write_new_values(time_value,new_kw):

    '''
    Creates a list of values to write to the modbus register
    '''

    write_values = []
    write_values.append(int(time_value.year))
    write_values.append(int(time_value.month))
    write_values.append(int(time_value.day))
    write_values.append(int(time_value.hour))
    write_values.append(int(time_value.minute))
    write_values.append(int(time_value.second))
    write_values.append(int(new_kw))

    return write_values


def read_registers(registers,bess_dict):

    '''
    Reads modbus holding registers
    '''

    # First 6 values are a datetime - either live (if first register is 0) or historical
    if registers[0] == 0:
        registers[0] == datetime.now().year
    dt = datetime(registers[0],
                  registers[1],
                  registers[2],
                  registers[3],
                  registers[4],
                  registers[5])
    
    # Convert unsigned integers to signed
    for i in np.arange(6,11):
        if registers[i] >= 2**15:
            registers[i] = registers[i]-2**16
    
    # Identify remaining registers
    bess_pset = registers[6]
    bess_kw = registers[7]
    bess_soc = float(registers[8])/10
    wind_kw = registers[9]
    pv_kw = registers[10]

    # Pass BESS values to dict
    bess_dict['soc'] = bess_soc
    bess_dict['kw'] = bess_kw
    bess_dict['pset_kw'] = bess_pset

    return dt, wind_kw, pv_kw, bess_dict


if __name__ == '__main__':

    # Set up simulation conditions
    plotting = True
    sim_bess = True
    force_soc_start = True
    soc_start = 50

    # Set time window to test with
    hist_start_time = datetime(2022,7,28,0)
    write_values = write_new_values(hist_start_time,0)
    hist_end_time = hist_start_time + timedelta(minutes=10)
    hist_duration = (hist_end_time-hist_start_time).total_seconds()
    
    # Set up BESS dict
    test_power_setpoint = 300
    bess_dict = {'cap_kw':1000,
                'cap_kwh':10,
                'kw':0,
                'soc':soc_start,
                'min':10,
                'max':90,
                'max_kw_s':4000,
                'pset_kw':test_power_setpoint,
                'hopp_control':False}
    
    # Locate command directory
    current_dir = Path(__file__).parent.absolute()
    command_dir = current_dir/'..'/'..'/'results'/'weeklong_sim'

    # Test modbus connection
    mbClient = ModbusTcpClient(host='192.174.56.29',port=502)
    connected = mbClient.connect()
    if not connected:
        raise ConnectionError('Cannot connect to modbus host!')
    
    # TODO: check if in control of BESS (pretending we are for testing purposes)
    bess_dict['hopp_control'] = True

    # Start dataframes
    idx_name = 'Timestamp - Real Time'
    df_cols = [ 'BESS P Set [kW]',
                'BESS P Out [kW]',
                'BESS SOC [%]',
                'Plant Set [kW]'
                'Timestamp - History',
                'Wind [kW]',
                'PV [kW]']
    df_idxs = pd.Index(pd.DatetimeIndex([]),name=idx_name)
    hopp_df = pd.DataFrame(None,index=df_idxs,columns=df_cols[:1])
    aries_df = pd.DataFrame(None,index=df_idxs,columns=df_cols)
    
    # Make blank plot with fake data (before clock starts, to save time)
    if plotting:
        plt.ion()
        axes = []
        ax = plt.subplot(2,1,1)
        axes.append(ax)
        plt.title('Power Output [kW]')
        p1 = plt.plot(hist_start_time,0,'.-',label='BESS Set kW from HOPP to ARIES')
        p2 = plt.plot(hist_start_time,0,'.-',label='BESS Set kW from ARIES to HOPP')
        p3 = plt.plot(hist_start_time,0,'.-',label='BESS Actual kW from ARIES to HOPP')
        p4 = plt.plot(hist_start_time,0,'.-',label='Wind kW from ARIES to HOPP')
        p5 = plt.plot(hist_start_time,0,'.-',label='PV kW from ARIES to HOPP')
        p6 = plt.plot(hist_start_time,0,'.-',label='Total kW from ARIES to HOPP')
        plt.xlim([hist_start_time,hist_end_time])
        plt.ylim([-bess_dict['cap_kw'],bess_dict['cap_kw']])
        plt.legend(loc='lower right')
        ax = plt.subplot(2,1,2)
        axes.append(ax)
        plt.title('BESS SOC [%]')
        p7 = plt.plot(hist_start_time,0,'.-')
        plt.xlim([hist_start_time,hist_end_time])
        plt.ylim([-bess_dict['min'],bess_dict['max']])
        plt.gcf().set_tight_layout(True)
        plt.gcf().set_size_inches(12,8)
        plt.pause(0.001)

    # Push real start time forward to an even second on the clock
    real_start_time = datetime.now()
    us_lag = 1e5
    if 1e6-real_start_time.microsecond < us_lag:
        real_start_time = real_start_time + timedelta(seconds=4)
    else:       
        real_start_time = real_start_time + timedelta(seconds=3)
    if real_start_time.second%2 == 1:
        real_start_time = real_start_time + timedelta(seconds=1)
    real_start_time = real_start_time.replace(microsecond=0)
    offset = real_start_time-hist_start_time

    # Write initial values and start timing loop 2 seconds before real_start_time is reached
    while datetime.now() < (real_start_time - timedelta(seconds=2)):
        pass
    timer_start = time()
    mbClient.write_registers(address=0,values=write_values)
    df_idx = pd.Index(pd.DatetimeIndex([real_start_time - timedelta(seconds=1)]),name=idx_name)
    hopp_df = pd.concat([hopp_df,pd.DataFrame(0,index=df_idx,columns=df_cols[:1])])
    timer_now = time()

    # Wait 1 second and then read response
    while (timer_now-timer_start)<1:
        timer_now = time()
    timer_start = time()+1
    res = mbClient.read_holding_registers(address=0, count=11)
    dt, wind_kw, pv_kw, bess_dict = read_registers(res.registers,bess_dict)
    df_data = [[bess_dict['pset_kw'], bess_dict['kw'], bess_dict['soc'], dt, wind_kw, pv_kw]]
    bess_dict['pset_kw'] = balance_power(test_power_setpoint, wind_kw, pv_kw, bess_dict)
    aries_df = pd.concat([aries_df,pd.DataFrame(df_data,index=df_idx,columns=df_cols)])
    if sim_bess:
        if force_soc_start:
            bess_dict['soc'] = soc_start
        prev_soc = copy.deepcopy(bess_dict['soc'])
        prev_kw = copy.deepcopy(bess_dict['kw'])
        
    # Execute for:
    # - the full duration of history, 
    # - until modbus indicates control has been interrupted on ARIES side,
    # - or user interrupts by pressing escape
    user_quit = False
    sec_elapsed = -1
    if plotting:
        plot_made_in_time = False
    while ((timer_now-timer_start)<=hist_duration) and bess_dict['hopp_control'] and not user_quit:
        # Generate timestamps for the upcoming second while waiting
        upcoming_hist_time = hist_start_time + timedelta(seconds=sec_elapsed+2)
        upcoming_real_time = real_start_time + timedelta(seconds=sec_elapsed+1)
        df_idx = pd.Index(pd.DatetimeIndex([upcoming_real_time]),name=idx_name)
        # Wait for the next second
        timer_now = time()
        while (timer_now-timer_start)<(sec_elapsed+1):
            # Check that plot generation did not already delay time past t=0
            if plotting:
                if not plot_made_in_time:
                    plot_made_in_time = True
            timer_now = time()
        sec_elapsed += 1
        # Read on the odd seconds
        if sec_elapsed%2 == 1:
            res = mbClient.read_holding_registers(address=0, count=11)
            dt, wind_kw, pv_kw, bess_dict = read_registers(res.registers,bess_dict)
            if sim_bess:
                bess_dict['kw'] = bess_dict['pset_kw']
                bess_dict['soc'] = prev_soc - (prev_kw+bess_dict['kw'])/3600/bess_dict['cap_kwh']*100
                prev_soc = copy.deepcopy(bess_dict['soc'])
                prev_kw = copy.deepcopy(bess_dict['kw'])
            df_data = [[bess_dict['pset_kw'], bess_dict['kw'], bess_dict['soc'], dt, wind_kw, pv_kw]]
            bess_dict['pset_kw'] = balance_power(test_power_setpoint, wind_kw, pv_kw, bess_dict)
            aries_df = pd.concat([aries_df,pd.DataFrame(df_data,index=df_idx,columns=df_cols)])
            print('Read loop time: {:.3f} sec'.format(time()-timer_start-sec_elapsed))
        # Write on the evens
        else:        
            new_kw = bess_dict['pset_kw']
            hopp_df = pd.concat([hopp_df,pd.DataFrame(new_kw,index=df_idx,columns=df_cols[:1])])
            if new_kw < 0:
                new_kw = 2**16+new_kw
            write_values = write_new_values(upcoming_hist_time,new_kw)
            mbClient.write_registers(address=0,values=write_values)
            print('Write loop time: {:.3f} sec'.format(time()-timer_start-sec_elapsed))
        if keyboard.is_pressed('esc'):
            user_quit = True
        if plotting:
            if plot_made_in_time:
                plot_start = time()
                plt.sca(axes[0])
                p1[0].set_data(hopp_df.index.values,hopp_df['BESS P Set [kW]'].values)
                p2[0].set_data(aries_df.index.values,aries_df['BESS P Set [kW]'].values)
                p3[0].set_data(aries_df.index.values,aries_df['BESS P Out [kW]'].values)
                p4[0].set_data(aries_df.index.values,aries_df['Wind [kW]'].values)
                p5[0].set_data(aries_df.index.values,aries_df['PV [kW]'].values)
                p6[0].set_data(aries_df.index.values,aries_df['BESS P Set [kW]'].values+\
                                                        aries_df['Wind [kW]'].values+\
                                                        aries_df['PV [kW]'].values)
                plt.xlim([real_start_time-timedelta(seconds=1),upcoming_real_time+timedelta(seconds=1)])
                plt.ylim([-bess_dict['cap_kw'],bess_dict['cap_kw']])
                plt.sca(axes[1])
                p7[0].set_data(aries_df.index.values,aries_df['BESS SOC [%]'].values)
                plt.xlim([real_start_time-timedelta(seconds=1),upcoming_real_time+timedelta(seconds=1)])
                plt.ylim([0,100])
                print('Plot loop time: {:.3f} sec'.format(time()-plot_start))
                plt.pause(0.001)
                plot_made_in_time = False
            else:
                raise TimeoutError('Took longer than polling interval to generate plot - turn off plotting!')

    if plotting:
        plt.ioff()
        plt.show()