import numpy as np
import pandas as pd
import copy
from datetime import datetime, timedelta
from time import time, sleep
from pymodbus.client.sync import ModbusTcpClient
from pathlib import Path
import keyboard
import matplotlib.pyplot as plt


def sim_response(bess_dict,new_bess_kw,response_delay_ms,time_interval_ms,plot_flag=False):

    '''
    Simulates the response of the BESS to a new power setpoint.
    bess_dict is the state of the battery, read from controller at timestep 0
    new_bess_kw is the new power setpoint, to be written at timestep 1
    Controller will respond to this new setpoint with a delay of response_delay_ms
    The response will simulated at the next read cycle - timestep 2 (returned in bess_dict_next_read)
    AND the next write cycle - timestep 3 (returned in bess_dict_next_write)
    '''

    # Parse BESS dict
    bess_kwh = bess_dict['cap_kwh']
    bess_kw = bess_dict['kw']
    bess_soc = bess_dict['soc']
    bess_max_kw_s = bess_dict['max_kw_s']
    
    # Set up time series
    time_series = np.arange(time_interval_ms*3+1)
    kw_series = np.full(time_series.shape, bess_kw)
    soc_series = np.full(time_series.shape, bess_soc)

    # Calculate SOC % change rate per kw-ms
    soc_change_pct_kw_ms = 100/bess_kwh/3600/1000

    # Apply new setpoint at ramp rate
    sign = 1 if new_bess_kw > bess_kw else -1
    for i, ms in enumerate(np.arange(time_interval_ms+response_delay_ms,time_interval_ms*3+1)):
        kw_series[ms] = bess_kw+sign*min(i*bess_max_kw_s*i/1000,abs(new_bess_kw-bess_kw))

    # Calculate SOC change
    for ms in np.arange(1,time_interval_ms*3+1):
        soc_series[ms] = soc_series[ms-1] - (kw_series[ms-1]+kw_series[ms])/2*soc_change_pct_kw_ms

    if plot_flag:
        plt.ioff()
        plt.clf()
        plt.plot(time_series,kw_series)
        ax2 = plt.twinx()
        ax2.plot(time_series,soc_series)
        plt.show()
    
    # Make return dicts
    bess_dict_next_read = copy.deepcopy(bess_dict)
    bess_dict_next_write = copy.deepcopy(bess_dict)
    bess_dict_next_read['kw'] = kw_series[time_interval_ms*2]
    bess_dict_next_read['soc'] = soc_series[time_interval_ms*2]
    bess_dict_next_write['kw'] = kw_series[time_interval_ms*3]
    bess_dict_next_write['soc'] = soc_series[time_interval_ms*3]

    return bess_dict_next_read, bess_dict_next_write


def balance_power(target_kw,wind_kw,pv_kw,bess_dict,response_delay_ms,time_interval_ms,plot_flag=False):

    '''
    Balances the target plant output (produced by HOPP optimization) with the actual wind and PV generation
    Calculates the necessary BESS setpoint and checks that it will not exceed limits
    '''

    # Parse BESS dict
    bess_cap_kw = bess_dict['cap_kw']
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
    bess_kw_s = abs(new_bess_kw-bess_kw)/time_interval_ms*1000
    if bess_kw_s > bess_max_kw_s:
        if new_bess_kw > bess_kw:
            new_bess_kw = bess_kw+bess_max_kw_s*time_interval_ms/1000
        else:
            new_bess_kw = bess_kw-bess_max_kw_s*time_interval_ms/1000

    # Make sure not exceeding min/max SOC
    if new_bess_kw == bess_kw:
        plot_flag = False
    bess_dict_next_read, bess_dict_next_write = sim_response(bess_dict,new_bess_kw,response_delay_ms,time_interval_ms,plot_flag)
    new_soc = bess_dict_next_write['soc']
    if new_soc < bess_min:
        new_bess_kw = bess_kw+(new_bess_kw-bess_kw)*(bess_soc-bess_min)/(bess_soc-new_soc)
        bess_dict_next_read, _ = sim_response(bess_dict,new_bess_kw,response_delay_ms,time_interval_ms,plot_flag)
    elif new_soc > bess_max:
        new_bess_kw = bess_kw+(new_bess_kw-bess_kw)*(bess_soc-bess_max)/(bess_soc-new_soc)
        bess_dict_next_read, _ = sim_response(bess_dict,new_bess_kw,response_delay_ms,time_interval_ms,plot_flag)
    next_read_soc = bess_dict_next_read['soc']

    return new_bess_kw, next_read_soc


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
    write_values.append(int(time_value.microsecond/1e3))
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
                  registers[5],
                  registers[6])
    
    # Convert unsigned integers to signed
    for i in np.arange(7,12):
        if registers[i] >= 2**15:
            registers[i] = registers[i]-2**16
    
    # Identify remaining registers
    bess_pset = registers[7]
    bess_kw = registers[8]
    bess_soc = float(registers[9])/10
    wind_kw = registers[10]
    pv_kw = registers[11]

    # Pass BESS values to dict
    bess_dict['soc'] = bess_soc
    bess_dict['kw'] = bess_kw
    bess_dict['pset_kw'] = bess_pset

    return dt, wind_kw, pv_kw, bess_dict


if __name__ == '__main__':

    # Set up simulation conditions
    plotting = True
    plot_balancer = True
    sim_bess = True
    force_soc_start = True
    soc_start = 50.0
    response_delay_ms = 40
    time_interval_ms = 100

    # Set time window to test with
    hist_start_time = datetime(2022,8,5,13,30)
    # hist_start_time = datetime(2022,7,28,0,0)
    write_values = write_new_values(hist_start_time,0)
    hist_end_time = hist_start_time + timedelta(minutes=10)
    hist_duration = (hist_end_time-hist_start_time).total_seconds()
    
    # Set up BESS dict
    test_power_setpoint = 300
    bess_dict = {'cap_kw':1000,
                'cap_kwh':1000,
                'kw':0,
                'soc':soc_start,
                'min':10,
                'max':90,
                'max_kw_s':100000,
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
        p1 = plt.plot(hist_start_time,0,'.-',label='BESS Setpoint kW from HOPP to ARIES')
        p2 = plt.plot(hist_start_time,0,'.-',label='BESS Setpoint kW from ARIES to HOPP')
        p3 = plt.plot(hist_start_time,0,'.-',label='BESS Output kW from ARIES to HOPP - SIMULATED')
        p4 = plt.plot(hist_start_time,0,'.-',label='Wind kW from ARIES to HOPP')
        p5 = plt.plot(hist_start_time,0,'.-',label='PV kW from ARIES to HOPP')
        p6 = plt.plot(hist_start_time,0,'.-',label='Total kW from ARIES to HOPP')
        plt.xlim([hist_start_time,hist_end_time])
        plt.ylim([-bess_dict['cap_kw'],bess_dict['cap_kw']])
        plt.legend(loc='lower left')
        plt.xlabel('Real Time (BESS)')
        ax2 = plt.twiny()
        plt.xlim([hist_start_time,hist_end_time])
        plt.xlabel('Historical Time (Wind & PV)')
        axes.append(ax2)
        ax = plt.subplot(2,1,2)
        axes.append(ax)
        plt.title('BESS SOC [%] - SIMULATED @ 1% capacity to trigger SOC limit')
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

    # Write initial values and start timing loop 2 intervals before real_start_time is reached
    while datetime.now() < (real_start_time - timedelta(microseconds=2*time_interval_ms*1000)):
        pass
    timer_start = time()
    mbClient.write_registers(address=0,values=write_values)
    df_idx = pd.Index(pd.DatetimeIndex([real_start_time - timedelta(microseconds=time_interval_ms*1000)]),name=idx_name)
    hopp_df = pd.concat([hopp_df,pd.DataFrame(0,index=df_idx,columns=df_cols[:1])])
    timer_now = time()

    # Wait 1 interval and then read response
    while (timer_now-timer_start)*1e4<time_interval_ms:
        timer_now = time()
    timer_start = time()+time_interval_ms/1e3
    res = mbClient.read_holding_registers(address=0, count=12)
    dt, wind_kw, pv_kw, bess_dict = read_registers(res.registers,bess_dict)
    if sim_bess:
        bess_dict['kw'] = bess_dict['pset_kw']
        if force_soc_start:
            bess_dict['soc'] = soc_start
        prev_soc = copy.deepcopy(bess_dict['soc'])
        prev_kw = copy.deepcopy(bess_dict['kw'])
    bess_dict['pset_kw'], next_read_soc = balance_power(test_power_setpoint, wind_kw, pv_kw, bess_dict, response_delay_ms, time_interval_ms, False)
    df_data = [[bess_dict['pset_kw'], bess_dict['kw'], bess_dict['soc'], dt, wind_kw, pv_kw]]
    # aries_df = pd.concat([aries_df,pd.DataFrame(df_data,index=df_idx,columns=df_cols)])
        
    # Execute for:
    # - the full duration of history, 
    # - until modbus indicates control has been interrupted on ARIES side,
    # - or user interrupts by pressing escape
    user_quit = False
    ms_elapsed = -time_interval_ms
    if plotting:
        plot_made_in_time = False
    while ((timer_now-timer_start)<=hist_duration) and bess_dict['hopp_control'] and not user_quit:
        # Generate timestamps for the upcoming second while waiting
        upcoming_hist_time = hist_start_time + timedelta(milliseconds=ms_elapsed+2*time_interval_ms)
        upcoming_real_time = real_start_time + timedelta(milliseconds=ms_elapsed+time_interval_ms)
        df_idx = pd.Index(pd.DatetimeIndex([upcoming_real_time]),name=idx_name)
        # Wait for the next second
        timer_now = time()
        while (timer_now-timer_start)*1e3<(ms_elapsed+time_interval_ms):
            # Check that plot generation did not already delay time past t=0
            if plotting:
                if not plot_made_in_time:
                    plot_made_in_time = True
            timer_now = time()
        ms_elapsed += time_interval_ms
        # Read on the odd time intervals
        if ms_elapsed%(2*time_interval_ms) == time_interval_ms:
            res = mbClient.read_holding_registers(address=0, count=12)
            dt, wind_kw, pv_kw, bess_dict = read_registers(res.registers,bess_dict)
            if sim_bess:
                bess_dict['kw'] = bess_dict['pset_kw']
                bess_dict['soc'] = next_read_soc
                prev_soc = copy.deepcopy(bess_dict['soc'])
                prev_kw = copy.deepcopy(bess_dict['kw'])
            df_data = [[bess_dict['pset_kw'], bess_dict['kw'], bess_dict['soc'], dt, wind_kw, pv_kw]]
            bess_dict['pset_kw'], next_read_soc = balance_power(test_power_setpoint, wind_kw, pv_kw, bess_dict, response_delay_ms, time_interval_ms, False)
            aries_df = pd.concat([aries_df,pd.DataFrame(df_data,index=df_idx,columns=df_cols)])
            print('Read loop time: {:.3f} ms'.format((time()-timer_start)*1e3-ms_elapsed))
        # Write on the evens
        else:        
            new_kw = bess_dict['pset_kw']
            hopp_df = pd.concat([hopp_df,pd.DataFrame(new_kw,index=df_idx,columns=df_cols[:1])])
            if new_kw < 0:
                new_kw = 2**16+new_kw
            write_values = write_new_values(upcoming_hist_time,new_kw)
            mbClient.write_registers(address=0,values=write_values)
            print('Write loop time: {:.3f} ms'.format((time()-timer_start)*1e3-ms_elapsed))
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
                plt.xlim([hist_start_time-timedelta(seconds=1),upcoming_hist_time])
                plt.ylim([-bess_dict['cap_kw'],bess_dict['cap_kw']])
                plt.sca(axes[2])
                p7[0].set_data(aries_df.index.values,aries_df['BESS SOC [%]'].values)
                plt.xlim([real_start_time-timedelta(seconds=1),upcoming_real_time+timedelta(seconds=1)])
                plt.ylim([0,100])
                print('Plot loop time: {:.3f} ms'.format((time()-plot_start)*1e3))
                plt.pause(0.001)
                plot_made_in_time = False
            else:
                # raise TimeoutError('Took longer than polling interval to generate plot - turn off plotting!')
                pass

    if plotting:
        plt.ioff()
        plt.show()