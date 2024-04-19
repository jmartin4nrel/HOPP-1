import socket
import json
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from comms_tracking import setup_tracking, update_trackers, updateSOCplot
from aries_comms import aries_output_unpack, aries_input_pack
from hopp import ROOT_DIR

def batt_balance(HOPPdict, ARIESdict, trackers, simulate_SOC):

    aries_time = trackers[1]
    aries_xdata = trackers[2]

    # Get commanded electrolyzer output from HOPP
    elyzer_kw = HOPPdict['elyzer_kw']

    # Get generation from ARIES
    wind_kw = ARIESdict['wind']
    wave_kw = ARIESdict['wave']
    solar_kw = ARIESdict['solar']
    batt_kw = ARIESdict['batt']

    # Look up battery parameters
    cap_kw = HOPPdict['batt_limits']['cap_kw']
    cap_kwh = HOPPdict['batt_limits']['cap_kwh']
    max_SOC = HOPPdict['batt_limits']['max_SOC']
    min_SOC = HOPPdict['batt_limits']['min_SOC']
    hoppSOC = HOPPdict['soc']
    ariesSOC = aries_xdata['soc']

    if simulate_SOC:
        # Calc new SOC
        if len(ariesSOC) == 0:
            initSOC = hoppSOC[0]
            sec_elapsed = float(np.diff(pd.DatetimeIndex(ARIESdict['aries_time'])))/1e9
        else:
            initSOC = aries_xdata['soc'][-1]
            sec_elapsed = float(np.diff(aries_time[[-3,-1]]))/1e9
        newSOC = (initSOC/100-sec_elapsed*batt_kw[0]/cap_kwh/3600)*100
        aries_xdata['soc'].append(newSOC)
    else:
        newSOC = ariesSOC[-1]
        if len(ariesSOC) == 1:
            sec_elapsed = float(np.diff(pd.DatetimeIndex(ARIESdict['aries_time'])))/1e9
        else:
            sec_elapsed = float(np.diff(aries_time[[-3,-1]]))/1e9

    # Calculate battery generation needed
    new_batt_kw = elyzer_kw-wind_kw[0]-wave_kw[0]-solar_kw[0]

    # Put 'bumpers' at capacity, SOC limits
    new_batt_kw = np.max([-cap_kw,new_batt_kw])
    new_batt_kw = np.min([cap_kw,new_batt_kw])
    expectedSOC = (newSOC/100-sec_elapsed*new_batt_kw/cap_kwh/3600)*100
    # if (newSOC<max_SOC) and (expectedSOC>max_SOC):
    if expectedSOC>max_SOC:
        if newSOC<max_SOC:
            over_fraction = (expectedSOC-max_SOC)/(expectedSOC-newSOC)
            new_batt_kw = new_batt_kw*(1-over_fraction)
        else:
            new_batt_kw = np.max([new_batt_kw,0])
    elif expectedSOC<min_SOC:
        if newSOC>min_SOC:
            under_fraction = (expectedSOC-min_SOC)/(expectedSOC-newSOC)
            new_batt_kw = new_batt_kw*(1-under_fraction)
        else:
            new_batt_kw = np.min([new_batt_kw,0])

    HOPPdict['commands']['bess_kw'] = new_batt_kw
    
    return HOPPdict, trackers

def realtime_balancer(simulate_aries=True, acceleration=1, simulate_SOC=True):

    bufferSize_HOPP  = 4096
    bufferSize_ARIES  = 40*4
    plotting = True

    # Setup UDP receive from HOPP
    localIP     = "127.0.0.1"
    localPort   = 20001
    serverAddressPort   = (localIP, localPort)
    recvHOPPsocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    recvHOPPsocket.bind(serverAddressPort)
    recvHOPPsocket.settimeout(60)

    if simulate_aries:
        # Setup UDP receive from ARIES
        localIP     = "127.0.0.1"
        localPort   = 20002
        serverAddressPort   = (localIP, localPort)
        recvARIESsocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        recvARIESsocket.bind(serverAddressPort)
        recvARIESsocket.settimeout(60)

        # Setup UDP send to ARIES
        localIP     = "127.0.0.1"
        localPort   = 20003
        sendARIESaddress  = (localIP, localPort)
        sendARIESsocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

    else:
        
        # Setup UDP send to ARIES
        remoteIP     = "10.81.15.88"
        remotePort   = 9010
        sendARIESaddress  = (remoteIP, remotePort)
        sendARIESsocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        sendARIESsocket.connect(sendARIESaddress)

    # Setup UDP send to HOPP
    localIP     = "127.0.0.1"
    localPort   = 20004
    sendHOPPaddress  = (localIP, localPort)
    sendHOPPsocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

    # Read in ARIES placeholder signal
    aries_sig_fn = ROOT_DIR.parent / 'examples' / 'outputs' / 'placeholder_ARIES_with_wind_1s.csv'
    aries_signals = pd.read_csv(aries_sig_fn,parse_dates=True,index_col=0,infer_datetime_format=True)

    # Set up trackers and plots if necessary
    if plotting:
        plt.ion()
    trackers = setup_tracking(plotting)
    last_aries_time = None

    # Indicate ready to start ARIES
    print("Ready to start ARIES")

    # Sync start time to clock
    real_start_time = time.time()
    hopp_start_time = float(aries_signals.index.values[0])/1e9

    while(True):

        # Setup UDP receive from ARIES
        remoteIP     = "10.81.17.104"
        remotePort   = 9010
        serverAddressPort   = (remoteIP, remotePort)
        recvARIESsocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        recvARIESsocket.bind(serverAddressPort)
        recvARIESsocket.settimeout(60)

        # Receive data from ARIES
        ARIESpair = recvARIESsocket.recvfrom(bufferSize_ARIES)
        ARIESraw = ARIESpair[0]
        ARIES_output_dict = aries_output_unpack(ARIESraw)
        # ARIESdict = json.loads(ARIESraw)

        # Keep track of aries time
        if last_aries_time is not None: last_aries_time = copy.deepcopy(aries_time)
        aries_time = pd.Timestamp(((time.time()-real_start_time)*acceleration+hopp_start_time)*1e9)
        if last_aries_time is None: last_aries_time = pd.Timestamp(hopp_start_time*1e9)

        # Receive data from HOPP
        HOPPpair = recvHOPPsocket.recvfrom(bufferSize_HOPP)
        HOPPraw = HOPPpair[0]
        HOPPdict = json.loads(HOPPraw)

        # Replace insolation and wind speeds with real-time wind speeds
        most_recent_time = aries_signals.index.values[np.argmin(aries_time>aries_signals.index)]
        insol = aries_signals.loc[most_recent_time,'poa']
        HOPPdict['commands']['poa'] = insol
        num_turbs = sum(pd.Series(list(HOPPdict['commands'].keys())).str.contains('wind_spd').values)
        for turb_num in range(num_turbs):
            wind_spd = aries_signals.loc[most_recent_time,'wind_vel_{:02}'.format(turb_num)]
            HOPPdict['commands']['wind_vel_{:02}'.format(turb_num)] = wind_spd

        # Condense wind and wave generation to one dictionary item
        wind_gen = 0
        num_turbs = sum(pd.Series(list(HOPPdict['commands'].keys())).str.contains('wind_spd').values)
        for turb_num in range(num_turbs):
            wind_gen += ARIES_output_dict['wind'+str(turb_num)]
        ARIES_output_dict['wind'] = wind_gen
        wave_gen = 0
        num_wecs = sum(pd.Series(list(HOPPdict['commands'].keys())).str.contains('wave_prod').values)
        for wec_num in range(num_wecs):
            wave_gen += ARIES_output_dict['wave'+str(wec_num)]
        ARIES_output_dict['wave'] = wave_gen

        # Turn the ARIES output dict into a two-item list
        ARIES_output_dict['aries_time'] = [last_aries_time, aries_time]
        for key in ARIES_output_dict.keys():
            if key != 'aries_time':
                ARIES_output_dict[key] = [ARIES_output_dict[key],ARIES_output_dict[key]]

        trackers = update_trackers(trackers, HOPPdict, ARIES_output_dict, plotting, simulate_SOC)

        # Balance battery output from real-time output
        HOPPdict, trackers = batt_balance(HOPPdict, ARIES_output_dict, trackers, simulate_SOC)

        if plotting:
            trackers = updateSOCplot(trackers, HOPPdict)

        # Send ARIES time back to HOPP
        bytesToSend = str.encode(json.dumps(str(aries_time)))
        sendHOPPsocket.sendto(bytesToSend, sendHOPPaddress)

        # Send command back to ARIES
        # bytesToSend = str.encode(json.dumps(HOPPdict))
        ARIES_input_dict = HOPPdict['commands']
        bytesToSend = aries_input_pack(ARIES_input_dict)
        if simulate_aries:
            sendARIESsocket.sendto(bytesToSend, sendARIESaddress)
        else:
            sendARIESsocket.send(bytesToSend)

        

if __name__ == '__main__':

    realtime_balancer(True, 1, False)