import socket
import json
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from comms_tracking import setup_tracking, update_trackers, updateSOCplot
from aries_comms import aries_output_unpack, aries_input_pack
from hopp import ROOT_DIR

def batt_balance(HOPPdict, ARIESdict, trackers):

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

    # Calc new SOC
    if len(ariesSOC) == 0:
        initSOC = hoppSOC[0]
        sec_elapsed = float(np.diff(pd.DatetimeIndex(ARIESdict['aries_time'])))/1e9
    else:
        initSOC = aries_xdata['soc'][-1]
        sec_elapsed = float(np.diff(aries_time[[-3,-1]]))/1e9
    newSOC = (initSOC/100-sec_elapsed*batt_kw[0]/cap_kwh/3600)*100
    if len(ariesSOC) == 0:
        aries_xdata['soc'].extend([initSOC,newSOC])
    else:
        aries_xdata['soc'].append(newSOC)

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

def realtime_balancer(simulate_aries=True):

    bufferSize_HOPP  = 4096*2
    bufferSize_ARIES  = 4096*2
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
        # Setup UDP receive from ARIES
        localIP     = "RTDS MACHINE"
        localPort   = 9000
        serverAddressPort   = (localIP, localPort)
        recvARIESsocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        recvARIESsocket.bind(serverAddressPort)
        recvARIESsocket.settimeout(60)

        # Setup UDP send to ARIES
        localIP     = "RTDS MACHINE"
        localPort   = 9001
        sendARIESaddress  = (localIP, localPort)
        sendARIESsocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

    # Setup UDP send to HOPP
    localIP     = "127.0.0.1"
    localPort   = 20004
    sendHOPPaddress  = (localIP, localPort)
    sendHOPPsocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

    # Read in ARIES placeholder signal
    aries_sig_fn = ROOT_DIR.parent / 'examples' / 'outputs' / 'placeholder_ARIES.csv'
    aries_signals = pd.read_csv(aries_sig_fn,parse_dates=True,index_col=0,infer_datetime_format=True)

    # TODO: Read in wave generation signals
    # wave_gen_fn = ???
    # wave_gen_signals = pd.read_csv()

    # Set up trackers and plots if necessary
    if plotting:
        plt.ion()
    trackers = setup_tracking(plotting)

    while(True):

        # Receive data from ARIES
        ARIESpair = recvARIESsocket.recvfrom(bufferSize_ARIES)
        ARIESraw = ARIESpair[0]
        # ARIESdict = aries_output_unpack(ARIESraw)
        ARIESdict = json.loads(ARIESraw)

        # Receive data from HOPP
        HOPPpair = recvHOPPsocket.recvfrom(bufferSize_HOPP)
        HOPPraw = HOPPpair[0]
        HOPPdict = json.loads(HOPPraw)

        # Replace insolation and wind speeds with real-time wind speeds
        aries_time = trackers[1]
        if len(aries_time) == 0:
            aries_time = [aries_signals.index.values[0]]
        insol = aries_signals.loc[aries_time[-1],'poa']
        HOPPdict['commands']['pv_insol'] = insol
        for turb_num in range(len(HOPPdict['gen']['wind'])):
            wind_spd = aries_signals.loc[aries_time[-1],'wind_vel_{:02}'.format(turb_num)]
            HOPPdict['commands']['wind_spd_'+str(turb_num+1)] = wind_spd

        #TODO: Read in real-time wave power profiles from WEC_sim outputs (from Naveen?)
        # for wec_num in range(len(HOPPdict['gen']['wave'])):
        # if len(aries_time) == 0:
        #     aries_time = [aries_signals.index.values[0]]
        # for turb_num in range(len(HOPPdict['gen']['wind'])):
        #     wave_gen = wave_gen_signals.loc[aries_time[-1],'???']
        #     HOPPdict['commands']['wave_gen_'+str(turb_num+1)] = wind_spd

        trackers = update_trackers(trackers, HOPPdict, ARIESdict, plotting)

        # Balance battery output from real-time output
        HOPPdict, trackers = batt_balance(HOPPdict, ARIESdict, trackers)

        if plotting:
            trackers = updateSOCplot(trackers, HOPPdict)

        # Send ARIES time back to HOPP
        bytesToSend = str.encode(json.dumps(str(ARIESdict['aries_time'][-1])))
        sendHOPPsocket.sendto(bytesToSend, sendHOPPaddress)

        # Send command back to ARIES
        # bytesToSend = str.encode(json.dumps(HOPPdict))
        bytesToSend = aries_input_pack(HOPPdict)
        sendARIESsocket.sendto(bytesToSend, sendARIESaddress)

        

if __name__ == '__main__':

    realtime_balancer()