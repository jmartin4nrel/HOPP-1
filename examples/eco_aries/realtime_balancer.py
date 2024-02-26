import socket
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from comms_tracking import setup_tracking, update_trackers, updateSOCplot

def batt_balance(HOPPdict, ARIESdict, trackers):

    aries_time = trackers[1]
    aries_xdata = trackers[2]

    # Get commanded electrolyzer output from HOPP
    comm_dict = HOPPdict['commands']
    elyzer_kw = comm_dict['elyzer_kw']

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

    HOPPdict['batt_command_kw'] = new_batt_kw
    
    return HOPPdict, trackers

def realtime_balancer():

    bufferSize  = 4096
    plotting = True

    # Setup UDP receive from HOPP
    localIP     = "127.0.0.1"
    localPort   = 20001
    serverAddressPort   = (localIP, localPort)
    recvHOPPsocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    recvHOPPsocket.bind(serverAddressPort)
    recvHOPPsocket.settimeout(60)

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

    # Setup UDP send to HOPP
    localIP     = "127.0.0.1"
    localPort   = 20004
    sendHOPPaddress  = (localIP, localPort)
    sendHOPPsocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

    # Set up trackers and plots if necessary
    if plotting:
        plt.ion()
    trackers = setup_tracking(plotting)

    while(True):

        # Receive data from HOPP
        HOPPpair = recvHOPPsocket.recvfrom(bufferSize)
        HOPPraw = HOPPpair[0]
        HOPPdict = json.loads(HOPPraw)
        comm_dict = HOPPdict['commands']

        # Receive data from ARIES
        ARIESpair = recvARIESsocket.recvfrom(bufferSize)
        ARIESraw = ARIESpair[0]
        ARIESdict = json.loads(ARIESraw)

        trackers = update_trackers(trackers, HOPPdict, ARIESdict, plotting)

        # Balance battery output from real-time output
        HOPPdict, trackers = batt_balance(HOPPdict, ARIESdict, trackers)

        if plotting:
            trackers = updateSOCplot(trackers, HOPPdict)

        # Send command back to ARIES
        bytesToSend = str.encode(json.dumps(HOPPdict))
        sendARIESsocket.sendto(bytesToSend, sendARIESaddress)

        # Send ARIES time back to HOPP
        bytesToSend = str.encode(json.dumps(str(ARIESdict['aries_time'][-1])))
        sendHOPPsocket.sendto(bytesToSend, sendHOPPaddress)


if __name__ == '__main__':

    realtime_balancer()