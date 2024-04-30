import socket
import json
import struct
import numpy as np
import pandas as pd
from pathlib import Path
from random import random


def aries_input_pack(HOPPdict):

    # Packs dict being sent from balancer to ARIES into a labeled dict
    # Keys match the "From UDP" variable names in IO_variables

    from_UDP_HOPP = ['bess_kw','poa','wind_vel_00','wind_vel_01',
                 'wind_vel_02','wind_vel_03','wind_vel_04','wind_vel_05',
                 'wind_vel_06','wind_vel_07','wind_vel_08','wind_vel_09',
                 'wind_vel_10','wind_vel_11','wind_vel_12','wind_vel_13',
                 'wind_vel_14','wind_vel_15','wind_vel_16','wind_vel_17',
                 'wind_vel_18','wind_vel_19','wind_vel_20','wind_vel_21',
                 'wind_vel_22','wind_vel_23','peripheral_load','elyzer_kw']

    strs_list = []

    for key in from_UDP_HOPP:
        strs_list.append((struct.pack('!f', HOPPdict[key])))
    raw_input = b"".join(strs_list)

    return raw_input

def aries_input_unpack(raw_input):

    # Unpacks raw bytes being sent from balancer to ARIES into a labeled dict
    # Keys match the "From UDP" variable names in IO_variables

    from_UDP_ARIES = ['PORD','INSOL','WindSPD',
                    'WindSPD1','WindSPD2','WindSPD3','WindSPD4',
                    'WindSPD5','WindSPD6','WindSPD7','WindSPD8',
                    'WindSPD9','WindSPD10','WindSPD11','WindSPD12',
                    'WindSPD13','WindSPD14','WindSPD15','WindSPD16',
                    'WindSPD17','WindSPD18','WindSPD19','WindSPD20',
                    'WindSPD21','WindSPD22','WindSPD23','LoadPeriph','Electrolyzer_Ps']

    num_bytes = len(raw_input)

    dl_vals = []
    idx = 0
    while idx < num_bytes:
        data = struct.unpack(
            '!f', raw_input[idx:idx+4])[0]
        idx = idx + 4
        dl_vals.append(data)
    
    ARIES_dict = {}
    for i in range(0, len(dl_vals)):
        ARIES_dict[from_UDP_ARIES[i]] = dl_vals[i]

    return ARIES_dict


def aries_output_pack(ARIESdict):

    # Packs dict being sent from ARIES to balancer into raw bytes
    # Keys match the "To UDP" variable names in IO_variables

    to_UDP_ARIES = ['PML6','BESS_SOC','Ps_PV',
                    'Electrolyzer_Ps','Electrolyzer_kg_s','WF1PGfilt',
                    'WF1PGfilt1','WF1PGfilt2','WF1PGfilt3','WF1PGfilt4',
                    'WF1PGfilt5','WF1PGfilt6','WF1PGfilt7','WF1PGfilt8',
                    'WF1PGfilt9','WF1PGfilt10','WF1PGfilt11','WF1PGfilt12',
                    'WF1PGfilt13','WF1PGfilt14','WF1PGfilt15','WF1PGfilt16',
                    'WF1PGfilt17','WF1PGfilt18','WF1PGfilt19','WF1PGfilt20',
                    'WF1PGfilt21','WF1PGfilt22','WF1PGfilt23','WAVE1Pcon1Filt',
                    'WAVE1Pcon1Filt1','WAVE1Pcon1Filt2','WAVE1Pcon1Filt3',
                    'WAVE1Pcon1Filt4','WAVE1Pcon1Filt5','WAVE1Pcon1Filt6',
                    'WAVE1Pcon1Filt7','WAVE1Pcon1Filt8','WAVE1Pcon1Filt9','DUMMY']

    strs_list = []

    for key in to_UDP_ARIES:
        strs_list.append((struct.pack('!f', ARIESdict[key])))
    ARIESraw = b"".join(strs_list)

    return ARIESraw
 

def aries_output_unpack(raw_output):

    # Unpacks raw bytes being sent from ARIES to balancer into dict
    # Keys match the "To UDP" variable names in IO_variables

    to_UDP_HOPP = ['batt','soc','solar',
                    'elyzer','elyzer_kg_s','wind0',
                    'wind1','wind2','wind3','wind4',
                    'wind5','wind6','wind7','wind8',
                    'wind9','wind10','wind11','wind12',
                    'wind13','wind14','wind15','wind16',
                    'wind17','wind18','wind19','wind20',
                    'wind21','wind22','wind23','wave0',
                    'wave1','wave2','wave3',
                    'wave4','wave5','wave6',
                    'wave7','wave8','wave9','DUMMY']

    num_bytes = len(raw_output)

    dl_vals = []
    idx = 0
    while idx < num_bytes:
        data = struct.unpack(
            '!f', raw_output[idx:idx+4])[0]
        idx = idx + 4
        dl_vals.append(data)
    
    HOPP_dict = {}
    for i in range(0, len(dl_vals)):
        HOPP_dict[to_UDP_HOPP[i]] = dl_vals[i]

    return HOPP_dict


def aries_comms(num_inputs=28, initial_SOC=50.0, acceleration=1):

    bufferSize  = num_inputs*4

    # Read in ARIES placeholder signal
    aries_sig_fn = Path(__file__).parent / 'outputs' / 'placeholder_ARIES_no_wind_100ms.csv'
    aries_signals = pd.read_csv(aries_sig_fn,parse_dates=True,index_col=0,infer_datetime_format=True)

    # Setup UDP send
    localIP     = "127.0.0.1"
    localPort   = 20002
    sendAddress   = (localIP, localPort)
    sendSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

    # Setup UDP receive
    localIP     = "127.0.0.1"
    localPort   = 20003
    serverAddressPort   = (localIP, localPort)
    recvSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    recvSocket.bind(serverAddressPort)
    recvSocket.settimeout(60)

    # Set the time index to the start of the ARIES signal
    start_time = aries_signals.index.values[0]
    end_time = aries_signals.index.values[-1]
    timer_start = pd.Timestamp.now()
    time_index = start_time

    # Send out "ARIES-generated" data
    started = False
    while((end_time-time_index)>pd.Timedelta(0)):

        # Measure elapsed time and find correct place in ARIES signal
        elpased_time = (pd.Timestamp.now()-timer_start)*acceleration
        aries_time = start_time+elpased_time
        if not started:
            new_time_index = start_time
        else:
            new_time_index = aries_time.floor('100ms')

        # If at least two 1 s cycle have passed, send out the signals
        if (new_time_index > time_index + pd.Timedelta('200ms')) or not started:
            row = aries_signals.loc[new_time_index]

            # Use battery command to calculate electrolyzer output
            sum = 0.
            for col in row.index.values:
                if col == 'batt':
                    if not started:
                        row[col] = 0
                    else:
                        row[col] = HOPPdict['PORD']
                else:
                    if not started:
                        row[col] = aries_signals.loc[time_index,col]
                    else:
                        row[col] = np.mean(aries_signals.loc[time_index:(new_time_index-pd.Timedelta('100ms')),col])
                if col != 'elyzer':
                    sum += row[col]
                else:
                    row[col] = sum

            # Add variables to ARIES dict
            ARIESdict = {}
            ARIESdict['PML6'] = row['batt']
            ARIESdict['BESS_SOC'] = initial_SOC+random()*10-5 # Randomly perturb the SOC so we will see it change when running in simulated ARIES mode
            ARIESdict['Ps_PV'] = row['solar']
            ARIESdict['Electrolyzer_Ps'] = row['elyzer']
            ARIESdict['Electrolyzer_kg_s'] = row['elyzer']/54.66/60/60
            ARIESdict['WF1PGfilt'] = row['wind']/24
            for turb_num in range(1,24):
                ARIESdict['WF1PGfilt'+str(turb_num)] = row['wind']/24
            ARIESdict['WAVE1Pcon1Filt'] = row['wave']/10
            for turb_num in range(1,10):
                ARIESdict['WAVE1Pcon1Filt'+str(turb_num)] = row['wave']/10
            ARIESdict['DUMMY'] = 0

            # Pack and send ARIES dict over UDP
            # bytesToSend = str.encode(json.dumps(ARIESdict))
            bytesToSend = aries_output_pack(ARIESdict)
            sendSocket.sendto(bytesToSend, sendAddress)
            time_index = new_time_index
            if not started:
                print("Ready to start HOPP")
            
            # Wait to receive command from balancer
            HOPPrecv = recvSocket.recvfrom(bufferSize)
            HOPPdict = aries_input_unpack(HOPPrecv[0])

            started = True


if __name__ == '__main__':

    aries_comms()