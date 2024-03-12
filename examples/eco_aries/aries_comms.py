import socket
import json
import struct
import numpy as np
import pandas as pd
from hopp import ROOT_DIR


def aries_input_pack(HOPPdict):

    comm_list = list(HOPPdict["commands"].values())
    
    strs_list = []
    idx = -1
    for x in comm_list:
        idx += 1
        strs_list.append((struct.pack('!d', x)))
    raw_input = b"".join(strs_list)

    return raw_input

def aries_input_unpack(raw_input):

    num_inputs = 37
    num_bytes = num_inputs*8

    dl_vals = []
    idx = 0
    while idx < num_bytes:
        data = struct.unpack(
            '!d', raw_input[idx:idx+8])[0]
        idx = idx + 8
        dl_vals.append(data)

    dl_names = ['bess_kw', 'pv_insol']
    for i in range(1, 25):
        dl_names.append('wind_spd_' + str(i))
    for i in range(1, 11):
        dl_names.append('wave_prod_' + str(i))
    dl_names.append('peripheral_load')
    
    HOPP_dict = {}
    for i in range(0, len(dl_vals)):
        HOPP_dict[dl_names[i]] = dl_vals[i]

    return HOPP_dict


def aries_output_pack(ARIESdict):

    measurements = list([val[0] for val in ARIESdict.values()])
    
    strs_list = []

    s = bytes(measurements[0], 'utf-8')
    strs_list.append(struct.pack("I%ds" % (len(s),), len(s), s))

    idx = -1
    for x in range(1,len(measurements)):
        idx += 1
        strs_list.append((struct.pack('!d', measurements[x])))
    ARIESraw = b"".join(strs_list)

    return ARIESraw
 

def aries_output_unpack(raw_output):

    date_bytes = 33
    num_inputs = 39
    num_bytes = num_inputs*8 + date_bytes

    dl_vals = []
    idx = 0
    (i,), data = struct.unpack("I", raw_output[:4]), raw_output[4:]
    s, data = data[:i], data[i:]
    idx = idx + date_bytes
    dl_vals.append(str(s.decode('UTF-8')))

    while idx < num_bytes:
        data = struct.unpack(
            '!d', raw_output[idx:idx+8])[0]
        idx = idx + 8
        dl_vals.append(data)

    dl_names = ['aries_time', 'pv_insol','bess_kw','elyzer','poa']
    for i in range(1, 25):
        dl_names.append('wind_spd_' + str(i))
    dl_names.append('batt_soc')
    for i in range(1, 11):
        dl_names.append('wave_prod_' + str(i))
    
    ARIES_dict = {}
    for i in range(0, len(dl_vals)):
        ARIES_dict[dl_names[i]] = dl_vals[i]

    return ARIES_dict


def aries_comms():

    bufferSize  = 4096*2

    # Set up faster-than-realtime
    times_realtime = 60

    # Read in ARIES placeholder signal
    aries_sig_fn = ROOT_DIR.parent / 'examples' / 'outputs' / 'placeholder_ARIES.csv'
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
    while((end_time-time_index)>pd.Timedelta(0)):

        # Measure elapsed time and find correct place in ARIES signal
        elpased_time = (pd.Timestamp.now()-timer_start)*times_realtime
        aries_time = start_time+elpased_time
        new_time_index = aries_time.floor('1s')

        # If at least two 1 s cycle have passed, send out the signals
        if new_time_index > time_index + pd.Timedelta('2s'):
            rows = aries_signals.loc[time_index:new_time_index]
            rows = rows.iloc[[0,-2]]

            # Send data to balancer, using battery command to calculate electrolyzer output
            ARIESdict = {'aries_time':[str(i) for i in rows.index.values]}
            sum = 0.
            for col in rows.columns.values:
                rows[col] = np.mean(aries_signals.loc[time_index:(new_time_index-pd.Timedelta('1s')),col])
                if col != 'elyzer':
                    sum += rows[col]
                else:
                    rows[col] = sum
                ARIESdict[col] = list(rows[col].values)
            
            #Removing and Adding extra columns from and to ARIESdict to match number of bytes from ARIES
            #May not match actual measurement keys or order recieved from ARIES
            #Sends 1 date item and 39 measurements
            del ARIESdict['wind']
            del ARIESdict['wind_vel_max']
            del ARIESdict['wind_vel_min']
            ARIESdict['batt_soc'] = ARIESdict['batt']
            for i in range(1, 11):
                ARIESdict[str('wave_prod_' + str(i))] = ARIESdict['wave']
            del ARIESdict['wave']

            # bytesToSend = str.encode(json.dumps(ARIESdict))
            bytesToSend = aries_output_pack(ARIESdict)
            sendSocket.sendto(bytesToSend, sendAddress)
            time_index = new_time_index
            
            # Wait to receive command from balancer
            HOPPrecv = recvSocket.recvfrom(bufferSize)
            HOPPdict = aries_input_unpack(HOPPrecv[0])
            HOPPcommand = HOPPdict['bess_kw']
            aries_signals.loc[(new_time_index+pd.Timedelta('1s')):,'batt'] = HOPPcommand


if __name__ == '__main__':

    aries_comms()