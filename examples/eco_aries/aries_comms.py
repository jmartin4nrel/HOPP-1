import socket
import json
import numpy as np
import pandas as pd
from hopp import ROOT_DIR

if __name__ == '__main__':

    bufferSize  = 4096

    # Set up faster-than-realtime
    times_realtime = 100

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
        new_time_index = aries_time.floor('100ms')

        # If at least two 100 ms cycle have passed, send out the signals
        if new_time_index > time_index + pd.Timedelta('200ms'):
            rows = aries_signals.loc[time_index:new_time_index]
            rows = rows.iloc[[0,-2]]

            # Send data to balancer, using battery command to calculate electrolyzer output
            response_dict = {'aries_time':[str(i) for i in rows.index.values]}
            sum = 0.
            for col in rows.columns.values:
                rows[col] = np.mean(aries_signals.loc[time_index:(new_time_index-pd.Timedelta('100ms')),col])
                if col != 'elyzer':
                    sum += rows[col]
                else:
                    rows[col] = sum
                response_dict[col] = list(rows[col].values) 
            bytesToSend = str.encode(json.dumps(response_dict))
            sendSocket.sendto(bytesToSend, sendAddress)
            time_index = new_time_index
            
            # Wait to receive command from balancer
            pair = recvSocket.recvfrom(bufferSize)
            HOPPdict = json.loads(pair[0])
            HOPPcommand = HOPPdict['batt_command_kw']
            aries_signals.loc[(new_time_index+pd.Timedelta('100ms')):,'batt'] = HOPPcommand