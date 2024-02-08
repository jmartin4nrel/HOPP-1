import socket
import numpy as np
import pandas as pd
from hopp import ROOT_DIR

# Read in ARIES placeholder signal
aries_sig_fn = ROOT_DIR.parent / 'examples' / 'outputs' / 'placeholder_ARIES.csv'
aries_signals = pd.read_csv(aries_sig_fn,parse_dates=True,index_col=0,infer_datetime_format=True)

# Setup UDP send
localIP     = "127.0.0.1"
localPort   = 20001
serverAddressPort   = (localIP, localPort)
SendSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

# Set the time index to the start of the ARIES signal
start_time = aries_signals.index.values[0]
end_time = aries_signals.index.values[-1]
timer_start = pd.Timestamp.now()
time_index = start_time

# Listen for incoming datagrams
while((end_time-time_index)>pd.Timedelta(0)):

    # Measure elapsed time and find correct place in ARIES signal
    elpased_time = (pd.Timestamp.now()-timer_start)*4
    aries_time = start_time+elpased_time
    new_time_index = aries_time.floor('100ms')

    # If at least one 100 ms cycle has passed, send out the signals
    if new_time_index != time_index:
        time_index = new_time_index
        msgFromServer = str(aries_signals.loc[time_index])
        bytesToSend = str.encode(msgFromServer)
        SendSocket.sendto(bytesToSend, serverAddressPort)