import random
import sys
import struct
import os
import socket
import time

# Setup UDP send to ARIES
remoteIP     = "10.81.15.41"
remotePort   = 9010
sendARIESaddress  = (remoteIP, remotePort)
sendARIESsocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
sendARIESsocket.connect(sendARIESaddress)

while True:
    
    testvals = [ ] # sending 26 ones
    for i in range(26):
        testvals.append(struct.pack('!f', 1.0+float(i)))
    bytes_to_send = b"".join(testvals)
    sendARIESsocket.send(bytes_to_send)
    time.sleep(1)