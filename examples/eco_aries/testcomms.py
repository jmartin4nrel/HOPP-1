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
    testvals = [ struct.pack('!f', float(random.randint(1,100))) for i in range(26)] # sending 26 random numbers
    # testvals = [ struct.pack('!f', float(1)) for i in range(26)] # sending all 1s 
    bytes_to_send = b"".join(testvals)
    sendARIESsocket.send(bytes_to_send)
    time.sleep(1)