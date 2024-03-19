import socket
import struct
import time

# Setup UDP send to ARIES
remoteIP     = "10.81.15.41"
remotePort   = 9010
sendARIESaddress  = (remoteIP, remotePort)
sendARIESsocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
sendARIESsocket.connect(sendARIESaddress)

while True:
    
    bytes_to_send = b"".join([struct.pack('!f',1)])
    sendARIESsocket.send(bytes_to_send)
    time.sleep(1)