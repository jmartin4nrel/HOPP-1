import socket
import time

# Setup UDP receive from ARIES
localIP     = "127.0.0.1"
localPort   = 20001
remotePort  = 20001
bufferSize  = 1024
serverAddressPort   = (localIP, localPort)
ARIESrcvSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

while(True):
    bytesAddressPair = ARIESrcvSocket.recvfrom(bufferSize)
    message = bytesAddressPair[0]
    address = bytesAddressPair[1]

    clientMsg = "Message from Client:{}".format(message)
    clientIP  = "Client IP Address:{}".format(address)

    print(clientMsg)