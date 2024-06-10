import socket

hostname = socket.gethostname()

IPSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
IPSocket.connect(('10.0.0.0', 0))
IPSocketName = IPSocket.getsockname()[0]

print(hostname + "'s IP Address is: " + str(IPSocketName))
