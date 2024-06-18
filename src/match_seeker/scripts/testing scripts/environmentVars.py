import os
import subprocess
import tempfile
import socket

# Getting the computer's hostname and displaying it in the shell
hostname = socket.gethostname()
IPSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
IPSocket.connect(('10.0.0.0', 0))
IPSocketName = IPSocket.getsockname()[0]
print(hostname + "'s IP Address is: " + str(IPSocketName))

# Gets the name of the desired robot
robotName = raw_input("What robot are we working with? (Speedy2 or Cutie2) ")

# Input accepts any of the reasonable input as stated below
if robotName in ["Speedy2", "S", "s", "s2", "SP", "SP2", "sp", "sp2"]:
    new_uri = "http://141.140.243.85:11311"
elif robotName in ["Cutie2", "C", "c", "ct", "CT", "c2", "CT2"]:
    new_uri = "http://141.140.243.153:11311"
else:
    print("Invalid robot name. Please try again.")


# Adds the hostname and master uri to the .bashrc file
def set_variables(uri, socketname):
    bashrc_path = os.path.expanduser("~/.bashrc")
    with open(bashrc_path, "a") as bashrc:
        bashrc.write("\nexport ROS_MASTER_URI={}\n".format(uri))
        bashrc.write("\nexport ROS_HOSTNAME={}\n".format(socketname))

    print "ROS_MASTER_URI set to {}".format(uri)
    print "ROS_HOSTNAME set to {}".format(socketname)

    # Creates temporary shell script to source the .bashrc file
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.sh') as temp_script:
        temp_script.write("/bin/bash\n")
        temp_script.write("source {}\n".format(bashrc_path))
        temp_script.write("echo Current ROS_MASTER_URI: $ROS_MASTER_URI\n")
        temp_script.write("echo Current ROS_HOSTNAME: $ROS_HOSTNAME\n")
        temp_script_path = temp_script.name

    # Make temporary script executable, executes the temporary shell, and cleans it up
    subprocess.call(['chmod', '+x', temp_script_path])
    subprocess.call(['bash', '-c', temp_script_path])
    os.remove(temp_script_path)


set_variables(new_uri, IPSocketName)
