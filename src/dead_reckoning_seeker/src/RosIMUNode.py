#!/usr/bin/env python

import rospy
import serial
import struct
import math
from sensor_msgs.msg import Imu
from std_msgs.msg import Header

class ICM20948Node:
    def __init__(self):
        rospy.init_node('icm20948_node')
        
        # Get parameters
        self.port = rospy.get_param('~port', '/dev/ttyACM0')
        self.baud = rospy.get_param('~baud', 115200)
        self.frame_id = rospy.get_param('~frame_id', 'imu_link')
        
        # Setup publisher
        self.imu_pub = rospy.Publisher('imu/data_raw', Imu, queue_size=10)
        
        # Setup serial connection
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=1.0
            )
        except serial.SerialException as e:
            rospy.logerr(f"Failed to open serial port {self.port}: {str(e)}")
            raise
        
        # Packet format: 13 floats (3 accel, 3 gyro, 3 mag, 4 quat) + 1 uint32
        self.packet_format = '13f1I'
        self.packet_size = struct.calcsize(self.packet_format)

        # ICM-20948 noise characteristics from datasheet
        # Accelerometer noise = 230 μg/√Hz (page 12)
        self.accel_noise_density = 230.0 * 1e-6 * 9.81  # Convert μg to m/s^2
        
        # Gyroscope noise = 0.015 dps/√Hz (page 11)
        self.gyro_noise_density = 0.015 * math.pi / 180.0  # Convert dps to rad/s
        
        # Magnetometer resolution = 0.15 μT (page 13)
        self.mag_resolution = 0.15  # μT
        
    def read_packet(self):
        """Read a packet from the serial port with start/end markers"""
        # Wait for start marker
        while not rospy.is_shutdown():
            if self.serial.read() == b'\xff' and self.serial.read() == b'\xaa':
                break
        
        # Read packet data
        data = self.serial.read(self.packet_size)
        
        # Wait for end marker
        end_marker = self.serial.read(2)
        if end_marker != b'\x55\xff':
            rospy.logwarn("Invalid end marker")
            return None
            
        try:
            # Unpack the data
            values = struct.unpack(self.packet_format, data)
            return values
        except struct.error as e:
            rospy.logwarn(f"Failed to unpack packet: {str(e)}")
            return None
            
    def create_imu_msg(self, data):
        """Create ROS IMU message from packet data"""
        if data is None:
            return None
            
        msg = Imu()
        
        # Set header
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.frame_id
        
        # Set linear acceleration (first 3 values)
        msg.linear_acceleration.x = data[0]
        msg.linear_acceleration.y = data[1]
        msg.linear_acceleration.z = data[2]
        
        # Set angular velocity (next 3 values)
        msg.angular_velocity.x = data[3]
        msg.angular_velocity.y = data[4]
        msg.angular_velocity.z = data[5]
        
        # Set orientation (quaternion values start at index 9)
        # Note: Index shifted by -1 due to removal of temperature
        msg.orientation.w = data[8]
        msg.orientation.x = data[9]
        msg.orientation.y = data[10]
        msg.orientation.z = data[11]
        
        # Calculate covariances based on magnetometer resolution
        orientation_variance = math.pow(math.atan2(self.mag_resolution, 50.0), 2)
        msg.orientation_covariance = [
            orientation_variance, 0.0, 0.0,
            0.0, orientation_variance, 0.0,
            0.0, 0.0, orientation_variance
        ]
        
        # Calculate angular velocity: based on gyroscope noise density
        gyro_variance = math.pow(self.gyro_noise_density, 2) * 50
        msg.angular_velocity_covariance = [
            gyro_variance, 0.0, 0.0,
            0.0, gyro_variance, 0.0,
            0.0, 0.0, gyro_variance
        ]
        
        # Calculate linear acceleration: based on accelerometer noise density
        accel_variance = math.pow(self.accel_noise_density, 2) * 50
        msg.linear_acceleration_covariance = [
            accel_variance, 0.0, 0.0,
            0.0, accel_variance, 0.0,
            0.0, 0.0, accel_variance
        ]
        
        return msg
        
    def run(self):
        """Main node loop"""
        rospy.loginfo("ICM-20948 node started")
        
        while not rospy.is_shutdown():
            try:
                data = self.read_packet()
                if data:
                    msg = self.create_imu_msg(data)
                    if msg:
                        self.imu_pub.publish(msg)
            except serial.SerialException as e:
                rospy.logerr(f"Serial error: {str(e)}")
                break
            except Exception as e:
                rospy.logerr(f"Unexpected error: {str(e)}")
                continue
                
            rospy.sleep(0.001)
        
        # Cleanup
        self.serial.close()

if __name__ == '__main__':
    try:
        node = ICM20948Node()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Failed to start node: {str(e)}")