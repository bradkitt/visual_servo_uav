#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import socket
import json
import time
import os
import rospy 
from geometry_msgs.msg import Point
from visual_servo_uav_msgs.msg import DroneCommand


class send_cmd_to_socket():
	FORMAT = "utf-8"
	DISCONNECT_MESSAGE = "!DISCONNECT"

	def __init__(self, ip_server, port):

		rospy.init_node('remote_ctrl_interface' )
		self.PORT = port
		self.SERVER = ip_server 
		self.ADDR = (self.SERVER,self.PORT)
		try:
			json_dir = os.path.dirname( __file__ )
			rel_path = "command.JSON"
			abs_file_path = os.path.join(json_dir,rel_path)
			abs_file_path="/home/federico/catkin_ws/src/visual_servo_uav/visual_servo_uav_pkg/scripts/command.JSON"
			rospy.loginfo( "path: ", abs_file_path )
			# path = os.path.dirname(os.path.abspath(__file__))+"/command.JSON"
			# print( path )

		except:
			print( " issues in finding the correct path for JSON file ... " )
           
		self.data = json.load( open(abs_file_path) )
		rospy.loginfo( " JSON file correctly uploaded " )

		self.sub = rospy.Subscriber("/cmd_vel", DroneCommand, self.send_cmd_to_app, queue_size=1)

	
	def send_cmd_to_app(self, cmd):
		self.data[ " yaw " ] =   cmd.yaw
		self.data[ " pitch " ] = cmd.pitch
		self.data[ " roll " ] =  cmd.roll
		self.data[ " throttle " ] = cmd.throttle

		msg = json.dumps( self.data )
		
		# client initialisation
		client = socket.socket( socket.AF_INET, socket.SOCK_STREAM ) 
		# server adress to be connected with 
		client.connect( self.ADDR )  
	   
		# binary conversion before sendinf 
		message = msg.encode( self.FORMAT )  
		# sending connection
		client.send( message )  
		rospy.loginfo( " JSON message sent ... " )

		# closing the client 
		client.close( )


if __name__ == "__main__":
	
	ip_server = input( " Enter IP of your mobile device:  " )
	port = 8080
	client = send_cmd_to_socket( ip_server , port )
	try:
		rospy.spin( )
	except KeyboardInterrupt:
		print ( " Shutting down ... " )
