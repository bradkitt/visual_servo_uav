
import shutil
# from visual_servo_uav_msgs.msg import DroneCommand

import rospy 
from sensor_msgs.msg import Image as  CompressedImage
import time
import os
import cv2
from pathlib import Path
from typing import Iterable
from PIL import Image, ImageDraw
from cv_bridge import CvBridge
import numpy as np





# global Var
outSavedPath = "/home/federico/catkin_ws/src/visual_servo_uav/detections"
# global image format for CV
cvImage = None
# counter: for debugging 
counter = 0 


def frame_info(fr):
	"""print information about frames
	"""
	print( " Image type:\n ", type(fr) ,
		 " ", " \nImage shape:\n ", fr.shape ,
		 " ", " \nImage size(px):\n ", fr.size, 
		 " ", " \nImage dtype:\n ", fr.dtype
		 )



def frame_cbk(data):
	global cvImage
	global outSavedPath
	global counter
	cvObject = CvBridge( )
	cvImage = cvObject.imgmsg_to_cv2( data , 'bgr8' )
	rospy.loginfo(' cbk: cvImage object created! ')
	# change directory 
	os.chdir(outSavedPath)
	frame_info(cvImage)
	img_ret = cv2.imwrite(f'/home/federico/catkin_ws/src/visual_servo_uav/detections/image_{counter}.jpg',cvImage)
	if img_ret:
		rospy.loginfo(f' Successfully saved image_{counter}.jpg')
		counter += 1
		# print an image every two seconds
		time.sleep(2)	
	else:
		rospy.loginfo(' Error while saving ...')

	

if __name__ == '__main__':
	rospy.init_node("frame_saver")
	rospy.loginfo(" Frame saver ready to log! ")
	frame_sub = rospy.Subscriber("/output/image_raw", CompressedImage, frame_cbk, queue_size=1 )
	# fr = rospy.Rate(15)
	rospy.spin()



