#!/usr/bin/env python

# Python libs
import sys, time
import math

# numpy and scipy
import numpy as np
from scipy.ndimage import filters

import imutils
import os 

# OpenCV
import cv2

# Ros libraries
import roslib
import rospy

# Ros Messages
from sensor_msgs.msg import CompressedImage
from main.msg import data

VERBOSE=False

class image_feature:

	def __init__(self):
		'''Initialize ros publisher, ros subscriber'''
		# topics where we publish
		self.image_pub = rospy.Publisher("/output/ErrorFromBbox",
			CompressedImage, queue_size=1) # TODO check 

		self.data_pub = rospy.Publisher("/DistAndAngle", data,
			queue_size=1)

		# subscribed Topic
		self.subscriber = rospy.Subscriber("/output/image_raw/compressed",
			CompressedImage, self.callback,  queue_size = 1)
		if VERBOSE :
			print("subscribed to /raspicam_node/image/compressed")


	def callback(self, ros_data):
		'''Callback function of subscribed topic. 
		Here images get converted and features detected'''
		if VERBOSE :
			print('received image of type: "%s"' % ros_data.format)

		#### direct conversion to CV2 ####
		np_arr = np.fromstring(ros_data.data, np.uint8)
		image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenV >= 3.0:


	#color level of the ball
	redLower = (160, 150, 15)
	redUpper = (180, 255, 255)
	
	#Focal length in pixels
	focalLength = 360		 #for 480x270 resolution
	#focalLength=978 	 #for 1280x960 resolution

	#radius of the ball in centimeters
	knownWidth = 10 

	#center of the image in pixels
	C = 480/2 
	Cy = 270/2
	#Cx=1280/2 
	#Cy=960/2

	blurred = cv2.GaussianBlur(image_np, (11, 11), 0)
	hsv =  cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV) #convert from rgb to hsv
	mask = cv2.inRange(hsv, redLower, redUpper) #create a mask of redlower/redupper range
	mask = cv2.erode(mask, None, iterations=2)  #clean the mask with erode and dilatate
	mask = cv2.dilate(mask, None, iterations=2)
	#cv2.imshow('mask', mask)
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, #mask.copy create a copy of the element mask (if we use a=b instead,each time we change a, also b changes)
									# RETR_EXTERNAL takes the most external contour in the contour hierarchy
									# CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments and leaves only their end points
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts) #take the contours computed before
	center = None #initialize the center of the ball

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea) #find the maximum contour
		((x, y), radius) = cv2.minEnclosingCircle(c) #compute the minimum enclosing circle of the ball
		M = cv2.moments(c) #compute the center of that circle
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) 
 
		distance= (knownWidth * focalLength) / radius		# compute the distance of the ball in centimeters

		#compute the center of the detected ball in meters
		centerxM=(center[0] * distance) / focalLength;
		centeryM=(center[1] * distance) / focalLength;

		#compute the center of the image plane in meters (to put it in the plane where the ball is positioned)
		CxM= (Cx * distance) / focalLength; 
		CyM= (Cy * distance) / focalLength;

		#compute the distance between the center of the image plane in meters and the center of the ball
		DistFromCenter=math.sqrt( (centerxM-CxM)**2 + (centeryM-CyM)**2 )  

		#take only the horizontal component
		Distx=abs(centerxM-CxM);

		#find the distance and the angle between the center of the ball and the center camera
		Hyp=math.sqrt((DistFromCenter)**2+distance**2);
		angle=np.arcsin(DistFromCenter/Hyp);

				#angle considering only horizontal component
		anglex=np.arcsin(Distx/Hyp);
		if center[0] > Cx:
			anglex=-anglex
			angle_grades=180*anglex/(math.pi) #compute the angle in grades

		#put all the distance and angle in "value", an object of type data
		value = data()
		if(distance>=180 or distance <=15 or angle_grades>=26 or angle_grades<=-26): #ball is for sure out of the arena or too near the robot or too near to the edge of the scene
			value.distance = float('nan')
			value.angle = float('nan')
		else:
			value.distance = distance
			value.angle = anglex
				
		print("The distance is:", value.distance)
		print("The angle is:", angle_grades)
		self.data_pub.publish(value)

#The moment is a particular weighted average of image pixel intensities, with the help of which we can find some specific properties of an image, like radius, area, centroid
#The center can be found using the above formula
 
		# only proceed if the radius meets a minimum size of 10 pixels
		if radius > 10:  
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(image_np, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			cv2.circle(image_np, center, 5, (0, 0, 255), -1)

	else: #I didn't find a ball
		value = data()
		value.distance = float('nan')
		value.angle = float('nan')
		print("The distance is:", value.distance)
		print("The angle is:", value.angle)
 		self.data_pub.publish(value)


		cv2.imshow('window', image_np) #show the image in a window
		cv2.waitKey(2)  #displays the image for specified milliseconds
 
		#self.subscriber.unregister()


def get_images(in_path):
    img_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))

    return img_files

def main(args):
	'''Initializes and cleanup ros node'''
	
	ic = image_feature()
	rospy.init_node('error_from_center', anonymous=True)
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down ROS Image feature detector module...")
	
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)
