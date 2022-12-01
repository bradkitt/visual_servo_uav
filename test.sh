#!/bin/bash

xterm -e bash -c "roslaunch visual_servo_uav_pkg launch_image_view.launch ; exec bash" &
sleep 2
xterm -e bash -c "rosrun visual_servo_uav_pkg img_server.py ; exec bash" &
xterm -e bash -c "rosrun visual_servo_uav_pkg remote_ctrl_interface.py ; exec bash" &

echo "script finished"
