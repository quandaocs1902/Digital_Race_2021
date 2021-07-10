#!/usr/bin/env python
# license removed for brevity
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32

def Control():
    speed = rospy.Publisher('team1/set_speed', Float32, queue_size=10)
    angle = rospy.Publisher('team1/set_angle', Float32, queue_size=10)
    rospy.init_node('control', anonymous=True)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        v = 50
        theta = 0
        #rospy.loginfo(v)
        #rospy.loginfo(theta)
        speed.publish(v)
        angle.publish(theta)
        rate.sleep()

if __name__ == '__main__':
    try:
        Control()
    except rospy.ROSInterruptException:
        pass

# <node name="control" pkg="python_pkg" type ="control.py" output="screen" />
