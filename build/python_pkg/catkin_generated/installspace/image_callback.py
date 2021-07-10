# %%
#!/usr/bin/env python
# license removed for brevity
import rospy
import cv2
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32
import numpy as np
import sys
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
from numpy.lib.polynomial import polyfit
from sklearn.metrics import mean_squared_error
from python_pkg import lane_detect_module

def get_depth(depth_img):
    points_1 = [50, 90, 110, 150, 170, 210, 230, 270]

    a_1 = []
    depth_1 = []
    org_1 = []
    fontScale = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(4):
        depth_img = cv2.rectangle((depth_img), (points_1[i*2], 100), (points_1[i*2 + 1], 140), (0,0,255), 2)
        a_1.append([depth_img[100:140, points_1[i*2]:points_1[i*2 + 1]]])
        depth_1.append(np.mean(a_1[i]))
        org_1.append((points_1[i*2]-5, 140 + 15))
        depth_img = cv2.putText(depth_img, str(round(depth_1[i])), org_1[i], font, fontScale, (0,0,255), 1, cv2.LINE_AA)

    points_2 = [85, 115, 145, 175, 205, 235]
    a_2 = []
    depth_2 = []
    org_2 = []
    for i in range(3):
        depth_img = cv2.rectangle((depth_img), (points_2[i*2], 70), (points_2[i*2 + 1], 100), (0,0,255), 2)
        a_2.append([depth_img[70:100, points_2[i*2]:points_2[i*2 + 1]]])
        depth_2.append(np.mean(a_2[i]))
        org_2.append((points_2[i*2]-5, 70-5))
        depth_img = cv2.putText(depth_img, str(round(depth_2[i])), org_2[i], font, fontScale, (0,0,255), 1, cv2.LINE_AA)

    return depth_img, depth_1, depth_2

class image_feature:
	
	def __init__(self):
		self.subscriber1 = rospy.Subscriber("/team1/camera/depth/compressed",CompressedImage, self.callback_depth,  queue_size = 1)
		self.subscriber2 = rospy.Subscriber("/team1/camera/rgb/compressed",CompressedImage, self.callback_rgb,  queue_size = 1)
		self.speed = rospy.Publisher('team1/set_speed', Float32, queue_size=10)
		self.angle = rospy.Publisher('team1/set_angle', Float32, queue_size=10)

	def callback_depth(self, ros_data):
		np_arr_depth = np.frombuffer(ros_data.data, np.uint8)
		depth0 = cv2.imdecode(np_arr_depth, cv2.IMREAD_COLOR)
		depth, depth_list_1, depth_list_2 = get_depth(depth0)
		self.speed.publish(50)
		cv2.imshow('depth', depth)
		# cv2.waitKey(2)
	
	def callback_rgb(self, ros_data):
		np_arr_rgb = np.frombuffer(ros_data.data, np.uint8)
		rgb = cv2.imdecode(np_arr_rgb, cv2.IMREAD_COLOR)
		canny_img = lane_detect_module.preprocess(rgb)
		self.angle.publish(0)
        

def main(args):
	ic = image_feature()
	rospy.init_node('image', anonymous=True)
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print('e')
	cv2.destroyAllWindows()

# %%	
if __name__ == '__main__':
	try:
		# main()
		main(sys.argv)
	except rospy.ROSInterruptException:
		pass



	
