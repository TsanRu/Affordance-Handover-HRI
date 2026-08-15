import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def image_callback(msg):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")  # 轉換成 OpenCV 格式
    cv2.imshow("Gazebo Camera", cv_image)
    cv2.waitKey(1)

rospy.init_node("camera_listener")
rospy.Subscriber("/usb_cam/image_raw", Image, image_callback)
rospy.spin()