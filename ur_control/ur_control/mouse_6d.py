import time

from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Joy


class Mouse6D:
    """Subscribe to 3Dconnexion spacenav twist/joy topics."""

    def __init__(self, node: Node):
        self.node = node
        self.twist = None
        self.joy_axes = None
        self.joy_buttons = None

        node.create_subscription(
            Twist, 'spacenav/twist', self.twist_cb, qos_profile_sensor_data)
        node.create_subscription(
            Joy, 'spacenav/joy', self.joy_cb, qos_profile_sensor_data)

        time.sleep(0.01)

    def twist_cb(self, msg):
        self.twist = [
            msg.linear.x,
            msg.linear.y,
            msg.linear.z,
            msg.angular.x,
            msg.angular.y,
            msg.angular.z,
        ]

    def joy_cb(self, msg):
        self.joy_axes = msg.axes
        self.joy_buttons = msg.buttons
