"""Latch the active gripper name on /active_gripper for ur_control clients.

Publishes ONCE (transient-local) and stays alive so late subscribers still get it.
Used by the simulation bringups so the gripper is defined once, in the launcher.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import String

from ur_control.gripper_configs import ACTIVE_GRIPPER_TOPIC


def main(args=None):
    rclpy.init(args=args)
    node = Node("active_gripper_publisher")
    name = str(node.declare_parameter("gripper", "none").value)
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST)
    pub = node.create_publisher(String, ACTIVE_GRIPPER_TOPIC, qos)
    pub.publish(String(data=name))  # once; transient-local serves late subscribers
    node.get_logger().info("Latched %s = '%s'" % (ACTIVE_GRIPPER_TOPIC, name))
    try:
        rclpy.spin(node)  # keep the publisher (and its latched sample) alive
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
