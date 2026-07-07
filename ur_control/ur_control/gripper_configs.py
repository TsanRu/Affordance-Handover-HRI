# Gripper client configuration registry.
#
# Instead of passing a per-gripper --params-file to every client, a bringup declares which
# gripper is present (e.g. the simulation launcher publishes its name on the latched
# /active_gripper topic), and the client resolves the matching parameters from here.
#
# Each entry is the set of node parameters that ur_control.grippers.GripperController reads.

import time

import rclpy
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import String

ACTIVE_GRIPPER_TOPIC = "/active_gripper"

GRIPPER_CONFIGS = {
    # PickNik 2F-85 in gz: GripperActionController (gripper_cmd) on the single knuckle.
    "robotiq_2f85": {
        "gripper_type": "85",
        "joint": "robotiq_85_left_knuckle_joint",
    },
    # Robotiq Hand-E in gz: JointTrajectoryController on both prismatic fingers.
    "robotiq_hande": {
        "gripper_type": "hand-e",
        # only the actuated finger; hande_right_finger_joint is a mimic
        "joint": "finger_joint",
        "gripper_action_interface": "trajectory",
        "gripper_trajectory_controller": "gripper_controller",
        "gripper_finger_max_position": 0.02,
    },
    "none": {},
}


def apply_gripper_config(node, name):
    """Declare/set the parameters for gripper C{name} on C{node}.

    Returns the config dict (empty for 'none'/unknown). Call before constructing the
    gripper so GripperController reads these parameters.
    """
    cfg = GRIPPER_CONFIGS.get(name)
    if cfg is None:
        node.get_logger().warn("Unknown gripper '%s'; known: %s" % (name, list(GRIPPER_CONFIGS)))
        return {}
    for key, value in cfg.items():
        if node.has_parameter(key):
            node.set_parameters([Parameter(key, value=value)])
        else:
            node.declare_parameter(key, value)
    return cfg


def read_active_gripper(node, timeout=5.0):
    """Return the gripper name published (latched) on /active_gripper, or None on timeout.

    Requires C{node} to be spun by an executor (e.g. a background MultiThreadedExecutor).
    """
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST)
    holder = {}
    sub = node.create_subscription(String, ACTIVE_GRIPPER_TOPIC,
                                   lambda m: holder.setdefault("name", m.data), qos)
    start = time.time()
    while "name" not in holder and (time.time() - start) < timeout and rclpy.ok():
        time.sleep(0.05)
    node.destroy_subscription(sub)
    return holder.get("name")
