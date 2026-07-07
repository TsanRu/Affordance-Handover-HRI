# The MIT License (MIT)
#
# Copyright (c) 2018-2021 Cristian Beltran
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Author: Cristian Beltran

"""UR Joint Position Example: keyboard control (ROS 2)."""

import argparse
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from ur_control import transformations
from ur_control.arm import Arm
from ur_control.constants import GripperType, IKSolverType
from ur_control.gripper_configs import apply_gripper_config, read_active_gripper, GRIPPER_CONFIGS
from ur_control.getch import getch

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)


def map_keyboard(arm, relative_to_tcp):
    delta_q = np.deg2rad(1.0)
    delta_x = 0.005

    def print_robot_state():
        print("Joint angles:", np.round(arm.joint_angles(), 4).tolist())
        print("EE Pose:", np.round(arm.end_effector(), 5).tolist())
        print("EE Pose (euler):", np.round(arm.end_effector(rot_type="euler"), 5).tolist())
        if arm.gripper:
            print("Gripper position:", np.round(arm.gripper.get_position(), 4))
            # opening_width only exists on RobotiqGripper (CModel status feedback).
            if hasattr(arm.gripper, "opening_width"):
                print("Gripper opening_width:", np.round(arm.gripper.opening_width, 4))
            print("Gripper percentage:", np.round(arm.gripper.get_opening_percentage(), 4))

    def set_j(joint_name, sign):
        nonlocal delta_q
        current_position = arm.joint_angles()
        current_position[joint_name] += delta_q * sign
        arm.set_joint_positions(positions=current_position, target_time=0.25)

    def update_d(delta, increment):
        nonlocal delta_q, delta_x
        if delta == 'q':
            delta_q += np.deg2rad(increment)
            print(("delta_q", np.rad2deg(delta_q)))
        if delta == 'x':
            delta_x += increment
            print(("delta_x", delta_x))

    def set_pose_ik(dim, sign):
        nonlocal delta_q, delta_x
        x = arm.end_effector()
        delta = np.zeros(6)

        if dim <= 2:  # position
            delta[dim] += delta_x * sign
        else:  # rotation
            delta[dim] += delta_q * sign

        xc = transformations.transform_pose(x, delta, rotated_frame=relative_to_tcp)
        arm.set_target_pose(pose=xc, target_time=0.25)

    def open_gripper():
        arm.gripper.open(wait=False)

    def close_gripper():
        arm.gripper.close(wait=False)

    def move_gripper(delta):
        cpose = arm.gripper.get_position()
        cpose += delta
        arm.gripper.command(cpose, wait=False)

    bindings = {
        #   key: (function, args, description)
        'z': (set_j, [0, 1], "shoulder_pan_joint increase"),
        'v': (set_j, [0, -1], "shoulder_pan_joint decrease"),
        'x': (set_j, [1, 1], "shoulder_lift_joint increase"),
        'c': (set_j, [1, -1], "shoulder_lift_joint decrease"),
        'a': (set_j, [2, 1], "elbow_joint increase"),
        'f': (set_j, [2, -1], "elbow_joint decrease"),
        's': (set_j, [3, 1], "wrist_1_joint increase"),
        'd': (set_j, [3, -1], "wrist_1_joint decrease"),
        'q': (set_j, [4, 1], "wrist_2_joint increase"),
        'r': (set_j, [4, -1], "wrist_2_joint decrease"),
        'w': (set_j, [5, 1], "wrist_3_joint increase"),
        'e': (set_j, [5, -1], "wrist_3_joint decrease"),
        'p': (print_robot_state, [], "right: printing"),
        # Task Space
        'h': (set_pose_ik, [0, 1], "x increase"),
        'k': (set_pose_ik, [0, -1], "x decrease"),
        'y': (set_pose_ik, [1, 1], "y increase"),
        'i': (set_pose_ik, [1, -1], "y decrease"),
        'u': (set_pose_ik, [2, 1], "z increase"),
        'j': (set_pose_ik, [2, -1], "z decrease"),
        'n': (set_pose_ik, [3, 1], "ax increase"),
        'm': (set_pose_ik, [3, -1], "ax decrease"),
        ',': (set_pose_ik, [4, 1], "ay increase"),
        '.': (set_pose_ik, [4, -1], "ay decrease"),
        'o': (set_pose_ik, [5, 1], "az increase"),
        'l': (set_pose_ik, [5, -1], "az decrease"),

        # Increase or decrease delta
        '1': (update_d, ['q', 0.25], "delta_q increase"),
        '2': (update_d, ['q', -0.25], "delta_q decrease"),
        '6': (update_d, ['x', 0.0001], "delta_x increase"),
        '7': (update_d, ['x', -0.0001], "delta_x decrease"),

        # Gripper
        '5': (move_gripper, [0.005], "open gripper a bit"),
        't': (open_gripper, [], "open gripper"),
        'g': (close_gripper, [], "close gripper"),
        'b': (move_gripper, [-0.005], "close gripper a bit"),
    }
    done = False
    print("Controlling joints. Press ? for help, Esc to quit.")
    while not done and rclpy.ok():
        c = getch()
        if c:
            # catch Esc or ctrl-c
            if c in ['\x1b', '\x03']:
                done = True
            elif c in bindings:
                cmd = bindings[c]
                cmd[0](*cmd[1])
                print(("command: %s" % (cmd[2], )))
            else:
                print("key bindings: ")
                print("  Esc: Quit")
                print("  ?: Help")
                for key, val in sorted(
                        list(bindings.items()), key=lambda x: x[1][2]):
                    print(("  %s: %s" % (key, val[2])))


def main(args=None):
    """Joint Position Example: Keyboard Control

    Use your dev machine's keyboard to control joint positions.

    Each key corresponds to increasing or decreasing the angle
    of a joint on the robot arm.
    """
    epilog = """
See help inside the example with the '?' key for key bindings.
    """
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(
        formatter_class=arg_fmt, description=main.__doc__, epilog=epilog)
    parser.add_argument(
        '--relative', action='store_true', help='Motion Relative to ee')
    parser.add_argument(
        '--namespace', type=str, help='Namespace of arm (useful when having multiple arms)', default=None)
    parser.add_argument(
        '--gripper', type=str, default='auto',
        help="gripper: 'auto' (read /active_gripper from the bringup), a registry name "
             "(robotiq_2f85, robotiq_hande, none), or legacy 'generic'/'robotiq'.")
    parser.add_argument(
        '--tcp', type=str, help='Tool Center Point or End-Effector frame for IK without joint prefix', default='tool0'
    )

    argv = remove_ros_args(args if args is not None else sys.argv)
    cli_args = parser.parse_args(argv[1:])

    rclpy.init(args=args)
    node = Node("joint_position_keyboard")

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    time.sleep(1.0)

    try:
        tcp_link = cli_args.tcp
        joints_prefix = cli_args.namespace + '_' if cli_args.namespace else None
        # Resolve which gripper to use. 'auto' reads the latched /active_gripper topic
        # published by the bringup, so the gripper is defined once (in the launcher).
        gripper_name = cli_args.gripper
        if gripper_name in (None, 'auto'):
            gripper_name = read_active_gripper(node) or 'none'
            node.get_logger().info("Active gripper (from /active_gripper): %s" % gripper_name)

        if gripper_name in ('none', None):
            gripper = None
        elif gripper_name == 'robotiq':      # legacy: real Robotiq CModel gripper
            gripper = GripperType.ROBOTIQ
        elif gripper_name == 'generic':      # legacy: GripperController, params provided externally
            gripper = GripperType.GENERIC
        elif gripper_name in GRIPPER_CONFIGS:  # registry gripper -> config resolved internally
            apply_gripper_config(node, gripper_name)
            gripper = GripperType.GENERIC
        else:
            node.get_logger().error("Unknown gripper '%s'; loading without gripper" % gripper_name)
            gripper = None

        arm = Arm(node,
                  namespace=cli_args.namespace,
                  gripper_type=gripper,
                  joint_names_prefix=joints_prefix,
                  ee_link=tcp_link,
                  #   ik_solver=IKSolverType.KDL,
                  )

        arm.dashboard_services.activate_ros_control_on_ur()

        map_keyboard(arm, relative_to_tcp=cli_args.relative)
        print("Done.")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
