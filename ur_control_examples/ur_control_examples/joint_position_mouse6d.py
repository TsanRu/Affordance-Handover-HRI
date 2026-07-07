#!/usr/bin/env python

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

"""UR Joint Position Example: 3Dconnexion mouse (ROS 2)."""

import argparse
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from ur_control import transformations, utils
from ur_control.arm import Arm
from ur_control.mouse_6d import Mouse6D

np.set_printoptions(suppress=True)


def print_robot_state(arm):
    print("Joint angles:", np.round(arm.joint_angles(), 3))
    print("End Effector:", np.round(arm.end_effector(rot_type='euler'), 3))


def start_control(arm, mouse6d, motion_type="linear"):
    print("Start moving. type", motion_type)
    rate = utils.Rate(125)
    delta_x = 0.01
    delta_q = np.deg2rad(1)

    while rclpy.ok():
        if mouse6d.twist is None:
            rate.sleep()
            continue

        x = arm.end_effector()
        xd = np.array(mouse6d.twist)

        xd[:3] = [delta_x * np.sign(xd[i]) if abs(xd[i]) > 0.15 else 0.0 for i in range(3)]
        xd[3:] = [delta_q * np.sign(xd[3 + i]) if abs(xd[3 + i]) > 0.15 else 0.0 for i in range(3)]
        if motion_type == "rotated":
            xd[2] *= -1
        elif motion_type != "linear":
            print("motion_type not supported", motion_type)
            break

        x = transformations.pose_from_angular_velocity(x, xd, dt=0.25)
        if mouse6d.joy_buttons and mouse6d.joy_buttons[0] == 1:
            print_robot_state(arm)

        arm.set_target_pose(pose=x, target_time=0.25, wait=False)
        rate.sleep()


def main(args=None):
    """3D mouse control."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('-r', action='store_true', help='move using relative rotation of end-effector')
    parser.add_argument('--namespace', type=str, default=None, help='robot namespace')
    parser.add_argument('--tcp', type=str, default='tool0', help='end-effector link for IK')

    argv = remove_ros_args(args if args is not None else sys.argv)
    cli_args = parser.parse_args(argv[1:])

    rclpy.init(args=args)
    node = Node('joint_position_mouse6d')

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    time.sleep(1.0)

    try:
        joints_prefix = cli_args.namespace + '_' if cli_args.namespace else None
        arm = Arm(node,
                  namespace=cli_args.namespace,
                  joint_names_prefix=joints_prefix,
                  ee_link=cli_args.tcp,
                  gripper_type=None)
        arm.dashboard_services.activate_ros_control_on_ur()

        mouse6d = Mouse6D(node)
        motion_type = "rotated" if cli_args.r else "linear"
        start_control(arm, mouse6d, motion_type=motion_type)
        print("Done.")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
