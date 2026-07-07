#!/usr/bin/env python

# The MIT License (MIT)
#
# Copyright (c) 2023 Cristian Beltran
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

"""FZI cartesian compliance controller examples (ROS 2)."""

import argparse
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from ur_control import traj_utils
from ur_control.fzi_cartesian_compliance_controller import CompliantController

np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)


def move_joints(arm):
    q = [1.5689, -1.0497, 1.4306, -1.998, -1.5684, 0.0003]
    arm.set_joint_positions(positions=q, target_time=3, wait=True)


def move_cartesian(arm, node):
    q = [1.3524, -1.5555, 1.7697, -1.7785, -1.5644, 1.3493]
    arm.set_joint_positions(positions=q, target_time=3, wait=True)

    arm.set_position_control_mode(True)
    arm.set_control_mode(mode="parallel")
    arm.set_solver_parameters(error_scale=0.5, iterations=1)
    arm.update_stiffness([1500, 1500, 1500, 100, 100, 100])

    selection_matrix = [1, 1, 0, 1, 1, 1]
    arm.update_selection_matrix(selection_matrix)

    p_gains = [0.05, 0.05, 0.05, 1.5, 1.5, 1.5]
    d_gains = [0.005, 0.005, 0.005, 0, 0, 0]
    arm.update_pd_gains(p_gains, d_gains=d_gains)

    theta = q
    arm.set_joint_positions(positions=theta, target_time=3, wait=True)

    ee = arm.end_effector()
    print("EE", ee)

    steps = 100
    x = -0.17 * np.ones(steps)
    y = 0.51 * np.ones(steps)
    z = 0.0755 * np.ones(steps)
    trajectory_quat = ee[3:] * np.ones((100, 4))
    trajectory_pos = np.stack((x, y, z)).T
    ref_traj = np.zeros((steps, 7))
    ref_traj[:, :3] = trajectory_pos
    ref_traj[:, 3:] = trajectory_quat

    target_force = np.zeros(6) * np.ones((steps, 6))

    arm.set_target_pose(
        pose=ref_traj[0, :] + np.array([0, 0, 0.01, 0, 0, 0, 0]),
        target_time=3,
        wait=True,
    )

    def f(x, _w):
        node.get_logger().info('x: %s' % str(x[:3]), throttle_duration_sec=0.25)

    arm.zero_ft_sensor()
    arm.execute_compliance_control(
        ref_traj,
        target_wrench=target_force,
        max_force_torque=[50., 50., 50., 5., 5., 5.],
        duration=15,
        func=f,
        scale_up_error=True,
        max_scale_error=3.0,
        auto_stop=False,
    )
    print("EE total displacement", np.round(ee - arm.end_effector(), 4))
    arm.set_joint_positions(positions=theta, target_time=3, wait=True)


def move_force(arm):
    arm.zero_ft_sensor()

    arm.set_control_mode("parallel")
    arm.set_solver_parameters(error_scale=0.8, iterations=1)

    selection_matrix = [1, 1, 0, 1, 1, 1]
    arm.update_selection_matrix(selection_matrix)

    arm.update_stiffness([500, 500, 500, 100, 100, 100])

    p_gains = [0.05, 0.05, 0.05, 1.5, 1.5, 1.5]
    d_gains = [0.0, 0.0, 0.0, 0, 0, 0]
    arm.update_pd_gains(p_gains, d_gains)

    ee = arm.end_effector()

    target_force = [0, 0, -5, 0, 0, 0]
    stop_at_wrench = np.copy(target_force)
    stop_at_wrench *= -1

    res = arm.execute_compliance_control(
        ee,
        target_wrench=target_force,
        max_force_torque=[30., 30., 30., 4., 4., 4.],
        duration=15,
        stop_at_wrench=stop_at_wrench,
        stop_on_target_force=True,
    )
    print(res)
    print("EE change", ee - arm.end_effector())


def slicing(arm):
    arm.zero_ft_sensor()

    selection_matrix = [1, 1, 0, 1, 1, 1]
    arm.update_selection_matrix(selection_matrix)

    pd_gains = [0.03, 0.03, 0.03, 1.0, 1.0, 1.0]
    arm.update_pd_gains(pd_gains)

    ee = arm.end_effector()

    trajectory = traj_utils.compute_sinusoidal_trajectory(
        ee, dimension=1, period=3, amplitude=0.02, num_of_points=100)
    target_force = [0, 0, -3, 0, 0, 0]

    res = arm.execute_compliance_control(
        trajectory, target_wrench=target_force,
        max_force_torque=[50., 50., 50., 5., 5., 5.], duration=20)
    print(res)
    print("EE change", ee - arm.end_effector())


def admittance_control(arm, node):
    node.get_logger().info('START ADMITTANCE')

    arm.set_control_mode(mode="spring-mass-damper")

    ee = arm.end_effector()
    target_force = np.zeros(6)
    arm.execute_compliance_control(
        ee, target_wrench=target_force,
        max_force_torque=[50., 50., 50., 5., 5., 5.], duration=10,
        stop_on_target_force=False)

    node.get_logger().info('STOP ADMITTANCE')


def free_drive(arm, node):
    node.get_logger().info('START FREE DRIVE')
    time.sleep(0.5)
    arm.zero_ft_sensor()
    arm.set_control_mode("spring-mass-damper")
    arm.set_solver_parameters(error_scale=1.0, iterations=1.0)
    pd_gains = [0.1, 0.1, 0.1, 2.0, 2.0, 3.0]
    d_gains = [0.0, 0.0, 0.0, 0, 0, 0]
    arm.update_pd_gains(pd_gains, d_gains)
    arm.update_stiffness([100, 100, 100, 5, 5, 5])

    ee = arm.end_effector()
    target_force = np.zeros(6)

    res = arm.execute_compliance_control(
        ee, target_wrench=target_force,
        max_force_torque=[50., 50., 50., 5., 5., 5.], duration=60,
        stop_on_target_force=False)
    print(res)
    print("EE change", ee - arm.end_effector())
    node.get_logger().info('STOP FREE DRIVE')


def test(arm):
    move_joints(arm)

    arm.move_relative(
        transformation=[0, 0, -0.03, 0, 0, 0],
        relative_to_tcp=False,
        target_time=0.5,
        wait=True,
    )
    arm.move_relative(
        transformation=[0, 0, -0.03, 0, 0, 0],
        relative_to_tcp=False,
        target_time=0.5,
        wait=True,
    )
    for _ in range(3):
        arm.move_relative(
            transformation=[0, 0, -0.03, 0, 0, 0],
            relative_to_tcp=False,
            target_time=0.5,
            wait=True,
        )

    arm.move_relative(
        transformation=[0, 0, 0.03, 0, 0, 0],
        relative_to_tcp=False,
        target_time=0.25,
        wait=True,
    )
    arm.move_relative(
        transformation=[0, 0.01, 0, 0, 0, 0],
        relative_to_tcp=False,
        target_time=0.25,
        wait=True,
    )


def enable_compliance_control(arm):
    q = [1.3524, -1.5555, 1.7697, -1.7785, -1.5644, 1.3493]
    arm.set_joint_positions(q, target_time=1, wait=True)

    arm.zero_ft_sensor()
    arm.set_control_mode(mode="spring-mass-damper")

    arm.activate_cartesian_controller()
    arm.set_cartesian_target_pose(arm.end_effector(tip_link="b_bot_gripper_tip_link"))

    for _ in range(30):
        print("current target pose", arm.current_target_pose[:3])
        print("error", arm.current_target_pose[:3] -
              arm.end_effector(tip_link="b_bot_gripper_tip_link")[:3])
        time.sleep(1)

    arm.activate_joint_trajectory_controller()


def main(args=None):
    parser = argparse.ArgumentParser(description='Test force control')
    parser.add_argument('-m', '--move_joints', action='store_true',
                        help='move to joint configuration')
    parser.add_argument('-mc', '--move_cartesian', action='store_true',
                        help='move to cartesian configuration')
    parser.add_argument('-mf', '--move_force', action='store_true',
                        help='move towards target force')
    parser.add_argument('-fd', '--free_drive', action='store_true',
                        help='move the robot freely')
    parser.add_argument('-a', '--admittance', action='store_true',
                        help='Spring-mass-damper force control demo')
    parser.add_argument('-s', '--slicing', action='store_true',
                        help='Push down while oscillating on X-axis')
    parser.add_argument('-t', '--test', action='store_true', help='Test')
    parser.add_argument('-teleop', '--teleoperation', action='store_true',
                        help='Enable cartesian controllers for teleoperation')
    parser.add_argument('--namespace', type=str, default=None)

    argv = remove_ros_args(args if args is not None else sys.argv)
    cli_args = parser.parse_args(argv[1:])

    rclpy.init(args=args)
    node = Node('cartesian_compliance_controller_examples')

    # These examples target the gz sim by default. use_gazebo_sim scales solver.error_scale
    # (see CompliantController.set_solver_parameters) and, with use_real_robot=false, keeps the
    # arm from calling UR-driver-only services (e.g. hardware FT zeroing). Override for real HW:
    #   ros2 run ... --ros-args -p use_gazebo_sim:=false -p use_real_robot:=true
    node.declare_parameter('use_gazebo_sim', True)
    node.declare_parameter('use_real_robot', False)

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    time.sleep(1.0)

    try:
        joints_prefix = cli_args.namespace + '_' if cli_args.namespace else None
        arm = CompliantController(
            node=node,
            namespace=cli_args.namespace,
            joint_names_prefix=joints_prefix,
            ee_link='gripper_tip_link',
            ft_topic='wrench',
            gripper_type=None)

        if not arm.dashboard_services.activate_ros_control_on_ur():
            return

        if cli_args.move_joints:
            move_joints(arm)
        if cli_args.move_cartesian:
            move_cartesian(arm, node)
        if cli_args.move_force:
            move_force(arm)
        if cli_args.admittance:
            admittance_control(arm, node)
        if cli_args.free_drive:
            free_drive(arm, node)
        if cli_args.slicing:
            slicing(arm)
        if cli_args.test:
            test(arm)
        if cli_args.teleoperation:
            enable_compliance_control(arm)
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
