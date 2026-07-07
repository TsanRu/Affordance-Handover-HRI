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

import collections
import numbers
import threading
import time
import types
import numpy as np

import rclpy
from rclpy.qos import qos_profile_sensor_data

from ur_control import utils, conversions
from ur_control.arm import Arm
from ur_control.constants import CARTESIAN_COMPLIANCE_CONTROLLER, ExecutionResult
from ur_control.fzi_utils import (
    is_more_extreme,
    convert_selection_matrix_to_parameters,
    convert_stiffness_to_parameters,
    convert_pd_gains_to_parameters,
    switch_cartesian_controllers
)

from geometry_msgs.msg import WrenchStamped, PoseStamped
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType


def _to_parameter_value(value):
    """Build an rcl_interfaces ParameterValue, inferring the type from a Python/numpy value."""
    pv = ParameterValue()
    # bool must be checked before Integral (bool is a subclass of int).
    if isinstance(value, bool):
        pv.type = ParameterType.PARAMETER_BOOL
        pv.bool_value = bool(value)
    elif isinstance(value, numbers.Integral):
        pv.type = ParameterType.PARAMETER_INTEGER
        pv.integer_value = int(value)
    elif isinstance(value, numbers.Real):
        pv.type = ParameterType.PARAMETER_DOUBLE
        pv.double_value = float(value)
    elif isinstance(value, str):
        pv.type = ParameterType.PARAMETER_STRING
        pv.string_value = value
    else:
        raise TypeError("Unsupported parameter type for value %r" % (value,))
    return pv


class _ControllerParamClient:
    """Replaces ROS 1 dynamic_reconfigure: sets parameters on a ros2_control controller
    node via its ``<controller>/set_parameters`` service. Assumes the owning node is spun
    by a background executor (so the synchronous ``call`` resolves)."""

    def __init__(self, node, controller_node_name):
        self.node = node
        self._client = node.create_client(SetParameters, controller_node_name.rstrip('/') + '/set_parameters')

    def set(self, parameters: dict, timeout=1.0):
        if not self._client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError("set_parameters service unavailable: %s" % self._client.srv_name)
        request = SetParameters.Request()
        for name, value in parameters.items():
            request.parameters.append(Parameter(name=name, value=_to_parameter_value(value)))
        return self._client.call(request)


class CompliantController(Arm):
    """
    A compliant controller using FZI Cartesian Compliance controllers.

    This class extends the Arm class to provide compliant control capabilities
    using the FZI Cartesian Compliance controllers. It allows for setting
    target poses and wrenches, and provides methods for controlling the robot
    in a compliant manner.
    """

    def __init__(self, **kwargs):
        """
        Initialize the CompliantController.

        Args:
            **kwargs: Additional arguments to pass to the Arm constructor
        """
        Arm.__init__(self, **kwargs)

        # use_gazebo_sim is a node parameter (ROS 2 has no global param server).
        self.is_gazebo_sim = bool(utils.read_parameter(self.node, "use_gazebo_sim", False))

        self.rate = utils.Rate(self.joint_traj_controller.rate)
        self.min_dt = 1. / self.joint_traj_controller.rate

        self.auto_switch_controllers = True  # Safety switching back to safe controllers

        self.current_target_pose = np.zeros(7)
        self.current_wrench_pose = np.zeros(6)

        # Monitor external goals
        self.node.create_subscription(PoseStamped, '%s%s/target_frame' % (self.ns, CARTESIAN_COMPLIANCE_CONTROLLER), self.target_pose_cb, qos_profile_sensor_data)
        self.node.create_subscription(WrenchStamped, '%s%s/target_wrench' % (self.ns, CARTESIAN_COMPLIANCE_CONTROLLER), self.target_wrench_cb, qos_profile_sensor_data)

        self.cartesian_target_pose_pub = self.node.create_publisher(PoseStamped, '%s%s/target_frame' % (self.ns, CARTESIAN_COMPLIANCE_CONTROLLER), 10)
        self.cartesian_target_wrench_pub = self.node.create_publisher(WrenchStamped, '%s%s/target_wrench' % (self.ns, CARTESIAN_COMPLIANCE_CONTROLLER), 10)

        # ROS 2: the controller's tunables are node parameters on the single controller
        # node, set via one SetParameters service (replaces ROS 1 dynamic_reconfigure clients).
        self.param_client = _ControllerParamClient(self.node, '%s%s' % (self.ns, CARTESIAN_COMPLIANCE_CONTROLLER))

        self.param_update_queue = collections.deque(maxlen=15)
        self.update_thread = None
        self.update_lock = threading.Lock()
        self.update_condition = threading.Condition()
        self.update_thread_stopped = False
        self.async_mode = False

        self.set_hand_frame_control(False)
        self.set_end_effector_link(self.ee_link)
        self.min_scale_error = 1.5

        # ROS 2 has no rospy.on_shutdown; register on the node's context instead.
        try:
            self.node.context.on_shutdown(self.activate_joint_trajectory_controller)
        except Exception:
            pass

    def __del__(self):
        """
        Destructor that ensures the update thread is stopped.
        """
        # wake up thread and stop it
        if hasattr(self, 'update_condition'):
            with self.update_condition:
                self.update_thread_stopped = True
                self.update_condition.notify_all()

    def target_pose_cb(self, data):
        """
        Callback for target pose messages.

        Args:
            data (PoseStamped): The target pose message
        """
        self.current_target_pose = conversions.from_pose_to_list(data.pose)

    def target_wrench_cb(self, data):
        """
        Callback for target wrench messages.

        Args:
            data (WrenchStamped): The target wrench message
        """
        self.current_target_wrench = conversions.from_wrench(data.wrench)

    def activate_cartesian_controller(self):
        """
        Activate the cartesian compliance controller.

        Returns:
            bool: True if the controller was activated successfully, False otherwise
        """
        if self.running_controller_name == CARTESIAN_COMPLIANCE_CONTROLLER:
            return True
        if self.controller_manager.switch_controllers(controllers_on=[CARTESIAN_COMPLIANCE_CONTROLLER],
                                                      controllers_off=[self.running_controller_name]):
            self.running_controller_name = CARTESIAN_COMPLIANCE_CONTROLLER
            return True
        raise Exception("Failed to activate cartesian compliance controller")
        return False

    def set_cartesian_target_wrench(self, wrench: list):
        """
        Set the target wrench for the cartesian compliance controller.

        Args:
            wrench (list): A 6-element list representing the target wrench [fx, fy, fz, tx, ty, tz]
        """
        # Publish the target wrench
        try:
            target_wrench = WrenchStamped()
            target_wrench.header.frame_id = self.base_link
            target_wrench.wrench = conversions.to_wrench(wrench)
            self.cartesian_target_wrench_pub.publish(target_wrench)
        except Exception as e:
            self.node.get_logger().error("Fail to set_target_wrench(): %s" % e)

    def set_cartesian_target_pose(self, pose: list):
        """
        Set the target pose for the cartesian compliance controller.

        Args:
            pose (list): A 7-element list representing the target pose [x, y, z, qx, qy, qz, qw]
        """
        # Publish the target pose
        try:
            target_pose = conversions.to_pose_stamped(self.base_link, pose)
            self.cartesian_target_pose_pub.publish(target_pose)
        except Exception as e:
            self.node.get_logger().error("Fail to set_target_pose(): %s" % e)

    def publish_parameter_update(self, parameters):
        """
        Publish parameter updates to the controller.

        Args:
            parameters (dict): A flat {param_name: value} dict of controller node parameters
        """
        try:
            self.param_client.set(parameters)
        except Exception as e:
            self.node.get_logger().error("failed publish_parameter_update %s" % e, throttle_duration_sec=1.0)

    def __update_controller_parameter_loop__(self):
        """
        Internal method that runs in a separate thread to update controller parameters.
        """
        while rclpy.ok():
            if self.update_thread_stopped:
                return

            # Sleep until new request is available
            if not self.param_update_queue:
                with self.update_condition:
                    self.update_condition.wait(timeout=0.5)
                return
            else:
                # Lock queue update
                with self.update_lock:
                    parameters = self.param_update_queue.pop()
                if parameters:
                    self.publish_parameter_update(parameters)
                    parameters = None

    def update_controller_parameters(self, parameters: dict):
        """
        Update controller parameters.

        Args:
            parameters (dict): A dictionary containing the parameters to update
        """
        if self.async_mode:
            with self.update_lock:
                self.param_update_queue.append(parameters)
            if self.update_thread is None or not self.update_thread.is_alive():
                del self.update_thread
                self.update_thread = threading.Thread(target=self.__update_controller_parameter_loop__)
                self.update_thread.start()
            with self.update_condition:
                self.update_condition.notify()
        else:
            self.publish_parameter_update(parameters)

    def update_selection_matrix(self, selection_matrix):
        """
        Update the selection matrix for the controller.

        Args:
            selection_matrix (numpy.ndarray): A 6-element array representing the selection matrix
        """
        parameters = convert_selection_matrix_to_parameters(selection_matrix)
        self.update_controller_parameters(parameters)

    def update_pd_gains(self, p_gains, d_gains=[0, 0, 0, 0, 0, 0]):
        """
        Update the P and D gains for the controller.

        Args:
            p_gains (numpy.ndarray): A 6-element array representing the P gains
            d_gains (numpy.ndarray, optional): A 6-element array representing the D gains. Defaults to zeros.
        """
        parameters = convert_pd_gains_to_parameters(p_gains, d_gains)
        self.update_controller_parameters(parameters)

    def update_stiffness(self, stiffness):
        """
        Update the stiffness values for the controller.

        Args:
            stiffness (numpy.ndarray): A 6-element array representing the stiffness values
        """
        parameters = convert_stiffness_to_parameters(stiffness)
        self.update_controller_parameters(parameters)

    def set_control_mode(self, mode="parallel"):
        """
        Set the control mode for the controller.

        Args:
            mode (str, optional): The control mode to set. Options are "parallel" or "spring-mass-damper". Defaults to "parallel".

        Raises:
            ValueError: If an unknown control mode is specified
        """
        if mode == "parallel":
            parameters = {"stiffness.use_parallel_force_position_control": True}
        elif mode == "spring-mass-damper":
            parameters = {"stiffness.use_parallel_force_position_control": False}
        else:
            raise ValueError("Unknown control mode %s" % mode)
        self.update_controller_parameters(parameters)

    def set_position_control_mode(self, enable=True):
        """
        Set the position control mode for the controller.

        Args:
            enable (bool, optional): Whether to enable position control mode. Defaults to True.
        """
        parameters = convert_selection_matrix_to_parameters(np.ones(6))
        parameters.update({"stiffness.use_parallel_force_position_control": enable})
        parameters.update({"stiffness.use_selection_matrix_in_gripper_frame": enable})
        self.update_controller_parameters(parameters)

    def set_hand_frame_control(self, enable):
        """
        Set whether to use hand frame control.

        Args:
            enable (bool): Whether to enable hand frame control
        """
        parameters = {"hand_frame_control": enable}
        self.update_controller_parameters(parameters)

    def set_end_effector_link(self, end_effector_link):
        """
        Change the end effector link used in the Cartesian Compliance Controllers.

        Args:
            end_effector_link (str): The name of the end effector link
        """
        parameters = {"end_effector_link": end_effector_link}
        self.update_controller_parameters(parameters)

    def set_solver_parameters(self, error_scale=None, iterations=None, publish_state_feedback=None):
        """
        Set solver parameters for the controller.

        Args:
            error_scale (float, optional): The error scale parameter. Defaults to None.
            iterations (int, optional): The number of iterations. Defaults to None.
            publish_state_feedback (bool, optional): Whether to publish state feedback. Defaults to None.
        """
        parameters = {}
        if error_scale:
            error_scale = error_scale
            parameters["solver.error_scale"] = round(error_scale, 4)
        if iterations:
            parameters["solver.iterations"] = int(iterations)
        if publish_state_feedback:
            parameters["solver.publish_state_feedback"] = publish_state_feedback
        self.update_controller_parameters(parameters)

    @switch_cartesian_controllers
    def execute_compliance_control(self, trajectory: np.array, target_wrench: np.array, max_force_torque: list,
                                   duration: float, stop_on_target_force=False, termination_criteria=None,
                                   auto_stop=True, func=None, scale_up_error=False, max_scale_error=None,
                                   stop_at_wrench=None):
        """
        Execute compliance control with a trajectory and target wrench.

        Args:
            trajectory (np.array): The trajectory to follow
            target_wrench (np.array): The target wrench to apply
            max_force_torque (list): The maximum force and torque limits
            duration (float): The duration of the trajectory in seconds
            stop_on_target_force (bool, optional): Whether to stop when the target force is reached. Defaults to False.
            termination_criteria (callable, optional): A function that returns True when the execution should stop. Defaults to None.
            auto_stop (bool, optional): Whether to automatically stop at the end of the trajectory. Defaults to True.
            func (callable, optional): A function to call during execution. Defaults to None.
            scale_up_error (bool, optional): Whether to scale up the error. Defaults to False.
            max_scale_error (float, optional): The maximum scale error. Defaults to None.
            stop_at_wrench (list, optional): The wrench at which to stop. Defaults to None.

        Returns:
            ExecutionResult: The result of the execution
        """
        # Set initial pose and wrench to zero
        self.set_cartesian_target_pose(self.end_effector())
        self.set_cartesian_target_wrench(np.zeros(6))

        # Space out the trajectory points
        trajectory = trajectory.reshape((-1, 7))  # Assuming this format [x,y,z,qx,qy,qz,qw]
        step_duration = max(self.min_dt, duration / float(trajectory.shape[0]))
        trajectory_index = 0

        # loop throw target trajectory
        initial_time = time.monotonic()
        step_initial_time = time.monotonic()

        result = ExecutionResult.DONE
        if stop_on_target_force and stop_at_wrench is None:
            raise ValueError("'stop_at_wrench' not specify when requesting 'stop_on_target_force'")

        if stop_on_target_force:
            stop_at_wrench = np.array(stop_at_wrench)
            stop_target_wrench_mask = np.flatnonzero(stop_at_wrench)
            self.node.get_logger().info('TARGET F/T {}'.format(np.round(stop_at_wrench[stop_target_wrench_mask], 2)), throttle_duration_sec=1.0)

        # Publish target wrench only once
        self.set_cartesian_target_wrench(target_wrench)

        # Publish first trajectory point
        self.set_cartesian_target_pose(trajectory[trajectory_index])

        if scale_up_error and max_scale_error:
            self.sliding_error(trajectory[trajectory_index], max_scale_error)

        while rclpy.ok() and (time.monotonic() - initial_time) < duration:

            current_wrench = self.get_wrench(base_frame_control=True)

            if termination_criteria is not None:
                assert isinstance(termination_criteria, types.LambdaType), "Invalid termination criteria, expecting lambda/function with one argument[current pose array[7]]"
                if termination_criteria(self.end_effector()):
                    self.node.get_logger().info("Termination criteria returned True, stopping force control")
                    result = ExecutionResult.TERMINATION_CRITERIA
                    break

            self.node.get_logger().info('F/T {}'.format(np.round(current_wrench[:3], 2)), throttle_duration_sec=1.0)

            # Check if any of the monitored wrench dimensions have exceeded their target values
            if stop_on_target_force:
                # Check if any dimension has exceeded its target
                if is_more_extreme(current_wrench[stop_target_wrench_mask], stop_at_wrench[stop_target_wrench_mask]):
                    self.node.get_logger().info('Target F/T reached {}'.format(np.round(current_wrench, 2)) + ' Stopping!')
                    result = ExecutionResult.STOP_ON_TARGET_FORCE
                    break

            # Safety limits: max force
            if np.any(np.abs(current_wrench) > max_force_torque):
                self.node.get_logger().error('Maximum force/torque exceeded {}'.format(np.round(current_wrench, 3)))
                result = ExecutionResult.FORCE_TORQUE_EXCEEDED
                break

            if (time.monotonic() - step_initial_time) > step_duration:
                step_initial_time = time.monotonic()
                trajectory_index += 1
                if trajectory_index >= trajectory.shape[0]:
                    break
                # push next point to the controller
                self.set_cartesian_target_pose(trajectory[trajectory_index])

                if scale_up_error and max_scale_error:
                    self.sliding_error(trajectory[trajectory_index], max_scale_error)

            if func:
                func(self.end_effector())

            self.rate.sleep()

        if auto_stop:
            # Stop moving
            # set position control only, then fix the pose to the current one
            self.set_position_control_mode()
            self.set_cartesian_target_pose(self.end_effector())
            self.set_cartesian_target_wrench(np.zeros(6))
            self.wait_for_robot_to_stop(wait_time=5)

        return result

    def sliding_error(self, target_pose, max_scale_error):
        """
        Scale error_scale as position error decreases until a max scale error.

        Args:
            target_pose (np.array): The target pose
            max_scale_error (float): The maximum scale error
        """
        # Scale error_scale as position error decreases until a max scale error
        position_error = np.linalg.norm(target_pose[:3] - self.end_effector()[:3])
        # from position_error < 0.01m increase scale error
        factor = 1 - np.tanh(100 * position_error)
        # scale_error = np.interp(factor, [0, 1], [0.01, max_scale_error])
        scale_error = np.interp(factor, [0, 1], [self.min_scale_error, max_scale_error])
        self.set_solver_parameters(error_scale=np.round(scale_error, 3))
