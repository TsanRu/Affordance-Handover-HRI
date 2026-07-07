#!/usr/bin/env python
import os
import yaml
import copy
import collections
import time
import math

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data
from ament_index_python.packages import get_package_share_directory

from ur_control import utils, constants
import numpy as np
from std_msgs.msg import Float64, Float64MultiArray
from controller_manager_msgs.srv import ListControllers
# Joint trajectory action
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory


# NOTE (ROS 2): every controller class takes an rclpy ``node`` and assumes that node
# is being spun by an executor in a background thread (a MultiThreadedExecutor). The
# joint_states callback firing and the synchronous service/action waits below all rely
# on that external spinning.


def degrees_constructor(loader, node):
    """Custom YAML constructor for !degrees tag that converts degrees to radians."""
    value = loader.construct_scalar(node)
    return math.radians(float(value))


# Register the custom constructor with the YAML loader
yaml.SafeLoader.add_constructor('!degrees', degrees_constructor)


class JointControllerBase(object):
    """
    Base class for the Joint Position Controllers. It subscribes to the C{joint_states} topic by default.
    """

    def __init__(self, node, namespace, timeout, joint_names=None):
        """
        JointControllerBase constructor. It subscribes to the C{joint_states} topic and informs after
        successfully reading a message from the topic.
        @type  node: rclpy.node.Node
        @param node: shared node (spun by a background executor).
        @type namespace: string
        @param namespace: Override ROS namespace manually. Useful when controlling several robots
        @type  timeout: float
        @param timeout: Time in seconds that will wait for the controller
        """
        self.node = node
        self.valid_joint_names = constants.JOINT_ORDER if joint_names is None else joint_names

        self.ns = utils.solve_namespace(namespace, node=node)
        self._jnt_positions_hist = collections.deque(maxlen=24)
        # Set-up publishers/subscribers
        self._js_sub = node.create_subscription(JointState, 'joint_states', self.joint_states_cb, qos_profile_sensor_data)
        retry = False
        self.node.get_logger().debug('Waiting for [%sjoint_states] topic' % self.ns)
        start_time = time.time()
        while not hasattr(self, '_joint_names'):
            if (time.time() - start_time) > timeout and not retry:
                # Re-try with namespace
                self._js_sub = node.create_subscription(JointState, '%sjoint_states' % self.ns, self.joint_states_cb, qos_profile_sensor_data)
                start_time = time.time()
                retry = True
                continue
            elif (time.time() - start_time) > timeout and retry:
                self.node.get_logger().error('Timed out waiting for joint_states topic')
                return
            time.sleep(0.01)
            if not rclpy.ok():
                return

        # joint_state publish rate (node-scoped; ROS 2 has no global param server).
        self.rate = utils.read_parameter(node, "joint_state_publish_rate", 500)

        self._num_joints = len(self._joint_names)
        self.node.get_logger().debug('Topic [%sjoint_states] found' % self.ns)

    def disconnect(self):
        """
        Disconnects from the joint_states topic. Useful to ligthen the use of system resources.
        """
        self.node.destroy_subscription(self._js_sub)

    def get_joint_efforts(self):
        """
        Returns the current joint efforts of the UR robot.
        @rtype: numpy.ndarray
        @return: Current joint efforts of the UR robot.
        """
        return np.array(self._current_jnt_efforts)

    def get_joint_positions(self):
        """
        Returns the current joint positions of the UR robot.
        @rtype: numpy.ndarray
        @return: Current joint positions of the UR robot.
        """
        return np.array(self._current_jnt_positions)

    def get_joint_positions_hist(self):
        """
        Returns the current joint positions of the UR robot.
        @rtype: numpy.ndarray
        @return: Current joint positions of the UR robot.
        """
        return list(self._jnt_positions_hist)

    def get_joint_velocities(self):
        """
        Returns the current joint velocities of the UR robot.
        @rtype: numpy.ndarray
        @return: Current joint velocities of the UR robot.
        """
        return np.array(self._current_jnt_velocities)

    def joint_states_cb(self, msg):
        """
        Callback executed every time a message is publish in the C{joint_states} topic.
        @type  msg: sensor_msgs/JointState
        @param msg: The JointState message published by the RT hardware interface.
        """
        position = []
        velocity = []
        effort = []
        name = []
        for joint_name in self.valid_joint_names:
            if joint_name in msg.name:
                idx = msg.name.index(joint_name)
                name.append(msg.name[idx])
                if msg.effort:
                    effort.append(msg.effort[idx])
                if msg.velocity:
                    velocity.append(msg.velocity[idx])
                position.append(msg.position[idx])
        if set(name) == set(self.valid_joint_names):
            self._current_jnt_positions = np.array(position)
            self._jnt_positions_hist.append(self._current_jnt_positions)
            self._current_jnt_velocities = np.array(velocity)
            self._current_jnt_efforts = np.array(effort)
            self._joint_names = list(name)


class JointPositionController(JointControllerBase):
    """
    Interface class to control the UR robot using a Joint Position Control approach.
    If you C{set_joint_positions} to a value very far away from the current robot position,
    it will move at its maximum speed/acceleration and will even move the base of the robot, so, B{use with caution}.
    """

    def __init__(self, node, namespace='', timeout=5.0, joint_names=None):
        super(JointPositionController, self).__init__(node, namespace, timeout=timeout, joint_names=joint_names)
        if not hasattr(self, '_joint_names'):
            raise RuntimeError('JointPositionController timed out waiting joint_states topic: {0}'.format(namespace))
        self._cmd_pub = dict()
        for joint in self._joint_names:
            self._cmd_pub[joint] = node.create_publisher(Float64, '%s%scommand' % (self.ns, joint), 3)
        # Wait for the joint position controllers
        controller_list_srv = self.ns + 'controller_manager/list_controllers'
        self.node.get_logger().debug('Waiting for the joint position controllers...')
        list_controllers = node.create_client(ListControllers, controller_list_srv)
        if not list_controllers.wait_for_service(timeout_sec=timeout):
            raise RuntimeError('JointPositionController timed out waiting for controller_manager: {0}'.format(namespace))
        expected_controllers = (joint_names if joint_names is not None else constants.JOINT_ORDER)
        start_time = time.time()
        while rclpy.ok():
            if (time.time() - start_time) > timeout:
                raise RuntimeError('JointPositionController timed out waiting for the controller_manager: {0}'.format(namespace))
            time.sleep(0.01)
            found = 0
            try:
                res = list_controllers.call(ListControllers.Request())
                for state in res.controller:
                    if state.name in expected_controllers:
                        found += 1
            except Exception:
                pass
            if found == len(expected_controllers):
                break
        self.node.get_logger().info('JointPositionController initialized. ns: {0}'.format(namespace))

    def set_joint_positions(self, jnt_positions):
        """
        Sets the joint positions of the robot. The values are send directly to the robot.
        @type jnt_positions: list
        @param jnt_positions: Joint positions command.
        """
        if not self.valid_jnt_command(jnt_positions):
            self.node.get_logger().warn('A valid joint positions command should have %d elements' % (self._num_joints))
            return
        # Publish the point for each joint
        for name, q in zip(self._joint_names, jnt_positions):
            try:
                self._cmd_pub[name].publish(Float64(data=float(q)))
            except Exception:
                self.node.get_logger().error("Failed to publish joint command")

    def valid_jnt_command(self, command):
        """
        It validates that the length of a joint command is equal to the number of joints
        @type command: list
        @param command: Joint command to be validated
        @rtype: bool
        @return: True if the joint command is valid
        """
        return (len(command) == self._num_joints)


class JointVelocityController(JointControllerBase):
    """
    Interface class to control the UR robot using a Joint Velocity Control approach.
    """

    def __init__(self, node, controller_name='joint_group_vel_controller', namespace='', timeout=5.0, joint_names=None, robot_version=None):
        super(JointVelocityController, self).__init__(node, namespace, timeout=timeout, joint_names=joint_names)
        if not hasattr(self, '_joint_names'):
            raise RuntimeError('JointVelocityController timed out waiting joint_states topic: {0}'.format(namespace))

        # Publisher for velocity commands. ROS 2 forward_command_controller
        # (JointGroupVelocityController) subscribes to <controller>/commands.
        self.controller_name = controller_name
        self.velocity_pub = node.create_publisher(Float64MultiArray, f'/{self.controller_name}/commands', 1)

        # Initialize joint limits
        self.joint_limits = self._load_joint_limits(robot_version.lower())

        self.node.get_logger().info('JointVelocityController initialized. ns: {0}'.format(namespace))

    def _load_joint_limits(self, robot_version=None):
        """Load joint limits from YAML file."""
        # Try to find the default UR5e joint limits file
        try:
            ur_desc_path = get_package_share_directory('ur_description')
            joint_limits_file = os.path.join(ur_desc_path, 'config', robot_version, 'joint_limits.yaml')
        except Exception as e:
            self.node.get_logger().warn(f"Could not find ur_description package: {e}")
            return self._get_default_limits()

        try:
            with open(joint_limits_file, 'r') as f:
                limits_data = yaml.safe_load(f)

            joint_limits = {}
            for joint_name in self._joint_names:
                # Remove '_joint' suffix to match YAML keys
                yaml_key = joint_name.replace('_joint', '').replace('_', '_')
                if yaml_key == 'shoulder_pan':
                    yaml_key = 'shoulder_pan'
                elif yaml_key == 'shoulder_lift':
                    yaml_key = 'shoulder_lift'
                elif yaml_key == 'elbow':
                    yaml_key = 'elbow_joint'
                elif yaml_key.startswith('wrist'):
                    yaml_key = yaml_key.replace('_', '_')

                if yaml_key in limits_data['joint_limits']:
                    limits = limits_data['joint_limits'][yaml_key]
                    joint_limits[joint_name] = {
                        'min_position': limits.get('min_position', math.radians(-360.0)),
                        'max_position': limits.get('max_position', math.radians(360.0)),
                        'max_velocity': limits.get('max_velocity', math.radians(180.0)),
                        'max_effort': limits.get('max_effort', 150.0)
                    }
                else:
                    self.node.get_logger().warn(f"Joint {joint_name} not found in limits file, using defaults")
                    joint_limits[joint_name] = self._get_default_limits()[joint_name]

            self.node.get_logger().info("Successfully loaded joint limits from YAML file")
            return joint_limits

        except Exception as e:
            self.node.get_logger().warn(f"Failed to load joint limits from {joint_limits_file}: {e}")
            return self._get_default_limits()

    def _get_default_limits(self):
        """Get default joint limits if YAML file is not available."""
        return {
            'shoulder_pan_joint': {
                'min_position': np.radians(-360.0),
                'max_position': np.radians(360.0),
                'max_velocity': np.radians(180.0),
                'max_effort': 150.0
            },
            'shoulder_lift_joint': {
                'min_position': np.radians(-360.0),
                'max_position': np.radians(360.0),
                'max_velocity': np.radians(180.0),
                'max_effort': 150.0
            },
            'elbow_joint': {
                'min_position': np.radians(-180.0),
                'max_position': np.radians(180.0),
                'max_velocity': np.radians(180.0),
                'max_effort': 150.0
            },
            'wrist_1_joint': {
                'min_position': np.radians(-360.0),
                'max_position': np.radians(360.0),
                'max_velocity': np.radians(180.0),
                'max_effort': 28.0
            },
            'wrist_2_joint': {
                'min_position': np.radians(-360.0),
                'max_position': np.radians(360.0),
                'max_velocity': np.radians(180.0),
                'max_effort': 28.0
            },
            'wrist_3_joint': {
                'min_position': np.radians(-360.0),
                'max_position': np.radians(360.0),
                'max_velocity': np.radians(180.0),
                'max_effort': 28.0
            }
        }

    def set_joint_velocities(self, jnt_velocities):
        """
        Sets the joint velocities of the robot. The values are send directly to the robot.
        @type jnt_velocities: list
        @param jnt_velocities: Joint velocities command.
        """
        if len(jnt_velocities) != 6:
            self.node.get_logger().error("Velocity command must have exactly 6 values")
            return

        # Apply velocity limits
        vel_array = self._enforce_velocity_limits(jnt_velocities)

        # Create and publish message
        msg = Float64MultiArray()
        msg.data = [float(v) for v in vel_array]

        self.velocity_pub.publish(msg)
        self.node.get_logger().debug(f"Sent velocities: {vel_array}")

    def _enforce_velocity_limits(self, velocities):
        """Enforce velocity limits on the command."""
        limited_velocities = velocities.copy()

        for i, (joint_name, vel) in enumerate(zip(self._joint_names, velocities)):
            max_vel = self.joint_limits[joint_name]['max_velocity']

            if abs(vel) > max_vel:
                limited_vel = np.sign(vel) * max_vel
                self.node.get_logger().warn(
                    f"Joint {joint_name} velocity limited from {vel:.3f} to {limited_vel:.3f} rad/s",
                    throttle_duration_sec=1.0)
                limited_velocities[i] = limited_vel

        return limited_velocities

    def stop_all_joints(self):
        """Stop all joint motion immediately."""
        self.set_joint_velocities([0.0] * 6)


class JointTrajectoryController(JointControllerBase):
    """
    This class creates a C{rclpy.action.ActionClient} that connects to the
    C{trajectory_controller/follow_joint_trajectory} action server. Using this
    interface you can control the robot by adding points to the trajectory.

    The synchronous (actionlib-style) API of the ROS 1 version is re-expressed over
    the asynchronous ROS 2 action API: goal acceptance and the result are captured by
    done-callbacks fired on the (background) executor thread, and ``wait()`` polls them.
    """

    def __init__(self, node, publisher_name='scaled_joint_trajectory_controller', namespace='', timeout=5.0, joint_names=None):
        super(JointTrajectoryController, self).__init__(node, namespace, timeout=timeout, joint_names=joint_names)

        # ROS 2 JointTrajectoryController command topic is <controller>/joint_trajectory.
        trajectory_publisher_topic = self.ns + publisher_name + '/joint_trajectory'
        self.trajectory_pub = node.create_publisher(JointTrajectory, trajectory_publisher_topic, 10)

        action_server = self.ns + publisher_name + '/follow_joint_trajectory'
        self._client = ActionClient(node, FollowJointTrajectory, action_server)
        self._goal = FollowJointTrajectory.Goal()
        self._goal_handle = None
        self._result = None
        self._status = None
        self.node.get_logger().debug('Waiting for [%s] action server' % action_server)
        if not self._client.wait_for_server(timeout_sec=timeout):
            self.node.get_logger().error('Timed out waiting for Joint Trajectory'
                                         ' Action Server to connect. Start the action server'
                                         ' before running this node.')
            raise RuntimeError('JointTrajectoryController timed out: {0}'.format(action_server))
        self.node.get_logger().debug('Successfully connected to [%s]' % action_server)
        # Get a copy of joint_names
        if not hasattr(self, '_joint_names'):
            raise RuntimeError('JointTrajectoryController timed out waiting joint_states topic: {0}'.format(self.ns))
        self._goal.trajectory.joint_names = copy.deepcopy(self._joint_names)
        self.node.get_logger().info('JointTrajectoryController initialized. ns: {0}'.format(self.ns))

    def add_point(self, target_time, positions, velocities=None, accelerations=None):
        """
        Adds a point to the trajectory. Each point must be specified by the goal position and
        the goal time. The velocity and acceleration are optional.
        @type  positions: list
        @param positions: The goal position in the joint space
        @type  target_time: float
        @param target_time: The time B{from start} when the robot should arrive at the goal position.
        """
        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in positions]
        if velocities is None:
            point.velocities = [0.0] * self._num_joints
        else:
            point.velocities = [float(v) for v in velocities]
        if accelerations is None:
            point.accelerations = [0.0] * self._num_joints
        else:
            point.accelerations = [float(a) for a in accelerations]
        point.time_from_start = Duration(seconds=target_time).to_msg()
        self._goal.trajectory.points.append(point)

    def clear_points(self):
        """
        Clear all points in the trajectory.
        """
        self._goal.trajectory.points = []

    def get_num_points(self):
        """
        Returns the number of points currently added to the trajectory
        @rtype: int
        @return: Number of points currently added to the trajectory
        """
        return len(self._goal.trajectory.points)

    def get_result(self):
        """
        Returns the result B{after} the execution of the trajectory
        (control_msgs/action/FollowJointTrajectory.Result), or None if not finished.
        """
        return self._result

    def get_state(self):
        """
        Returns the action goal status B{during/after} execution as an
        action_msgs/msg/GoalStatus value, or None. Note: ROS 2 GoalStatus values differ
        from ROS 1 actionlib (e.g. SUCCEEDED == 4 in ROS 2, not 3).
        """
        return self._status

    def set_trajectory(self, trajectory):
        """
        Sets the goal trajectory directly. B{It only copies} the C{trajectory.points} field.
        @type  trajectory: trajectory_msgs/JointTrajectory
        @param trajectory: The goal trajectory
        """
        self._goal.trajectory.points = copy.deepcopy(trajectory.points)

    def _goal_response_cb(self, future):
        self._goal_handle = future.result()
        if self._goal_handle is not None and self._goal_handle.accepted:
            self._goal_handle.get_result_async().add_done_callback(self._result_cb)

    def _result_cb(self, future):
        response = future.result()
        self._result = response.result
        self._status = response.status

    def start(self, delay=0.1, wait=False):
        """
        Starts the trajectory. It sends the C{FollowJointTrajectory.Goal} to the action server.
        @type  delay: float
        @param delay: Delay (in seconds) before executing the trajectory
        """
        num_points = len(self._goal.trajectory.points)
        self.node.get_logger().debug('Executing Joint Trajectory with {0} points'.format(num_points))
        if delay == 0:
            stamp = Time().to_msg()  # zero stamp => start as soon as possible
        else:
            stamp = (self.node.get_clock().now() + Duration(seconds=delay)).to_msg()
        self._goal.trajectory.header.stamp = stamp
        self._result = None
        self._status = None
        self._goal_handle = None
        send_future = self._client.send_goal_async(self._goal)
        send_future.add_done_callback(self._goal_response_cb)
        if wait:
            self.wait()

    def stop(self):
        """
        Stops an active trajectory.
        """
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()

    def wait(self, timeout=15.0):
        """
        Waits synchronously (with a timeout) until the trajectory action gives a result.
        @type  timeout: float
        @param timeout: The amount of time we will wait
        @rtype: bool
        @return: True if a result was received in the allocated time; False on timeout/rejection.
        """
        start_time = time.time()
        while (time.time() - start_time) < timeout and rclpy.ok():
            if self._result is not None:
                return True
            if self._goal_handle is not None and not self._goal_handle.accepted:
                return False  # goal rejected
            time.sleep(0.01)
        return self._result is not None

    def start_no_action_server(self):
        """
        Start the trajectory without expecting any feedback from the action server.
        """
        self.trajectory_pub.publish(self._goal.trajectory)
