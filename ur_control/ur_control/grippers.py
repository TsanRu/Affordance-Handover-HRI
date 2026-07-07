# Gripper action
import time
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from control_msgs.action import GripperCommand, FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from ur_control import utils




# NOTE (ROS 2): gripper config (joint names, gripper_type, max_gap, ...) is read from
# the shared node's parameters (launch-provided) rather than a global param server.
# Functional validation of the grippers depends on the robotiq_control ROS 2 port.


class GripperControllerBase():
    def __init__(self, node, namespace='', node_name='', prefix=None, timeout=5.0) -> None:
        self.node = node
        self.ns = namespace
        self.prefix = prefix if prefix is not None else ''
        self._goal_handle = None
        self._result = None
        self._status = None

        # Gripper joint name(s), provided to the shared node as parameters.
        joint = utils.read_parameter(node, "joint", None)
        joints = utils.read_parameter(node, "joints", None)
        joint_name = utils.read_parameter(node, "joint_name", None)
        self.valid_joint_names = []
        if joint is not None:
            self.valid_joint_names = [joint]
        elif joints is not None:
            self.valid_joint_names = joints
        elif joint_name is not None:
            if isinstance(joint_name, str):
                self.valid_joint_names = [self.prefix + joint_name]
            else:
                self.valid_joint_names = joint_name
        else:
            raise RuntimeError(
                "No gripper joint params found for '%s'. Provide 'joint'/'joints'/'joint_name' "
                "via a gripper client params file, e.g. --ros-args --params-file "
                "$(ros2 pkg prefix ur_gripper_gz)/share/ur_gripper_gz/config/gripper_2f85_client.yaml" % node_name)

        self._js_sub = node.create_subscription(JointState, '/joint_states', self.joint_states_cb, qos_profile_sensor_data)

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
                self.node.get_logger().error('Timed out waiting for gripper joint_states topic')
                return
            # Rely on the shared node's background executor (do NOT spin_once here:
            # spinning a node that an executor is already spinning corrupts its wait set).
            time.sleep(0.01)
            if not rclpy.ok():
                return

    def open(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def get_position(self):
        return self._current_jnt_positions[0]

    def get_velocity(self):
        return self._current_jnt_velocities[0]

    def get_opening_percentage(self):
        raise NotImplementedError()

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
                effort.append(msg.effort[idx])
                velocity.append(msg.velocity[idx])
                position.append(msg.position[idx])

        if set(name) == set(self.valid_joint_names):
            self._current_jnt_positions = np.array(position)
            self._current_jnt_velocities = np.array(velocity)
            self._current_jnt_efforts = np.array(effort)
            self._joint_names = list(name)

    # --- shared async-action helpers (ROS 2) ---------------------------------
    def _goal_response_cb(self, future):
        self._goal_handle = future.result()
        if self._goal_handle is not None and self._goal_handle.accepted:
            self._goal_handle.get_result_async().add_done_callback(self._result_cb)

    def _result_cb(self, future):
        response = future.result()
        self._result = response.result
        self._status = response.status

    def _send_goal(self, client, goal, wait, timeout=5.0):
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()
        self._result = None
        self._status = None
        self._goal_handle = None
        send_future = client.send_goal_async(goal)
        send_future.add_done_callback(self._goal_response_cb)
        if wait:
            return self._wait_result(timeout)
        return True

    def _wait_result(self, timeout=15.0):
        start_time = time.time()
        while (time.time() - start_time) < timeout and rclpy.ok():
            if self._result is not None:
                return True
            if self._goal_handle is not None and not self._goal_handle.accepted:
                return False
            # Poll; results arrive via done-callbacks on the shared background executor.
            time.sleep(0.01)
        return self._result is not None

    def _make_trajectory_goal(self, finger_position, duration=1.0):
        goal = FollowJointTrajectory.Goal()
        pos = float(finger_position)
        # Sim safety clamp: gz_ros2_control ignores URDF joint limits, and over-closing
        # the Hand-E fingers past finger/body contact jams them in gz. _finger_max defaults
        # to a no-op for real hardware; set it (e.g. 0.02) via the client params for sim.
        pos = float(np.clip(pos, 0.0, getattr(self, "_finger_max", 1.0)))
        if self.gripper_type == "hand-e" and getattr(self, "_use_trajectory", False):
            # Only the actuated finger is commanded; hande_right_finger_joint is a URDF
            # <mimic> of finger_joint (enforced by gz bullet-featherstone), so it is not a
            # controller joint and must not appear in the trajectory.
            goal.trajectory.joint_names = [self.prefix + "finger_joint"]
            point = JointTrajectoryPoint()
            point.positions = [pos]
        else:
            goal.trajectory.joint_names = list(self.valid_joint_names)
            point = JointTrajectoryPoint()
            point.positions = [pos] * len(goal.trajectory.joint_names)
        point.velocities = [0.0] * len(point.positions)
        point.time_from_start = Duration(seconds=duration).to_msg()
        goal.trajectory.points = [point]
        goal.trajectory.header.stamp = Time().to_msg()
        return goal


class GripperController(GripperControllerBase):
    def __init__(self, node, namespace='', prefix=None, timeout=5.0):
        node_name = "gripper_controller"
        super().__init__(node, namespace, node_name, prefix, timeout)
        self.gripper_type = str(utils.read_parameter(node, "gripper_type", "85"))

        if self.gripper_type == "hand-e":
            self._max_gap = 0.025 * 2.0
            self._to_open = 0.0
            self._to_close = self._max_gap
        elif self.gripper_type == "85":
            self._max_gap = 0.085
            self._to_open = self._max_gap
            self._to_close = 0.001
            self._max_angle = 0.8028
        elif self.gripper_type == "140":
            self._max_gap = 0.140
            self._to_open = self._max_gap
            self._to_close = 0.001
            self._max_angle = 0.69

        self._use_trajectory = (
            self.gripper_type == "hand-e"
            and str(utils.read_parameter(node, "gripper_action_interface", "gripper_command")) == "trajectory")

        # Per-finger position clamp (meters). Defaults to a no-op (real hardware uses the
        # full URDF travel); set to e.g. 0.02 in the sim client params to avoid the gz
        # over-close jam.
        self._finger_max = float(utils.read_parameter(node, "gripper_finger_max_position", 1.0))

        if self._use_trajectory:
            traj_controller = str(utils.read_parameter(node, "gripper_trajectory_controller", "gripper_controller"))
            action_server = self.ns + traj_controller + '/follow_joint_trajectory'
            self._client = ActionClient(node, FollowJointTrajectory, action_server)
            self.node.get_logger().debug('Waiting for [%s] trajectory action server' % action_server)
            if not self._client.wait_for_server(timeout_sec=timeout):
                self.node.get_logger().error('Timed out waiting for gripper trajectory action server: %s' % action_server)
                raise RuntimeError('Gripper trajectory action timed out: {0}'.format(action_server))
            self.node.get_logger().info('Hand-E gripper trajectory action initialized. ns: {0}'.format(self.ns))
        else:
            # Gripper action server
            action_server = self.ns + node_name + '/gripper_cmd'
            self._client = ActionClient(node, GripperCommand, action_server)
            self._goal = GripperCommand.Goal()
            self.node.get_logger().debug('Waiting for [%s] action server' % action_server)
            if not self._client.wait_for_server(timeout_sec=timeout):
                self.node.get_logger().error('Timed out waiting for Gripper Command'
                                             ' Action Server to connect. Start the action server'
                                             ' before running this node.')
                raise RuntimeError('GripperCommand action timed out: {0}'.format(action_server))
            self.node.get_logger().debug('Successfully connected to [%s]' % action_server)
            self.node.get_logger().info('GripperCommand action initialized. ns: {0}'.format(self.ns))

    def close(self, wait=True):
        return self.command(0.0, percentage=True, wait=wait)

    def percentage_command(self, value, wait=True):
        return self.command(value, percentage=True, wait=wait)

    def command(self, value, percentage=False, wait=True):
        """ assume command given in percentage otherwise meters
            percentage bool: If True value value assumed to be from 0.0 to 1.0
                                     where 1.0 is open and 0.0 is close
                             If False value value assume to be from 0.0 to max_gap
        """
        if value == "close":
            return self.close()
        elif value == "open":
            return self.open()

        if self.gripper_type == "85" or self.gripper_type == "140":
            if percentage:
                value = np.clip(value, 0.0, 1.0)
                cmd = (value) * self._max_gap
            else:
                cmd = np.clip(value, 0.0, self._max_gap)
                cmd = (value)
            angle = self._distance_to_angle(cmd)
            self._goal.command.position = float(angle)
        if self.gripper_type == "hand-e":
            cmd = 0.0
            if percentage:
                value = np.clip(value, 0.0, 1.0)
                cmd = (1.0 - value) * self._max_gap / 2.0
            else:
                cmd = np.clip(value, 0.0, self._max_gap)
                cmd = (self._max_gap - value) / 2.0
            if self._use_trajectory:
                goal = self._make_trajectory_goal(cmd)
            else:
                self._goal.command.position = float(cmd)
                goal = self._goal
        else:
            goal = self._goal
        if wait:
            self._send_goal(self._client, goal, wait=True, timeout=5.0)
            time.sleep(0.05)
        else:
            self._send_goal(self._client, goal, wait=False)
        return True

    def _distance_to_angle(self, distance):
        distance = np.clip(distance, 0, self._max_gap)
        angle = (self._max_gap - distance) * self._max_angle / self._max_gap
        return angle

    def _angle_to_distance(self, angle):
        angle = np.clip(angle, 0, self._max_angle)
        distance = (self._max_angle - angle) * self._max_gap / self._max_angle
        return distance

    def get_result(self):
        return self._result

    def get_state(self):
        return self._status

    def open(self, wait=True):
        return self.command(1.0, percentage=True, wait=wait)

    def stop(self):
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()

    def wait(self, timeout=15.0):
        return self._wait_result(timeout)

    def get_position(self):
        """
        Returns the current joint positions of the gripper.
        @rtype: numpy.ndarray
        @return: Current joint positions of the gripper.
        """
        if self.gripper_type == "hand-e":
            return self._max_gap - (self._current_jnt_positions[0] * 2.0)
        else:
            return self._angle_to_distance(self._current_jnt_positions[0])

    def get_opening_percentage(self):
        return self.get_position() / self._max_gap


class RobotiqGripper(GripperControllerBase):
    try:
        import robotiq_msgs.action
    except ImportError:
        print("Robotiq gripper can't be loaded. robotiq_msgs required.")

    def __init__(self, node, namespace="", prefix="", timeout=2):
        node_name = "gripper_action_controller"
        super().__init__(node, namespace, node_name, prefix, timeout)
        if not namespace or namespace == "/":
            self.ns = ""
        else:
            self.ns = namespace

        self.opening_width = 0.0

        self.gripper = ActionClient(node, robotiq_msgs.action.CModelCommand, self.ns + "gripper_action_controller")
        self.sub_gripper_status_ = node.create_subscription(robotiq_msgs.action.CModelCommand.Feedback, "%sgripper_status" % self.ns, self._gripper_status_callback, qos_profile_sensor_data)

        joint_name = utils.read_parameter(node, "counts_to_meters", None)
        if utils.read_parameter(node, "joint_name", None) is not None:
            self.gripper_type = utils.read_parameter(node, "joint_name", "finger_joint")
            self._max_gap = float(utils.read_parameter(node, "max_gap", 0.085))
            self._max_angle = float(utils.read_parameter(node, "counts_to_meters", 0.8))
        else:
            self.node.get_logger().warn("Robotiq gripper parameters not found. Assuming Robotiq Gripper 85")
            self.gripper_type = "finger_joint"
            self._max_gap = 0.085
            self._max_angle = 0.8

        if self.gripper_type == "robotiq_hande_joint_finger":
            self._max_gap = self._max_gap * 2.0
            self._to_open = 0.0
            self._to_close = self._max_gap
        elif self.gripper_type == "finger_joint":
            self._to_open = self._max_gap
            self._to_close = 0.001

        if self.gripper.wait_for_server(timeout_sec=timeout):
            self.node.get_logger().info("=== Connected to ROBOTIQ gripper ===")
        else:
            self.node.get_logger().error("Unable to connect to ROBOTIQ gripper")

    def _gripper_status_callback(self, msg):
        self.opening_width = msg.position  # [m]

    def get_opening_percentage(self):
        return self.opening_width / self._max_gap

    def close(self, force=40.0, velocity=1.0, wait=True):
        return self.command("close", force=force, velocity=velocity, wait=wait)

    def open(self, velocity=1.0, wait=True, opening_width=None):
        command = opening_width if opening_width else "open"
        return self.command(command, wait=wait, velocity=velocity)

    def convert_percentage_to_width(self, width):
        if self.gripper_type == "finger_joint":
            width = np.clip(width, 0.0, self._max_gap)
            percentage = width / self._max_gap
        if self.gripper_type == "hand-e":
            raise ValueError("Unimplemented")
        return percentage

    def convert_width_to_percentage(self, percentage):
        if self.gripper_type == "finger_joint":
            percentage = np.clip(percentage, 0.0, 1.0)
            width = (percentage) * self._max_gap
        if self.gripper_type == "hand-e":
            percentage = np.clip(percentage, 0.0, 1.0)
            width = (1.0 - percentage) * self._max_gap / 2.0
        return width

    def percentage_command(self, value, wait=True):
        """
        0.0 = Fully Close
        1.0 = Fully Open
        """
        if self.gripper_type == "finger_joint":
            value = np.clip(value, 0.0, 1.0)
            cmd = (value) * self._max_gap
            return self.command(cmd, wait=wait)
        if self.gripper_type == "hand-e":
            value = np.clip(value, 0.0, 1.0)
            cmd = (1.0 - value) * self._max_gap / 2.0
            return self.command(cmd, wait=wait)

    def command(self, command, force=40.0, velocity=1.0, wait=True):
        """
        command: "open", "close" or opening width
        force: Gripper force in N. From 40 to 100
        velocity: Gripper speed. From 0.013 to 0.1

        Use a slow closing speed when using a low gripper force, or the force might be unexpectedly high.
        """
        goal = robotiq_msgs.action.CModelCommand.Goal()
        goal.velocity = float(velocity)
        goal.force = float(force)
        if command == "close":
            goal.position = 0.0
        elif command == "open":
            goal.position = 0.140
        else:
            goal.position = float(command)     # This sets the opening width directly

        self.node.get_logger().debug("Sending command " + str(command) + " to gripper: " + self.ns)
        if wait:
            ok = self._send_goal(self.gripper, goal, wait=True, timeout=5.0)
            return bool(ok and self._result is not None)
        else:
            self._send_goal(self.gripper, goal, wait=False)
            return True
