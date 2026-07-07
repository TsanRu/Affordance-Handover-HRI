# The MIT License (MIT)
#
# Copyright (c) 2018-2023 Cristian Beltran
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
import time
import numpy as np

from geometry_msgs.msg import WrenchStamped
from std_srvs.srv import Empty, SetBool, Trigger
from rclpy.qos import qos_profile_sensor_data

from ur_control import utils, spalg, conversions, transformations
from ur_control.exceptions import InverseKinematicsException
from ur_control.controllers_connection import ControllersConnection
from ur_control.controllers import JointTrajectoryController, JointVelocityController
from ur_control.constants import BASE_LINK, CARTESIAN_COMPLIANCE_CONTROLLER, EE_LINK,  FT_SUBSCRIBER, JOINT_POSITION_TRAJECTORY_CONTROLLER, JOINT_VELOCITY_TRAJECTORY_CONTROLLER, VELOCITY_CONTROLLER_NAME,  \
    ExecutionResult, IKSolverType, GripperType, \
    get_arm_joint_names
from ur_control.ur_services import URServices
from ur_control.eaik_kinematics import EAIKKinematics

from ur_pykdl import ur_kinematics, get_robot_description

# ros2_control lifecycle state name for an activated controller (ROS 1 used "running").
_ACTIVE = "active"


class Arm(object):
    """ Universal Robots arm controller """

    def __init__(self,
                 node,
                 namespace: str = None,
                 ik_solver: IKSolverType = IKSolverType.EAIK,
                 gripper_type: GripperType = GripperType.GENERIC,
                 ft_topic: str = None,
                 base_link: str = None,
                 ee_link: str = None,
                 joint_names_prefix: str = None,
                 use_velocity_interface: bool = False,
                 robot_version: str = "UR5e",
                 skip_ros_control: bool = False):
        """

        Parameters
        ----------
        node : rclpy.node.Node
            shared node, assumed to be spun by an executor (e.g. a MultiThreadedExecutor)
            in a background thread. All sub-controllers, service clients and subscriptions
            created here are owned by this node.
        namespace : optional
            ROS namespace of the robot e.g., '/ns/robot_description'
        ik_solver : optional
            inverse kinematic solver to be used
        gripper_type : optional
            gripper control approach. Generic or specific for real Robotiq gripper
        ft_topic: optional
            topic from which to read the wrench
        base_link: optional
            robot base frame. Excluding the prefix used in 'joint_names_prefix'
        ee_link: optional
            end-effector frame. Excluding the prefix used in 'joint_names_prefix'
        joint_names_prefix: optional
            optionally specify a prefix when multiple robots are defined. For example,
            if 'a_bot' is defined, all joints will be consider like 'a_bot_link_name'
        use_velocity_interface: optional
            use velocity interface instead of position interface
        robot_version: optional
            robot version. Currently supported: 'UR5e'
        Raises
        ------
        ValueError
            If a non-supported gripper type or ik solver is defined

        """

        self.node = node
        self.ns = utils.solve_namespace(namespace, node=node)

        base_link = utils.resolve_parameter(value=base_link, default_value=BASE_LINK)
        ee_link = utils.resolve_parameter(value=ee_link, default_value=EE_LINK)

        # Support for joint prefixes
        self.joint_names_prefix = utils.resolve_parameter(joint_names_prefix, '')
        self.base_link = base_link if joint_names_prefix is None else joint_names_prefix + base_link
        self.ee_link = ee_link if joint_names_prefix is None else joint_names_prefix + ee_link

        self.ik_solver = ik_solver

        self.ft_topic = utils.resolve_parameter(value=ft_topic, default_value=FT_SUBSCRIBER)
        self.current_ft_value = np.zeros(6)
        self.wrench_queue = collections.deque(maxlen=25)  # store history of FT data

        # self.max_joint_speed = np.deg2rad([100, 100, 100, 200, 200, 200]) # deg/s -> rad/s
        self.max_joint_speed = np.deg2rad([191, 191, 191, 371, 371, 371])

        self.use_velocity_interface = use_velocity_interface
        self.skip_ros_control = skip_ros_control

        self.node.get_logger().info("Initializing ur robot with parameters")
        self.node.get_logger().info("gripper: {}, ft_sensor_topic: {}, base_link: {}, ee_link: {}"
                                     .format(gripper_type, self.ft_topic, self.base_link, self.ee_link))

        self.controller_manager = ControllersConnection(self.node, self.ns)
        self.dashboard_services = URServices(self.node, self.ns)

        self.__init_controllers__(gripper_type, joint_names_prefix, robot_version)
        self.__init_ik_solver__(self.base_link, self.ee_link)

        self.__init_ft_sensor__()

        # ROS 2 has no rospy.on_shutdown; register on the node's context instead.
        try:
            self.node.context.on_shutdown(self.__on_shutdown__)
        except Exception:
            pass

### private methods ###

    def __on_shutdown__(self):
        try:
            if self.use_velocity_interface:
                # self.joint_vel_controller.stop_all_joints()
                self.activate_joint_trajectory_controller()
            self.joint_traj_controller.stop()
        except Exception:
            pass

    def __init_controllers__(self, gripper_type, joint_names_prefix=None, robot_version="UR5e"):
        self.joint_names = None if joint_names_prefix is None else get_arm_joint_names(joint_names_prefix)

        self.running_controller_name = None
        if not self.skip_ros_control:
            if self.dashboard_services.activate_ros_control_on_ur():
                in_conflict_controllers = [JOINT_POSITION_TRAJECTORY_CONTROLLER, JOINT_VELOCITY_TRAJECTORY_CONTROLLER, CARTESIAN_COMPLIANCE_CONTROLLER]
                for controller in in_conflict_controllers:
                    # get_controller_state raises if the controller is not loaded; treat that as "not active".
                    try:
                        state = self.controller_manager.get_controller_state(controller)
                    except ValueError:
                        continue
                    if state == _ACTIVE:
                        self.running_controller_name = controller
                        break

        self.joint_traj_controller_name = JOINT_VELOCITY_TRAJECTORY_CONTROLLER if self.use_velocity_interface else JOINT_POSITION_TRAJECTORY_CONTROLLER
        self.joint_traj_controller = JointTrajectoryController(self.node,
                                                               publisher_name=self.joint_traj_controller_name,
                                                               namespace=self.ns,
                                                               joint_names=self.joint_names,
                                                               timeout=1.0)

        if self.use_velocity_interface:
            self.joint_vel_controller = JointVelocityController(self.node,
                                                                controller_name=VELOCITY_CONTROLLER_NAME,
                                                                namespace=self.ns,
                                                                joint_names=self.joint_names,
                                                                robot_version=robot_version,
                                                                timeout=1.0)

        self.gripper = None

        if not gripper_type:
            self.node.get_logger().warn("Loading without gripper")
            return

        if gripper_type == GripperType.GENERIC:
            from ur_control.grippers import GripperController
            self.gripper = GripperController(self.node, namespace=self.ns, prefix=self.joint_names_prefix, timeout=2.0)
        elif gripper_type == GripperType.ROBOTIQ:
            from ur_control.grippers import RobotiqGripper
            self.gripper = RobotiqGripper(self.node, namespace=self.ns, prefix=self.joint_names_prefix, timeout=2.0)
        else:
            raise ValueError("Invalid gripper type %s" % gripper_type)

    def __init_ik_solver__(self, base_link, ee_link):
        # ROS 2 has no global parameter server; fetch the URDF from the
        # /robot_description topic (transient-local) and share it with every solver.
        robot_description = get_robot_description(self.node)
        if not robot_description:
            raise ValueError("robot_description not available on the /robot_description topic")
        self._robot_description = robot_description

        # Instantiate KDL kinematics solver to compute forward kinematics
        self.kdl = ur_kinematics(base_link=base_link, ee_link=ee_link, robot_description=robot_description)

        # Instantiate inverse kinematics solver (KDL FK is always available above).
        self.eaik = None
        if self.ik_solver == IKSolverType.EAIK:
            try:
                self.eaik = EAIKKinematics(self.kdl, logger=self.node.get_logger(),
                                           robot_description=robot_description)
            except (ImportError, ValueError) as exc:
                self.node.get_logger().warn(
                    "EAIK unavailable ({}); falling back to KDL IK solver".format(exc))
                self.ik_solver = IKSolverType.KDL
        elif self.ik_solver == IKSolverType.KDL:
            pass
        else:
            raise ValueError("unsupported ik_solver: %s" % self.ik_solver)

    def __init_ft_sensor__(self):
        # Subscriber of wrench. Use best-effort sensor QoS (compatible with both
        # reliable and best-effort publishers).
        filtered_topic = self.ns + self.ft_topic + '/filtered'
        if not utils.topic_exist(self.node, filtered_topic):
            self.node.get_logger().warn("Filtered FT topic not found. Using raw sensor directly.")
            # Try the raw FT topic
            raw_topic = self.ns + self.ft_topic
            self._ft_sub = self.node.create_subscription(WrenchStamped, raw_topic, self.__ft_callback__, qos_profile_sensor_data)
            self._zero_ft_filtered = lambda: None
            self._ft_filtered = lambda active=True: None
        else:
            self._ft_sub = self.node.create_subscription(WrenchStamped, filtered_topic, self.__ft_callback__, qos_profile_sensor_data)

            # Only bind the (synchronous) service calls if the service is actually up:
            # client.call() has no timeout and blocks forever against a missing service.
            zero_ft_filtered_client = self.node.create_client(Empty, filtered_topic + '/zero_ftsensor')
            if zero_ft_filtered_client.wait_for_service(timeout_sec=2.0):
                self._zero_ft_filtered = lambda: zero_ft_filtered_client.call(Empty.Request())
            else:
                self.node.get_logger().warn('%s unavailable; FT zeroing disabled' % zero_ft_filtered_client.srv_name)
                self._zero_ft_filtered = lambda: None

            ft_filter_client = self.node.create_client(SetBool, filtered_topic + '/enable_filtering')
            if ft_filter_client.wait_for_service(timeout_sec=1.0):
                self._ft_filtered = lambda active=True: ft_filter_client.call(SetBool.Request(data=bool(active)))
            else:
                self._ft_filtered = lambda active=True: None

            # Check that the FT topic is publishing
            if not utils.wait_for(lambda: self.current_ft_value is not None, timeout=2.0):
                self.node.get_logger().error('Timed out waiting for {0} topic'.format(filtered_topic))

        # use_gazebo_sim is a node parameter (ROS 2 has no global param server).
        self.use_gazebo_sim = bool(utils.read_parameter(self.node, "use_gazebo_sim", False))
        if not self.use_gazebo_sim:
            zero_ft_client = self.node.create_client(Trigger, self.ns + 'ur_hardware_interface/zero_ftsensor')
            # Guard the blocking call: without the UR driver (e.g. gz sim mistakenly run with
            # use_gazebo_sim=false) this service never exists and client.call() hangs forever.
            if zero_ft_client.wait_for_service(timeout_sec=2.0):
                self._zero_ft = lambda: zero_ft_client.call(Trigger.Request())
            else:
                self.node.get_logger().warn('%s unavailable; skipping hardware FT zeroing' % zero_ft_client.srv_name)
                self._zero_ft = lambda: None
        else:
            self._zero_ft = lambda: None

    def __ft_callback__(self, msg):
        self.current_ft_value = conversions.from_wrench(msg.wrench)
        self.wrench_queue.append(self.current_ft_value)

    def activate_joint_velocity_controller(self):
        """
        Activate the joint velocity controller.

        Returns:
            bool: True if the controller was activated successfully, False otherwise
        """
        if not self.use_velocity_interface or self.running_controller_name == VELOCITY_CONTROLLER_NAME:
            return True
        if self.controller_manager.switch_controllers(controllers_on=[VELOCITY_CONTROLLER_NAME],
                                                      controllers_off=[self.running_controller_name]):
            self.running_controller_name = VELOCITY_CONTROLLER_NAME
            return True
        return False

    def activate_joint_trajectory_controller(self):
        """
        Activate the joint trajectory controller.

        Returns:
            bool: True if the controller was activated successfully, False otherwise
        """
        if self.running_controller_name == self.joint_traj_controller_name:
            return True

        if self.controller_manager.switch_controllers(controllers_on=[self.joint_traj_controller_name],
                                                      controllers_off=[self.running_controller_name]):
            self.running_controller_name = self.joint_traj_controller_name
            return True
        return False


### Data access methods ###

    def inverse_kinematics(self,
                           pose: np.ndarray,
                           seed: np.ndarray = None,
                           attempts: int = 0,
                           verbose: bool = True) -> np.ndarray:
        """
        return a joint configuration for a given Cartesian pose of the end-effector
        (ee_link) if any.

        Parameters
        ----------
        pose :
            Cartesian pose of the end-effector defined as ee_link
        seed : optional
            if given, attempt to return a joint configuration closer to the seed
        attempts : int, optional
            number of attempts to find an IK solution. Retries only help the
            numerical KDL solver; analytical EAIK returns the same result.
        verbose : bool, optional
            print a warning message when IK solutions are not found

        Returns
        -------
        res : ndarray
            Joint configuration if any or None

        Raises
        ------
        ValueError
            If the rot_type is different from 'quaternion' or 'euler'
        """
        q_guess_ = seed if seed is not None else self.joint_angles()

        if self.ik_solver == IKSolverType.EAIK:
            ik = self.eaik.inverse_kinematics(pose, seed=q_guess_)
        elif self.ik_solver == IKSolverType.KDL:
            ik = self.kdl.inverse_kinematics(pose[:3], pose[3:], seed=q_guess_)

        if ik is None:
            if attempts > 0:
                return self.inverse_kinematics(pose, seed, attempts-1)
            if verbose:
                self.node.get_logger().warn(f"{self.ik_solver}: solution not found!")
            raise InverseKinematicsException(f"{self.ik_solver}: solution not found for pose {pose}!")
        return ik

    def end_effector(self,
                     joint_angles=None,
                     rot_type='quaternion',
                     tip_link=None) -> np.ndarray:
        """
        Return the Cartesian pose of the end-effector in the robot base frame (base_link).

        Parameters
        ----------
        joint_angles : ndarray, optional
            If not given, the current joint configuration will be used.
            If provided, the joint configuration is expected in the order given by constants.JOINT_ORDER
        rot_type : str, optional
            Rotation representation to be returned.
            Valid types "quaternion" or "euler"
        tip_link : str, optional
            Return the Cartesian pose of the tip_link if provided.
            Otherwise, use the default ee_link

        Returns
        -------
        res : ndarray
            The Cartesian pose in the form of
            quaternion: [x, y, z, aw, ax, ay, az] or
            euler: [x, y, z, roll, pitch, yaw]
            in radians.

        Raises
        ------
        ValueError
            If the rot_type is different from 'quaternion' or 'euler'
        """

        joint_angles = self.joint_angles() if joint_angles is None else joint_angles

        if rot_type == 'quaternion':
            # forward kinematics
            return self.kdl.forward(joint_angles, tip_link)

        elif rot_type == 'euler':
            x = self.end_effector(joint_angles, tip_link=tip_link)
            euler = np.array(transformations.euler_from_quaternion(x[3:], axes='sxyz'))
            return np.concatenate((x[:3], euler))
        elif rot_type == 'ortho6':
            x = self.end_effector(joint_angles, tip_link=tip_link)
            ortho6 = np.array(transformations.ortho6_from_quaternion(x[3:]))
            return np.concatenate((x[:3], ortho6))
        else:
            raise ValueError("Rotation Type not supported", rot_type)

    def end_effector_velocity(self,
                              joint_angles=None,
                              joint_velocities=None,
                              tip_link=None) -> np.ndarray:
        """
        Return the Cartesian velocity of the end-effector in the robot base frame (base_link).

        Parameters
        ----------
        joint_angles : ndarray, optional
            If not given, the current joint configuration will be used.
            If provided, the joint configuration is expected in the order given by constants.JOINT_ORDER
        joint_velocities : ndarray, optional
            If not given, the current joint velocities will be used.
            If provided, the joint velocities are expected in the order given by constants.JOINT_ORDER
        tip_link : str, optional
            Return the Cartesian velocity of the tip_link if provided.
            Otherwise, use the default ee_link

        Returns
        -------
        res : ndarray
            The Cartesian velocity in the form of [vx, vy, vz, wx, wy, wz]
            where vx,vy,vz are linear velocities and wx,wy,wz are angular velocities
        """
        joint_angles = self.joint_angles() if joint_angles is None else joint_angles
        joint_velocities = self.joint_velocities() if joint_velocities is None else joint_velocities

        # Compute the Jacobian matrix at current joint configuration
        end_effector_vel = self.kdl.forward_velocity(joint_angles, joint_velocities, tip_link)

        return end_effector_vel

    def joint_angle(self, joint: str) -> float:
        """
        Return the requested joint angle in radians.
        """
        joint_idx = self.joint_traj_controller.valid_joint_names.index(joint)
        return self.joint_traj_controller.get_joint_positions()[joint_idx]

    def joint_angles(self) -> np.ndarray:
        """
        Returns the current joint positions in radians and
        in the order given by constants.JOINT_ORDER.
        """
        return self.joint_traj_controller.get_joint_positions()

    def joint_velocity(self, joint: str) -> float:
        """
        Return the requested joint velocity in radians/secs.
        """
        joint_idx = self.joint_traj_controller.valid_joint_names.index(joint)
        return self.joint_traj_controller.get_joint_velocities()[joint_idx]

    def joint_velocities(self) -> np.ndarray:
        """
        Returns the current joint velocities.
        """
        return self.joint_traj_controller.get_joint_velocities()

    def joint_effort(self, joint: str) -> float:
        """
        Return the requested joint effort.
        """
        joint_idx = self.joint_traj_controller.valid_joint_names.index(joint)
        return self.joint_traj_controller.get_joint_efforts()[joint_idx]

    def joint_efforts(self) -> np.ndarray:
        """
        Returns the current joint efforts.
        """
        return self.joint_traj_controller.get_joint_efforts()

    def get_wrench_history(self, hist_size=24, hand_frame_control=False):
        if self.current_ft_value is None:
            raise Exception("FT Sensor not initialized")

        ft_hist = np.array(self.wrench_queue)[:hist_size]

        if hand_frame_control:
            q_hist = self.joint_traj_controller.get_joint_positions_hist()[:hist_size]
            poses_hist = [self.end_effector(q, tip_link=self.ee_link) for q in q_hist]
            wrench_hist = [spalg.convert_wrench(wft, p).tolist() for p, wft in zip(poses_hist, ft_hist)]
        else:
            wrench_hist = ft_hist

        return np.array(wrench_hist)

    def get_wrench(self,
                   base_frame_control=False,
                   hand_frame_control=False) -> np.ndarray:
        """
        Returns the wrench (force/torque) in task-space.
        By default, return the wrench as read from the sensor topic.

        Parameters
        ----------
        base_frame_control : bool, optional
            If True, returns the wrench with respect to the robot base frame
        hand_frame_control : bool, optional
            If True, returns the wrench with respect to the end-effector frame
            If both base_frame_control and hand_frame_control are set to True.
            the former is considered.
        Returns
        -------
        res : np.ndarray
            Returns the wrench in the requested frame
        """
        if self.current_ft_value is None:
            # No values have been received from sensor's topic
            return np.zeros(6)

        wrench_force = self.current_ft_value
        if not hand_frame_control and not base_frame_control:
            return wrench_force

        if base_frame_control:
            # Transform force/torque from sensor to robot base frame
            transform = self.end_effector(tip_link=self.joint_names_prefix + "wrist_3_link")
            ee_wrench_force = spalg.convert_wrench(wrench_force, transform)

            return ee_wrench_force
        else:
            # Transform force/torque from sensor to end effector frame
            transform = self.end_effector(tip_link=self.ee_link)
            ee_wrench_force = spalg.convert_wrench(wrench_force, transform)

            return ee_wrench_force

### Control Methods ###

    def set_joint_positions(self,
                            target_time: float,
                            positions: np.ndarray,
                            velocities: np.ndarray = None,
                            accelerations: np.ndarray = None,
                            wait: bool = False) -> ExecutionResult:
        """
        Run the joint trajectory controller towards a single waypoint starting now.

        Parameters
        ----------
        target_time : float
            time at which target joint should be reach. It can be understood as the
            duration of the trajectory.
        positions : numpy.ndarray
            target joint configuration in the order given by constants.JOINT_ORDER
        velocities : numpy.ndarray, optional
            target joint velocities
        accelerations : numpy.ndarray, optional
            target joint accelerations
        wait : bool, optional
            whether to block code execution until the trajectory is completed or not.

        Returns
        -------
        res : bool
            True if the trajectory is successful when waiting for the execution to be
            completed. Otherwise returns true if the trajectory was started.
        """
        self.activate_joint_trajectory_controller()
        self.joint_traj_controller.add_point(positions=positions,
                                             velocities=velocities,
                                             accelerations=accelerations,
                                             target_time=target_time)
        if wait:
            self.joint_traj_controller.start(delay=0, wait=True)
        else:
            self.joint_traj_controller.start_no_action_server()

        # Always clear the trajectory goal
        self.joint_traj_controller.clear_points()

        if wait:
            res = self.joint_traj_controller.get_result()
            return ExecutionResult.DONE if res.error_code == 0 else ExecutionResult.CONTROLLER_FAILED
        return ExecutionResult.DONE

    def set_joint_velocities(self,
                             velocities: np.ndarray):
        """
        Set the joint velocities.
        """
        self.activate_joint_velocity_controller()
        self.joint_vel_controller.set_joint_velocities(velocities)

    def set_joint_trajectory(self,
                             target_time: float,
                             trajectory: np.ndarray,
                             velocities: np.ndarray = None,
                             accelerations: np.ndarray = None,
                             wait: bool = False) -> ExecutionResult:
        """
        Start the joint trajectory controller with a multi-waypoint trajectory.

        Parameters
        ----------
        target_time : float
            time at which target joint should be reach. It can be understood as the
            duration of the trajectory.
        positions : 2-D numpy.ndarray
            list of target joint configuration for each waypoint in the order given by constants.JOINT_ORDER
        velocities : 2-D numpy.ndarray, optional
            list of target joint velocities for each waypoint
        accelerations : 2-D numpy.ndarray, optional
            list of target joint accelerations for each waypoint
        wait : bool, optional
            whether to block code execution until the trajectory is completed or not.

        Returns
        -------
        res : bool
            True if the trajectory is succesfully executed.
        """
        dt = target_time/len(trajectory)

        for i in range(len(trajectory)):
            self.joint_traj_controller.add_point(positions=trajectory[i],
                                                 target_time=(i+1) * dt,
                                                 velocities=velocities[i] if velocities is not None else None,
                                                 accelerations=accelerations[i] if accelerations is not None else None)
        if wait:
            self.joint_traj_controller.start(delay=0, wait=True)
        else:
            self.joint_traj_controller.start_no_action_server()

        self.joint_traj_controller.clear_points()

        if wait:
            res = self.joint_traj_controller.get_result()
            return ExecutionResult.DONE if res.error_code == 0 else ExecutionResult.CONTROLLER_FAILED
        return ExecutionResult.DONE

    def set_target_pose(self,
                        target_time: float,
                        pose: np.ndarray,
                        wait: bool = False) -> ExecutionResult:
        """
        Reach a given target cartesian pose with the end-effector (ee_link)

        Parameters
        ----------
        target_time : float
            time at which target joint should be reach. It can be understood as the
            duration of the trajectory.
        pose : numpy.ndarray
            Cartesian target pose. Only the quaternion representation is supported
             in the form: [x, y, z, aw, ax, ay, az]
        wait : bool, optional
            whether to block code execution until the trajectory is completed or not.

        Returns
        -------
        res : bool
            True if the trajectory is succesfully executed.
        """
        q = self.inverse_kinematics(pose)
        if q is None:
            self.node.get_logger().debug("IK not found")
            raise InverseKinematicsException("IK solver failed to find a solution")
        else:
            return self.set_joint_positions(positions=q, target_time=target_time, wait=wait)

    def set_pose_trajectory(self,
                            target_time: float,
                            trajectory: np.ndarray) -> ExecutionResult:
        """
        Reach a given target cartesian pose with the end-effector (ee_link)

        Parameters
        ----------
        target_time : float
            time at which target joint should be reach. It can be understood as the
            duration of the trajectory.
        pose : numpy.ndarray
            Cartesian target pose. Only the quaternion representation is supported
             in the form: [x, y, z, aw, ax, ay, az]
        wait : bool, optional
            whether to block code execution until the trajectory is completed or not.

        Returns
        -------
        res : bool or str
            True if the trajectory is succesfully executed.
            If the IK solver fails, return "ik_not_found"
        """
        joint_trajectory = []
        previous_q = self.joint_angles()
        for i, pose in enumerate(trajectory):
            q = self.inverse_kinematics(pose, seed=previous_q)
            if q is None:
                raise InverseKinematicsException("IK solver failed to find a solution")

            previous_q = q
            joint_trajectory.append(q)
        return self.set_joint_trajectory(trajectory=joint_trajectory, target_time=target_time)

    def move_relative(self,
                      target_time: float,
                      transformation: np.array,
                      relative_to_tcp: bool = True,
                      wait: bool = True) -> ExecutionResult:
        """
        Move end-effector (ee_link) relative to its current position

        Parameters
        ----------
        target_time : float
            time at which target joint should be reach. It can be understood as the
            duration of the trajectory.
        pose : numpy.ndarray
            Cartesian target pose. Only the quaternion representation is supported
             in the form: [x, y, z, aw, ax, ay, az]
        relative_to_tcp : bool, optional
            if True, consider the current position relative to the end-effector frame.
            if False, consider the current position relative to the robot base frame.
        wait : bool, optional
            whether to block code execution until the trajectory is completed or not.

        Returns
        -------
        res : bool or str
            True if the trajectory is successfully executed.
            If the IK solver fails, return "ik_not_found"
        """
        new_pose = transformations.transform_pose(self.end_effector(), transformation, rotated_frame=relative_to_tcp)
        return self.set_target_pose(pose=new_pose, target_time=target_time, wait=wait)

    def wait_for_robot_to_stop(self, wait_time=5):
        """
        Wait for the robot to stop moving.

        Args:
            wait_time (float, optional): The maximum time to wait in seconds. Defaults to 5.
        """
        remaining_time = wait_time
        # Monotonic wall-clock (ROS 2 has no global rospy clock here).
        start_time = time.monotonic()

        prev_state = self.joint_angles()

        no_motion_count = 0

        period = 1.0 / 500

        while remaining_time > 0 and no_motion_count < 10:
            time.sleep(period)
            remaining_time = wait_time - (time.monotonic() - start_time)
            curr_state = self.joint_angles()
            if np.allclose(prev_state, curr_state, atol=0.0001):
                no_motion_count += 1
            else:
                no_motion_count = 0
                prev_state = curr_state.copy()

    def stop_robot(self):
        if self.use_velocity_interface:
            self.set_joint_velocities(velocities=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.wait_for_robot_to_stop()
        else:
            self.set_joint_positions(positions=self.joint_angles(), target_time=0.1, wait=True)

### FT sensor control ###

    def zero_ft_sensor(self, sleep_time=0.05):
        """
        Reset force-torque sensor readings to zeros.
        """
        if not self.use_gazebo_sim:
            # First try to zero FT from ur_driver
            self._zero_ft()
            time.sleep(sleep_time)
        # Then update filtered one
        self._zero_ft_filtered()
        time.sleep(sleep_time)

    def set_ft_filtering(self, active=True):
        """
        Enable/disable a low-pass filter.
        If active, the readings returned from self.get_wrench will have been filtered.
        otherwise, the raw data from the sensor's topic will be returned.

        The filtering is done in an external topic. See ur_control_examples ft_filter node.
        """
        self._ft_filtered(active)
