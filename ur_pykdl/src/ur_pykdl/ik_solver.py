#!/usr/bin/env python

import rospy
import numpy as np
from ur_pykdl import ur_kinematics
import PyKDL

from ur_pykdl.ur_pykdl import frame_to_list


def kdl_to_np(kdl_array):
    return np.array([kdl_array[i] for i in range(kdl_array.rows())])


class IKSolver(ur_kinematics):
    """IK Solver class that inherits from ur_kinematics and adds additional functionality"""

    def __init__(self, base_link=None, ee_link=None, robot=None, prefix=None, rospackage=None):
        super(IKSolver, self).__init__(base_link=base_link, ee_link=ee_link, robot=robot, prefix=prefix, rospackage=rospackage)

        self.end_effector_pose = PyKDL.Frame()
        self.end_effector_vel = np.zeros(6)  # 6D vector for linear and angular velocity

        # Initialize FK solvers
        self.fk_pos_solver = None
        self.fk_vel_solver = None

        self.number_joints = self._arm_chain.getNrOfJoints()

        # Initialize joint arrays
        self.current_positions = np.zeros(self.number_joints)
        self.current_velocities = np.zeros(self.number_joints)
        self.current_accelerations = np.zeros(self.number_joints)
        self.last_positions = np.zeros(self.number_joints)
        self.last_velocities = np.zeros(self.number_joints)

        # Parse joint limits from URDF model
        self.upper_pos_limits = np.zeros(self.number_joints)
        self.lower_pos_limits = np.zeros(self.number_joints)
        self.velocity_limits = np.zeros(self.number_joints)

        for i, joint_name in enumerate(self._joint_names):
            joint = self._ur.joint_map[joint_name]

            if joint.type == 'continuous':
                # For continuous joints, use NaN to indicate no limits
                self.upper_pos_limits[i] = np.nan
                self.lower_pos_limits[i] = np.nan
            elif joint.limit:
                # Use limits from URDF if defined
                self.upper_pos_limits[i] = joint.limit.upper
                self.lower_pos_limits[i] = joint.limit.lower
            else:
                # Default to no limits if not specified
                self.upper_pos_limits[i] = np.nan
                self.lower_pos_limits[i] = np.nan

            # Set velocity limits
            if joint.limit and joint.limit.velocity:
                self.velocity_limits[i] = joint.limit.velocity
            else:
                self.velocity_limits[i] = 0.0  # No velocity limit defined

        # Initialize solvers
        self.fk_pos_solver = PyKDL.ChainFkSolverPos_recursive(self._arm_chain)
        self.fk_vel_solver = PyKDL.ChainFkSolverVel_recursive(self._arm_chain)

        self._jac_kdl = PyKDL.ChainJntToJacSolver(self._arm_chain)
        self._dyn_kdl = PyKDL.ChainDynParam(self._arm_chain, PyKDL.Vector.Zero())

        rospy.loginfo("Forward dynamics solver initialized")
        rospy.loginfo(f"Forward dynamics solver has control over {self.number_joints} joints")

    def get_end_effector_pose(self):
        """Get the current end effector pose

        Returns:
            KDL.Frame: Current end effector pose
        """
        return frame_to_list(self.end_effector_pose)

    def get_end_effector_vel(self):
        """Get end effector velocity

        Returns:
            numpy.ndarray: 6D vector of linear and angular velocity
        """
        return self.end_effector_vel

    def get_positions(self):
        """Get current joint positions

        Returns:
            KDL.JntArray: Current joint positions
        """
        return self.current_positions

    def get_velocity(self):
        """Get current joint velocities

        Returns:
            KDL.JntArray: Current joint velocities
        """
        return self.current_velocities

    def set_start_state(self, joint_positions, joint_velocities):
        """Set initial joint states

        Args:
            joint_positions: List of joint positions
            joint_velocities: List of joint velocities
        """
        self.current_positions = joint_positions.copy()
        self.current_velocities = joint_velocities.copy()
        self.current_accelerations = np.zeros(self.number_joints)

        self.last_positions = self.current_positions.copy()
        self.last_velocities = self.current_velocities.copy()

    def synchronize_joint_positions(self, joint_positions):
        """Synchronize joint positions with the current hardware state

        Args:
            joint_positions: List of joint handles containing current joint states
        """
        self.current_positions = joint_positions.copy()
        self.last_positions = self.current_positions.copy()

    def update_kinematics(self):
        """Update forward kinematics for current joint states"""
        # Update end effector pose
        self.fk_pos_solver.JntToCart(self.joints_to_kdl('positions', self.current_positions), self.end_effector_pose)

        # Update end effector velocity
        vel_frame = PyKDL.FrameVel()
        joint_array_vel = PyKDL.JntArrayVel(self.joints_to_kdl('positions', self.current_positions),
                                            self.joints_to_kdl('positions', self.current_velocities))
        self.fk_vel_solver.JntToCart(joint_array_vel, vel_frame)

        # Extract linear and angular velocity
        self.end_effector_vel[0] = vel_frame.deriv().vel[0]
        self.end_effector_vel[1] = vel_frame.deriv().vel[1]
        self.end_effector_vel[2] = vel_frame.deriv().vel[2]
        self.end_effector_vel[3] = vel_frame.deriv().rot[0]
        self.end_effector_vel[4] = vel_frame.deriv().rot[1]
        self.end_effector_vel[5] = vel_frame.deriv().rot[2]

    def apply_joint_velocity_limits(self):
        """Apply velocity limits to joint velocities"""
        velocity_scaling_factor = 1.0

        for i in range(self.number_joints):
            # Skip if limit is not defined (0.0)
            if self.velocity_limits[i] == 0:
                continue

            unbounded_velocity = self.current_velocities[i]
            if abs(unbounded_velocity) > 1e-10:  # Avoid division by zero
                # Clamp velocity and compute scaling factor
                bounded_velocity = np.clip(
                    unbounded_velocity,
                    -self.velocity_limits[i],
                    self.velocity_limits[i]
                )
                velocity_scaling_factor = min(
                    velocity_scaling_factor,
                    bounded_velocity / unbounded_velocity
                )

        # if velocity_scaling_factor < 1.0:
        #     print(f"Scaling down joint velocities by a factor of {velocity_scaling_factor}")
        #     print(f"self.current_velocities {self.current_velocities}")
        #     exit(0)

        # Apply scaling
        for i in range(self.number_joints):
            self.current_velocities[i] *= velocity_scaling_factor

    def apply_joint_limits(self):
        """Apply position limits to joint positions"""
        for i in range(self.number_joints):
            if np.isnan(self.lower_pos_limits[i]) or np.isnan(self.upper_pos_limits[i]):
                # Joint is continuous
                continue

            self.current_positions[i] = np.clip(
                self.current_positions[i],
                self.lower_pos_limits[i],
                self.upper_pos_limits[i]
            )

    def build_generic_model(self):
        # Set all masses and inertias to minimal (yet stable) values
        ip_min = 0.000001
        min_mass = 0.1

        for i in range(self._arm_chain.getNrOfSegments()):
            segment = self._arm_chain.getSegment(i)

            # Fixed joint segment
            if segment.getJoint().getType() == None:
                segment.setInertia(PyKDL.RigidBodyInertia.Zero())

            # Relatively moving segment
            else:
                segment.setInertia(PyKDL.RigidBodyInertia(
                    min_mass,  # mass
                    PyKDL.Vector(0, 0, 0),  # center of gravity
                    PyKDL.RotationalInertia(
                        ip_min,  # ixx
                        ip_min,  # iyy
                        ip_min   # izz
                        # ixy, ixy, iyz default to 0.0
                    )
                ))
        # See https://arxiv.org/pdf/1908.06252.pdf for motivation
        m = 1.0
        ip = 1.0

        # Set inertia for last segment
        last_segment = self._arm_chain.getSegment(self._arm_chain.getNrOfSegments()-1)
        last_segment.setInertia(PyKDL.RigidBodyInertia(
            m,
            PyKDL.Vector(0, 0, 0),
            PyKDL.RotationalInertia(ip, ip, ip)
        ))

    def get_joint_control_cmds(self, period, net_force):
        """Compute joint control commands using forward dynamics

        Args:
            period (rospy.Duration): Time period for the control command
            net_force (numpy.ndarray): 6D vector of net force/torque

        Returns:
            trajectory_msgs.JointTrajectoryPoint: Joint trajectory control command
        """
        # Compute joint space inertia matrix with actualized link masses
        self.build_generic_model()
        jnt_space_inertia = PyKDL.JntSpaceInertiaMatrix(self.number_joints)
        self._dyn_kdl.JntToMass(self.joints_to_kdl('positions', self.current_positions), jnt_space_inertia)

        # Compute joint jacobian
        jnt_jacobian = PyKDL.Jacobian(self.number_joints)
        self._jac_kdl.JntToJac(self.joints_to_kdl('positions', self.current_positions), jnt_jacobian)

        # Convert KDL matrices to numpy for computation
        H = np.array([[jnt_space_inertia[i, j] for j in range(self.number_joints)]
                     for i in range(self.number_joints)])
        J = np.array([[jnt_jacobian[i, j] for j in range(self.number_joints)]
                     for i in range(self.number_joints)])

        # Compute joint accelerations: ddot{q} = H^{-1} (J^T f)
        self.current_accelerations = np.linalg.inv(H).dot(J.T.dot(net_force))

        # Numerical time integration with Euler forward method
        dt = period
        self.current_positions = (self.last_positions + self.last_velocities * dt)
        self.current_velocities = (self.last_velocities + self.current_accelerations * dt)

        # 10% global damping against unwanted null space motion
        self.current_velocities *= 0.9

        # Apply joint limits
        # self.apply_joint_velocity_limits()
        self.apply_joint_limits()

        # Create control command
        control_cmd = {
            'positions': self.current_positions,
            'velocities': self.current_velocities
        }

        # Leave accelerations empty as they are interpreted as max tolerances

        # Update for next cycle
        self.last_positions = self.current_positions
        self.last_velocities = self.current_velocities

        return control_cmd
