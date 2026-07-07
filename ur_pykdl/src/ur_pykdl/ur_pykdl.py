#!/usr/bin/python

# Copyright (c) 2013-2014, Rethink Robotics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the Rethink Robotics nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import os
import re
import time

import numpy as np
import PyKDL

import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory

from ur_kdl.kdl_parser import kdl_tree_from_urdf_model
from urdf_parser_py.urdf import URDF
from ur_control import transformations

# Set constants for joints
SHOULDER_PAN_JOINT = 'shoulder_pan_joint'
SHOULDER_LIFT_JOINT = 'shoulder_lift_joint'
ELBOW_JOINT = 'elbow_joint'
WRIST_1_JOINT = 'wrist_1_joint'
WRIST_2_JOINT = 'wrist_2_joint'
WRIST_3_JOINT = 'wrist_3_joint'

# Set constants for links
BASE_LINK = 'base_link'
SHOULDER_LINK = 'shoulder_link'
UPPER_ARM_LINK = 'upper_arm_link'
FOREARM_LINK = 'forearm_link'
WRIST_1_LINK = 'wrist_1_link'
WRIST_2_LINK = 'wrist_2_link'
WRIST_3_LINK = 'wrist_3_link'
EE_LINK = 'ur3_robotiq_85_gripper'
EE_LINK = 'ee_link'

# Only edit these when editing the robot joints and links.
# The lengths of these arrays define numerous parameters in GPS.
JOINT_ORDER = [SHOULDER_PAN_JOINT, SHOULDER_LIFT_JOINT, ELBOW_JOINT,
               WRIST_1_JOINT, WRIST_2_JOINT, WRIST_3_JOINT]
LINK_NAMES = [BASE_LINK, SHOULDER_LINK, UPPER_ARM_LINK, FOREARM_LINK,
              WRIST_1_LINK, WRIST_2_LINK, WRIST_3_LINK]


def frame_to_list(frame):
    pos = frame.p
    rot = PyKDL.Rotation(frame.M)
    rot = rot.GetQuaternion()
    return np.array([pos[0], pos[1], pos[2],
                     rot[0], rot[1], rot[2], rot[3]])


def get_robot_description(node, topic='/robot_description', timeout=10.0):
    """Fetch the URDF string from the (transient-local) robot_description topic.

    Replaces ROS 1's ``URDF.from_parameter_server()`` (ROS 2 has no global parameter
    server). Requires ``node`` to be spun by an executor in a background thread.
    """
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST)
    holder = {}
    sub = node.create_subscription(String, topic, lambda m: holder.setdefault('urdf', m.data), qos)
    start = time.time()
    while 'urdf' not in holder and (time.time() - start) < timeout and rclpy.ok():
        time.sleep(0.05)
    node.destroy_subscription(sub)
    return holder.get('urdf')


def _parse_urdf(xml_string):
    """Parse a URDF string into a urdf_parser_py model.

    Strips a leading ``<?xml ... ?>`` declaration first: urdf_parser_py feeds the
    string to lxml, which rejects unicode strings that carry an encoding declaration.
    """
    xml_string = re.sub(r"<\?xml[^>]*\?>", "", xml_string, count=1).lstrip()
    return URDF.from_xml_string(xml_string)


class ur_kinematics(object):
    """
    UR Kinematics with PyKDL
    """

    def __init__(self, base_link=None, ee_link=None, robot=None, prefix=None,
                 rospackage=None, node=None, robot_description=None):
        # Source the URDF: from a packaged file (robot=...), a URDF string
        # (robot_description=...), or the /robot_description topic (node=...).
        if robot:
            pkg_dir = get_package_share_directory(rospackage if rospackage is not None else 'ur_pykdl')
            TREE_PATH = os.path.join(pkg_dir, 'urdf', robot + '.urdf')
            with open(TREE_PATH, 'r') as f:
                self._ur = _parse_urdf(f.read())
        elif robot_description:
            self._ur = _parse_urdf(robot_description)
        elif node is not None:
            urdf_str = get_robot_description(node)
            if not urdf_str:
                raise RuntimeError("ur_kinematics: could not obtain URDF from /robot_description")
            self._ur = _parse_urdf(urdf_str)
        else:
            raise ValueError("ur_kinematics requires one of: robot (packaged file), "
                             "robot_description (URDF string), or node (/robot_description topic)")

        self._kdl_tree = kdl_tree_from_urdf_model(self._ur)
        self._base_link = BASE_LINK if base_link is None else base_link

        self._tip_link = EE_LINK if ee_link is None else ee_link
        self._tip_frame = PyKDL.Frame()
        self._arm_chain = self._kdl_tree.getChain(self._base_link,
                                                  self._tip_link)

        self.chain_dict = {f'{self._base_link}-{self._tip_link}': self._arm_chain}

        # UR Interface Limb Instances
        self._joint_names = JOINT_ORDER if prefix is None else [prefix + joint for joint in JOINT_ORDER]
        self._num_jnts = len(self._joint_names)

        # KDL Solvers
        self._fk_p_kdl = PyKDL.ChainFkSolverPos_recursive(self._arm_chain)
        self._fk_v_kdl = PyKDL.ChainFkSolverVel_recursive(self._arm_chain)
        self._ik_v_kdl = PyKDL.ChainIkSolverVel_pinv(self._arm_chain)
        self._ik_p_kdl = PyKDL.ChainIkSolverPos_NR(self._arm_chain,
                                                   self._fk_p_kdl,
                                                   self._ik_v_kdl)
        self._jac_kdl = PyKDL.ChainJntToJacSolver(self._arm_chain)
        self._dyn_kdl = PyKDL.ChainDynParam(self._arm_chain,
                                            PyKDL.Vector.Zero())

    def print_robot_description(self):
        nf_joints = 0
        for j in self._ur.joints:
            if j.type != 'fixed':
                nf_joints += 1
        print(("URDF non-fixed joints: %d;" % nf_joints))
        print(("URDF total joints: %d" % len(self._ur.joints)))
        print(("URDF links: %d" % len(self._ur.links)))
        print(("KDL joints: %d" % self._kdl_tree.getNrOfJoints()))
        print(("KDL segments: %d" % self._kdl_tree.getNrOfSegments()))

    def print_kdl_chain(self):
        for idx in range(self._arm_chain.getNrOfSegments()):
            print(('* ' + self._arm_chain.getSegment(idx).getName()))

    def joints_to_kdl(self, type, values):
        kdl_array = PyKDL.JntArray(self._num_jnts)

        cur_type_values = values

        for idx in range(self._num_jnts):
            kdl_array[idx] = cur_type_values[idx]
        if type == 'velocities':
            kdl_array = PyKDL.JntArrayVel(kdl_array)
        return kdl_array

    def kdl_to_mat(self, data):
        mat = np.mat(np.zeros((data.rows(), data.columns())))
        for i in range(data.rows()):
            for j in range(data.columns()):
                mat[i, j] = data[i, j]
        return mat

    def end_effector_transform(self, joint_values, tip_link=None):
        pose = self.forward(joint_values, tip_link)
        translation = np.array([pose[:3]])
        transform = transformations.rotation_matrix_from_quaternion(pose[3:])
        transform[:3, 3] = translation
        return transform

    def forward(self, joint_values, tip_link=None):
        if not tip_link or tip_link == self._tip_link:
            return self.forward_position_kinematics(joint_values)

        chain_key = f'{self._base_link}-{tip_link}'
        arm_chain = self.chain_dict.get(chain_key, None)

        if arm_chain is None:
            arm_chain = self._kdl_tree.getChain(self._base_link, tip_link)
            self.chain_dict.update({chain_key: arm_chain})
        fk_p_kdl = PyKDL.ChainFkSolverPos_recursive(arm_chain)
        end_frame = PyKDL.Frame()
        fk_p_kdl.JntToCart(self.joints_to_kdl('positions', joint_values),
                           end_frame)
        return frame_to_list(end_frame)

    def forward_position_kinematics(self, joint_values):
        end_frame = PyKDL.Frame()
        self._fk_p_kdl.JntToCart(self.joints_to_kdl('positions', joint_values),
                                 end_frame)
        return frame_to_list(end_frame)

    def forward_velocity(self, joint_positions, joint_velocities, tip_link=None):
        if not tip_link or tip_link == self._tip_link:
            return self.forward_velocity_kinematics(joint_positions, joint_velocities)

        chain_key = f'{self._base_link}-{tip_link}'
        arm_chain = self.chain_dict.get(chain_key, None)

        if arm_chain is None:
            arm_chain = self._kdl_tree.getChain(self._base_link, tip_link)
            self.chain_dict.update({chain_key: arm_chain})
        fk_v_kdl = PyKDL.ChainFkSolverVel_recursive(arm_chain)
        end_frame = PyKDL.FrameVel()

        q = PyKDL.JntArray(self._num_jnts)
        qdot = PyKDL.JntArray(self._num_jnts)
        for idx in range(self._num_jnts):
            q[idx] = joint_positions[idx]
            qdot[idx] = joint_velocities[idx]
        kdl_joint_vel = PyKDL.JntArrayVel(q, qdot)

        fk_v_kdl.JntToCart(kdl_joint_vel, end_frame)

        twist = end_frame.GetTwist()
        return [twist.vel[0], twist.vel[1], twist.vel[2], twist.rot[0], twist.rot[1], twist.rot[2]]

    def forward_velocity_kinematics(self, joint_positions, joint_velocities):
        end_frame = PyKDL.FrameVel()

        q = PyKDL.JntArray(self._num_jnts)
        qdot = PyKDL.JntArray(self._num_jnts)
        for idx in range(self._num_jnts):
            q[idx] = joint_positions[idx]
            qdot[idx] = joint_velocities[idx]
        kdl_joint_vel = PyKDL.JntArrayVel(q, qdot)

        self._fk_v_kdl.JntToCart(kdl_joint_vel, end_frame)

        twist = end_frame.GetTwist()
        return [twist.vel[0], twist.vel[1], twist.vel[2], twist.rot[0], twist.rot[1], twist.rot[2]]

    def inverse_kinematics(self, position, orientation=None, seed=None):
        pos = PyKDL.Vector(position[0], position[1], position[2])
        if isinstance(orientation, (np.ndarray, np.generic, list)):
            rot = PyKDL.Rotation()
            rot = rot.Quaternion(orientation[0], orientation[1],
                                 orientation[2], orientation[3])
        # Populate seed with current angles if not provided
        seed_array = PyKDL.JntArray(self._num_jnts)
        if seed is not None:
            seed_array.resize(len(seed))
            for idx, jnt in enumerate(seed):
                seed_array[idx] = jnt

        # Make IK Call
        if orientation is not None:
            goal_pose = PyKDL.Frame(rot, pos)
        else:
            goal_pose = PyKDL.Frame(pos)
        result_angles = PyKDL.JntArray(self._num_jnts)

        if self._ik_p_kdl.CartToJnt(seed_array, goal_pose, result_angles) >= 0:
            result = np.array(list(result_angles))
            return result
        else:
            return None

    def jacobian(self, joint_values=None):
        jacobian = PyKDL.Jacobian(self._num_jnts)
        self._jac_kdl.JntToJac(self.joints_to_kdl('positions', joint_values), jacobian)
        return self.kdl_to_mat(jacobian)

    def jacobian_transpose(self, joint_values=None):
        return self.jacobian(joint_values).T

    def jacobian_pseudo_inverse(self, joint_values=None):
        return np.linalg.pinv(self.jacobian(joint_values))

    def inertia(self, joint_values=None):
        inertia = PyKDL.JntSpaceInertiaMatrix(self._num_jnts)
        self._dyn_kdl.JntToMass(self.joints_to_kdl('positions', joint_values), inertia)
        return self.kdl_to_mat(inertia)

    def cart_inertia(self, joint_values=None):
        js_inertia = self.inertia(joint_values)
        jacobian = self.jacobian(joint_values)
        return np.linalg.inv(jacobian * np.linalg.inv(js_inertia) * jacobian.T)
