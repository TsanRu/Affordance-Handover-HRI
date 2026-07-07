# The MIT License (MIT)
#
# Copyright (c) 2026 Cristian Beltran
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

"""Analytical inverse kinematics via EAIK (https://ostermd.github.io/EAIK/)."""

import io
import os
import re

import numpy as np

from ur_control import transformations

try:
    import eaik.pybindings.EAIK as EAIK
    from eaik.IK_Robot import IKRobot
    from urchin import URDF
except ImportError as exc:
    IKRobot = object
    EAIK = None
    URDF = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _strip_xml_prolog(xml: str) -> str:
    """Remove the XML declaration so lxml accepts in-memory URDF strings."""
    return re.sub(r"<\?xml[^>]*\?>", "", xml, count=1).lstrip()


def _strip_non_urdf_tags(xml: str) -> str:
    """Drop Gazebo / ros2_control extensions that urchin cannot parse."""
    xml = _strip_xml_prolog(xml)
    xml = re.sub(r"<ros2_control[\s\S]*?</ros2_control>", "", xml)
    xml = re.sub(r"<gazebo[\s\S]*?</gazebo>", "", xml)
    xml = re.sub(r"<gazebo[^>]*/>", "", xml)
    return xml


_REVOLUTE_JOINT_TYPES = ("revolute", "continuous")


def _revolute_chain_joints(robot: "URDF", base_link: str, tip_link: str) -> list:
    """Revolute joints on the kinematic chain from base_link to tip_link (root to tip)."""
    child_to_joint = {joint.child: joint for joint in robot.joints}

    chain = []
    link = tip_link
    while link != base_link:
        if link not in child_to_joint:
            raise ValueError(
                "URDF chain broken: link '%s' has no parent joint while walking toward '%s'"
                % (link, base_link))
        joint = child_to_joint[link]
        if joint.joint_type in _REVOLUTE_JOINT_TYPES:
            chain.append(joint)
        link = joint.parent

    chain.reverse()
    if not chain:
        raise ValueError(
            "No revolute joints on chain from '%s' to '%s'" % (base_link, tip_link))
    return chain


def _load_urdf(*, urdf: str = None, file_path: str = None) -> "URDF":
    """Load a URDF model from a file path or an in-memory XML string."""
    if (urdf is None) == (file_path is None):
        raise ValueError("Provide exactly one of urdf or file_path")

    if file_path is not None:
        return URDF.load(file_path, lazy_load_meshes=True)

    buf = io.BytesIO(_strip_non_urdf_tags(urdf).encode())
    buf.name = "robot.urdf"
    return URDF.load(buf, lazy_load_meshes=True)


class UrdfRobot(IKRobot):
    """EAIK robot whose kinematic chain is parsed from a URDF file or string."""

    def __init__(self,
                 urdf: str = None,
                 file_path: str = None,
                 base_link: str = None,
                 tip_link: str = None,
                 fixed_axes: list[tuple[int, float]] = None):
        """
        Parameters
        ----------
        urdf : str, optional
            URDF XML string (e.g. from the ``/robot_description`` topic).
        file_path : str, optional
            Path to a URDF file on disk.
        base_link : str, optional
            Root link of the IK chain. When set together with ``tip_link``, only
            revolute joints on that chain are used (gripper / extra DOFs are
            ignored). Matches the KDL chain used by :class:`ur_kinematics`.
        tip_link : str, optional
            Tip link of the IK chain (e.g. ``tool0``).
        fixed_axes : list[tuple[int, float]], optional
            Fixed joints as zero-indexed ``(joint_index, angle)`` pairs.
        """
        if EAIK is None:
            raise ImportError(
                "EAIK is not installed (pip install EAIK). Original error: %s" % _IMPORT_ERROR)

        if fixed_axes is None:
            fixed_axes = []

        super().__init__()
        robot = _load_urdf(urdf=urdf, file_path=file_path)

        if base_link is not None and tip_link is not None:
            joints = _revolute_chain_joints(robot, base_link, tip_link)
        else:
            joints = robot._sort_joints(robot.actuated_joints)

        fk_zero_pose = robot.link_fk()
        # link_fk() is in the URDF root (world) frame. A fixed world->base offset
        # (e.g. spawn yaw in sim xacros) must not enter the arm chain model.
        if base_link is not None:
            T_base_inv = np.linalg.inv(fk_zero_pose[robot.link_map[base_link]])
        else:
            T_base_inv = np.eye(4)

        parent_p = np.zeros(3)
        H = np.array([], dtype=np.int64).reshape(0, 3)
        P = np.array([], dtype=np.int64).reshape(0, 3)
        for joint in joints:
            joint_child_link = robot.link_map[joint.child]
            T_child = T_base_inv @ fk_zero_pose[joint_child_link]
            h, p = self.urdf_to_sp_conv(T_child, joint.axis, parent_p)
            H = np.vstack([H, h])
            P = np.vstack([P, p])
            parent_p += p

        P = np.vstack([P, np.zeros(3)])
        self._robot = EAIK.Robot(H.T, P.T, np.eye(3), fixed_axes, True)


class EAIKKinematics(object):
    """EAIK analytical IK for a URDF chain, aligned to a KDL tip frame."""

    def __init__(self, kdl, logger=None, robot_description: str = None, file_path: str = None):
        if EAIK is None:
            raise ImportError(
                "EAIK is not installed (pip install EAIK). Original error: %s" % _IMPORT_ERROR)

        self._kdl = kdl
        self._logger = logger
        self._num_joints = kdl._num_jnts

        self._bot = UrdfRobot(urdf=robot_description, file_path=file_path,
                              base_link=kdl._base_link, tip_link=kdl._tip_link)
        if not self._bot.hasKnownDecomposition():
            raise ValueError(
                "EAIK has no known decomposition for this URDF (kinematic family: %s)"
                % self._bot.getKinematicFamily())

        q_zero = np.zeros(self._num_joints)
        try:
            self._bot.fwdKin(q_zero)
        except Exception as exc:
            raise ValueError(
                "EAIK joint count does not match the arm chain (%d joints): %s"
                % (self._num_joints, exc)) from exc

        T_eaik = self._bot.fwdKin(q_zero)
        T_ee = transformations.pose_to_transform(self._kdl.forward(q_zero))
        self._T_eaik_to_ee = np.linalg.inv(T_eaik) @ T_ee

        if logger is not None:
            source = file_path if file_path is not None else "robot_description"
            logger.info(
                "EAIK initialized (family={}, joints={}, source={})".format(
                    self._bot.getKinematicFamily(), self._num_joints, source))

    @classmethod
    def from_file(cls, file_path: str, kdl, logger=None):
        """Construct from a URDF file path."""
        if not os.path.isfile(file_path):
            raise ValueError("URDF file not found: %s" % file_path)
        return cls(kdl, logger=logger, file_path=file_path)

    def inverse_kinematics(self, pose: np.ndarray, seed: np.ndarray = None) -> np.ndarray:
        """Return the IK solution closest to seed, or None if no solution exists."""
        T_target = transformations.pose_to_transform(pose) @ np.linalg.inv(self._T_eaik_to_ee)
        ik = self._bot.IK(T_target)
        solutions = ik.Q
        if solutions is None or len(solutions) == 0:
            return None

        seed = np.zeros(self._num_joints) if seed is None else np.asarray(seed, dtype=float)
        best = min(solutions, key=lambda q: np.linalg.norm(np.asarray(q) - seed))
        return np.asarray(best, dtype=float)
