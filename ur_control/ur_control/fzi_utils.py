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

import numpy as np
from ur_control.constants import ExecutionResult


def is_more_extreme(value, target):
    """
    Check if a value is more extreme than a target value.

    For each dimension:
    - If target is positive, check if value is greater than target
    - If target is negative, check if value is less than target

    Args:
        value (numpy.ndarray): The value to check
        target (numpy.ndarray): The target value to compare against

    Returns:
        bool: True if the value is more extreme than the target, False otherwise
    """
    # Check each dimension individually
    for i in range(len(target)):
        # For positive targets, check if value exceeds target
        if target[i] > 0 and value[i] > target[i]:
            return True
        # For negative targets, check if value is more negative than target
        elif target[i] < 0 and value[i] < target[i]:
            return True

    # If we get here, no dimension exceeded its target
    return False


# NOTE (ROS 2): the FZI cartesian_compliance_controller exposes all of its tunables as
# standard ROS 2 node parameters (flat, dot-separated names) instead of ROS 1
# dynamic_reconfigure groups. The converters below return {param_name: value} dicts ready
# for a single SetParameters call. Names match cartesian_controllers' declarations:
#   stiffness.{trans_*,rot_*,sel_*}, pd_gains.<axis>.{p,d}, solver.*, end_effector_link, ...
_AXES = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]


def convert_selection_matrix_to_parameters(selection_matrix):
    """
    Convert a selection matrix to controller parameters.

    Args:
        selection_matrix (numpy.ndarray): A 6-element array representing the selection matrix

    Returns:
        dict: {param_name: value} for stiffness.sel_* parameters
    """
    keys = ["sel_x", "sel_y", "sel_z", "sel_ax", "sel_ay", "sel_az"]
    return {"stiffness.%s" % k: float(selection_matrix[i]) for i, k in enumerate(keys)}


def convert_stiffness_to_parameters(stiffness):
    """
    Convert stiffness values to controller parameters.

    Args:
        stiffness (numpy.ndarray): A 6-element array representing the stiffness values

    Returns:
        dict: {param_name: value} for stiffness.{trans_*,rot_*} parameters
    """
    return {"stiffness.%s" % k: float(stiffness[i]) for i, k in enumerate(_AXES)}


def convert_pd_gains_to_parameters(p_gains, d_gains=[0, 0, 0, 0, 0, 0]):
    """
    Convert P and D gains to controller parameters.

    Args:
        p_gains (numpy.ndarray): A 6-element array representing the P gains
        d_gains (numpy.ndarray, optional): A 6-element array representing the D gains. Defaults to zeros.

    Returns:
        dict: {param_name: value} for pd_gains.<axis>.{p,d} parameters
    """
    parameters = {}
    for i, axis in enumerate(_AXES):
        parameters["pd_gains.%s.p" % axis] = float(p_gains[i])
        parameters["pd_gains.%s.d" % axis] = float(d_gains[i])
    return parameters


def switch_cartesian_controllers(func):
    """
    Decorator that switches from cartesian to joint trajectory controllers and back.

    This decorator ensures that the cartesian controller is activated before the function is called
    and that the joint trajectory controller is activated after the function returns.

    Args:
        func (callable): The function to decorate

    Returns:
        callable: The decorated function
    """
    def wrap(*args, **kwargs):
        if not args[0].auto_switch_controllers:
            return func(*args, **kwargs)

        args[0].activate_cartesian_controller()

        try:
            res = func(*args, **kwargs)
        except Exception as e:
            args[0].node.get_logger().error("Exception: %s" % e)
            res = ExecutionResult.DONE

        args[0].activate_joint_trajectory_controller()

        return res
    return wrap
