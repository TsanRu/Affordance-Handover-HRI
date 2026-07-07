# ROS utilities used by the CRI group
#! /usr/bin/env python
import os
import sys
import copy
import time
import numpy as np
import quaternion

import rclpy
from ament_index_python.packages import get_package_share_directory

from ur_control import transformations, spalg
from ur_control.log import TextColors
from sensor_msgs.msg import JointState


def load_urdf_string(package, filename):
    package_dir = get_package_share_directory(package)
    urdf_file = os.path.join(package_dir, 'urdf', filename + '.urdf')
    with open(urdf_file) as f:
        urdf = f.read()
    return urdf


class PDRotation:
    def __init__(self, kp, kd=None):
        self.kp = np.array(kp)
        self.kd = np.array(kd)
        self.reset()

    def reset(self):
        # Monotonic wall-clock seconds (float). ROS 2 has no global rospy clock;
        # pass dt explicitly to update() when sim-time-accurate timing is required.
        self.last_time = time.monotonic()
        self.last_error = np.quaternion(1, 0, 0, 0)

    def set_gains(self, kp=None, kd=None):
        if kp is not None:
            self.kp = np.array(kp)
        if kd is not None:
            self.kd = np.array(kd)

    def update(self, quaternion_error, dt=None):
        now = time.monotonic()
        if dt is None:
            dt = now - self.last_time

        k_prime = 2 * quaternion_error.w*np.identity(3)-spalg.skew(quaternion_error.vec)
        p_term = np.dot(self.kp, k_prime)

        # delta_error = quaternion_error - self.last_error
        w = transformations.angular_velocity_from_quaternions(quaternion_error, self.last_error, dt)
        d_term = self.kd * w

        output = p_term + d_term
        # Save last values
        self.last_error = quaternion_error
        self.last_time = now
        return output

class PID:
    def __init__(self, Kp, Ki=None, Kd=None, dynamic_pid=False, max_gain_multiplier=10.0):
        # Proportional gain. Gains are real-valued; cast to float so the integral
        # term stays float64 and supports in-place accumulation (`integral += error*dt`).
        self.Kp = np.array(Kp, dtype=float)
        self.Ki = np.zeros_like(self.Kp)
        self.Kd = np.zeros_like(self.Kp)
        # Integral gain
        if Ki is not None:
            self.Ki = np.array(Ki, dtype=float)
        # Derivative gain
        if Kd is not None:
            self.Kd = np.array(Kd, dtype=float)
        self.set_windup(np.ones_like(self.Kp))
        # Reset
        self.reset()
        self.scale_gains = dynamic_pid
        self.max_gain_multiplier = max_gain_multiplier

    def reset(self):
        # Monotonic wall-clock seconds (float); see PDRotation.reset note.
        self.last_time = time.monotonic()
        self.last_error = np.zeros_like(self.Kp)
        self.integral = np.zeros_like(self.Kp)

    def set_gains(self, Kp=None, Ki=None, Kd=None):
        if Kp is not None:
            self.Kp = np.array(Kp, dtype=float)
        if Ki is not None:
            self.Ki = np.array(Ki, dtype=float)
        if Kd is not None:
            self.Kd = np.array(Kd, dtype=float)

    def set_windup(self, windup):
        self.i_min = -np.array(windup)
        self.i_max = np.array(windup)

    def update(self, error, dt=None):
        # CAUTION: naive scaling of the Kp parameter based on the error
        # The main idea, the smaller the error the higher the gain
        if self.scale_gains:
            kp = np.zeros_like(self.Kp)
            kd = np.zeros_like(self.Kd)

            for i in range(len(error)):
                # from position_error < 0.01m increase scale error
                factor = 1 - np.tanh(100 * error[i])
                kp[i] = np.interp(factor, [0.0, 1.0], [self.Kp[i], self.Kp[i] * self.max_gain_multiplier])
                kd[i] = np.interp(factor, [0.0, 1.0], [self.Kd[i], self.Kd[i] * self.max_gain_multiplier])

            kd = self.Kd
            ki = self.Ki
        else:
            kp = self.Kp
            kd = self.Kd
            ki = self.Ki

        now = time.monotonic()
        if dt is None:
            dt = now - self.last_time
        delta_error = error - self.last_error
        # Compute terms
        self.integral += error * dt
        p_term = kp * error
        i_term = ki * self.integral
        i_term = np.maximum(self.i_min, np.minimum(i_term, self.i_max))

        # First delta error is huge since it was initialized at zero first, avoid considering
        if not np.allclose(self.last_error, np.zeros_like(self.last_error)):
            d_term = kd * delta_error / dt
        else:
            d_term = kd * np.zeros_like(delta_error) / dt

        output = p_term + i_term + d_term
        # Save last values
        self.last_error = np.array(error)
        self.last_time = now
        return output


# TextColors is the roscore-independent console logger; defined once in ur_control.log
# and re-exported here for backwards compatibility (ur_control.utils.TextColors).


## Helper Functions ##
def assert_shape(variable, name, shape):
    """
    Asserts the shape of an np.array
    @type  variable: Object
    @param variable: variable to be asserted
    @type  name: string
    @param name: variable name
    @type  shape: tuple
    @param ttype: expected shape of the np.array
    """
    assert variable.shape == shape, '%s must have a shape %r: %r' % (name, shape, variable.shape)


def assert_type(variable, name, ttype):
    """
    Asserts the type of a variable with a given name
    @type  variable: Object
    @param variable: variable to be asserted
    @type  name: string
    @param name: variable name
    @type  ttype: Type
    @param ttype: expected variable type
    """
    assert type(variable) is ttype,  '%s must be of type %r: %r' % (name, ttype, type(variable))


def db_error_msg(name, logger=TextColors()):
    """
    Prints out an error message appending the given database name.
    @type  name: string
    @param name: database name
    @type  logger: Object
    @param logger: Logger instance.
    """
    msg = 'Database %s not found. Please generate it. [rosrun denso_openrave generate_databases.py]' % name
    logger.logerr(msg)


def clean_cos(value):
    """
    Limits the a value between the range C{[-1, 1]}
    @type value: float
    @param value: The input value
    @rtype: float
    @return: The limited value in the range C{[-1, 1]}
    """
    return min(1, max(value, -1))


def has_keys(data, keys):
    """
    Checks whether a dictionary has all the given keys.
    @type   data: dict
    @param  data: Parameter name
    @type   keys: list
    @param  keys: list containing the expected keys to be found in the dict.
    @rtype: bool
    @return: True if all the keys are found in the dict, false otherwise.
    """
    if not isinstance(data, dict):
        return False
    has_all = True
    for key in keys:
        if key not in data:
            has_all = False
            break
    return has_all


def raise_not_implemented():
    """
    Raises a NotImplementedError exception
    """
    raise NotImplementedError()


def topic_exist(node, topic):
    """Whether C{topic} is currently advertised, as seen by C{node}."""
    topic_names = [name for name, _ in node.get_topic_names_and_types()]
    return topic in topic_names

def read_key(echo=False):
    """
    Reads a key from the keyboard
    @type   echo: bool, optional
    @param  echo: if set, will show the input key in the console.
    @rtype: str
    @return: The limited value in the range C{[-1, 1]}
    """
    if not echo:
        os.system("stty -echo")
    key = sys.stdin.read(1)
    if not echo:
        os.system("stty echo")
    return key.lower()

def resolve_parameter(value, default_value):
    if value:
        return value
    else:
        return default_value

def read_parameter(node, name, default):
    """
    Read a parameter from C{node}, declaring it with C{default} if undeclared.

    NOTE: ROS 2 parameters are node-scoped (there is no global parameter server),
    so 'global' resources such as robot_description are fetched from topics, not here.
    @type  node: rclpy.node.Node
    @type  name: string
    @param name: Parameter name (a flat name, not a slashed global path)
    @param default: Default value used to declare the parameter if missing.
    @return: The resulting parameter value (or C{default}).
    """
    from rcl_interfaces.msg import ParameterDescriptor
    if not node.has_parameter(name):
        node.declare_parameter(name, default, ParameterDescriptor(dynamic_typing=True))
    value = node.get_parameter(name).value
    return default if value is None else value


def read_parameter_err(node, name):
    """
    Read a parameter from C{node}. If it is not declared/set, log an error.
    @rtype: (bool, any)
    @return: (found, value); value is None when not found.
    """
    if not node.has_parameter(name):
        node.get_logger().error("Parameter [%s] not found" % (name))
        return False, None
    return True, node.get_parameter(name).value


def read_parameter_fatal(node, name):
    """
    Read a required parameter from C{node}; raise if it is not declared/set.
    @rtype: any
    @return: The resulting parameter value.
    """
    if not node.has_parameter(name):
        node.get_logger().fatal("Parameter [%s] not found" % (name))
        raise Exception('Required parameter {0} not found'.format(name))
    return node.get_parameter(name).value


def solve_namespace(namespace=None, node=None):
    """
    Appends neccessary slashes required for a proper ROS namespace.
    @type namespace: string
    @param namespace: namespace to be fixed.
    @type node: rclpy.node.Node
    @param node: node used to resolve the current namespace when C{namespace} is empty.
    @rtype: string
    @return: Proper ROS namespace.
    """
    if namespace is None or len(namespace) == 0:
        namespace = node.get_namespace() if node is not None else '/'
        if not namespace.endswith('/'):
            namespace += '/'
    elif len(namespace) == 1:
        if namespace != '/':
            namespace = '/' + namespace + '/'
    else:
        if namespace[0] != '/':
            namespace = '/' + namespace
        if namespace[-1] != '/':
            namespace += '/'
    return namespace


def sorted_joint_state_msg(msg, joint_names):
    """
    Returns a sorted C{sensor_msgs/JointState} for the given joint names
    @type  msg: sensor_msgs/JointState
    @param msg: The input message
    @type  joint_names: list
    @param joint_names: The sorted joint names
    @rtype: sensor_msgs/JointState
    @return: The C{JointState} message with the fields in the order given by joint names
    """
    valid_names = set(joint_names).intersection(set(msg.name))
    valid_position = len(msg.name) == len(msg.position)
    valid_velocity = len(msg.name) == len(msg.velocity)
    valid_effort = len(msg.name) == len(msg.effort)
    num_joints = len(valid_names)
    retmsg = JointState()
    retmsg.header = copy.deepcopy(msg.header)
    for name in joint_names:
        if name not in valid_names:
            continue
        idx = msg.name.index(name)
        retmsg.name.append(name)
        if valid_position:
            retmsg.position.append(msg.position[idx])
        if valid_velocity:
            retmsg.velocity.append(msg.velocity[idx])
        if valid_effort:
            retmsg.effort.append(msg.effort[idx])
    return retmsg


def unique(data):
    """
    Finds the unique elements of an array. B{row-wise} and
    returns the sorted unique elements of an array.
    @type  data: np.array
    @param data: Input array.
    @rtype: np.array
    @return: The sorted unique array.
    """
    order = np.lexsort(data.T)
    data = data[order]
    diff = np.diff(data, axis=0)
    ui = np.ones(len(data), 'bool')
    ui[1:] = (diff != 0).any(axis=1)
    return data[ui]


def wait_for(predicate, timeout=5.0):
    start_time = time.time()
    while not predicate():
        now = time.time()
        if (now - start_time) > timeout:
            return False
        time.sleep(0.001)
    return True


class Rate:
    """Minimal ``rospy.Rate`` replacement based on the monotonic wall clock.

    ROS 2 offers ``node.create_rate()``, but that couples to a spinning executor
    and can deadlock when called from a callback thread. This is a dependency-free
    fixed-rate sleeper for the synchronous control loops.

    NOTE: it uses the monotonic wall clock, not the ROS clock, so it is not
    sim-time aware. Pass an explicit ``dt`` to the control models where sim-time
    accuracy matters.
    """

    def __init__(self, hz):
        self._period = 1.0 / float(hz)
        self._next = time.monotonic() + self._period

    def sleep(self):
        now = time.monotonic()
        remaining = self._next - now
        if remaining > 0:
            time.sleep(remaining)
            self._next += self._period
        else:
            # Missed the deadline: resync to "now" to avoid a burst of catch-up sleeps.
            self._next = now + self._period
