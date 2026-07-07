from enum import Enum


class IKSolverType(Enum):
    KDL = 'kdl'
    EAIK = 'eaik'


class GripperType(Enum):
    GENERIC = 'generic'
    ROBOTIQ = 'robotiq'


class ExecutionResult(Enum):
    DONE = 'success'
    CONTROLLER_FAILED = 'controller_failed'
    FORCE_TORQUE_EXCEEDED = 'force_exceeded'
    STOP_ON_TARGET_FORCE = 'stop_on_target_force'
    IK_NOT_FOUND = 'ik_not_found'
    SPEED_LIMIT_EXCEEDED = 'speed_limit_exceeded'
    TERMINATION_CRITERIA = 'termination_criteria_achieved'


CARTESIAN_COMPLIANCE_CONTROLLER = 'cartesian_compliance_controller'
# ROS 2 UR driver (ros2_control) controller names. The ROS 1 scaled *position* and
# scaled *velocity* joint-trajectory controllers collapse into a single
# scaled_joint_trajectory_controller; raw joint velocities are commanded through the
# forward_velocity_controller (a forward_command_controller, not a trajectory action).
JOINT_POSITION_TRAJECTORY_CONTROLLER = 'scaled_joint_trajectory_controller'
JOINT_VELOCITY_TRAJECTORY_CONTROLLER = 'scaled_joint_trajectory_controller'
JOINT_SUBSCRIBER = '/arm_controller/state'
JOINT_STATE_SUBSCRIBER = 'joint_states'
FT_SUBSCRIBER = 'wrench'

VELOCITY_CONTROLLER_NAME = 'forward_velocity_controller'

# Set constants for joints
SHOULDER_PAN_JOINT = 'shoulder_pan_joint'
SHOULDER_LIFT_JOINT = 'shoulder_lift_joint'
ELBOW_JOINT = 'elbow_joint'
WRIST_1_JOINT = 'wrist_1_joint'
WRIST_2_JOINT = 'wrist_2_joint'
WRIST_3_JOINT = 'wrist_3_joint'

BASE_LINK = 'base_link'
EE_LINK = 'tool0'
FT_LINK = 'tool0'

JOINT_ORDER = [
    SHOULDER_PAN_JOINT, SHOULDER_LIFT_JOINT, ELBOW_JOINT, WRIST_1_JOINT,
    WRIST_2_JOINT, WRIST_3_JOINT
]


def get_arm_joint_names(prefix):
    return [prefix + joint for joint in JOINT_ORDER]
