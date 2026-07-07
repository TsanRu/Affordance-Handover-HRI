# The MIT License (MIT)
#
# Copyright (c) 2018, 2019 Cristian Beltran
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

import ur_control.transformations as tr
import numpy as np
from ur_control import transformations


def spiral(radius, theta_offset, revolutions, steps):
    theta = np.linspace(0, 2*np.pi*revolutions, steps) + theta_offset
    x = radius*np.cos(theta)
    y = radius*np.sin(theta)
    return x, y


def get_conical_helix_trajectory(p1, p2, steps, revolutions=5.0):
    """ Compute Cartesian conical helix between 2 points"""
    euclidean_dist = np.linalg.norm(np.array(p2[:2])-np.array(p1[:2]))
    radius = np.linspace(euclidean_dist, 0, steps)
    theta_offset = np.arctan2((p1[1] - p2[1]), (p1[0]-p2[0]))

    x, y = spiral(radius, theta_offset, revolutions, steps)
    x += p2[0]
    y += p2[1]
    z = np.linspace(p1[2]-(p1[2]-p2[2])/2, p2[2], steps)
    return concat_vec(x, y, z, steps)


def get_spiral_trajectory(p1, p2, steps, revolutions=5.0, from_center=False, inverse=False):
    """ Compute Cartesian conical helix between 2 points"""
    euclidean_dist = np.linalg.norm(np.array(p2[:2])-np.array(p1[:2]))
    if from_center:  # start the spiral as if p1 is the center and p2 is the farthest point
        radius = np.linspace(0, euclidean_dist, steps)
        theta_offset = 0.0
    else:
        # Compute the distance from p1 to p2 and start the spiral as if p2 is the center
        radius = np.linspace(euclidean_dist, 0, steps)
        # Hack for some reason this offset does not work for changes w.r.t Z
        if inverse:
            theta_offset = np.arctan2((p2[1] - p1[1]), (p2[0]-p1[0]))
        else:
            theta_offset = np.arctan2((p1[1] - p2[1]), (p1[0]-p2[0]))
    x, y = spiral(radius, theta_offset, revolutions, steps)
    sign = 1. if not inverse else -1.  # super hack, there is something fundamentally wrong with Z
    x += p2[0] * sign
    y += p2[1] * sign
    z = np.linspace(p1[2]-(p1[2]-p2[2])/2, p2[2], steps)
    return concat_vec(x, y, z, steps)


def get_circular_trajectory(p1, p2, steps, revolutions=1.0, from_center=False, inverse=False):
    euclidean_dist = np.linalg.norm(np.array(p2[:2])-np.array(p1[:2]))
    if from_center:  # start the spiral as if p1 is the center and p2 is the farthest point
        theta_offset = 0.0
    else:
        # Compute the distance from p1 to p2 and start the spiral as if p2 is the center
        # Hack for some reason this offset does not work for changes w.r.t Z
        if inverse:
            theta_offset = np.arctan2((p2[1] - p1[1]), (p2[0]-p1[0]))
        else:
            theta_offset = np.arctan2((p1[1] - p2[1]), (p1[0]-p2[0]))
    x, y = spiral(euclidean_dist, theta_offset, revolutions, steps)
    x += p2[0]
    y += p2[1]
    z = np.zeros(steps)+p1[2]
    return concat_vec(x, y, z, steps)


def concat_vec(x, y, z, steps):
    x = x.reshape(-1, steps)
    y = y.reshape(-1, steps)
    z = z.reshape(-1, steps)
    return np.concatenate((x, y, z), axis=0).T


def get_plane_direction(plane, radius):
    VALID_DIRECTIONS = ('+X', '+Y', '+Z', '-X', '-Y', '-Z')
    DIRECTION_INDEX = {'X': 0, 'Y': 1, 'Z': 2}

    assert plane in VALID_DIRECTIONS, "Invalid direction: %s" % plane

    direction_array = [0., 0., 0., 0., 0., 0.]
    sign = 1. if '+' in plane else -1.
    direction_array[DIRECTION_INDEX.get(plane[1])] = radius * sign

    return np.array(direction_array, dtype=np.float64)


def compute_rotation_wiggle(initial_orientation, direction, angle, steps, revolutions):
    """
        Compute a sinusoidal trajectory for the orientation of the end-effector in one given direction.
        To keep it simple, only supports one direction.
        initial_orientation: array[4], orientation in quaternion form
        direction: string, 'X','Y', or 'Z' w.r.t the robot's base
        angle: float, magnitude of rotation in radians
        steps: int, number of steps for the resulting trajectory
        revolutions: int, number of revolutions 
    """
    assert direction in ('X', 'Y', 'Z'), "Invalid direction: %s" % direction
    DIRECTION_INDEX = {'X': 0, 'Y': 1, 'Z': 2}

    euler = np.array(transformations.euler_from_quaternion(initial_orientation, axes='rxyz'))
    theta = np.linspace(0, 2*np.pi*revolutions, steps)
    deltas = angle * np.sin(theta)

    direction_array = np.zeros(3)
    direction_array[DIRECTION_INDEX.get(direction)] = 1.0
    deltas = deltas.reshape(-1, 1) * direction_array.reshape(1, 3)

    new_eulers = deltas + euler

    cmd_orientations = [transformations.quaternion_from_euler(*new_euler, axes='rxyz') for new_euler in new_eulers]

    return cmd_orientations


def compute_trajectory(initial_pose, plane, radius, radius_direction, steps=100, revolutions=5, from_center=True,  trajectory_type="circular",
                       wiggle_direction=None, wiggle_angle=0.0, wiggle_revolutions=0.0):
    """
        Compute a trajectory "circular" or "spiral":
        plane: string, only 3 valid options "XY", "XZ", "YZ", is the plane w.r.t to the robot base where the trajectory will be drawn
        radius: float, size of the trajectory
        radius_direction: string, '+X', '+Y', '+Z', '-X', '-Y', '-Z' direction to compute the radius, valid directions depend on the plane selected
        steps: int, number of steps for the trajectory
        revolutions: int, number of times that the circle is drawn or the spiral's revolutions before reaching its end.
        from_center: bool, whether to start the trajectory assuming the current position as center (True) or the radius+radius_direction as center (False)
                            [True] is better for spiral trajectory while [False] is better for the circular trajectory, though other options are okay.
        trajectory_type: string, "circular" or "spiral"
        wiggle_direction: string, 'X','Y', or 'Z' w.r.t the robot's base
        wiggle_angle: float, magnitude of wiggle-rotation in radians
        wiggle_revolutions: int, number of wiggle-revolutions 
    """
    from pyquaternion import Quaternion

    direction = get_plane_direction(radius_direction, radius)

    if plane == "XZ":
        assert "Y" not in radius_direction, "Invalid radius direction %s for plane %s" % (radius_direction, plane)
        to_plane = [np.pi/2, 0, 0]
    elif plane == "YZ":
        assert "X" not in radius_direction, "Invalid radius direction %s for plane %s" % (radius_direction, plane)
        to_plane = [0, np.pi/2, 0]
    elif plane == "XY":
        assert "Z" not in radius_direction, "Invalid radius direction %s for plane %s" % (radius_direction, plane)
        to_plane = [0, 0, 0]
    else:
        raise ValueError("Invalid value for plane: %s" % plane)

    target_pose = transformations.transform_pose(initial_pose, direction, rotated_frame=False)

    # print("Initial", np.round(spalg.translation_rotation_error(target_pose, arm.end_effector()), 4))
    aux_orientation = transformations.quaternion_from_euler(*to_plane)

    target_orientation = Quaternion(np.roll(aux_orientation, 1))

    initial_pose = initial_pose[:3]
    final_pose = target_pose[:3]

    if from_center:
        p1 = np.zeros(3)
        p2 = transformations.quaternion_rotate_vector(target_orientation, initial_pose - final_pose)
    else:
        p1 = transformations.quaternion_rotate_vector(target_orientation, initial_pose - final_pose)
        p2 = np.zeros(3)

    if trajectory_type == "circular":
        # Hack for some reason this offset does not work for changes w.r.t Z
        traj = get_circular_trajectory(p1, p2, steps, revolutions, from_center=from_center, inverse=("Z" in radius_direction))
    elif trajectory_type == "spiral":
        # Hack for some reason this offset does not work for changes w.r.t Z
        traj = get_spiral_trajectory(p1, p2, steps, revolutions, from_center=from_center, inverse=("Z" in radius_direction))
    else:
        raise ValueError("Unsupported trajectory type: %s" % trajectory_type)

    traj = np.apply_along_axis(transformations.quaternion_rotate_vector, 1, target_orientation, traj)
    trajectory = traj + final_pose

    if wiggle_direction is None:
        trajectory = [np.concatenate([t, target_pose[3:]]) for t in trajectory]
    else:
        target_orientation = compute_rotation_wiggle(target_pose[3:], wiggle_direction, wiggle_angle, steps, wiggle_revolutions)
        trajectory = [np.concatenate([tp, to]) for tp, to in zip(trajectory, target_orientation)]

    return np.array(trajectory)


def compute_1d_sinusoidal_trajectory(num_of_points, amplitude=0.01, period=1):
    points = np.linspace(-np.pi, np.pi, num_of_points)
    trajectory = np.sin(points * period) * amplitude
    return trajectory


def compute_sinusoidal_trajectory(current_pose, dimension: int, num_of_points=30, amplitude=0.01, period=1):
    trajectory_1d = compute_1d_sinusoidal_trajectory(num_of_points, amplitude, period)
    trajectory = np.array([current_pose for _ in range(num_of_points)]).reshape((-1, 7))
    trajectory[:, dimension] = trajectory_1d + current_pose[dimension]
    return trajectory


def generate_mortar_trajectory(mortar_diameter, desired_height, n_steps, default_quat=np.array([0, -1, 0, 0]), fraction=None):
    """
    Generate a trajectory to trace the surface of an upward-facing bowl at a given height.
    The pen orientation at the center (0,0,0) is represented by quaternion [0,-1,0,0].

    Args:
        mortar_diameter (float): Diameter of the bowl in meters
        desired_height (float): Desired height from the bottom of the bowl in meters
        n_steps (int): Number of points in the trajectory
        default_quat (list): Quaternion [qx, qy, qz, qw] representing orientation at the bowl center at (0,0,0)
        fraction: (float): If defined, the final quaternion returned is the slerp fraction from the default_quat to the 
                           corresponding normal vector

    Returns:
        np.array: Array of shape (n_steps, 7) containing [x, y, z, qx, qy, qz, qw]
                 for each point in the trajectory
    """
    # Step 1: Calculate bowl parameters
    radius = mortar_diameter / 2

    # Step 2: Verify if desired height is valid
    if desired_height > radius:
        raise ValueError("Desired height cannot be greater than bowl radius")

    # Step 3: Calculate radius of the circle at desired height
    # For upward facing bowl: r^2 = R^2 - (R-h)^2
    circle_radius = np.sqrt(radius**2 - (radius - desired_height)**2)

    # Step 4: Generate points along a circle at the desired height
    theta = np.linspace(0, 2*np.pi, n_steps)
    x = circle_radius * np.cos(theta)
    y = circle_radius * np.sin(theta)
    z = np.full_like(theta, desired_height)

    # Step 5: Calculate normal vectors at each point
    # For an upward facing bowl, the normal vector points outward from the center of curvature
    normals = np.zeros((n_steps, 3))
    for i in range(n_steps):
        point = np.array([x[i], y[i], z[i] - radius])  # Shift center of curvature to (0,0,-R)
        normal = -point / np.linalg.norm(point)  # Normalize and negate for outward normal
        normals[i] = normal

    # Step 6: Convert normal vectors to quaternions considering initial orientation
    quaternions = np.zeros((n_steps, 4))

    initial_rotation = tr.rotation_matrix_from_quaternion(default_quat)[:3, :3]

    for i in range(n_steps):
        normal = normals[i]

        # Calculate rotation from [0, 0, 1] to normal vector
        z_axis = np.array([0, 0, 1])
        v = np.cross(z_axis, normal)
        s = np.linalg.norm(v)

        if s < 1e-10:  # If vectors are parallel
            if normal[2] > 0:  # Same direction
                surface_rotation = tr.rotation_matrix_from_quaternion([0, 0, 0, 1])
            else:  # Opposite direction
                surface_rotation = tr.rotation_matrix_from_quaternion([1, 0, 0, 0])
        else:
            c = np.dot(z_axis, normal)
            v_skew = np.array([[0, -v[2], v[1]],
                               [v[2], 0, -v[0]],
                               [-v[1], v[0], 0]])
            R_matrix = np.eye(3) + v_skew + np.matmul(v_skew, v_skew) * (1 - c) / (s * s)
            surface_rotation = R_matrix

        # Compose rotations: first apply initial orientation, then surface normal rotation
        final_rotation = surface_rotation @ initial_rotation
        final_quaternion = tr.quaternion_from_matrix(final_rotation)
        if fraction:
            final_quaternion = tr.quaternion_slerp(default_quat, final_quaternion, fraction=fraction)
        quaternions[i] = final_quaternion

    # Step 7: Combine positions and orientations
    trajectory = np.column_stack((x, y, z, quaternions))
    trajectory = np.concatenate([trajectory, [trajectory[0]]])

    return trajectory
