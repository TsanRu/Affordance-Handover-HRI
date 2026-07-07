# Universal Robot UR / URe — ROS 2 utilities

Custom ROS 2 packages for Universal Robots with Robotiq grippers (2F-85 and Hand-E): control
library, gz-sim bringup, MoveIt 2 config, and examples.

**ROS 2 Jazzy** on **Ubuntu 24.04** with **Gazebo Harmonic (gz-sim 8)**. For ROS 1 (Noetic /
Melodic + Gazebo Classic), use the `noetic-devel` branch.

---

## Packages

| Package | Description |
|---|---|
| `ur_control` | Core Python control library — arm, grippers, FK/IK, compliance / force control |
| `ur_pykdl` | Forward / inverse kinematics via PyKDL |
| `ur_control_examples` | Example nodes (keyboard teleop, FT filter, …) |
| `ur_gripper_gz` | gz-sim bringup for UR + Hand-E or 2F-85 (ros2_control, FT sensor, Cartesian compliance) |
| `ur_gripper_gz_moveit_config` | MoveIt 2 config (arm + gripper), parameterized by `ur_type` and `gripper` |

Simulation is handled by `ur_gripper_gz` plus apt
`robotiq_description` (2F-85 geometry).

---

## Installation

Three ways to get a working workspace: **pixi**
(recommended, no Docker/root required), **Docker**, or a manual colcon workspace.

### With pixi (recommended)

Installs ROS 2 Jazzy and all dependencies into a local, project-scoped
[pixi](https://pixi.sh) environment via [RoboStack](https://robostack.github.io/) — no
Docker, no `sudo` (besides the one-time `rosdep init`), no polluting your system Python/apt.

Requires Linux with glibc ≥ 2.36 (e.g. Ubuntu 24.04 Noble) and
[pixi](https://pixi.sh/latest/#installation) installed.

```bash
pixi install                  # creates .pixi/envs/default with ROS 2 Jazzy + build tools

pixi run rosdep-init           # one-time per machine, requires sudo
pixi run rosdep-update
pixi run rosdep-install        # resolve workspace package.xml deps

pixi run build                 # symlinks this repo's packages + cartesian_controllers
                                # (dependencies.repos) into ws/src/, then colcon builds
```

`pixi run build` skips `cartesian_controller_simulation` / `cartesian_controller_tests`
(they need a separate MuJoCo install, not needed for this repo's own packages).

Then, in every new shell:

```bash
pixi shell
ros2 launch ur_gripper_gz ur_2f85_gz_control.launch.py
```

See [`pixi.toml`](pixi.toml) for the full task list (`build-debug`, `test`, `clean`, …).

### With Docker

ROS 2 Jazzy image with CUDA, MoveIt 2, gz Harmonic, and UR stack pre-installed:

```bash
./docker/BUILD-DOCKER-IMAGE.sh
./docker/RUN-DOCKER.sh

# inside the container — first-time workspace build:
ur-build-workspace
```

The repo is bind-mounted at `/root/ws/src/ur_python_utilities`. Third-party sources
(`cartesian_controllers`, required for compliance control) are imported from
`dependencies.repos` during `ur-build-workspace`.

Set `DOCKER_RUNTIME=runc` on machines without an NVIDIA GPU (Intel GL fallback is handled in the
container entry script).

### Compile from source manually

In a colcon workspace:

```bash
cd /path/to/ws/src
git clone -b jazzy <this-repo> ur_python_utilities
vcs import . < ur_python_utilities/dependencies.repos

cd ..
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install \
  --packages-skip cartesian_controller_simulation cartesian_controller_tests
source install/setup.bash
```

When used inside the parent [osx_robot_env](https://github.com/omron-sinicx/osx_robot_env) repo,
build `underlay_ws` first, then any overlay workspace on top.

---

## Quick start

### Simulation

```bash
# UR5e + Robotiq 2F-85 (gui:=false for headless)
ros2 launch ur_gripper_gz ur_2f85_gz_control.launch.py

# UR + Robotiq Hand-E
ros2 launch ur_gripper_gz ur_gz_control.launch.py
```

Both launches force the **bullet-featherstone** physics engine (required for gripper mimic joints)
and publish the active gripper name on `/active_gripper`.

### Keyboard teleop

In a second terminal (with the workspace sourced):

```bash
ros2 run ur_control_examples joint_position_keyboard
```

`--gripper auto` (default) reads `/active_gripper` from the running bringup. Press **SPACE** for
the command list.

### MoveIt 2

Terminal 1 — sim:

```bash
ros2 launch ur_gripper_gz ur_2f85_gz_control.launch.py gui:=false
# Hand-E: ros2 launch ur_gripper_gz ur_gz_control.launch.py gui:=false
```

Terminal 2 — MoveIt + RViz:

```bash
ros2 launch ur_gripper_gz_moveit_config ur_moveit.launch.py ur_type:=ur5e gripper:=robotiq_2f85
# Hand-E: ur_type:=ur3e gripper:=hande
```

In RViz **MotionPlanning**: plan & execute with group `ur_manipulator` (arm) or `gripper`
(open / close named states).

---

## Features

### `ur_control`

Python library ported from ROS 1 (`rospy` → `rclpy`). All classes take a shared `rclpy.node.Node`
injected by the application; spin a `MultiThreadedExecutor` in a background thread before
constructing `Arm` or gripper clients.

Highlights:

- Arm trajectory / velocity control via ros2_control (`scaled_joint_trajectory_controller`,
  `forward_velocity_controller`)
- Generic and Robotiq gripper drivers (`GripperCommand` and `FollowJointTrajectory` paths)
- FK / IK (KDL + EAIK analytical IK)
- FZI `cartesian_compliance_controller` wrapper (`CompliantController`) with runtime parameter tuning
  via `SetParameters`
- Gripper config registry + `/active_gripper` auto-detection

### `ur_gripper_gz`

Self-contained gz_ros2_control bringups:

- **Hand-E** — single actuated `finger_joint`; right finger follows via URDF `<mimic>`
- **2F-85** — PickNik `robotiq_description` geometry; one actuated knuckle + five mimic joints;
  `GripperActionController` matches `ur_control` `gripper_type="85"`

Both variants include:

- Wrist **force-torque sensor** on `/wrench` (`force_torque_sensor_broadcaster`)
- **`ft_filter`** — Butterworth filter to `/wrench/filtered` plus a zero service for taring distal-mass bias
- **`cartesian_compliance_controller`** (loaded inactive; switched on by `CompliantController`)

Zero the FT sensor at a known no-contact pose before using compliance or force control.

### `ur_gripper_gz_moveit_config`

MoveIt 2 config parameterized by `ur_type` + `gripper`. Includes mandatory SRDF
`disable_collisions` for gripper links and gripper joint acceleration limits. Hand-E full reopen
in gz is limited by the finger model; 2F-85 open/close works end-to-end.

---

## Design notes

1. **Shared node + background executor.** Never call `rclpy.spin_once(node)` on a node the
   executor is already spinning.
2. **Mimic joints need bullet-featherstone.** DART does not support mimic constraints; both gripper
   launches pass `--physics-engine gz-physics-bullet-featherstone-plugin`.
3. **`gz_ros2_control` ignores URDF command limits.** Cap gripper travel in the URDF and/or via
   client `gripper_finger_max_position` to avoid over-close jamming.
4. **TRAC-IK Python is unavailable on Jazzy.** `IKSolverType.TRAC_IK`; using EAIK
   for analytical IK instead.
5. **Controller renames (ROS 1 → 2):** `scaled_pos_joint_traj_controller` →
   `scaled_joint_trajectory_controller`; `joint_group_vel_controller` → `forward_velocity_controller`.

---

## Known limitations

- **FT zeroing** tares distal-mass gravity bias only at the pose where you zero; bias is
  pose-dependent. Model-based gravity compensation would improve robustness across the workspace.
- **Robotiq real-hardware driver** (`robotiq_control` / pymodbus) is deferred; re-port when needed.
- **Hand-E MoveIt reopen** is limited by gz finger fidelity (close works; full open is capped).
- Legacy modules kept off the main import path: `simple_controllers`, `mouse_6d` (spacenav teleop).

---

## License

See [LICENSE](LICENSE).
