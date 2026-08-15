# Affordance-Handover-HRI

Semantic-guided robot-to-human object handover — ensures objects land functional-side-first
in the receiver's hand, validated by a custom affordance metric.

A dual-UR3 Gazebo simulation stands in for one robot arm handing an object to a human hand
(the second arm acts as a receiver-hand proxy, since simulating a realistic human hand is out
of scope). A vision-language model (Gemini) reasons about *where* on the object to grasp and
*which end* should face the receiver, so that e.g. a hammer arrives handle-first rather than
head-first. A custom **Affordance GT** metric — computed from the object's real Gazebo pose and
the receiving fingertip's real TF position, invisible to the system at runtime — measures
whether that actually happened, and ablation studies quantify what the semantic layer buys you.

<!-- TODO: 有demo影片/GIF的話放在這裡 -->

## Why this exists

A naive success/fail metric ("did the handover complete") doesn't capture *where* the object
landed. Adding a semantic reasoning layer on top of a purely geometric grasp pipeline can even
*lower* raw success rate (more moving parts, more failure modes) while *improving* the thing
that actually matters for a human receiver: getting handed the correct end of the tool. The
Affordance GT metric exists to make that trade-off measurable instead of anecdotal.

## Architecture

Four independently-run ROS nodes, plus this control script, communicate over topics:

| Role | File | What it does |
|---|---|---|
| Arm control / orchestration | [`ur_control/scripts/simple_grasp_controller.py`](ur_control/scripts/simple_grasp_controller.py) | Drives the whole mission: trigger detection, grasp, in-air handover, retreat. Also computes the Affordance GT and HOE (Hand-Object-Error orientation) metrics post-hoc. |
| Semantic reasoning + grasp region selection | [`semantic_layer/brain.py`](semantic_layer/brain.py) | OWL-v2 (zero-shot detection) → Gemini (which grid cells = giver/receiver region, functional-end vs geometric strategy) → SAM (precise segmentation mask). |
| Grasp pose generation | [`semantic_layer/anygrasp_ros.py`](semantic_layer/anygrasp_ros.py) | Feeds the segmented point cloud to [AnyGrasp](https://github.com/graspnet/anygrasp_sdk) for 6-DoF grasp candidates, per arm. |
| Pose completion | [`pose_completion/foundationpose_node.py`](pose_completion/foundationpose_node.py) | [FoundationPose](https://github.com/NVlabs/FoundationPose) estimates full object pose from a partial view during re-detection at the handover zone. |

The `ros_ur3` package tree (everything outside `semantic_layer/` and `pose_completion/`) is a
fork of [cambel/ur3](https://github.com/cambel/ur3) (`noetic-devel` branch), extended to a
dual-arm UR3 + Robotiq 85 setup with YCB object spawning, an affordance-axis calibration module,
and the metrics described below.

### The Affordance GT metric

For each object, its mesh's PCA principal axis is calibrated once (`ur_control/scripts/affordance_gt.py`)
into a 0–1 position range with a boundary separating "handle" from "functional end", plus which
side is the handle. At handover time, the real left-fingertip TF position (captured right before
the gripper closes) is projected onto that axis using the object's **real Gazebo ground-truth
pose** (not the FoundationPose estimate — this keeps pose-estimation error from contaminating the
metric) to get a HIT/MISS. The system never sees this value; it's computed and logged purely for
post-hoc evaluation.

## Results

Ablation comparing the full semantic pipeline against a `no-llm` baseline (OWL-v2 + SAM only,
no Gemini reasoning, so the giver-side grasp region is the *entire* segmented object instead of
a semantically-chosen sub-region) — first 10 attempts per condition:

| Object | Mode | GSR (grasp) | HSR (handover) | TSR (task) | Affordance HIT rate |
|---|---|---:|---:|---:|---:|
| hammer | full-system | 40% | 10% | 40% | — |
| hammer | no-llm | 90% | 70% | 70% | 70% |
| scissors | full-system | 80% | 70% | 70% | — |
| scissors | no-llm | 60% | 40% | 60% | 80% |
| spatula | full-system | 50% | 40% | 40% | — |
| spatula | no-llm | 10% | 0% | 0% | — |
| large_clamp | full-system | 60% | 40% | 50% | — |
| large_clamp | no-llm | 0% | 0% | 0% | — |

Full per-object CSVs and raw trial logs are in [`ur_control/scripts/affordance_experiments/`](ur_control/scripts/affordance_experiments/).

**Finding**: for objects with roughly uniform grip geometry along their length (hammer, scissors),
removing semantic guidance barely hurts — or even helps — raw grasp success, because AnyGrasp's
candidates are diverse enough to include a good handle grasp regardless. For objects with a sharp
width transition (spatula's thin handle vs. wide blade, large_clamp's arms vs. jaw hinge), the
dense cluster of candidates on the wide/complex end statistically crowds the sparse handle
candidates out of the top-K AnyGrasp ever tries — without semantic region-narrowing, the system
can go from "usually succeeds" to "essentially never grasps the object at all," not just a lower
success rate. The semantic layer's value here isn't only picking the *correct* affordance end —
it's also what makes basic grasp feasibility reliable for non-uniform objects in the first place.

## Setup

This is a large, multi-repo research stack. Expect to spend real time on this — it is **not**
a `pip install` away.

### 1. ROS / simulation base

- Ubuntu 20.04, ROS Noetic, Gazebo 11
- This repo (`ros_ur3` fork) plus these sibling packages under the same catkin workspace `src/`:
  - [`robotiq`](https://github.com/cambel/robotiq) — gripper description + Gazebo plugin
  - [`universal_robot`](https://github.com/ros-industrial/universal_robot) — for the `ur_description` package (arm meshes)
  - [`trac_ik`](https://bitbucket.org/traclabs/trac_ik) — IK solver (`ur_control`'s package.xml depends on `trac_ik_python`)
  - [`gazebo_ros_link_attacher`](https://github.com/pal-robotics/gazebo_ros_link_attacher) — simulated rigid grasp attach/detach
  - [`realsense-ros`](https://github.com/IntelRealSense/realsense-ros) + [`realsense_gazebo_plugin`](https://github.com/pal-robotics/realsense_gazebo_plugin) — simulated camera
- `catkin build` the workspace, `source devel/setup.bash`

### 2. YCB object models

The object mesh/SDF folders under `ur_gripper_gazebo/models/` are **not included** in this repo
(third-party YCB dataset assets). Download the ones you need from the
[YCB Object Set](https://www.ycbbenchmarks.com/object-models/) — the `google_16k` variant — and
place each under `ur_gripper_gazebo/models/<NNN>_<name>/`, matching the entries in
`OBJECT_SDF` / `OBJECT_MESH_MAP` in `random_object_placement.py` and `simple_grasp_controller.py`.
An SDF + `.material` + `model.config` needs to accompany each mesh (see any existing non-YCB
model folder for the format, e.g. `ur_gripper_gazebo/models/floor/`).

### 3. Semantic layer (AnyGrasp + Gemini)

- Follow [`semantic_layer/install_anygrasp.txt`](semantic_layer/install_anygrasp.txt) to set up
  the `anygrasp_ros` conda environment (PyTorch, MinkowskiEngine, pointnet2, graspnetAPI).
- **AnyGrasp requires its own academic license** — register at the
  [AnyGrasp SDK repo](https://github.com/graspnet/anygrasp_sdk) to get a license file. It cannot
  be shared; each user needs their own.
- A separate `lang-sam` conda env runs `brain.py` (needs `transformers`, `torch`, `google-generativeai`).
- Set a Gemini API key (get one free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)):
  ```bash
  export GEMINI_API_KEY="your-key-here"   # add to ~/.bashrc to persist
  ```

### 4. Pose completion (FoundationPose)

Follow FoundationPose's own [installation instructions](https://github.com/NVlabs/FoundationPose)
(it's typically run inside a Docker container with its own conda env). `pose_completion/foundationpose_node.py`
in this repo still has this project's original absolute paths (`/home/rvl/ros_ws/...`) — update
`FP_ROOT`, `YCB_MODELS_ROOT`, `GRIPPER_MESH_DIR`, `SAVE_DIR` near the top of the file to match
your own workspace layout before running it.

### 5. Running it

Each of the four nodes above runs in its own environment/terminal:

```bash
# 1) simulation
roslaunch ur_gripper_gazebo ur_gripper_85_dual_arm.launch
roslaunch ur_gripper_85_moveit_config start_sim_dual_ur3e_moveit.launch

# 2) semantic layer (conda env: lang-sam)
python3 semantic_layer/brain.py

# 3) grasp detection (conda env: anygrasp_ros)
python3 semantic_layer/anygrasp_ros.py

# 4) pose completion (inside the FoundationPose docker container)
python3 pose_completion/foundationpose_node.py

# 5) place an object, then run a mission
python3 ur_control/scripts/random_object_placement.py hammer
python3 ur_control/scripts/simple_grasp_controller.py
# → prompts for a natural-language object description, then runs the full pipeline
```

`ur_control/scripts/check_nodes.sh` / `restart_node.sh` / `restart_node_bg.sh` are convenience
scripts for polling node liveness and restarting individual nodes without restarting everything.

## Attribution / licensing

- The `ros_ur3` package tree is forked from [cambel/ur3](https://github.com/cambel/ur3) (MIT
  License, © Cristian Beltran) — see `LICENSE`.
- Grasp pose generation via [AnyGrasp](https://github.com/graspnet/anygrasp_sdk) (academic
  license, register separately).
- Pose completion via [FoundationPose](https://github.com/NVlabs/FoundationPose) (NVlabs).
- Object detection/segmentation via [OWL-v2](https://huggingface.co/google/owlv2-base-patch16-ensemble)
  and [SAM](https://huggingface.co/facebook/sam-vit-base) (Hugging Face `transformers`).
- Semantic reasoning via Google Gemini.

This repo contains only the code written for this research (dual-arm control logic, the
semantic-layer glue code, the affordance metric, and the ablation experiment tooling) — not the
upstream AnyGrasp / FoundationPose codebases themselves.
