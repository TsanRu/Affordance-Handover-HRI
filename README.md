# Affordance-Handover-HRI

Semantic-guided robot-to-human object handover — ensures objects land functional-side-first
in the receiver's hand, validated in both a dual-arm Gazebo simulation and a real UR3 handing
objects to a human receiver.

A vision-language model (GPT-5.4, prompted over a Set-of-Mask-style labeled grid) reasons about
*where* on the object to grasp and *which end* should face the receiver, so that e.g. a hammer
arrives handle-first rather than head-first. A PCA-based wrist rotation step then adjusts the
object's delivered orientation to match. The approach is first validated in a dual-UR3 Gazebo
simulation (one arm handing to a second arm acting as a receiver proxy, with a custom
**Affordance GT** metric measuring whether the correct end was actually grasped) and then
deployed to a physical UR3 handing objects to a real human hand.

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
| Arm control / orchestration + orientation adjustment | [`ur_control/scripts/simple_grasp_controller.py`](ur_control/scripts/simple_grasp_controller.py) | Drives the whole mission: trigger detection, grasp, in-air handover, retreat. After the giver arm grasps, runs a **PCA-based wrist rotation** step — computing the object's principal axis from its point cloud and rotating the wrist so the functional end faces the receiver — before moving to the handover zone. Also computes the Affordance GT and HOE (Hand-Object-Error orientation) metrics post-hoc. |
| Semantic reasoning + grasp region selection | [`semantic_layer/brain.py`](semantic_layer/brain.py) | OWL-v2 (zero-shot detection) → GPT-5.4 reasoning over a **Set-of-Mask (SoM) style 5×5 labeled grid** overlaid on the image (which grid cells = giver/receiver region, functional-end vs geometric strategy) → SAM (precise segmentation mask). |
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

### Simulation: Multi-Object Dual-Arm Handover (8 object categories)

Full system evaluated across 8 YCB objects (4 functional-end objects: hammer, spatula,
scissors, clamp; 4 non-functional objects: banana, sugar box, tomato soup can, bowl),
10 trials each, random position/orientation per trial.

| Metric | Result |
|---|---:|
| Grasp Success Rate (GSR) | 78% |
| Handover Success Rate (HSR) | 46% |
| Task Success Rate (TSR) | 58% |

The GSR→HSR gap (32 points) reflects that a feasible grasp doesn't guarantee a feasible
in-air handover — this is what the pose-completion module and the retry/regrasp fallback
exist to close.

### Orientation Adjustment Accuracy (Handover Orientation Error, HOE)

For functional-end objects, the PCA-based wrist rotation mechanism was evaluated on how
close the delivered orientation lands to the ideal functional-end-first target:

| Metric | Result |
|---|---:|
| Mean HOE | 22.92° |
| Within 30° | 77% |
| Within 45° | 97% |

### Real-World Human-to-Robot Handover (10 everyday objects)

Deployed to a physical UR3 + human receiver setup. Baseline uses only the top-ranked
AnyGrasp candidate pose; the multi-pose variant retries up to 3 ranked candidates on
planning failure; the orientation-adjustment variant additionally applies the PCA wrist
rotation to the 5 functional-end objects in the set (screwdriver, ladle, spoon, hammer,
pliers).

| Setting | Objects | GSR | HSR | TSR |
|---|---|---:|---:|---:|
| Single-pose baseline | 10 | 70% | 59% | 58% |
| Multi-pose fallback | 10 | 76% | 65% | 61% |
| + Orientation adjustment | 5 (functional-end subset) | 76% | 72% | 62% |

### Ablations (simulation)

| Removed component | Effect |
|---|---|
| Semantic reasoning (GPT + SoM) | TSR unchanged, but functional-correctness rate (does the receiver grab the *correct* end) drops ~10–20 points — success without semantic guidance often means grabbing the wrong end. |
| Point cloud completion | HSR drops from 46% → 36% — this module mainly helps the receiver arm plan a grasp on the occluded object. |
| Orientation adjustment | TSR *rises* to 71% (removing a failure-prone extra motion step) — this doesn't mean the module isn't worth it; see the paper's discussion of the completion-rate-vs-functional-correctness trade-off. |

Full per-object breakdowns and per-trial logs are in the paper's appendix; raw experiment
CSVs for the simulation ablations are in
[`ur_control/scripts/affordance_experiments/`](ur_control/scripts/affordance_experiments/).

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

### 3. Semantic layer (AnyGrasp + GPT) — two separate conda envs

`brain.py` and `anygrasp_ros.py` intentionally run in **different** conda environments (different
PyTorch/CUDA/Python version requirements), even though both live in `semantic_layer/`.

- **`anygrasp_ros` env** (runs `anygrasp_ros.py`): follow
  [`semantic_layer/install_anygrasp.txt`](semantic_layer/install_anygrasp.txt) — sets up PyTorch,
  MinkowskiEngine, pointnet2, graspnetAPI, and AnyGrasp itself.
  - **AnyGrasp requires its own academic license** — register at the
    [AnyGrasp SDK repo](https://github.com/graspnet/anygrasp_sdk) to get a license file. It cannot
    be shared; each user needs their own.
- **`lang-sam` env** (runs `brain.py`): follow
  [`semantic_layer/install_lang_sam.txt`](semantic_layer/install_lang_sam.txt) — PyTorch,
  `transformers` (OWL-v2 + SAM), `openai`.
- Set an OpenAI API key (`brain.py` uses `gpt-5.4` for grid-region reasoning):
  ```bash
  export OPENAI_API_KEY="your-key-here"   # add to ~/.bashrc to persist
  ```

### 4. Pose completion (FoundationPose)

Follow [`pose_completion/install_foundationpose.txt`](pose_completion/install_foundationpose.txt) —
pulls a pre-built Docker image with the FoundationPose environment baked in, downloads the model
weights/demo data, and adds a ROS bridge (`rospy` + friends) inside the container so
`foundationpose_node.py` can talk to the rest of the system. `pose_completion/foundationpose_node.py`
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
- Semantic reasoning via OpenAI GPT.

This repo contains only the code written for this research (dual-arm control logic, the
semantic-layer glue code, the affordance metric, and the ablation experiment tooling) — not the
upstream AnyGrasp / FoundationPose codebases themselves.

## Author

Developed at the [Robot Vision Lab](https://ntut-rvl.github.io/web/).

## Citation

A conference paper based on this work is currently under submission. Citation information will
be added here once the paper is accepted / publicly available.
