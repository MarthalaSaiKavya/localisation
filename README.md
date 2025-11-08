# PF Localization – Project 3 (Deliverables 1 & 2)

This package contains everything needed to submit Deliverable 1 (motion + sensor models) and Deliverable 2 (full particle filter) for Project 3. Use the instructions below to recreate the map/likelihood field artifacts and to run the complete localization stack in simulation.

---

## Prerequisites
- Docker container `noetic-dev` with ROS Noetic, TurtleBot3 packages, and Gazebo assets installed.
- `catkin_ws` workspace with this package in `~/catkin_ws/src/pf_localization`.
- `catkin_make` already run so the workspace is built.
- `python3-opencv` available in the container (needed for distance fields).
- (Optional) ImageMagick’s `convert` if you prefer generating PNGs via CLI, though the provided node already emits PNGs.

All steps assume three host terminals attached to the same container.

---

## Common Terminal Setup
1. **Attach to the container**
   ```bash
   docker start -ai noetic-dev
   ```
2. **Open two extra terminals** (Terminal 2/3):
   ```bash
   docker exec -it noetic-dev bash
   ```
3. **Source ROS + workspace in every terminal**
   ```bash
   source /opt/ros/noetic/setup.bash
   source ~/catkin_ws/devel/setup.bash
   export TURTLEBOT3_MODEL=burger
   ```

---

## Deliverable 1 Workflow

### 1. Build the House Map
Use three terminals as in Project 1:
- **Terminal 1 – Gazebo (headless recommended)**  
  `roslaunch turtlebot3_gazebo turtlebot3_house.launch gui:=false`
- **Terminal 2 – SLAM / gmapping**  
  `roslaunch turtlebot3_slam turtlebot3_slam.launch slam_methods:=gmapping`
- **Terminal 3 – Teleop**  
  `roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch`

Drive the robot with WASD for ~3 minutes to cover the entire house. When satisfied:
```bash
rosrun map_server map_saver -f ~/catkin_ws/src/pf_localization/maps/house_map
convert ~/catkin_ws/src/pf_localization/maps/house_map.pgm \
        ~/catkin_ws/src/pf_localization/images/house_map.png
```
(The `convert` step is optional because the next node can also make a PNG.)

### 2. Generate the Likelihood Field (distance matrix)
```bash
roslaunch pf_localization likelihood_field.launch
```
This loads `maps/house_map.yaml`, builds the distance transform, and saves `images/likelihood_field.png`. The `pf_localization/map_utils.py` helpers are also used later by the particle filter, so the PNG you submit directly reflects the data used at runtime.

Artifacts to include in the Deliverable 1 tarball:
- `maps/house_map.yaml` + `maps/house_map.pgm`
- `images/house_map.png`
- `images/likelihood_field.png`
- Entire ROS package.

---

## Deliverable 2 Workflow – Particle Filter Localization

### Option A: One-command bringup (spawns Gazebo + PF)
1. **Terminal 1** – start the full stack (headless by default):
   ```bash
   roslaunch pf_localization bringup_house.launch gui:=false
   ```
   This launches the TurtleBot3 house world and the custom `particle_filter_node`. The node:
   - Subscribes to `/odom` and `/scan`
   - Samples 800 particles in free space
   - Publishes `particle_filter/particle_cloud`, `particle_filter/pose`, `particle_filter/odom`
   - Broadcasts `map → base_footprint` TF using the best particle estimate
2. **Terminal 2** – teleoperate the robot:
   ```bash
   roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
   ```
3. **Terminal 3 (optional)** – visualize in RViz:
   ```bash
   rviz
   ```
   Add the `Map`, `LaserScan`, `PoseArray` (`/particle_filter/particle_cloud`), and `Pose` (`/particle_filter/pose`) displays to watch the filter converge.

### Option B: Attach to an already-running simulator
If Gazebo is already up (e.g., via the starter `particle_filter.launch` from the course repo), start only the PF node:
```bash
roslaunch pf_localization particle_filter.launch
```

### Node outputs and parameters
- Customize runtime via `rosparam`, e.g., `beam_step`, `num_particles`, `sigma_hit`, etc. All defaults live inside `particle_filter_node.py`.
- The node writes TF and publishes a `PoseArray` of every particle plus a `PoseStamped` best estimate so you can record Deliverable 2 videos directly.

---

## Troubleshooting
- Run `catkin_make` in `~/catkin_ws` whenever Python files change (installs the `pf_localization` helpers defined in `setup.py`).
- If Gazebo fails, ensure the TurtleBot3 models are installed (`export TURTLEBOT3_MODEL=burger` before launching).
- `particle_filter_node` relies on `/scan` and `/odom`. Launch order matters: wait for Gazebo to publish both topics before expecting particles to converge.
- If the map path inside `maps/house_map.yaml` is absolute and invalid for your machine, the helper automatically falls back to the local `maps/house_map.pgm`.

With the above steps plus the included PNGs/videos, the package is ready for submission for both project deliverables.
