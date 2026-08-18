# 无 odom Go2 Gazebo 策略仿真

本分支 `feat/no-odom-quad-mwm` 用于在 Gazebo 中验证 Go2 的 `quad_mwm` 策略，运行时不启用 Gazebo 真值里程计，也不发布 `odom -> base` TF。高程图改为在 `base` 坐标系内构建，策略控制器使用 `/imu`、关节状态和局部高程图。

本分支仅用于仿真验证，不包含 Unitree SDK，不能用于实机低层控制。

## 当前链路

```text
Gazebo
  ├─ /imu
  ├─ /robot_joint_controller/state
  └─ /utlidar/cloud
       ↓
点云自滤波 → /utlidar/cloud_filtered
       ↓
elevation_mapping_cupy（map_frame=base）
       ↓
/elevation_mapping_node/elevation_map_filter
       ↓
rl_policy_runtime（quad_mwm）
       ↓
/robot_joint_controller/command
```

无 odom 模式下不应存在 `/odom` 或 `odom -> base` TF。高程图的消息头坐标系必须是 `base`。

## 启动

容器和工作区已准备好后，打开三个终端。每个终端均执行：

```bash
cd /sim2sim/sim2gazebo
source /opt/ros/humble/setup.bash
source install/setup.bash
```

源码修改后，先重新构建相关包：

```bash
cd /sim2sim/sim2gazebo
source /opt/ros/humble/setup.bash
colcon build --packages-select rl_sar elevation_mapping_cupy rl_policy_runtime --symlink-install
source install/setup.bash
```

终端 1，启动无 odom Gazebo：

```bash
ros2 launch rl_sar gazebo.launch.py \
  wname:=terrain_track \
  enable_truth_tf:=false
```

终端 2，启动基于 `base` 坐标系的高程图：

```bash
ros2 launch elevation_mapping_cupy elevation_mapping.launch.py \
  use_sim_time:=true \
  no_odom:=true
```

终端 3，启动 `quad_mwm` 控制器：

```bash
ros2 launch rl_policy_runtime controller.launch.py \
  policy:=quad_mwm \
  keyboard_enabled:=true \
  control_enabled:=false
```

控制器启动后先按 `0` 执行起身流程；起身完成后按 `1` 进入策略控制。不要在机器人仰躺时直接通过 `~/enable` 服务进入 `LOCOMOTION`。

## 当前问题与结论

### 启动后保持四脚朝天

Gazebo 中机器人生成后可能会先仰躺。控制器启动时，`PASSIVE` 状态必须持续发送固定站立关节角，并且默认重置 Gazebo 世界；两者共同保证机器人会尝试自动起身。

若启动控制器后仍保持仰躺，先确认运行的不是旧构建：

```bash
ros2 param get /rl_policy_runtime reset_world_on_start
ros2 topic hz /imu
ros2 topic hz /robot_joint_controller/state
ros2 topic hz /robot_joint_controller/command
ros2 control list_controllers
```

预期：

- `reset_world_on_start` 为 `true`；
- `/imu` 约为 `100 Hz`；
- `/robot_joint_controller/state` 有持续数据；
- `/robot_joint_controller/command` 约为 `50 Hz`；
- `joint_state_broadcaster` 与 `robot_joint_controller` 均为 `active`。

如果命令频率为 `50 Hz` 但机器人仍不响应，应检查控制器日志、关节命名和 Gazebo 中的关节控制插件；如果命令话题没有数据，优先检查 `/imu` 和 `/robot_joint_controller/state`。

### 高程图报找不到 `odom`

出现以下日志说明高程图仍运行在有 odom 配置，而不是无 odom 配置：

```text
Frame 'odom' or 'utlidar_lidar' does not exist
```

必须使用 `no_odom:=true` 启动高程图。随后检查地图是否正常发布：

```bash
ros2 topic echo --once /elevation_mapping_node/elevation_map_filter --field header.frame_id
ros2 topic hz /elevation_mapping_node/elevation_map_filter
```

预期输出坐标系为 `base`，发布频率约为 `10 Hz`。没有高程图时，`quad_mwm` 会因必需传感器未就绪而保持站立，不执行网络推理。

### 仰躺时进入策略控制仍可能失败

这是当前无 odom 方案的核心限制。高程图使用会随机体旋转的 `base` 坐标系；当机器人四脚朝天时，机体 `z` 轴指向世界下方，地面在地图中的高度会由训练时的负值变成正值。`quad_mwm` 的高度输入裁剪范围为 `[-1.2, 0]`，正高度会被裁剪为 `0`，同时地形的平面方向也与训练时不一致。

因此，起身阶段应由固定关节姿态控制完成，机器人站稳后再按 `1` 进入策略。若需要策略直接从任意翻转姿态恢复，需将地形图改为重力对齐的局部坐标系，而不是直接使用随机身完整旋转的 `base` 坐标系。

## 运行检查清单

```bash
ros2 topic echo --once /elevation_mapping_node/elevation_map_filter --field header.frame_id
ros2 topic hz /elevation_mapping_node/elevation_map_filter
ros2 topic hz /imu
ros2 topic hz /robot_joint_controller/state
ros2 topic hz /robot_joint_controller/command
ros2 control list_controllers
```

满足以下条件后再进入策略控制：高程图坐标系为 `base`、高程图频率约 `10 Hz`、控制命令频率约 `50 Hz`、两个 controller 均为 `active`，且机器人已通过起身流程恢复站立。
