# rl_sar：Go2 Gazebo 策略仿真

基于 **Gazebo Classic** 的 Go2 强化学习策略部署工程。策略运行时由 Python 包 `rl_policy_runtime` 提供，新增策略只需在 `policy/go2/<name>/` 下添加模型、`manifest.yaml` 与 `policy.py`，无需改动框架代码。此分支不含 Unitree SDK，**不能用于实机低层控制**。

```text
rl_sar                          Gazebo 世界、机器人生成、TF 与点云过滤
elevation_mapping_cupy          通用高程图建图（/filtered_map）
rl_policy_runtime               FSM、策略加载、终端交互与关节命令
└── policy/go2/<name>           策略模型、manifest 与预处理代码
```

数据流：`Gazebo → /imu、/odom、关节状态 → elevation_mapping_cupy → /filtered_map → rl_policy_runtime → /robot_joint_controller/command`。若策略声明 `elevation_map` 传感器，运行时按 manifest 自动启动高程图适配器生成 `/ame_elevation_map`。

## 环境部署

### 1. 构建 Docker 镜像

为执行 shell 脚本设置权限，然后运行脚本构建镜像（约 15–20 分钟）。`build.sh` 默认使用 FastDDS/FastRTPS 构建：

```bash
chmod +x docker/build.sh docker/run.sh
ROS_DISTRO=humble ./build.sh        # build.sh 默认 ROS_DISTRO=jazzy，本项目使用 Humble
```

镜像构建细节与排障记录见 [evelutionMap_docker_build.md](src/elevation_mapping/evelutionMap_docker_build.md)。

### 2. 进入容器

使用 `src/elevation_mapping/em_gpu_humble.sh` 中的函数进入容器（首次自动创建，挂载宿主机工作区到 `/sim2sim`）：

```bash
source src/elevation_mapping/em_gpu_humble.sh   # 提供 em_create / em_start / em_enter 等函数
em_enter
```

常用函数：`em_create` 创建容器、`em_start` 启动、`em_enter` 进入、`em_stop` 停止、`em_remove` 删除后重建。

### 3. 容器内安装依赖

进入容器后执行（换源、安装 ROS 2 依赖与 Gazebo 模型）：

```bash
# Ubuntu基础源换清华
sudo sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list
sudo sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

# ROS 2 Humble 源（TUNA 镜像）
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu jammy main" | \
  sudo tee /etc/apt/sources.list.d/ros2.list

# 刷新源
sudo apt update

sudo apt install -y \
  ros-humble-xacro \
  ros-humble-control-toolbox \
  ros-humble-hardware-interface \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-joy \
  ros-humble-demo-nodes-cpp \
  ros-humble-gazebo-ros2-control \
  ros-humble-joint-state-publisher-gui \
  ros-humble-rviz2

pip install python3-rich

mkdir -p ~/.gazebo/models
git -c http.version=HTTP/1.1 clone --depth 1 --filter=blob:none --sparse \
  https://github.com/osrf/gazebo_models.git /tmp/gazebo_models
git -C /tmp/gazebo_models sparse-checkout set sun ground_plane
cp -a /tmp/gazebo_models/sun /tmp/gazebo_models/ground_plane ~/.gazebo/models/
```

### 4. 构建工作区

```bash
cd /sim2sim/rl_sar
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

仅修改 Python 策略运行时时可缩小构建范围：`colcon build --packages-select rl_policy_runtime --symlink-install`。

## 启动仿真

三个终端，均需先加载 ROS 与工作区环境：

1. **Gazebo**：`ros2 launch rl_sar gazebo.launch.py` — 可选 `rname:=go2`（机器人）、`wname:=stairs|terrain_track`（世界）
2. **高程图**：`ros2 launch elevation_mapping_cupy elevation_mapping.launch.py use_sim_time:=true`
3. **策略控制器**：`ros2 run rl_policy_runtime deploy_node --ros-args -p policy:=quad_mwm -p keyboard_enabled:=true`

可选策略：`quad_mwm`（高程图 17×25×3）、`isaaclab_ame`（21×33×3）。控制器会自动启动 `robot_joint_controller` 与高程图适配器；**不要同时运行多个 `deploy_node`**。

## 操作

键盘状态机（Rich 面板显示 FSM 状态、起立/趴下进度、速度指令与传感器新鲜度）：

| 按键 | 行为 |
| --- | --- |
| `0` / `1` / `9` | 起立 / 起立完成后进入 locomotion / 趴下 |
| `P` | 停止策略，切换被动站立 |
| `W`/`S`、`A`/`D`、`Q`/`E` | 调节 vx / vy / yaw |
| `Space` | 速度命令清零 |
| `N` | 切换键盘命令与 `/cmd_vel` |
| `R` / `Enter` | 重置世界 / 暂停·恢复物理 |

典型顺序：等状态话题就绪 → `0` → 等进度条完成 → `1`。面板显示 `elevation_map: waiting/stale` 时策略保持站立、不进入推理。

非交互启动（自动化）：

```bash
ros2 launch rl_policy_runtime controller.launch.py policy:=quad_mwm keyboard_enabled:=false
ros2 service call /rl_policy_runtime/enable std_srvs/srv/SetBool "{data: true}"
ros2 service call /rl_policy_runtime/reset std_srvs/srv/Trigger "{}"
```

## 新增策略

见 [新增策略模型指南](src/rl_policy_runtime/ADDING_POLICY.md)：策略目录与 `manifest.yaml` 模板、`policy.py` 观测构造与推理模板、关节顺序与 PD 约束、接入与验收清单。

## 常见问题

- **`rosidl_adapter` 找不到**：ROS Python 环境未加载，重新 `source /opt/ros/humble/setup.bash` 后再构建。
- **`Ignoring elevation_map: got ... expected ...`**：数据长度 = `(size_x/resolution + 1) × (size_y/resolution + 1) × 3`，须与 manifest `shape` 对应；同一时刻只保留一个 `deploy_node`。
- **键盘无效或面板不显示**：用 `ros2 run` 而非 `ros2 launch` 操作键盘，确认 `keyboard_enabled:=true`，并安装 `python3-rich`。
