# 新增策略模型指南

本文说明如何将一个新的 Go2 强化学习策略接入 `rl_policy_runtime`。新增策略只需要在自己的策略目录中提供：模型文件、`manifest.yaml` 和 `policy.py`。无需修改 `deploy_node.py`、Gazebo 或 `elevation_mapping_cupy`。

当前运行时的边界如下：

- 模型后端：TorchScript（`torch.jit.load`）。
- 机器人：当前 Go2 配置为 12 个关节，策略输出必须是 12 个归一化动作。
- 控制：运行时将动作缩放并转换为 `q/dq/tau/kp/kd`；底层 `robot_joint_controller` 负责 PD 力矩计算。
- 传感器：部署节点订阅 manifest 中声明的 `std_msgs/msg/Float32MultiArray` 话题，并在进入策略前检查数据长度和新鲜度。
- 高程图：若声明名为 `elevation_map` 的传感器，部署节点会自动启动通用 `/filtered_map` 适配器。高程图几何由策略 manifest 决定，建图包不需要也不接受策略名。

---

## 1. 新建策略目录

以 `new_policy` 为例，目录必须位于机器人目录下：

```text
src/rl_policy_runtime/policy/go2/new_policy/
├── manifest.yaml
├── policy.py
└── model.pt
```

不要把策略专用代码放入 `rl_policy_runtime/deploy_node.py`。策略之间的观测拼接、历史帧、归一化、Torch 模型调用和后处理都应写在自己的 `policy.py` 中。

`setup.py` 会递归安装 `policy/` 下的 `.yaml`、`.py`、`.pt`、`.onnx` 和 `.engine` 文件。新增或替换模型后仍必须重新执行 `colcon build`，以便安装目录同步更新。

---

## 2. 编写 manifest.yaml

下面是一个带高程图输入的完整模板。数值仅是示例，必须替换为训练时使用的配置。

```yaml
name: new_policy
robot_name: go2

runtime:
  backend: torchscript
  model: model.pt
  control_hz: 50

# 只用于键盘状态机的起立/趴下，不是 locomotion 策略的默认关节姿态。
fsm:
  standing_joint_pos: [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 0.8, -1.5]
  pre_get_up_pos: [0.0, 1.36, -2.65, 0.0, 1.36, -2.65, 0.0, 1.36, -2.65, 0.0, 1.36, -2.65]
  fixed_kp: 80.0
  fixed_kd: 3.0

sensors:
  elevation_map:
    # 运行时适配器发布的话题，同时也是 policy.py 中 sensors 的键。
    topic: /ame_elevation_map
    # 仅作长度校验；数据仍以一维 float 数组传入 policy.py。
    shape: [17, 25, 3]
    required: true
    timeout_sec: 0.25

    # 以下字段由高程图适配器使用。采样点数应满足
    # (size_x / resolution + 1) * (size_y / resolution + 1) * 3 == prod(shape)。
    layer: elevation
    size: [1.2, 0.8]
    offset: [0.375, 0.0]
    resolution: 0.05
    clip: [-1.2, 0.0]

robot:
  num_joints: 12
  joint_names: ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"]
  gazebo_joint_mapping: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

  # Gazebo 数组转换到训练策略关节顺序的索引排列。
  gazebo_to_policy: [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
  # 必须是 gazebo_to_policy 的逆排列，用于把模型动作转回 Gazebo 顺序。
  policy_to_gazebo: [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]

control:
  # 均为 Gazebo 关节顺序，长度必须为 12。
  default_joint_pos: [-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5]
  action_scale: [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
  kp: [40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40]
  kd: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  torque_limits: [35, 40, 40, 35, 40, 40, 35, 40, 40, 35, 40, 40]
  clip_actions_lower: [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]
  clip_actions_upper: [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

# policy.py 自己使用的任意训练配置。运行时不会解释这里的字段。
policy_config:
  ang_vel_scale: 1.0
  dof_vel_scale: 0.05
  default_dof_pos: [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5]
```

### 关键字段与约束

| 字段 | 作用 | 要求 |
| --- | --- | --- |
| `runtime.model` | 模型相对策略目录的路径 | 文件必须存在 |
| `runtime.backend` | 模型加载器 | 当前只能为 `torchscript` |
| `runtime.control_hz` | 控制频率 | 正数；应与训练部署频率一致 |
| `robot_name` | 策略所属机器人 | 启动参数 `robot` 必须相同 |
| `robot.num_joints` | 关节数 | 当前 Go2 为 `12` |
| `gazebo_to_policy` | 输入关节顺序变换 | 0 到 11 的无重复排列 |
| `policy_to_gazebo` | 输出关节顺序变换 | `gazebo_to_policy` 的逆排列 |
| `control.*` | 低层位置控制参数 | 每个数组长度必须为 12 |
| `sensors.<name>.shape` | 传感器长度校验 | `prod(shape)` 必须等于实际数据个数 |
| `timeout_sec` | 数据失效时间 | 超时不会进入策略，机器人保持站立姿态 |

关节映射容易出错。若 `gazebo_to_policy = mapping`，策略关节向量由 `gazebo_values[mapping]` 得到；因此 `policy_to_gazebo` 应满足 `mapping[policy_to_gazebo] == [0, 1, ..., 11]`。

---

## 3. 编写 policy.py 与常用基类工具

运行时会动态导入策略目录下的 `policy.py`，并构造其中名为 `Policy` 的类：

```python
Policy(
    model=torch.jit.ScriptModule,
    config=dict,
    gazebo_to_policy=np.ndarray,
    policy_to_gazebo=np.ndarray,
)
```

`rl_policy_runtime.policy_api` 提供了开箱即用的基类方法与常用数学工具，避免在每个策略中重复编写样板代码。

### 3.1 提供的基础设施

- **`BasePolicy.build_proprio_features(state, default_joint_pos, ang_vel_scale, dof_vel_scale, include_lin_vel)`**：一键生成 45/48 维标准化本体感知向量（包含角速度、重力投影、命令、关节差值、关节速度、历史动作）。
- **`BasePolicy.map_to_policy(values)`** 与 **`BasePolicy.map_to_gazebo(values)`**：关节顺序转换。
- **`HistoryBuffer(history_len)`**：多帧观测时序缓存管理器，内置自动填充、更新与扁平/堆叠输出。
- **`project_gravity(quaternion_wxyz)`** / **`quat_rotate_inverse(q, v)`**：四元数空间变换。

### 3.2 最小策略模板（标准 MLP + 高程图输入）

```python
from typing import Any, Dict, Mapping

import numpy as np
import torch

from rl_policy_runtime.policy_api import Policy as BasePolicy, RobotState


class Policy(BasePolicy):
    def __init__(
        self,
        model: torch.jit.ScriptModule,
        config: Dict[str, Any],
        gazebo_to_policy: np.ndarray,
        policy_to_gazebo: np.ndarray,
    ) -> None:
        super().__init__(model, config, gazebo_to_policy, policy_to_gazebo)
        self.ang_vel_scale = float(config.get("ang_vel_scale", 1.0))
        self.dof_vel_scale = float(config.get("dof_vel_scale", 0.05))
        self.default_dof_pos = np.array(config["default_dof_pos"], dtype=np.float32)

    def infer(
        self,
        state: RobotState,
        sensors: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        # 1. 组装 45 维本体感知特征
        proprio = self.build_proprio_features(
            state=state,
            default_joint_pos=self.default_dof_pos,
            ang_vel_scale=self.ang_vel_scale,
            dof_vel_scale=self.dof_vel_scale,
            include_lin_vel=False,
        )

        # 2. 提取传感器高程图 (1275 维)
        elevation_map = sensors["elevation_map"].astype(np.float32).reshape(-1)

        # 3. 拼接得到网络输入 (1320 维)
        observation = np.concatenate([proprio, elevation_map], dtype=np.float32)

        # 4. TorchScript 前向传播
        with torch.no_grad():
            action = self.model(torch.from_numpy(observation).unsqueeze(0))

        # 5. 返回训练策略顺序的 12 维动作
        return action.squeeze(0).cpu().numpy().astype(np.float32)
```

### 3.3 带时序历史帧的策略模板（如 Quad MWM）

```python
from typing import Any, Dict, Mapping
import numpy as np
import torch
from rl_policy_runtime.policy_api import HistoryBuffer, Policy as BasePolicy, RobotState

class Policy(BasePolicy):
    def __init__(self, model, config, gazebo_to_policy, policy_to_gazebo):
        super().__init__(model, config, gazebo_to_policy, policy_to_gazebo)
        self.default_dof_pos = np.array(config["default_dof_pos"], dtype=np.float32)
        # 初始化 4 帧历史缓存
        self.history_buffer = HistoryBuffer(history_len=int(config.get("history_length", 4)))

    def reset(self) -> None:
        self.history_buffer.reset()

    def infer(self, state: RobotState, sensors: Mapping[str, np.ndarray]) -> np.ndarray:
        proprio = self.build_proprio_features(state, self.default_dof_pos, dof_vel_scale=0.05)
        elevation_map = sensors["elevation_map"].astype(np.float32).reshape(-1)
        observation = np.concatenate([proprio, elevation_map], dtype=np.float32)

        # 推入当前帧并获取拼接的历史向量
        history = self.history_buffer.update(proprio)

        with torch.no_grad():
            action = self.model(
                torch.from_numpy(observation).unsqueeze(0),
                torch.from_numpy(history).unsqueeze(0),
            )
        return action.squeeze(0).cpu().numpy().astype(np.float32)
```

### 3.4 状态定义与控制换算

`RobotState` 提供下列字段：

| 字段 | 顺序/单位 |
| --- | --- |
| `joint_pos`、`joint_vel` | Gazebo 关节顺序，弧度和弧度每秒 |
| `quaternion_wxyz` | `[w, x, y, z]` |
| `angular_velocity`、`linear_velocity` | 基座坐标系速度，单位为 SI |
| `command` | `[vx, vy, yaw_rate]`；来自键盘或 `/cmd_vel` |
| `previous_action` | Gazebo 关节顺序的上一帧模型动作 |

`infer()` 必须返回形状为 `(12,)` 的 `float32` 数组，且顺序必须是训练策略顺序。运行时会按以下流程处理：

```text
策略动作（训练顺序）
  -> policy_to_gazebo
  -> clip_actions_lower / clip_actions_upper
  -> action_scale
  -> default_joint_pos + scaled_action
  -> RobotCommand(q, dq=0, tau=0, kp, kd)
```

不要在 `policy.py` 再计算 PD 力矩，也不要把模型输出直接当作 `tau`；当前关节控制器会使用下发的 `q/dq/kp/kd` 完成 PD 控制。

---

## 4. 不使用外部感知的策略

没有外部感知时，将 `sensors` 留空或删除整个字段即可：

```yaml
sensors: {}
```

`policy.py` 不应访问不存在的 `sensors` 键。此类策略不会启动高程图适配器，也不依赖 elevation mapping 节点。

---

## 5. 使用其他外部感知

当前部署节点会为 manifest 中每一个传感器订阅 `Float32MultiArray`。例如外部节点已发布一个 64 维点云特征：

```yaml
sensors:
  pointcloud_feature:
    topic: /go2/perception/feature
    shape: [64]
    required: true
    timeout_sec: 0.10
```

策略中直接读取：

```python
feature = sensors["pointcloud_feature"].reshape(64)
```

对于不同的网络输入结构，保持 ROS 消息为一维 float 数组，在策略内部恢复为训练时的形状即可，例如 `reshape(4, 32, 32)` 或 CNN 二维特征 `reshape(1, 1, 17, 25)`。这使部署节点保持不变。

---

## 6. 高程图策略配置

`elevation_map_adapter` 从 `/filtered_map` 和 `/odom` 采样，并输出每个点的 `[x, y, relative_height]`。点云以机器人 yaw 对齐，`relative_height` 会按 `clip` 裁剪。

常用策略采样几何对照：

| 策略 | `size` | `offset` | `resolution` | `shape` | 数据长度 |
| --- | --- | --- | --- | --- | --- |
| `cnn` | `[1.2, 0.8]` | `[0.375, 0.0]` | `0.05` | `[17, 25, 3]` | 1275 |
| `ame` | `[1.2, 0.8]` | `[0.375, 0.0]` | `0.05` | `[17, 25, 3]` | 1275 |
| `quad_mwm` | `[1.2, 0.8]` | `[0.375, 0.0]` | `0.05` | `[17, 25, 3]` | 1275 |
| `isaaclab_ame` | `[1.6, 1.0]` | `[0.0, 0.0]` | `0.05` | `[21, 33, 3]` | 2079 |

---

## 7. 构建与运行

在 `em_gpu_humble` 容器中执行：

```bash
cd /sim2sim/rl_sar
source /opt/ros/humble/setup.bash
colcon build --packages-select rl_policy_runtime --symlink-install
source install/setup.bash
```

启动仿真与通用建图：

```bash
ros2 launch rl_sar gazebo.launch.py
ros2 launch elevation_mapping_cupy elevation_mapping.launch.py use_sim_time:=true
```

在新的终端启动策略；`ros2 run` 保留终端 stdin，因此可使用键盘状态机：

```bash
source /sim2sim/rl_sar/install/setup.bash
ros2 run rl_policy_runtime deploy_node --ros-args -p policy:=cnn
```

在交互式终端中，控制器会使用 `rich` 显示状态机进度、当前速度命令及传感器新鲜度。

运行后依次按：

1. `0`：从四脚朝天等初始姿态执行起立序列。
2. 等待起立完成。
3. `1`：加载模型并进入 locomotion。
4. `9`：趴下并回到 passive。

---

## 8. 接入检查清单

- [ ] 模型已导出为可在 CPU 上 `torch.jit.load` 的 TorchScript 文件。
- [ ] `manifest.yaml` 中 `runtime.model` 文件存在。
- [ ] `robot_name`、启动参数 `robot` 和策略目录名称一致。
- [ ] 模型输入的每一个归一化、坐标系、关节顺序和历史帧都在 `policy.py` 中复现（优先使用 `BasePolicy` 内置方法）。
- [ ] `infer()` 返回训练策略顺序的 12 维动作。
- [ ] `gazebo_to_policy` 与 `policy_to_gazebo` 互为逆排列。
- [ ] 所有 `control` 数组长度都是 12，并使用训练/部署验证过的 PD 参数。
- [ ] 每个 required 传感器都有稳定发布者，数据长度等于 `prod(shape)`。
- [ ] 高程图策略的采样点数与 `shape` 一致。
- [ ] 先在仿真中验证起立、静止、前进、转向和趴下，再接入实机。
