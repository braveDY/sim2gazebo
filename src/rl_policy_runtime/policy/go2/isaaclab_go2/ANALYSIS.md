# IsaacLab Go2 策略部署与高程图分析报告

本文档针对 `isaaclab_go2` 策略模型在 Gazebo 仿真环境中部署时出现的“**机器狗不敢向前迈步/原地发颤/抗拒前进**”问题，提供完整的全链路对比、数学推导、测试验证与修复方案。

---

## 1. 策略背景与训练配置溯源

* **模型来源**：IsaacLab 官方预训练权重 `Isaac-Velocity-Rough-Unitree-Go2-v0` (`policy.pt`)
* **训练配置**：[/home/brave/open_src/ssh_env_hub/task/isaaclab/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/rough_env_cfg.py](file:///home/brave/open_src/ssh_env_hub/task/isaaclab/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/rough_env_cfg.py)
* **环境基类**：`LocomotionVelocityRoughEnvCfg`
* **部署目录**：[src/rl_policy_runtime/policy/go2/isaaclab_go2](file:///home/brave/sim2sim/sim2gazebo/src/rl_policy_runtime/policy/go2/isaaclab_go2)

---

## 2. 训练端 vs 部署端 全参数对照表

| 检查项 | IsaacLab 官方训练配置 (`rough_env_cfg.py`) | 本地部署配置 (`isaaclab_go2`) | 一致性与问题 |
|---|---|---|---|
| **控制频率** | $dt = 0.005 \times 4 = 0.02\text{s}$ (50 Hz) | `runtime.control_hz: 50` | ✅ 完全一致 |
| **动作缩放** | `action_scale: 0.25` | `control.action_scale: 0.25` | ✅ 完全一致 |
| **PD 参数** | `kp: 25.0, kd: 0.5` | `kp: 25.0, kd: 0.5` | ✅ 完全一致 |
| **关节顺序** | PhysX 顺序 (FL, FR, RL, RR) | `gazebo_to_policy: [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]` | ✅ 映射完全正确 |
| **默认关节位置** | `[0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5]` | `policy_config.default_joint_pos` 一致 | ✅ 完全一致 |
| **重力投影** | $v' = q^{-1} \cdot [0, 0, -1] \cdot q$ | `project_gravity(quaternion_wxyz)` | ✅ 数学等价 |
| **高度扫描采样** | 17x11=187 维，网格 $[1.6\text{m}, 1.0\text{m}]$，分辨率 $0.1\text{m}$ | `elevation_map_adapter` 生成 $[17, 11]$ 网格 | ✅ 采样几何一致 |
| **高度扫描公式** | $Z_{base} - Z_{hit} - 0.5$ | `-values - 0.5` (其中 $values = Z_{hit} - Z_{base}$) | ⚠️ **数学公式一致，但异常值填充有致命错误** |
| **线速度观测** | 机体系速度 `root_lin_vel_b` | 直接读取 `/odom` 的 `twist.twist.linear` | ❌ **严重错误：传入了世界系速度** |
| **盲区/NaN 填充** | 射线穿透地面计算 | NaN 填 `-1.2` $\to$ policy 得到 `+0.7` | ❌ **严重错误：被模型误判为 1 米深悬崖** |

---

## 3. 核心根因深度剖析

### 根因 1：高程图盲区/NaN 默认值填充导致机器狗感知到“1 米深悬崖”

#### 1. 代码逻辑与数学推导
在 [elevation_map_adapter.py](file:///home/brave/sim2sim/sim2gazebo/src/rl_policy_runtime/rl_policy_runtime/elevation_map_adapter.py) 中：
```python
clip = config.get("clip", [-1.2, 0.0])  # manifest 中为 [-1.2, 0.0]
values = np.full(len(self.points), self.clip[0], dtype=np.float32)  # 默认填 -1.2
sampled = elevation[ri[valid], ci[valid]] - base[2]
values[valid] = np.clip(np.nan_to_num(sampled, nan=self.clip[0]), *self.clip)  # NaN 也填 -1.2
```
在 [policy.py](file:///home/brave/sim2sim/sim2gazebo/src/rl_policy_runtime/policy/go2/isaaclab_go2/policy.py) 中：
```python
def _height_scan(self, sensor: np.ndarray) -> np.ndarray:
    values = np.asarray(sensor, dtype=np.float32).reshape(-1, 3)[:, 2]
    return np.clip(-values - self.height_scan_offset, -1.0, 1.0)  # offset = 0.5
```

* **正常平地情况**：
  * 地面高度 $Z_{terrain} = 0.0\text{m}$，机器狗基座高度 $Z_{base} \approx 0.3246\text{m}$。
  * `sampled` $= Z_{terrain} - Z_{base} \approx -0.3246\text{m}$。
  * 进入 Policy 后：$-(-0.3246) - 0.5 = \mathbf{-0.1754}$（符合训练时的平地数值 $\approx -0.18$）。
* **雷达盲区或未建图区域（NaN / 越界）**：
  * Adapter 自动填入默认值 `self.clip[0] = -1.2`。
  * 进入 Policy 后：$-(-1.2) - 0.5 = \mathbf{+0.70}$！
  * **物理意义**：$+0.70$ 等价于 $Z_{base} - Z_{terrain} - 0.5 = 0.70 \implies Z_{terrain} = Z_{base} - 1.2 = \mathbf{-0.88\text{m}}$（即面前存在深达近 1 米的悬崖）。

#### 2. 闭环仿真验证对比
使用官方 `model.pt` 在相同状态下输入不同高程图的动作输出测试：

| 输入高度扫描特征 | 模型动作输出范数 (Action Norm) | 大腿关节动作 (`FL_thigh`) | 机器狗行为表征 |
|---|---|---|---|
| **平地标准值 (`-0.18`)** | **`3.77`** | **`-0.30 ~ -0.91`** | **节奏性迈步向前移动** |
| **NaN 错误填充值 (`+0.70`)** | **`20.95 ~ 24.99` (暴增 6~7 倍)** | **`+9.85` (折合 +2.46 rad)** | **关节完全打满限位，极度蜷缩抗拒迈步** |

---

### 根因 2：线速度观测 (`linear_velocity`) 坐标系错误（世界系 vs 机体系）

#### 1. 代码逻辑
在 [deploy_node.py](file:///home/brave/sim2sim/sim2gazebo/src/rl_policy_runtime/rl_policy_runtime/deploy_node.py) 中：
```python
linear_velocity = np.array([
    odom.twist.twist.linear.x,
    odom.twist.twist.linear.y,
    odom.twist.twist.linear.z,
], dtype=np.float32)
```

#### 2. 坐标系差异
* **IsaacLab 训练时**：`base_lin_vel` 为 `asset.data.root_lin_vel_b`（**机体 Base 坐标系**）。
* **Gazebo 部署时**：Gazebo 的 `libgazebo_ros_p3d.so` 插件发布的 `odom.twist.twist.linear` 是 **世界坐标系 (World Frame)** 速度。
* **影响**：
  * 一旦机器狗初始偏航角 $\text{Yaw} \neq 0$（例如朝向世界坐标系的 Y 轴），当机器狗尝试向前走时，世界系速度是 $[0, v_y, 0]$。
  * 策略模型接收到机体系 $v_x = 0, v_y = v_y$，误认为机器狗正在发生严重的横向漂移，从而输出巨大的反向纠偏力矩进行刹车，导致机器狗锁死不敢向前。

---

### 根因 3：`manifest.yaml` 中的 `clip: [-1.2, 0.0]` 截断了正高度

* 在 [manifest.yaml](file:///home/brave/sim2sim/sim2gazebo/src/rl_policy_runtime/policy/go2/isaaclab_go2/manifest.yaml) 中：
  `clip: [-1.2, 0.0]`
* `sampled = elevation - base[2]` 是相对于 Base 的高度差。若遇到高于 Base 的地形/上坡/台阶（$Z_{terrain} - Z_{base} > 0$），会被硬截断到 `0.0`，导致无法正确感知上凸障碍物。

---

### 根因 4：高程图刷新延迟与超时保护机制

* 在 [deploy_node.py](file:///home/brave/sim2sim/sim2gazebo/src/rl_policy_runtime/rl_policy_runtime/deploy_node.py) 中：
  ```python
  if not self._runtime.enabled or not self._check_sensor_freshness():
      return self._runtime.position_command(state, self._fsm_standing_joint_pos, self._fixed_kp, self._fixed_kd)
  ```
* `manifest.yaml` 中配置 `timeout_sec: 0.25` 且 `required: true`。
* 若 `elevation_mapping_cupy` 点云处理延迟超过 250ms，系统会自动降级为纯 PD 原地定点站立，终端 UI 会提示 `elevation_map: stale/waiting`，此时策略完全不执行。

---

## 4. 建议修复代码与实施方案

### 方案 1：修复线速度机体系旋转转换
在 [deploy_node.py](file:///home/brave/sim2sim/sim2gazebo/src/rl_policy_runtime/rl_policy_runtime/deploy_node.py) 的 `_build_robot_state()` 中：
```python
from .policy_api import quat_rotate_inverse

# 将 odom 世界坐标系线速度逆旋转到机体 Base 坐标系
if self._latest_odom is not None:
    odom = self._latest_odom
    v_world = np.array([
        odom.twist.twist.linear.x,
        odom.twist.twist.linear.y,
        odom.twist.twist.linear.z,
    ], dtype=np.float32)
    linear_velocity = quat_rotate_inverse(quaternion, v_world)
```

### 方案 2：修复高程图 NaN / 盲区默认填充值
在 [elevation_map_adapter.py](file:///home/brave/sim2sim/sim2gazebo/src/rl_policy_runtime/rl_policy_runtime/elevation_map_adapter.py) 的 `publish_map()` 中：
```python
# 默认填充平地高度差（即相对 base 高度差为 -base[2]，平地时约为 -0.3246m）
flat_ground_sampled = -float(base[2])
values = np.full(len(self.points), flat_ground_sampled, dtype=np.float32)
sampled = elevation[ri[valid], ci[valid]] - base[2]
values[valid] = np.nan_to_num(sampled, nan=flat_ground_sampled)
```

### 方案 3：更新 `manifest.yaml` 中的裁剪范围
在 [manifest.yaml](file:///home/brave/sim2sim/sim2gazebo/src/rl_policy_runtime/policy/go2/isaaclab_go2/manifest.yaml) 中放宽 clip 限制：
```yaml
sensors:
  elevation_map:
    topic: /isaaclab_elevation_map
    size: [1.6, 1.0]
    offset: [0.0, 0.0]
    resolution: 0.1
    clip: [-1.5, 1.0]
    shape: [17, 11, 3]
    timeout_sec: 0.25
    required: true
```
