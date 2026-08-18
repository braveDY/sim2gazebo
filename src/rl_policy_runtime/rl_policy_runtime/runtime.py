import importlib.util
import os
from typing import Any, Dict, Optional

import numpy as np
import yaml

from .policy_api import Policy, RobotState


class PolicyRuntime:
    def __init__(self, policy_root: str, policy_name: str, robot: str):
        self._policy_root = policy_root
        self._policy_name = policy_name
        self._robot = robot

        policy_dir = os.path.join(policy_root, robot, policy_name)
        manifest_path = os.path.join(policy_dir, "manifest.yaml")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, "r") as f:
            self._manifest = yaml.safe_load(f)

        if self._manifest.get("robot_name") != robot:
            raise ValueError(
                f"Manifest robot_name '{self._manifest.get('robot_name')}' "
                f"!= expected '{robot}'"
            )

        self._control = self._manifest["control"]
        self._robot_cfg = self._manifest["robot"]
        self._runtime_cfg = self._manifest["runtime"]
        self._sensors_cfg = self._manifest.get("sensors", {})
        self._policy_config = self._manifest.get("policy_config", {})

        self._num_joints = self._robot_cfg["num_joints"]
        self._gazebo_to_policy = np.array(self._robot_cfg["gazebo_to_policy"], dtype=np.int64)
        self._policy_to_gazebo = np.array(self._robot_cfg["policy_to_gazebo"], dtype=np.int64)
        self._default_joint_pos = np.array(self._control["default_joint_pos"], dtype=np.float32)
        self._action_scale = np.array(self._control["action_scale"], dtype=np.float32)
        self._kp = np.array(self._control["kp"], dtype=np.float32)
        self._kd = np.array(self._control["kd"], dtype=np.float32)
        self._torque_limits = np.array(self._control["torque_limits"], dtype=np.float32)
        self._clip_lower = np.array(self._control["clip_actions_lower"], dtype=np.float32)
        self._clip_upper = np.array(self._control["clip_actions_upper"], dtype=np.float32)

        for name, values in {
            "gazebo_to_policy": self._gazebo_to_policy,
            "policy_to_gazebo": self._policy_to_gazebo,
            "default_joint_pos": self._default_joint_pos,
            "action_scale": self._action_scale,
            "kp": self._kp,
            "kd": self._kd,
            "torque_limits": self._torque_limits,
            "clip_actions_lower": self._clip_lower,
            "clip_actions_upper": self._clip_upper,
        }.items():
            if values.shape != (self._num_joints,):
                raise ValueError(
                    f"Manifest field '{name}' must contain {self._num_joints} values, "
                    f"got {values.shape}"
                )
        if not np.array_equal(np.sort(self._gazebo_to_policy), np.arange(self._num_joints)):
            raise ValueError("gazebo_to_policy must be a joint-index permutation")
        if not np.array_equal(np.sort(self._policy_to_gazebo), np.arange(self._num_joints)):
            raise ValueError("policy_to_gazebo must be a joint-index permutation")

        self._control_hz = float(self._runtime_cfg.get("control_hz", 50))
        if self._control_hz <= 0:
            raise ValueError("runtime.control_hz must be positive")
        self._control_period = 1.0 / self._control_hz

        self._policy: Optional[Policy] = None
        self._model = None
        self._policy_module = None
        self._last_action = np.zeros(self._num_joints, dtype=np.float32)
        self._last_safe_action = np.column_stack(
            [self._default_joint_pos, np.zeros(self._num_joints),
             np.zeros(self._num_joints), self._kp, self._kd]
        ).astype(np.float32)
        self._enabled = False
        self._loaded = False

    @property
    def manifest(self) -> Dict[str, Any]:
        return self._manifest

    @property
    def control_hz(self) -> float:
        return self._control_hz

    @property
    def control_period(self) -> float:
        return self._control_period

    @property
    def default_joint_pos(self) -> np.ndarray:
        return self._default_joint_pos.copy()

    @property
    def torque_limits(self) -> np.ndarray:
        return self._torque_limits.copy()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        value = bool(value)
        if value and not self._loaded:
            self._load()
        if value:
            self._policy.reset()
        self._enabled = value

    @property
    def loaded(self) -> bool:
        return self._loaded

    def required_sensor_topics(self) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for sensor_name, sensor_cfg in self._sensors_cfg.items():
            if sensor_cfg.get("required", False):
                result[sensor_name] = sensor_cfg["topic"]
        return result

    def sensor_topics(self) -> Dict[str, str]:
        return {
            sensor_name: sensor_cfg["topic"]
            for sensor_name, sensor_cfg in self._sensors_cfg.items()
            if "topic" in sensor_cfg
        }

    def sensor_shape(self, sensor_name: str) -> Optional[tuple]:
        shape = self._sensors_cfg.get(sensor_name, {}).get("shape")
        return tuple(int(value) for value in shape) if shape else None

    def sensor_timeout(self, sensor_name: str) -> float:
        cfg = self._sensors_cfg.get(sensor_name, {})
        return cfg.get("timeout_sec", 0.5)

    def sensor_type(self, sensor_name: str) -> str:
        return self._sensors_cfg.get(sensor_name, {}).get("type", "float32_multiarray")

    def _load(self) -> None:
        if self._loaded:
            return

        policy_dir = os.path.join(self._policy_root, self._robot, self._policy_name)

        policy_file = os.path.join(policy_dir, "policy.py")
        if not os.path.isfile(policy_file):
            raise FileNotFoundError(f"Policy module not found: {policy_file}")
        module_name = f"rl_policy_{self._robot}_{self._policy_name}"
        spec = importlib.util.spec_from_file_location(module_name, policy_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load policy module: {policy_file}")
        self._policy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._policy_module)

        model_path = os.path.join(policy_dir, self._runtime_cfg["model"])
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        backend = self._runtime_cfg.get("backend", "torchscript")
        if backend == "torchscript":
            import torch
            self._model = torch.jit.load(model_path, map_location="cpu")
            self._model.eval()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        PolicyClass = getattr(self._policy_module, "Policy", None)
        if PolicyClass is None:
            raise AttributeError(
                f"Policy class not found in {self._robot}.{self._policy_name}.policy"
            )

        self._policy = PolicyClass(
            model=self._model,
            config=self._policy_config,
            gazebo_to_policy=self._gazebo_to_policy,
            policy_to_gazebo=self._policy_to_gazebo,
        )
        self._loaded = True

    def step(
        self, state: RobotState, sensors: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        if not self._enabled or not self._loaded:
            return None

        for sensor_name in self.required_sensor_topics():
            if sensor_name not in sensors:
                return self._last_safe_action.copy()

        if self._last_action is not None:
            state.previous_action = self._last_action.copy()

        raw_action = np.asarray(self._policy.infer(state, sensors), dtype=np.float32).reshape(-1)
        if raw_action.shape != (self._num_joints,):
            raise ValueError(
                f"Policy '{self._policy_name}' returned {raw_action.shape}, "
                f"expected {(self._num_joints,)}"
            )

        raw_action = np.clip(raw_action, self._clip_lower, self._clip_upper)

        action_gazebo = raw_action[self._policy_to_gazebo]

        self._last_action = action_gazebo.copy()

        action_scaled = action_gazebo * self._action_scale

        q_target = self._default_joint_pos + action_scaled

        self._last_safe_action = np.column_stack(
            [q_target, np.zeros(self._num_joints), np.zeros(self._num_joints), self._kp, self._kd]
        ).astype(np.float32)

        return self._last_safe_action

    def position_command(
        self,
        state: RobotState,
        target_pos: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ) -> np.ndarray:
        target_pos = np.asarray(target_pos, dtype=np.float32)
        kp = np.asarray(kp, dtype=np.float32)
        kd = np.asarray(kd, dtype=np.float32)
        if target_pos.shape != (self._num_joints,):
            raise ValueError(f"target_pos must have shape {(self._num_joints,)}")
        if kp.shape != (self._num_joints,) or kd.shape != (self._num_joints,):
            raise ValueError(f"kp and kd must have shape {(self._num_joints,)}")

        return np.column_stack(
            [target_pos, np.zeros(self._num_joints), np.zeros(self._num_joints), kp, kd]
        ).astype(np.float32)

    def reset(self) -> None:
        if self._policy is not None:
            self._policy.reset()
        self._last_action = np.zeros(self._num_joints, dtype=np.float32)
        self._last_safe_action = np.zeros((self._num_joints, 5), dtype=np.float32)

    def shutdown(self) -> None:
        self._enabled = False
        self._loaded = False
        self._policy = None
        self._model = None
