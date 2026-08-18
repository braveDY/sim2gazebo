import os
import select
import subprocess
import sys
import termios
import tempfile
import time
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Twist
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import Odometry
from rclpy.node import Node
from robot_msgs.msg import MotorCommand, RobotCommand, RobotState as RobotStateMsg
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty, SetBool, Trigger

from .policy_api import RobotState as PolicyRobotState
from .runtime import PolicyRuntime
from .terminal_ui import TerminalDashboard


class ControllerState(Enum):
    PASSIVE = "passive"
    GET_UP = "get_up"
    LOCOMOTION = "locomotion"
    GET_DOWN = "get_down"


class DeployNode(Node):
    def __init__(self) -> None:
        super().__init__("rl_policy_runtime")

        self.declare_parameter("robot", "go2")
        self.declare_parameter("policy", "quad_mwm")
        self.declare_parameter("policy_root", "")
        self.declare_parameter("control_enabled", False)
        self.declare_parameter("command_topic", "/robot_joint_controller/command")
        self.declare_parameter("robot_state_topic", "/robot_joint_controller/state")
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("keyboard_enabled", True)
        self.declare_parameter("terminal_ui_enabled", True)
        self.declare_parameter("terminal_ui_hz", 10.0)
        self.declare_parameter("keyboard_command_step", 0.1)
        self.declare_parameter("fixed_kp", 80.0)
        self.declare_parameter("fixed_kd", 3.0)
        self.declare_parameter("get_up_pre_duration", 1.0)
        self.declare_parameter("get_up_duration", 2.0)
        self.declare_parameter("get_down_duration", 2.0)
        self.declare_parameter("start_joint_controller", True)
        self.declare_parameter("reset_world_on_start", False)

        robot = self.get_parameter("robot").value
        policy_name = self.get_parameter("policy").value
        policy_root = self.get_parameter("policy_root").value
        control_enabled = self._parse_bool(
            self.get_parameter("control_enabled").value
        )

        if not policy_root:
            try:
                from ament_index_python.packages import get_package_share_directory
                policy_root = os.path.join(
                    get_package_share_directory("rl_policy_runtime"), "policy"
                )
            except Exception:
                policy_root = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                    "policy",
                )
        self.get_logger().info(f"Policy root: {policy_root}")

        try:
            self._runtime = PolicyRuntime(policy_root, policy_name, robot)
        except Exception as e:
            self.get_logger().fatal(f"Failed to initialize runtime: {e}")
            raise

        self._runtime.enabled = control_enabled
        self._robot_name = robot
        self._policy_name = policy_name

        self._control_hz = self._runtime.control_hz
        self._control_period = self._runtime.control_period

        self._latest_imu: Optional[Imu] = None
        self._latest_odom: Optional[Odometry] = None
        self._latest_cmd_vel: Optional[Twist] = None
        self._latest_robot_state: Optional[RobotStateMsg] = None
        self._latest_sensors: Dict[str, Any] = {}
        self._sensor_timestamps: Dict[str, float] = {}
        self._sensor_subscriptions: Dict[str, Any] = {}
        self._num_joints = self._runtime.manifest["robot"]["num_joints"]
        fsm_config = self._runtime.manifest.get("fsm", {})
        self._fsm_standing_joint_pos = np.asarray(
            fsm_config.get("standing_joint_pos", self._runtime.default_joint_pos),
            dtype=np.float32,
        )
        if self._fsm_standing_joint_pos.shape != (self._num_joints,):
            raise ValueError("fsm.standing_joint_pos must have one value per joint")
        self._controller_state = (
            ControllerState.LOCOMOTION if control_enabled else ControllerState.PASSIVE
        )
        self._state_started_at = time.monotonic()
        self._transition_start_pos: Optional[np.ndarray] = None
        self._get_up_origin: Optional[np.ndarray] = None
        self._get_up_complete = False
        self._stand_from_passive = True
        self._keyboard_command = np.zeros(3, dtype=np.float32)
        self._navigation_mode = False
        self._keyboard_enabled = self._parse_bool(
            self.get_parameter("keyboard_enabled").value
        )
        self._keyboard_step = float(self.get_parameter("keyboard_command_step").value)
        self._fixed_kp = self._as_joint_vector(
            fsm_config.get("fixed_kp", self.get_parameter("fixed_kp").value), "fsm.fixed_kp"
        )
        self._fixed_kd = self._as_joint_vector(
            fsm_config.get("fixed_kd", self.get_parameter("fixed_kd").value), "fsm.fixed_kd"
        )
        self._get_up_pre_duration = float(self.get_parameter("get_up_pre_duration").value)
        self._get_up_duration = float(self.get_parameter("get_up_duration").value)
        self._get_down_duration = float(self.get_parameter("get_down_duration").value)
        self._pre_get_up_pos = np.asarray(
            fsm_config.get(
                "pre_get_up_pos",
                [0.0, 1.36, -2.65, 0.0, 1.36, -2.65,
                 0.0, 1.36, -2.65, 0.0, 1.36, -2.65],
            ),
            dtype=np.float32,
        )
        if self._pre_get_up_pos.shape != (self._num_joints,):
            raise ValueError("fsm.pre_get_up_pos must have one value per joint")
        self._terminal_settings = None
        self._terminal_dashboard: Optional[TerminalDashboard] = None
        self._last_key = "-"

        self._command_pub = self.create_publisher(
            RobotCommand, self.get_parameter("command_topic").value, 10
        )

        if self._parse_bool(self.get_parameter("start_joint_controller").value):
            self._start_joint_controller()

        self._imu_sub = self.create_subscription(
            Imu,
            self.get_parameter("imu_topic").value,
            self._imu_callback,
            rclpy.qos.qos_profile_sensor_data,
        )
        self._odom_sub = self.create_subscription(
            Odometry,
            self.get_parameter("odom_topic").value,
            self._odom_callback,
            rclpy.qos.qos_profile_sensor_data,
        )
        self._cmd_vel_sub = self.create_subscription(
            Twist,
            self.get_parameter("cmd_vel_topic").value,
            self._cmd_vel_callback,
            rclpy.qos.qos_profile_sensor_data,
        )
        self._robot_state_sub = self.create_subscription(
            RobotStateMsg,
            self.get_parameter("robot_state_topic").value,
            self._robot_state_callback,
            rclpy.qos.qos_profile_sensor_data,
        )

        for sensor_name, topic in self._runtime.sensor_topics().items():
            self._subscribe_sensor(sensor_name, topic)

        self._enable_srv = self.create_service(
            SetBool, "~/enable", self._enable_callback
        )
        self._reset_srv = self.create_service(
            Trigger, "~/reset", self._reset_callback
        )
        self._gazebo_reset_client = self.create_client(Empty, "/reset_world")
        self._gazebo_pause_client = self.create_client(Empty, "/pause_physics")
        self._gazebo_unpause_client = self.create_client(Empty, "/unpause_physics")
        self._simulation_running = True
        if self._parse_bool(self.get_parameter("reset_world_on_start").value):
            self._call_empty_service(self._gazebo_reset_client, "/reset_world")

        self._timer = self.create_timer(self._control_period, self._control_loop)
        self._keyboard_timer = self.create_timer(0.05, self._poll_keyboard)

        if self._keyboard_enabled:
            self._configure_terminal()
        if (
            self._keyboard_enabled
            and self._parse_bool(self.get_parameter("terminal_ui_enabled").value)
        ):
            terminal_ui_hz = float(self.get_parameter("terminal_ui_hz").value)
            if terminal_ui_hz <= 0:
                raise ValueError("terminal_ui_hz must be positive")
            self._terminal_dashboard = TerminalDashboard(robot, policy_name)
            self._terminal_dashboard.start()
            self._terminal_ui_timer = self.create_timer(
                1.0 / terminal_ui_hz, self._refresh_terminal_dashboard
            )

        self.get_logger().info(
            f"DeployNode started: robot={robot}, policy={policy_name}, "
            f"control_hz={self._control_hz}, state={self._controller_state.value}. "
            "Keys: 0=get up, 1=locomotion, 9=get down, P=passive, "
            "W/S/A/D/Q/E=command, Space=clear, N=toggle cmd_vel"
        )

    def _subscribe_sensor(self, sensor_name: str, topic: str) -> None:
        sensor_type = self._runtime.sensor_type(sensor_name)
        qos = rclpy.qos.qos_profile_sensor_data

        if sensor_type == "grid_map":
            def callback(msg: GridMap) -> None:
                self._latest_sensors[sensor_name] = msg
                self._sensor_timestamps[sensor_name] = time.time()

            sub = self.create_subscription(GridMap, topic, callback, qos)
        else:
            def make_callback(name: str):
                def callback(msg: Float32MultiArray) -> None:
                    value = np.asarray(msg.data, dtype=np.float32)
                    expected_shape = self._runtime.sensor_shape(name)
                    if expected_shape is not None and value.size != int(np.prod(expected_shape)):
                        self.get_logger().warning(
                            f"Ignoring {name}: got {value.size} values, "
                            f"expected {int(np.prod(expected_shape))}"
                        )
                        return
                    self._latest_sensors[name] = value
                    self._sensor_timestamps[name] = time.time()
                return callback

            sub = self.create_subscription(
                Float32MultiArray, topic, make_callback(sensor_name), qos
            )

        self._sensor_subscriptions[sensor_name] = sub

    def _start_joint_controller(self) -> None:
        controller_name = "robot_joint_controller"
        joints = self._runtime.manifest["robot"].get("joint_names")
        if not joints:
            raise ValueError("manifest robot.joint_names is required to start robot_joint_controller")

        controller_config = {
            f"/{controller_name}": {"ros__parameters": {"joints": joints}}
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            yaml.safe_dump(controller_config, config_file)
            config_path = config_file.name
        try:
            result = subprocess.run(
                [
                    "ros2", "run", "controller_manager", "spawner", controller_name,
                    "-p", config_path,
                    "--controller-manager", "/controller_manager",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=20,
                check=False,
            )
        finally:
            os.unlink(config_path)

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start {controller_name}: {result.stdout.strip()}"
            )
        self.get_logger().info(f"Started {controller_name}")

    def _imu_callback(self, msg: Imu) -> None:
        self._latest_imu = msg

    def _odom_callback(self, msg: Odometry) -> None:
        self._latest_odom = msg

    def _cmd_vel_callback(self, msg: Twist) -> None:
        self._latest_cmd_vel = msg

    def _robot_state_callback(self, msg: RobotStateMsg) -> None:
        self._latest_robot_state = msg

    def _enable_callback(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        try:
            self._runtime.enabled = request.data
        except Exception as exc:
            response.success = False
            response.message = f"Failed to enable policy: {exc}"
            self.get_logger().error(response.message)
            return response
        if request.data:
            self._set_controller_state(ControllerState.LOCOMOTION)
        else:
            self._set_controller_state(ControllerState.PASSIVE)
        response.success = True
        response.message = f"Policy {'enabled' if request.data else 'disabled'}"
        self.get_logger().info(response.message)
        return response

    def _reset_callback(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        self._runtime.reset()
        response.success = True
        response.message = "Policy reset"
        self.get_logger().info(response.message)
        return response

    def _build_robot_state(self) -> Optional[PolicyRobotState]:
        if self._latest_imu is None or self._latest_robot_state is None:
            return None

        imu = self._latest_imu
        robot_state = self._latest_robot_state

        joint_pos = np.array(
            [m.q for m in robot_state.motor_state], dtype=np.float32
        )
        joint_vel = np.array(
            [m.dq for m in robot_state.motor_state], dtype=np.float32
        )

        quaternion = np.array(
            [
                imu.orientation.w,
                imu.orientation.x,
                imu.orientation.y,
                imu.orientation.z,
            ],
            dtype=np.float32,
        )

        angular_velocity = np.array(
            [
                imu.angular_velocity.x,
                imu.angular_velocity.y,
                imu.angular_velocity.z,
            ],
            dtype=np.float32,
        )

        base_position = np.zeros(3, dtype=np.float32)
        linear_velocity = np.zeros(3, dtype=np.float32)
        if self._latest_odom is not None:
            odom = self._latest_odom
            base_position = np.array(
                [
                    odom.pose.pose.position.x,
                    odom.pose.pose.position.y,
                    odom.pose.pose.position.z,
                ],
                dtype=np.float32,
            )
            linear_velocity = np.array(
                [
                    odom.twist.twist.linear.x,
                    odom.twist.twist.linear.y,
                    odom.twist.twist.linear.z,
                ],
                dtype=np.float32,
            )

        command = self._keyboard_command.copy()
        if self._navigation_mode and self._latest_cmd_vel is not None:
            cmd = self._latest_cmd_vel
            command = np.array(
                [cmd.linear.x, cmd.linear.y, cmd.angular.z], dtype=np.float32
            )

        return PolicyRobotState(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            quaternion_wxyz=quaternion,
            angular_velocity=angular_velocity,
            linear_velocity=linear_velocity,
            command=command,
            base_position=base_position,
        )

    def _check_sensor_freshness(self) -> bool:
        return self._sensor_health()[0]

    def _sensor_health(self) -> tuple[bool, str]:
        now = time.time()
        health = []
        for sensor_name in self._runtime.required_sensor_topics():
            timeout = self._runtime.sensor_timeout(sensor_name)
            if sensor_name not in self._sensor_timestamps:
                health.append(f"{sensor_name}: waiting")
                continue
            age = now - self._sensor_timestamps[sensor_name]
            if age > timeout:
                health.append(f"{sensor_name}: stale {age * 1000:.0f} ms")
            else:
                health.append(f"{sensor_name}: OK {age * 1000:.0f} ms")
        is_fresh = all("OK" in item for item in health) if health else True
        return is_fresh, " | ".join(health) if health else "no external sensors"

    def _dashboard_phase(self) -> tuple[str, float]:
        elapsed = time.monotonic() - self._state_started_at
        if self._controller_state == ControllerState.GET_UP:
            if self._stand_from_passive and elapsed < self._get_up_pre_duration:
                return "Pre getting up", elapsed / self._get_up_pre_duration
            if self._stand_from_passive and not self._get_up_complete:
                return "Getting up", (elapsed - self._get_up_pre_duration) / self._get_up_duration
            if not self._get_up_complete:
                return "Getting up", elapsed / self._get_up_duration
            return "Ready: press 1 for locomotion", 1.0
        if self._controller_state == ControllerState.GET_DOWN:
            return "Getting down", elapsed / self._get_down_duration
        if self._controller_state == ControllerState.LOCOMOTION:
            return "Locomotion active", 1.0
        return "Passive standing hold", 1.0

    def _dashboard_command(self) -> tuple[str, tuple]:
        if self._navigation_mode and self._latest_cmd_vel is not None:
            return "cmd_vel", (
                self._latest_cmd_vel.linear.x,
                self._latest_cmd_vel.linear.y,
                self._latest_cmd_vel.angular.z,
            )
        source = "keyboard" if not self._navigation_mode else "keyboard (cmd_vel waiting)"
        return source, tuple(float(value) for value in self._keyboard_command)

    def _refresh_terminal_dashboard(self) -> None:
        if self._terminal_dashboard is None:
            return
        phase, progress = self._dashboard_phase()
        source, command = self._dashboard_command()
        _, sensor_health = self._sensor_health()
        self._terminal_dashboard.update(
            self._controller_state.value,
            phase,
            float(np.clip(progress, 0.0, 1.0)),
            source,
            command,
            sensor_health,
            self._last_key,
            self._runtime.enabled,
        )

    def _control_loop(self) -> None:
        state = self._build_robot_state()
        if state is None:
            return

        result = self._command_for_state(state)
        if result is None:
            return

        cmd = RobotCommand()
        cmd.motor_command = [
            MotorCommand(
                q=float(result[i, 0]),
                dq=float(result[i, 1]),
                tau=float(result[i, 2]),
                kp=float(result[i, 3]),
                kd=float(result[i, 4]),
            )
            for i in range(self._num_joints)
        ]
        self._command_pub.publish(cmd)

    def _command_for_state(self, state: PolicyRobotState) -> Optional[np.ndarray]:
        if self._controller_state == ControllerState.PASSIVE:
            target = self._fsm_standing_joint_pos if self._get_up_complete else state.joint_pos
            return self._runtime.position_command(
                state, target, self._fixed_kp, self._fixed_kd
            )

        elapsed = time.monotonic() - self._state_started_at
        if self._controller_state == ControllerState.GET_UP:
            origin = self._get_up_origin if self._get_up_origin is not None else state.joint_pos
            if not self._get_up_complete and self._stand_from_passive and elapsed < self._get_up_pre_duration:
                ratio = elapsed / self._get_up_pre_duration
                target = self._interpolate(origin, self._pre_get_up_pos, ratio)
            elif not self._get_up_complete and self._stand_from_passive and elapsed < self._get_up_pre_duration + self._get_up_duration:
                ratio = (elapsed - self._get_up_pre_duration) / self._get_up_duration
                target = self._interpolate(self._pre_get_up_pos, self._fsm_standing_joint_pos, ratio)
            elif not self._get_up_complete and not self._stand_from_passive and elapsed < self._get_up_duration:
                target = self._interpolate(origin, self._fsm_standing_joint_pos, elapsed / self._get_up_duration)
            else:
                self._get_up_complete = True
                target = self._fsm_standing_joint_pos
            return self._runtime.position_command(state, target, self._fixed_kp, self._fixed_kd)

        if self._controller_state == ControllerState.GET_DOWN:
            origin = self._transition_start_pos if self._transition_start_pos is not None else state.joint_pos
            ratio = min(elapsed / self._get_down_duration, 1.0)
            target = self._interpolate(origin, self._get_up_origin if self._get_up_origin is not None else self._pre_get_up_pos, ratio)
            if ratio >= 1.0:
                self._set_controller_state(ControllerState.PASSIVE)
            return self._runtime.position_command(state, target, self._fixed_kp, self._fixed_kd)

        if not self._runtime.enabled or not self._check_sensor_freshness():
            return self._runtime.position_command(
                state, self._fsm_standing_joint_pos, self._fixed_kp, self._fixed_kd
            )
        try:
            return self._runtime.step(state, dict(self._latest_sensors))
        except Exception as exc:
            self.get_logger().error(f"Policy step failed; switching to passive: {exc}")
            self._runtime.enabled = False
            self._set_controller_state(ControllerState.PASSIVE)
            return self._runtime.position_command(
                state, self._fsm_standing_joint_pos, self._fixed_kp, self._fixed_kd
            )

    def _set_controller_state(self, new_state: ControllerState) -> None:
        if new_state == self._controller_state:
            return
        self._controller_state = new_state
        self._state_started_at = time.monotonic()
        if new_state == ControllerState.GET_UP:
            self._get_up_complete = False
        self.get_logger().info(f"Controller state: {new_state.value}")

    def _poll_keyboard(self) -> None:
        if not self._keyboard_enabled or not sys.stdin.isatty():
            return
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        except (OSError, ValueError) as exc:
            self._disable_keyboard(f"Keyboard input unavailable: {exc}")
            return
        if not ready:
            return
        try:
            key = os.read(sys.stdin.fileno(), 1).decode(errors="ignore")
        except OSError as exc:
            self._disable_keyboard(f"Keyboard input unavailable: {exc}")
            return
        if not key:
            self._disable_keyboard("Keyboard input closed")
            return
        self._handle_key(key)

    def _handle_key(self, key: str) -> None:
        key_lower = key.lower()
        self._last_key = "Space" if key == " " else ("Enter" if key in {"\n", "\r"} else key.upper())
        if key == "0":
            state = self._build_robot_state()
            self._get_up_origin = state.joint_pos.copy() if state else None
            self._stand_from_passive = self._controller_state == ControllerState.PASSIVE
            self._runtime.enabled = False
            self._set_controller_state(ControllerState.GET_UP)
        elif key == "1":
            if self._controller_state == ControllerState.GET_UP and self._get_up_complete:
                try:
                    self._runtime.enabled = True
                    self._runtime.reset()
                    self._set_controller_state(ControllerState.LOCOMOTION)
                except Exception as exc:
                    self.get_logger().error(f"Failed to enable policy: {exc}")
            else:
                self.get_logger().warning("Press 1 only after get_up completes")
        elif key == "9":
            if self._controller_state in {ControllerState.GET_UP, ControllerState.LOCOMOTION}:
                state = self._build_robot_state()
                self._transition_start_pos = state.joint_pos.copy() if state else None
                self._runtime.enabled = False
                self._set_controller_state(ControllerState.GET_DOWN)
        elif key_lower == "p":
            self._runtime.enabled = False
            self._set_controller_state(ControllerState.PASSIVE)
        elif key_lower == "w":
            self._keyboard_command[0] += self._keyboard_step
        elif key_lower == "s":
            self._keyboard_command[0] -= self._keyboard_step
        elif key_lower == "a":
            self._keyboard_command[1] += self._keyboard_step
        elif key_lower == "d":
            self._keyboard_command[1] -= self._keyboard_step
        elif key_lower == "q":
            self._keyboard_command[2] += self._keyboard_step
        elif key_lower == "e":
            self._keyboard_command[2] -= self._keyboard_step
        elif key == " ":
            self._keyboard_command.fill(0.0)
        elif key_lower == "n":
            self._navigation_mode = not self._navigation_mode
            self.get_logger().info(f"Navigation mode: {'ON' if self._navigation_mode else 'OFF'}")
        elif key_lower == "r":
            self._call_empty_service(self._gazebo_reset_client, "/reset_world")
        elif key in {"\n", "\r"}:
            client = self._gazebo_pause_client if self._simulation_running else self._gazebo_unpause_client
            service_name = "/pause_physics" if self._simulation_running else "/unpause_physics"
            self._call_empty_service(client, service_name)
            self._simulation_running = not self._simulation_running

    def _configure_terminal(self) -> None:
        try:
            self._terminal_settings = termios.tcgetattr(sys.stdin.fileno())
            settings = termios.tcgetattr(sys.stdin.fileno())
            settings[3] &= ~(termios.ICANON | termios.ECHO)
            settings[6][termios.VMIN] = 0
            settings[6][termios.VTIME] = 0
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, settings)
        except termios.error as exc:
            self._disable_keyboard(f"Terminal keyboard disabled: {exc}")

    def _restore_terminal(self) -> None:
        if self._terminal_settings is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, self._terminal_settings)
            except termios.error:
                pass
            self._terminal_settings = None

    def _disable_keyboard(self, message: str) -> None:
        if self._keyboard_enabled:
            self.get_logger().warning(message)
        self._keyboard_enabled = False
        if self._terminal_dashboard is not None:
            self._terminal_dashboard.stop()
            self._terminal_dashboard = None
        self._restore_terminal()

    def destroy_node(self):
        if self._terminal_dashboard is not None:
            self._terminal_dashboard.stop()
            self._terminal_dashboard = None
        self._restore_terminal()
        return super().destroy_node()

    @staticmethod
    def _interpolate(start: np.ndarray, target: np.ndarray, ratio: float) -> np.ndarray:
        return start + np.clip(ratio, 0.0, 1.0) * (target - start)

    def _call_empty_service(self, client, name: str) -> None:
        if not client.service_is_ready():
            self.get_logger().warning(f"Gazebo service unavailable: {name}")
            return
        client.call_async(Empty.Request())

    def _as_joint_vector(self, value: Any, field_name: str) -> np.ndarray:
        if np.isscalar(value):
            return np.full(self._num_joints, float(value), dtype=np.float32)
        result = np.asarray(value, dtype=np.float32)
        if result.shape != (self._num_joints,):
            raise ValueError(f"{field_name} must be a scalar or {self._num_joints}-element list")
        return result

    @staticmethod
    def _parse_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}


def main(args=None):
    rclpy.init(args=args)
    node = DeployNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
