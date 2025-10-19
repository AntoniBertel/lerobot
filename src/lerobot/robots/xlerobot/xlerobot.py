#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from functools import cached_property
from itertools import chain
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_xlerobot import XLerobotConfig

logger = logging.getLogger(__name__)


class XLerobot(Robot):
    """
    The robot includes a three omniwheel mobile base and a remote follower arm.
    The leader arm is connected locally (on the laptop) and its joint positions are recorded and then
    forwarded to the remote follower arm (after applying a safety clamp).
    In parallel, keyboard teleoperation is used to generate raw velocity commands for the wheels.
    """

    config_class = XLerobotConfig
    name = "xlerobot"

    def __init__(self, config: XLerobotConfig):
        super().__init__(config)
        self.config = config
        self.teleop_keys = config.teleop_keys
        # Define three speed levels and a current index
        self.speed_levels = [
            {"xy": 0.1, "theta": 30},  # slow
            {"xy": 0.2, "theta": 60},  # medium
            {"xy": 0.3, "theta": 90},  # fast
        ]
        self.speed_index = 0  # Start at slow
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        
        # Extract calibration for each bus
        calibration_left_arm = {
            k: v for k, v in self.calibration.items() if k.startswith("left_arm_")
        } if self.calibration else {}
        
        calibration_right_arm = {
            k: v for k, v in self.calibration.items() if k.startswith("right_arm_")
        } if self.calibration else {}
        
        calibration_head = {
            k: v for k, v in self.calibration.items() if k.startswith("head_")
        } if self.calibration else {}
        
        calibration_base = {
            k: v for k, v in self.calibration.items() if k.startswith("base_")
        } if self.calibration else {}
        
        # Bus 1: Left hand/arm
        self.bus1 = FeetechMotorsBus(
            port=self.config.port1,
            motors={
                "left_arm_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "left_arm_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "left_arm_elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "left_arm_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "left_arm_wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "left_arm_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=calibration_left_arm,
        )
        
        # Bus 2: Right hand/arm
        self.bus2 = FeetechMotorsBus(
            port=self.config.port2,
            motors={
                "right_arm_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "right_arm_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "right_arm_elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "right_arm_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "right_arm_wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "right_arm_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=calibration_right_arm,
        )
        
        # Bus 3: Neck and camera
        self.bus3 = FeetechMotorsBus(
            port=self.config.port3,
            motors={
                "head_motor_1": Motor(1, "sts3215", norm_mode_body),
                "head_motor_2": Motor(2, "sts3215", norm_mode_body),
            },
            calibration=calibration_head,
        )
        
        # Bus 4: Wheels
        self.bus4 = FeetechMotorsBus(
            port=self.config.port4,
            motors={
                "base_left_wheel": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_back_wheel": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_right_wheel": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
            },
            calibration=calibration_base,
        )
        
        self.left_arm_motors = list(self.bus1.motors.keys())
        self.right_arm_motors = list(self.bus2.motors.keys())
        self.head_motors = list(self.bus3.motors.keys())
        self.base_motors = list(self.bus4.motors.keys())
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _state_ft(self) -> dict[str, type]:
        return dict.fromkeys(
            (
                "left_arm_shoulder_pan.pos",
                "left_arm_shoulder_lift.pos",
                "left_arm_elbow_flex.pos",
                "left_arm_wrist_flex.pos",
                "left_arm_wrist_roll.pos",
                "left_arm_gripper.pos",
                "right_arm_shoulder_pan.pos",
                "right_arm_shoulder_lift.pos",
                "right_arm_elbow_flex.pos",
                "right_arm_wrist_flex.pos",
                "right_arm_wrist_roll.pos",
                "right_arm_gripper.pos",
                "head_motor_1.pos",
                "head_motor_2.pos",
                "x.vel",
                "y.vel",
                "theta.vel",
            ),
            float,
        )

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        return (
            self.bus1.is_connected 
            and self.bus2.is_connected 
            and self.bus3.is_connected 
            and self.bus4.is_connected 
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus1.connect()
        self.bus2.connect()
        self.bus3.connect()
        self.bus4.connect()
        
        if not self.is_calibrated and calibrate:
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return (
            self.bus1.is_calibrated 
            and self.bus2.is_calibrated 
            and self.bus3.is_calibrated 
            and self.bus4.is_calibrated
        )

    def calibrate(self) -> None:
        logger.info(f"\nRunning calibration of {self}")
        
        # Calibrate Bus 1: Left arm
        self.bus1.disable_torque()
        for name in self.left_arm_motors:
            self.bus1.write("Operating_Mode", name, OperatingMode.POSITION.value)
        input("Move left arm motors to the middle of their range of motion and press ENTER....")
        homing_offsets_left = self.bus1.set_half_turn_homings(self.left_arm_motors)
        
        print(
            "Move all left arm joints sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins_left, range_maxes_left = self.bus1.record_ranges_of_motion(self.left_arm_motors)
        
        calibration_left = {}
        for name, motor in self.bus1.motors.items():
            calibration_left[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=homing_offsets_left[name],
                range_min=range_mins_left[name],
                range_max=range_maxes_left[name],
            )
        self.bus1.write_calibration(calibration_left)
        
        # Calibrate Bus 2: Right arm
        self.bus2.disable_torque()
        for name in self.right_arm_motors:
            self.bus2.write("Operating_Mode", name, OperatingMode.POSITION.value)
        input("Move right arm motors to the middle of their range of motion and press ENTER....")
        homing_offsets_right = self.bus2.set_half_turn_homings(self.right_arm_motors)
        
        print(
            "Move all right arm joints sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins_right, range_maxes_right = self.bus2.record_ranges_of_motion(self.right_arm_motors)
        
        calibration_right = {}
        for name, motor in self.bus2.motors.items():
            calibration_right[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=homing_offsets_right[name],
                range_min=range_mins_right[name],
                range_max=range_maxes_right[name],
            )
        self.bus2.write_calibration(calibration_right)
        
        # Calibrate Bus 3: Head/neck motors
        self.bus3.disable_torque()
        for name in self.head_motors:
            self.bus3.write("Operating_Mode", name, OperatingMode.POSITION.value)
        input("Move head motors to the middle of their range of motion and press ENTER....")
        homing_offsets_head = self.bus3.set_half_turn_homings(self.head_motors)
        
        print(
            "Move all head joints sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins_head, range_maxes_head = self.bus3.record_ranges_of_motion(self.head_motors)
        
        calibration_head = {}
        for name, motor in self.bus3.motors.items():
            calibration_head[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=homing_offsets_head[name],
                range_min=range_mins_head[name],
                range_max=range_maxes_head[name],
            )
        self.bus3.write_calibration(calibration_head)
        
        # Calibrate Bus 4: Base wheels (full rotation motors, no homing offset needed)
        # Wheels have full rotation capability
        calibration_base = {}
        for name, motor in self.bus4.motors.items():
            calibration_base[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=0,
                range_min=0,
                range_max=4095,
            )
        self.bus4.write_calibration(calibration_base)
        
        # Combine all calibrations and save
        self.calibration = {**calibration_left, **calibration_right, **calibration_head, **calibration_base}
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)
        

    def configure(self):
        # Set-up all actuators
        # We assume that at connection time, robot is in a rest position,
        # and torque can be safely disabled to run configuration
        self.bus1.disable_torque()
        self.bus2.disable_torque()
        self.bus3.disable_torque()
        self.bus4.disable_torque()
        
        self.bus1.configure_motors()
        self.bus2.configure_motors()
        self.bus3.configure_motors()
        self.bus4.configure_motors()
        
        # Configure Bus 1: Left arm motors (position mode)
        for name in self.left_arm_motors:
            self.bus1.write("Operating_Mode", name, OperatingMode.POSITION.value)
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            self.bus1.write("P_Coefficient", name, 16)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            self.bus1.write("I_Coefficient", name, 0)
            self.bus1.write("D_Coefficient", name, 43)
        
        # Configure Bus 2: Right arm motors (position mode)
        for name in self.right_arm_motors:
            self.bus2.write("Operating_Mode", name, OperatingMode.POSITION.value)
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            self.bus2.write("P_Coefficient", name, 16)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            self.bus2.write("I_Coefficient", name, 0)
            self.bus2.write("D_Coefficient", name, 43)
        
        # Configure Bus 3: Head motors (position mode)
        for name in self.head_motors:
            self.bus3.write("Operating_Mode", name, OperatingMode.POSITION.value)
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            self.bus3.write("P_Coefficient", name, 16)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            self.bus3.write("I_Coefficient", name, 0)
            self.bus3.write("D_Coefficient", name, 43)
        
        # Configure Bus 4: Base wheel motors (velocity mode)
        for name in self.base_motors:
            self.bus4.write("Operating_Mode", name, OperatingMode.VELOCITY.value)
        
        # Enable torque on all buses
        self.bus1.enable_torque()
        self.bus2.enable_torque()
        self.bus3.enable_torque()
        self.bus4.enable_torque()
        

    def setup_motors(self) -> None:
        # Setup Bus 1: Left arm motors (only if port1 is provided)
        if self.config.port1:
            print("\n=== Setting up Bus 1: Left Arm Motors ===")
            for motor in reversed(self.left_arm_motors):
                input(f"Connect the controller board to the '{motor}' motor only and press enter.")
                self.bus1.setup_motor(motor)
                print(f"'{motor}' motor id set to {self.bus1.motors[motor].id}")
        else:
            print("\n⏭️  Skipping Bus 1 (Left Arm) - no port1 provided")
        
        # Setup Bus 2: Right arm motors (only if port2 is provided)
        if self.config.port2:
            print("\n=== Setting up Bus 2: Right Arm Motors ===")
            for motor in reversed(self.right_arm_motors):
                input(f"Connect the controller board to the '{motor}' motor only and press enter.")
                self.bus2.setup_motor(motor)
                print(f"'{motor}' motor id set to {self.bus2.motors[motor].id}")
        else:
            print("\n⏭️  Skipping Bus 2 (Right Arm) - no port2 provided")
        
        # Setup Bus 3: Head motors (only if port3 is provided)
        if self.config.port3:
            print("\n=== Setting up Bus 3: Head Motors ===")
            for motor in reversed(self.head_motors):
                input(f"Connect the controller board to the '{motor}' motor only and press enter.")
                self.bus3.setup_motor(motor)
                print(f"'{motor}' motor id set to {self.bus3.motors[motor].id}")
        else:
            print("\n⏭️  Skipping Bus 3 (Head) - no port3 provided")
        
        # Setup Bus 4: Base wheel motors (only if port4 is provided)
        if self.config.port4:
            print("\n=== Setting up Bus 4: Base Wheel Motors ===")
            for motor in reversed(self.base_motors):
                input(f"Connect the controller board to the '{motor}' motor only and press enter.")
                self.bus4.setup_motor(motor)
                print(f"'{motor}' motor id set to {self.bus4.motors[motor].id}")
        else:
            print("\n⏭️  Skipping Bus 4 (Base Wheels) - no port4 provided")
        

    @staticmethod
    def _degps_to_raw(degps: float) -> int:
        steps_per_deg = 4096.0 / 360.0
        speed_in_steps = degps * steps_per_deg
        speed_int = int(round(speed_in_steps))
        # Cap the value to fit within signed 16-bit range (-32768 to 32767)
        if speed_int > 0x7FFF:
            speed_int = 0x7FFF  # 32767 -> maximum positive value
        elif speed_int < -0x8000:
            speed_int = -0x8000  # -32768 -> minimum negative value
        return speed_int

    @staticmethod
    def _raw_to_degps(raw_speed: int) -> float:
        steps_per_deg = 4096.0 / 360.0
        magnitude = raw_speed
        degps = magnitude / steps_per_deg
        return degps

    def _body_to_wheel_raw(
        self,
        x: float,
        y: float,
        theta: float,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
        max_raw: int = 3000,
    ) -> dict:
        """
        Convert desired body-frame velocities into wheel raw commands.

        Parameters:
          x_cmd      : Linear velocity in x (m/s).
          y_cmd      : Linear velocity in y (m/s).
          theta_cmd  : Rotational velocity (deg/s).
          wheel_radius: Radius of each wheel (meters).
          base_radius : Distance from the center of rotation to each wheel (meters).
          max_raw    : Maximum allowed raw command (ticks) per wheel.

        Returns:
          A dictionary with wheel raw commands:
             {"base_left_wheel": value, "base_back_wheel": value, "base_right_wheel": value}.

        Notes:
          - Internally, the method converts theta_cmd to rad/s for the kinematics.
          - The raw command is computed from the wheels angular speed in deg/s
            using _degps_to_raw(). If any command exceeds max_raw, all commands
            are scaled down proportionally.
        """
        # Convert rotational velocity from deg/s to rad/s.
        theta_rad = theta * (np.pi / 180.0)
        # Create the body velocity vector [x, y, theta_rad].
        velocity_vector = np.array([x, y, theta_rad])

        # Define the wheel mounting angles with a -90° offset.
        angles = np.radians(np.array([240, 0, 120]) - 90)
        # Build the kinematic matrix: each row maps body velocities to a wheel’s linear speed.
        # The third column (base_radius) accounts for the effect of rotation.
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        # Compute each wheel’s linear speed (m/s) and then its angular speed (rad/s).
        wheel_linear_speeds = m.dot(velocity_vector)
        wheel_angular_speeds = wheel_linear_speeds / wheel_radius

        # Convert wheel angular speeds from rad/s to deg/s.
        wheel_degps = wheel_angular_speeds * (180.0 / np.pi)

        # Scaling
        steps_per_deg = 4096.0 / 360.0
        raw_floats = [abs(degps) * steps_per_deg for degps in wheel_degps]
        max_raw_computed = max(raw_floats)
        if max_raw_computed > max_raw:
            scale = max_raw / max_raw_computed
            wheel_degps = wheel_degps * scale

        # Convert each wheel’s angular speed (deg/s) to a raw integer.
        wheel_raw = [self._degps_to_raw(deg) for deg in wheel_degps]

        return {
            "base_left_wheel": wheel_raw[0],
            "base_back_wheel": wheel_raw[1],
            "base_right_wheel": wheel_raw[2],
        }

    def _wheel_raw_to_body(
        self,
        left_wheel_speed,
        back_wheel_speed,
        right_wheel_speed,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
    ) -> dict[str, Any]:
        """
        Convert wheel raw command feedback back into body-frame velocities.

        Parameters:
          wheel_raw   : Vector with raw wheel commands ("base_left_wheel", "base_back_wheel", "base_right_wheel").
          wheel_radius: Radius of each wheel (meters).
          base_radius : Distance from the robot center to each wheel (meters).

        Returns:
          A dict (x.vel, y.vel, theta.vel) all in m/s
        """

        # Convert each raw command back to an angular speed in deg/s.
        wheel_degps = np.array(
            [
                self._raw_to_degps(left_wheel_speed),
                self._raw_to_degps(back_wheel_speed),
                self._raw_to_degps(right_wheel_speed),
            ]
        )

        # Convert from deg/s to rad/s.
        wheel_radps = wheel_degps * (np.pi / 180.0)
        # Compute each wheel’s linear speed (m/s) from its angular speed.
        wheel_linear_speeds = wheel_radps * wheel_radius

        # Define the wheel mounting angles with a -90° offset.
        angles = np.radians(np.array([240, 0, 120]) - 90)
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        # Solve the inverse kinematics: body_velocity = M⁻¹ · wheel_linear_speeds.
        m_inv = np.linalg.inv(m)
        velocity_vector = m_inv.dot(wheel_linear_speeds)
        x, y, theta_rad = velocity_vector
        theta = theta_rad * (180.0 / np.pi)
        return {
            "x.vel": x,
            "y.vel": y,
            "theta.vel": theta,
        }  # m/s and deg/s
    
    def _from_keyboard_to_base_action(self, pressed_keys: np.ndarray):
        # Speed control
        if self.teleop_keys["speed_up"] in pressed_keys:
            self.speed_index = min(self.speed_index + 1, 2)
        if self.teleop_keys["speed_down"] in pressed_keys:
            self.speed_index = max(self.speed_index - 1, 0)
        speed_setting = self.speed_levels[self.speed_index]
        xy_speed = speed_setting["xy"]  # e.g. 0.1, 0.25, or 0.4
        theta_speed = speed_setting["theta"]  # e.g. 30, 60, or 90

        x_cmd = 0.0  # m/s forward/backward
        y_cmd = 0.0  # m/s lateral
        theta_cmd = 0.0  # deg/s rotation

        if self.teleop_keys["forward"] in pressed_keys:
            x_cmd += xy_speed
        if self.teleop_keys["backward"] in pressed_keys:
            x_cmd -= xy_speed
        if self.teleop_keys["left"] in pressed_keys:
            y_cmd += xy_speed
        if self.teleop_keys["right"] in pressed_keys:
            y_cmd -= xy_speed
        if self.teleop_keys["rotate_left"] in pressed_keys:
            theta_cmd += theta_speed
        if self.teleop_keys["rotate_right"] in pressed_keys:
            theta_cmd -= theta_speed
            
        return {
            # "head_motor_1.pos": 0.0,  # Head motors are not controlled by keyboard
            # "head_motor_2.pos": 0.0,  # TODO: implement head control
            "x.vel": x_cmd, 
            "y.vel": y_cmd,
            "theta.vel": theta_cmd,
        }

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read actuators position for arms and head, velocity for base
        start = time.perf_counter()
        left_arm_pos = self.bus1.sync_read("Present_Position", self.left_arm_motors)
        right_arm_pos = self.bus2.sync_read("Present_Position", self.right_arm_motors)
        head_pos = self.bus3.sync_read("Present_Position", self.head_motors)
        base_wheel_vel = self.bus4.sync_read("Present_Velocity", self.base_motors)
        
        base_vel = self._wheel_raw_to_body(
            base_wheel_vel["base_left_wheel"],
            base_wheel_vel["base_back_wheel"],
            base_wheel_vel["base_right_wheel"],
        )
        
        left_arm_state = {f"{k}.pos": v for k, v in left_arm_pos.items()}
        right_arm_state = {f"{k}.pos": v for k, v in right_arm_pos.items()}
        head_state = {f"{k}.pos": v for k, v in head_pos.items()}
        # Combine all arm and head states
        obs_dict = {**left_arm_state, **right_arm_state, **head_state, **base_vel}

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command lekiwi to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            np.ndarray: the action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        left_arm_pos = {k: v for k, v in action.items() if k.startswith("left_arm_") and k.endswith(".pos")}
        right_arm_pos = {k: v for k, v in action.items() if k.startswith("right_arm_") and k.endswith(".pos")}
        head_pos = {k: v for k, v in action.items() if k.startswith("head_") and k.endswith(".pos")}
        base_goal_vel = {k: v for k, v in action.items() if k.endswith(".vel")}
        base_wheel_goal_vel = self._body_to_wheel_raw(
            base_goal_vel.get("x.vel", 0.0),
            base_goal_vel.get("y.vel", 0.0),
            base_goal_vel.get("theta.vel", 0.0),
        )
        
        
        if self.config.max_relative_target is not None:
            # Read present positions for left arm, right arm, and head
            present_pos_left = self.bus1.sync_read("Present_Position", self.left_arm_motors)
            present_pos_right = self.bus2.sync_read("Present_Position", self.right_arm_motors)
            present_pos_head = self.bus3.sync_read("Present_Position", self.head_motors)

            # Combine all present positions
            present_pos = {**present_pos_left, **present_pos_right, **present_pos_head}

            # Ensure safe goal position for each arm and head
            goal_present_pos = {
                key: (g_pos, present_pos[key]) for key, g_pos in chain(left_arm_pos.items(), right_arm_pos.items(), head_pos.items())
            }
            safe_goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

            # Update the action with the safe goal positions
            left_arm_pos = {k: v for k, v in safe_goal_pos.items() if k in left_arm_pos}
            right_arm_pos = {k: v for k, v in safe_goal_pos.items() if k in right_arm_pos}
            head_pos = {k: v for k, v in safe_goal_pos.items() if k in head_pos}
        
        left_arm_pos_raw = {k.replace(".pos", ""): v for k, v in left_arm_pos.items()}
        right_arm_pos_raw = {k.replace(".pos", ""): v for k, v in right_arm_pos.items()}
        head_pos_raw = {k.replace(".pos", ""): v for k, v in head_pos.items()}
        
        # Only sync_write if there are motors to write to
        if left_arm_pos_raw:
            self.bus1.sync_write("Goal_Position", left_arm_pos_raw)
        if right_arm_pos_raw:
            self.bus2.sync_write("Goal_Position", right_arm_pos_raw)
        if head_pos_raw:
            self.bus3.sync_write("Goal_Position", head_pos_raw)
        if base_wheel_goal_vel:
            self.bus4.sync_write("Goal_Velocity", base_wheel_goal_vel)
        return {
            **left_arm_pos,
            **right_arm_pos,
            **head_pos,
            **base_goal_vel,
        }

    def stop_base(self):
        self.bus4.sync_write("Goal_Velocity", dict.fromkeys(self.base_motors, 0), num_retry=5)
        logger.info("Base motors stopped")

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.stop_base()
        self.bus1.disconnect(self.config.disable_torque_on_disconnect)
        self.bus2.disconnect(self.config.disable_torque_on_disconnect)
        self.bus3.disconnect(self.config.disable_torque_on_disconnect)
        self.bus4.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
