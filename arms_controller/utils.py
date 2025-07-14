"""
Utility functions for high-level G1 arm control.
"""

import math
import numpy as np

# Map IK joint names to G1 SDK motor names
ik_to_motor_joint_map = {
    "left_shoulder_pitch_joint": "LeftShoulderPitch",
    "left_shoulder_roll_joint": "LeftShoulderRoll",
    "left_shoulder_yaw_joint": "LeftShoulderYaw",
    "left_elbow_joint": "LeftElbow",
    "left_wrist_roll_joint": "LeftWristRoll",
    "left_wrist_pitch_joint": "LeftWristPitch",
    "left_wrist_yaw_joint": "LeftWristYaw",
    "right_shoulder_pitch_joint": "RightShoulderPitch",
    "right_shoulder_roll_joint": "RightShoulderRoll",
    "right_shoulder_yaw_joint": "RightShoulderYaw",
    "right_elbow_joint": "RightElbow",
    "right_wrist_roll_joint": "RightWristRoll",
    "right_wrist_pitch_joint": "RightWristPitch",
    "right_wrist_yaw_joint": "RightWristYaw",
}



def remap_ik_joints_to_motor(joint_dict):
    """
    Remap IK solution joint names to SDK motor names.
    """
    return {
        ik_to_motor_joint_map[j]: v
        for j, v in joint_dict.items()
        if j in ik_to_motor_joint_map
    }


def map_motor_state(motor_state):
    """
    Map SDK motor state names to IK joint names for forward kinematics.
    """
    motor_to_fk = {
        "RightShoulderPitch": "right_shoulder_pitch_joint",
        "RightShoulderRoll": "right_shoulder_roll_joint",
        "RightShoulderYaw": "right_shoulder_yaw_joint",
        "RightElbow": "right_elbow_joint",
        "RightWristRoll": "right_wrist_roll_joint",
        "RightWristPitch": "right_wrist_pitch_joint",
        "RightWristYaw": "right_wrist_yaw_joint",
        "LeftShoulderPitch": "left_shoulder_pitch_joint",
        "LeftShoulderRoll": "left_shoulder_roll_joint",
        "LeftShoulderYaw": "left_shoulder_yaw_joint",
        "LeftElbow": "left_elbow_joint",
        "LeftWristRoll": "left_wrist_roll_joint",
        "LeftWristPitch": "left_wrist_pitch_joint",
        "LeftWristYaw": "left_wrist_yaw_joint",
    }
    return {
        fk: motor_state[m]
        for m, fk in motor_to_fk.items()
        if m in motor_state
    }
