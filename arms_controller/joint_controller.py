import numpy as np
import time, math, csv, os
from datetime import datetime

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

from . import unitree_legged_const  as h1

H1_NUM_MOTOR = 20


# Joint mapping as a dictionary: joint name -> motor index

# Joint mapping as a dictionary: joint name -> motor index

joint_mapping = {
    "RightHipYaw": 8,
    "RightHipRoll": 0,
    "RightHipPitch": 1,
    "RightKnee": 2,
    "RightAnkle": 11,
    "LeftHipYaw": 7,
    "LeftHipRoll": 3,
    "LeftHipPitch": 4,
    "LeftKnee": 5,
    "LeftAnkle": 10,
    "WaistYaw": 6,
    "NotUsedJoint": 9,
    "RightShoulderPitch": 12,
    "RightShoulderRoll": 13,
    "RightShoulderYaw": 14,
    "RightElbow": 15,
    "LeftShoulderPitch": 16,
    "LeftShoulderRoll": 17,
    "LeftShoulderYaw": 18,
    "LeftElbow": 19,
}

arm_joint_names = [ "LeftShoulderPitch", 
                   "LeftShoulderRoll", 
                   "LeftShoulderYaw", 
                   "LeftElbow", 
                   "RightShoulderPitch", 
                   "RightShoulderRoll", 
                   "RightShoulderYaw", 
                   "RightElbow", 
             ]


# List of indices for weak motors (if needed for different PD gains)
weak_motors_indices = [11, 10, 12, 13, 14, 15, 16, 17, 18, 19]


class UnitreeH1ArmController: 
    def __init__(self, control_dt=0.02, results_dir=None, dds_topic="l"): 

        self.control_dt_ = control_dt
        self.target_positions = {joint: 0.0 for joint in arm_joint_names}

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = None
        self.crc = CRC()
        self.time_ = 0.0
        self.running = True

        # PD gains 
        # Per-joint gains
        self.Kp_init = 45.0
        self.Kd_init = 3.3 

        self.Kp_map = { "RightShoulderPitch": self.Kp_init,"RightShoulderRoll": self.Kp_init,
                        "RightShoulderYaw": self.Kp_init,"RightElbow": self.Kp_init,
                        "LeftShoulderPitch": self.Kp_init,"LeftShoulderRoll": self.Kp_init,
                        "LeftShoulderYaw": self.Kp_init, "LeftElbow": self.Kp_init
                        }
        
        self.Kd_map =  { "RightShoulderPitch": self.Kd_init,"RightShoulderRoll": self.Kd_init,
                        "RightShoulderYaw": self.Kd_init,"RightElbow": self.Kd_init,
                        "LeftShoulderPitch": self.Kd_init,"LeftShoulderRoll": self.Kd_init,
                        "LeftShoulderYaw": self.Kd_init, "LeftElbow": self.Kd_init
                        }

        # Logs
        self.log_data = []
        self.logging_enabled = False

        self.results_dir = results_dir or os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = os.path.join(self.results_dir, f"joint_log_{timestamp}.csv")
        
        # intialize msgs
        self.dds_topic_type = dds_topic
        self._init_low_cmd()
        self._init_topics(dds_topic)
        self.control_thread = None

        self._initialize_targets_from_current_state()
        
    def _init_low_cmd(self):
        # Set message header and default values.
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(H1_NUM_MOTOR):
            if i in weak_motors_indices:
                self.low_cmd.motor_cmd[i].mode = 0x01
            else:
                self.low_cmd.motor_cmd[i].mode = 0x0A
            self.low_cmd.motor_cmd[i].q = h1.PosStopF
            self.low_cmd.motor_cmd[i].dq = h1.VelStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def _init_topics(self, topic_type):
        topic_map = {"l": "rt/lowcmd", "h": "rt/arm_sdk"}
        topic_name = topic_map.get(topic_type, "rt/lowcmd")

        self.armcmd_publisher = ChannelPublisher(topic_name, LowCmd_)
        self.armcmd_publisher.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self._low_state_handler, 10)

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()
            

    def _low_state_handler(self, msg: LowState_):
        self.low_state = msg

    def _initialize_targets_from_current_state(self):
        if self.low_state is None:
            print("[WARN] Cannot initialize targets, low_state not available.")
            return
        for joint in arm_joint_names:
            q = self.low_state.motor_state[joint_mapping[joint]].q
            self.target_positions[joint] = q
    
    
    def write_arm_command(self):
        """
        Called periodically by the control loop.
        It updates only the arm joints to move toward their target positions,
        and sets the "weight" parameter in the NotUsedJoint.
        """
        if self.low_state is None:
            return
        
        log_entry = {"time": time.time()}
   
        # Update arm joints.
        for joint in arm_joint_names:
            idx = joint_mapping[joint]
            current_q = self.low_state.motor_state[idx].q
            target_q = self.target_positions.get(joint, current_q)
            
            self.low_cmd.motor_cmd[idx].q = target_q
            self.low_cmd.motor_cmd[idx].dq = 0.0
            self.low_cmd.motor_cmd[idx].kp = self.Kp_map[joint]
            self.low_cmd.motor_cmd[idx].kd = self.Kd_map[joint]
            self.low_cmd.motor_cmd[idx].tau = 0.0


            # save logs
            if self.logging_enabled:
                log_entry[f"{joint}_target"] = target_q
                log_entry[f"{joint}_pos"] = self.low_state.motor_state[idx].q
                # log_entry[f"{joint}_pos_raw"] = self.low_state.motor_state[idx].q_raw               
                log_entry[f"{joint}_vel"] = self.low_state.motor_state[idx].dq
                log_entry[f"{joint}_tau"] = self.low_state.motor_state[idx].tau_est

        # save state values
        if self.logging_enabled:
            self.log_data.append(log_entry)
 

        # Update the weight pasrameter using NotUsedJoint.
        self.low_cmd.motor_cmd[joint_mapping["NotUsedJoint"]].q = 0.1

        # idx_waist = joint_mapping["WaistYaw"]
        # self.low_cmd.motor_cmd[idx_waist].q = 0.0
        # self.low_cmd.motor_cmd[idx_waist].dq = 0.0
        # self.low_cmd.motor_cmd[idx_waist].kp = 200
        # self.low_cmd.motor_cmd[idx_waist].kd = 5.0
        # self.low_cmd.motor_cmd[idx_waist].tau = 0.0
        
        # Compute and attach the CRC.
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        # Write the command over the high-level arm topic.
        self.armcmd_publisher.Write(self.low_cmd)

    def start_control_loop(self):
        """
        Start a control loop that continuously sends arm commands.
        """
        self.running = True
        self.control_thread = RecurrentThread(
            interval=self.control_dt_, target=self.write_arm_command, name="g1_arm_control_loop"
        )
        self.control_thread.Start()

    def stop_control_loop(self):
        """
        Stop the arm control loop.
        """
        self.running = False
        if self.control_thread is not None:
            self.control_thread.Wait()  # Proper way to request loop exit and join
      
    def read_motor_state(self):
        """
        Returns a dictionary of the current motor positions (radians) for arm joints.
        """
        if self.low_state is None:
            return {}
        state = {}
        for joint in arm_joint_names:
            idx = joint_mapping[joint]
            state[joint] = self.low_state.motor_state[idx].q
        return state
    

    def read_torque_state(self):
        """
        Returns a dictionary of the current motor positions (radians).
        """
        if self.low_state is None:
            return {}
        return {joint: self.low_state.motor_state[idx].tau_est for joint, idx in joint_mapping.items()}
    
    def read_raw_motor_state(self):
        """
        Returns a dictionary of the current motor positions (radians).
        """
        if self.low_state is None:
            return {}
        return {joint: self.low_state.motor_state[idx].q_raw for joint, idx in joint_mapping.items()}

    
    def update_target_positions(self, new_targets: dict):
        """ G1 SDK2, 
        Update the target positions for the arm joints.
        Expect keys from the arm joint list or "NotUsedJoint".
        """
        for joint, target in new_targets.items():
            if joint in self.target_positions:
                self.target_positions[joint] = target
            else:
                print(f"Warning: {joint} not found in target positions.")

    def print_imu_state(self):
        """
        Returns the current IMU sensor values: RPY, gyroscope, and accelerometer.
        return: 
        {
            "rpy": (roll, pitch, yaw),
            "gyroscope": (x, y, z),
            "accelerometer": (x, y, z)
        }
        """
        if self.low_state and hasattr(self.low_state, "imu_state"):
            imu = self.low_state.imu_state
            rpy = imu.rpy
            gyro = imu.gyroscope
            accel = imu.accelerometer
            imu_data = {
                "rpy": tuple(rpy),
                "gyroscope": tuple(gyro),
                "accelerometer": tuple(accel)
            }
            # print(f"[IMU RPY]         Roll: {rpy[0]:.4f}, Pitch: {rpy[1]:.4f}, Yaw: {rpy[2]:.4f}")
            # print(f"[IMU Gyroscope]   X: {gyro[0]:.4f}, Y: {gyro[1]:.4f}, Z: {gyro[2]:.4f}")
            # print(f"[IMU Accelerometer] X: {accel[0]:.4f}, Y: {accel[1]:.4f}, Z: {accel[2]:.4f}")
            return imu_data
        
        else:
            print("IMU state not available yet.")
            return None

    def enable_logging(self, filename = None):
        self.logging_enabled = True
        if filename is not None:
            self.log_filename = filename
        self.log_data = []

    def save_log_to_csv(self):
        if not self.logging_enabled or not self.log_data:
            return
        keys = self.log_data[0].keys()
        with open(self.log_filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.log_data)
        print(f"[INFO] Log saved to {self.log_filename}")



if __name__ == "__main__":

    print("[INFO] Initializing DDS...")
    ChannelFactoryInitialize(1, "lo")
    time.sleep(0.5)

    print("[INFO] Initializing controller...")
    controller = UnitreeH1ArmController(control_dt=0.02, dds_topic="l")
    controller.start_control_loop()
    time.sleep(1.0)

    print("[INFO] Testing each joint individually...")
    for joint in arm_joint_names:
        print(f"[TEST] Moving joint: {joint}")
        target = {j: 0.0 for j in arm_joint_names}
        target[joint] = 0.5
        controller.update_target_positions(target)
        time.sleep(2.0)
        target[joint] = -0.2
        controller.update_target_positions(target)
        time.sleep(2.0)
        target[joint] = 0.00
        controller.update_target_positions(target)
        time.sleep(1.0)

    print("[INFO] Final joint positions:")
    print(controller.read_motor_state())

    controller.stop_control_loop()
    
    

            
