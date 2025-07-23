import math
import time
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorCmd_, MotorStates_

class WristController:
    def __init__(self, dt = 0.01 , kp=5.0, kd=1.0):
       
        # DDS initialization
        self.publisher = ChannelPublisher("rt/wrist/cmd", MotorCmds_)
        self.subscriber = ChannelSubscriber("rt/wrist/state", MotorStates_)
        self.publisher.Init()
        self.subscriber.Init()
        self.dt = dt 
        self.kp = kp
        self.kd = kd

        # Move wrist to zero on init
        self.write_position_deg(0.0)

    def write_position_deg(self, pos_deg):
        """Send a position command in degrees to the wrist motor at constant speed."""
        target_rad = math.radians(pos_deg)
        current_q = self._get_current_position()
        
        steps = 50
        err = target_rad - current_q
        for i in range(steps + 1):
            q_interp = current_q + (err * i / steps)
            cmd = MotorCmd_(
                mode=0x01,
                q=q_interp,
                dq=0.0,
                tau=0.0,
                kp=self.kp,
                kd=self.kd,
                reserve=[0, 0, 0]
            )
            msg = MotorCmds_()
            msg.cmds.append(cmd)
            self.publisher.Write(msg)
            

    def _get_current_state(self):
        """Return the latest wrist state message."""
        while True:
            msg = self.subscriber.Read()
            if msg and msg.states:
                return msg.states[0]
            time.sleep(0.01)

    def _get_current_position(self):
        return self._get_current_state().q

    def get_position_rad(self):
        return self._get_current_state().q

    def get_velocity(self):
        return self._get_current_state().dq

    def get_torque(self):
        return self._get_current_state().tau_est

    def get_temperature(self):
        return self._get_current_state().temperature

    def get_mode(self):
        return self._get_current_state().mode

    def get_lost_count(self):
        return self._get_current_state().lost

# Example main usage
if __name__ == "__main__":

    interface_name = "enp2s0"
    ChannelFactoryInitialize(0, interface_name)
    controller = WristController()

    while True:
        try:
            pos = float(input("Enter wrist angle (degrees): "))
            controller.write_position_deg(pos)
            time.sleep(1.0)
            print(f"Current Pos: {controller.get_position_rad():.2f}rad")
            print(f"Current Pos: {math.degrees(controller.get_position_rad()):.2f}°")
            print(f"Velocity: {controller.get_velocity():.3f} rad/s")
            print(f"Torque: {controller.get_torque():.3f} Nm")
            print(f"Temperature: {controller.get_temperature()} °C")
            print(f"Mode: {controller.get_mode()}")
            print(f"Lost Packets: {controller.get_lost_count()}\n")

        except KeyboardInterrupt:
            print("Exiting.")
            break
        except Exception as e:
            print(f"[Error] {e}")
            time.sleep(1)
