import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl import MotorCmds_, unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl import MotorStates_, unitree_go_msg_dds__MotorState_

class InspireHandController:
    def __init__(self, topic="rt/inspire/cmd",topic_state="rt/inspire/state"):
        """Initialize the DDS publisher for the Inspire hand."""
        self.publisher = ChannelPublisher(topic, MotorCmds_)
        self.publisher.Init()
        self.subscriber = ChannelSubscriber(topic_state, MotorStates_)
        self.subscriber.Init()
        self.num_fingers = 6  # 5 fingers + 1 wrist rotor in many setups
        self.open_position = 1.0
        self.close_position = 0
        self.half_open_position = 0.5 
        self.last_hold_state = None
        self.last_grap_state = None
        time.sleep(0.5)

    def send_positions(self, positions):
        cmds = MotorCmds_(cmds=[unitree_go_msg_dds__MotorCmd_() for _ in range(12)])
        for i in range(self.num_fingers):
            cmds.cmds[i].q = positions[i]
        self.publisher.Write(cmds)

    def open(self):
        """Send command to open all Inspire hand fingers."""
        cmds = MotorCmds_(cmds=[unitree_go_msg_dds__MotorCmd_() for _ in range(12)])
        for i in range(self.num_fingers):
            cmds.cmds[i].q = self.open_position
        self.publisher.Write(cmds)
        print("[InspireHand] Open command sent.")

    def point(self):
        # Safety: fully open first
        self.open()
        time.sleep(1)
        positions = [self.close_position]*self.num_fingers
        positions[3] = self.open_position
        positions[5] = 1.0 # half open pointer
        self.send_positions(positions)
        print("[InspireHand] Point gesture command sent.")


    def point_half(self):
        self.open()
        time.sleep(0.5)
        positions = [self.close_position]*self.num_fingers
        positions[3] = self.half_open_position
        self.send_positions(positions)
        print("[InspireHand] Point half gesture command sent.")

    def point_half(self):
        positions = [self.close_position]*self.num_fingers
        positions[3] = self.half_open_position  # half open pointer
        positions[5] = self.open_position # half open pointer
        self.send_positions(positions)
        print("[InspireHand] Point half gesture command sent.")


    def hold(self):
        if self.last_hold_state == 1:
            # Go to state 2: all closed
            positions = [self.half_open_position]*self.num_fingers
            positions[5] = self.half_open_position  # Thumb DoF2 closed
            self.send_positions(positions)
            print("[InspireHand] Hold: state 2 (fully closed)")
            self.last_hold_state = 2
        else:
            # Go to state 1: all open except Thumb DoF2 closed
            positions = [self.open_position]*self.num_fingers
            positions[5] = self.half_open_position  # Thumb DoF2 closed
            self.send_positions(positions)
            print("[InspireHand] Hold: state 1 (all open except thumb2 closed)")
            self.last_hold_state = 1

 

    def close(self):
        # Close fingers first
        positions = [self.close_position if i < 4 else self.open_position for i in range(self.num_fingers)]
        self.send_positions(positions)
        print("[InspireHand] Closing fingers (pinky to pointer)...")
        time.sleep(0.5)
        # Close thumbs
        positions = [self.close_position]*self.num_fingers
        self.send_positions(positions)
        print("[InspireHand] Close command sent.")

    def grap(self):
        if self.last_grap_state == 1:
            # Go to state 2: close thumb and pointer
            positions = [self.close_position]*self.num_fingers
            positions[3] = self.half_open_position # Pointer
            positions[4] = self.half_open_position  # Thumb DoF1

            self.send_positions(positions)
            print("[InspireHand] Grap: state 2 (thumb and pointer closed)")
            self.last_grap_state = 2
        else:
            # Go to state 1: all closed except thumb and pointer open
            positions = [self.close_position]*self.num_fingers
            positions[3] = self.open_position  # Pointer
            positions[4] = self.open_position  # Thumb DoF1
            # positions[5] = self.open_position  # Thumb DoF2
            self.send_positions(positions)
            print("[InspireHand] Grap: state 1 (only thumb and pointer open)")
            self.last_grap_state = 1

    # Toggle functions per DoF:
    def toggle_pinky(self):
        self._toggle_single(0, "Pinky")

    def toggle_ring(self):
        self._toggle_single(1, "Ring")

    def toggle_middle(self):
        self._toggle_single(2, "Middle")

    def toggle_pointer(self):
        self._toggle_single(3, "Pointer")

    def toggle_thumb1(self):
        self._toggle_single(4, "Thumb DoF1")


    def _toggle_single(self, index, name):
        positions = self.read_state()
        current_pos = positions[index]
        new_value = self.close_position if current_pos > 0.5 else self.open_position
        positions[index] = new_value
        positions[5] = self.open_position
        self.send_positions(positions)
        print(f"[InspireHand] {name} toggled to {new_value}")

    def read_state(self):
        state = self.subscriber.Read()
        positions = []
        if state is not None:
            for i in range(self.num_fingers):
                positions.append(state.states[i].q)
            print("[InspireHand] Current positions:", positions)
            return positions
        else:
            print("[InspireHand] No state received yet.")
            return [self.open_position]*self.num_fingers

 
if __name__ == "__main__":

    '''
    cd ~/h1_inspire_service/build
    sudo LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib ./inspire_hand -s /dev/ttyUSB0

    '''
    ChannelFactoryInitialize(0, "enp2s0")
    time.sleep(0.5)
    hand = InspireHandController()
    time_sleep = 2

    hand.open()
    time.sleep(time_sleep)
    hand.close()
    time.sleep(time_sleep)
    hand.open()
    time.sleep(time_sleep)

    hand.point()
    time.sleep(time_sleep)
    hand.point_half()
    time.sleep(time_sleep)

    hand.open()
    time.sleep(time_sleep)
    hand.hold()
    time.sleep(time_sleep)
    hand.hold()
    time.sleep(time_sleep)

    hand.open()
    time.sleep(time_sleep)

    hand.grap()
    time.sleep(time_sleep)
    hand.grap()
    time.sleep(time_sleep)

    hand.open()
    time.sleep(time_sleep)

    hand.toggle_pinky()
    time.sleep(time_sleep)
    hand.toggle_ring()
    time.sleep(time_sleep)
    hand.toggle_middle()
    time.sleep(time_sleep)
    hand.toggle_pointer()
    time.sleep(time_sleep)
    hand.toggle_thumb1()
    time.sleep(time_sleep)
    hand.open()
