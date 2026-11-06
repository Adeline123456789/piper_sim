"""Track individual CAN feedback frames: SDK aggregate timestamps are insufficient."""
from copy import deepcopy
import struct
import threading
import time


def tracked_sdk(base):
    class TrackedSDK(base):
        # Do not reuse a legacy controller's singleton on the same CAN channel.
        _instances = {}
        _lock = threading.Lock()

        def __init__(self, *args, **kwargs):
            self.feedback_lock = threading.RLock()
            self.frame_times = {}
            self.raw_feedback = {}
            super().__init__(*args, **kwargs)
            # This supplied SDK logs failed sends instead of raising from
            # JointCtrl/GripperCtrl. Promote transport failures at its CAN
            # boundary so PiperBackend can stop both sides and report a fault.
            port = getattr(self, f"_{base.__name__}__arm_can", None)
            if port is None:
                raise RuntimeError("Unsupported Piper SDK: CAN transport is unavailable")
            original_send = port.SendCanMessage

            def checked_send(*args, **kwargs):
                status = original_send(*args, **kwargs)
                if status is not port.CAN_STATUS.SEND_MESSAGE_SUCCESS:
                    raise OSError(f"Piper CAN send failed: {status}")
                return status

            port.SendCanMessage = checked_send

        def ParseCANFrame(self, message):
            with self.feedback_lock:
                super().ParseCANFrame(message)
                if message is not None and not message.is_error_frame and message.dlc == 8:
                    self.frame_times[message.arbitration_id] = time.monotonic()
                    if 0x2A5 <= message.arbitration_id <= 0x2A8:
                        self.raw_feedback[message.arbitration_id] = bytes(message.data)

        def GetArmJointMsgs(self):
            # Some SDK releases clamp received feedback with the command limits.
            # Preserve signed encoder feedback while keeping SDK command limiting active.
            with self.feedback_lock:
                result = deepcopy(super().GetArmJointMsgs())
                for index, identifier in enumerate((0x2A5, 0x2A6, 0x2A7)):
                    if identifier in self.raw_feedback:
                        for offset, angle in enumerate(struct.unpack(">ii", self.raw_feedback[identifier])):
                            setattr(result.joint_state, f"joint_{2*index+offset+1}", angle)
                return result

        def GetArmGripperMsgs(self):
            with self.feedback_lock:
                result = deepcopy(super().GetArmGripperMsgs())
                if 0x2A8 in self.raw_feedback:
                    result.gripper_state.grippers_angle = int.from_bytes(
                        self.raw_feedback[0x2A8][:4], "big", signed=True)
                return result

    return TrackedSDK
