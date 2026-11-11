"""Reuse home_zero.py's ReplayController and its 100 Hz command thread.

Construction does not enable, start the command loop, home, or touch the gripper.
Only this session's explicit enable starts that same controller's handshake.
"""
import json
import platform
import subprocess
import threading

import numpy as np


def verify_mapping(mapping_path, interface, side):
    mapping = json.loads(mapping_path.read_text())
    link = json.loads(subprocess.check_output(
        ['ip', '-details', '-json', 'link', 'show', 'dev', interface], text=True))[0]
    saved = mapping.get('interfaces', {}).get(interface, {})
    if (mapping.get('hostname') != platform.node() or saved.get('physical_side') != side or
            not saved.get('usb_interface') or saved.get('usb_interface') != link.get('parentdev')):
        raise RuntimeError(f'{interface}: USB/side mapping differs from home_zero mapping')


class HomeZeroTransport:
    def __init__(self, interface, controller_factory=None):
        if controller_factory is None:
            from replay_controller import ReplayController
            controller_factory = ReplayController
        self.controller = controller_factory(can_interface=interface, robot_rate=100,
                                             auto_enable=False, raw_joint_feedback=True)
        self.raw = self.controller.piper
        self.enable_thread = None
        self.enable_error = None
        self.cancel_enable = threading.Event()

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def MotionCtrl_2(self, *args):
        # ReplayController._send_joint_command sends this at 100 Hz.
        if args != (1, 1, 5, 0):
            raise ValueError('Only the home_zero low-speed joint mode is supported')

    def JointCtrl(self, *angles_millidegrees):
        if self.enable_error or self.controller.stream_fault:
            raise RuntimeError(self.enable_error or self.controller.stream_fault)
        self.controller.set_target_joint_angles(np.array(angles_millidegrees)/1000.)

    def EnableArm(self, motor):
        if self.enable_error:
            raise RuntimeError(self.enable_error)
        if self.enable_thread is not None:
            return
        self.controller.start_control_loop()
        self.cancel_enable.clear()
        def progress():
            if self.cancel_enable.is_set():
                raise RuntimeError('Enable cancelled by stop')
        def enable():
            try:
                self.controller.enable_robot(sync_targets=False, timeout_s=5, progress_check=progress)
            except BaseException as exc:
                self.enable_error = str(exc)
        self.enable_thread = threading.Thread(target=enable, daemon=True)
        self.enable_thread.start()

    def MotionCtrl_1(self, mode, a, b):
        if mode == 1:
            self.cancel_enable.set()
            self.controller._set_command_stream_enabled(False)
            if self.enable_thread and self.enable_thread is not threading.current_thread():
                self.enable_thread.join(timeout=.5)
        self.raw.MotionCtrl_1(mode, a, b)

    def DisableArm(self, motor):
        self.controller._set_command_stream_enabled(False)
        self.raw.DisableArm(motor)

    def DisconnectPort(self):
        self.cancel_enable.set()
        self.controller._set_command_stream_enabled(False)
        self.controller.stop_control_loop()
        if self.enable_thread:
            self.enable_thread.join(timeout=5.5)
        self.raw.DisconnectPort()

    def clear_disabled_session(self):
        if self.enable_thread and self.enable_thread.is_alive():
            raise RuntimeError('Enable handshake still exiting; reconnect before another session')
        if not self.controller.reconcile_disabled_feedback():
            raise RuntimeError('Cannot confirm disabled controller state')
        self.enable_thread = None
        self.enable_error = None
        self.controller.stream_fault = None
