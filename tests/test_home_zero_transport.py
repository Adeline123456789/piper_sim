from types import SimpleNamespace
import threading
import time
import numpy as np
import pytest
from piper_sim.home_zero_transport import HomeZeroTransport


class Controller:
    def __init__(self, **kwargs):
        assert kwargs == dict(can_interface='fake', robot_rate=100,auto_enable=False,raw_joint_feedback=True)
        self.calls=[]
        self.stream_fault=None
        self.piper=SimpleNamespace(MotionCtrl_1=lambda *a:self.calls.append(('stop',a)),
            DisableArm=lambda i:self.calls.append(('disable',i)),
            DisconnectPort=lambda:self.calls.append(('disconnect',)))
    def start_control_loop(self): self.calls.append(('start_loop',))
    def stop_control_loop(self): self.calls.append(('stop_loop',))
    def set_target_joint_angles(self, q): self.calls.append(('target',q.copy()))
    def _set_command_stream_enabled(self, state): self.calls.append(('gate',state))
    def reconcile_disabled_feedback(self): return True
    def enable_robot(self, **kwargs):
        assert kwargs['sync_targets'] is False and kwargs['timeout_s']==5
        kwargs['progress_check']()
        self.calls.append(('enable',))


def test_same_controller_is_passive_then_uses_100hz_loop_and_enable_api():
    t=HomeZeroTransport('fake',Controller)
    assert t.controller.calls==[]
    t.JointCtrl(1000,0,0,0,0,0)
    np.testing.assert_array_equal(t.controller.calls[-1][1],[1,0,0,0,0,0])
    t.MotionCtrl_2(1,1,5,0)
    for i in range(1,7): t.EnableArm(i)
    t.enable_thread.join(1)
    assert t.controller.calls.count(('enable',))==1
    assert t.controller.calls.count(('start_loop',))==1
    t.MotionCtrl_1(1,0,0)
    assert t.controller.calls[-2:]==[('gate',False),('stop',(1,0,0))]
    t.DisconnectPort()
    assert ('stop_loop',) in t.controller.calls


def test_stop_cancels_enable_handshake_before_disable():
    entered=threading.Event()
    class WaitingController(Controller):
        def enable_robot(self, **kwargs):
            entered.set()
            while True:
                kwargs['progress_check']()
                time.sleep(.005)
    t=HomeZeroTransport('fake',WaitingController)
    t.EnableArm(1)
    assert entered.wait(1)
    t.MotionCtrl_1(1,0,0)
    assert not t.enable_thread.is_alive()
    assert 'cancelled' in t.enable_error
    t.DisableArm(1)
    assert t.controller.calls[-1]==('disable',1)


def test_source_stream_fault_cannot_be_silently_ignored():
    t=HomeZeroTransport('fake',Controller)
    t.controller.stream_fault='failed send'
    with pytest.raises(RuntimeError,match='failed send'):
        t.JointCtrl(0,0,0,0,0,0)
