import copy
import threading

import numpy as np
import pytest

from piper_sim.joint_sync import JointSync


class SDK:
    def __init__(self):
        self.sent = []
        self.state = dict(joints_deg=[0., -2., .8, 0., 23., 2.], motors_enabled=[False]*6,
                          gripper_enabled=False, arm_status=0, err_code=0, ctrl_mode=0,
                          driver_faults={})

    def MotionCtrl_1(self, mode, *_):
        self.sent.append(("stop/resume", mode))
        self.state["arm_status"] = 1 if mode == 1 else 0

    def MotionCtrl_2(self, *args):
        self.sent.append(("mode", args))

    def JointCtrl(self, *args):
        self.sent.append(("joints", args))
        self.state["joints_deg"] = np.array(args)/1000

    def EnableArm(self, i):
        self.sent.append(("enable", i))
        self.state["motors_enabled"][i-1] = True

    def DisableArm(self, i):
        self.sent.append(("disable", i))
        self.state["motors_enabled"][i-1] = False


@pytest.fixture
def rig():
    sdk = SDK()
    clock = [10.]
    c = JointSync(sdk, lambda s: copy.deepcopy(s.state),
                  [[-150,150],[0,180],[-170,0],[-100,100],[-70,70],[-120,120]],
                  clock=lambda: clock[0])
    return c, sdk, clock


def activate(rig):
    c, sdk, clock = rig
    c.start(sdk.state["joints_deg"])
    for _ in range(70):
        clock[0] += 1/30
        c.tick(lambda q: q)
    assert c.state == "active"


def test_passive_connect_and_preview_send_nothing(rig):
    c, sdk, _ = rig
    c.tick(lambda _: pytest.fail("Must not advance on passive tick"))
    initial, entry = c.prepare(c.read())
    assert entry[1:3].tolist() == [0., 0.]
    assert initial[1:3].tolist() == [-2., .8]
    assert not sdk.sent
    with pytest.raises(RuntimeError, match="Start"):
        c.target(5)


def test_real_follows_returned_sim_position_not_requested_goal(rig):
    activate(rig)
    c, sdk, clock = rig
    c.target(5.)
    previous = c.command.copy()
    clock[0] += 1/30
    c.tick(lambda q: (q+previous)/2)
    last = [v for k, v in sdk.sent if k == "joints"][-1]
    assert last[0] == 8  # 0.5 deg/s / 30 Hz / 2 -> 0.008 deg
    np.testing.assert_allclose(np.array(last[1:])/1000, previous[1:])
    assert not any(k == "gripper" for k, _ in sdk.sent)


def test_all_six_joint_selections_and_bounds(rig):
    activate(rig)
    c, _, clock = rig
    for j in range(6):
        c.select(j)
        target = c.command[j] + (-.01 if j == 2 else .01)
        c.target(target)
        old = c.command.copy()
        clock[0] += 1/30
        c.tick(lambda q: q)
        mask = np.arange(6) != j
        np.testing.assert_allclose(c.command[mask], old[mask])
        assert c.command[j] == pytest.approx(target)
    with pytest.raises(ValueError):
        c.target(1000)


def test_watchdog_stops_but_does_not_release_gravity_load(rig):
    activate(rig)
    c, sdk, clock = rig
    sdk.sent.clear()
    clock[0] += .4
    c.watchdog()
    assert c.state == "stopped"
    assert sdk.sent == [("stop/resume", 1)]
    c.request_disable()
    for _ in range(2):
        c.tick(lambda q: q)
    assert c.state == "idle" and not c.touched
    assert not any(sdk.state["motors_enabled"])


def test_stalled_simulation_does_not_block_watchdog_or_send_after_stop(rig):
    activate(rig)
    c, sdk, clock = rig
    entered, finish = threading.Event(), threading.Event()
    def blocked(q):
        entered.set()
        assert finish.wait(2)
        return q
    sdk.sent.clear()
    worker = threading.Thread(target=lambda: c.tick(blocked))
    worker.start()
    assert entered.wait(2)
    clock[0] += .4
    c.watchdog()
    assert c.state == "stopped"
    finish.set()
    worker.join(2)
    assert not worker.is_alive()
    assert sdk.sent == [("stop/resume", 1)]


def test_bad_sim_or_feedback_stops_without_sending_target(rig):
    activate(rig)
    c, sdk, clock = rig
    sdk.sent.clear()
    clock[0] += 1/30
    with pytest.raises(RuntimeError, match="deviated"):
        c.tick(lambda q: q+10)
    assert sdk.sent == [("stop/resume", 1)]


def test_faulted_or_unexpected_entry_rejected_before_enable(rig):
    c, sdk, _ = rig
    sdk.state["joints_deg"][1] = -4
    with pytest.raises(RuntimeError, match="outside limits"):
        c.start(sdk.state["joints_deg"])
    assert not sdk.sent


def test_software_stop_resumes_without_enabling_until_stable(rig):
    c, sdk, clock = rig
    sdk.state["arm_status"] = 1
    c.start(sdk.state["joints_deg"])
    assert sdk.sent == [("stop/resume", 2)]
    for _ in range(20):
        clock[0] += 1/30
        c.tick(lambda q: q)
    assert not any(k == "enable" for k, _ in sdk.sent)
    for _ in range(100):
        clock[0] += 1/30
        c.tick(lambda q: q)
    assert c.state == "active"


def test_home_zero_firmware_reset_transient_is_waited_out_with_motors_off(rig):
    c, sdk, clock = rig
    sdk.state["arm_status"] = 1
    c.start(sdk.state["joints_deg"])
    for _ in range(15):
        clock[0] += 1/30
        c.tick(lambda q:q)
    sdk.state.update(arm_status=5, err_code=63)
    original=list(sdk.state['joints_deg'])
    sdk.state['joints_deg']=[3.,-2.,4.,7.,8.,6.]
    for _ in range(3):
        clock[0] += 1/30
        c.tick(lambda q:q)
    assert c.state == 'resuming'
    assert not any(k=='enable' for k,_ in sdk.sent)
    sdk.state.update(arm_status=0, err_code=0)
    sdk.state['joints_deg']=original
    for _ in range(100):
        clock[0] += 1/30
        c.tick(lambda q:q)
    assert c.state == 'active'


def test_reset_transient_exception_does_not_mask_enabled_or_high_fault_bits(rig):
    c,sdk,clock=rig
    sdk.state['arm_status']=1
    c.start(sdk.state['joints_deg'])
    sdk.state.update(arm_status=5,err_code=64)
    clock[0]+=.5
    with pytest.raises(RuntimeError,match='fault'):
        c.tick(lambda q:q)
    assert c.state=='stopped'
