"""Actual MuJoCo integration with a fake CAN endpoint: never connects to hardware."""
import copy
import numpy as np
import pytest

from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.config import ROOT, load
from piper_sim.contracts import LOWER, UPPER
from piper_sim.joint_sync import JointSync
from piper_sim.sync_view import NativeJointInput
from test_joint_sync import SDK


@pytest.fixture(scope="module")
def sim():
    cfg = load()
    cfg["task_scene"] = ROOT/"configs/tasks/real_connector.yaml"
    backend = MujocoBackend(cfg)
    yield backend
    backend.close()


@pytest.mark.parametrize("joint", range(6))
def test_each_joint_reaches_requested_delta_from_mujoco_pose(sim, joint):
    sdk, clock = SDK(), [10.]
    bounds = np.rad2deg(np.column_stack((LOWER[7:13], UPPER[7:13])))
    control = JointSync(sdk, lambda s: copy.deepcopy(s.state), bounds, clock=lambda: clock[0])
    control.start(sdk.state["joints_deg"])
    control.select(joint)
    sim.reset()
    pose = np.clip(sim.read().position, LOWER, UPPER)
    pose[7:13] = np.deg2rad(control.reference)
    sim.set_observation_state(pose)
    def advance(q):
        pose[7:13] = np.deg2rad(q)
        sim.set_observation_state(pose)
        return np.rad2deg(sim.read().position[7:13])
    for _ in range(70):
        clock[0] += 1/30
        control.tick(advance)
    assert control.state == "active"
    delta = -2. if joint == 2 else 2.
    goal = control.reference[joint]+delta
    # Exercise the same native Control buffer and readback as the visible UI.
    native = NativeJointInput(sim.control_targets(), 7)
    sim.data.ctrl[sim.indices["right"]["ctrl"][joint]] = np.deg2rad(goal)
    changed_joint, degrees = native.changed(sim.control_targets())
    assert changed_joint == joint
    control.target(degrees)
    for _ in range(180):
        clock[0] += 1/30
        previous = control.command.copy()
        control.tick(advance)
        assert np.max(abs(control.command-previous)) <= .5/30+1e-8
    assert control.command[joint] == pytest.approx(goal, abs=.15)
    other = np.arange(6) != joint
    np.testing.assert_allclose(control.command[other], control.reference[other])
