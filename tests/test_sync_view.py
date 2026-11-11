from types import SimpleNamespace
import mujoco
import numpy as np
import pytest
from piper_sim.sync_view import NativeJointInput, FeedbackGhost


def test_native_control_detects_joint_and_converts_radians_without_committing():
    original = np.zeros(14)
    inputs = NativeJointInput(original, 7)
    edited = original.copy()
    edited[9] = np.deg2rad(-5)
    assert inputs.changed(edited) == pytest.approx((2, -5))
    np.testing.assert_array_equal(inputs.expected, original)
    inputs.reset(edited)
    assert inputs.changed(edited) is None


@pytest.mark.parametrize("indices", [(0,), (6,), (13,), (7,8)])
def test_other_arm_gripper_or_multiple_changes_rejected(indices):
    inputs = NativeJointInput(np.zeros(14), 7)
    q = np.zeros(14)
    q[list(indices)] = .1
    with pytest.raises(ValueError):
        inputs.changed(q)


def test_feedback_ghost_uses_raw_pose_and_does_not_modify_leader():
    from piper_sim.backends.mujoco import MujocoBackend
    from piper_sim.config import load
    sim = MujocoBackend(load())
    # Headless viewer stand-in: no actual CAN or window required.
    sim.viewer = SimpleNamespace(user_scn=mujoco.MjvScene(sim.model,maxgeom=2000),
                                 opt=mujoco.MjvOption(), cam=mujoco.MjvCamera(), close=lambda:None)
    try:
        ghost = FeedbackGhost(sim, "right")
        before = sim.data.qpos.copy()
        gap = ghost.update([5., -2.4, .8, 0, 0, 0])
        assert gap > 1
        assert sim.viewer.user_scn.ngeom > 0
        np.testing.assert_array_equal(sim.data.qpos, before)
        j2 = sim.indices["right"]["qpos"][1]
        assert ghost.data.qpos[j2] == pytest.approx(np.deg2rad(-2.4))
    finally:
        sim.close()
