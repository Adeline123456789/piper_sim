"""The thin straw must rest on the mat; both holes must admit a part and block an offset."""
import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("shapely")
pytest.importorskip("mujoco")
from piper_sim.recorded_env import RecordedPiperEnv
from piper_sim.tasks.recorded_scenes import matrix


@pytest.mark.parametrize("name", ["real_straw", "real_connector"])
def test_recorded_task_contacts(name):
    env = RecordedPiperEnv(name, cameras=False)
    try:
        initial, _ = env.reset(seed=1)
        task = env.sim.task
        sim = env.sim
        for _ in range(60):
            sim.step()
        poses = sim.task_adapter.task_poses(sim.data, task)
        for part in task.dynamic:
            np.testing.assert_allclose(poses[part]["position"], task.layout["poses"][part]["position"], atol=.001)
        for offset in (0., .012):
            env.reset(seed=1)
            if name == "real_connector":
                part = "yellow_plug"
                p = task.layout["poses"]["strip"]
                x, y = task.outlets[task.config["target_index"]]
                xyz = (matrix(p) @ [x+offset, y, task.config["strip_size_m"][2]+.022, 1])[:3]
                quat = p["quaternion_wxyz"]
            else:
                part = "straw"
                p = task.layout["poses"]["cup"]
                xyz = np.array(p["position"]) + [offset, 0, task.config["cup"]["height_m"]
                                                   + task.config["straw"]["length_m"]/2 + .001]
                quat = [1, 0, 0, 0]
            q = sim.model.joint(f"task/{part}/free").qposadr[0]
            sim.data.qpos[q:q+7] = [*xyz, *quat]
            success = False
            hit_surface = False
            for _ in range(35):
                obs, reward, done, _, info = env.step(initial["state"].astype(float))
                assert env.observation_space.contains(obs)
                success |= done
                if offset:
                    for contact in sim.data.contact:
                        bodies = {sim.model.body(sim.model.geom(g).bodyid[0]).name for g in contact.geom}
                        hit_surface |= bodies == {f"task/{part}", "task/strip" if name == "real_connector" else "task/cup"}
                assert reward == float(info["success"])
            assert success == (offset == 0)
            if offset:
                assert hit_surface
    finally:
        env.close()


def test_white_strip_usb_and_three_ac_modules():
    env = RecordedPiperEnv("real_connector", cameras=False)
    try:
        sim = env.sim
        task = sim.task
        for index, (x, y) in enumerate(task.outlets):
            # Drop over each actual panel; train on the selected AC outlet.
            task.config["target_index"] = max(1, index)
            sim.reset()
            strip = matrix(task.layout["poses"]["strip"])
            xyz = (strip @ [x, y, task.socket_face_z+.021, 1])[:3]
            q = sim.model.joint("task/yellow_plug/free").qposadr[0]
            sim.data.qpos[q:q+7] = [*xyz, *task.layout["poses"]["strip"]["quaternion_wxyz"]]
            for _ in range(20):
                sim.step()
            metrics = sim.read().extras["task"]["metrics"]
            assert metrics["success"] == (index > 0)
            if index > 0:
                assert abs(metrics["face_gap_m"]) < .0015
            else:
                # The USB panel physically arrests the three pins at first contact.
                sim.data.qpos[q:q+7] = [*xyz, *task.layout["poses"]["strip"]["quaternion_wxyz"]]
                v = sim.model.joint("task/yellow_plug/free").dofadr[0]
                sim.data.qvel[v:v+6] = 0
                sim.step()
                assert sim.data.body("task/yellow_plug").xpos[2] - strip[2,3] > task.socket_face_z+.014
    finally:
        env.close()


def test_connector_arm_mounts_and_wrist_reference_frames():
    from piper_sim.backends.mujoco import MujocoBackend
    from piper_sim.config import ROOT, load
    import yaml

    scene = yaml.safe_load((ROOT/"configs/tasks/real_connector.yaml").read_text())
    current = MujocoBackend(dict(load(), task_scene=scene))
    reference = MujocoBackend(dict(load(), task_scene=dict(scene, arm_outward_offset_m=0.,
                                                           workstation_background="recording")))
    try:
        for side, sign in (("left",1), ("right",-1)):
            shift = [0,sign*.010,0]
            for body in ("base_link","mount_plate","gripper_center"):
                name = f"{side}/{body}"
                np.testing.assert_allclose(current.data.body(name).xpos - reference.data.body(name).xpos,
                                           shift,atol=1e-10)
            cam_name = side+"_wrist"
            np.testing.assert_allclose(current.data.camera(cam_name).xpos - reference.data.camera(cam_name).xpos,
                                       shift,atol=1e-10)
            meta = current.rig["cameras"][cam_name]
            old = reference.rig["cameras"][cam_name]
            np.testing.assert_allclose(np.array(meta["reference_world_pose"])[:3,3] -
                                       np.array(old["reference_world_pose"])[:3,3],shift,atol=1e-10)
        np.testing.assert_allclose(current.data.camera("overhead").xpos,reference.data.camera("overhead").xpos)
        for name in ("task/board","task/strip"):
            np.testing.assert_allclose(current.data.body(name).xpos,reference.data.body(name).xpos)
        # Face screw must be on the side looking down the board toward the open end.
        board = current.data.body("task/board")
        plug = current.data.body("task/yellow_plug")
        offset = board.xmat.reshape(3,3).T @ (current.data.geom("plug/screw").xpos - plug.xpos)
        assert offset[0] < -.015 and abs(offset[1]) < .001
    finally:
        current.close()
        reference.close()
