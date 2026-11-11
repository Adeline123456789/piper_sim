"""Visual variation must leave collision dynamics and camera provenance intact."""
from copy import deepcopy

import numpy as np
import pytest
import yaml

pytest.importorskip('mujoco')
pytest.importorskip('shapely')
from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.config import ROOT, load


def configs():
    office = yaml.safe_load((ROOT/'configs/tasks/real_connector.yaml').read_text())
    before = deepcopy(office)
    before.pop('appearance')
    return before, office


@pytest.mark.parametrize('height,neck_scale', [(.054,[1.,1.]), (.0394,[1.,1.]), (.0451,[.902,.907])])
def test_plug_collision_preserves_waist_volume_without_visual_hull(height,neck_scale):
    _, cfg = configs()
    cfg['yellow_plug'].update(housing_height_m=height, rim_height_m=.007*height/.054, neck_scale_xy=neck_scale)
    sim = MujocoBackend(dict(load(), task_scene=cfg))
    try:
        model = sim.model
        visual = model.geom('plug/grip')
        assert visual.contype[0] == visual.conaffinity[0] == 0
        mid = int(visual.dataid[0])
        start = model.mesh_vertadr[mid]
        vertices = model.mesh_vert[start:start+model.mesh_vertnum[mid]].astype(float)
        start = model.mesh_faceadr[mid]
        triangles = vertices[model.mesh_face[start:start+model.mesh_facenum[mid]]]
        visual_volume = abs(np.einsum('ij,ij->i', triangles[:,0],
                                     np.cross(triangles[:,1], triangles[:,2])).sum()/6)
        volumes = []
        for gid in range(model.ngeom):
            if not (model.geom(gid).name or '').startswith('plug/grip_contact/'):
                continue
            mid = model.geom_dataid[gid]; start = model.mesh_vertadr[mid]
            v = model.mesh_vert[start:start+model.mesh_vertnum[mid]].astype(float)
            start = model.mesh_faceadr[mid]
            tri = v[model.mesh_face[start:start+model.mesh_facenum[mid]]]
            volumes.append(abs(np.einsum('ij,ij->i', tri[:,0], np.cross(tri[:,1],tri[:,2])).sum())/6)
        assert len(volumes) == 291
        assert sum(volumes) == pytest.approx(visual_volume, rel=1e-5)
    finally:
        sim.close()


def test_office_profile_preserves_contacts_and_physics():
    before, office = configs()
    a = MujocoBackend(dict(load(), task_scene=before))
    b = MujocoBackend(dict(load(), task_scene=office))
    try:
        for key in ('body_mass', 'body_inertia', 'body_pos', 'geom_size', 'geom_pos',
                    'geom_quat', 'geom_contype', 'geom_conaffinity', 'geom_friction',
                    'geom_solref', 'geom_solimp', 'jnt_range', 'actuator_ctrlrange'):
            np.testing.assert_array_equal(getattr(a.model, key), getattr(b.model, key), err_msg=key)
        for _ in range(20):
            a.step(); b.step()
            np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
            np.testing.assert_array_equal(a.data.qvel, b.data.qvel)
        assert b.model.vis.quality.shadowsize == 4096
        assert b.read().extras['appearance']['randomized'] is False
    finally:
        a.close(); b.close()


def test_seeded_variation_resets_without_drift_and_reports_camera_pose():
    _, cfg = configs()
    cfg['appearance']['randomization']['enabled'] = True
    sim = MujocoBackend(dict(load(), task_scene=cfg))
    try:
        q = sim.data.qpos.copy()
        sim.reset(seed=23)
        sample = deepcopy(sim.appearance.sample)
        position = sim.model.cam_pos.copy()
        lights = sim.model.light_diffuse.copy()
        sim.reset(seed=24)
        assert not np.array_equal(lights, sim.model.light_diffuse)
        sim.reset(seed=23)
        assert sim.appearance.sample == sample
        np.testing.assert_array_equal(position, sim.model.cam_pos)
        np.testing.assert_array_equal(lights, sim.model.light_diffuse)
        np.testing.assert_array_equal(q, sim.data.qpos)
        for name, delta in sample['cameras'].items():
            assert max(abs(np.array(delta['parent_translation_delta_m']))) <= .002
        sim.appearance.config['randomization']['enabled'] = False
        sim.reset()
        np.testing.assert_array_equal(sim.model.cam_pos, sim.appearance.baseline['cam_pos'])
    finally:
        sim.close()


def test_drop_candidate_changes_only_yellow_rotational_contact():
    from piper_sim.mixed_recording import configure_rigid_collection
    _, cfg = configs()
    sim = MujocoBackend(dict(load(), task_scene=cfg))
    try:
        configure_rigid_collection(sim)
        friction = sim.model.geom_friction.copy()
        dimensions = sim.model.geom_condim.copy()
        yellow = np.array([sim.model.body(i).name == 'task/yellow_plug'
                           for i in sim.model.geom_bodyid])
        sim.task.config['yellow_torsional_friction_m'] = .00013
        profile = configure_rigid_collection(sim)
        np.testing.assert_array_equal(sim.model.geom_friction[~yellow], friction[~yellow])
        np.testing.assert_array_equal(sim.model.geom_friction[:, [0, 2]], friction[:, [0, 2]])
        np.testing.assert_array_equal(sim.model.geom_condim[~yellow], dimensions[~yellow])
        assert np.all(sim.model.geom_condim[yellow] == 4)
        assert np.all(sim.model.geom_friction[yellow, 1] == .00013)
        assert profile['yellow_contact_override']['condim'] == 4
    finally:
        sim.close()


def test_recorded_commands_can_change_between_camera_frames(tmp_path):
    from piper_sim.mixed_recording import (configure_rigid_collection, checked_physics_step,
                                          read_recorded_commands, recorded_command_player)
    _, cfg = configs()
    sim = MujocoBackend(dict(load(), task_scene=cfg))
    try:
        configure_rigid_collection(sim)
        path = tmp_path/'commands.csv'
        path.write_text('monotonic,target_j1,target_j2,target_j3,target_j4,target_j5,target_j6,target_gripper_open\n'
                        '100,0,0,0,0,0,0,0.5\n100.01,1,0,0,0,0,0,0.5\n')
        _, sends, targets = read_recorded_commands(path)
        np.testing.assert_allclose(sends, [0, .01])
        assert targets[1, 0] == np.deg2rad(1)
        assert targets[1, 6] == .0475
        target = sim.control_targets()
        left = target[:7].copy()
        player = recorded_command_player(sim, sends, targets, target)
        times = []; commands = []
        actuator = sim.model.actuator('right/joint1').id
        def command(t):
            times.append(t)
            player(t)
            commands.append(sim.data.ctrl[actuator])
        result = checked_physics_step(sim, before_step=command)
        assert result['valid']
        assert len(times) == sim.substeps*sim.decimation
        np.testing.assert_allclose(np.diff(times), sim.model.opt.timestep)
        assert times[0] < .01 < times[-1] < 1/30
        np.testing.assert_array_equal(target[:7], left)
        np.testing.assert_allclose(commands, np.where(np.array(times) < sends[1], 0, np.deg2rad(1)))
        assert sim.data.ctrl[actuator] == np.deg2rad(1)
        assert sim.data.qpos[sim.indices['right']['qpos'][0]] > 0
    finally:
        sim.close()


def test_identified_gain_and_pin_material_reach_solver():
    import mujoco
    from piper_sim.mixed_recording import configure_rigid_collection
    _, cfg = configs()
    cfg.update(right_arm_position_kp=4000, yellow_pin_sliding_friction=.2, yellow_sliding_friction=1.2,
               right_gripper_velocity_kv=97.6)
    sim = MujocoBackend(dict(load(), task_scene=cfg))
    try:
        configure_rigid_collection(sim)
        assert sim.model.actuator('right/gripper').biasprm[2] == -97.6
        assert sim.model.geom('plug/grip_contact/3/0/0').friction[0] == 1.2
        sim.data.qvel[:] = 0
        for side in ('left', 'right'):
            joint = sim.model.joint(f'{side}/joint1')
            sim.data.ctrl[sim.model.actuator(f'{side}/joint1').id] = sim.data.qpos[joint.qposadr[0]] + .001
        plug = sim.model.joint('task/yellow_plug/free').qposadr[0]
        sim.data.qpos[plug] += .003
        mujoco.mj_forward(sim.model, sim.data)
        np.testing.assert_allclose([sim.data.actuator_force[sim.model.actuator(f'{s}/joint1').id]
                                   for s in ('left', 'right')], [.4, 4.], atol=1e-8)
        pin_contacts = [c for c in sim.data.contact if any(
            sim.model.geom(g).name.startswith('task/yellow_plug/pin_') for g in (c.geom1, c.geom2))]
        assert pin_contacts
        assert all(c.dim == 3 and np.isclose(c.friction[0], .2) for c in pin_contacts)
    finally:
        sim.close()


def test_housing_offset_preserves_pin_pattern_and_body_frame():
    _, cfg = configs()
    baseline = MujocoBackend(dict(load(), task_scene=cfg))
    shift = np.array([-.001, .0048, 0.])
    cfg['yellow_plug']['housing_offset_xy_m'] = shift[:2].tolist()
    changed = MujocoBackend(dict(load(), task_scene=cfg))
    try:
        bid = baseline.model.body('task/yellow_plug').id
        shell = np.array([baseline.model.geom_bodyid[i] == bid and
                          not baseline.model.geom(i).name.startswith('task/yellow_plug/pin_')
                          for i in range(baseline.model.ngeom)])
        np.testing.assert_allclose(changed.model.geom_pos[shell]-baseline.model.geom_pos[shell],
                                   np.tile(shift, (shell.sum(), 1)), atol=1e-9)
        np.testing.assert_array_equal(changed.model.geom_pos[~shell], baseline.model.geom_pos[~shell])
        for field in ['body_pos', 'body_ipos', 'body_mass', 'body_inertia']:
            np.testing.assert_array_equal(getattr(changed.model, field), getattr(baseline.model, field))
    finally:
        baseline.close(); changed.close()


def test_fitted_center_of_mass_changes_only_inertial_position():
    _, cfg = configs()
    baseline = MujocoBackend(dict(load(), task_scene=cfg))
    center = [-.0009826436, .0048003229, .005]
    cfg['yellow_plug']['center_of_mass_m'] = center
    changed = MujocoBackend(dict(load(), task_scene=cfg))
    try:
        expected = baseline.model.body_ipos.copy()
        expected[baseline.model.body('task/yellow_plug').id] = center
        np.testing.assert_allclose(changed.model.body_ipos, expected, atol=1e-12)
        for field in ['body_pos', 'body_mass', 'body_inertia', 'geom_pos', 'geom_quat',
                      'geom_size', 'geom_friction', 'geom_contype', 'geom_conaffinity']:
            np.testing.assert_array_equal(getattr(changed.model, field), getattr(baseline.model, field))
    finally:
        baseline.close(); changed.close()
