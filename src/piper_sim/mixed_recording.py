"""Connector-compatible episodes with explicit real/sim provenance and validity."""
import csv
import json
from pathlib import Path

import av
import h5py
import numpy as np

from .contracts import Command

CAMERAS = ('cam_external', 'cam_left_wrist', 'cam_right_wrist')


def read_recorded_commands(path):
    with Path(path).open() as source:
        rows = list(csv.DictReader(source))
    times = np.array([float(r['monotonic']) for r in rows]); times -= times[0]
    actions = np.array([[*np.deg2rad([float(r[f'target_j{i}']) for i in range(1, 7)]),
                         float(r['target_gripper_open'])*.095] for r in rows])
    assert np.all(np.diff(times) > 0) and np.isfinite(actions).all()
    return rows, times, actions


def recorded_command_player(sim, times, actions, target):
    """Update the shared target at each physics substep, independently of camera FPS."""
    start = sim.data.time
    def apply(physics_time):
        index = max(0, np.searchsorted(times, physics_time-start, side='right')-1)
        target[7:] = actions[index]
        sim.apply(Command(target, physics_time, index))
    return apply


class MixedEpisodeWriter:
    """Actions are targets active during the interval ending at this observation.

    Unknown real actions must be NaN with action_valid=False, never copied from
    observations. Sim-only ground truth belongs in privileged.jsonl, not state.
    Images are RGB uint8. Joint positions: degrees; EE position: mm; quat: xyzw;
    legacy gripper_open: measured width / 95 mm (raw width is also retained).
    """
    def __init__(self, root, session, episode, domain, purpose, fps=30, metadata=None):
        if domain not in ('real', 'sim'):
            raise ValueError('domain must be real or sim')
        self.root, self.session, self.episode = Path(root), session, episode
        self.path = self.root / 'proprio_stats' / session / episode
        self.path.mkdir(parents=True, exist_ok=False)
        self.video_path = self.root / 'observations' / session / episode / 'videos'
        self.video_path.mkdir(parents=True, exist_ok=False)
        self.meta = dict(schema='connector_mixed.v1', domain=domain, purpose=purpose,
                         fps=fps, session_id=session, episode_id=episode,
                         task_name='plug_in', action_alignment='target active during interval ending at observation',
                         units=dict(joint_positions='degree', end_effector_pos='mm',
                                    end_effector_quat='xyzw', gripper_open='width_mm / 95',
                                    gripper_width_m='metre'), **(metadata or {}))
        self.meta.setdefault('timestamp_clock', 'synthetic Unix origin + simulation time' if domain=='sim'
                             else 'RealSense global_time Unix seconds')
        self.meta.setdefault('timestamp_monotonic_clock', 'local host receipt/recording monotonic')
        self.fps, self.domain = fps, domain
        self.rows, self.videos, self.streams = [], {}, {}
        self.privileged = None

    def append(self, row, images, privileged=None):
        if set(images) != set(CAMERAS):
            raise ValueError('One image from each of the three cameras is required')
        if self.rows and set(row) != set(self.rows[0]):
            raise ValueError('Episode fields must be consistent')
        if privileged is not None and self.domain != 'sim':
            raise ValueError('Real frames cannot be assigned simulator ground truth')
        if bool(row.get('valid/action', False)):
            if not all(np.isfinite(v).all() for k,v in row.items() if k.startswith('action/')):
                raise ValueError('Valid actions cannot contain missing values')
        for camera, rgb in images.items():
            if camera not in self.videos:
                container = av.open(str(self.video_path / (camera+'_rgb.mp4')), 'w')
                stream = container.add_stream('libx264', rate=self.fps)
                stream.width, stream.height = rgb.shape[1], rgb.shape[0]
                stream.pix_fmt = 'yuv420p'
                stream.options = {'crf': '18', 'preset': 'fast'}
                self.videos[camera], self.streams[camera] = container, stream
            frame = av.VideoFrame.from_ndarray(np.asarray(rgb, dtype=np.uint8), format='rgb24')
            for packet in self.streams[camera].encode(frame):
                self.videos[camera].mux(packet)
        if privileged is not None:
            if self.privileged is None:
                self.privileged = (self.path/'privileged.jsonl').open('w')
            self.privileged.write(json.dumps(dict(frame=len(self.rows), **privileged))+'\n')
        self.rows.append({k: np.asarray(v).copy() for k, v in row.items()})

    def close(self, success=False, policy_training_eligible=False, **quality):
        for name, video in self.videos.items():
            for packet in self.streams[name].encode():
                video.mux(packet)
            video.close()
        if self.privileged:
            self.privileged.close()
        self.meta.update(num_frames=len(self.rows), success=success,
                         policy_training_eligible=policy_training_eligible, quality=quality)
        with h5py.File(self.path/'proprio_stats.h5', 'w') as f:
            for key in self.rows[0] if self.rows else []:
                f.create_dataset(key, data=np.stack([r[key] for r in self.rows]))
            for key, value in self.meta.items():
                f.attrs[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
        (self.path/'metadata.json').write_text(json.dumps(self.meta, indent=2)+'\n')
        return self.meta


def collision_snapshot(model, data):
    """All active contacts, in metres; negative distance is solver penetration."""
    contacts = []
    for c in data.contact:
        contacts.append(dict(geom1=model.geom(c.geom1).name, geom2=model.geom(c.geom2).name,
                             body1=model.body(model.geom_bodyid[c.geom1]).name,
                             body2=model.body(model.geom_bodyid[c.geom2]).name,
                             distance_m=float(c.dist), position_world_m=c.pos.tolist()))
    return contacts


def configure_rigid_collection(sim):
    """Numerically resolved rigid contacts; parameters remain unmeasured estimates.

    1/3840 s integration, 30 Hz observations. This is an explicit collection
    profile and does not mutate the reference task/robot asset files.
    """
    sim.substeps = 32
    sim.model.opt.timestep = sim.dt/sim.substeps
    config = sim.task.config if sim.task else {}
    arm_gain = config.get('right_arm_position_kp')
    pin_friction = config.get('yellow_pin_sliding_friction')
    sliding = config.get('yellow_sliding_friction')
    pad_sliding = config.get('right_gripper_pad_sliding_friction')
    gripper_kv = config.get('right_gripper_velocity_kv')
    for name, value in [('right_arm_position_kp', arm_gain), ('yellow_pin_sliding_friction', pin_friction),
                        ('yellow_sliding_friction', sliding), ('right_gripper_pad_sliding_friction', pad_sliding),
                        ('right_gripper_velocity_kv', gripper_kv)]:
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f'{name} must be finite and positive')
    arm_ids = sim.indices['right']['ctrl'][:6]
    if arm_gain is not None:
        sim.model.actuator_gainprm[arm_ids, 0] = arm_gain
        sim.model.actuator_biasprm[arm_ids, 1] = -arm_gain
    if gripper_kv is not None:
        sim.model.actuator_biasprm[sim.model.actuator('right/gripper').id, 2] = -gripper_kv
    torsion = sim.task.config.get('yellow_torsional_friction_m') if sim.task else None
    if torsion is not None and (not np.isfinite(torsion) or torsion <= 0):
        raise ValueError('yellow_torsional_friction_m must be finite and positive')
    for i in range(sim.model.ngeom):
        if sim.model.body(sim.model.geom_bodyid[i]).name.startswith('task/'):
            sim.model.geom_solref[i] = [.0006, 1]
            sim.model.geom_solimp[i, :3] = [.99, .999, .0001]
            sim.model.geom_priority[i] = 1
            # Finite finger pads resist twist/roll. With condim=3 the existing
            # torsional/rolling friction coefficients are ignored, allowing a
            # grasped tapered plug to rotate about two point contacts.
            sim.model.geom_condim[i] = 6
            # Experimental failure-replay fit; keep separate from nominal task settings.
            if torsion is not None and sim.model.body(sim.model.geom_bodyid[i]).name == 'task/yellow_plug':
                sim.model.geom_condim[i] = 4
                sim.model.geom_friction[i, 1] = torsion
            if sliding is not None and sim.model.body(sim.model.geom_bodyid[i]).name == 'task/yellow_plug':
                sim.model.geom_friction[i, 0] = sliding
            if pin_friction is not None and sim.model.geom(i).name.startswith('task/yellow_plug/pin_'):
                # Higher priority preserves the metal/plastic fit against the holder's friction.
                sim.model.geom_priority[i] = 2
                sim.model.geom_condim[i] = 3
                sim.model.geom_friction[i, 0] = pin_friction
    if pad_sliding is not None:
        # Keep the plug/fixture friction unchanged while fitting the rubber pads.
        model = sim.model
        reference = model.geom('task/yellow_plug/rim').id
        for name in ('gripper_right_36', 'gripper_right_46'):
            pad = model.geom(name).id
            for field in ('geom_condim', 'geom_friction', 'geom_solref', 'geom_solimp'):
                getattr(model, field)[pad] = getattr(model, field)[reference]
            model.geom_priority[pad] = 3
            model.geom_friction[pad, 0] = pad_sliding
    return dict(timestep_s=sim.model.opt.timestep, task_solref=[.0006,1], task_condim=6,
                right_arm_kp=sim.model.actuator_gainprm[arm_ids, 0].tolist(),
                right_arm_kv=(-sim.model.actuator_biasprm[arm_ids, 2]).tolist(),
                right_gripper_velocity_kv=float(-sim.model.actuator_biasprm[sim.model.actuator('right/gripper').id, 2]),
                yellow_pin_sliding_friction=pin_friction,
                yellow_sliding_friction=sliding,
                right_gripper_pad_sliding_friction=pad_sliding,
                yellow_contact_override=None if torsion is None else dict(condim=4, torsional_friction_m=torsion),
                task_solimp_first3=[.99,.999,.0001],
                parameter_status='numerical rigid-contact profile, not experimentally identified')


def checked_physics_step(sim, max_penetration_m=.0002, before_step=None):
    """Inspect EVERY integrator step, including contacts between camera frames.

    Exceeding the tolerance rejects the rollout; it does not delete the collision
    or move qpos out of the obstacle. The caller retains the failed trial report.
    """
    import mujoco
    worst = 0.
    for _ in range(sim.decimation*sim.substeps):
        if before_step is not None:
            before_step(sim.data.time)
        mujoco.mj_step(sim.model, sim.data)
        mujoco.mj_forward(sim.model, sim.data)
        if sim.data.ncon:
            worst = max(worst, -min(c.dist for c in sim.data.contact))
        if worst > max_penetration_m:
            return dict(valid=False, max_penetration_m=worst,
                        contacts=collision_snapshot(sim.model, sim.data))
    mujoco.mj_forward(sim.model, sim.data)
    sim._images = None
    sim.task.metrics(sim.task_adapter.task_poses(sim.data, sim.task), sim.dt*sim.decimation)
    return dict(valid=True, max_penetration_m=worst,
                contacts=collision_snapshot(sim.model, sim.data))
