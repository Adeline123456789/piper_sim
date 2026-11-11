"""Replay recorded real targets through MuJoCo physics; no hardware imports."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.config import load
from piper_sim.mixed_recording import (configure_rigid_collection, checked_physics_step,
                                      read_recorded_commands, recorded_command_player)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--trial', type=Path, required=True)
    p.add_argument('--task', type=Path, help='Offline candidate scene; objects start seated in its holder')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gripper-force-limit-n', type=float, help='Offline right-gripper force limit candidate')
    a = p.parse_args()
    if a.gripper_force_limit_n is not None and not (np.isfinite(a.gripper_force_limit_n) and a.gripper_force_limit_n > 0):
        p.error('gripper force limit must be finite and positive')
    a.output.mkdir(parents=True, exist_ok=False)
    rows, times, actions = read_recorded_commands(a.capture/'commanded_probe.csv')
    feedback = np.array([[*np.deg2rad([float(r[f'feedback_j{i}']) for i in range(1, 7)]), float(r['feedback_gripper_open'])*.095] for r in rows])
    assert np.isfinite(feedback).all()
    task = yaml.safe_load((a.task or a.trial/'task.yaml').read_text())
    task['robot_gravity_compensation'] = True
    (a.output/'task.yaml').write_text(yaml.safe_dump(task))
    sim = MujocoBackend(dict(load(), task_scene=task))
    profile = configure_rigid_collection(sim)
    if a.gripper_force_limit_n is not None:
        sim.model.actuator_forcerange[sim.model.actuator('right/gripper').id] = [-a.gripper_force_limit_n, a.gripper_force_limit_n]
    profile['right_gripper_force_range_n'] = sim.model.actuator_forcerange[sim.model.actuator('right/gripper').id].tolist()
    initial = np.load(a.trial/'initial_state.npz')
    sim.data.qpos[:] = initial['qpos']; sim.data.qvel[:] = initial['qvel']
    sim.data.ctrl[:] = initial['ctrl']
    if a.task:
        sim.task_adapter.reset_task(sim.model, sim.data, sim.task)
    q = sim.control_targets()
    q[7:13] = np.deg2rad([float(rows[0][f'feedback_j{i}']) for i in range(1, 7)])
    q[13] = float(rows[0]['feedback_gripper_open'])*.095
    sim.set_observation_state(q)
    np.savez(a.output/'initial_state.npz', qpos=sim.data.qpos, qvel=sim.data.qvel, ctrl=sim.data.ctrl)
    initial_time = sim.data.time
    history = []; stages = {}; worst = 0.; valid = True
    period = sim.dt*sim.decimation
    apply_recorded_target = recorded_command_player(sim, times, actions, q)
    try:
        for t in np.arange(0, times[-1], period):
            index = np.searchsorted(times, t, side='right')-1
            phase = rows[index]['phase']
            check = checked_physics_step(sim, .0002, before_step=apply_recorded_target)
            worst = max(worst, check['max_penetration_m'])
            obs = sim.read(); plug = sim.data.body('task/yellow_plug')
            grip = sim.data.body('right/gripper_center')
            feedback_time = min(sim.data.time-initial_time, times[-1])
            real_q = np.array([np.interp(feedback_time, times, feedback[:, j]) for j in range(7)])
            relative = grip.xmat.reshape(3, 3).T@(plug.xpos-grip.xpos)
            row = dict(time=sim.sim_time, phase=phase, q=obs.position.tolist(), action=q.tolist(),
                       physics_qpos=sim.data.qpos.tolist(), physics_qvel=sim.data.qvel.tolist(),
                       physics_ctrl=sim.data.ctrl.tolist(),
                       simulated_actuator_force=sim.data.actuator_force.tolist(),
                       real_feedback_gripper_m=float(real_q[6]),
                       real_feedback_si=real_q.tolist(), feedback_time_s=float(feedback_time),
                       joint_error_deg=np.rad2deg(obs.position[7:13]-real_q[:6]).tolist(),
                       gripper_error_mm=float((obs.position[13]-real_q[6])*1000),
                       plug=plug.xpos.tolist(), plug_quaternion_wxyz=plug.xquat.tolist(),
                       plug_in_gripper_m=relative.tolist(), metrics=obs.extras['task']['metrics'],
                       penetration=check['max_penetration_m'])
            history.append(row)
            if phase not in stages:
                stages[phase] = dict(first_frame=len(history)-1)
                print('PHYSICS_REPLAY', phase, flush=True)
            stages[phase].update(last_frame=len(history)-1, metrics=row['metrics'],
                                 sim_gripper_mm=float(obs.position[13]*1000),
                                 real_gripper_mm=row['real_feedback_gripper_m']*1000,
                                 plug_in_gripper_mm=(relative*1000).tolist())
            if not check['valid']:
                valid = False
                (a.output/'collision.json').write_text(json.dumps(check, indent=2))
                break
        sim.save_overview(a.output/'final.png')
        errors = np.array([r['joint_error_deg'] for r in history])
        report = dict(joint_rmse_deg=np.sqrt(np.mean(errors**2, axis=0)).tolist(),
                      gripper_rmse_mm=float(np.sqrt(np.mean([r['gripper_error_mm']**2 for r in history]))),
                      complete_pick_insert=bool(valid and history[-1]['metrics']['success']),
                      all_substeps_valid=valid, frames=len(history), stages=stages,
                      max_penetration_m=worst, hardware_commands=0, profile=profile,
                      initial_robot_pose='measured entry; subsequent states integrated by physics',
                      speed_deg_s=12., trajectory_provenance=str(a.capture/'commanded_probe.csv'),
                      observation_hz=1/period,
                      command_replay='zero-order hold of original sends, updated every physics substep',
                      limitations='Nominal object geometry, friction and fixture registration are estimates; no real object poses injected.')
        (a.output/'trajectory.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
        (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2), flush=True)
    finally:
        sim.close()


if __name__ == '__main__':
    main()
