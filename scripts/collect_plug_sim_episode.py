"""Re-run a verified grasp/insertion trial through physics and save three-view data."""
import argparse
import json
from pathlib import Path
import sys
import time
import shutil
import mujoco
import numpy as np
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.config import ROOT,load
from piper_sim.contracts import Command
from piper_sim.mixed_recording import (MixedEpisodeWriter,configure_rigid_collection,checked_physics_step,
                                      read_recorded_commands,recorded_command_player)
from collect_plug_contact_probes import state_row
from piper_sim.task_archive import archive_episode

PHASES = ['settle','descend','grasp','lift','transport','insert','release','approach',
          'retreat','align','fold','prepare_gripper','abort_retreat','final_hold']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--session',default='20261108_sim_yellow_insert')
    p.add_argument('--episode',default='000001')
    a=p.parse_args()
    report=json.loads((a.trial/'report.json').read_text())
    task_path=a.trial/'task.yaml'
    # Historical trials predate task snapshots and used the nominal scene.
    task=yaml.safe_load((task_path if task_path.exists() else ROOT/'configs/tasks/real_connector.yaml').read_text())
    task['robot_gravity_compensation']=True
    sim=MujocoBackend(dict(load(),task_scene=task),cameras=True)
    profile=configure_rigid_collection(sim);profile['robot_gravity_compensation']=True
    initial=np.load(a.trial/'initial_state.npz')
    sim.data.qpos[:]=initial['qpos'];sim.data.qvel[:]=initial['qvel'];sim.data.ctrl[:]=initial['ctrl']
    mujoco.mj_forward(sim.model,sim.data)
    trajectory=[json.loads(l) for l in (a.trial/'trajectory.jsonl').open()]
    action=sim.control_targets();player=None
    command_path=Path(report['trajectory_provenance']) if 'command_replay' in report else None
    if command_path is not None:
        _,times,targets=read_recorded_commands(command_path)
        player=recorded_command_player(sim,times,targets,action)
    writer=MixedEpisodeWriter(a.root,a.session,a.episode,'sim','yellow_pick_insert',metadata=dict(
        source_trial=str(a.trial),force_sensor_present=False,physics=profile,task_config=task,
        source_trial_report=report,
        trajectory_provenance=report.get('trajectory_provenance',str(a.trial)),
        command_replay=report.get('command_replay','one command per simulation observation'),
        opening_mapping=report.get('opening_mapping',{}),
        fixture_registration=task.get('alignment_status','nominal layout, not calibrated physical ground truth'),
        camera_calibration=(json.loads((ROOT/task['camera_calibration']).read_text())
                            if task.get('camera_calibration') else 'nominal camera rig'),ee_pose_frame='each arm base',
        success_definition='lift yellow plug from holder, transport, insert in white outlet, release, stable seating',
        initial_robot_pose=report['initial_robot_pose'],
        simulation_joint_speed_deg_s=report['speed_deg_s'],privileged_information='simulator only'))
    shutil.copyfile(a.trial/'initial_state.npz',writer.path/'initial_state.npz')
    if command_path is not None:
        shutil.copyfile(command_path,writer.path/'recorded_commands.csv')
    origin=time.time();worst=0.;max_error=0.;max_object_error=0.;max_rotation_error=0.;valid=True
    try:
        for i,reference in enumerate(trajectory):
            if player is None:
                action[:]=reference['action'];sim.apply(Command(action,sim.sim_time,i))
            check=checked_physics_step(sim,.0002,before_step=player);worst=max(worst,check['max_penetration_m'])
            row,obs=state_row(sim,origin+sim.sim_time,action,True)
            error=float(np.max(abs(obs.position-reference['q'])));max_error=max(max_error,error)
            plug=sim.data.body('task/yellow_plug')
            object_error=float(np.linalg.norm(plug.xpos-reference['plug']))
            rotation_error=float(2*np.arccos(np.clip(abs(plug.xquat@reference['plug_quaternion_wxyz']),0,1)))
            max_object_error=max(max_object_error,object_error);max_rotation_error=max(max_rotation_error,rotation_error)
            row['valid/collision']=check['valid'];row['contact/max_penetration_m']=check['max_penetration_m']
            row['phase_index']=PHASES.index(reference['phase'])
            writer.append(row,{'cam_external':obs.images['overhead'],'cam_left_wrist':obs.images['left_wrist'],
                              'cam_right_wrist':obs.images['right_wrist']},privileged=dict(
                phase=reference['phase'],objects=obs.extras['objects'],task_metrics=obs.extras['task']['metrics'],
                contacts=check['contacts'],cameras=obs.extras['cameras'],
                appearance=obs.extras.get('appearance'),qvel=sim.data.qvel.tolist()))
            if not check['valid'] or error>1e-5 or object_error>1e-5 or rotation_error>1e-4:valid=False;break
            if i%150==0:print('recorded',i+1,'/',len(trajectory),reference['phase'],flush=True)
        success=bool(report['complete_pick_insert'] and valid and sim.read().extras['task']['metrics']['success'] and len(writer.rows)==len(trajectory))
        mujoco.mj_saveLastXML(str(writer.path/'model.xml'),sim.model)
        result=writer.close(success=success,policy_training_eligible=success,
            complete_pick_insert=success,all_substeps_valid=valid,max_penetration_m=worst,
            deterministic_replay_max_state_error=max_error,
            deterministic_replay_max_object_error_m=max_object_error,
            deterministic_replay_max_object_rotation_error_rad=max_rotation_error,
            phase_names=PHASES)
        print(json.dumps({k:result[k] for k in ['num_frames','success','policy_training_eligible','quality']},indent=2),flush=True)
        if valid and len(writer.rows)==len(trajectory):
            print('ARCHIVED',archive_episode(a.root,a.session,a.episode),flush=True)
    finally:sim.close()


if __name__=='__main__':main()
