"""Collect physics-only seated/blocked contact probes, not full pick-insert demos."""
import argparse
import json
from pathlib import Path
import sys
import time
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.config import load,ROOT
from piper_sim.contracts import LOWER,UPPER
from piper_sim.mixed_recording import MixedEpisodeWriter,checked_physics_step,configure_rigid_collection
from piper_sim.tasks.recorded_scenes import matrix


def state_row(sim, timestamp, action, valid):
    obs=sim.read();row=dict(timestamp=timestamp,timestamp_monotonic=time.monotonic(),sim_time=sim.sim_time)
    row['valid/action']=valid;row['valid/state']=True
    for i,side in enumerate(('left','right')):
        q=obs.position[7*i:7*i+7];target=action[7*i:7*i+7]
        # Match the reference controller EE frame: link6 relative to arm base.
        base=sim.data.body(side+'/base_link');tip=sim.data.body(side+'/link6')
        r=base.xmat.reshape(3,3);p=r.T@(tip.xpos-base.xpos)
        quat=Rotation.from_matrix(r.T@tip.xmat.reshape(3,3)).as_quat()
        prefix=f'state/{side}_arm/'
        row[prefix+'joint_positions']=np.rad2deg(q[:6])
        row[prefix+'gripper_open']=q[6]/.095
        row[prefix+'gripper_width_m']=q[6]
        row[prefix+'end_effector_pos']=p*1000
        row[prefix+'end_effector_quat']=quat
        row[f'action/{side}_arm/joint_positions']=np.rad2deg(target[:6])
        row[f'action/{side}_arm/gripper_open']=target[6]/.095
    return row,obs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--session',default='20261108_sim_contact_probes')
    a=p.parse_args()
    sim=MujocoBackend(dict(load(),task_scene=ROOT/'configs/tasks/real_connector.yaml'),cameras=True)
    profile=configure_rigid_collection(sim)
    try:
        for index,offset in enumerate((0.,.012),1):
            sim.reset()
            # Initialization only. All subsequent movement is mj_step; no weld,
            # object teleport or kinematic placement during recorded frames.
            task=sim.task;strip=task.layout['poses']['strip'];x,y=task.outlets[3]
            xyz=(matrix(strip)@[x+offset,y,task.socket_face_z+.021,1])[:3]
            adr=sim.model.joint('task/yellow_plug/free').qposadr[0]
            sim.data.qpos[adr:adr+7]=[*xyz,*strip['quaternion_wxyz']]
            # Park the empty arms so this probe isolates plug/socket contacts.
            q=np.array([0,0,0,0,0,0,.075]*2,dtype=float)
            sim.set_observation_state(q)
            writer=MixedEpisodeWriter(a.root,a.session,f'{index:06d}','sim','gravity_contact_probe',
                metadata=dict(force_sensor_present=False,physics=profile,offset_m=offset,
                              initialization='yellow plug placed above outlet before rollout',
                              fixture_registration='nominal reference layout; not real ground truth',
                              camera_calibration='nominal render cameras',ee_pose_frame='each arm base',
                              task_description='numerical contact calibration, not a pick-insert demonstration'))
            worst=0.;valid=True;origin=time.time()
            for frame in range(45):
                check=checked_physics_step(sim,.0002)
                worst=max(worst,check['max_penetration_m'])
                row,obs=state_row(sim,origin+sim.sim_time,q,True)
                row['valid/collision']=check['valid'];row['contact/max_penetration_m']=check['max_penetration_m']
                images={ 'cam_external':obs.images['overhead'],
                         'cam_left_wrist':obs.images['left_wrist'],
                         'cam_right_wrist':obs.images['right_wrist'] }
                writer.append(row,images,privileged=dict(objects=obs.extras['objects'],
                    task_metrics=obs.extras['task']['metrics'],contacts=check['contacts'],
                    cameras=obs.extras['cameras'],joint_velocity=sim.data.qvel.tolist()))
                if not check['valid']:valid=False;break
            metrics=sim.read().extras['task']['metrics']
            print(json.dumps(writer.close(success=False,policy_training_eligible=False,
                  complete_pick_insert=False,contact_probe_seated=metrics['success'],
                  all_substeps_valid=valid,max_penetration_m=worst,
                  reason='contact calibration only'),indent=2),flush=True)
    finally:sim.close()


if __name__=='__main__':main()
