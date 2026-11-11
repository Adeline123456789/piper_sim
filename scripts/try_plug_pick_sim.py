"""Offline physical grasp trial. Never imports a CAN controller or sends hardware commands."""
import argparse
import json
from pathlib import Path
import sys
import mujoco
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.config import ROOT,load
from piper_sim.contracts import LOWER,UPPER,Command
from piper_sim.mixed_recording import configure_rigid_collection,checked_physics_step


class PoseIK:
    def __init__(self, sim, side='right'):
        self.model=sim.model;self.data=mujoco.MjData(sim.model)
        self.data.qpos[:]=sim.data.qpos
        self.ids=sim.indices[side]['qpos'][:6]
        self.bid=sim.model.body(side+'/gripper_center').id
        self.seed=np.deg2rad([-11,95,-65,0,65,-101])
        # Fingers straddle the short direction of the holder, avoiding the
        # black charger alongside the yellow plug on its long direction.
        self.rotation=np.array([[0.,-1.,0.],[0.,0.,1.],[-1.,0.,0.]])

    def solve(self, xyz, rotation=None):
        desired=self.rotation if rotation is None else rotation
        def error(q):
            self.data.qpos[self.ids]=q;mujoco.mj_forward(self.model,self.data)
            return np.r_[10*(self.data.xpos[self.bid]-xyz),
                Rotation.from_matrix(desired.T@self.data.xmat[self.bid].reshape(3,3)).as_rotvec()]
        fit=least_squares(error,np.clip(self.seed,LOWER[7:13]+1e-7,UPPER[7:13]-1e-7),
                          bounds=(LOWER[7:13],UPPER[7:13]),max_nfev=150)
        residual = error(fit.x)
        if np.linalg.norm(residual[:3])/10 > .0001 or np.linalg.norm(residual[3:]) > np.deg2rad(.1):
            raise ValueError(f'Unreachable pose {xyz}: residual {residual}')
        self.seed=fit.x
        return fit.x.copy()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--task',type=Path,default=ROOT/'configs/tasks/real_connector.yaml')
    p.add_argument('--speed',type=float,default=10.,help='simulation trial joint speed, deg/s')
    p.add_argument('--smooth-timing',action='store_true',help='Quintic moves with shorter dwell; simulation only')
    p.add_argument('--insertion-offset-mm',type=float,nargs=2,default=[0.,0.],
                   help='Sim-only deliberate XY misalignment in strip frame for failure data')
    p.add_argument('--grasp-tilt-deg',type=float,default=0.,help='gripper tilt about world Y during grasp')
    p.add_argument('--grasp-roll-deg',type=float,default=0.,help='rotate the grasp about the gripper approach axis')
    p.add_argument('--align-corrections',type=int,default=1,help='pose feedback corrections above the socket')
    p.add_argument('--grasp-height',type=float,default=.030,help='gripper center height above plug face, metres')
    p.add_argument('--grasp-opening',type=float,default=.027,help='gripper actuator target during grasp, metres in simulation coordinates')
    p.add_argument('--preinsert-gap',type=float,default=.045,help='plug face clearance above socket, metres')
    p.add_argument('--start-right-deg',type=float,nargs=6,help='Start from these six right-joint angles, then physically move to the grasp approach')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    task_cfg=yaml.safe_load(a.task.read_text())
    task_cfg['robot_gravity_compensation']=True
    if task_cfg.get('camera_calibration'):
        source=ROOT/task_cfg['camera_calibration']
        snapshot=(a.output/'cameras.json').resolve()
        snapshot.write_text(source.read_text())
        task_cfg['camera_calibration']=str(snapshot)
    (a.output/'task.yaml').write_text(yaml.safe_dump(task_cfg,sort_keys=False))
    sim=MujocoBackend(dict(load(),task_scene=task_cfg))
    profile=configure_rigid_collection(sim)
    # Approximate the real controller's gravity-supported joint-position mode.
    # Objects retain gravity. This parameterization is recorded, not identified.
    profile['robot_gravity_compensation']=True
    ik=PoseIK(sim);plug=sim.data.body('task/yellow_plug').xpos.copy()
    board_rotation=sim.data.body('task/board').xmat.reshape(3,3)
    ik.rotation=board_rotation@Rotation.from_euler('z',90.,degrees=True).as_matrix()@ik.rotation
    ik.rotation=Rotation.from_euler('y',a.grasp_tilt_deg,degrees=True).as_matrix()@ik.rotation
    ik.rotation=ik.rotation@Rotation.from_euler('x',a.grasp_roll_deg,degrees=True).as_matrix()
    ik.seed[5]=np.deg2rad(-101+a.grasp_roll_deg)
    q=np.array([0,0,0,0,0,0,.060]*2,dtype=float)
    approach=ik.solve(plug+[0,0,.074])
    q[7:13]=approach if a.start_right_deg is None else np.deg2rad(a.start_right_deg)
    sim.set_observation_state(q)
    np.savez(a.output/'initial_state.npz',qpos=sim.data.qpos,qvel=sim.data.qvel,ctrl=sim.data.ctrl)
    history=[];worst=0.;phase='initialization';success=False;complete=False
    def step(command):
        nonlocal worst
        sim.apply(Command(command,sim.sim_time,len(history)))
        check=checked_physics_step(sim,.0002);worst=max(worst,check['max_penetration_m'])
        history.append(dict(time=sim.sim_time,phase=phase,q=sim.read().position.tolist(),
            action=command.tolist(),plug=sim.data.body('task/yellow_plug').xpos.tolist(),
            plug_quaternion_wxyz=sim.data.body('task/yellow_plug').xquat.tolist(),
            grip=sim.data.body('right/gripper_center').xpos.tolist(),
            penetration=check['max_penetration_m']))
        if not check['valid']:
            (a.output/'collision.json').write_text(json.dumps(check,indent=2))
            raise RuntimeError('Penetration exceeds collection tolerance')
    def move(target, duration=None):
        nonlocal q
        joint_delta=np.max(abs((target-q)[[0,1,2,3,4,5,7,8,9,10,11,12]]))
        # Quintic progress has peak derivative 1.875; keep the stated speed bound.
        contact_phase=phase in ('descend','grasp','align','insert','release')
        speed=min(a.speed,12.) if contact_phase else a.speed
        travel=(1.875 if a.smooth_timing else 1.)*joint_delta/np.deg2rad(speed)
        # Match the real executor's 47.5 mm/s gripper-target limit before physics.
        grip_travel=(1.875 if a.smooth_timing else 1.)*np.max(abs((target-q)[[6,13]]))/.0475
        duration=max(duration or (.35 if a.smooth_timing else 1.),travel,grip_travel)
        before=q.copy()
        for alpha in np.linspace(0,1,int(np.ceil(duration*30))+1)[1:]:
            progress=alpha**3*(10.-15.*alpha+6.*alpha**2) if a.smooth_timing else alpha
            step(before+(target-before)*progress)
        q=target.copy()
        for _ in range(0 if a.smooth_timing else 15):step(q)
        sim.save_overview(a.output/(phase+'_'+str(len(history))+'.png'))
        from PIL import Image
        camera=mujoco.MjvCamera();camera.lookat[:]=sim.data.body('task/yellow_plug').xpos+[0,0,.025]
        camera.distance=.32;camera.azimuth=90;camera.elevation=-15
        Image.fromarray(sim._render(camera,800,600)).save(a.output/(phase+'_'+str(len(history))+'_detail.png'))
        print(phase,'t',sim.sim_time,'plug',sim.data.body('task/yellow_plug').xpos,
              'grip',sim.data.body('right/gripper_center').xpos,'max_mm',worst*1000,flush=True)
    error=None
    try:
        if a.start_right_deg is not None:
            phase='approach';target=q.copy();target[7:13]=approach;move(target)
        if not a.smooth_timing:
            phase='settle';move(q,1.)
        plug=sim.data.body('task/yellow_plug').xpos.copy()
        phase='descend';target=q.copy();target[7:13]=ik.solve(plug+[0,0,a.grasp_height]);move(target)
        phase='grasp';target=q.copy();target[13]=a.grasp_opening;move(target,1. if a.smooth_timing else 3.)
        phase='lift';target=q.copy();target[7:13]=ik.solve(plug+[0,0,.074]);move(target)
        body=sim.data.body('task/yellow_plug')
        pins=np.concatenate([part['points'] for part in sim.task.fixture.parts['yellow_plug']
                             if part['name'].startswith('pin_')])
        clearance=float((pins@body.xmat.reshape(3,3).T+body.xpos)[:,2].min()-plug[2])
        success=bool(clearance>.002)
        if not success:raise RuntimeError(f'Plug pins have not cleared the holder: {clearance:.6f} m')
        def body_transform(name):
            body=sim.data.body(name);t=np.eye(4);t[:3,:3]=body.xmat.reshape(3,3);t[:3,3]=body.xpos
            return t
        def place_over_socket(gap):
            relative=np.linalg.inv(body_transform('right/gripper_center'))@body_transform('task/yellow_plug')
            socket=body_transform('task/strip')
            desired=np.eye(4);desired[:2,3]=sim.task.outlets[sim.task.config['target_index']]
            desired[:2,3]+=np.array(a.insertion_offset_mm)/1000.
            desired[2,3]=sim.task.socket_face_z+.0002+gap
            # A small carry tilt keeps the raised approach within the J5 limit.
            if gap>.03:desired[:3,:3]=Rotation.from_euler('x',5.,degrees=True).as_matrix()
            desired=socket@desired
            grip=desired@np.linalg.inv(relative)
            target=q.copy();target[7:13]=ik.solve(grip[:3,3],grip[:3,:3])
            return target
        phase='transport';preinsert=place_over_socket(a.preinsert_gap);move(preinsert)
        # Straighten above the pin tips, before making insertion contact.
        if a.preinsert_gap>.03:
            phase='align'
            for _ in range(a.align_corrections):move(place_over_socket(.025),.5 if a.smooth_timing else None)
        phase='insert';move(place_over_socket(0.),1.8 if a.smooth_timing else 4.)
        phase='release';target=q.copy();target[13]=.060;move(target,.8 if a.smooth_timing else 2.)
        # Retrace the already reachable insertion approach with fingers open.
        phase='retreat';target=preinsert.copy();target[13]=.060
        move(target,1.5 if a.smooth_timing else 2.)
        phase='fold';target=q.copy();target[7:13]=0.;move(target)
        for _ in range(3 if a.smooth_timing else 30):step(q)
        complete=bool(sim.read().extras['task']['metrics']['success'])
        if not complete:raise RuntimeError('Released plug did not meet stable insertion criteria')
    except (RuntimeError,ValueError) as exc:
        error=str(exc);print(phase,error,flush=True)
        sim.save_overview(a.output/'failure.png')
    finally:
        np.savez(a.output/'final_state.npz',qpos=sim.data.qpos,qvel=sim.data.qvel,ctrl=sim.data.ctrl)
        (a.output/'trajectory.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
        (a.output/'report.json').write_text(json.dumps(dict(profile=profile,
            grasp_lift_success=success,complete_pick_insert=complete,error=error,last_phase=phase,
            max_penetration_m=worst,frames=len(history),hardware_commands=0,
            initial_robot_pose=('IK approach pose; initialization only' if a.start_right_deg is None
                                else 'explicit right-joint entry; physical approach included'),
            initial_right_joint_deg=a.start_right_deg,speed_deg_s=a.speed,
            contact_joint_speed_deg_s=min(a.speed,12.),
            gripper_target_speed_m_s=.0475,
            timing_profile=('quintic shorter-dwell simulation' if a.smooth_timing else 'original linear'),
            deliberate_insertion_offset_mm=a.insertion_offset_mm,
            grasp_tilt_deg=a.grasp_tilt_deg,grasp_height_m=a.grasp_height,
            grasp_opening_m=a.grasp_opening,grasp_roll_deg=a.grasp_roll_deg,
            align_corrections=a.align_corrections,alignment_observations="simulator privileged pose",
            preinsert_gap_m=a.preinsert_gap,preinsert_tilt_deg=5. if a.preinsert_gap>.03 else 0.),indent=2))
        sim.close()


if __name__=='__main__':main()
