"""Build a separate, provisional RGB-D-aligned scene and render diagnostics.

No robot commands. Raw feedback qpos placement is used only for static camera
diagnostics, never emitted as a physical training rollout.
"""
import argparse
import json
from pathlib import Path
import struct
import sys
import mujoco
import numpy as np
from PIL import Image,ImageDraw
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from piper_sim.config import ROOT,load
from piper_sim.backends.mujoco import MujocoBackend
from register_plug_fixture_views import fit_camera


def body_transform(body):
    t=np.eye(4);t[:3,:3]=body.xmat.reshape(3,3);t[:3,3]=body.xpos
    return t


def apply_feedback_for_diagnostic(sim,frames):
    angles={}
    for side,channel in [('left','can_arm1'),('right','can_arm2')]:
        f=frames[channel]
        q=np.array([v/1000 for i in (0x2a5,0x2a6,0x2a7)
                    for v in struct.unpack('>ii',bytes.fromhex(f[hex(i)]['data_hex']))])
        width=struct.unpack('>i',bytes.fromhex(f['0x2a8']['data_hex'])[:4])[0]*1e-6
        sim.data.qpos[sim.indices[side]['qpos']]=[*np.deg2rad(q),width/2,width/2]
        angles[side]=dict(joints_deg=q.tolist(),reported_gripper_width_m=width)
    mujoco.mj_forward(sim.model,sim.data);sim._images=None
    return angles


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    registration=json.loads((a.capture/'fixture_view_registration.json').read_text())
    table=json.loads((a.capture/'table_registration.json').read_text())
    world_external=np.array(table['T_world_rgb_estimate'])
    strip=fit_camera(a.capture,'external',np.array([[259,243],[365,242],[370,275],[255,276]],float),
                     dimensions=(.212,.076,.026),occluded=(),label='strip')
    world_strip=world_external@np.array(strip['T_rgb_holder'])
    frames=json.loads((a.capture/'can_latest.json').read_text())
    cfg=yaml.safe_load((ROOT/'configs/tasks/real_connector.yaml').read_text())
    sim=MujocoBackend(dict(load(),task_scene=cfg),cameras=True)
    try:
        angles=apply_feedback_for_diagnostic(sim,frames)
        baseline=sim.read().images
        link6=body_transform(sim.data.body('right/link6'))
        right_local=np.linalg.inv(link6)@np.array(registration['T_world_right_rgb_estimate'])
    finally:sim.close()
    cameras={}
    for real,name in [('external','overhead'),('left_wrist','left_wrist'),('right_wrist','right_wrist')]:
        raw=json.loads((a.capture/real/'camera.json').read_text())
        c=dict(K=raw['K'],serial=raw['serial'],raw_distortion=dict(model=raw['distortion_model'],coeffs=raw['distortion_coeffs']),
               status='SDK intrinsics; CAD extrinsics not calibrated')
        if name in ('overhead','right_wrist'):
            c.update(parent_body='world' if name=='overhead' else 'right/link6',
                T_parent_rgb=(world_external if name=='overhead' else right_local).tolist(),
                status='SDK intrinsics; provisional single-pose RGB-D extrinsics with CAD world XY/yaw anchor')
        cameras[name]=c
    calibration=dict(source_capture=str(a.capture),motion_qualified=False,cameras=cameras,
        raw_feedback_at_calibration=angles,fixtures=dict(holder=registration,strip=strip),
        limitations=['single-pose fit; not a validated hand-eye calibration',
                     'raw distorted images differ from the pinhole render',
                     'left wrist extrinsics remain CAD-based',
                     'gripper encoder opening is not yet physically calibrated'])
    cal_path=(a.output/'cameras.json').resolve();cal_path.write_text(json.dumps(calibration,indent=2)+'\n')
    cfg['camera_calibration']=str(cal_path)
    cfg['overhead_row_offset_px']=0
    cfg.pop('right_wrist_optical_rotvec',None)
    holder=np.array(registration['T_world_holder_estimate'])
    for name,t in [('holder',holder),('strip',world_strip)]:
        cfg[name+'_xy_m']=t[:2,3].tolist();cfg[name+'_yaw_deg']=float(np.rad2deg(np.arctan2(t[1,0],t[0,0])))
    cfg['alignment_status']='provisional RGB-D visual registration, not qualified for hardware motion'
    task_path=(a.output/'task.yaml').resolve();task_path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    sim=MujocoBackend(dict(load(),task_scene=task_path),cameras=True)
    try:
        apply_feedback_for_diagnostic(sim,frames);after=sim.read().images
        canvas=Image.new('RGB',(1920,1512),'white');draw=ImageDraw.Draw(canvas)
        for row,(real,name) in enumerate([('external','overhead'),('right_wrist','right_wrist'),('left_wrist','left_wrist')]):
            for col,(title,im) in enumerate([('REAL '+real,Image.open(a.capture/real/'last.png')),
                                           ('REFERENCE SIM '+name,Image.fromarray(baseline[name])),
                                           ('PROVISIONAL ALIGNED SIM '+name,Image.fromarray(after[name]))]):
                draw.text((col*640+8,row*504+5),title,fill='black');canvas.paste(im,(col*640,row*504+24))
        canvas.save(a.output/'comparison.png')
        for name,im in after.items():Image.fromarray(im).save(a.output/(name+'_aligned.png'))
        print(json.dumps(dict(task=str(task_path),cameras=str(cal_path),holder_xy_m=cfg['holder_xy_m'],
            holder_yaw_deg=cfg['holder_yaw_deg'],strip_xy_m=cfg['strip_xy_m'],strip_yaw_deg=cfg['strip_yaw_deg'],
            strip_reprojection_rms_px=strip['reprojection_rms_px']),indent=2))
    finally:sim.close()


if __name__=='__main__':main()
