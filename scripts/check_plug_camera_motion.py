"""Held-out RGB optical-flow versus CAN/FK camera motion; no hardware access.

Track features on the exposed planar holder surface. Lift their first-frame rays
onto the known holder plane, then predict their subsequent pixels using actual
CAN feedback and the independently saved camera-mount estimate. This checks
relative motion only; it does not measure absolute hand-eye accuracy.
"""
import argparse
import bisect
import csv
import json
from pathlib import Path
import struct
import sys

import cv2
import mujoco
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.config import load


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--task',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    directory=a.capture/'right_wrist'
    camera=json.loads((directory/'camera.json').read_text())
    k=np.array(camera['K'])
    if np.any(camera['distortion_coeffs']):
        raise ValueError('This motion check expects the zero-distortion right RGB stream')
    samples=camera['samples']
    if any(s['timestamp_domain']!='timestamp_domain.global_time' for s in samples):
        raise ValueError('Global exposure timestamps are required')
    raw={i:[] for i in (0x2a5,0x2a6,0x2a7)}
    for line in (a.capture/'raw_can.jsonl').open():
        r=json.loads(line)
        if r['can']=='can_arm2' and r['id'] in raw and not r['host_local']:
            raw[r['id']].append((r['host_monotonic'],struct.unpack('>ii',bytes.fromhex(r['data_hex']))))
    times={i:[r[0] for r in rows] for i,rows in raw.items()}
    cfg=yaml.safe_load(a.task.read_text())
    if cfg.get('appearance'):cfg['appearance']['randomization']['enabled']=False
    sim=MujocoBackend(dict(load(),task_scene=cfg))
    plane=np.array(json.loads(a.registration.read_text())['T_world_holder_estimate'])
    normal=plane[:3,2];point=plane[:3,3]+normal*.033
    records=[]
    try:
        def pose(index):
            s=samples[index];t=s['rgb_device_ms']/1000+s['host_monotonic']-s['host_unix']
            pairs=[];ages=[]
            for ident in raw:
                j=bisect.bisect_right(times[ident],t)-1
                if j<0:raise ValueError('No feedback before exposure')
                pairs.extend(raw[ident][j][1]);ages.append(t-times[ident][j])
            q=np.array(pairs)/1000
            sim.data.qpos[sim.indices['right']['qpos'][:6]]=np.deg2rad(q)
            mujoco.mj_forward(sim.model,sim.data)
            c=sim.data.camera('right_wrist');R=c.xmat.reshape(3,3)@np.diag([1.,-1.,-1.])
            return c.xpos.copy(),R,q,t,max(ages)
        first=5  # avoid the very first startup exposure preceding CAN subscription
        initial=cv2.imread(str(directory/f'{first:04d}.jpg'))
        gray=cv2.cvtColor(initial,cv2.COLOR_BGR2GRAY)
        mask=np.zeros_like(gray)
        cv2.fillConvexPoly(mask,np.array([[243,320],[322,290],[339,368],[250,401]]),255)
        features=cv2.goodFeaturesToTrack(gray,100,.02,7,mask=mask,blockSize=5).reshape(-1,2)
        cam0,R0,q0,_,_=pose(first)
        rays=np.column_stack([features,np.ones(len(features))])@np.linalg.inv(k).T@R0.T
        distance=((point-cam0)@normal)/(rays@normal)
        world=cam0+rays*distance[:,None]
        tracked=features.copy();alive=np.ones(len(features),bool)
        indices=sorted(set([first,*range(35,len(samples),30),
                            *range(max(first,len(samples)-120),len(samples),3),len(samples)-1]))
        previous=gray
        for index in indices:
            image=cv2.imread(str(directory/f'{index:04d}.jpg'));current=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            forward,status,_=cv2.calcOpticalFlowPyrLK(previous,current,tracked.astype('float32'),None,
                winSize=(25,25),maxLevel=3)
            backward,back_status,_=cv2.calcOpticalFlowPyrLK(current,previous,forward,None,
                winSize=(25,25),maxLevel=3)
            alive &= status.ravel().astype(bool)&back_status.ravel().astype(bool)&(np.linalg.norm(backward-tracked,axis=1)<1.)
            tracked=forward;previous=current
            cam,R,q,t,age=pose(index)
            xyz=(world-cam)@R
            predicted=xyz@k.T;predicted=predicted[:,:2]/predicted[:,2:]
            error=np.linalg.norm(tracked-predicted,axis=1)
            use=alive&(xyz[:,2]>0)
            if not use.any():raise RuntimeError('All planar features lost')
            records.append(dict(frame=index,exposure_monotonic=t,joints_deg=q.tolist(),feedback_age_s=age,
                tracked_points=int(use.sum()),median_error_px=float(np.median(error[use])),
                p95_error_px=float(np.percentile(error[use],95)),
                observed_median_shift_px=np.median(tracked[use]-features[use],axis=0).tolist(),
                predicted_median_shift_px=np.median(predicted[use]-features[use],axis=0).tolist()))
        overlay=image.copy()
        for observed,expected in zip(tracked[use],predicted[use]):
            cv2.circle(overlay,tuple(np.round(observed).astype(int)),3,(0,255,0),1)
            cv2.drawMarker(overlay,tuple(np.round(expected).astype(int)),(0,0,255),cv2.MARKER_CROSS,6,1)
            cv2.line(overlay,tuple(np.round(observed).astype(int)),tuple(np.round(expected).astype(int)),(255,180,0),1)
        cv2.putText(overlay,'GREEN observed | RED CAN/FK predicted',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        cv2.imwrite(str(a.output/'final_motion_check.png'),overlay)
        (a.output/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        report=dict(source_capture=str(a.capture),task=str(a.task),first_frame=first,last_frame=indices[-1],
            feature_count=len(features),last_frame_result=records[-1],
            max_feedback_age_s=max(r['feedback_age_s'] for r in records),
            observed_joint_ranges_deg=np.ptp([r['joints_deg'] for r in records],axis=0).tolist(),
            status='relative-motion diagnostic only; not absolute calibration',
            limitations=['Feature depth initialized from the provisional holder plane',
                         'Small single-axis motion cannot identify full hand-eye extrinsics'])
        report['maximum_median_error_px']=max(r['median_error_px'] for r in records)
        report['inconsistent_frames_over_5px']=[r['frame'] for r in records if r['median_error_px']>5]
        command_file=a.capture/'commanded_probe.csv'
        if command_file.exists():
            with command_file.open() as f:commands=list(csv.DictReader(f))
            start=float(commands[0]['monotonic']);end=float(commands[-1]['monotonic'])
            covered=[r for r in records if start<=r['exposure_monotonic']<=end]
            report['command_interval_check']=dict(covered_seconds=max(0,records[-1]['exposure_monotonic']-start),
                complete_video_coverage=records[-1]['exposure_monotonic']>=end,
                frames=len(covered),maximum_median_error_px=max(r['median_error_px'] for r in covered),
                joint_ranges_deg=np.ptp([r['joints_deg'] for r in covered],axis=0).tolist())
            moving=[r for r in commands if r['phase'] in ('outbound','return')]
            if moving:
                report['command_interval_check']['complete_movement_video_coverage']=bool(
                    records[0]['exposure_monotonic']<=float(moving[0]['monotonic']) and
                    records[-1]['exposure_monotonic']>=float(moving[-1]['monotonic']))
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    finally:sim.close()


if __name__=='__main__':main()
