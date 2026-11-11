"""Refine the four observable wrist-camera parameters from a J6 round trip.

Fit only outbound frames and evaluate return frames. RGB-D supplies initial
feature depths; no assumed world/holder pose is used. Translation along the
joint axis and the commuting rotation gauge remain fixed to the old mount.
This is a candidate relative-motion calibration, not a full hand-eye solution.
"""
import argparse
import bisect
import csv
from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    directory=a.capture/'right_wrist'
    meta=json.loads((directory/'camera.json').read_text());k=np.asarray(meta['K'])
    if np.any(meta['distortion_coeffs']):raise ValueError('Expected zero-distortion right RGB intrinsics')
    samples=meta['samples']
    exposure=np.array([s['rgb_device_ms']/1000+s['host_monotonic']-s['host_unix'] for s in samples])
    with (a.capture/'commanded_probe.csv').open() as f:commands=list(csv.DictReader(f))
    times=np.array([float(c['monotonic']) for c in commands])
    joints=np.array([[float(c[f'feedback_j{i}']) for i in range(1,7)] for c in commands])
    phases=[c['phase'] for c in commands]
    first=int(np.searchsorted(exposure,times[0]+.7))
    reference=cv2.imread(str(directory/f'{first:04d}.jpg'))
    gray=cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY);mask=np.zeros_like(gray)
    cv2.fillConvexPoly(mask,np.array([[242,319],[321,288],[341,359],[247,390]]),255)
    mask[319:355,270:307]=0  # exclude the aperture rim's mixed-depth pixels
    points=cv2.goodFeaturesToTrack(gray,100,.01,5,mask=mask,blockSize=5).reshape(-1,2)
    with np.load(directory/'depth.npz') as archive:
        depth=archive['depth_raw'][first].astype(float)*meta['depth_scale_m']
    keep=[];depths=[]
    for index,(u,v) in enumerate(np.round(points).astype(int)):
        patch=depth[v-2:v+3,u-2:u+3];valid=patch[patch>0]
        if len(valid)>=20 and np.std(valid)<.003:
            keep.append(index);depths.append(float(np.median(valid)))
    points=points[keep];points0=points.copy()
    if len(points)<12:raise RuntimeError('Insufficient reliable RGB-D features on exposed holder')
    xyz0=np.column_stack([points,np.ones(len(points))])@np.linalg.inv(k).T*np.array(depths)[:,None]
    calibration=json.loads(a.calibration.read_text());old=np.array(calibration['cameras']['right_wrist']['T_parent_rgb'])
    q0=np.interp(exposure[first],times,joints[:,5])
    observed=[];angles=[];frame_ids=[];phase_ids=[];valid_masks=[]
    alive=np.ones(len(points),bool);previous=gray
    for index in sorted(set([*range(first,len(samples),3),len(samples)-1])):
        im=cv2.imread(str(directory/f'{index:04d}.jpg'));current=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        forward,status,_=cv2.calcOpticalFlowPyrLK(previous,current,points.astype('float32'),None,winSize=(25,25),maxLevel=3)
        backward,backstatus,_=cv2.calcOpticalFlowPyrLK(current,previous,forward,None,winSize=(25,25),maxLevel=3)
        alive &= status.ravel().astype(bool)&backstatus.ravel().astype(bool)&(np.linalg.norm(backward-points,axis=1)<.7)
        observed.append(forward.copy());valid_masks.append(alive.copy());points=forward;previous=current
        angles.append(np.deg2rad(np.interp(exposure[index],times,joints[:,5])-q0))
        frame_ids.append(index);phase_ids.append(phases[max(0,bisect.bisect_right(times,exposure[index])-1)])
    observed=np.asarray(observed);valid=np.asarray(valid_masks)
    rotations=Rotation.from_euler('z',-np.array(angles)).as_matrix()
    train=np.isin(phase_ids,['outbound','outbound_hold']);heldout=np.isin(phase_ids,['return','return_hold'])
    def mount(parameters):
        x=old.copy();x[:2,3]+=parameters[:2]
        x[:3,:3]=Rotation.from_rotvec([*parameters[2:],0.]).as_matrix()@old[:3,:3]
        return x
    def predict(parameters):
        x=mount(parameters);r=x[:3,:3];t=x[:3,3]
        p0=xyz0@r.T+t
        current=np.einsum('fij,nj->fni',rotations,p0)
        camera=(current-t)@r
        pixels=camera@k.T
        return pixels[:,:,:2]/pixels[:,:,2:]
    def residual(parameters):
        delta=predict(parameters)-observed
        return delta[valid&train[:,None]].ravel()
    bounds=np.array([.025,.025,np.deg2rad(12),np.deg2rad(12)])
    fit=least_squares(residual,np.zeros(4),bounds=(-bounds,bounds),loss='soft_l1',f_scale=1.,x_scale=bounds)
    def metrics(parameters,select):
        e=np.linalg.norm(predict(parameters)-observed,axis=2)[valid&select[:,None]]
        return dict(median_px=float(np.median(e)),p95_px=float(np.percentile(e,95)),rms_px=float(np.sqrt(np.mean(e*e))),points=len(e))
    report=dict(source_capture=str(a.capture),reference_frame=first,features=len(points0),
        training_phases=['outbound','outbound_hold'],heldout_phases=['return','return_hold'],
        before=dict(train=metrics(np.zeros(4),train),heldout=metrics(np.zeros(4),heldout)),
        candidate=dict(train=metrics(fit.x,train),heldout=metrics(fit.x,heldout)),
        correction_parent_xy_m=fit.x[:2].tolist(),correction_parent_tilt_deg=np.rad2deg(fit.x[2:]).tolist(),
        optimization_success=bool(fit.success),parameters_at_bound=bool(np.any(abs(fit.x)>bounds*.99)),
        stationary_other_joint_ranges_deg=np.ptp(joints[times>=exposure[first],:5],axis=0).tolist(),
        status='candidate J6-axis calibration only; full hand-eye and world registration remain unverified',
        depth_source='aligned RealSense depth at initial tracked feature locations',
        limitations=['single-axis motion leaves two mount degrees of freedom unobservable',
                     'return-phase validation shares objects and session with fitting data',
                     'camera timing, depth error and bracket flex may affect the estimate'])
    report['heldout_improved']=report['candidate']['heldout']['p95_px']<report['before']['heldout']['p95_px']
    candidate=deepcopy(calibration);candidate['motion_qualified']=False
    candidate['cameras']['right_wrist']['T_parent_rgb']=mount(fit.x).tolist()
    candidate['cameras']['right_wrist']['status']='candidate J6-axis fit; two mount DOFs and world pose remain unverified'
    candidate['relative_motion_fit']=report
    (a.output/'candidate_cameras.json').write_text(json.dumps(candidate,indent=2)+'\n')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(a.output/'tracks.npz',frames=frame_ids,observed_px=observed,valid=valid,
        before_px=predict(np.zeros(4)),candidate_px=predict(fit.x),initial_camera_xyz_m=xyz0,train=train,heldout=heldout)
    peak=int(np.argmax(abs(np.array(angles))))
    frame=cv2.imread(str(directory/f'{frame_ids[peak]:04d}.jpg'))
    for obs,pre,post,ok in zip(observed[peak],predict(np.zeros(4))[peak],predict(fit.x)[peak],valid[peak]):
        if ok:
            cv2.circle(frame,tuple(np.round(obs).astype(int)),3,(0,255,0),1)
            cv2.drawMarker(frame,tuple(np.round(pre).astype(int)),(0,0,255),cv2.MARKER_CROSS,6,1)
            cv2.drawMarker(frame,tuple(np.round(post).astype(int)),(255,255,0),cv2.MARKER_CROSS,6,1)
    cv2.putText(frame,'GREEN measured | RED before | CYAN candidate',(7,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
    cv2.imwrite(str(a.output/'peak_comparison.png'),frame)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
