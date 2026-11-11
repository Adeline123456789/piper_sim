"""Estimate external camera/table/fixture geometry from the captured RGB-D.

This is a provisional registration: camera horizontal translation and yaw are
anchored to the CAD mast, not a measured hand-eye calibration. Never real GT.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import pyrealsense2 as rs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    a = p.parse_args()
    path = a.capture/'external'
    meta = json.loads((path/'camera.json').read_text())
    dep = np.load(path/'depth.npz')['depth_raw'].astype(float)
    dep[dep == 0] = np.nan
    depth = np.nanmedian(dep, axis=0)*meta['depth_scale_m']
    k = np.asarray(meta['K'])
    v, u = np.mgrid[:480, :640]
    # Respect RealSense's INVERSE Brown-Conrady convention (not OpenCV's
    # forward distortion coefficients). The SDK handles the declared model.
    intr=rs.intrinsics();intr.width=640;intr.height=480
    intr.fx=k[0,0];intr.fy=k[1,1];intr.ppx=k[0,2];intr.ppy=k[1,2]
    intr.model=getattr(rs.distortion,meta['distortion_model'].split('.')[-1])
    intr.coeffs=meta['distortion_coeffs']
    rays=np.array([rs.rs2_deproject_pixel_to_point(intr,[float(px),float(py)],1.)
                   for py in range(480) for px in range(640)]).reshape(480,640,3)
    points=rays*depth[...,None]
    mask = ((v>160)&(v<230)&(u>55)&(u<590)) | ((v>305)&(v<355)&(u>190)&(u<430))
    cloud = points[mask & np.isfinite(depth)]
    rng = np.random.default_rng(7)
    best = np.zeros(len(cloud), bool)
    for _ in range(200):
        sample = cloud[rng.choice(len(cloud),3,replace=False)]
        n = np.cross(sample[1]-sample[0],sample[2]-sample[0])
        n /= np.linalg.norm(n)
        keep = abs((cloud-sample[0])@n)<.003
        if keep.sum()>best.sum(): best=keep
    inliers=cloud[best]; center=np.mean(inliers,axis=0)
    _,_,vt=np.linalg.svd(inliers-center,full_matrices=False)
    normal=vt[-1]
    if normal[2]>0: normal=-normal
    distance=-normal@center
    y=-np.array([1.,0,0]); y-=normal*(y@normal); y/=np.linalg.norm(y)
    x=np.cross(y,normal)
    r=np.stack([x,y,normal])
    t=np.array([.016575,-.0115,.025+distance])
    transform=np.eye(4);transform[:3,:3]=r;transform[:3,3]=t
    world=points@r.T+t
    # Pixel centres selected from the external RGB. Use a 5-pixel neighbourhood
    # on each unobstructed fixture top, not the yellow/black housings.
    samples={ 'strip_top':(310,250), 'holder_top':(474,272) }
    measured={}
    for name,(px,py) in samples.items():
        xyz=world[py-2:py+3,px-2:px+3].reshape(-1,3)
        xyz=xyz[np.isfinite(xyz).all(axis=1)]
        measured[name]=dict(pixel=[px,py],position_world_estimate_m=np.median(xyz,axis=0).tolist(),
                            valid_depth_pixels=len(xyz))
    report=dict(status='provisional, not hardware motion calibration',
        source_capture=str(a.capture),T_world_rgb_estimate=transform.tolist(),
        table_plane_normal_rgb=normal.tolist(),table_plane_distance_m=float(distance),
        table_inliers=int(best.sum()),table_rms_m=float(np.sqrt(np.mean(((inliers-center)@normal)**2))),
        assumptions=['table surface world z=0.025 m', 'camera XY and yaw anchored to CAD mast',
                     'fixture samples are surface points, not fixture origin or insertion targets'],
        samples=measured)
    (a.capture/'table_registration.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(a.capture/'external_pointcloud.npz',world_estimate_m=world,depth_m=depth)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
