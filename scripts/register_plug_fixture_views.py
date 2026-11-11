"""Fit the measured holder rectangle and RGB-D planes in two camera views.

World registration retains the table-fit CAD XY/yaw anchor. This is a provisional
visual alignment, not an independently calibrated robot/world transformation.
"""
import argparse
import json
from pathlib import Path
import sys
import warnings
import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def intrinsics(meta):
    k=np.asarray(meta['K']);i=rs.intrinsics();i.width=640;i.height=480
    i.fx=k[0,0];i.fy=k[1,1];i.ppx=k[0,2];i.ppy=k[1,2]
    i.model=getattr(rs.distortion,meta['distortion_model'].split('.')[-1]);i.coeffs=meta['distortion_coeffs']
    return i,k


def rays_at(intr,pixels):
    return np.array([rs.rs2_deproject_pixel_to_point(intr,list(map(float,p)),1.) for p in pixels])


def fit_camera(capture,name,corners,dimensions=(.17,.06,.033),occluded=(1,),label='holder'):
    directory=capture/name;meta=json.loads((directory/'camera.json').read_text());intr,k=intrinsics(meta)
    dep=np.load(directory/'depth.npz')['depth_raw'].astype(float);dep[dep==0]=np.nan
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='All-NaN slice encountered')
        depth=np.nanmedian(dep,axis=0)*meta['depth_scale_m']
    yy,xx=np.mgrid[:480,:640]
    # A long, uncluttered table patch gives a better normal than the tiny
    # holder patch, which is partly occluded by the yellow/black housings.
    mask=((xx>55)&(xx<590)&(yy>160)&(yy<230)) if name=='external' else (
         (xx>100)&(xx<340)&(yy>120)&(yy<260))
    y,x=np.where(mask&np.isfinite(depth));pixels=np.column_stack([x,y])
    points=rays_at(intr,pixels)*depth[y,x,None]
    rng=np.random.default_rng(23)
    cloud=points[rng.choice(len(points),min(6000,len(points)),replace=False)]
    best=np.zeros(len(cloud),bool)
    for _ in range(220):
        a,b,c=cloud[rng.choice(len(cloud),3,replace=False)]
        n=np.cross(b-a,c-a);n/=np.linalg.norm(n)
        inside=abs((cloud-a)@n)<.0025
        if inside.sum()>best.sum():best=inside
    pts=cloud[best];center=np.mean(pts,axis=0);_,_,vt=np.linalg.svd(pts-center,full_matrices=False)
    normal=vt[-1]
    if normal@center>0:normal=-normal
    distance=-normal@center
    # A: long-axis left/back; B: right/back; C: right/front; D: left/front.
    # These are the top-face edges, not the lower shadow or side-wall edges.
    length,width,height=dimensions
    xyz=np.array([[-length/2,width/2,height],[length/2,width/2,height],
                  [length/2,-width/2,height],[-length/2,-width/2,height]])
    normalized=rays_at(intr,corners)[:,:2].copy()
    _,rvec,tvec=cv2.solvePnP(xyz,normalized,np.eye(3),None,flags=cv2.SOLVEPNP_IPPE)
    e1=np.array([1.,0,0]);e1-=normal*(e1@normal);e1/=np.linalg.norm(e1)
    e2=np.cross(normal,e1);basis=np.column_stack([e1,e2,normal]);origin=-distance*normal
    initial_r=Rotation.from_rotvec(rvec.ravel()).as_matrix()
    initial_yaw=np.arctan2(e2@initial_r[:,0],e1@initial_r[:,0])
    def transform_at(v):
        r=basis@Rotation.from_rotvec([0,0,v[0]]).as_matrix()
        t=origin+e1*v[1]+e2*v[2]
        return r,t
    # The measured solid holder rests flat on the table. Solve its in-plane
    # pose rather than allowing occlusion/depth noise to tilt it off the table.
    weights=np.ones((4,1));weights[list(occluded)]=0
    def error(v):
        r,t=transform_at(v);cam=xyz@r.T+t
        projection=cam[:,:2]/cam[:,2:]
        pixel=(projection-normalized)*[k[0,0],k[1,1]]
        return (pixel*weights).ravel()/1.5
    fit=least_squares(error,[initial_yaw,e1@(tvec.ravel()-origin),e2@(tvec.ravel()-origin)],max_nfev=150)
    r,t=transform_at(fit.x)
    transform=np.eye(4);transform[:3,:3]=r;transform[:3,3]=t
    projected=xyz@r.T+t;projected=projected[:,:2]/projected[:,2:]
    residual=(projected-normalized)*[k[0,0],k[1,1]]
    # Store undistorted pixel coordinates so the residual convention is explicit.
    report=dict(T_rgb_holder=transform.tolist(),K=k.tolist(),raw_corners_px=corners.tolist(),
        undistorted_corners_px=(normalized*[k[0,0],k[1,1]]+[k[0,2],k[1,2]]).tolist(),
        fit_corners_undistorted_px=(projected*[k[0,0],k[1,1]]+[k[0,2],k[1,2]]).tolist(),
        reprojection_rms_px=float(np.sqrt(np.mean(residual**2))),
        visible_corner_reprojection_rms_px=float(np.sqrt(np.mean(residual[weights.ravel()>0]**2))),
        occluded_corner_indices=list(occluded),dimensions_m=list(dimensions),
        corner_weights=weights.ravel().tolist(),pose_model='known-size holder resting flat on measured table plane',
        plane_normal_rgb=normal.tolist(),table_plane_distance_m=float(distance),plane_points=len(pts),
        plane_rms_m=float(np.sqrt(np.mean(((pts-center)@normal)**2))),
        fitted_bottom_plane_residual_m=float(normal@t+distance))
    image=cv2.imread(str(directory/'last.png'))
    cv2.polylines(image,[np.round(corners).astype(np.int32)],True,(0,255,255),1)
    for i,(u,v) in enumerate(corners):cv2.putText(image,'ABCD'[i],(int(u)+3,int(v)-4),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1)
    cv2.imwrite(str(capture/(name+'_'+label+'_annotation.png')),image)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',type=Path,required=True)
    a=p.parse_args()
    # Manually inspected corners in the stored 640x480 RGB frames. Approximate
    # annotation uncertainty is 1.5 pixels; corners behind the black body are
    # extrapolated from visible straight edges and should be rechecked.
    corners={'external':np.array([[455,257],[541,249],[551,272],[461,280]],float),
             'right_wrist':np.array([[231,318],[563,216],[620,270],[242,416]],float)}
    views={name:fit_camera(a.capture,name,px) for name,px in corners.items()}
    table=json.loads((a.capture/'table_registration.json').read_text())
    world_external=np.array(table['T_world_rgb_estimate'])
    world_holder=world_external@np.array(views['external']['T_rgb_holder'])
    world_right=world_holder@np.linalg.inv(np.array(views['right_wrist']['T_rgb_holder']))
    output=dict(status='provisional visual alignment; not motion-qualified',
        source_capture=str(a.capture),holder_dimensions_m=[.17,.06,.033],views=views,
        T_world_holder_estimate=world_holder.tolist(),T_world_right_rgb_estimate=world_right.tolist(),
        limitations=['world XY/yaw inherited from CAD camera anchor',
                     'corner annotations include an occluded-corner extrapolation',
                     'single-pose camera registration is not a multi-pose hand-eye calibration'])
    (a.capture/'fixture_view_registration.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='views'},indent=2))
    for name,r in views.items():print(name,'reprojection_rms_px',r['reprojection_rms_px'],'depth_rms_mm',r['plane_rms_m']*1000)


if __name__=='__main__':main()
