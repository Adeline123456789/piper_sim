"""Locate the three insertion slots in a recorded wrist RGB-D view. No CAN access."""
import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import mujoco
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from piper_sim.config import load
from piper_sim.backends.mujoco import MujocoBackend


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--frame', type=int, required=True)
    p.add_argument('--roi', type=int, nargs=4, required=True, help='Five-slot area x0 y0 x1 y1')
    p.add_argument('--task', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    directory = a.capture/'right_wrist'
    meta = json.loads((directory/'camera.json').read_text())
    k = np.array(meta['K'])
    im = cv2.imread(str(directory/f'{a.frame:04d}.jpg'))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    x0, y0, x1, y1 = a.roi
    mask = np.zeros_like(gray)
    mask[y0:y1, x0:x1] = 255
    contours, _ = cv2.findContours(((gray < 80)*mask).astype('uint8'), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 20 < area < 500:
            m = cv2.moments(contour)
            centers.append([m['m10']/m['m00'], m['m01']/m['m00']])
    if len(centers) != 5:
        raise ValueError(f'Expected five visible slots in this ROI, found {len(centers)}')
    centers = np.array(sorted(centers))
    # This recorded view has the angled pair on the left and earth slot at right centre.
    pair = centers[:2][np.argsort(centers[:2, 1])]
    earth = centers[2:][np.argsort(centers[2:, 1])][1]
    pixels = np.vstack([pair, earth])
    with np.load(directory/'depth.npz') as f:
        depth = f['depth_raw'][a.frame].astype(float)*meta['depth_scale_m']
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    surface = (mask > 0) & (gray > 90) & (hsv[:, :, 1] > 40) & (depth > 0)
    v, u = np.where(surface)
    cloud = np.c_[u, v, np.ones(len(u))]@np.linalg.inv(k).T*depth[v, u, None]
    if len(cloud) < 30:
        report = dict(frame=a.frame, slot_centers_px=pixels.tolist(),
            target_center_px=pixels[:2].mean(axis=0).tolist(), depth_plane_points=len(cloud),
            position_3d_available=False, hardware_commands=0,
            status='2D slots located; insufficient depth on socket face for a 3D estimate')
        (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
        return
    for _ in range(3):
        center = np.median(cloud, axis=0)
        _, _, vt = np.linalg.svd(cloud-center, full_matrices=False)
        normal = vt[-1]
        cloud = cloud[abs((cloud-center)@normal) < .002]
    rays = np.c_[pixels, np.ones(3)]@np.linalg.inv(k).T
    points = rays*((normal@center)/(rays@normal))[:, None]
    origin = points[:2].mean(axis=0)
    x = points[1]-points[0]; x /= np.linalg.norm(x)
    y = points[2]-origin; y -= x*(x@y); y /= np.linalg.norm(y)
    z = np.cross(x, y)
    measured = np.eye(4); measured[:3, :3] = np.column_stack([x, y, z]); measured[:3, 3] = origin
    sample = meta['samples'][a.frame]
    timestamp = sample['rgb_device_ms']/1000+sample['host_monotonic']-sample['host_unix']
    rows = list(csv.DictReader((a.capture/'commanded_probe.csv').open()))
    times = np.array([float(r['monotonic']) for r in rows])
    q = [np.interp(timestamp, times, [float(r[f'feedback_j{i}']) for r in rows]) for i in range(1, 7)]
    config = yaml.safe_load(a.task.read_text())
    sim = MujocoBackend(dict(load(), task_scene=config))
    try:
        sim.data.qpos[sim.indices['right']['qpos'][:6]] = np.deg2rad(q)
        mujoco.mj_forward(sim.model, sim.data)
        calibration = json.loads(Path(config['camera_calibration']).read_text())
        mount = np.array(calibration['cameras']['right_wrist']['T_parent_rgb'])
        body = sim.data.body('right/link6')
        camera = np.eye(4); camera[:3, :3] = body.xmat.reshape(3, 3); camera[:3, 3] = body.xpos
        camera = camera@mount
        strip = sim.data.body('task/strip')
        target = strip.xpos+strip.xmat.reshape(3, 3)@np.r_[sim.task.outlets[config['target_index']],sim.task.socket_face_z]
        predicted = (np.linalg.inv(camera)@np.r_[target, 1])[:3]
    finally:
        sim.close()
    projected = k@predicted; projected = projected[:2]/projected[2]
    image_origin = k@origin; image_origin = image_origin[:2]/image_origin[2]
    report = dict(source_capture=str(a.capture), frame=a.frame, roi=a.roi,
        position_3d_available=True,
        slot_centers_px=pixels.tolist(), target_center_px=image_origin.tolist(),
        T_rgb_target_estimate=measured.tolist(), predicted_target_rgb_m=predicted.tolist(),
        predicted_target_px=projected.tolist(), observed_minus_predicted_rgb_m=(origin-predicted).tolist(),
        observed_minus_predicted_world_m=(camera[:3,:3]@(origin-predicted)).tolist(),
        measured_pair_spacing_m=float(np.linalg.norm(points[1]-points[0])),
        measured_earth_offset_m=float(np.linalg.norm(points[2]-origin)),
        depth_plane_rms_m=float(np.sqrt(np.mean(((cloud-center)@normal)**2))), depth_plane_points=len(cloud),
        measured_joint_deg=q, hardware_commands=0,
        status='recorded-view estimate; world correction inherits unvalidated wrist mount, not executable calibration')
    for i, pixel in enumerate(pixels):
        cv2.circle(im, tuple(np.round(pixel).astype(int)), 4, (0,255,0), 1)
        cv2.putText(im, ['pin A','pin B','earth'][i], tuple((pixel+[5,-6]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, .35, (0,255,0), 1)
    for label, pixel, color in [('OBSERVED',image_origin,(0,255,0)),('MODEL',projected,(0,0,255))]:
        cv2.drawMarker(im, tuple(np.round(pixel).astype(int)), color, cv2.MARKER_CROSS, 12, 2)
        cv2.putText(im, label, tuple((pixel+[5,18]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
    cv2.imwrite(str(a.output/'hole_alignment.png'), im)
    (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
