"""Offline six-DOF wrist mount fit from static RGB-D tracks and measured joints.

Outbound segments fit the mount; return segments evaluate it. Uses measured
joint FK, not commanded joints. Writes a candidate only, never controls hardware.
"""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys

import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.config import load


def transform(body):
    t = np.eye(4)
    t[:3, :3] = body.xmat.reshape(3, 3)
    t[:3, 3] = body.xpos
    return t


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mask-polygon', type=float, nargs='+', help='Static reference-image ROI: x y pairs')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    directory = a.capture/'right_wrist'
    meta = json.loads((directory/'camera.json').read_text())
    k = np.array(meta['K'])
    if np.any(meta['distortion_coeffs']):
        raise ValueError('This fitter expects zero-distortion right RGB intrinsics')
    exposure = np.array([s['rgb_device_ms']/1000+s['host_monotonic']-s['host_unix']
                         for s in meta['samples']])
    rows = list(csv.DictReader((a.capture/'commanded_probe.csv').open()))
    times = np.array([float(r['monotonic']) for r in rows])
    q = np.array([[float(r[f'feedback_j{i}']) for i in range(1, 7)] for r in rows])
    first = int(np.searchsorted(exposure, times[0]+.7))
    frames = np.arange(first, np.searchsorted(exposure, times[-1]), 3)
    reference = cv2.imread(str(directory/f'{first:04d}.jpg'), 0)
    # Static scene only: exclude the gripper in the lower left and image edges.
    mask = np.zeros_like(reference)
    mask[12:460, 120:625] = 255
    if a.mask_polygon:
        mask[:] = 0
        cv2.fillPoly(mask, [np.array(a.mask_polygon, np.int32).reshape(-1, 2)], 255)
    points = cv2.goodFeaturesToTrack(reference, 250, .01, 6, mask=mask).reshape(-1, 2)
    with np.load(directory/'depth.npz') as archive:
        depth = archive['depth_raw'][first].astype(float)*meta['depth_scale_m']
    good, depths = [], []
    for i, (u, v) in enumerate(np.round(points).astype(int)):
        patch = depth[v-2:v+3, u-2:u+3]
        values = patch[patch > 0]
        if len(values) >= 20 and np.std(values) < .003:
            good.append(i)
            depths.append(np.median(values))
    points = points[good]
    annotated = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
    for point in points:
        cv2.circle(annotated, tuple(np.round(point).astype(int)), 2, (0, 255, 0), 1)
    cv2.imwrite(str(a.output/'reference_features.png'), annotated)
    xyz = np.c_[points, np.ones(len(points))]@np.linalg.inv(k).T*np.array(depths)[:, None]
    if len(points) < 12:
        raise ValueError('Insufficient static RGB-D features')
    observed, valid = [], []
    previous = reference
    alive = np.ones(len(points), bool)
    for frame in frames:
        current = cv2.imread(str(directory/f'{frame:04d}.jpg'), 0)
        forward, s1, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None,
                                               winSize=(25, 25), maxLevel=3)
        backward, s2, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None,
                                                winSize=(25, 25), maxLevel=3)
        alive &= s1.ravel().astype(bool) & s2.ravel().astype(bool)
        alive &= np.linalg.norm(backward-points, axis=1) < .7
        observed.append(forward.copy())
        valid.append(alive.copy())
        points, previous = forward, current
    observed, valid = np.array(observed), np.array(valid)
    phases = [rows[np.searchsorted(times, exposure[i], side='right')-1]['phase'] for i in frames]
    train = np.array(['outbound' in phase for phase in phases])
    heldout = np.array(['return' in phase for phase in phases])
    sim = MujocoBackend(load())
    poses = []
    try:
        for frame in frames:
            angles = np.array([np.interp(exposure[frame], times, q[:, i]) for i in range(6)])
            # FK diagnostics only: measured positions may be outside command limits.
            sim.data.qpos[sim.indices['right']['qpos'][:6]] = np.deg2rad(angles)
            mujoco.mj_forward(sim.model, sim.data)
            poses.append(transform(sim.data.body('right/link6')))
    finally:
        sim.close()
    relative = np.linalg.inv(np.array(poses))@poses[0]
    calibration = json.loads(a.calibration.read_text())
    old = np.array(calibration['cameras']['right_wrist']['T_parent_rgb'])

    def mount(v):
        x = old.copy()
        x[:3, 3] += v[:3]
        x[:3, :3] = Rotation.from_rotvec(v[3:6]).as_matrix()@old[:3, :3]
        return x

    def predict(v):
        x = mount(v)
        cam = np.linalg.inv(x)@relative@x
        points = np.einsum('fij,nj->fni', cam, np.c_[xyz, np.ones(len(xyz))])[..., :3]
        pixels = points@k.T
        return pixels[..., :2]/pixels[..., 2:]

    def residual(v):
        return (predict(v)-observed)[valid & train[:, None]].ravel()

    scale = np.r_[[.05]*3, [np.deg2rad(20)]*3]
    zero = np.zeros(len(scale))
    fit = least_squares(residual, zero, bounds=(-scale, scale),
                        loss='soft_l1', f_scale=1., x_scale=scale)

    def metrics(v, selection):
        e = np.linalg.norm(predict(v)-observed, axis=2)[valid & selection[:, None]]
        return dict(points=len(e), median_px=float(np.median(e)),
                    p95_px=float(np.percentile(e, 95)), rms_px=float(np.sqrt(np.mean(e**2))))

    singular = np.linalg.svd(fit.jac*scale, compute_uv=False)
    report = dict(source_capture=str(a.capture), features=len(xyz), frames=len(frames), reference_frame=first,
        mask_polygon=a.mask_polygon,
        before={name: metrics(zero, sel) for name, sel in [('train', train), ('heldout', heldout)]},
        candidate={name: metrics(fit.x, sel) for name, sel in [('train', train), ('heldout', heldout)]},
        translation_correction_m=fit.x[:3].tolist(), rotation_correction_deg=np.rad2deg(fit.x[3:6]).tolist(),
        scaled_jacobian_singular_values=singular.tolist(), scaled_condition_number=float(singular[0]/singular[-1]),
        optimization_success=bool(fit.success), parameters_at_bound=bool(np.any(abs(fit.x) > .99*scale)),
        train_phases=sorted(set(np.array(phases)[train])), heldout_phases=sorted(set(np.array(phases)[heldout])),
        status='offline candidate only; no hardware commands or active calibration changes',
        limitations=['return validation shares scene and session with fitting',
                     'world fixture registration and absolute hole alignment require independent validation',
                     'depth, joint FK, timing and optical-flow errors affect the estimate'])
    candidate = deepcopy(calibration)
    candidate['motion_qualified'] = False
    candidate['cameras']['right_wrist']['T_parent_rgb'] = mount(fit.x).tolist()
    candidate['cameras']['right_wrist']['status'] = report['status']
    candidate['multiaxis_fit'] = report
    (a.output/'candidate_cameras.json').write_text(json.dumps(candidate, indent=2)+'\n')
    (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(a.output/'tracks.npz', frames=frames, observed_px=observed, valid=valid,
                        before_px=predict(zero), candidate_px=predict(fit.x),
                        train=train, heldout=heldout, initial_camera_xyz_m=xyz)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
