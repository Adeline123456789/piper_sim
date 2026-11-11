"""Read-only RGB-D and CAN capture for fixture/hand-eye calibration. No CAN writes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import select
import socket
import struct
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs

CAMERAS = {'left_wrist': '243422071995', 'right_wrist': '243722074652',
           'external': '419222301415'}


def capture_camera(name, serial, out, frames, wait_trigger=False):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    metadata = dict(serial=serial, resolution=[640, 480],
                    K=[[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
                    distortion_model=str(intr.model), distortion_coeffs=intr.coeffs,
                    depth_scale_m=profile.get_device().first_depth_sensor().get_depth_scale(),
                    depth_frame='aligned to RGB optical frame', samples=[])
    depth = []
    directory = out / name
    directory.mkdir()
    try:
        for _ in range(30):
            pipe.wait_for_frames(5000)
        (directory/'READY').write_text(str(time.monotonic()))
        while wait_trigger and not (out/'START').exists():
            pipe.wait_for_frames(5000)
            if (out/'STOP').exists():
                raise RuntimeError('Capture stopped before trigger')
        for i in range(frames):
            fs = align.process(pipe.wait_for_frames(5000))
            color, dep = fs.get_color_frame(), fs.get_depth_frame()
            now = time.monotonic()
            unix = time.time()
            rgb = np.asanyarray(color.get_data()).copy()
            depth.append(np.asanyarray(dep.get_data()).copy())
            cv2.imwrite(str(directory / f'{i:04d}.jpg'), rgb)
            metadata['samples'].append(dict(host_monotonic=now, host_unix=unix,
                rgb_frame=color.get_frame_number(), depth_frame=dep.get_frame_number(),
                rgb_device_ms=color.get_timestamp(), depth_device_ms=dep.get_timestamp(),
                timestamp_domain=str(color.get_frame_timestamp_domain())))
            if (out/'STOP').exists():
                break
        np.savez_compressed(directory / 'depth.npz', depth_raw=np.stack(depth))
        cv2.imwrite(str(directory / 'last.png'), rgb)
        (directory / 'camera.json').write_text(json.dumps(metadata, indent=2)+'\n')
        return dict(camera=name, frames=len(depth), first=metadata['samples'][0], last=metadata['samples'][-1])
    finally:
        pipe.stop()


def receive_can(out, done):
    buses = {}
    latest = {name: {} for name in ('can_arm1', 'can_arm2')}
    for name in latest:
        s = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        s.bind((name,))
        buses[s] = name
    try:
        with (out / 'raw_can.jsonl').open('w') as f:
            while not done.is_set():
                for s in select.select(list(buses), [], [], .05)[0]:
                    raw, _, flags, _ = s.recvmsg(16)
                    ident, dlc, data = struct.unpack('=IB3x8s', raw)
                    row = dict(host_monotonic=time.monotonic(), can=buses[s], id=ident,
                               data_hex=data[:dlc].hex(), host_local=bool(flags & socket.MSG_DONTROUTE))
                    f.write(json.dumps(row)+'\n')
                    latest[buses[s]][hex(ident)] = row
        (out / 'can_latest.json').write_text(json.dumps(latest, indent=2)+'\n')
    finally:
        for s in buses:
            s.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frames', type=int, default=45)
    p.add_argument('--wait-trigger', action='store_true', help='Warm cameras, then wait for START file')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    done = threading.Event()
    with ThreadPoolExecutor(max_workers=4) as pool:
        can = pool.submit(receive_can, a.output, done)
        jobs = [pool.submit(capture_camera, n, s, a.output, a.frames, a.wait_trigger) for n, s in CAMERAS.items()]
        try:
            results = [j.result() for j in jobs]
        finally:
            done.set()
            can.result()
    summary = dict(purpose='calibration only, not a policy demonstration', can_writes=0,
                   cameras=results, camera_sync='independent clocks; host receipt timestamps retained')
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
