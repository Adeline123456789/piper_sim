"""Read live Piper feedback and compare it to the SDK pose mirrored in MuJoCo. No actuation."""
import argparse
import csv
import socket
import struct
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.config import load
from piper_sim.contracts import LOWER, UPPER
from piper_sim.piper_connection import PiperConnection


def raw_snapshot(bus):
    # Drain old frames, then collect one fresh set of feedback IDs.
    bus.setblocking(False)
    for _ in range(4096):
        try:
            bus.recv(16)
        except BlockingIOError:
            break
    else:
        raise RuntimeError("Raw feedback backlog exceeded")
    bus.settimeout(.25)
    frames = {}
    required = {0x2A1, 0x2A5, 0x2A6, 0x2A7, 0x2A8, *range(0x261, 0x267)}
    deadline = time.monotonic() + .25
    while frames.keys() != required:
        if time.monotonic() >= deadline:
            raise TimeoutError("Incomplete raw Piper feedback")
        identifier, length, payload = struct.unpack("=IB3x8s", bus.recv(16))
        if identifier in required and length == 8:
            frames[identifier] = payload
    joints = [angle/1000 for i in (0x2A5,0x2A6,0x2A7)
              for angle in struct.unpack(">ii", frames[i])]
    width = int.from_bytes(frames[0x2A8][:4], "big", signed=True)/1000
    enabled = sum(bool(frames[i][5] & 0x40) for i in range(0x261,0x267))
    return joints+[width], frames[0x2A1][1], enabled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-path", required=True)
    parser.add_argument("--can-left", default="can_arm1")
    parser.add_argument("--can-right", default="can_arm2")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cfg = load()
    cfg["hardware"].update(sdk_path=args.sdk_path, can_left=args.can_left, can_right=args.can_right)
    with ExitStack() as stack:
        buses = []
        for port in (args.can_left,args.can_right):
            bus = stack.enter_context(socket.socket(socket.AF_CAN,socket.SOCK_RAW,socket.CAN_RAW))
            bus.bind((port,))
            bus.settimeout(.25)
            buses.append(bus)
        sim = MujocoBackend(cfg)
        stack.callback(sim.close)
        robot = stack.enter_context(PiperConnection(cfg["hardware"], allow_actuation=False))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = csv.writer(stack.enter_context(args.output.open("w", newline="")))
        writer.writerow(["sample","side","channel","unit","status","enabled_motors","raw","sim_limit_projection","sim_minus_raw","sdk_unclipped","sdk_minus_raw"])
        for sample in range(30):
            raw = [raw_snapshot(bus) for bus in buses]
            feedback = robot.read().position
            # Display the nearest supported model pose; never send this clipped pose to hardware.
            sim.set_observation_state(np.clip(feedback, LOWER, UPPER))
            for side, q, physical, (values,status,enabled) in zip(("left","right"),sim.read().position.reshape(2,7),feedback.reshape(2,7),raw):
                mirrored = np.r_[np.rad2deg(q[:6]), q[6]*1000]
                physical_units = np.r_[np.rad2deg(physical[:6]), physical[6]*1000]
                for i,(actual,virtual) in enumerate(zip(values,mirrored)):
                    writer.writerow([sample,side,f"J{i+1}" if i<6 else "gripper","deg" if i<6 else "mm",
                                     status,enabled,actual,virtual,virtual-actual,physical_units[i],physical_units[i]-actual])
            time.sleep(.02)


if __name__ == "__main__":
    main()
