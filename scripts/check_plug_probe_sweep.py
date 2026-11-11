"""Offline kinematic contact audit of the proposed right wrist calibration.

Uses measured joint angles, including folded poses outside the model's normal
joint limits. This is a geometry diagnostic, not a physics rollout or permission
to execute hardware motion. No CAN modules are imported.
"""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.config import load
from piper_sim.contracts import LOWER, UPPER
from probe_plug_right_wrist import probe_segments


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--task', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    state = json.loads(a.state.read_text())
    sim = MujocoBackend(dict(load(), task_scene=yaml.safe_load(a.task.read_text())))
    entry = np.array([[*state['right']['joint_deg'], .1028]])
    measured = entry.copy()
    entry[0, :6] = np.clip(entry[0, :6], np.rad2deg(LOWER[7:13]), np.rad2deg(UPPER[7:13]))
    # Include the known controller's near-zero folded-entry recovery, rather
    # than auditing wrist motion from an unreachable control target.
    if np.max(abs(entry[0, :6]-measured[0, :6])) > 3:
        raise ValueError('Measured pose is outside the 3 degree folded-entry envelope')
    segments = [('folded_entry', measured, entry, 1.)] + list(probe_segments(entry, True))
    contacts = {}
    samples = 0
    try:
        for side in ('left', 'right'):
            # Actual gripper width exceeds model travel; retain this limitation.
            sim.data.qpos[sim.indices[side]['qpos']] = [
                *np.deg2rad(state[side]['joint_deg']), .0475, .0475]
        for phase, start, end, duration in segments:
            for alpha in np.linspace(0, 1, round(duration*100)+1):
                sim.data.qpos[sim.indices['right']['qpos'][:6]] = np.deg2rad(
                    (start+(end-start)*alpha)[0, :6])
                mujoco.mj_forward(sim.model, sim.data)
                samples += 1
                for c in sim.data.contact[:sim.data.ncon]:
                    if c.dist >= 0:
                        continue
                    bodies = [sim.model.body(int(sim.model.geom_bodyid[g])).name
                              for g in (c.geom1, c.geom2)]
                    names = [sim.model.geom(int(g)).name or f'geom_{g}'
                             for g in (c.geom1, c.geom2)]
                    key = '|'.join(sorted(names))
                    if key not in contacts or -c.dist > contacts[key]['max_penetration_m']:
                        contacts[key] = dict(geoms=names, bodies=bodies,
                            involves_right=any(b.startswith('right/') for b in bodies),
                            max_penetration_m=float(-c.dist), phase=phase,
                            fraction=float(alpha))
        report = dict(source_state=str(a.state), task=str(a.task), samples=samples,
            hardware_commands=0, kinematic_only=True,
            planned_duration_s=25., speed_deg_s=2.,
            entry_joint_deg=entry[0, :6].tolist(),
            folded_entry_adjustment_deg=(entry-measured)[0, :6].tolist(),
            contacts=sorted(contacts.values(), key=lambda c:-c['max_penetration_m']),
            limitations=['provisional robot-base and fixture registration',
                         'gripper geometry uses 95 mm opening; physical width not calibrated',
                         'existing folded-pose overlaps are reported, not filtered',
                         'no physical clearance guarantee from this model'])
        a.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
    finally:
        sim.close()


if __name__ == '__main__':
    main()
