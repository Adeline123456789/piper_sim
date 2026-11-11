"""Bounded, non-contact right-wrist calibration using the working home_zero lifecycle.

Run the independent RGB-D/CAN recorder first. This script records commanded
targets and feedback, returns each tested joint to entry, then waits for supported
shutdown. It never controls the gripper or left arm.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np


def probe_segments(entry, multi_axis=False):
    """Joint-degree targets; each axis returns to entry before the next one."""
    for joint, delta in ([(5, 8.), (4, -5.), (3, 5.)] if multi_axis else [(5, 8.)]):
        goal = entry.copy()
        goal[0, joint] += delta
        prefix = f'j{joint+1}_' if multi_axis else ''
        for phase, start, end in [('outbound', entry, goal), ('return', goal, entry)]:
            yield prefix+phase, start.copy(), end.copy(), abs(delta)/2.


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--multi-axis',action='store_true',help='J6 +8°, J5 -5°, J4 +5°; each returns, at 2°/s')
    p.add_argument('--approach-plan',type=Path,help='Reviewed task approach JSON; moves to its joint goal and holds, no gripper actuation')
    a=p.parse_args()
    if not a.execute:raise SystemExit('Use --execute after the scene and calibration path are checked')
    plan=json.loads(a.approach_plan.read_text()) if a.approach_plan else None
    label=('取插任务起步：展开至黄色插头前上方并保持' if plan else
           '当前姿态；J6 +8°、J5 −5°、J4 +5°，逐轴往返' if a.multi_axis else
           '当前姿态，随后右腕 J6 +8° 并返回')
    root=Path('/media/midea/F496AC8096AC4548/act_td3/piper_use')
    for path in (root,root/'example_agilex',root/'example_agilex/piper_sdk'):sys.path.insert(0,str(path))
    from replay_connector import Hardware,parse_args
    args=parse_args([]);args.folded_entry=True;args.control_hz=100;args.max_joint_speed=3.
    hardware=Hardware(args,['right'])
    f=(a.capture/'commanded_probe.csv').open('w',newline='')
    writer=csv.writer(f);writer.writerow(['monotonic','phase']+[f'{kind}_j{i}' for kind in ('target','feedback') for i in range(1,7)])
    def send(q,phase):
        tick=time.monotonic();hardware.check();hardware.send(q)
        measured=hardware.feedback('right',True)[0]
        writer.writerow([tick,phase,*q[0,:6],*measured[:6]]);f.flush()
        time.sleep(max(0,.01-(time.monotonic()-tick)))
    try:
        for name in ('left_wrist','right_wrist','external'):
            if not (a.capture/name/'READY').exists():
                raise RuntimeError('Camera not warmed: '+name)
        episode=SimpleNamespace(episode_id='RGBD_WRIST_CALIBRATION',actions={'right':np.zeros((1,7))})
        expected=np.zeros((1,7))
        if plan:expected[0,:6]=plan['goal_joint_deg']
        entry=hardware.connect(episode,expected,activate_gripper=False,
            confirmation='TASK right' if plan else 'PROBE right',target_label=label,
            before_enable=lambda _: (a.capture/'START').write_text(str(time.monotonic())))
        segments=list(probe_segments(entry,a.multi_axis))
        if plan:
            goal=entry.copy();goal[0,:6]=plan['goal_joint_deg']
            segments=[('approach',entry,goal,max(1.,float(np.max(abs(goal-entry)))/3.))]
        limits=hardware.controllers['right'].JOINT_LIMITS
        for _,_,end,_ in segments:
            if np.any(end[0,:6]<limits[:,0]) or np.any(end[0,:6]>limits[:,1]):
                raise ValueError('Probe exceeds hardware joint limits')
        for _ in range(100):send(entry,'entry_hold')
        for phase,start,end,duration in segments:
            for alpha in np.linspace(0,1,round(duration*100)+1)[1:]:send(start+(end-start)*alpha,phase)
            for _ in range(100):send(end,phase+'_hold')
            print(phase,'completed',hardware.feedback('right',True)[0][:6].tolist(),flush=True)
        print(('APPROACH_COMPLETE: holding task approach pose' if plan else
               'PROBE_COMPLETE: returned to entry')+'; holding enable for supported shutdown',flush=True)
        # Cameras deliver frames after exposure. Keep recording a short tail so
        # the final command is covered even with USB/SDK receipt latency.
        time.sleep(.15)
    finally:
        f.close()
        (a.capture/'STOP').write_text('probe complete or stopped\n')
        hardware.shutdown()


if __name__=='__main__':main()
