"""Export static or wrist-probe calibration into reference HDF5/MP4 layout.

With --commands, retain recorded requested right-joint targets during the common
video interval. Unknown left/gripper targets remain NaN. Calibration episodes
are excluded from full-task policy training.
"""
import argparse
import bisect
import csv
import json
from pathlib import Path
import struct
import sys
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from piper_sim.mixed_recording import MixedEpisodeWriter
from piper_sim.task_archive import archive_episode


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--session',default='20261108_real_calibration')
    p.add_argument('--episode',default='000001')
    p.add_argument('--commands',type=Path,help='commanded_probe.csv; right joint targets only')
    p.add_argument('--purpose',choices=['wrist_alignment_probe','task_approach','yellow_pick_insert_attempt'],help='Purpose of a commanded real episode')
    p.add_argument('--execution-report',type=Path,help='Observed task outcome; preserved independently from command completion')
    a=p.parse_args()
    cameras={n:json.loads((a.capture/n/'camera.json').read_text()) for n in ('external','left_wrist','right_wrist')}
    times={n:np.array([r['rgb_device_ms']/1000 for r in m['samples']]) for n,m in cameras.items()}
    if any(r['timestamp_domain']!='timestamp_domain.global_time' for m in cameras.values() for r in m['samples']):
        raise ValueError('This export requires RealSense global_time timestamps')
    raw={}
    for line in (a.capture/'raw_can.jsonl').open():
        row=json.loads(line);raw.setdefault((row['can'],row['id']),[]).append(row)
    raw_times={k:[r['host_monotonic'] for r in v] for k,v in raw.items()}
    def feedback(channel,ident,t):
        key=(channel,ident);i=bisect.bisect_right(raw_times[key],t)-1
        if i<0 or t-raw_times[key][i]>.1: raise ValueError('Missing fresh feedback')
        return bytes.fromhex(raw[key][i]['data_hex']),t-raw_times[key][i]
    commands=[]
    if a.commands:
        with a.commands.open() as f:commands=list(csv.DictReader(f))
    command_times=[float(r['monotonic']) for r in commands]
    grip_known=bool(commands and 'target_gripper_open' in commands[0])
    outcome=json.loads(a.execution_report.read_text()) if a.execution_report else None
    writer=MixedEpisodeWriter(a.root,a.session,a.episode,'real',
        (a.purpose or 'wrist_alignment_probe') if commands else 'static_rgbd_calibration',
        metadata=dict(source_capture=str(a.capture),camera_calibration={k:{x:y for x,y in v.items() if x!='samples'} for k,v in cameras.items()},
                      ee_pose_source='controller CAN pose, not externally measured',
                      ee_pose_frame='each arm base',force_sensor_present=False,
                      timestamp_monotonic_clock='RealSense RGB global exposure mapped to host monotonic clock',
                      action_source=('requested right joint targets before asynchronous CAN sender; not transmit timestamps'
                                     if commands else 'unavailable'),
                      source_command_log=str(a.commands) if commands else '',
                      execution_outcome=outcome,
                      missing_action_fields=('left arm' if grip_known else 'left arm and both grippers') if commands else 'all'))
    start=max(t[0] for t in times.values());end=min(t[-1] for t in times.values())
    timeline=times['external'][(times['external']>=start)&(times['external']<=end)]
    skews=[]
    for timestamp in timeline:
        images={};row={'timestamp':timestamp,'valid/action':False,'valid/state':True}
        ids={n:int(np.argmin(abs(t-timestamp))) for n,t in times.items()}
        ext=cameras['external']['samples'][ids['external']]
        sample_time=timestamp+(ext['host_monotonic']-ext['host_unix'])
        if commands and not command_times[0]<=sample_time<=command_times[-1]:
            continue
        row['timestamp_monotonic']=sample_time
        for n,i in ids.items():
            images['cam_'+n]=cv2.cvtColor(cv2.imread(str(a.capture/n/f'{i:04d}.jpg')),cv2.COLOR_BGR2RGB)
            row[f'camera/cam_{n}/timestamp']=times[n][i]
            row[f'camera/cam_{n}/source_frame']=i
        skew=max(times[n][i] for n,i in ids.items())-min(times[n][i] for n,i in ids.items())
        row['camera/max_skew_s']=skew;skews.append(skew)
        for arm,channel in [('left','can_arm1'),('right','can_arm2')]:
            ages=[]
            def pairs(ident):
                b,age=feedback(channel,ident,sample_time);ages.append(age)
                return np.array(struct.unpack('>ii',b))/1000
            joints=np.concatenate([pairs(i) for i in (0x2a5,0x2a6,0x2a7)])
            ee=np.concatenate([pairs(i) for i in (0x2a2,0x2a3,0x2a4)])
            g,age=feedback(channel,0x2a8,sample_time);ages.append(age)
            width=struct.unpack('>i',g[:4])[0]*1e-6
            prefix=f'state/{arm}_arm/'
            row[prefix+'joint_positions']=joints
            row[prefix+'end_effector_pos']=ee[:3]
            row[prefix+'end_effector_quat']=Rotation.from_euler('xyz',ee[3:],degrees=True).as_quat()
            row[prefix+'gripper_open']=width/.095
            row[prefix+'gripper_width_m']=width
            row[prefix+'feedback_age_s']=max(ages)
            row[f'action/{arm}_arm/joint_positions']=np.full(6,np.nan)
            row[f'action/{arm}_arm/gripper_open']=np.nan
            row[f'valid/{arm}_arm_joint_action']=False
            row[f'valid/{arm}_gripper_action']=False
        if commands:
            command=commands[bisect.bisect_right(command_times,sample_time)-1]
            row['action/right_arm/joint_positions']=np.array([float(command[f'target_j{i}']) for i in range(1,7)])
            row['valid/right_arm_joint_action']=True
            if grip_known:
                row['action/right_arm/gripper_open']=float(command['target_gripper_open'])
                row['valid/right_gripper_action']=True
            row['action/right_arm/request_age_s']=sample_time-float(command['monotonic'])
        writer.append(row,images)
    movement=[r for r in commands if r['phase'].split('_')[-1] in
              ('outbound','return','approach','descend','grasp','lift','transport','align','insert','release','retreat','fold')]
    movement_covered=bool(movement and writer.rows and
        float(writer.rows[0]['timestamp_monotonic'])<=float(movement[0]['monotonic']) and
        float(writer.rows[-1]['timestamp_monotonic'])>=float(movement[-1]['monotonic']))
    task_success=bool(a.purpose=='yellow_pick_insert_attempt' and outcome and
                      outcome.get('physical_task_success') is True)
    print(json.dumps(writer.close(success=task_success,policy_training_eligible=False,
        reason=('observed real task outcome retained; missing action fields remain invalid'
                if a.purpose=='yellow_pick_insert_attempt' and outcome and outcome.get('physical_task_success') is not None else
                'partial task approach, no grasp or insertion; unknown action fields remain invalid'
                if a.purpose=='task_approach' else 'calibration only; unknown action fields remain invalid'),
        max_camera_skew_s=max(skews),
        complete_movement_video_coverage=movement_covered,
        complete_command_video_coverage=bool(commands and writer.rows and
            float(writer.rows[0]['timestamp_monotonic'])<=command_times[0]+1/30 and
            float(writer.rows[-1]['timestamp_monotonic'])>=command_times[-1]-1/30)),indent=2))
    if outcome and outcome.get('physical_task_success') is not None:
        print('ARCHIVED',archive_episode(a.root,a.session,a.episode))


if __name__=='__main__':main()
