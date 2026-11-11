"""Decode all episode videos and verify sample counts and domain/label integrity."""
import argparse
import json
from pathlib import Path
import av
import h5py
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();rows=[]
    for path in sorted((a.root/'proprio_stats').glob('*/*/proprio_stats.h5')):
        session,episode=path.parent.parent.name,path.parent.name
        with h5py.File(path) as f:
            count=int(f.attrs['num_frames']);domain=f.attrs['domain']
            assert np.all(np.diff(f['timestamp'][:])>0),path
            assert f['timestamp'].shape==(count,),path
            for arm in ('left','right'):
                assert f[f'state/{arm}_arm/joint_positions'].shape==(count,6),path
            valid=f['valid/action'][:]
            for arm in ('left','right'):
                assert np.isfinite(f[f'action/{arm}_arm/joint_positions'][:][valid]).all(),path
                partial=f'valid/{arm}_arm_joint_action'
                if partial in f:
                    assert np.isfinite(f[f'action/{arm}_arm/joint_positions'][:][f[partial][:]]).all(),path
                grip_valid=f'valid/{arm}_gripper_action'
                if grip_valid in f:
                    grip=f[f'action/{arm}_arm/gripper_open'][:][f[grip_valid][:]]
                    assert np.isfinite(grip).all() and ((grip>=0)&(grip<=1)).all(),path
            eligible=bool(f.attrs['policy_training_eligible'])
            if eligible:
                assert bool(f.attrs['success']) and valid.all(),path
                assert f.attrs['purpose']=='yellow_pick_insert',path
                if domain=='sim':assert f['valid/collision'][:].all(),path
            videos={}
            for camera in ('cam_external','cam_left_wrist','cam_right_wrist'):
                video_path=a.root/'observations'/session/episode/'videos'/(camera+'_rgb.mp4')
                with av.open(str(video_path)) as video:
                    frames=sum(1 for _ in video.decode(video=0))
                assert frames==count,(video_path,frames,count)
                videos[camera]=frames
            privileged=path.parent/'privileged.jsonl'
            if domain=='sim':
                records=[json.loads(line) for line in privileged.open()]
                assert len(records)==count and [r['frame'] for r in records]==list(range(count)),path
            else:assert not privileged.exists(),path
            rows.append(dict(session=session,episode=episode,domain=domain,num_frames=count,
                purpose=f.attrs['purpose'],success=bool(f.attrs['success']),
                policy_training_eligible=eligible,video_frames=videos))
            rows[-1]['known_right_joint_action_frames']=int(np.sum(f['valid/right_arm_joint_action'][:])) \
                if 'valid/right_arm_joint_action' in f else int(np.sum(valid))
            rows[-1]['known_right_gripper_action_frames']=int(np.sum(f['valid/right_gripper_action'][:])) \
                if 'valid/right_gripper_action' in f else int(np.sum(valid))
    assert not (a.root/'sensors').exists(),'Unexpected force-sensor directory'
    report=dict(episodes=rows,verified_frames=sum(r['num_frames'] for r in rows),
                full_sim_demonstrations=sum(r['domain']=='sim' and r['policy_training_eligible'] for r in rows),
                full_real_demonstrations=sum(r['domain']=='real' and r['policy_training_eligible'] for r in rows),
                real_task_attempts=sum(r['domain']=='real' and r['purpose']=='yellow_pick_insert_attempt' for r in rows),
                force_sensor_directory_present=False)
    (a.root/'dataset_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
