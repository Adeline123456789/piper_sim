"""Dataset integrity: finalized videos, matching frame counts and truthful labels."""
import json
from pathlib import Path
import tempfile
import unittest

import av
import h5py
import numpy as np

from piper_sim.mixed_recording import CAMERAS, MixedEpisodeWriter


class MixedRecordingTests(unittest.TestCase):
    def test_reference_layout_and_missing_real_actions(self):
        with tempfile.TemporaryDirectory() as folder:
            w=MixedEpisodeWriter(folder,'real','000001','real','calibration')
            for i in range(3):
                row={'timestamp':float(i)/30,'valid/action':False}
                for side in ('left','right'):
                    row[f'state/{side}_arm/joint_positions']=np.arange(6)+i
                    row[f'action/{side}_arm/joint_positions']=np.full(6,np.nan)
                    row[f'state/{side}_arm/gripper_open']=.5
                    row[f'action/{side}_arm/gripper_open']=np.nan
                w.append(row,{n:np.full((32,48,3),i*80,np.uint8) for n in CAMERAS})
            w.close()
            with h5py.File(w.path/'proprio_stats.h5') as f:
                self.assertEqual(f['state/right_arm/joint_positions'].shape,(3,6))
                self.assertTrue(np.isnan(f['action/right_arm/joint_positions'][:]).all())
                self.assertFalse(f.attrs['policy_training_eligible'])
                self.assertFalse(f['valid/action'][:].any())
            for n in CAMERAS:
                with av.open(str(w.video_path/(n+'_rgb.mp4'))) as video:
                    decoded=list(video.decode(video=0))
                    self.assertEqual(len(decoded),3)
                    self.assertGreater(decoded[-1].to_ndarray(format='rgb24').mean(),100)
            self.assertFalse((Path(folder)/'sensors').exists())
            self.assertFalse((w.path/'privileged.jsonl').exists())

    def test_real_cannot_claim_simulator_ground_truth(self):
        with tempfile.TemporaryDirectory() as folder:
            w=MixedEpisodeWriter(folder,'real','000001','real','calibration')
            with self.assertRaisesRegex(ValueError,'ground truth'):
                w.append({'timestamp':0.},{n:np.zeros((32,48,3),np.uint8) for n in CAMERAS},
                         privileged={'objects':{}})
            w.close()

    def test_missing_command_cannot_be_marked_valid(self):
        with tempfile.TemporaryDirectory() as folder:
            w=MixedEpisodeWriter(folder,'real','000001','real','calibration')
            with self.assertRaisesRegex(ValueError,'missing'):
                w.append({'timestamp':0.,'valid/action':True,'action/right_arm/joint_positions':[np.nan]*6},
                         {n:np.zeros((32,48,3),np.uint8) for n in CAMERAS})
            w.close()


if __name__=='__main__':unittest.main()
