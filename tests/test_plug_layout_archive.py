import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from piper_sim.config import ROOT, load
from piper_sim.tasks.recorded_scenes import RealConnector, matrix
from piper_sim.task_archive import archive_episode


def test_random_fixtures_keep_both_plugs_seated_in_holder_frame():
    config = yaml.safe_load((ROOT/'configs/tasks/real_connector_sim_randomized.yaml').read_text())
    task = RealConnector(config, load())
    nominal = task.reset(randomize=False)['poses']
    expected = {name: np.linalg.inv(matrix(nominal['board']))@matrix(nominal[name])
                for name in task.dynamic}
    boards = []
    for seed in range(20):
        poses = task.reset(seed, randomize=True)['poses']
        boards.append(poses['board']['position'])
        for name in task.dynamic:
            relative = np.linalg.inv(matrix(poses['board']))@matrix(poses[name])
            np.testing.assert_allclose(relative, expected[name], atol=1e-12)
    assert np.ptp(boards, axis=0)[:2].min() > .02
    fixed = task.reset(42, randomize=False)['poses']
    assert fixed == nominal


@pytest.mark.parametrize('domain,result,branch', [
    ('sim', True, 'complete_sim/succeed'), ('sim', False, 'complete_sim/failed'),
    ('real', True, 'complete_real/successed'), ('real', False, 'complete_real/failed')])
def test_archive_routes_observed_outcomes(tmp_path, domain, result, branch):
    source = tmp_path/'proprio_stats/session/000001'
    source.mkdir(parents=True)
    metadata = dict(domain=domain, success=result, policy_training_eligible=False,
                    execution_outcome={'physical_task_success': result})
    (source/'metadata.json').write_text(json.dumps(metadata))
    (tmp_path/'observations/session/000001/videos').mkdir(parents=True)
    target = archive_episode(tmp_path, 'session', '000001')
    assert target == tmp_path/branch/'session_000001'
    assert json.loads((target/'outcome.json').read_text())['success'] is result
    assert (source/'metadata.json').exists()


def test_unobserved_real_outcome_not_archived_as_success(tmp_path):
    source = tmp_path/'proprio_stats/session/000001'
    source.mkdir(parents=True)
    (source/'metadata.json').write_text(json.dumps(dict(domain='real', success=True)))
    with pytest.raises(ValueError, match='observed'):
        archive_episode(tmp_path, 'session', '000001')
