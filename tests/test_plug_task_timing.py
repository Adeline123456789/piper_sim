"""Offline timing checks: no SDK imports or CAN access."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'continue_plug_task', Path(__file__).parents[1]/'scripts/continue_plug_task.py')
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)


def test_resampling_preserves_path_endpoints_phases_and_speed_limits():
    rows = []
    for t, q, grip, phase in [(0, 0, .06, 'transport'), (.1, 3, .06, 'align'),
                              (.2, 6, .04, 'insert'), (.3, 6, .04, 'release')]:
        rows.append(dict(time=t, phase=phase, action=[0]*7+[np.deg2rad(q)]+[0]*5+[grip]))
    timeline, targets, phases = task.resample(rows)
    velocity = abs(np.diff(targets, axis=0)/np.diff(timeline)[:, None])
    assert velocity[:, :6].max() <= 12+1e-9
    assert velocity[:, 6].max() <= .5+1e-9
    np.testing.assert_allclose(targets[[0, -1], 0], [0, 6])
    np.testing.assert_allclose(targets[[0, -1], 6], [.06/.095, .04/.095])
    assert list(dict.fromkeys(phases)) == ['transport', 'align', 'insert', 'release']
    assert timeline[-1] == pytest.approx(.3*(.02/.095/.1/.5))
    assert np.all(np.diff(timeline) > 0)
    assert np.diff(timeline).max() <= .01+1e-9
    rows[1]['time'] = 0
    with pytest.raises(ValueError, match='increasing timestamps'):
        task.resample(rows)


@pytest.mark.parametrize('grip', [.63, .027/.095])
def test_fold_is_smooth_speed_limited_and_keeps_gripper_opening(grip):
    start = np.array([[45.7, 112.9, -88.6, -4., 67.1, -43.3, grip]])
    samples = np.concatenate([start, *task.fold_targets(start)])
    np.testing.assert_allclose(samples[-1, :6], 0, atol=1e-12)
    np.testing.assert_allclose(samples[:, 6], grip)
    assert np.max(abs(np.diff(samples[:, :6], axis=0)))*100 <= 12+1e-9
    assert np.max(abs(samples[1, :6]-samples[0, :6])) < .001
    assert np.max(abs(samples[-1, :6]-samples[-2, :6])) < .001


def test_retired_real_plan_is_rejected_before_hardware_import(tmp_path, monkeypatch):
    (tmp_path/'trajectory.jsonl').write_text('')
    (tmp_path/'report.json').write_text(json.dumps({'complete_pick_insert': True}))
    plan = tmp_path/'real_actions.json'
    plan.write_text(json.dumps({'execution_status': 'rejected', 'rejection_reason': 'Repeated insertion slip'}))
    monkeypatch.setattr(sys, 'argv', ['task', '--trial', str(tmp_path), '--capture', str(tmp_path),
                                    '--real-actions', str(plan), '--execute'])
    with pytest.raises(ValueError, match='retired'):
        task.main()
