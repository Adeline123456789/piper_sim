"""Exercise the new entry point with real MuJoCo and an entirely fake robot.

SDK/controller imports are replaced before main(), so these tests cannot open CAN.
"""
import contextlib
import importlib.util
import json
import sys
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from piper_sim.config import ROOT
from piper_sim.contracts import LOWER, UPPER


@pytest.mark.parametrize('execute,joint,finish,mode,controlled', [
    (True, 0, 'stop', 'right', 'right'), (True, 5, 'disable', 'left', 'left'),
    (False, 5, 'close', 'both', 'left')] +
    [(True, joint, 'stop', 'both', side) for side in ('left', 'right') for joint in range(6)])
def test_native_input_reaches_hardware_send_and_stops(monkeypatch, tmp_path, execute, joint, finish, mode, controlled):
    spec = importlib.util.spec_from_file_location('native_home_under_test', ROOT/'scripts/sync_mujoco_home.py')
    app = importlib.util.module_from_spec(spec); spec.loader.exec_module(app)
    bounds = np.rad2deg(np.column_stack((LOWER[7:13], UPPER[7:13])))
    state = SimpleNamespace(sent=[], stops=[], disabled=False, shutdown=False, count_at_stop=None,
                            viewer_warmed=False, connected=False, input_sent=False)
    sides = ['left', 'right'] if mode == 'both' else [mode]
    angle = -1. if joint == 2 else 1.
    class Hardware:
        def __init__(self, args, requested):
            assert requested == sides
            self.enable_attempted = False
            self.controllers = {side: SimpleNamespace(
                JOINT_LIMITS=bounds, can_interface='FAKE_ONLY',
                _set_command_stream_enabled=lambda value: state.stops.append(value),
                piper=SimpleNamespace(MotionCtrl_1=lambda *args: state.stops.append(args))) for side in sides}
            self.q = np.zeros((len(sides), 7))
        def connect(self, episode, expected, **kwargs):
            assert not self.enable_attempted
            kwargs['before_enable'](self.q.copy())
            assert state.viewer_warmed
            state.connected = True
            assert kwargs['activate_gripper'] is False
            self.enable_attempted = not kwargs['preview']
            return self.q.copy()
        def check(self): pass
        def send(self, target):
            assert self.enable_attempted
            self.q = target.copy()
            state.sent.append((time.monotonic(), self.q.copy()))
        def feedback(self, side, **kwargs): return self.q[sides.index(side)].copy(), {}
        def emergency(self): state.disabled = True
        def shutdown(self): state.shutdown = True
    monkeypatch.setitem(sys.modules, 'replay_connector', SimpleNamespace(
        Hardware=Hardware, parse_args=lambda _: SimpleNamespace(),
        prepare_entry=lambda q, *args: q))
    for name in ('replay_controller', 'piper_controller', 'piper_sdk'):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(__file__='FAKE_NO_CAN'))
    class Viewer:
        def __init__(self, model, data, key_callback, **kwargs):
            for side in sides:
                np.testing.assert_allclose(model.actuator(f'{side}/joint6').ctrlrange, np.deg2rad([-15, 15]))
            self.model, self.data, self.key_callback = model, data, key_callback
            self.cam = mujoco.MjvCamera(); self.calls = 0; self.closed = False
        def lock(self): return contextlib.nullcontext()
        def set_texts(self, *args): pass
        def is_running(self): return not self.closed and self.calls < 24
        def close(self): self.closed = True
        def sync(self):
            self.calls += 1
            state.viewer_warmed = True
            if state.connected and self.calls == 2:
                # MuJoCo can replay Backspace/Clear-all queued before enabling.
                self.data.ctrl[:] = 0
            if state.connected and self.calls >= 4 and not state.input_sent:
                self.data.ctrl[self.model.actuator(f'{controlled}/joint{joint+1}').id] = np.deg2rad(angle)
                state.input_sent = True
            if self.calls == 18:
                state.count_at_stop = len(state.sent)
                if finish == 'stop': self.key_callback(298)
                if finish == 'disable': self.key_callback(299)
    monkeypatch.setattr(app.viewer, 'launch_passive', Viewer)
    monkeypatch.setattr(app, 'FeedbackGhost', lambda *args: SimpleNamespace(enabled=True, update=lambda q, **kwargs: 0.))
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr(sys, 'argv', ['sync_mujoco_home', '--side', mode, '--output', str(tmp_path/'run')] + (['--execute'] if execute else []))
    app.main()
    assert state.shutdown
    events = [json.loads(line) for line in (tmp_path/'run/events.jsonl').read_text().splitlines()]
    target = next(e for e in events if e['event'] == 'native_control_target')
    assert target['joint'] == joint+1 and target['target_deg'] == pytest.approx(angle)
    assert target['side'] == controlled
    assert target['physical_output'] == execute
    if execute:
        sent = np.array([q for _, q in state.sent]).reshape(len(state.sent), -1)
        active = sides.index(controlled)*7+joint
        assert sent[-1, active] == pytest.approx(angle)
        assert np.max(np.abs(np.diff(sent, axis=0))) <= .060001
        np.testing.assert_allclose(sent[:, [i for i in range(7*len(sides)) if i != active]], 0)
        assert len(state.sent) == state.count_at_stop
        assert state.stops
        assert state.disabled == (finish == 'disable')
    else:
        assert not state.sent and not state.disabled and not state.stops
