"""Run a complete simulated right-arm task through the existing home_zero lifecycle.

Dry-run is the default. Simulation success does not establish real hole alignment.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np


def resample(rows, max_joint_speed=12., hz=100):
    """Keep the local smooth path; stretch its time axis to the real joint limit."""
    source_time = np.array([r['time'] for r in rows])
    source_time -= source_time[0]
    targets = np.array([[*np.rad2deg(r['action'][7:13]), r['action'][13]/.095] for r in rows])
    dt = np.diff(source_time)
    if not np.isfinite(targets).all() or not np.isfinite(source_time).all() or np.any(dt <= 0):
        raise ValueError('Trajectory must have finite targets and increasing timestamps')
    velocity = np.abs(np.diff(targets, axis=0) / dt[:, None])
    scale = max(1., velocity[:, :6].max()/max_joint_speed, velocity[:, 6].max()/.5)
    timeline = np.r_[np.arange(0, source_time[-1]*scale, 1/hz), source_time[-1]*scale]
    query = timeline/scale
    samples = np.column_stack([np.interp(query, source_time, targets[:, j]) for j in range(7)])
    phases = [rows[i]['phase'] for i in np.searchsorted(source_time, query, side='right')-1]
    return timeline, samples, phases


def fold_targets(start, max_joint_speed=12., hz=100):
    """Return all six joints to zero, preserving the supplied gripper target."""
    goal = start.copy()
    goal[:, :6] = 0
    seconds = max(1., 1.875*np.max(abs(start[:, :6]))/max_joint_speed)
    for alpha in np.linspace(0, 1, int(np.ceil(seconds*hz))+1)[1:]:
        yield start+(goal-start)*(alpha**3*(10-15*alpha+6*alpha**2))


def yellow_top(image):
    import cv2
    # shortcut: fixed task wrist view; revise this ROI when the camera mount changes.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([15, 90, 60]), np.array([45, 255, 255]))
    mask[:250] = 0; mask[:, :180] = 0; mask[:, 470:] = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or cv2.contourArea(max(contours, key=cv2.contourArea)) < 300:
        return None
    return cv2.boundingRect(max(contours, key=cv2.contourArea))[1]


def adopt_held(hardware, pid, targets):
    import os
    import signal
    from replay_controller import ReplayController
    from piper_session import CanLocks
    from check_joint_control import limits
    old = Path(f'/proc/{pid}')
    if old.exists():
        command = (old/'cmdline').read_bytes()
        if b'fold_held.py' not in command and b'continue_plug_task.py' not in command:
            raise RuntimeError('Handoff source is not the task controller')
    if int(Path(f'/tmp/piper_control_{os.getuid()}_can_arm2.lock').read_text()) != pid:
        raise RuntimeError('Task controller does not own the right CAN channel')
    ctrl = ReplayController(can_interface='can_arm2', robot_rate=100,
                            auto_enable=False, raw_joint_feedback=True)
    hardware.controllers['right'] = ctrl
    time.sleep(1)
    bound = limits(ctrl.piper)
    ctrl.JOINT_LIMITS = np.column_stack((np.maximum(ctrl.JOINT_LIMITS[:, 0], bound[:, 0]),
                                       np.minimum(ctrl.JOINT_LIMITS[:, 1], bound[:, 1])))
    hardware.validate_limits('right', targets[:, :6], 'next task')
    q, _ = hardware.feedback('right', True)
    time.sleep(.1)
    again, _ = hardware.feedback('right', True)
    if np.max(abs(q[:6]-again[:6])) > .5 or np.max(abs(again[:6]-targets[0, :6])) > 1:
        raise RuntimeError('Held arm is not stationary at the next task entry')
    if old.exists():
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic()+2
    while old.exists() and (old/'stat').read_text().split(') ')[1][0] != 'Z':
        if time.monotonic() > deadline:
            raise RuntimeError('Previous task sender has not exited')
        time.sleep(.01)
    # A dead process leader can precede its final SDK thread releasing the flock.
    while True:
        try:
            hardware._can_locks = CanLocks(['can_arm2']).acquire()
            break
        except RuntimeError:
            if time.monotonic() > deadline:
                raise
            time.sleep(.02)
    hardware.enable_attempted = True
    q, _ = hardware.feedback('right', True)
    ctrl.set_target_joint_angles(q[:6]); ctrl.set_gripper_open_value(q[6])
    ctrl._set_robot_status_enabled(True); ctrl._set_command_stream_enabled(True)
    ctrl.start_control_loop()
    ctrl.gripper_active = True
    return q[None, :]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--handoff-pid', type=int, help='Continue from the stationary enabled task controller')
    p.add_argument('--real-actions', type=Path, help='Measured residual action plan; not a simulated rollout')
    a = p.parse_args()
    rows = [json.loads(line) for line in (a.trial/'trajectory.jsonl').open()]
    if not json.loads((a.trial/'report.json').read_text())['complete_pick_insert']:
        raise ValueError('Source simulation task did not finish')
    residual = json.loads(a.real_actions.read_text()) if a.real_actions else None
    if a.execute and residual and residual.get('execution_status') == 'rejected':
        raise ValueError('This real trajectory failed alignment and is retired: '+residual['rejection_reason'])
    if residual:
        rows = residual['rows']
    timeline, targets, phases = resample(rows)
    fold_seconds = 0. if phases[-1] == 'fold' else max(1., 1.875*np.max(abs(targets[-1, :6]))/12.)
    report = dict(source_trial=str(a.trial.resolve()), frames=len(rows), steps=len(targets),
                  trajectory_seconds=float(timeline[-1]), max_joint_speed_deg_s=12.,
                  fold_seconds=float(fold_seconds),
                  planned_with_fold_and_holds_seconds=float(timeline[-1]+fold_seconds+2),
                  task_sdk_speed_percent=15, phases=list(dict.fromkeys([*phases, 'fold'])),
                  first_target=targets[0].tolist(), last_target=targets[-1].tolist(),
                  folded_joint_goal_deg=[0.]*6,
                  real_action_plan=str(a.real_actions) if a.real_actions else None,
                  action_provenance='real residual IK, not simulated physics' if residual else 'simulated trajectory',
                  completed_sequence=False, physical_task_success=None)
    if not a.execute:
        print(json.dumps(dict(report, hardware_commands=0), indent=2))
        return
    import cv2
    for camera in ('external', 'left_wrist', 'right_wrist'):
        if not (a.capture/camera/'READY').exists():
            raise RuntimeError('Camera not ready: '+camera)
    root = Path('/media/midea/F496AC8096AC4548/act_td3/piper_use')
    for path in (root, root/'example_agilex', root/'example_agilex/piper_sdk'):
        sys.path.insert(0, str(path))
    from replay_connector import Hardware, Keyboard, parse_args, transition
    args = parse_args([])
    args.folded_entry = True
    args.max_joint_speed = 12.
    args.control_hz = 100
    hardware = Hardware(args, ['right'])
    with (a.capture/'commanded_probe.csv').open('x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['monotonic', 'phase']+
                        [f'{kind}_j{i}' for kind in ('target', 'feedback') for i in range(1, 7)]+
                        ['target_gripper_open', 'feedback_gripper_open'])
        try:
            def start_capture(entry):
                if np.max(abs(entry[0, :6]-targets[0, :6])) > 1:
                    raise RuntimeError('Current entry differs from the simulated start; regenerate the path')
                (a.capture/'START').write_text(str(time.monotonic()))
            episode = SimpleNamespace(episode_id='YELLOW_PICK_INSERT', actions={'right': targets})
            if a.handoff_pid:
                entry = adopt_held(hardware, a.handoff_pid, targets)
                start_capture(entry)
            else:
                entry = hardware.connect(episode, targets[:1], confirmation='TASK right',
                                         target_label='完整取插轨迹入口', before_enable=start_capture)
            hardware.controllers['right'].joint_speed_percent = 15
            print('TASK: joint target <=12 deg/s, SDK 15%, gripper SDK effort command 300 (not calibrated force/torque)', flush=True)
            with Keyboard() as keyboard:
                def emit(target, phase, seconds=.01):
                    tick = time.monotonic()
                    hardware.poll(keyboard)
                    hardware.send(target)
                    actual = hardware.feedback('right', True)[0]
                    writer.writerow([time.monotonic(), phase, *target[0, :6], *actual[:6],
                                     target[0, 6], actual[6]])
                    f.flush()
                    time.sleep(max(0, seconds-(time.monotonic()-tick)))
                for step in transition(entry, targets[:1], 1., args):
                    emit(step, 'prepare_gripper')
                last_phase = None
                baseline = None
                last_visual = 0.
                slipped = False
                for i, (target, phase) in enumerate(zip(targets, phases)):
                    if phase != last_phase:
                        print('TASK_PHASE', phase, flush=True)
                        (a.capture/'phase.json').write_text(json.dumps(dict(phase=phase, monotonic=time.monotonic())))
                        last_phase = phase
                    if phase in ('align', 'insert') and time.monotonic()-last_visual >= .1:
                        import cv2
                        frame = sorted((a.capture/'right_wrist').glob('*.jpg'))[-2]
                        top = yellow_top(cv2.imread(str(frame)))
                        last_visual = time.monotonic()
                        if baseline is None:
                            baseline = top
                        if top is None or top-baseline > 15:
                            report.update(physical_task_success=False, insertion_stopped_frame=frame.name,
                                          baseline_yellow_top_px=baseline, observed_yellow_top_px=top,
                                          insertion_stop_reason='yellow housing moved in gripper or disappeared')
                            print('INSERTION_SLIP: retreat without releasing, then fold', flush=True)
                            slipped = True
                            break
                    seconds = timeline[i+1]-timeline[i] if i+1 < len(timeline) else .01
                    emit(target[None, :], phase, seconds)
                if slipped:
                    start = hardware.feedback('right', True)[0][None, :]
                    start[0, 6] = target[6]
                    retreat = targets[phases.index('align')-1:phases.index('align')].copy()
                    retreat[0, 6] = start[0, 6]
                    seconds = max(1., 1.875*np.max(abs(retreat[0, :6]-start[0, :6]))/12.)
                    for alpha in np.linspace(0, 1, int(np.ceil(seconds*100))+1)[1:]:
                        emit(start+(retreat-start)*(alpha**3*(10-15*alpha+6*alpha**2)), 'abort_retreat')
                else:
                    hardware.wait_targets(targets[-1:], gripper=False, keyboard=keyboard)
                folded = hardware.feedback('right', True)[0][None, :]
                if slipped:
                    folded[0, 6] = target[6]
                if slipped or phases[-1] != 'fold':
                    print('TASK_PHASE fold', flush=True)
                    for folded in fold_targets(folded):
                        emit(folded, 'fold')
                hardware.wait_targets(folded, gripper=False, keyboard=keyboard)
                for _ in range(100):
                    emit(folded, 'final_hold')
            report['completed_sequence'] = not slipped
            report['fold_completed'] = True
            report['final_feedback'] = hardware.feedback('right', True)[0].tolist()
            print('ATTEMPT_FAILED: insertion slip; folded and holding enabled' if slipped else
                  'TASK_SEQUENCE_COMPLETE: physical success unverified; holding enabled', flush=True)
        except BaseException as exc:
            report['error'] = str(exc)
            raise
        finally:
            time.sleep(.15)
            (a.capture/'STOP').write_text('sequence ended\n')
            (a.capture/'execution_report.json').write_text(json.dumps(report, indent=2)+'\n')
            hardware.shutdown()


if __name__ == '__main__':
    main()
