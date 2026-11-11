#!/usr/bin/env python3
"""Native Control pose leader using the home_zero Hardware lifecycle."""
import argparse
import csv
import json
from pathlib import Path
import queue
import sys
import threading
import time
from types import SimpleNamespace

import mujoco
from mujoco import viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.config import load
from piper_sim.contracts import LOWER, UPPER
from piper_sim.sync_view import FeedbackGhost, NativeWorkcellInput


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--piper-use', type=Path, default=Path('/media/midea/F496AC8096AC4548/act_td3/piper_use'))
    parser.add_argument('--side', choices=['left', 'right', 'both'], default='right')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--speed', type=float, default=3.)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    if not np.isfinite(a.speed) or not 0 < a.speed <= 3:
        parser.error('speed must be in (0, 3] degrees/s')
    if a.execute and not sys.stdin.isatty():
        parser.error('--execute requires an interactive terminal')
    for path in (a.piper_use, a.piper_use/'example_agilex', a.piper_use/'example_agilex/piper_sdk'):
        sys.path.insert(0, str(path))
    from replay_connector import Hardware, parse_args, prepare_entry
    import replay_controller, piper_controller, piper_sdk

    a.output.mkdir(parents=True, exist_ok=False)
    events = (a.output/'events.jsonl').open('w')
    def event(kind, **values):
        row = dict(time_monotonic=time.monotonic(), event=kind, **values)
        events.write(json.dumps(row)+'\n'); events.flush()
        print(json.dumps(row), flush=True)
    event('runtime', python=sys.executable, numpy=np.__version__,
          controller=replay_controller.__file__, base_controller=piper_controller.__file__,
          sdk=piper_sdk.__file__, control_hz=100, speed_deg_s=a.speed,
          mode='kinematic MuJoCo pose leader; not dynamics calibration')
    config = load(); config['task_scene'] = ROOT/'configs/tasks/real_connector.yaml'
    sim = MujocoBackend(config)
    scene_qpos = sim.data.qpos.copy()
    scene_mocap_pos = sim.data.mocap_pos.copy()
    scene_mocap_quat = sim.data.mocap_quat.copy()
    def place_pose(pose):
        # Native Backspace resets all mjData, including the recorded workcell's
        # mocap fixtures. This pose-leader mode keeps the reference scene fixed.
        sim.data.qpos[:] = scene_qpos
        sim.data.mocap_pos[:] = scene_mocap_pos
        sim.data.mocap_quat[:] = scene_mocap_quat
        sim.set_observation_state(pose)
    args = parse_args([]); args.folded_entry = True; args.control_hz = 100
    args.max_joint_speed = a.speed
    sides = ['left', 'right'] if a.side == 'both' else [a.side]
    hardware = Hardware(args, sides)
    offsets = {'left': 0, 'right': 7}
    keys = queue.Queue(); done = threading.Event(); stopped = threading.Event()
    heartbeat = [time.monotonic()]
    log = (a.output/'trajectory.csv').open('w', newline='')
    writer = csv.writer(log)
    writer.writerow(['monotonic', 'state'] + [f'{side}_{kind}_j{i}_deg' for kind in ('goal', 'sim', 'can') for side in sides for i in range(1, 7)])
    def stop(reason):
        if not stopped.is_set():
            stopped.set()
            for ctrl in hardware.controllers.values():
                ctrl._set_command_stream_enabled(False)
                if hardware.enable_attempted:
                    ctrl.piper.MotionCtrl_1(1, 0, 0)
            event('stopped', reason=reason)
    def watchdog():
        while not done.wait(.05):
            if time.monotonic()-heartbeat[0] > .5:
                stop('Native control heartbeat exceeded 500 ms')
                return
    pose = np.clip(sim.read().position, LOWER, UPPER)
    low = np.zeros((len(sides), 6)); high = low.copy()
    ghosts = {}
    def initialize_view(measured):
        # Called by Hardware after fresh targets are prepared, before enabling.
        # Native MuJoCo caches slider ranges when its window is constructed.
        for index, side in enumerate(sides):
            offset = offsets[side]; ctrl = hardware.controllers[side]
            entry_pose = prepare_entry(measured[index], ctrl.JOINT_LIMITS, True)
            pose[offset:offset+6] = np.deg2rad(entry_pose[:6])
            low[index] = np.maximum.reduce([ctrl.JOINT_LIMITS[:, 0], np.rad2deg(LOWER[offset:offset+6]), entry_pose[:6]-15])
            high[index] = np.minimum.reduce([ctrl.JOINT_LIMITS[:, 1], np.rad2deg(UPPER[offset:offset+6]), entry_pose[:6]+15])
            sim.model.actuator_ctrlrange[sim.indices[side]['ctrl'][:6]] = np.deg2rad(np.column_stack((low[index], high[index])))
        place_pose(pose)
        sim.viewer = viewer.launch_passive(sim.model, sim.data, key_callback=keys.put,
                                          show_left_ui=False, show_right_ui=True)
        sim._overview_camera(sim.viewer.cam)
        if len(sides) == 1:
            sim.viewer.cam.lookat[:] = sim.data.body(f'{sides[0]}/link3').xpos
            sim.viewer.cam.distance = 1.25
        for side in sides:
            ghosts[side] = FeedbackGhost(sim, side)
        sim.viewer.sync()
        event('viewer_initialized_before_enable', sides=sides, limits_deg=[low.tolist(), high.tolist()])
    try:
        episode = SimpleNamespace(episode_id='NATIVE', actions={side: np.zeros((1, 7)) for side in sides})
        entry = hardware.connect(episode, np.zeros((len(sides), 7)), activate_gripper=False,
                                 preview=not a.execute, confirmation=f'SYNC {a.side}',
                                 target_label='原生 Control 当前姿态（不自动回零）',
                                 before_enable=initialize_view)
        if not a.execute:
            entry = np.stack([prepare_entry(entry[i], hardware.controllers[side].JOINT_LIMITS, True) for i, side in enumerate(sides)])
        for index, side in enumerate(sides):
            offset = offsets[side]
            pose[offset:offset+6] = np.deg2rad(entry[index, :6])
        place_pose(pose)
        inputs = NativeWorkcellInput(pose, sides)
        goal = entry[:, :6].copy(); q = goal.copy(); selected = None
        synced_frames = 0
        note = 'Synchronizing native controls with the fresh measured entry pose.'
        event('ready', execute=a.execute, entry_deg=q.tolist(), can={side: hardware.controllers[side].can_interface for side in sides})
        heartbeat[0] = time.monotonic()
        if a.execute:
            threading.Thread(target=watchdog, daemon=True).start()
        last = rendered = time.monotonic()
        while sim.is_running():
            now = time.monotonic(); dt = min(now-last, .02); last = now
            while not keys.empty():
                key = keys.get_nowait(); event('viewer_key', key=key)
                if key in (298, 32): stop('Operator stop')
                elif key == 299:
                    # F10 explicitly means physical support has been provided.
                    stop('Supported shutdown')
                    if a.execute: hardware.emergency()
                    event('supported_disable_requested')
                    return
                elif key == ord('G'):
                    for ghost in ghosts.values(): ghost.enabled = not ghost.enabled
            try:
                if not stopped.is_set():
                    # The first sync can replay GUI resets/edits queued while the
                    # terminal waited. Publish the fresh pose twice before input.
                    if synced_frames >= 2:
                        try:
                            change = inputs.changed(sim.control_targets())
                            if change is not None:
                                side, joint, target = change
                                index = sides.index(side); offset = offsets[side]
                                if not low[index, joint] <= target <= high[index, joint]:
                                    raise ValueError('Target outside session/model/hardware limits')
                                if selected is not None and (side, joint) != selected and np.max(abs(goal-q)) > .15:
                                    raise ValueError('Wait for the current joint before switching')
                                selected = (side, joint); goal[index, joint] = target
                                inputs.expected[offset:offset+6] = np.deg2rad(goal[index])
                                note = f'Native {side}/J{joint+1} target {target:.3f} deg'
                                event('native_control_target', side=side, joint=joint+1, target_deg=target, physical_output=a.execute)
                        except ValueError as exc:
                            # A locked slider/Clear all is not a robot command.
                            # Reject the edit and hold, without losing motor support.
                            event('rejected_native_input', reason=str(exc), controls=sim.control_targets().tolist())
                            goal = q.copy(); inputs.reset(pose)
                            note = 'Input rejected; holding: '+str(exc)
                    q += np.clip(goal-q, -a.speed*dt, a.speed*dt)
                    for index, side in enumerate(sides):
                        offset = offsets[side]
                        pose[offset:offset+6] = np.deg2rad(q[index])
                    place_pose(pose)
                    # Send the displayed MuJoCo qpos, not a separate control widget.
                    q = np.stack([np.rad2deg(sim.read().position[offsets[side]:offsets[side]+6]) for side in sides])
                    if a.execute:
                        hardware.check()
                        target = entry.copy(); target[:, :6] = q
                        hardware.send(target)
                for index, side in enumerate(('left', 'right')):
                    sim.data.ctrl[sim.indices[side]['ctrl']] = inputs.expected[7*index:7*index+7]
                actual = np.stack([hardware.feedback(side, require_enabled=a.execute and not stopped.is_set(),
                                                    allowed_states=(0, 1))[0] for side in sides])
                state = 'stopped' if stopped.is_set() else ('active' if a.execute else 'preview')
                writer.writerow([now, state, *goal.ravel(), *q.ravel(), *actual[:, :6].ravel()]); log.flush()
                if now-rendered >= 1/30:
                    with sim.viewer.lock():
                        gaps = [ghosts[side].update(actual[index, :6], clear=index==0) for index, side in enumerate(sides)]
                    lines = [f'HOME_ZERO HARDWARE | {state}', '        GOAL       SIM       CAN     CAN-SIM (deg)']
                    for index, side in enumerate(sides):
                        lines += [f'{side}: {hardware.controllers[side].can_interface}']
                        lines += [f'J{i+1} {goal[index,i]:9.3f} {q[index,i]:9.3f} {actual[index,i]:9.3f} {actual[index,i]-q[index,i]:+9.3f}' for i in range(6)]
                        lines += [f'FK gap {gaps[index]:.2f} mm (model only)']
                    help_text = f'Native Control: rad | {a.speed:g} deg/s | +/-15 deg/session\nF9 / SPACE: stop | F10: supported disable and exit | G: ghost\nGrey = MuJoCo; cyan = CAN. Connected: {", ".join(sides)}. Grippers locked.\n{note}'
                    sim.viewer.set_texts([
                        (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT, '\n'.join(lines), ''),
                        (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, help_text, '')])
                    sim.viewer.sync(); rendered = now
                    synced_frames += 1
                    if synced_frames == 2:
                        note = 'Native controls ready.'
                        event('native_controls_ready')
                heartbeat[0] = time.monotonic()
            except Exception as exc:
                stop(str(exc))
                break
            time.sleep(max(0, .01-(time.monotonic()-now)))
    finally:
        done.set()
        if a.execute and hardware.enable_attempted: stop('Viewer/application exit')
        sim.close()
        try:
            hardware.shutdown()
            event('shutdown_confirmed')
        finally:
            events.close(); log.close()


if __name__ == '__main__':
    main()
