#!/usr/bin/env python3
"""Native MuJoCo Control -> Piper; CAN pose ghost and errors in the SAME viewer.

F8 enables after the operator clears the arm. F9 stops. F10 means the operator has
supported the arm and requests disable. No separate control GUI is created.
"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import queue
import sys
import threading
import time

import mujoco
from mujoco import viewer as mj_viewer
import numpy as np

from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.home_zero_transport import HomeZeroTransport, verify_mapping
from piper_sim.config import ROOT, load
from piper_sim.contracts import LOWER, UPPER
from piper_sim.joint_sync import JointSync
from piper_sim.sync_view import FeedbackGhost, NativeJointInput


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--piper-use", type=Path, default=Path("/media/midea/F496AC8096AC4548/act_td3/piper_use"))
    p.add_argument("--can", default="can_arm2")
    p.add_argument("--side", choices=("left", "right"), default="right")
    p.add_argument("--excursion-deg", type=float, default=15.)
    p.add_argument("--speed-deg-s", type=float, default=.5)
    p.add_argument("--output", type=Path)
    p.add_argument("--execute", action="store_true", help="Use terminal SYNC confirmation instead of the F8 enable key")
    p.add_argument("--preview-only", action="store_true", help="Native Control preview; reject all enable requests")
    args = p.parse_args()
    if args.execute and args.preview_only:
        p.error('--execute and --preview-only are mutually exclusive')
    sys.path.insert(0, str(args.piper_use))
    sys.path.insert(0, str(args.piper_use/"example_agilex"))
    sys.path.insert(0, str(args.piper_use/"example_agilex/piper_sdk"))
    from check_joint_control import snapshot, limits
    from piper_session import CanLocks

    folder = args.output or ROOT/"outputs/live_piper"/datetime.now().strftime("native_sync_%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=False)
    cfg = load()
    cfg["task_scene"] = ROOT/"configs/tasks/real_connector.yaml"
    sdk = sim = controller = None
    done, keys = threading.Event(), queue.Queue()
    offset = 0 if args.side == "left" else 7
    model_bounds = np.rad2deg(np.column_stack((LOWER[offset:offset+6], UPPER[offset:offset+6])))
    log = (folder/"trajectory.csv").open("w", newline="")
    writer = csv.writer(log)
    writer.writerow(["host_monotonic_s", "state", "selected_joint"] +
                    [f"{kind}_j{i}_deg" for kind in ("goal", "sim", "sent", "feedback") for i in range(1, 7)] +
                    [f"motor_{kind}_j{i}_raw" for kind in ("pos", "speed", "current") for i in range(1, 7)])
    events = (folder/"events.jsonl").open("w")
    def event(kind, **details):
        row = dict(time_monotonic=time.monotonic(), event=kind, **details)
        events.write(json.dumps(row, ensure_ascii=False)+"\n")
        events.flush()
        print(json.dumps(row, ensure_ascii=False), flush=True)
    try:
        with CanLocks([args.can]):
            verify_mapping(args.piper_use/"example_agilex/scripts/piper_vision_ports.json", args.can, args.side)
            sdk = HomeZeroTransport(args.can)
            time.sleep(.6)
            bounds = limits(sdk)
            bounds[:, 0] = np.maximum(bounds[:, 0], model_bounds[:, 0])
            bounds[:, 1] = np.minimum(bounds[:, 1], model_bounds[:, 1])
            controller = JointSync(sdk, snapshot, bounds, speed=args.speed_deg_s, excursion=args.excursion_deg)
            initial = controller.read()
            sim = MujocoBackend(cfg)
            pose = np.clip(sim.read().position, LOWER, UPPER)
            pose[offset:offset+6] = np.clip(np.deg2rad(initial["joints_deg"]), LOWER[offset:offset+6], UPPER[offset:offset+6])
            sim.set_observation_state(pose)
            reference = np.rad2deg(pose[offset:offset+6])
            low = np.maximum(bounds[:, 0], reference-args.excursion_deg)
            high = np.minimum(bounds[:, 1], reference+args.excursion_deg)
            sim.model.actuator_ctrlrange[sim.indices[args.side]["ctrl"][:6]] = np.deg2rad(np.column_stack((low, high)))
            sim.viewer = mj_viewer.launch_passive(sim.model, sim.data, key_callback=keys.put,
                                                 show_left_ui=False, show_right_ui=True)
            sim._overview_camera(sim.viewer.cam)
            sim.viewer.cam.lookat[:] = sim.data.body(f"{args.side}/link3").xpos
            sim.viewer.cam.distance = 1.25
            inputs = NativeJointInput(sim.control_targets(), offset)
            ghost = FeedbackGhost(sim, args.side)
            preview_goal = pose.copy()
            note = "PREVIEW ONLY: Control moves the virtual arm. F8 enables real following."
            event("connected_read_only", can=args.can, configured_side=args.side, feedback=initial,
                  other_arm_commanded=False, gripper_commanded=False,
                  simulation_mode="kinematic_joint_pose_leader; not dynamics validation",
                  control_source="native_mujoco_control",
                  hardware_controller="home_zero.py / ReplayController / 100 Hz",
                  interpretation="Cyan ghost is CAN feedback FK, not visual confirmation of motion")
            if args.execute:
                if not sys.stdin.isatty():
                    raise RuntimeError('--execute requires an interactive terminal')
                print(f'确认已松开 {args.side} 臂并避开活动范围，输入 SYNC {args.side} 启用原生 Control 跟随；不会自动回零或往返：', flush=True)
                if input().strip() != f'SYNC {args.side}':
                    return
                keys.put(297)
            def watch():
                while not done.wait(.05):
                    try:
                        controller.watchdog()
                    except Exception:
                        return
            threading.Thread(target=watch, daemon=True).start()
            def advance(q):
                target = pose.copy()
                target[offset:offset+6] = np.deg2rad(q)
                # Pose teleoperation: read back the visual model's actual qpos.
                sim.set_observation_state(target)
                return np.rad2deg(sim.read().position[offset:offset+6])
            last_state = last_error = None
            while sim.is_running():
                began = time.monotonic()
                try:
                    while not keys.empty():
                        key = keys.get_nowait()
                        event("viewer_key", key=key)
                        if key == 297 and controller.state == "idle":
                            if args.preview_only:
                                note = 'PREVIEW-ONLY: enable rejected; no real motion commands.'
                                continue
                            state = controller.read()
                            controller.start(state["joints_deg"])
                            pose[offset:offset+6] = np.deg2rad(controller.reference)
                            inputs.reset(pose)
                            sim.model.actuator_ctrlrange[sim.indices[args.side]["ctrl"][:6]] = np.deg2rad(np.column_stack((controller.low, controller.high)))
                            note = "Enabling six arm motors; J2/J3 enter legal zero boundary if needed."
                            event("operator_started", initial_deg=controller.initial.tolist(), entry_deg=controller.reference.tolist())
                        elif key in (298, 32):
                            controller.stop("Operator stop")
                            note = "STOPPED. Support the arm, then press F10 to disable."
                        elif key == 299:
                            controller.request_disable()
                            event("operator_confirmed_support_and_disable")
                        elif key == ord("G"):
                            ghost.enabled = not ghost.enabled
                    if controller.state in ("active", "idle"):
                        change = inputs.changed(sim.control_targets())
                        if change is not None:
                            j, goal = change
                            if controller.state == 'active':
                                if j != controller.selected:
                                    controller.select(j)
                                controller.target(goal)
                            else:
                                if not low[j] <= goal <= high[j]:
                                    raise ValueError('Preview goal outside joint interval')
                                preview_goal[offset+j] = np.deg2rad(goal)
                                inputs.expected = preview_goal.copy()
                            event("native_control_target", joint=j+1, goal_deg=goal, physical_output=controller.state=='active')
                            note = f"Native Control: {args.side}/joint{j+1} -> {goal:.3f} deg"
                    state = controller.tick(advance)
                    if controller.state == 'idle':
                        virtual = sim.read().position
                        virtual[offset:offset+6] += np.clip(preview_goal[offset:offset+6]-virtual[offset:offset+6],
                                                           -np.deg2rad(args.speed_deg_s/30), np.deg2rad(args.speed_deg_s/30))
                        sim.set_observation_state(virtual)
                    elif controller.state in ('resuming', 'enabling', 'entry'):
                        q = controller.reference
                        pose[offset:offset+6] = np.clip(np.deg2rad(q), LOWER[offset:offset+6], UPPER[offset:offset+6])
                        sim.set_observation_state(pose)
                        inputs.reset(pose)
                    elif controller.state == 'active':
                        # Preserve the UI goal while the visual arm approaches it.
                        inputs.expected[offset:offset+6] = np.deg2rad(controller.desired)
                    for i, side in enumerate(("left", "right")):
                        sim.data.ctrl[sim.indices[side]["ctrl"]] = inputs.expected[7*i:7*i+7]
                    feedback = np.array(state["joints_deg"])
                    simulated = np.rad2deg(sim.read().position[offset:offset+6])
                    sent = controller.command if controller.command is not None else np.full(6, np.nan)
                    goal = controller.desired if controller.state != 'idle' and controller.desired is not None else np.rad2deg(preview_goal[offset:offset+6])
                    with sim.viewer.lock():
                        fk_gap = ghost.update(feedback)
                    error = feedback-simulated
                    left = [f"NATIVE CONTROL -> {args.can} ({args.side})",
                            f"State: {controller.state} | enabled {sum(state['motors_enabled'])}/6",
                            "Grey: virtual leader | CYAN: CAN feedback pose",
                            "         GOAL     SIM      CAN      CAN-SIM (deg)"]
                    left += [f"J{i+1}  {goal[i]:8.3f} {simulated[i]:8.3f} {feedback[i]:8.3f} {error[i]:+9.3f}" for i in range(6)]
                    left += [f"FK tip gap: {fk_gap:.2f} mm (model-based, not camera measurement)"]
                    help_text = "\n".join([
                        "Control units: rad.  5 deg = 0.08727 rad",
                        f"One joint at a time; max {args.speed_deg_s} deg/s; +/-{args.excursion_deg} deg/session",
                        "F8: enable (release/clear arm) | F9 / SPACE: STOP",
                        "F10: disable ONLY after support | G: toggle feedback ghost",
                        "CAN feedback alone does not confirm physical motion.",
                        controller.error or note])
                    sim.viewer.set_texts([
                        (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT, "\n".join(left), ""),
                        (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, help_text, "")])
                    sim.viewer.sync()
                    motor = sdk.GetArmHighSpdInfoMsgs()
                    fresh = -.1 <= time.time()-motor.time_stamp < .25 and motor.Hz > 0
                    raw = [getattr(getattr(motor, f"motor_{i}"), field) if fresh else ""
                           for field in ("pos", "motor_speed", "current") for i in range(1,7)]
                    writer.writerow([time.monotonic(), controller.state, controller.selected+1, *goal, *simulated, *sent, *feedback, *raw])
                    log.flush()
                    if last_state != controller.state:
                        event("state", state=controller.state, feedback=state, error=controller.error)
                        last_state = controller.state
                except Exception as exc:
                    controller.stop(str(exc))
                    note = str(exc)
                    sim.viewer.set_texts((None, None, "STOPPED: "+note+"\nSupport arm, press F10 to disable.", ""))
                    sim.viewer.sync()
                    for i, side in enumerate(("left", "right")):
                        sim.data.ctrl[sim.indices[side]["ctrl"]] = inputs.expected[7*i:7*i+7]
                    if str(exc) != last_error:
                        event("error", detail=str(exc))
                        last_error = str(exc)
                time.sleep(max(0., 1/30-(time.monotonic()-began)))
            if controller.touched:
                controller.stop("Viewer closed")
                event("viewer_closed_support_required")
                print("机械臂已请求停止。扶稳后在本终端输入 DISABLE 下使能。", flush=True)
                if sys.stdin.isatty():
                    while input().strip() != "DISABLE":
                        pass
                    controller.request_disable()
                    while controller.touched and controller.state == "disabling":
                        controller.tick(lambda q: q)
                        time.sleep(.05)
                    event("shutdown", confirmed=not controller.touched)
    finally:
        done.set()
        if controller and controller.touched:
            controller.stop("Application exit; physical support required before disable")
            event("exit_requires_supported_disable")
        if sdk:
            sdk.DisconnectPort()
        if sim:
            sim.close()
        log.close()
        events.close()


if __name__ == "__main__":
    main()
