"""Bounded right-arm MuJoCo-to-Piper probe using the installed teleoperation diagnostics."""
import argparse
import csv
import json
from pathlib import Path
import select
import sys
import time

import numpy as np

from piper_sim.config import load, ROOT
from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.contracts import Command, LOWER, UPPER
from piper_sim.backends.sdk_tracking import tracked_sdk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piper-use", type=Path, default=Path("/media/midea/F496AC8096AC4548/act_td3/piper_use"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--folded-entry", action="store_true")
    parser.add_argument("--j1-deg", type=float, default=1.)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--resume-software-stop", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.j1_deg) or not 1 <= args.j1_deg <= 5:
        parser.error("j1-deg must be in [1, 5]")
    sys.path.insert(0, str(args.piper_use))
    sys.path.insert(0, str(args.piper_use/"example_agilex/piper_sdk"))
    from check_joint_control import snapshot, limits, full_arm_plan
    from piper_session import CanLocks
    from piper_sdk import C_PiperInterface_V2

    if args.execute and not sys.stdin.isatty():
        parser.error("Execution requires a terminal for supported shutdown")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"executed":False, "passed":False, "can":"can_arm2", "gripper_commanded":False}
    cfg = load()
    cfg["task_scene"] = ROOT/"configs/tasks/real_connector.yaml"
    sim = MujocoBackend(cfg, viewer=args.viewer)
    sdk = None
    touched = False
    rows = []
    stream = (args.output/"trajectory.csv").open("w",newline="")
    writer = csv.writer(stream)
    writer.writerow(["time_s","phase"]+[f"{group}_j{i}_rad" for group in ("target","sent","measured") for i in range(1,7)])
    stream.flush()
    def save():
        (args.output/"result.json").write_text(json.dumps(report, indent=2)+"\n")
    try:
        with CanLocks(["can_arm2"]):
            sdk = tracked_sdk(C_PiperInterface_V2)("can_arm2", start_sdk_joint_limit=False,
                                                 start_sdk_gripper_limit=False)
            sdk.ConnectPort(piper_init=False)
            time.sleep(.6)
            state = snapshot(sdk)
            if state["arm_status"] == 1 and args.resume_software_stop and args.execute:
                if any(state["motors_enabled"]) or state["gripper_enabled"] or state["err_code"] or state["driver_faults"]:
                    raise RuntimeError("Resume requires disabled, fault-free hardware")
                original = np.array(state["joints_deg"])
                sdk.MotionCtrl_1(2,0,0)
                start = time.monotonic()
                stable = None
                while True:
                    time.sleep(.02)
                    state = snapshot(sdk)
                    if any(state["motors_enabled"]) or state["gripper_enabled"] or state["driver_faults"]:
                        raise RuntimeError("Unexpected enable or driver fault during resume")
                    now = time.monotonic()
                    if state["arm_status"] == 0 and state["err_code"] == 0:
                        if stable is None:
                            stable = now
                        if now-start >= .8 and now-stable >= .5:
                            break
                    else:
                        stable = None
                    if now-start > 3:
                        raise RuntimeError("Software stop did not recover")
                if np.max(abs(np.array(state["joints_deg"])-original)) > .3:
                    raise RuntimeError("Pose changed during software-stop recovery")
            initial, hold, goal, bounds = full_arm_plan(state, limits(sdk), args.j1_deg, args.folded_entry)
            report.update(initial_deg=initial.tolist(), entry_deg=hold.tolist(), goal_deg=goal.tolist(),
                          before=state, command_limits_deg=bounds.tolist())
            print(json.dumps(report, indent=2), flush=True)
            # Initial visualization places only the selected arm at the validated entry pose.
            pose = np.clip(sim.read().position, LOWER, UPPER)
            pose[7:13] = np.deg2rad(hold)
            sim.set_observation_state(pose)
            # Check the MuJoCo J1 dynamics offline before any enable or motion request.
            for i in range(round((args.j1_deg*2+1)*30)):
                target = pose.copy()
                target[7] += np.deg2rad(min(i/60,args.j1_deg))
                sim.apply(Command(target,time.monotonic(),i))
                sim.step()
                if abs(np.rad2deg(sim.read().position[7])-np.rad2deg(target[7])) > .3:
                    raise RuntimeError("Simulation J1 tracking failed the offline check")
            sim.set_observation_state(pose)
            report["preview_ok"] = True
            if not args.execute:
                save()
                return
            fresh = snapshot(sdk)
            full_arm_plan(fresh,bounds,args.j1_deg,args.folded_entry)
            if np.max(np.abs(np.array(fresh["joints_deg"])-initial)) > .2:
                raise RuntimeError("Robot pose changed during the offline check")
            low, high = np.minimum(initial,hold)-.7, np.maximum(initial,hold)+.7
            high[0] = goal[0]+.7
            sequence = 0
            began = time.monotonic()

            def tick(target, phase):
                nonlocal sequence
                start = time.monotonic()
                current = snapshot(sdk)
                allowed = (0,6) if phase == "enabling" else (0,)
                if current["arm_status"] not in allowed or current["err_code"] or current["driver_faults"]:
                    raise RuntimeError(f"Hardware fault: {current}")
                if current["gripper_enabled"]:
                    raise RuntimeError("Unexpected gripper enable")
                if phase != "enabling" and not all(current["motors_enabled"]):
                    raise RuntimeError("Lost motor enable")
                measured = np.array(current["joints_deg"])
                if np.any(measured<low) or np.any(measured>high):
                    raise RuntimeError("Measured pose left the bounded probe envelope")
                sim.apply(Command(target,time.monotonic(),sequence))
                sim.step()
                # J1 comes from actual integrated MuJoCo dynamics. Other joints
                # hold the entry pose throughout this one-axis identification test.
                simulated = np.rad2deg(sim.read().position[7:13])
                command = hold.copy()
                command[0] = simulated[0]
                if np.any(command<bounds[:,0]) or np.any(command>bounds[:,1]):
                    raise RuntimeError("MuJoCo output exceeds joint limits")
                if np.max(np.abs(command-measured)) > 3:
                    raise RuntimeError("Tracking error exceeded 3 degrees")
                sdk.MotionCtrl_2(1,1,5,0)
                sdk.JointCtrl(*np.rint(command*1000).astype(int).tolist())
                rows.append([time.monotonic()-began,phase,*target[7:13],*np.deg2rad(command),
                             *np.deg2rad(measured)])
                writer.writerow(rows[-1])
                stream.flush()
                sequence += 1
                if select.select([sys.stdin],[],[],0)[0]:
                    if sys.stdin.readline().strip() in ("q","e"):
                        raise RuntimeError("Operator stopped the probe")
                time.sleep(max(0,1/30-(time.monotonic()-start)))

            touched = True
            report["executed"] = True
            deadline = time.monotonic()+5
            while True:
                tick(pose,"enabling")
                for motor in range(1,7):
                    sdk.EnableArm(motor)
                state = snapshot(sdk)
                if all(state["motors_enabled"]) and state["arm_status"] == 0:
                    break
                if time.monotonic()>deadline:
                    raise RuntimeError("Enable acknowledgement timed out")
            for _ in range(45):
                tick(pose,"entry")
            if np.max(np.abs(np.array(snapshot(sdk)["joints_deg"])-hold)) > .4:
                raise RuntimeError("Entry target did not settle")
            for phase, start_angle, end_angle, seconds in [
                ("outbound",hold[0],goal[0],args.j1_deg*2),("peak",goal[0],goal[0],3),
                ("return",goal[0],hold[0],args.j1_deg*2),("settle",hold[0],hold[0],1)]:
                for fraction in np.linspace(0,1,round(seconds*30)):
                    target = pose.copy()
                    target[7] = np.deg2rad(start_angle+(end_angle-start_angle)*fraction)
                    tick(target,phase)
                report[phase] = snapshot(sdk)
            movement = report["peak"]["joints_deg"][0]-hold[0]
            error = report["settle"]["joints_deg"][0]-hold[0]
            report["passed"] = bool(abs(movement-args.j1_deg)<=.4 and abs(error)<=.3)
            data = np.array([r[2:] for r in rows if r[1] in ("outbound","peak","return","settle")])
            tracking = np.rad2deg(data[:,12:18]-data[:,6:12])
            report.update(j1_movement_deg=movement, return_error_deg=error,
                          tracking_rmse_deg=np.sqrt(np.mean(tracking**2,axis=0)).tolist(),
                          tracking_max_abs_deg=np.max(abs(tracking),axis=0).tolist())
            if not report["passed"]:
                report["error"] = "Measured J1 excursion or return error failed acceptance"
            save()
            print("PROBE_FINISHED "+json.dumps(report),flush=True)
            print("SUPPORT_REQUIRED: support the arm, then type DISABLE to turn motors off.",flush=True)
            # The validated targets remain in the controller while awaiting physical support.
            # Never release gravity-loaded joints based on elapsed time alone.
            while True:
                if input().strip() == "DISABLE":
                    break
            sdk.MotionCtrl_1(1,0,0)
            deadline=time.monotonic()+3
            while True:
                for motor in range(1,7):
                    sdk.DisableArm(motor)
                time.sleep(.05)
                state=snapshot(sdk)
                if not any(state["motors_enabled"]):
                    report["shutdown_confirmed"]=True
                    touched=False
                    break
                if time.monotonic()>deadline:
                    raise RuntimeError("Disable acknowledgement timed out")
    except BaseException as exc:
        report["error"]=str(exc)
        raise
    finally:
        if sdk is not None:
            if touched:
                sdk.MotionCtrl_1(1,0,0)
                report["requires_supported_shutdown"]=True
            sdk.DisconnectPort()
        sim.close()
        stream.close()
        save()


if __name__ == "__main__":
    main()
