"""Single-arm, one-joint-at-a-time synchronization. Angles here are degrees.

Connecting is passive. Start and supported disable are explicit operator actions.
The other five joints hold their accepted positions; no gripper command is issued.
"""
import threading
import time

import numpy as np


class JointSync:
    def __init__(self, sdk, snapshot, bounds, *, speed=.5, excursion=15., clock=time.monotonic):
        self.sdk, self.snapshot, self.clock = sdk, snapshot, clock
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.shape != (6, 2) or not np.isfinite(self.bounds).all() or np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError("Invalid joint limits")
        if not 0 < speed <= 1 or not 0 < excursion <= 15:
            raise ValueError("Speed must be <= 1 degree/s; excursion <= 15 degrees")
        self.speed, self.excursion = speed, excursion
        self.state, self.error, self.selected = "idle", None, 0
        self.touched = False
        self.lock = threading.RLock()
        self.last_tick = clock()
        self.command = self.reference = self.desired = None

    def read(self):
        s = self.snapshot(self.sdk)
        # The legacy snapshot checks aggregate timestamps; require every frame too.
        if hasattr(self.sdk, "frame_times"):
            with self.sdk.feedback_lock:
                now = self.clock()
                required = (0x2A1, 0x2A5, 0x2A6, 0x2A7, 0x2A8, *range(0x261, 0x267))
                if any(not 0 <= now-self.sdk.frame_times.get(i, -float("inf")) < .25 for i in required):
                    raise RuntimeError("Individual CAN feedback is stale")
        return s

    def prepare(self, s):
        if any(s["motors_enabled"]) or s["gripper_enabled"]:
            raise RuntimeError("Start requires six motors and gripper disabled")
        if s["err_code"] or s["driver_faults"] or s["arm_status"] not in (0, 1) or s["ctrl_mode"] not in (0, 1):
            raise RuntimeError("Resolve arm/controller fault before starting")
        initial = np.asarray(s["joints_deg"], dtype=float)
        entry = np.clip(initial, self.bounds[:, 0], self.bounds[:, 1])
        changed = np.flatnonzero(abs(entry-initial) > 1e-6)
        if any(i not in (1, 2) for i in changed) or (len(changed) and np.any(abs(initial[1:3]) > 3)):
            raise RuntimeError("Pose outside limits; only near-zero J2/J3 entry is supported")
        if len(changed) and (abs(self.bounds[1, 0]) > 1e-9 or abs(self.bounds[2, 1]) > 1e-9):
            raise RuntimeError("Unexpected folded-entry limits")
        return initial, entry

    def start(self, expected_pose):
        with self.lock:
            if self.state != "idle":
                raise RuntimeError("Disable/reconnect before starting again")
            s = self.read()
            initial, entry = self.prepare(s)
            if hasattr(self.sdk, 'clear_disabled_session'):
                self.sdk.clear_disabled_session()
            if np.max(abs(initial-np.asarray(expected_pose))) > .3:
                raise RuntimeError("Pose changed since the start confirmation")
            self.initial, self.reference = initial, entry.copy()
            self.command = self.desired = entry.copy()
            self.sim_target = entry.copy()
            self.low = np.maximum(self.bounds[:, 0], entry-self.excursion)
            self.high = np.minimum(self.bounds[:, 1], entry+self.excursion)
            self.started = self.last_tick = self.clock()
            self.stable = None
            self.touched = True
            if s["arm_status"] == 1:
                # Explicit Start includes recovery from this session's software stop.
                self.state = "resuming"
                self.sdk.MotionCtrl_1(2, 0, 0)
            else:
                self.state = "enabling"

    def select(self, joint):
        with self.lock:
            if not 0 <= joint < 6:
                raise ValueError("Joint index must be 0..5")
            if self.state == "active" and np.max(abs(self.desired-self.command)) > .15:
                raise RuntimeError("Wait for the current target before switching joints")
            self.selected = joint
            if self.command is not None:
                self.desired = self.command.copy()
                self.sim_target = self.command.copy()

    def target(self, degrees):
        with self.lock:
            if self.state != "active":
                raise RuntimeError("Start synchronization first")
            if not np.isfinite(degrees) or not self.low[self.selected] <= degrees <= self.high[self.selected]:
                raise ValueError("Target exceeds this session's joint interval")
            self.desired[self.selected] = degrees

    def stop(self, reason="Operator stop"):
        with self.lock:
            if self.touched:
                try:
                    self.sdk.MotionCtrl_1(1, 0, 0)
                finally:
                    self.state, self.error = "stopped", reason

    def watchdog(self):
        with self.lock:
            if self.state in ("resuming", "enabling", "entry", "active") and self.clock()-self.last_tick > .3:
                self.stop("Control loop stalled for more than 300 ms")

    def request_disable(self):
        """Caller must have obtained physical-support confirmation."""
        with self.lock:
            if not self.touched:
                return
            self.stop("Supported shutdown requested")
            self.state, self.started = "disabling", self.clock()

    def tick(self, advance):
        """advance(deg[6]) returns displayed MuJoCo joints, also in degrees."""
        try:
            proposal = simulated = None
            with self.lock:
                if self.state == "active":
                    dt = min(max(self.clock()-self.last_tick, 0), .05)
                    proposal = self.sim_target.copy()
                    j = self.selected
                    proposal[j] += np.clip(self.desired[j]-proposal[j], -self.speed*dt, self.speed*dt)
                    self.sim_target = proposal.copy()
            # Renderer/physics may stall. Never block the independent stop watchdog.
            if proposal is not None:
                simulated = np.asarray(advance(proposal))
            with self.lock:
                # If the watchdog stopped during advance(), _tick must not send.
                return self._tick(proposal, simulated)
        except BaseException as exc:
            self.stop(str(exc))
            raise

    def _tick(self, target, simulated):
        now = self.clock()
        dt = min(max(now-self.last_tick, 0), .05)
        self.last_tick = now
        s = self.read()
        measured = np.asarray(s["joints_deg"])
        if self.state == "disabling":
            for i in range(1, 7):
                self.sdk.DisableArm(i)
            if not any(s["motors_enabled"]):
                self.state, self.touched = "idle", False
            elif now-self.started > 3:
                raise RuntimeError("Disable acknowledgement timed out")
            return s
        if self.state in ("idle", "stopped"):
            return s
        # Same bounded reset settling condition as home_zero's wait_reset_ready:
        # this firmware briefly emits state 5 + low-six error bits with motors off.
        reset_transient = (self.state == "resuming" and not any(s["motors_enabled"]) and
                           s["arm_status"] == 5 and s["err_code"] & ~0x3F == 0)
        if s["driver_faults"] or (s["err_code"] and not reset_transient) or s["gripper_enabled"]:
            raise RuntimeError("Driver fault, controller error, or unexpected gripper enable")
        if self.state == "resuming":
            if any(s["motors_enabled"]):
                raise RuntimeError("Unexpected enable while recovering stop")
            if s["arm_status"] == 0 and s["err_code"] == 0:
                if self.stable is None:
                    self.stable = now
                if now-self.stable > .5 and now-self.started > .8:
                    # Like home_zero: compare positions only after reset telemetry
                    # has settled. During state 5 the firmware can emit garbage q.
                    if np.max(abs(measured-self.initial)) > .3:
                        raise RuntimeError("Stable pose changed during software-stop recovery")
                    self.state, self.started = "enabling", now
            else:
                self.stable = None
            if self.state == "resuming" and now-self.started > 3:
                raise RuntimeError("Controller did not leave software stop")
            return s
        if s["arm_status"] not in ((0, 1, 6) if self.state == "enabling" else (0,)):
            raise RuntimeError("Unexpected arm stop/status")
        if np.any(measured < np.minimum(self.initial, self.low)-.7) or np.any(measured > np.maximum(self.initial, self.high)+.7):
            raise RuntimeError("Feedback left the permitted session interval")
        if np.max(abs(self.command-measured)) > 3:
            raise RuntimeError("CAN feedback differs from command by over 3 degrees")
        if self.state != "enabling" and not all(s["motors_enabled"]):
            raise RuntimeError("Motor enable lost")
        if self.state == "active":
            j = self.selected
            if target is None or simulated is None or simulated.shape != (6,) or not np.isfinite(simulated).all():
                raise RuntimeError("Invalid MuJoCo output")
            if abs(simulated[j]-target[j]) > .5 or abs(simulated[j]-self.command[j]) > self.speed*dt+.1:
                raise RuntimeError("MuJoCo joint deviated from the bounded target")
            outgoing = self.command.copy()
            # Read-back MuJoCo position is the real-arm leader, not the UI goal.
            outgoing[j] += np.clip(simulated[j]-outgoing[j], -self.speed*dt, self.speed*dt)
            if np.any(outgoing < self.low-.1) or np.any(outgoing > self.high+.1):
                raise RuntimeError("MuJoCo output exceeds session bounds")
            # MuJoCo uses soft limits. <=0.1 degree penetration must never be sent
            # past the hardware limit; larger violations stop the session.
            self.command = np.clip(outgoing, self.low, self.high)
        self.sdk.MotionCtrl_2(1, 1, 5, 0)
        self.sdk.JointCtrl(*np.rint(self.command*1000).astype(int).tolist())
        if self.state == "enabling":
            for i in range(1, 7):
                self.sdk.EnableArm(i)
            if all(s["motors_enabled"]) and s["arm_status"] == 0:
                self.state, self.started = "entry", now
            elif now-self.started > 5:
                raise RuntimeError("Motor enable timed out")
        elif self.state == "entry":
            if now-self.started >= 1.5 and np.max(abs(measured-self.reference)) <= .4:
                self.state = "active"
            elif now-self.started > 5:
                raise RuntimeError("Entry pose did not settle")
        return s
