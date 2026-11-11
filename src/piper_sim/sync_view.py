"""Native Control changes and a feedback-only ghost for the MuJoCo viewer."""
import mujoco
import numpy as np


class NativeJointInput:
    def __init__(self, position, offset):
        self.offset = offset
        self.reset(position)

    def reset(self, position):
        self.expected = np.array(position, dtype=float).copy()

    def changed(self, controls):
        values = np.array(controls, dtype=float)
        if values.shape != (14,) or not np.isfinite(values).all():
            raise ValueError("Invalid Control input")
        changed = np.flatnonzero(abs(values-self.expected) > 1e-7)
        if not len(changed):
            return None
        if any(not self.offset <= i < self.offset+6 for i in changed):
            raise ValueError("Only connected arm J1..J6 sliders are active; grippers locked")
        if len(changed) != 1:
            raise ValueError("Change one joint at a time; Clear all is not a home command")
        i = int(changed[0])
        return i-self.offset, float(np.rad2deg(values[i]))


class NativeWorkcellInput(NativeJointInput):
    """One native slider at a time, on either explicitly connected arm."""
    def __init__(self, position, sides):
        self.sides = tuple(sides)
        self.reset(position)

    def changed(self, controls):
        values = np.array(controls, dtype=float)
        if values.shape != (14,) or not np.isfinite(values).all():
            raise ValueError('Invalid Control input')
        changed = np.flatnonzero(abs(values-self.expected) > 1e-7)
        if not len(changed):
            return None
        if len(changed) != 1:
            raise ValueError('Change one joint at a time; Clear all is not a home command')
        index = int(changed[0]); side = 'left' if index < 7 else 'right'
        joint = index % 7
        if side not in self.sides or joint == 6:
            raise ValueError('Only connected arm J1..J6 sliders are active; grippers locked')
        return side, joint, float(np.rad2deg(values[index]))


class FeedbackGhost:
    def __init__(self, sim, side):
        self.sim, self.side, self.enabled = sim, side, True
        self.data = mujoco.MjData(sim.model)
        self.scene = mujoco.MjvScene(sim.model, maxgeom=max(2000, sim.model.ngeom+50))
        self.tip = sim.model.body(f"{side}/gripper_center").id

    def update(self, raw_degrees, clear=True):
        model, sim = self.sim.model, self.sim
        self.data.qpos[:] = sim.data.qpos
        # Raw signed feedback, even outside nominal limits: visualization only.
        self.data.qpos[sim.indices[self.side]["qpos"][:6]] = np.deg2rad(raw_degrees)
        mujoco.mj_forward(model, self.data)
        gap = np.linalg.norm(self.data.xpos[self.tip]-sim.data.xpos[self.tip])*1000
        user = sim.viewer.user_scn
        if clear:
            user.ngeom = 0
        if not self.enabled:
            return float(gap)
        mujoco.mjv_updateScene(model, self.data, sim.viewer.opt, None, sim.viewer.cam,
                              mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        for source in self.scene.geoms[:self.scene.ngeom]:
            if source.objtype != mujoco.mjtObj.mjOBJ_GEOM or source.objid < 0:
                continue
            body = model.body(int(model.geom_bodyid[source.objid])).name
            if not body.startswith(self.side+"/"):
                continue
            if user.ngeom >= user.maxgeom:
                raise RuntimeError("Feedback ghost exceeds viewer geometry capacity")
            dest = user.geoms[user.ngeom]
            mujoco.mjv_initGeom(dest, source.type, source.size, source.pos, source.mat.ravel(),
                               np.array([0., .85, 1., .4], dtype=np.float32))
            dest.dataid = source.dataid
            dest.matid = -1
            dest.emission = .3
            dest.transparent = 1
            user.ngeom += 1
        return float(gap)
