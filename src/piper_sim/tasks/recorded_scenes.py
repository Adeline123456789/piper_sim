"""Tabletop task layouts inferred from the September connector and June straw recordings."""
import math
from copy import deepcopy

import numpy as np
import yaml

from ..config import ROOT
from ..transforms import pose_matrix
from .connector import ConnectorTask, quat_z


def pose(xy, z, quat=(1., 0., 0., 0.)):
    return {"position": [*map(float, xy), float(z)], "quaternion_wxyz": list(quat)}


def matrix(p):
    return pose_matrix(p["position"], p["quaternion_wxyz"])


class RecordedTask:
    def __init__(self, config, workcell):
        self.config = deepcopy(config)
        table = workcell["scene"]["table"]
        self.top = table["position"][2] + table["size"][2] / 2
        self.reset(config["seed"])

    def reset(self, seed=None, randomize=None):
        seed = self.config["seed"] if seed is None else int(seed)
        enabled = self.config["randomization"]["enabled"] if randomize is None else bool(randomize)
        self.layout = {"seed": seed, "randomized": enabled,
                       "poses": self.place(np.random.default_rng(seed), enabled)}
        self.hold = 0.
        return deepcopy(self.layout)

    def metrics(self, poses, dt=0.):
        values, inside = self.evaluate(poses)
        if dt:
            self.hold = self.hold + dt if inside else 0.
        return dict(values, hold_seconds=self.hold,
                    success=bool(inside and self.hold >= self.config["hold_seconds"]))

    def metadata(self):
        return {"name": self.config["name"], "config": self.config, "layout": self.layout,
                "object_pose_frame": "world metres, quaternion wxyz",
                "dimension_status": "CAD station, video-estimated task geometry and contact properties"}


class RealConnector(RecordedTask):
    dynamic = ("black_charger", "yellow_plug")

    def __init__(self, config, workcell):
        if config["target_index"] not in (1, 2, 3):
            raise ValueError("Choose one of AC outlets 1, 2, 3; module 0 is dual USB")
        source = yaml.safe_load((ROOT / "configs/tasks/connector.yaml").read_text())
        source["plugs"]["yellow_plug"].update(config["yellow_plug"])
        source["plugs"]["black_charger"].update(config["black_charger"])
        self.fixture = ConnectorTask(source, workcell)
        super().__init__(config, workcell)

    @property
    def outlets(self):
        x0, y = self.config["first_module_xy_m"]
        return np.array([[x0 + i*self.config["outlet_pitch_m"], y] for i in range(4)])

    @property
    def socket_face_z(self):
        return self.config["strip_size_m"][2] - self.config["key_projection_m"]

    def place(self, rng, enabled):
        c = self.config
        jitter = c["randomization"]["fixture_jitter_m"] * enabled
        def fixture(name):
            r = c['randomization']
            bounds = r.get(name+'_xy_bounds_m')
            xy = (rng.uniform(*np.asarray(bounds).T) if enabled and bounds is not None
                  else c[name+'_xy_m'] + rng.uniform(-jitter, jitter, 2))
            yaw_bounds = r.get(name+'_yaw_bounds_deg')
            yaw = (rng.uniform(*yaw_bounds) if enabled and yaw_bounds is not None
                   else c.get(name+'_yaw_deg', -90.))
            return pose(xy, self.top, quat_z(np.deg2rad(yaw)))
        board, strip = fixture('holder'), fixture('strip')
        poses = {"board": board, **{n: self.fixture.target_pose(n, board) for n in self.dynamic}}
        self.fixture.layout["poses"] = poses
        return {**poses, "strip": strip}

    def evaluate(self, poses):
        rel = np.linalg.inv(matrix(poses["strip"])) @ matrix(poses["yellow_plug"])
        delta = rel[:2, 3] - self.outlets[self.config["target_index"]]
        lateral = float(np.linalg.norm(delta))
        tilt = math.acos(np.clip(rel[2, 2], -1., 1.))
        yaw = math.atan2(rel[1, 0], rel[0, 0])
        gap = float(rel[2, 3] - self.socket_face_z)
        # At this clearance a yawed pin corner can miss even when its centre fits.
        fits = True
        for part in self.fixture.parts["yellow_plug"]:
            if part["name"].startswith("pin_"):
                vertices = np.asarray(part["points"])
                xy = vertices[:, :2]
                projected = vertices @ rel[:3, :3].T + rel[:3, 3]
                displacement = projected[:, :2] - self.outlets[self.config["target_index"]] - xy
                fits &= bool(np.max(np.linalg.norm(displacement, axis=1)) <= self.config["slot_clearance_m"])
        seated = fits and abs(yaw) < .085 and tilt < .085 and -.001 <= gap <= .0015
        return {"lateral_error_m": lateral, "face_gap_m": gap, "yaw_rad": yaw,
                "tilt_rad": tilt, "seated": seated}, seated


class RealStraw(RecordedTask):
    dynamic = ("cup", "straw")

    def place(self, rng, enabled):
        c = self.config
        r = c["randomization"]
        jitter = r["object_jitter_m"] * enabled
        yaw = np.deg2rad(c["straw"]["yaw_deg"] + rng.uniform(-1, 1) * r["straw_yaw_jitter_deg"] * enabled)
        # Local +Z lies along the rod. At reset it lies flat on the mat.
        a = np.sqrt(.5)
        quat = [a*np.cos(yaw/2), -a*np.sin(yaw/2), a*np.cos(yaw/2), a*np.sin(yaw/2)]
        z = self.top + c["mat_size_m"][2]
        return {"mat": pose(c["mat_xy_m"], self.top),
                "cup": pose(c["cup_xy_m"] + rng.uniform(-jitter, jitter, 2), z),
                "straw": pose(c["straw_xy_m"] + rng.uniform(-jitter, jitter, 2),
                              z + c["straw"]["radius_m"], quat)}

    def evaluate(self, poses):
        c = self.config
        cup = matrix(poses["cup"])
        rel = np.linalg.inv(cup) @ matrix(poses["straw"])
        direction, center = rel[:3, 2], rel[:3, 3]
        half = c["straw"]["length_m"] / 2
        axis_z = abs(direction[2])
        lid_z = c["cup"]["height_m"]
        depth = float(lid_z - (center[2] - half*axis_z))
        # Intersect the finite straw with the lid in the moving cup frame.
        t = (lid_z - center[2]) / direction[2] if axis_z > 1e-8 else 0.
        nearest = np.clip(t, -half, half)
        lateral = float(np.linalg.norm(center[:2] + nearest*direction[:2]))
        clearance = c["cup"]["aperture_radius_m"] - c["straw"]["radius_m"] / max(axis_z, 1e-8)
        through_aperture = axis_z > 1e-8 and abs(t) < half and lateral <= clearance
        depth = max(0., depth) if through_aperture else 0.
        inserted = bool(axis_z > .85 and c["minimum_depth_m"] <= depth < lid_z and cup[2, 2] > .9)
        return {"axis_lid_error_m": lateral, "depth_m": depth, "inserted": inserted}, inserted
