"""Contact geometry and video-mode cameras for the two recorded tabletop tasks."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from ..camera_models import load_profiles, stream_rgb
from ..config import ROOT
from . import mujoco as connector_mj
from .clean_table_meshes import vessel
from .mujoco import nums
from .recorded_scenes import RealConnector


def configure_cameras(model, rig, cfg, task):
    profiles, _ = load_profiles(cfg)
    for name, cam in rig["cameras"].items():
        stream = {"mode": "crop_nominal", "resolution": [640, 480]}
        rgb = stream_rgb(profiles["models"][cam["model"]], stream)
        if name == "overhead":
            # Effective image translation from the two observed table edges.
            # Keep the physical CAD sensor mount untouched.
            k = rgb["intrinsic_matrix"]
            k[1][2] += task.config["overhead_row_offset_px"]
            corrected = dict(stream, mode="calibrated_pinhole", intrinsic_matrix=k,
                             source="nominal crop, effective row offset from tabletop photo")
            rgb = stream_rgb(profiles["models"][cam["model"]], corrected)
            rgb["calibration_status"] = "table-edge image offset estimate; device K unmeasured"
        cam["rgb"] = rgb
        k = np.asarray(rgb["intrinsic_matrix"])
        i = model.camera(name).id
        if name in ("left_wrist", "right_wrist") and isinstance(task, RealConnector):
            side = name.split("_")[0]
            delta_y = task.config.get("arm_outward_offset_m", 0.) * (1 if side == "left" else -1)
            for key in ("reference_world_pose", "front_reference_world_pose"):
                cam[key][1][3] += delta_y
            rig["mounts"][side]["reference_link6_world"][1][3] += delta_y
        if name == "right_wrist" and "right_wrist_optical_rotvec" in task.config:
            import mujoco
            from ..kinematics import rotation
            v = np.array(task.config["right_wrist_optical_rotvec"])
            angle = np.linalg.norm(v)
            delta = rotation(v.copy(), angle) if angle else np.eye(3)
            mat = np.empty(9)
            mujoco.mju_quat2Mat(mat, np.array(cam["rotation_wxyz_ros"]))
            optical = mat.reshape(3,3) @ delta
            q = np.empty(4)
            mujoco.mju_mat2Quat(q, optical.ravel())
            cam["rotation_wxyz_ros"] = q.tolist()
            reference = np.array(cam["reference_world_pose"])
            reference[:3,:3] = reference[:3,:3] @ delta
            cam["reference_world_pose"] = reference.tolist()
            cam["extrinsic_source"] = "CAD with recorded connector RGB orientation fit"
            mujoco.mju_mat2Quat(model.cam_quat[i], (optical @ np.diag([1.,-1.,-1.])).ravel())
        model.cam_resolution[i] = [640, 480]
        model.cam_intrinsic[i] = [k[0, 0]/640, k[1, 1]/480,
                                  (320-k[0, 2]-.5)/640, (k[1, 2]+.5-240)/480]


def box(body, size, pos, color, **attrs):
    return ET.SubElement(body, "geom", type="box", size=nums(np.asarray(size)/2),
                         pos=nums(pos), rgba=nums(color), **attrs)


def mesh_geom(asset, body, name, vertices, faces, color, **attrs):
    ET.SubElement(asset, "mesh", name=name, vertex=nums(vertices), face=nums(faces), inertia="shell")
    ET.SubElement(body, "geom", name=name, type="mesh", mesh=name, rgba=nums(color), **attrs)


def body_at(world, name, task):
    p = task.layout["poses"][name]
    return ET.SubElement(world, "body", name="task/"+name, pos=nums(p["position"]),
                         quat=nums(p["quaternion_wxyz"]))


def add_straw_scene(asset, world, task):
    c = task.config
    mat = body_at(world, "mat", task)
    size = c["mat_size_m"]
    ET.SubElement(asset, "texture", name="woven_mat", type="2d",
                  file=str(ROOT/"assets/tasks/straw/mat.png"))
    ET.SubElement(asset, "material", name="woven_mat", texture="woven_mat", specular=".04")
    box(mat, size, [0, 0, size[2]/2], [1, 1, 1, 1], material="woven_mat", friction="1 .003 .0001")

    cup = body_at(world, "cup", task)
    ET.SubElement(cup, "freejoint", name="task/cup/free")
    spec = c["cup"]
    h, r0, r1 = spec["height_m"], spec["base_radius_m"], spec["lid_radius_m"]
    mass = spec["mass_kg"]
    inertia = mass*(3*r1*r1+h*h)/12
    ET.SubElement(cup, "inertial", pos=nums([0, 0, .44*h]), mass=str(mass),
                  diaginertia=nums([inertia, inertia, mass*r1*r1/2]))
    ET.SubElement(cup, "geom", type="cylinder", size=nums([r0, .0015]), pos="0 0 .0015",
                  rgba=".025 .23 .32 1")
    profiles = [
        ("cup_wall", [[r0, 0], [r1, h-.004], [r1-.002, h-.004], [r0-.002, .003]],
         [.025, .23, .32, 1]),
        ("lid", [[r1, h-.004], [r1, h], [spec["aperture_radius_m"], h],
                 [spec["aperture_radius_m"], h-.004]], [.72, .70, .58, 1])]
    for name, profile, color in profiles:
        for part in vessel(name, profile):
            collision = part["collision"]
            mesh_geom(asset, cup, part["name"], part["points"], part["faces"],
                      [0, 0, 0, 0] if collision else color,
                      group="3" if collision else "1", contype=str(int(collision)),
                      conaffinity=str(int(collision)), friction=".8 .003 .0001")

    straw = body_at(world, "straw", task)
    ET.SubElement(straw, "freejoint", name="task/straw/free")
    spec = c["straw"]
    r, length, mass = spec["radius_m"], spec["length_m"], spec["mass_kg"]
    inertia = mass*(3*r*r+length**2)/12
    ET.SubElement(straw, "inertial", pos="0 0 0", mass=str(mass),
                  diaginertia=nums([inertia, inertia, mass*r*r/2]))
    ET.SubElement(straw, "geom", type="cylinder", size=nums([r, length/2]),
                  rgba=".018 .28 .62 1", contype="0", conaffinity="0")
    contact = dict(priority="1", condim="6", friction=".8 .001 .00015",
                   solref=".005 1", solimp=".97 .995 .0005", rgba="0 0 0 0", group="3")
    # Analytic endpoints support both ends on the mat. A long cylinder/box
    # pair alone yields a single unstable point on this MuJoCo version.
    ET.SubElement(straw, "geom", type="capsule", size=nums([r-.0001, length/2-r]), **contact)
    for sign in (-1, 1):
        ET.SubElement(straw, "geom", type="sphere", size=str(r),
                      pos=nums([0, 0, sign*(length/2-r)]), **contact)
    for sign in (-1, 1):
        ET.SubElement(straw, "geom", type="cylinder", size=nums([.74*r, .00005]),
                      pos=nums([0, 0, sign*(length/2+.00003)]), rgba=".015 .08 .16 1",
                      contype="0", conaffinity="0")


def add_recorded_background(root, config):
    assets, world = root.find("asset"), root.find("worldbody")
    room_dir = ROOT / "assets/tasks/room"
    for name, file in [("floor", "floor.png"), ("background", config["name"]+".png")]:
        ET.SubElement(assets, "texture", name=name, type="2d", file=str(room_dir/file))
        ET.SubElement(assets, "material", name=name, texture=name, specular="0",
                      texrepeat="8 8" if name == "floor" else "1 1")
    ground = world.find("geom[@name='ground']")
    ground.attrib.update(pos="0 0 -.73", material="floor", rgba="1 1 1 1", size="5 5 .05")
    # Fixed distant photo plane. The two reference wrist poses differ between sessions.
    bg = config["room_backdrop"]
    ET.SubElement(world, "geom", type="box", size="2.36457 1.06617 .001",
                  pos=nums(bg["pos"]), quat=nums(bg["quat"]), material="background",
                  contype="0", conaffinity="0")


def compose_model(path, task):
    if isinstance(task, RealConnector):
        root = ET.fromstring(connector_mj.compose_model(path, task.fixture))
        from .connector_details import add_connector_details
        add_connector_details(root, body_at(root.find("worldbody"), "strip", task), task)
    else:
        path = Path(path)
        root = ET.parse(path).getroot()
        compiler = root.find("compiler")
        compiler.set("meshdir", str((path.parent / compiler.get("meshdir", ".")).resolve()))
        world = root.find("worldbody")
        world.remove(world.find("body[@name='object']"))
        root.remove(root.find("keyframe"))
        add_straw_scene(root.find("asset"), world, task)
    # Soft indoor light: the source CAD preset saturates the white table and lid.
    world = root.find("worldbody")
    if isinstance(task, RealConnector):
        from .connector_workstation import add_workstation
        add_workstation(root, task)
        if task.config.get("workstation_background", "recording") == "recording":
            add_recorded_background(root, task.config)
    else:
        add_recorded_background(root, task.config)
        for side in (-1, 1):
            box(world, [.14, .20, .008], [.015, side*.30, task.top+.004], [.045,.05,.045,1],
                contype="0", conaffinity="0")
    for light in world.findall("light"):
        light.set("diffuse", ".38 .38 .38")
        light.set("ambient", ".09 .09 .09")
        light.set("specular", ".06 .06 .06")
        light.set("castshadow", "false")
    ET.SubElement(root.find("visual"), "headlight", ambient=".19 .19 .19",
                  diffuse=".20 .20 .20", specular=".05 .05 .05")
    if isinstance(task, RealConnector):
        key = world.find("light")
        key.attrib.update(diffuse=".12 .12 .12", dir=".28 .18 -1", castshadow="true")
        world.findall("light")[1].set("diffuse", ".49 .49 .49")
        ET.SubElement(world, "light", pos="-1 0 0.8", dir="1 0 -.5", directional="true",
                      diffuse=".31 .31 .31", specular=".07 .07 .07", castshadow="false")
    return ET.tostring(root, encoding="unicode")


def reset_task(model, data, task):
    for name, p in task.layout["poses"].items():
        if name in task.dynamic:
            joint = model.joint(f"task/{name}/free")
            q = joint.qposadr[0]
            data.qpos[q:q+7] = [*p["position"], *p["quaternion_wxyz"]]
            data.qvel[joint.dofadr[0]:joint.dofadr[0]+6] = 0
        else:
            body = model.body("task/"+name)
            model.body_pos[body.id] = p["position"]
            model.body_quat[body.id] = p["quaternion_wxyz"]


def task_poses(data, task):
    return {name: {"position": data.body("task/"+name).xpos.tolist(),
                   "quaternion_wxyz": data.body("task/"+name).xquat.tolist()}
            for name in task.layout["poses"]}
