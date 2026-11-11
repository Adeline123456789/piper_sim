"""Office appearance presets and episode-level visual variation.

Inspired by LIBERO's separate table visual/collision assets and RoboTwin's
lighting/camera randomization. Uses MuJoCo rasterization, not SAPIEN ray tracing.
This module changes no contacts, masses, fixture poses or joint limits.
"""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .mujoco import nums


def office_appearance(root):
    asset, world, visual = root.find('asset'), root.find('worldbody'), root.find('visual')
    quality = visual.find('quality')
    if quality is None:
        quality = ET.SubElement(visual, 'quality')
    quality.attrib.update(shadowsize='4096', offsamples='4', numslices='48', numstacks='24')
    head = visual.find('headlight')
    head.attrib.update(ambient='.30 .30 .30', diffuse='.10 .10 .10', specular='.025 .025 .025')
    for light in list(world.findall('light')):
        world.remove(light)
    # Four low-energy overhead samples approximate an extended ceiling fixture.
    # They produce overlapping low-contrast shadows, not a ray-traced area light.
    for i, (x, y) in enumerate(((.24,-.06),(.24,.06),(.36,-.06),(.36,.06))):
        ET.SubElement(world, 'light', name=f'office_ceiling_{i}', pos=nums([x,y,1.65]),
                      dir='0 0 -1', directional='false', cutoff='75', exponent='1',
                      diffuse='.035 .035 .035', ambient='0 0 0', specular='.035 .035 .035',
                      attenuation='1 0 0', castshadow='true')
    ET.SubElement(world, 'light', name='office_window_fill', pos='1.1 -.8 1.3',
                  dir='-.4 .2 -1', directional='true', diffuse='.19 .20 .21',
                  ambient='0 0 0', specular='.04 .04 .04', castshadow='false')
    ET.SubElement(world, 'light', name='office_front_fill', pos='-.8 .5 1.2',
                  dir='.5 -.2 -1', directional='true', diffuse='.16 .16 .155',
                  ambient='0 0 0', specular='.025 .025 .025', castshadow='false')

    for name, specular, shininess in [('plastic',.24,.20), ('paint',.16,.12),
                                      ('metal',.42,.38), ('rubber',.04,.04)]:
        ET.SubElement(asset, 'material', name='office/'+name, specular=str(specular),
                      shininess=str(shininess), reflectance='0')
    for material in asset.findall('material'):
        name = material.get('name', '')
        if name in ('desktop', 'office_floor'):
            material.attrib.update(specular='.04', shininess='.08', reflectance='0')
        elif name == 'silver_grain':
            material.attrib.update(specular='.28', shininess='.22', reflectance='0')
    for body in world.iter('body'):
        name = body.get('name', '')
        for geom in body.findall('geom'):
            rgba = np.fromstring(geom.get('rgba', '1 1 1 1'), sep=' ')
            if rgba[3] == 0 or geom.get('material'):
                continue
            if name == 'task/yellow_plug' and rgba[0] > .7 and rgba[1] > .4:
                geom.set('material', 'office/plastic')
            elif name in ('task/strip', 'task/black_charger'):
                geom.set('material', 'office/plastic')
            elif name.startswith(('left/', 'right/')):
                geom.set('material', 'office/rubber' if rgba[:3].max() < .12 else 'office/metal')
            else:
                geom.set('material', 'office/paint')
    # Smooth shading affects only normals; preserve the convex collision hulls.
    for name in ('plug/grip', 'plug/entry_rim'):
        asset.find(f"mesh[@name='{name}']").set('smoothnormal', 'true')


class EpisodeAppearance:
    """Reset from a baseline on every episode; never accumulate camera drift."""
    def __init__(self, model, rig, config):
        self.model, self.config = model, config
        self.camera_ids = {name: model.camera(name).id for name in rig['cameras']}
        self.baseline = {key: getattr(model, key).copy() for key in
                         ('light_diffuse', 'light_specular', 'geom_rgba', 'cam_pos', 'cam_quat')}
        self.sample = {}

    def reset(self, seed):
        m = self.model
        for key, value in self.baseline.items():
            getattr(m, key)[:] = value
        setting = self.config.get('randomization', {})
        enabled = setting.get('enabled', False)
        self.sample = dict(profile=self.config['profile'], seed=int(seed), randomized=bool(enabled),
                           renderer='MuJoCo OpenGL; approximate ceiling lighting', cameras={})
        if not enabled:
            return
        rng = np.random.default_rng(seed)
        gain = rng.uniform(*setting.get('light_gain', [.90, 1.10]))
        tint = rng.uniform(-.025, .025, 3) + 1
        m.light_diffuse[:] *= gain * tint
        m.light_specular[:] *= gain
        # Small albedo variation preserves each object's semantic colour.
        albedo = rng.uniform(.97, 1.03)
        m.geom_rgba[:, :3] = np.clip(m.geom_rgba[:, :3] * albedo, 0, 1)
        translation = setting.get('camera_translation_m', .002)
        angle = np.deg2rad(setting.get('camera_rotation_deg', .4))
        for name, cid in self.camera_ids.items():
            delta = rng.uniform(-translation, translation, 3)
            axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
            rotation = float(rng.uniform(-angle, angle))
            dq = np.empty(4); mujoco.mju_axisAngle2Quat(dq, axis, rotation)
            result = np.empty(4); mujoco.mju_mulQuat(result, self.baseline['cam_quat'][cid], dq)
            m.cam_pos[cid] += delta; m.cam_quat[cid] = result
            self.sample['cameras'][name] = dict(parent_translation_delta_m=delta.tolist(),
                                               local_mujoco_rotation_rad=rotation,
                                               local_mujoco_rotation_axis=axis.tolist())
        self.sample.update(light_gain=float(gain), light_rgb_tint=tint.tolist(), albedo_gain=float(albedo))
