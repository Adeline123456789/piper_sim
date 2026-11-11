"""Render original/office/seeded-training views at identical robot poses; no CAN."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw
import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from piper_sim.config import ROOT, load
from piper_sim.backends.mujoco import MujocoBackend
from build_plug_alignment import apply_feedback_for_diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, default=ROOT/'configs/tasks/real_connector.yaml')
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = yaml.safe_load(args.task.read_text())
    base.pop('appearance', None)
    office = deepcopy(base)
    office['appearance'] = deepcopy(yaml.safe_load((ROOT/'configs/tasks/real_connector.yaml').read_text())['appearance'])
    # Preserve calibration alongside the new profile, rather than depend on a mutable file.
    if office.get('camera_calibration'):
        snapshot = (args.output/'cameras.json').resolve()
        snapshot.write_text((ROOT/office['camera_calibration']).read_text())
        office['camera_calibration'] = str(snapshot)
    training = deepcopy(office)
    training['appearance']['randomization']['enabled'] = True
    training['seed'] = 23
    frames = json.loads((args.capture/'can_latest.json').read_text()) if args.capture else None
    names = ['overhead', 'right_wrist', 'left_wrist', 'assembly_side', 'white_strip_detail']
    variants = [('before', base), ('office', office), ('training_seed23', training)]
    panel = Image.new('RGB', (640*3, 504*len(names)), 'white')
    draw = ImageDraw.Draw(panel)
    report = {}
    for column, (label, cfg) in enumerate(variants):
        (args.output/(label+'.yaml')).write_text(yaml.safe_dump(cfg, sort_keys=False))
        sim = MujocoBackend(dict(load(), task_scene=cfg), cameras=True)
        try:
            if frames:
                apply_feedback_for_diagnostic(sim, frames)
            report[label] = dict(appearance=sim.read().extras['appearance'], cameras=sim.read().extras['cameras'])
            for row, name in enumerate(names):
                im = Image.fromarray(sim._render(name, 640, 480))
                im.save(args.output/(label+'_'+name+'.png'))
                draw.text((640*column+8, row*504+6), label+' / '+name, fill='black')
                panel.paste(im, (640*column, row*504+24))
        finally:
            sim.close()
    panel.save(args.output/'comparison.png')
    if args.capture:
        paired = Image.new('RGB', (1920, 1008), 'white')
        labels = ImageDraw.Draw(paired)
        for row, (real, simulated) in enumerate([('external', 'overhead'), ('right_wrist', 'right_wrist')]):
            sources = [('REAL', args.capture/real/'last.png'),
                       ('BEFORE', args.output/('before_'+simulated+'.png')),
                       ('OFFICE', args.output/('office_'+simulated+'.png'))]
            for col, (label, path) in enumerate(sources):
                labels.text((col*640+8, row*504+6), label+' / '+real, fill='black')
                paired.paste(Image.open(path), (col*640, row*504+24))
        paired.save(args.output/'real_before_office.png')
        # One empty tabletop patch: a photometric check, not a scene-wide metric.
        patch = (220, 160, 420, 220)
        means = {}
        for label, path in [('real', args.capture/'external/last.png'),
                            ('before', args.output/'before_overhead.png'),
                            ('office', args.output/'office_overhead.png')]:
            means[label] = np.asarray(Image.open(path).crop(patch)).mean(axis=(0,1)).tolist()
        report['table_patch_check'] = dict(roi_xyxy=patch, mean_rgb_0_255=means,
            interpretation='single empty table patch; not geometric or whole-scene accuracy')
    (args.output/'render_metadata.json').write_text(json.dumps(report, indent=2)+'\n')
    print(args.output/'comparison.png')


if __name__ == '__main__':
    main()
