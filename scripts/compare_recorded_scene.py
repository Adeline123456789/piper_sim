"""Pair the initial real and simulated RGB frames at the same arm state."""
import argparse
import io
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw
from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.config import ROOT, load
from piper_sim.contracts import LOWER, UPPER
from replay_recorded_scene import DATASETS, read_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=DATASETS)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = args.dataset or DATASETS[args.task]
    if args.task == "real_connector":
        ep = 1
        paths = [data/f"observations/20260908/{ep:06}/videos/cam_{cam}_rgb.mp4"
                 for cam in ("external", "left_wrist", "right_wrist")]
    else:
        ep = 19
        paths = [data/f"videos/chunk-000/{cam}/episode_{ep:06}.mp4"
                 for cam in ("image", "wrist_image_l", "wrist_image_r")]
    states, _ = read_episode(args.task, data, ep)
    cfg = load()
    cfg["task_scene"] = ROOT/f"configs/tasks/{args.task}.yaml"
    sim = MujocoBackend(cfg, cameras=True)
    try:
        if np.any(states[0] < LOWER) or np.any(states[0] > UPPER):
            raise ValueError("Reference state exceeds scene limits")
        sim.set_observation_state(states[0])
        frames = sim.read().images
    finally:
        sim.close()

    sheet = Image.new("RGB", (1200, 648), "#242727")
    draw = ImageDraw.Draw(sheet)
    for i, (name, path) in enumerate(zip(("overhead", "left_wrist", "right_wrist"), paths)):
        frame = subprocess.check_output(["ffmpeg", "-loglevel", "error", "-i", str(path),
                                         "-frames:v", "1", "-c:v", "png", "-f", "image2pipe", "-"])
        for row, (source, im) in enumerate([("Recording", Image.open(io.BytesIO(frame))),
                                            ("MuJoCo", Image.fromarray(frames[name]))]):
            y = 324*row
            draw.text((400*i+7, y+5), f"{source}  /  {name}  /  ep {ep}, frame 0", fill="white")
            sheet.paste(im.resize((400, 300), Image.Resampling.LANCZOS), (400*i, y+24))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)


if __name__ == "__main__":
    main()
