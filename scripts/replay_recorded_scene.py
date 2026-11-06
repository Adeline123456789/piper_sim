"""Play a local real episode in MuJoCo only. Targets and observed states are distinct."""
import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np

from piper_sim.config import ROOT, load
from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.contracts import Command, GRIPPER_MAX_WIDTH_M, LOWER, UPPER

DATASETS = {
    "real_connector": Path("/media/midea/F496AC8096AC4548/act_td3/connector_plugin"),
    "real_straw": Path("/media/midea/61500ece-df8d-46ce-be73-e04b2378079c/datasets/piper_insert_straw_lerobot0630"),
}


def read_episode(task, directory, episode):
    if task == "real_straw":
        import pyarrow.parquet as pq
        path = directory/f"data/chunk-{episode//1000:03}/episode_{episode:06}.parquet"
        table = pq.read_table(path, columns=["state", "actions"])
        arrays = []
        for key in table.column_names:
            values = np.asarray(table[key].to_pylist(), dtype=float)
            if values.ndim != 2 or values.shape[1] != 15:
                raise ValueError(f"Unexpected straw {key} schema")
            values = values[:, :14].copy()
            values[:, [6, 13]] *= GRIPPER_MAX_WIDTH_M
            arrays.append(values)
    else:
        path = directory/f"proprio_stats/20260908/{episode:06}/proprio_stats.h5"
        arrays = []
        with h5py.File(path) as f:
            if int(f.attrs["fps"]) != 30:
                raise ValueError("Expected 30 Hz")
            for key in ("state", "action"):
                arms = []
                for side in ("left_arm", "right_arm"):
                    group = f[f"{key}/{side}"]
                    arms.append(np.column_stack([np.deg2rad(group["joint_positions"][:]),
                                                 group["gripper_open"][:] * GRIPPER_MAX_WIDTH_M]))
                arrays.append(np.column_stack(arms))
    if not all(np.isfinite(v).all() and v.shape == arrays[0].shape for v in arrays):
        raise ValueError(f"Invalid episode arrays: {path}")
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=DATASETS)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--episode", type=int)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    ep = args.episode if args.episode is not None else (1 if args.task == "real_connector" else 19)
    states, actions = read_episode(args.task, args.dataset or DATASETS[args.task], ep)
    # Float32 serialization of the nominal 1.22 rad limit can exceed it by 3e-8.
    for values in (states[:1], actions):
        if (values < LOWER-1e-6).any() or (values > UPPER+1e-6).any():
            raise ValueError("Episode exceeds model limits. Inspect units/calibration before replay.")
    cfg = load()
    cfg["task_scene"] = ROOT/f"configs/tasks/{args.task}.yaml"
    sim = MujocoBackend(cfg, viewer=args.viewer)
    try:
        sim.set_observation_state(np.clip(states[0], LOWER, UPPER))
        inactive = []
        if args.task == "real_connector":
            for j in (0, 7):
                if np.max(np.ptp(actions[:, j:j+7], axis=0)) < 1e-8:
                    # Stationary arm in this logger can record a placeholder zero gripper command.
                    actions[:, j:j+7] = states[0, j:j+7]
                    inactive.append("left" if j == 0 else "right")
        successful_frames = 0
        for i, action in enumerate(actions):
            if not sim.is_running():
                break
            start = time.monotonic()
            sim.apply(Command(np.clip(action, LOWER, UPPER), start, i))
            sim.step()
            info = sim.read().extras["task"]["metrics"]
            successful_frames += int(info["success"])
            if args.viewer:
                time.sleep(max(0, 1/30-(time.monotonic()-start)))
        print(json.dumps({"episode": ep, "simulated_frames": i+1, "success_frames": successful_frames,
                          "held_inactive_arms": inactive, "final_metrics": info}, indent=2))
    finally:
        sim.close()


if __name__ == "__main__":
    main()
