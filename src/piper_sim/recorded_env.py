"""Gymnasium interface to the recorded scenes. All actuation is local MuJoCo."""
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .backends.mujoco import MujocoBackend
from .config import ROOT, load
from .contracts import Command, LOWER, UPPER


class RecordedPiperEnv(gym.Env):
    """Absolute joint targets in SI at 30 Hz. The reward is stable insertion (0/1)."""
    metadata = {"render_modes": [], "render_fps": 30}

    def __init__(self, task, cameras=True, max_steps=900, randomize=False):
        if task not in ("real_connector", "real_straw"):
            raise ValueError("Choose real_connector or real_straw")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        cfg = load()
        cfg["task_scene"] = ROOT / f"configs/tasks/{task}.yaml"
        self.sim = MujocoBackend(cfg, cameras=cameras)
        self.max_steps = max_steps
        self.randomize = randomize
        self.action_space = spaces.Box(LOWER, UPPER, dtype=np.float64)
        obs = {"state": spaces.Box(-np.inf, np.inf, (14,), dtype=np.float32)}
        if cameras:
            for name in self.sim.rig["cameras"]:
                obs[name] = spaces.Box(0, 255, (480, 640, 3), dtype=np.uint8)
        self.observation_space = spaces.Dict(obs)

    def _observe(self):
        obs = self.sim.read()
        return ({"state": obs.position.astype(np.float32), **obs.images},
                obs.extras["task"]["metrics"])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        layout_seed = int(self.np_random.integers(0, 2**31)) if seed is None else seed
        self.sim.reset(seed=layout_seed, randomize=(options or {}).get("randomize", self.randomize))
        self.steps = 0
        return self._observe()

    def step(self, action):
        self.sim.apply(Command(action, time.monotonic(), self.steps))
        self.sim.step()
        self.steps += 1
        obs, info = self._observe()
        success = info["success"]
        return obs, float(success), success, self.steps >= self.max_steps and not success, info

    def close(self):
        self.sim.close()
