"""Run an exported ONNX policy on N parallel envs (CPU ok) and record an MP4.

The env's command manager samples its own velocity/head commands per robot,
so each duck walks its own way.

    uv run scripts/record_swarm.py --onnx policies/alpha_walking.onnx --num-envs 20
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio_ffmpeg
import mediapy
import numpy as np
import onnxruntime as ort
import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.viewer.viewer_config import ViewerConfig

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="Mjlab-Velocity-Flat-MicroDuck")
ap.add_argument("--onnx", default="policies/alpha_walking.onnx")
ap.add_argument("--num-envs", type=int, default=20)
ap.add_argument("--seconds", type=float, default=20.0)
ap.add_argument("--spacing", type=float, default=0.6)
ap.add_argument("--out", default="swarm.mp4")
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
args = ap.parse_args()

cfg = load_env_cfg(args.task, play=True)
cfg.scene.num_envs = args.num_envs
cfg.scene.env_spacing = args.spacing
cfg.viewer.origin_type = ViewerConfig.OriginType.WORLD
cfg.viewer.lookat = (0.0, 0.0, 0.1)
cfg.viewer.distance = 0.9 * args.spacing * np.sqrt(args.num_envs) + 1.0
cfg.viewer.elevation = -30.0
cfg.viewer.azimuth = 135.0
cfg.viewer.max_extra_envs = args.num_envs - 1
cfg.viewer.width, cfg.viewer.height = args.width, args.height

env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode="rgb_array")
sess = ort.InferenceSession(args.onnx)
in_name, out_name = sess.get_inputs()[0].name, sess.get_outputs()[0].name

obs, _ = env.reset()
steps = int(args.seconds / env.step_dt)
frames, falls = [], 0
for i in range(steps):
    actor = obs["actor"].cpu().numpy().astype(np.float32)
    # exported graph has a fixed [1, 61] input, so run one robot at a time
    action = np.concatenate([sess.run([out_name], {in_name: a[None]})[0] for a in actor])
    obs, _, terminated, _, _ = env.step(torch.from_numpy(action))
    falls += int(terminated.sum())
    frames.append(env.render())
    if i % 50 == 0:
        print(f"t={i * env.step_dt:5.1f}s  falls so far={falls}")

mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
mediapy.write_video(args.out, frames, fps=round(1 / env.step_dt))
print(f"Wrote {len(frames)} frames to {args.out}; {args.num_envs} robots, {falls} falls in {args.seconds:.0f}s")
