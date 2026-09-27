"""Record an offscreen MP4 of scripts/infer_policy.py driving the sim.

Swaps mujoco.viewer.launch_passive for an offscreen renderer and replays a
scripted keypress schedule instead of reading the terminal. All other CLI
args pass straight through to infer_policy.py, e.g.

    uv run scripts/record_demo.py --out demo.mp4 -- \
        --walking policies/alpha_walking.onnx --standing policies/alpha_stand.onnx --new-cmd-obs
"""
import argparse
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio_ffmpeg
import mediapy
import mujoco
import mujoco.viewer

# (seconds since start, key) — see the keyboard help in infer_policy.py
SCHEDULE = [
    (3.0, "up"),     # walk forward
    (9.0, "a"),      # turn left while walking
    (12.0, " "),     # stop
    (14.0, "left"),  # strafe left
    (17.0, " "),     # stop
    (18.5, "p"),     # random push -> recovery
    (21.0, "up"),    # walk forward again
    (25.0, " "),
]

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="demo.mp4")
ap.add_argument("--duration", type=float, default=27.0)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--width", type=int, default=960)
ap.add_argument("--height", type=int, default=540)
ours, rest = ap.parse_known_args()
if rest and rest[0] == "--":
    rest = rest[1:]

T0 = time.time()
frames = []


class RecordingViewer:
    def __init__(self, model, data):
        self.model, self.data = model, data
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, ours.width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, ours.height)
        self.renderer = mujoco.Renderer(model, ours.height, ours.width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = 1
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 0.9, -18, 135
        self.user_scn = mujoco.MjvScene(model, maxgeom=100)
        self._next = 0.0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.renderer.close()

    def is_running(self):
        return time.time() - T0 < ours.duration

    def sync(self):
        t = time.time() - T0
        if t >= self._next:
            self._next += 1.0 / ours.fps
            self.renderer.update_scene(self.data, self.cam)
            frames.append(self.renderer.render().copy())


mujoco.viewer.launch_passive = lambda model, data, **kw: RecordingViewer(model, data)

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
import infer_policy  # noqa: E402

pending = list(SCHEDULE)


def scripted_keys(self):
    keys = []
    while pending and time.time() - T0 >= pending[0][0]:
        keys.append(pending.pop(0)[1])
    return keys


infer_policy.TerminalInput.get_keys = scripted_keys
infer_policy.mujoco.viewer.launch_passive = mujoco.viewer.launch_passive

sys.argv = ["infer_policy.py", *rest]
try:
    infer_policy.main()
finally:
    if frames:
        mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
        mediapy.write_video(ours.out, frames, fps=ours.fps)
        print(f"Wrote {len(frames)} frames to {ours.out}")
