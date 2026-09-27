"""Drive N ducks as a group in one world direction and record an MP4.

Two layers:
  * low level  — the exported ONNX walking policy, one per robot, fed the
    61-D actor obs from the mjlab env (same contract as the real runtime);
  * high level — a centralized multi-robot planner that writes each robot's
    twist command every control step:
      1. migration:   shared desired velocity v_goal along --direction-deg
      2. formation:   consensus term pulling each robot to its slot in a grid
                      formation around the group centroid
      3. safety:      decentralized control-barrier-function QP (Wang, Ames &
                      Egerstedt 2017, "Safety Barrier Certificates for
                      Collisions-Free Multirobot Systems") keeping every pair
                      at least --safe-dist apart
      4. unicycle map: world velocity -> (forward speed, yaw rate) since the
                      walking policy tracks forward/turn far better than strafe

Note: mjlab envs are separate MuJoCo worlds, so robots cannot physically
collide — separation is enforced by the planner only.

    uv run scripts/swarm_navigate.py --num-envs 20 --direction-deg 0
"""
import argparse
import math
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio_ffmpeg
import mediapy
import numpy as np
import onnxruntime as ort
import torch
from scipy.optimize import minimize

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.viewer.viewer_config import ViewerConfig

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="Mjlab-Velocity-Flat-MicroDuck")
ap.add_argument("--onnx", default="policies/alpha_walking.onnx")
ap.add_argument("--num-envs", type=int, default=20)
ap.add_argument("--direction-deg", type=float, default=0.0, help="world direction to travel (0 = +x)")
ap.add_argument("--speed", type=float, default=0.10, help="group speed actually achieved, m/s")
ap.add_argument("--track-gain", type=float, default=0.37,
                help="achieved/commanded forward speed of the policy (measured ~0.11/0.30)")
ap.add_argument("--slot-spacing", type=float, default=0.6, help="formation grid spacing, m")
ap.add_argument("--safe-dist", type=float, default=0.40,
                help="planning min distance, m (inflated above the ~0.25 m body size to absorb policy tracking error)")
ap.add_argument("--k-form", type=float, default=0.4, help="formation consensus gain")
ap.add_argument("--gamma", type=float, default=2.0, help="CBF class-K gain")
ap.add_argument("--pushes", action="store_true", help="keep the random shoves from the play cfg")
ap.add_argument("--seconds", type=float, default=30.0)
ap.add_argument("--spacing", type=float, default=0.6, help="env spawn spacing, m")
ap.add_argument("--out", default="swarm_nav.mp4")
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
args = ap.parse_args()

N = args.num_envs
VX_MAX, WZ_MAX = 0.4, 1.0  # training command ranges
v_max = VX_MAX * args.track_gain  # fastest the robots can actually go

cfg = load_env_cfg(args.task, play=True)
cfg.scene.num_envs = N
cfg.scene.env_spacing = args.spacing
# one continuous episode (play cfg would reset every robot at 20 s)
cfg.episode_length_s = args.seconds + 10.0
# spawn exactly on the env grid (training scatters +-0.5 m, which overlaps robots)
cfg.events["reset_base"].params["pose_range"]["x"] = (0.0, 0.0)
cfg.events["reset_base"].params["pose_range"]["y"] = (0.0, 0.0)
# face the travel direction at spawn, and drop the play cfg's random shoves
# (push_robot fires every 0.5-1 s — a robustness test, not a coordination demo)
_th = math.radians(args.direction_deg)
cfg.events["reset_base"].params["pose_range"]["yaw"] = (_th, _th)
if not args.pushes:
    cfg.events.pop("push_robot", None)
cfg.viewer.origin_type = ViewerConfig.OriginType.WORLD
cfg.viewer.distance = 0.9 * args.spacing * math.sqrt(N) + 1.2
cfg.viewer.elevation = -35.0
cfg.viewer.azimuth = args.direction_deg + 180.0 + 45.0  # look along the travel direction
cfg.viewer.max_extra_envs = N - 1
cfg.viewer.width, cfg.viewer.height = args.width, args.height

env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode="rgb_array")
robot = env.scene["robot"]

# Take over the command terms: freeze their internal resampling and write
# them ourselves. Head/body pose commands stay at zero (neutral head).
twist = env.command_manager.get_term("twist")
head = env.command_manager.get_term("head_pose")
body = env.command_manager.get_term("body_pose")
for term in (twist, head, body):
    term.compute = lambda dt: None
    term.command.zero_()

sess = ort.InferenceSession(args.onnx)
in_name, out_name = sess.get_inputs()[0].name, sess.get_outputs()[0].name

# Formation: the spawn grid itself (env origins around their centroid), so the
# group starts in formation and simply translates.
th = math.radians(args.direction_deg)
d_goal = np.array([math.cos(th), math.sin(th)])
origins = env.scene.env_origins.cpu().numpy()[:, :2]
slots = (origins - origins.mean(0)) * (args.slot_spacing / args.spacing)
v_goal = args.speed * d_goal


def yaw_of(q):  # q = (w, x, y, z)
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def assign_slots(p):
    """Greedy nearest-slot assignment so robots don't cross the group to reach theirs."""
    c = p.mean(0)
    free, out = list(range(N)), np.zeros(N, dtype=int)
    order = np.argsort(-(p - c) @ d_goal)  # front-most robots pick first
    for i in order:
        j = min(free, key=lambda k: np.linalg.norm(c + slots[k] - p[i]))
        out[i] = j
        free.remove(j)
    return out


def cbf_filter(i, p, u_nom, u_prev):
    """min ||u - u_nom||^2  s.t.  dp_ij . u >= -gamma/4 * h_ij + dp_ij . u_j/2 ... per neighbor, |u| <= v_max.

    Decentralized split of  2 dp^T (u_i - u_j) >= -gamma h,  h = |dp|^2 - Ds^2:
    each robot takes half, using its neighbor's last velocity as the estimate of u_j.
    """
    cons = []
    for j in range(N):
        if j == i:
            continue
        dp = p[i] - p[j]
        h = dp @ dp - args.safe_dist ** 2
        if np.linalg.norm(dp) > 3 * args.safe_dist:
            continue
        b = -args.gamma * h / 4 + dp @ u_prev[j] / 2
        cons.append({"type": "ineq", "fun": lambda u, dp=dp, b=b: dp @ u - b, "jac": lambda u, dp=dp: dp})
    cons.append({"type": "ineq", "fun": lambda u: v_max ** 2 - u @ u, "jac": lambda u: -2 * u})
    if len(cons) == 1 and u_nom @ u_nom <= v_max ** 2:
        return u_nom
    res = minimize(lambda u: (u - u_nom) @ (u - u_nom), u_nom, jac=lambda u: 2 * (u - u_nom),
                   constraints=cons, method="SLSQP", options={"maxiter": 50})
    return res.x if res.success else np.zeros(2)


obs, _ = env.reset()
steps = int(args.seconds / env.step_dt)
fps = round(1 / env.step_dt)
frames, falls, min_dist = [], 0, np.inf
u_prev = np.zeros((N, 2))
slot_of = None
cam = env._offline_renderer._cam if env._offline_renderer else None

for k in range(steps):
    pos = robot.data.root_link_pos_w.cpu().numpy()
    p = pos[:, :2]
    yaw = np.array([yaw_of(q) for q in robot.data.root_link_quat_w.cpu().numpy()])
    if slot_of is None or k % fps == 0:  # re-assign once a second
        slot_of = assign_slots(p)
    c = p.mean(0)
    if k == 0:
        c0 = c.copy()

    cmd = np.zeros((N, 3), dtype=np.float32)
    u_all = np.zeros((N, 2))
    for i in range(N):
        u_nom = v_goal + args.k_form * (c + slots[slot_of[i]] - p[i])
        u = cbf_filter(i, p, u_nom, u_prev)
        u_all[i] = u
        # unicycle: turn toward u, walk forward in proportion to alignment
        speed = np.linalg.norm(u)
        if speed < 0.02:
            continue
        err = math.atan2(u[1], u[0]) - yaw[i]
        err = (err + math.pi) % (2 * math.pi) - math.pi
        cmd[i, 0] = np.clip(speed * max(math.cos(err), 0.0) / args.track_gain, 0.0, VX_MAX)
        cmd[i, 2] = np.clip(2.0 * err, -WZ_MAX, WZ_MAX)
    u_prev = u_all
    twist.command[:] = torch.from_numpy(cmd)

    dists = np.linalg.norm(p[:, None] - p[None], axis=-1) + np.eye(N) * 1e9
    min_dist = min(min_dist, dists.min())

    actor = obs["actor"].cpu().numpy().astype(np.float32)
    # exported graph has a fixed [1, 61] input, so run one robot at a time
    action = np.concatenate([sess.run([out_name], {in_name: a[None]})[0] for a in actor])
    obs, _, terminated, _, _ = env.step(torch.from_numpy(action))
    falls += int(terminated.sum())

    if cam is not None:  # follow the group
        cam.lookat[:] = [c[0], c[1], 0.1]
    frames.append(env.render())
    if k % fps == 0:
        spread = np.linalg.norm(p - (c + slots[slot_of]), axis=1).mean()
        print(f"t={k * env.step_dt:5.1f}s  centroid=({c[0]:+.2f},{c[1]:+.2f})  "
              f"formation err={spread:.3f} m  min pair dist={dists.min():.3f} m  falls={falls}")

p = robot.data.root_link_pos_w.cpu().numpy()[:, :2]
mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
mediapy.write_video(args.out, frames, fps=fps)
print(f"Wrote {len(frames)} frames to {args.out}")
travel = p.mean(0) - c0
print(f"Summary: {N} robots, direction {args.direction_deg:.0f} deg | group moved {travel @ d_goal:.2f} m along it, "
      f"{abs(travel @ np.array([-d_goal[1], d_goal[0]])):.2f} m sideways | falls={falls} | min pair distance={min_dist:.3f} m")
