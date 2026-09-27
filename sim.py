"""Shared-world MuJoCo sim: N Microducks + one throwable ball in ONE world.

Each duck runs the pretrained walking/standing ONNX policies (61-D obs, same
contract as microduck_rl/scripts/infer_policy.py) on BAM XL330 actuators.
The only thing a caller controls per duck is a twist command (vx, vy, wz) —
the same interface a real robot's runtime takes.
"""
import base64
import math
import os
import sys

import mujoco
import numpy as np
import onnxruntime as ort

# pollen-robotics/microduck_rl checkout (robot model, BAM actuator, policies); sibling dir by default
RL_DIR = os.environ.get("MICRODUCK_RL_DIR",
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "microduck_rl"))
sys.path.insert(0, os.path.join(RL_DIR, "scripts"))
from infer_policy import (  # noqa: E402  (keep constants in sync with the repo)
    BAM_KP_FW, BAM_STIFF_SOLIMP_FRICTION, BAM_STIFF_SOLREF_FRICTION, BAM_VIN_MIN,
    DEFAULT_POSE, load_bam_model,
)

ROBOT_XML = os.path.join(RL_DIR, "src/mjlab_microduck/robot/microduck/robot_walk.xml")
POLICY_DIR = os.path.join(RL_DIR, "policies")
BALL_RADIUS = 0.035
TIMESTEP = 0.005
DECIMATION = 4  # 50 Hz policy
STAND_Z = 0.125
SWITCH_THRESHOLD = 0.05  # |twist| below this -> standing policy


def _quat_rotate_inverse(q, v):
    w, xyz = q[0], q[1:4]
    t = np.cross(xyz, v) * 2
    return v - w * t + np.cross(xyz, t)


def build_model(n, spacing=0.5):
    world = mujoco.MjSpec()
    world.option.timestep = TIMESTEP
    world.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                             contype=1, conaffinity=1)
    cols = math.ceil(math.sqrt(n))
    for i in range(n):
        r, c = divmod(i, cols)
        child = mujoco.MjSpec.from_file(ROBOT_XML)
        frame = world.worldbody.add_frame(pos=[r * spacing, (c - (cols - 1) / 2) * spacing, 0])
        frame.attach_body(child.body("trunk_base"), f"d{i}_", "")
    ball = world.worldbody.add_body(name="ball", pos=[-0.5, 0, BALL_RADIUS])
    ball.add_freejoint(name="ball_free")
    # collides with the floor only (bit 2 vs floor's conaffinity) — pickup is virtual
    ball.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_RADIUS, 0, 0], mass=0.03,
                  rgba=[1.0, 0.35, 0.1, 1], contype=2, conaffinity=1, solref=[0.02, 0.3],
                  friction=[0.8, 0.02, 0.012], condim=6)  # rolling friction: stops like on carpet

    # BAM: position actuators -> torque motors; mirrors infer_policy.load_mujoco_with_bam
    bam_model = load_bam_model(BAM_KP_FW, 7.4, None)
    force_limit = bam_model.actuator.vin * bam_model.kt.value / bam_model.R.value
    act_names = []
    for act in world.actuators:
        tgt = act.target.name if hasattr(act.target, "name") else str(act.target)
        act.set_to_motor()
        act.forcelimited = True
        act.forcerange = (-force_limit, force_limit)
        act.ctrllimited = False
        act.gear = [1.0, 0, 0, 0, 0, 0]
        act_names.append(act.name)
        j = world.joint(tgt)
        j.damping = np.zeros((3, 1))
        j.frictionloss = 0.0
        j.solref_friction = BAM_STIFF_SOLREF_FRICTION
        j.solimp_friction = BAM_STIFF_SOLIMP_FRICTION
    model = world.compile()
    return model, bam_model, act_names


class Duck:
    """Index bookkeeping + observation for one duck inside the shared model."""

    def __init__(self, model, i):
        p = f"d{i}_"
        jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, p + n)  # noqa: E731
        self.i = i
        self.free_q = int(model.jnt_qposadr[jid("trunk_base_freejoint")])
        self.free_v = int(model.jnt_dofadr[jid("trunk_base_freejoint")])
        self.trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, p + "trunk_base")
        gyro = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, p + "imu_ang_vel")
        self.gyro_adr = int(model.sensor_adr[gyro])
        acts = [a for a in range(model.nu)
                if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a).startswith(p)]
        self.act_ids = np.array(acts)
        self.q_idx = np.array([model.jnt_qposadr[model.actuator_trnid[a, 0]] for a in acts])
        self.v_idx = np.array([model.jnt_dofadr[model.actuator_trnid[a, 0]] for a in acts])
        self.last_action = np.zeros(14, dtype=np.float32)
        self.twist = np.zeros(3, dtype=np.float32)
        self.fallen_t = 0.0

    def obs(self, data):
        quat = data.xquat[self.trunk]
        return np.concatenate([
            data.sensordata[self.gyro_adr:self.gyro_adr + 3],
            _quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0])),
            data.qpos[self.q_idx] - DEFAULT_POSE,
            data.qvel[self.v_idx],
            self.last_action,
            self.twist, np.zeros(10),  # head_pose(4) + body_pose(6) commands stay neutral
        ]).astype(np.float32)

    def pos(self, data):
        return data.qpos[self.free_q:self.free_q + 3]

    def yaw(self, data):
        w, x, y, z = data.qpos[self.free_q + 3:self.free_q + 7]
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def upright(self, data):
        return _quat_rotate_inverse(data.xquat[self.trunk], np.array([0.0, 0.0, -1.0]))[2] < -0.6

    def place(self, data, x, y, yaw):
        q = self.free_q
        data.qpos[q:q + 3] = [x, y, STAND_Z]
        data.qpos[q + 3:q + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        data.qvel[self.free_v:self.free_v + 6] = 0
        data.qpos[self.q_idx] = DEFAULT_POSE
        data.qvel[self.v_idx] = 0
        self.last_action[:] = 0


class SwarmSim:
    def __init__(self, n=5, spacing=0.5):
        from bam.mujoco import MujocoController

        self.model, bam_model, act_names = build_model(n, spacing)
        self.data = mujoco.MjData(self.model)
        self.ducks = [Duck(self.model, i) for i in range(n)]
        for d in self.ducks:
            p = d.pos(self.data).copy()
            d.place(self.data, p[0], p[1], 0.0)
        self.bam = MujocoController(bam_model, act_names, self.model, self.data,
                                    vin_drop_gain=0.1, vin_min=BAM_VIN_MIN)
        self.bam.reset(self.data.qpos)
        mujoco.mj_forward(self.model, self.data)
        # actuator order in the controller == model actuator order
        self.walk = ort.InferenceSession(os.path.join(POLICY_DIR, "alpha_walking.onnx"))
        self.stand = ort.InferenceSession(os.path.join(POLICY_DIR, "alpha_stand.onnx"))
        ball_j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self.ball_q = int(self.model.jnt_qposadr[ball_j])
        self.ball_v = int(self.model.jnt_dofadr[ball_j])
        self.ball_carrier = None  # duck index carrying the ball, or None
        self.ball_hold = None     # fixed xyz to pin the ball to (e.g. user's hand), or None
        self.t = 0.0
        self.dt = TIMESTEP * DECIMATION

    # ---- ball -------------------------------------------------------------
    def ball_pos(self):
        return self.data.qpos[self.ball_q:self.ball_q + 3].copy()

    def ball_vel(self):
        return self.data.qvel[self.ball_v:self.ball_v + 3].copy()

    def throw(self, pos, vel):
        self.ball_carrier, self.ball_hold = None, None
        self.data.qpos[self.ball_q:self.ball_q + 7] = [*pos, 1, 0, 0, 0]
        self.data.qvel[self.ball_v:self.ball_v + 6] = [*vel, 0, 0, 0]

    def _pin_ball(self):
        if self.ball_carrier is not None:
            d = self.ducks[self.ball_carrier]
            p, yaw = d.pos(self.data), d.yaw(self.data)
            target = [p[0] + 0.07 * math.cos(yaw), p[1] + 0.07 * math.sin(yaw), p[2] + 0.13]  # "in the beak"
        elif self.ball_hold is not None:
            target = self.ball_hold
        else:
            return
        self.data.qpos[self.ball_q:self.ball_q + 7] = [*target, 1, 0, 0, 0]
        self.data.qvel[self.ball_v:self.ball_v + 6] = 0

    # ---- stepping -----------------------------------------------------------
    def step(self):
        """One 50 Hz control step: policies -> BAM targets -> 4 physics substeps."""
        targets = np.empty(self.model.nu)
        for d in self.ducks:
            sess = self.walk if np.linalg.norm(d.twist) > SWITCH_THRESHOLD else self.stand
            a = sess.run(None, {"obs": d.obs(self.data)[None]})[0][0]
            d.last_action[:] = a
            targets[d.act_ids] = DEFAULT_POSE + a
        self.bam.q_target[:] = targets
        for _ in range(DECIMATION):
            self.bam.update()
            mujoco.mj_step(self.model, self.data)
            self._pin_ball()
        self.t += self.dt
        # fallen ducks: stand them back up in place after 1.5 s (prototype shortcut)
        for d in self.ducks:
            if d.upright(self.data):
                d.fallen_t = 0.0
                continue
            d.fallen_t += self.dt
            if d.fallen_t > 1.5:
                p = d.pos(self.data)
                d.place(self.data, p[0], p[1], d.yaw(self.data))
                self.bam.reset(self.data.qpos)
                d.fallen_t = 0.0

    # ---- rendering data -----------------------------------------------------
    def scene_description(self):
        """Static geometry for the web client: meshes + visual geoms per body."""
        m = self.model
        meshes, geoms = {}, []
        for g in range(m.ngeom):
            if m.geom_group[g] not in (0, 2) or m.geom_bodyid[g] == 0:
                continue
            gtype = int(m.geom_type[g])
            entry = {"body": int(m.geom_bodyid[g]), "type": gtype,
                     "size": m.geom_size[g].round(5).tolist(),
                     "pos": m.geom_pos[g].round(5).tolist(), "quat": m.geom_quat[g].round(5).tolist(),
                     "rgba": m.geom_rgba[g].round(3).tolist()}
            if m.geom_matid[g] >= 0:
                entry["rgba"] = m.mat_rgba[m.geom_matid[g]].round(3).tolist()
            if gtype == mujoco.mjtGeom.mjGEOM_MESH:
                mid = int(m.geom_dataid[g])
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid) or f"mesh{mid}"
                key = name.split("_", 1)[1] if name.startswith("d") and "_" in name else name
                if key not in meshes:
                    v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                    f0, nf = m.mesh_faceadr[mid], m.mesh_facenum[mid]
                    # packed little-endian float32 / uint32, base64 — ~4x smaller than JSON lists
                    meshes[key] = {
                        "vert": base64.b64encode(m.mesh_vert[v0:v0 + nv].astype("<f4").tobytes()).decode(),
                        "face": base64.b64encode(m.mesh_face[f0:f0 + nf].astype("<u4").tobytes()).decode()}
                entry["mesh"] = key
            geoms.append(entry)
        return {"nbody": int(m.nbody), "meshes": meshes, "geoms": geoms, "n": len(self.ducks),
                "ball_radius": BALL_RADIUS}

    def body_poses(self):
        return np.concatenate([self.data.xpos, self.data.xquat], axis=1).astype(np.float32)


if __name__ == "__main__":
    import time
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    sim = SwarmSim(n)
    for d in sim.ducks:
        d.twist[:] = [0.3, 0, 0]
    t0 = time.time()
    steps = 250
    for _ in range(steps):
        sim.step()
    wall = time.time() - t0
    print(f"{n} ducks: {steps * sim.dt:.1f}s sim in {wall:.1f}s wall -> {steps * sim.dt / wall:.2f}x realtime")
    for d in sim.ducks:
        print(f"duck {d.i}: pos={d.pos(sim.data).round(2)} upright={d.upright(sim.data)}")
    desc = sim.scene_description()
    print("meshes", len(desc["meshes"]), "geoms", len(desc["geoms"]), "bodies", desc["nbody"])
