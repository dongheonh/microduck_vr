"""VR-duck prototype server: swarm fetch game over WebSocket.

    ../microduck_rl/.venv/bin/python server.py --ducks 5
    open http://localhost:8000

Client -> server (JSON):
    {"type": "throw", "pos": [x,y,z], "vel": [vx,vy,vz]}   # raw release (VR controller)
    {"type": "throw_to", "target": [x,y]}                  # desktop: lob to a floor point
    {"type": "user", "pos": [x,y], "yaw": rad}             # where the thrower stands
    {"type": "hand", "pos": [x,y,z] | null}                # VR hand: the held ball follows it
    {"type": "config", "time_scale": 2.0, "escort": true}
    {"type": "reset"}
Server -> client: one JSON "scene" message on connect, then per frame a JSON
"state" message followed by a binary Float32Array of body poses (nbody x 7:
x y z qw qx qy qz, MuJoCo z-up world frame).
All coordinates are the sim's world frame (metres, z up).
"""
import argparse
import asyncio
import functools
import http.server
import json
import math
import os
import threading
import time

import numpy as np
import websockets

from sim import BALL_RADIUS, SwarmSim

G = 9.81
TRACK_GAIN = 0.37     # achieved / commanded forward speed
VX_MAX, WZ_MAX = 0.4, 1.0
# The alpha walking policy often fails to start stepping from a standstill when the
# forward command is small (vx=0.15, wz=+1 -> 9 deg in 5 s; vx=0 turns ~13 deg), but
# is fine once walking. So a moving duck always gets >= VX_MIN_MOVING forward (turns
# on a tight arc instead of in place) and a START_KICK burst when it sets off.
VX_MIN_MOVING = 0.25
START_KICK_VX, START_KICK_S = 0.3, 0.5
SAFE_DIST = 0.22      # planner min distance between duck centres
SAFE_DIST_FETCHER = 0.16  # the fetcher may pass closer (still clear of contact) to squeeze by
# Identified response of the alpha walking policy (sim measurements, 5 s runs):
# yaw rate ~0.55 x wz command, forward speed ~0.37 x vx, a bit slower while turning.
YAW_GAIN = 0.55
PLAN_PERIOD = 0.1     # s, DWA replanning period
STUCK_RELAX_S, STUCK_SAFE_SCALE = 2.0, 0.65  # blocked this long -> shrink safety distance
HORIZON_S, HORIZON_DT = 1.5, 0.1
HORIZON_STEPS = int(HORIZON_S / HORIZON_DT)
DWA_CANDIDATES = [(0.0, 0.0)] + [(vx, wz) for vx in (VX_MIN_MOVING, 0.32, VX_MAX)
                                  for wz in (-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0)]


def rollout(p, yaw, vx, wz):
    """Predicted xy path [HORIZON_STEPS, 2] of a duck given a twist command."""
    v = TRACK_GAIN * vx * (1 - 0.25 * abs(wz))
    w = YAW_GAIN * wz
    t = np.arange(1, HORIZON_STEPS + 1) * HORIZON_DT
    th = yaw + w * t
    if abs(w) < 1e-6:
        return p + v * t[:, None] * np.array([math.cos(yaw), math.sin(yaw)])
    return p + (v / w) * np.stack([np.sin(th) - math.sin(yaw), -np.cos(th) + math.cos(yaw)], axis=1)
SLOT_SPACING = 0.35
ARRIVE_TOL = 0.08
PICKUP_DIST = 0.11
DELIVER_OFFSET = 0.25  # the fetcher parks this far from the user to hand over
HAND_HEIGHT = 0.9
MAX_THROW_SPEED = 4.0  # m/s; ~2.5 m max range from hand height
# must cover everywhere the fetcher may stop (goal +- ARRIVE_TOL) or it waits forever
DELIVER_DIST = DELIVER_OFFSET + ARRIVE_TOL + 0.04


class FetchGame:
    """Ball state machine + centralized swarm planner writing each duck's twist."""

    def __init__(self, sim: SwarmSim):
        self.sim = sim
        self.n = len(sim.ducks)
        self.user = np.array([-0.7, 0.0])
        self.user_yaw = 0.0
        self.escort = True
        self.hand_pos = None
        self.fetcher = None
        self.landing = None
        self.plan_timer = 0.0
        self.stuck_time = [0.0] * self.n
        self.moving_since = [None] * self.n
        self.events = []
        self.give_ball_to_user()

    # ---- ball lifecycle ------------------------------------------------------
    def hand(self):
        if self.hand_pos is not None:  # tracked VR hand
            return np.array(self.hand_pos)
        f = np.array([math.cos(self.user_yaw), math.sin(self.user_yaw)])
        return np.array([*(self.user + 0.25 * f), HAND_HEIGHT])

    def give_ball_to_user(self):
        self.state = "held"
        self.fetcher = None
        self.landing = None
        self.sim.ball_carrier = None
        self.sim.ball_hold = self.hand().tolist()
        self.sim.throw(self.sim.ball_hold, [0, 0, 0])
        self.sim.ball_hold = self.hand().tolist()

    def throw(self, pos, vel):
        if self.state != "held":
            return
        # cap release speed: a wild controller flick would send the ball metres away and
        # the ducks walk ~0.1 m/s (also keeps the ball on the lawn)
        vel = np.asarray(vel, dtype=float)
        speed = float(np.linalg.norm(vel))
        if speed > MAX_THROW_SPEED:
            vel = vel * (MAX_THROW_SPEED / speed)
        vel = vel.tolist()
        self.sim.throw(pos, vel)
        self.state = "flying"
        self.landing = self.predict_landing(np.array(pos), np.array(vel))
        # auction-free allocation: nearest duck (by distance to predicted landing) fetches
        d = [np.linalg.norm(duck.pos(self.sim.data)[:2] - self.landing) for duck in self.sim.ducks]
        self.fetcher = int(np.argmin(d))
        self.events.append(f"throw -> landing ({self.landing[0]:+.2f},{self.landing[1]:+.2f}), duck {self.fetcher} fetches")

    def throw_to(self, target):
        p0 = self.hand()
        dxy = np.array(target) - p0[:2]
        dist = float(np.linalg.norm(dxy))
        T = 0.45 + 0.25 * dist  # flight time: a gentle lob
        vz = (BALL_RADIUS - p0[2] + 0.5 * G * T * T) / T
        self.throw(p0.tolist(), [*(dxy / T), vz])

    @staticmethod
    def predict_landing(p, v):
        # z(t) = pz + vz t - g t^2 / 2 = r  -> positive root
        a, b, c = -0.5 * G, v[2], p[2] - BALL_RADIUS
        t = (-b - math.sqrt(max(b * b - 4 * a * c, 0.0))) / (2 * a)
        return p[:2] + v[:2] * t

    # ---- planning ------------------------------------------------------------
    def goals(self, P):
        """Per-duck goal xy (None = hold still)."""
        goals = [None] * self.n
        f = self.fetcher
        fwd = np.array([math.cos(self.user_yaw), math.sin(self.user_yaw)])
        if f is not None:
            if self.state == "flying":
                goals[f] = self.landing
            elif self.state == "ground":
                goals[f] = self.sim.ball_pos()[:2]
            elif self.state == "carried":
                to_duck = P[f] - self.user
                goals[f] = self.user + DELIVER_OFFSET * to_duck / (np.linalg.norm(to_duck) + 1e-6)
        # everyone else: grid formation around home (idle) or trailing the fetcher (escort)
        others = [i for i in range(self.n) if i != f]
        # escort on the way OUT only: on the way back the escorts would have to cross the
        # fetcher to get behind it again (deadlock), so they walk home ahead of it instead
        if f is not None and self.escort and self.state in ("flying", "ground"):
            head = goals[f] - P[f]
            head = head / (np.linalg.norm(head) + 1e-6)
            center = P[f] - 0.45 * head
            ang = math.atan2(head[1], head[0])
        else:
            # park at the user's left, out of the delivery lane in front of them
            left = np.array([-fwd[1], fwd[0]])
            center = self.user + 0.9 * left
            ang = self.user_yaw
        cols = max(1, math.ceil(math.sqrt(len(others))))
        R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
        slots = []
        for k in range(len(others)):
            r, c = divmod(k, cols)
            slots.append(center + R @ np.array([-r * SLOT_SPACING, (c - (cols - 1) / 2) * SLOT_SPACING]))
        # greedy nearest-slot assignment keeps ducks from crossing through each other
        free = list(range(len(slots)))
        for i in sorted(others, key=lambda i: np.linalg.norm(P[i] - center)):
            k = min(free, key=lambda k: np.linalg.norm(slots[k] - P[i]))
            free.remove(k)
            goals[i] = slots[k]
        return goals

    def dwa(self, i, P, yaw, goal, committed, ignore_others=False):
        """Dynamic Window Approach (Fox, Burgard & Thrun 1997) for one duck.

        Forward-simulates each candidate (vx, wz) command through the identified
        response of the walking policy. A candidate is admissible if it keeps every
        other duck at least the pair's safety distance away (or, if already closer,
        never closes in further), checking higher-priority ducks along their
        committed trajectories and lower-priority ones where they stand. Among
        admissible candidates it maximizes progress to the goal; standing still is
        a candidate too, so a parked duck also steps aside when a higher-priority
        duck's path runs through it. If nothing is admissible, it takes the
        candidate that keeps the most clearance. Returns (twist, trajectory [H, 2]).
        """
        goal = P[i] if goal is None else goal
        at_goal = np.linalg.norm(goal - P[i]) < ARRIVE_TOL
        stay = np.repeat(P[i][None], HORIZON_STEPS, axis=0)
        ids = [] if ignore_others else [j for j in range(self.n) if j != i]
        others = np.stack([committed.get(j, np.repeat(P[j][None], HORIZON_STEPS, axis=0)) for j in ids]) \
            if ids else np.zeros((0, HORIZON_STEPS, 2))
        safe = np.array([SAFE_DIST_FETCHER if self.fetcher in (i, j) else SAFE_DIST for j in ids])
        if self.stuck_time[i] > STUCK_RELAX_S:
            safe = safe * STUCK_SAFE_SCALE  # jam breaker: squeeze past (still clear of contact)
        d_now = np.linalg.norm(P[i] - P[ids], axis=1) if ids else np.zeros(0)
        base = np.linalg.norm(goal - P[i])
        best = best_fallback = None
        best_score = best_clear = -np.inf
        for vx, wz in DWA_CANDIDATES:
            traj = stay if vx == 0.0 else rollout(P[i], yaw, vx, wz)
            dmin = np.linalg.norm(traj[None] - others, axis=2).min(axis=1) if ids else np.zeros(0)
            clear = float((dmin - safe).min()) if ids else np.inf
            crowded = ids and bool(np.any(d_now < safe))
            if clear > best_clear and not (crowded and vx == 0.0):  # crowded: must actively separate
                best_fallback, best_clear = (np.array([vx, 0.0, wz]), traj), clear
            bad = (dmin < safe) & ((dmin < d_now - 1e-3) | (d_now >= safe))
            if np.any(bad):
                continue
            if vx == 0.0:
                # stopping short of the goal is only chosen when every move is blocked
                score = 1.0 if at_goal else -1.0
            else:
                # progress over the window + how well it ends up facing the goal; the
                # heading term lets a duck facing away turn around (turning first costs
                # progress, which otherwise makes "stop" the local optimum)
                progress = base - np.linalg.norm(goal - traj[-1])
                h, g = traj[-1] - traj[-2], goal - traj[-1]
                score = progress + 0.1 * (h @ g) / (np.linalg.norm(h) * np.linalg.norm(g) + 1e-9)
            if score > best_score:
                best, best_score = (np.array([vx, 0.0, wz]), traj), score
        return best if best is not None else best_fallback

    def update(self):
        sim = self.sim
        P = np.array([d.pos(sim.data)[:2] for d in sim.ducks])
        ball = sim.ball_pos()

        # transitions
        if self.state == "flying" and ball[2] < BALL_RADIUS + 0.01 and abs(sim.ball_vel()[2]) < 0.3:
            self.state = "ground"
        if self.state == "ground" and np.linalg.norm(P[self.fetcher] - ball[:2]) < PICKUP_DIST \
                and np.linalg.norm(sim.ball_vel()) < 0.4:
            self.state = "carried"
            sim.ball_carrier = self.fetcher
            self.events.append(f"duck {self.fetcher} picked up the ball")
        if self.state == "carried" and np.linalg.norm(P[self.fetcher] - self.user) < DELIVER_DIST:
            self.events.append(f"duck {self.fetcher} delivered the ball")
            self.give_ball_to_user()
        if self.state == "held":
            sim.ball_hold = self.hand().tolist()

        # control: replan at 10 Hz, fetcher first, then by distance to goal (closest first)
        self.plan_timer -= sim.dt
        if self.plan_timer > 0:
            return
        self.plan_timer = PLAN_PERIOD
        goals = self.goals(P)
        yaws = [d.yaw(sim.data) for d in sim.ducks]
        order = sorted(range(self.n), key=lambda i: (i != self.fetcher,
                       np.linalg.norm(goals[i] - P[i]) if goals[i] is not None else 0.0))
        committed = {}
        for i in order:
            duck = sim.ducks[i]
            twist, traj = self.dwa(i, P, yaws[i], goals[i], committed)
            committed[i] = traj
            wants_to_move = goals[i] is not None and np.linalg.norm(goals[i] - P[i]) >= ARRIVE_TOL
            self.stuck_time[i] = self.stuck_time[i] + PLAN_PERIOD if (wants_to_move and twist[0] == 0.0) else 0.0
            if i == self.fetcher:
                # broadcast where the fetcher WANTS to go (planned as if the floor were
                # empty), so lower-priority ducks clear its lane even while it waits —
                # otherwise a blocked fetcher and a parked duck wait on each other forever
                committed[i] = self.dwa(i, P, yaws[i], goals[i], {}, ignore_others=True)[1]
            if twist[0] == 0.0:
                duck.twist[:] = 0
                self.moving_since[i] = None
                continue
            if self.moving_since[i] is None:
                self.moving_since[i] = sim.t
            if sim.t - self.moving_since[i] < START_KICK_S:
                twist[0] = max(twist[0], START_KICK_VX)
            duck.twist[:] = twist

    def snapshot(self):
        sim = self.sim
        return {
            "type": "state", "t": round(sim.t, 2), "ball_state": self.state,
            "fetcher": self.fetcher,
            "landing": None if self.landing is None else self.landing.round(3).tolist(),
            "user": self.user.round(3).tolist(), "user_yaw": self.user_yaw,
            "ducks": [{"pos": d.pos(sim.data).round(3).tolist(), "twist": d.twist.round(2).tolist()}
                      for d in sim.ducks],
            "events": self.events[-6:],
        }


async def main_async(args):
    sim = SwarmSim(args.ducks)
    game = FetchGame(sim)
    clients = {}  # ws -> asyncio.Event; each client has its own sender task
    latest = {"frame": None}
    cfg = {"time_scale": args.time_scale}
    scene_msg = json.dumps({"type": "scene", **sim.scene_description()})

    async def sender(ws, ready):
        """Send only the newest frame: a slow client drops frames instead of stalling the sim."""
        while True:
            await ready.wait()
            ready.clear()
            state, poses = latest["frame"]
            await ws.send(state)
            await ws.send(poses)

    async def handler(ws):
        await ws.send(scene_msg)
        ready = asyncio.Event()
        clients[ws] = ready
        send_task = asyncio.create_task(sender(ws, ready))
        try:
            async for raw in ws:
                msg = json.loads(raw)
                t = msg.get("type")
                if t == "throw":
                    game.throw(msg["pos"], msg["vel"])
                elif t == "throw_to":
                    game.throw_to(msg["target"])
                elif t == "user":
                    game.user = np.array(msg["pos"], dtype=float)
                    game.user_yaw = float(msg.get("yaw", game.user_yaw))
                elif t == "hand":
                    game.hand_pos = None if msg.get("pos") is None else [float(v) for v in msg["pos"]]
                elif t == "config":
                    cfg["time_scale"] = float(np.clip(msg.get("time_scale", cfg["time_scale"]), 0.25, 6))
                    game.escort = bool(msg.get("escort", game.escort))
                elif t == "reset":
                    for i, d in enumerate(sim.ducks):
                        d.place(sim.data, 0.3 * (i // 3), (i % 3 - 1) * 0.35, 0.0)
                        d.twist[:] = 0
                    sim.bam.reset(sim.data.qpos)
                    game.give_ball_to_user()
        finally:
            clients.pop(ws, None)
            send_task.cancel()

    async def sim_loop():
        frame_every = 1 / args.fps
        next_frame = time.perf_counter()
        sim_debt = 0.0
        last = time.perf_counter()
        while True:
            now = time.perf_counter()
            sim_debt = min(sim_debt + (now - last) * cfg["time_scale"], 0.5)
            last = now
            while sim_debt >= sim.dt:
                game.update()
                sim.step()
                sim_debt -= sim.dt
            if now >= next_frame and clients:
                next_frame = now + frame_every
                latest["frame"] = (json.dumps(game.snapshot()), sim.body_poses().tobytes())
                for ready in clients.values():
                    ready.set()
            await asyncio.sleep(0.004)

    web_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
    httpd = http.server.ThreadingHTTPServer(
        ("0.0.0.0", args.http_port),
        functools.partial(http.server.SimpleHTTPRequestHandler, directory=web_dir))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"[vrduck] {args.ducks} ducks | open http://localhost:{args.http_port}  (ws :{args.ws_port})")
    async with websockets.serve(handler, "0.0.0.0", args.ws_port, max_size=None):
        await sim_loop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ducks", type=int, default=5)
    ap.add_argument("--time-scale", type=float, default=2.0, help="sim speed vs wall clock (ducks are slow)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--http-port", type=int, default=8000)
    ap.add_argument("--ws-port", type=int, default=8765)
    asyncio.run(main_async(ap.parse_args()))
