# microduck_vr — VR duck fetch (prototype)

Throw a virtual ball; a swarm of simulated Microducks fetches it and brings it back.

```
browser / Quest (Three.js + WebXR)  <-- WebSocket -->  server.py  -->  sim.py (MuJoCo, one shared world)
   click / controller throw                             fetch game + swarm planner      pretrained ONNX walk/stand policies
                                                        (writes a twist per duck)       + BAM XL330 actuators
```

## Setup

Needs the robot model, actuator model and pretrained policies from
[pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) (not vendored here —
its 3D models are CC BY-SA-NC). Clone it next to this repo, or point `MICRODUCK_RL_DIR` at it:

```bash
git clone https://github.com/pollen-robotics/microduck_rl
cd microduck_rl && uv sync && cd ..
# pretrained walking/standing policies
microduck_rl/.venv/bin/python -c "from huggingface_hub import hf_hub_download as d; \
  [d('pollen-robotics/microduck-policies', f, local_dir='microduck_rl/policies') for f in ('alpha_walking.onnx', 'alpha_stand.onnx')]"
```

## Run

```bash
cd microduck_vr
../microduck_rl/.venv/bin/python server.py --ducks 5
# open http://localhost:8000
```

(If ROS is sourced in your shell, prefix commands with `env -u PYTHONPATH`.)

**Overview mode:** click the floor to throw there, Shift+click to move yourself, drag to orbit.

**First-person VR, no headset needed:** press **ENTER VR** (bottom of the page). The page emulates
a Meta Quest 3 with [IWER](https://github.com/meta-quest/immersive-web-emulation-runtime), so the
real WebXR code path runs: click once to capture the mouse, then mouse = look, WASD = walk,
**hold click** = grab + wind up (longer = stronger), **release** = throw, **X** = exit.
The ducks bring the ball to wherever you're standing. `?native` uses a real headset instead;
`?lite` turns off shadows and halves resolution for slow GPUs.

`--time-scale` (or the Speed slider) runs the sim faster than real time — the ducks walk at ~0.1 m/s.

Uses the `microduck_rl` venv and its downloaded policies (`microduck_rl/policies/alpha_*.onnx`).

## How it works

- **sim.py** — N ducks attached into one MuJoCo world (`robot_walk.xml`, prefixes `d0_`, `d1_`…)
  plus a ball. Each duck: 61-D obs → walking policy (or standing policy when the command is ~0)
  → BAM actuator targets, 50 Hz. Pickup is virtual: the ball is pinned to the carrier's beak.
  Fallen ducks are stood back up in place after 1.5 s (prototype shortcut).
- **server.py** — ball state machine `held → flying → ground → carried → held`. On release the
  landing point is predicted ballistically and the nearest duck is assigned. Every 0.1 s a
  priority Dynamic Window Approach planner (Fox et al. 1997) picks each duck's (vx, wz):
  candidate commands are rolled out through the policy's measured response and rejected if they
  come within the safety distance of another duck; the fetcher plans first and broadcasts its
  intended path so parked ducks step aside. Jam breakers: a crowded duck must actively separate,
  and a duck blocked for 2 s may pass at 65 % of the safety distance.
- **web/index.html** — renders the streamed body poses. In VR the throw is measured the way a
  headset app would: trigger = grab, release velocity from the last ~100 ms of hand motion; the
  held ball follows the hand (`hand` message). The emulator drives IWER's headset + right
  controller from mouse/keyboard and stretches the swing over >= 5 frames at low frame rates, so
  throw strength doesn't depend on FPS. The server caps release speed at 4 m/s.

Policy quirks the planner works around (measured in sim): the walking policy barely starts from a
standstill with small commands and cannot turn in place, so moving ducks always get vx ≥ 0.25
(+ a 0.5 s kick at 0.3) and turn on arcs.

Offline demo scripts for the `microduck_rl` repo (videos of one duck, 20-duck swarms, group
navigation) are in [`microduck_rl_scripts/`](microduck_rl_scripts/).

## Tests

```bash
env -u PYTHONPATH ../microduck_rl/.venv/bin/python test_fetch.py 5      # 12 fixed throws
env -u PYTHONPATH ../microduck_rl/.venv/bin/python stress_test.py 5 8   # random idles/throws, 3 seeds
env -u PYTHONPATH ../microduck_rl/.venv/bin/python e2e_vr_test.py       # headless Chrome: emulated-VR throws (server running)
```

## Known gaps

- Robots collide only at the feet (walk model); spacing is enforced by the planner alone.
- Only tested on the emulated Quest. A real Quest needs the page over HTTPS and the socket over
  `wss://` (browsers block `ws://` from an https page) — not set up yet.
- The scene message is ~10 MB (full-detail meshes); decimate before streaming over Wi-Fi.
