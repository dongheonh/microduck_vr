# Scripts for the microduck_rl repo

These run inside a `pollen-robotics/microduck_rl` checkout (they import `scripts/infer_policy.py`
and the mjlab task registry). Copy them into `microduck_rl/scripts/` and run with `uv run`:

| Script | What it does |
|---|---|
| `record_demo.py` | Offscreen MP4 of `infer_policy.py` with a scripted keypress schedule (walk, turn, strafe, push) |
| `record_swarm.py` | N parallel mjlab envs running an ONNX policy with random commands, recorded to MP4 |
| `swarm_navigate.py` | N ducks moving as a group in one direction: formation keeping + CBF collision avoidance |

```bash
uv run scripts/record_demo.py --out demo.mp4 -- --walking policies/alpha_walking.onnx --standing policies/alpha_stand.onnx --new-cmd-obs
uv run scripts/record_swarm.py --num-envs 20
uv run scripts/swarm_navigate.py --num-envs 20 --direction-deg 45
```
