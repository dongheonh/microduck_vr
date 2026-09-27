"""Randomized stress test: random idle times + random throws, several seeds and swarm sizes."""
import sys
import numpy as np
from sim import SwarmSim
from server import FetchGame

def run(n, seed, throws=6):
    rng = np.random.default_rng(seed)
    sim = SwarmSim(n); game = FetchGame(sim)
    ok, times, min_sep = 0, [], np.inf
    for _ in range(throws):
        for _ in range(int(rng.uniform(1, 60) / sim.dt)):  # random idle 1-60 s
            game.update(); sim.step()
        target = [rng.uniform(-0.3, 2.0), rng.uniform(-1.3, 1.3)]
        game.throw_to(target); t0 = sim.t
        while sim.t - t0 < 120:
            game.update(); sim.step()
            P = np.array([d.pos(sim.data)[:2] for d in sim.ducks])
            min_sep = min(min_sep, min(np.linalg.norm(P[i] - P[j]) for i in range(n) for j in range(i + 1, n)))
            if game.state == "held" and sim.t - t0 > 1:
                break
        if game.state == "held":
            ok += 1; times.append(sim.t - t0)
        else:
            print(f"   n={n} seed={seed}: throw {np.round(target, 2)} FAILED in state {game.state}")
            game.give_ball_to_user()
    return ok, times, min_sep

for n in [int(a) for a in sys.argv[1:]] or [5]:
    for seed in range(3):
        ok, times, sep = run(n, seed)
        print(f"n={n} seed={seed}: {ok}/6 delivered, median {np.median(times) if times else float('nan'):.1f}s, "
              f"max {max(times) if times else float('nan'):.1f}s sim | closest pair {sep:.3f} m", flush=True)
