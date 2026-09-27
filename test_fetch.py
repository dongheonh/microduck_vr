"""Headless regression: idle into formation, then throws in many directions; each must be delivered."""
import sys
import numpy as np
from sim import SwarmSim
from server import FetchGame

n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
sim = SwarmSim(n)
game = FetchGame(sim)
for _ in range(750):  # 15 s idle: ducks form up and stand (the case that used to deadlock)
    game.update(); sim.step()
ok, min_sep = 0, np.inf
targets = [[0.9, 0.6], [0.2, -0.9], [1.3, -0.2], [0.6, 1.0], [1.5, 0.8], [0.0, 0.8], [1.0, -1.0], [1.8, 0.1], [-0.2, -1.2], [2.0, -0.6], [0.4, 0.3], [1.1, 1.3]]
for target in targets:
    game.throw_to(target); t0, picked = sim.t, None
    while sim.t - t0 < 120:
        game.update(); sim.step()
        P = np.array([d.pos(sim.data)[:2] for d in sim.ducks])
        sep, a, b = min((np.linalg.norm(P[i] - P[j]), i, j) for i in range(n) for j in range(i + 1, n))
        if sep < min_sep:
            min_sep, worst = sep, (round(sim.t - t0, 1), a, b, game.state, game.fetcher)
        if game.state == "carried" and picked is None:
            picked = sim.t - t0
        if game.state == "held" and sim.t - t0 > 1:
            break
    done = game.state == "held"
    ok += done
    print(f"throw {target}: {'DELIVERED' if done else 'FAILED (' + game.state + ')'}  pickup {picked and round(picked, 1)}s  total {sim.t - t0:.1f}s sim")
    for _ in range(250):
        game.update(); sim.step()
print(f"{ok}/{len(targets)} delivered | closest two ducks ever got: {min_sep:.3f} m "
      f"(t+{worst[0]}s, ducks {worst[1]}&{worst[2]}, ball {worst[3]}, fetcher {worst[4]})")
