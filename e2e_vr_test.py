"""End-to-end check of the emulated-VR path in headless Chrome (server must be running).

Enters VR on the IWER-emulated Quest, does scripted wind-up + throws via the page's
__emuThrow hook, and checks each ball lands in front of the user and gets fetched.
Screenshots go to --shots.

    env -u PYTHONPATH ../microduck_rl/.venv/bin/python e2e_vr_test.py --shots /tmp/shots
"""
import argparse
import asyncio
import base64
import json
import math
import os
import subprocess
import tempfile
import time
import urllib.request

import websockets

ap = argparse.ArgumentParser()
ap.add_argument("--shots", default=tempfile.mkdtemp())
ap.add_argument("--url", default="http://localhost:8000/?lite")
ap.add_argument("--holds", type=int, nargs="+", default=[300, 900])
args = ap.parse_args()
os.makedirs(args.shots, exist_ok=True)

chrome = subprocess.Popen(
    ["google-chrome", "--headless=new", "--remote-debugging-port=9333", "--window-size=960,560",
     "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist",
     f"--user-data-dir={tempfile.mkdtemp()}", "about:blank"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


async def main():
    for _ in range(50):
        try:
            tabs = json.load(urllib.request.urlopen("http://localhost:9333/json"))
            break
        except OSError:
            await asyncio.sleep(0.2)
    page = [t for t in tabs if t["type"] == "page"][0]
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=None) as cdp:
        n = 0

        async def call(method, **params):
            nonlocal n
            n += 1
            await cdp.send(json.dumps({"id": n, "method": method, "params": params}))
            while True:
                r = json.loads(await cdp.recv())
                if r.get("method") == "Runtime.consoleAPICalled":
                    print("   console:", " ".join(str(a.get("value", "")) for a in r["params"]["args"])[:600])
                if r.get("id") == n:
                    return r

        async def ev(expr):
            r = await call("Runtime.evaluate", expression=expr, userGesture=True)
            return r["result"]["result"].get("value")

        async def shot(name):
            r = await call("Page.captureScreenshot", format="png")
            with open(os.path.join(args.shots, name + ".png"), "wb") as f:
                f.write(base64.b64decode(r["result"]["data"]))

        await call("Runtime.enable")
        await call("Page.navigate", url=args.url)
        await asyncio.sleep(15)  # 10 MB scene + mesh build under software GL
        await ev("document.getElementById('VRButton').click()")
        await asyncio.sleep(6)
        print("render rate in VR:", await ev("document.getElementById('fps').textContent"))
        async with websockets.connect("ws://localhost:8765", max_size=None) as ws:
            await ws.recv()

            async def state():
                while True:
                    m = await ws.recv()
                    if isinstance(m, str):
                        return json.loads(m)

            ok = 0
            for k, hold in enumerate(args.holds):
                s = await state()
                while s["ball_state"] != "held":
                    s = await state()
                user, yaw = s["user"], s["user_yaw"]
                await shot(f"t{k}_before")
                await ev(f"__emuThrow({hold})")
                t0, last, landing = time.time(), None, None
                while time.time() - t0 < 90:
                    s = await state()
                    if s["ball_state"] != last:
                        last = s["ball_state"]
                        landing = s["landing"] or landing
                        if last in ("ground", "carried"):
                            await shot(f"t{k}_{last}")
                    if last == "held" and landing is not None:
                        break
                if landing is None:
                    print(f"hold {hold} ms: no throw registered")
                    continue
                dx, dy = landing[0] - user[0], landing[1] - user[1]
                ahead = dx * math.cos(yaw) + dy * math.sin(yaw)
                side = -dx * math.sin(yaw) + dy * math.cos(yaw)
                good = ahead > 0.2 and last == "held"
                ok += good
                print(f"hold {hold} ms: landed {ahead:+.2f} m ahead, {side:+.2f} m left | "
                      f"{'fetched back' if last == 'held' else 'NOT fetched (' + last + ')'} in "
                      f"{time.time() - t0:.1f}s wall -> {'OK' if good else 'FAIL'}")
            print(f"{ok}/{len(args.holds)} throws OK | screenshots in {args.shots}")


try:
    asyncio.run(main())
finally:
    chrome.terminate()
