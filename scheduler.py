"""Part 3: scheduler for the drawer of old phones.

  python scheduler.py                 plan, stream simulation, dropout test, plot
  python scheduler.py --validate      check the cost model against Part 2 measurements

Goal: MAXIMISE THROUGHPUT (a camera feed is a stream), ties broken by lower latency.

Assumptions (all stated on purpose):
  * Block time on a device = laptop time from Part 1 x (laptop_gflops / device_gflops).
    laptop_gflops = 2 x total MACs / total measured time. Using measured times keeps
    memory-bound blocks (like maxpool) realistic.
  * Star network: a hop between two devices has bandwidth = min of their two links and
    delay = sum of their two link latencies. Links do not interfere with each other.
  * Activations are sent in --dtype (default int8, Part 2 showed no accuracy loss).
  * Each device appears at most once, in one contiguous range of blocks.
  * The Pi (camera host) must start every image. It may hold zero blocks and just forward.
  * The final answer is tiny, so its return trip costs only link delay.
  * Battery rule (duty cap): battery >= 50% or mains: may be busy 100% of the time.
    25 to 50%: at most 50%. Below 25%: not used at all.
"""
import argparse
import csv
import itertools
import statistics
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DTYPE_FACTOR = {"fp32": 1.0, "fp16": 0.5, "int8": 0.25}


# ------------------------------------------------------------------ loading
def load_profile(path):
    rows = list(csv.DictReader(open(path)))
    inp, blocks = rows[0], rows[1:]
    names = [r["block"] for r in blocks]
    t = [float(r["lat_mean_ms"]) / 1000 for r in blocks]
    macs = [float(r["MMACs"]) * 1e6 for r in blocks]
    # act[c] = bytes (FP32) crossing a cut at position c. c=0 is the raw image.
    act = [float(inp["fp32_KB"]) * 1024] + [float(r["fp32_KB"]) * 1024 for r in blocks]
    return names, t, macs, act


def duty_cap(dev):
    if dev["mains"]:
        return 1.0
    if dev["battery"] < 25:
        return 0.0
    if dev["battery"] < 50:
        return 0.5
    return 1.0


def load_fleet(path):
    fleet = []
    for r in csv.DictReader(open(path)):
        b = r["battery_pct"].strip()
        dev = {
            "name": r["device"], "gflops": float(r["effective_gflops"]),
            "mains": b == "mains", "battery": 100.0 if b == "mains" else float(b),
            "mbps": float(r["link_mbps"]), "lat": float(r["link_latency_ms"]) / 1000,
            "drop": float(r["drop_chance_per_hour"]),
            "host": "camera host" in r["device"].lower(),
        }
        dev["cap"] = duty_cap(dev)
        fleet.append(dev)
    return fleet


# ------------------------------------------------------------------ cost model and search
class Model:
    def __init__(self, names, t, macs, act, dtype):
        self.names, self.n, self.act = names, len(t), act
        self.factor = DTYPE_FACTOR[dtype]
        self.cum = [0.0]
        for x in t:
            self.cum.append(self.cum[-1] + x)
        self.laptop_gflops = 2 * sum(macs) / sum(t) / 1e9

    def compute_time(self, dev, a, b):
        return (self.cum[b] - self.cum[a]) * self.laptop_gflops / dev["gflops"]

    def evaluate(self, order, cuts):
        """order[0] is the Pi. Stage i runs blocks bounds[i]..bounds[i+1]-1."""
        bounds = [0, *cuts, self.n]
        comp, eff = [], []
        for i, dev in enumerate(order):
            c = self.compute_time(dev, bounds[i], bounds[i + 1])
            comp.append(c)
            eff.append(c / dev["cap"])   # duty cap: a capped device acts slower
        tx, prop = [], []
        for i in range(len(order) - 1):
            a, b = order[i], order[i + 1]
            nbytes = self.act[bounds[i + 1]] * self.factor
            tx.append(nbytes * 8 / (min(a["mbps"], b["mbps"]) * 1e6))
            prop.append(a["lat"] + b["lat"])
        ret = order[-1]["lat"] + order[0]["lat"] if len(order) > 1 else 0.0
        bottleneck = max(eff + tx)
        latency = sum(comp) + sum(tx) + sum(prop) + ret
        return bottleneck, latency, comp, tx, prop, ret

    def describe(self, order, cuts):
        bott, lat, comp, tx, prop, ret = self.evaluate(order, cuts)
        survive = 1.0
        for i, d in enumerate(order):
            if comp[i] > 0:
                survive *= 1 - d["drop"]
        return {"order": order, "cuts": cuts, "bounds": [0, *cuts, self.n],
                "comp": comp, "tx": tx, "prop": prop, "ret": ret,
                "bottleneck": bott, "latency": lat, "throughput": 1 / bott,
                "survive": survive}

    def plan(self, fleet):
        """Brute force over every device order and every set of cut points."""
        host = next(d for d in fleet if d["host"])
        helpers = [d for d in fleet if not d["host"] and d["cap"] > 0]
        best = None
        for m in range(len(helpers) + 1):
            for perm in itertools.permutations(helpers, m):
                order = [host, *perm]
                for cuts in itertools.combinations(range(self.n), m):
                    bott, lat = self.evaluate(order, cuts)[:2]
                    key = (round(bott, 4), lat)   # best throughput, then best latency
                    if best is None or key < best[0]:
                        best = (key, order, cuts)
        return self.describe(best[1], best[2])


# ------------------------------------------------------------------ simulation
def steps_from(p):
    """Turn a plan into the list of things one image passes through."""
    steps = []
    n = len(p["order"])
    for i in range(n):
        if p["comp"][i] > 0:
            steps.append(("res", p["comp"][i]))     # a device computing
        if i < n - 1:
            steps.append(("res", p["tx"][i]))       # a link transmitting
            steps.append(("delay", p["prop"][i]))   # travel time, no queue
    if p["ret"] > 0:
        steps.append(("delay", p["ret"]))
    return steps


def run_frames(steps, frames, t_start):
    """frames: list of (id, arrival). Each resource handles one frame at a time, in order."""
    free = [t_start] * len(steps)
    out = {}
    for fid, arrival in frames:
        t = max(arrival, t_start)
        for k, (kind, s) in enumerate(steps):
            if kind == "res":
                t = max(t, free[k]) + s
                free[k] = t
            else:
                t += s
        out[fid] = t
    return out


def run_dropout(model, fleet, p0, victim, a):
    n = int(a.fps * a.duration)
    frames = [(i, i / a.fps) for i in range(n)]
    arr = dict(frames)
    fin0 = run_frames(steps_from(p0), frames, 0.0)

    t0 = time.perf_counter()
    p1 = model.plan([d for d in fleet if d is not victim])
    replan_s = time.perf_counter() - t0
    t_resume = a.t_drop + a.detect_s + replan_s

    # Frames not finished at the drop are replayed from the Pi under the new plan
    pending = [(i, arr[i]) for i in range(n) if fin0[i] > a.t_drop]
    fin1 = run_frames(steps_from(p1), pending, t_resume)
    fin = {i: f for i, f in fin0.items() if f <= a.t_drop}
    fin.update(fin1)
    lat = [fin[i] - arr[i] for i in range(n)]

    steady_before = statistics.median(fin[i] - arr[i] for i in fin0 if fin0[i] <= a.t_drop)
    thr = 1.2 * p1["latency"]
    post = [i for i, _ in pending]
    bad = [i for i in post if lat[i] > thr]
    if not bad:
        recover = min(fin1.values()) - a.t_drop
    elif bad[-1] == post[-1]:
        recover = None
    else:
        recover = fin[bad[-1] + 1] - a.t_drop
    return {
        "victim": victim["name"], "plan": p1, "replan_s": replan_s, "t_resume": t_resume,
        "replayed": sum(1 for i in post if arr[i] <= a.t_drop),
        "first_output_s": min(fin1.values()) - a.t_drop,
        "recover_s": recover, "peak_ms": 1000 * max(lat), "steady_before_ms": 1000 * steady_before,
        "arrivals": [arr[i] for i in range(n)], "lat": lat,
    }


# ------------------------------------------------------------------ printing
def print_plan(title, p, model):
    print(f"\n== {title}")
    print(f"{'device':30}{'blocks':44}{'compute ms':>11}{'busy %':>8}")
    for i, d in enumerate(p["order"]):
        a, b = p["bounds"][i], p["bounds"][i + 1]
        blocks = f"{model.names[a]} .. {model.names[b - 1]}" if b > a else "(forwards raw image)"
        print(f"{d['name']:30}{blocks:44}{p['comp'][i] * 1000:11.1f}"
              f"{100 * p['comp'][i] / p['bottleneck']:8.0f}")
    for i, t in enumerate(p["tx"]):
        print(f"  link {p['order'][i]['name']} -> {p['order'][i + 1]['name']}: "
              f"{t * 1000:.1f} ms sending (+{p['prop'][i] * 1000:.0f} ms delay), "
              f"busy {100 * t / p['bottleneck']:.0f}%")
    print(f"throughput {p['throughput']:.2f} img/s | latency {p['latency'] * 1000:.0f} ms | "
          f"chance no used helper drops in 1 h: {100 * p['survive']:.0f}%")


def validate(model, path="part2_results.csv"):
    """Predict each Part 2 run with the same cost model and compare to what you measured."""
    print(f"{'label':20}{'pred img/s':>11}{'meas img/s':>11}{'err %':>7}"
          f"{'pred ms':>9}{'meas ms':>9}{'err %':>7}")
    for r in csv.DictReader(open(path)):
        total = model.cum[model.n]
        if r["mode"] == "baseline":
            pt, pl = 1 / total, total
        else:
            k = int(r["cut"])
            s0, s1 = model.cum[k], total - model.cum[k]
            mbps, delay = float(r["mbps"]), float(r["delay_ms"]) / 1000
            tx = model.act[k] * DTYPE_FACTOR[r["dtype"]] * 8 / (mbps * 1e6) if mbps > 0 else 0
            pt, pl = 1 / max(s0, tx, s1), s0 + tx + s1 + 2 * delay
        mt, ml = float(r["tput_mean"]), float(r["lat_mean_ms"]) / 1000
        print(f"{r['label']:20}{pt:11.2f}{mt:11.2f}{100 * (pt - mt) / mt:7.1f}"
              f"{pl * 1000:9.1f}{ml * 1000:9.1f}{100 * (pl - ml) / ml:7.1f}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleet", default="fleet.csv")
    ap.add_argument("--profile", default="part1_profile.csv")
    ap.add_argument("--dtype", default="int8", choices=list(DTYPE_FACTOR))
    ap.add_argument("--fps", type=float, default=5.0, help="camera frame rate")
    ap.add_argument("--duration", type=float, default=60.0, help="seconds simulated")
    ap.add_argument("--t-drop", type=float, default=20.0, help="when the device dies (s)")
    ap.add_argument("--detect-s", type=float, default=1.0, help="time to notice a dead device")
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()

    names, t, macs, act = load_profile(a.profile)
    model = Model(names, t, macs, act, a.dtype)
    print(f"Laptop effective speed from Part 1: {model.laptop_gflops:.0f} GFLOPS, "
          f"whole model {model.cum[-1] * 1000:.1f} ms. Activations sent as {a.dtype}.")
    if a.validate:
        validate(model)
        return

    fleet = load_fleet(a.fleet)
    host = next(d for d in fleet if d["host"])
    print("\nBattery rule (duty cap):")
    for d in fleet:
        print(f"  {d['name']:30} battery {d['battery']:5.0f}%  cap {d['cap']:.1f}")

    pi_only = model.plan([host])
    print_plan("Raspberry Pi alone (baseline)", pi_only, model)

    t0 = time.perf_counter()
    plan = model.plan(fleet)
    print(f"(planning took {time.perf_counter() - t0:.2f} s)")
    print_plan("Drawer, WITH battery rule", plan, model)

    nocap = [dict(d, cap=1.0) for d in fleet]
    plan_nocap = model.plan(nocap)
    print_plan("Drawer, battery rule OFF (for comparison)", plan_nocap, model)

    print("\n== Does the drawer beat the Pi alone?")
    print(f"throughput: {plan['throughput']:.2f} vs {pi_only['throughput']:.2f} img/s "
          f"= {plan['throughput'] / pi_only['throughput']:.1f}x")
    print(f"latency:    {plan['latency'] * 1000:.0f} vs {pi_only['latency'] * 1000:.0f} ms "
          f"= {pi_only['latency'] / plan['latency']:.1f}x lower")
    print(f"battery rule costs {100 * (1 - plan['throughput'] / plan_nocap['throughput']):.0f}% "
          f"of throughput")
    print(f"camera runs at {a.fps:.0f} fps: Pi alone "
          f"{'cannot keep up (queue grows)' if pi_only['throughput'] < a.fps else 'keeps up'}, "
          f"drawer {'keeps up' if plan['throughput'] >= a.fps else 'cannot keep up'}")

    # Where does the time go, steady state, no failures
    n = int(a.fps * a.duration)
    frames = [(i, i / a.fps) for i in range(n)]
    fin = run_frames(steps_from(plan), frames, 0.0)
    mean_lat = statistics.mean(fin[i] - i / a.fps for i in range(n))
    comp, tx = sum(plan["comp"]), sum(plan["tx"])
    delay = sum(plan["prop"]) + plan["ret"]
    wait = max(mean_lat - plan["latency"], 0.0)
    print(f"\n== Where the time goes (mean latency {mean_lat * 1000:.0f} ms at {a.fps:.0f} fps)")
    for name, v in [("compute", comp), ("transfer", tx), ("network delay", delay), ("waiting in queues", wait)]:
        print(f"  {name:18}{v * 1000:7.1f} ms  {100 * v / mean_lat:5.1f}%")

    # Dropout test for every helper in the plan
    helpers = [d for i, d in enumerate(plan["order"]) if not d["host"] and plan["comp"][i] > 0]
    results = [run_dropout(model, fleet, plan, v, a) for v in helpers]
    print(f"\n== Dropout test: device dies at t={a.t_drop:.0f} s, noticed after {a.detect_s:.1f} s, "
          f"camera {a.fps:.0f} fps")
    print(f"steady latency before the drop: {results[0]['steady_before_ms']:.0f} ms")
    print(f"{'device lost':30}{'new img/s':>10}{'new ms':>8}{'peak ms':>9}{'replayed':>9}"
          f"{'1st out s':>10}{'recovered s':>12}")
    for r in results:
        rec = f"{r['recover_s']:.1f}" if r["recover_s"] is not None else "never"
        print(f"{r['victim']:30}{r['plan']['throughput']:10.2f}{r['plan']['latency'] * 1000:8.0f}"
              f"{r['peak_ms']:9.0f}{r['replayed']:9d}{r['first_output_s']:10.1f}{rec:>12}")
    print("(replanning itself took "
          f"{max(r['replan_s'] for r in results) * 1000:.0f} ms at most)")

    worst = max(results, key=lambda r: r["peak_ms"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(worst["arrivals"], [x * 1000 for x in worst["lat"]], ".", ms=3)
    ax.axvline(a.t_drop, color="r", ls="--", label=f"{worst['victim']} drops")
    ax.axvline(worst["t_resume"], color="g", ls=":", label="new plan running")
    ax.set_xlabel("time the frame was captured (s)")
    ax.set_ylabel("latency (ms)")
    ax.set_title("Latency per frame around a device dropout")
    ax.legend()
    fig.tight_layout()
    fig.savefig("part3_dropout.png", dpi=150)
    print("\nSaved part3_dropout.png (worst case: " + worst["victim"] + ")")


if __name__ == "__main__":
    main()
