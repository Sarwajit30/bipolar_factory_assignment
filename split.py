"""Part 2: run ResNet-18 as two stages in two separate processes.

  python split.py baseline                       whole model in one process
  python split.py split --cut 8                  stage0 = blocks[:8], stage1 = blocks[8:]
  python split.py split --cut 8 --mbps 20 --delay-ms 30 --dtype int8

Stage 0 is the "camera host" (Raspberry Pi role). It runs in this process.
Stage 1 is a separate server process that this script starts for you.
They talk over a TCP socket on localhost.

Simulation (all stated assumptions):
  * Network: sending N bytes takes N*8/mbps seconds (the link is busy, so the
    sender waits), then the message arrives delay_ms later (propagation).
  * Compute: --slow F multiplies a stage's compute time by F (used in Part 3).
"""
import argparse
import csv
import os
import pickle
import queue
import socket
import statistics as st
import struct
import subprocess
import sys
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision

torch.set_num_threads(1)   # one core per process, like a weak phone
PORT = 5055
NP_DTYPE = {"fp32": np.float32, "fp16": np.float16, "int8": np.int8}


# ---------------------------------------------------------------- model
def load_blocks():
    """ResNet-18 as 12 sequential blocks (same cut points as Part 1)."""
    m = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.DEFAULT).eval()
    blocks = [nn.Sequential(m.conv1, m.bn1, m.relu), m.maxpool]
    for name in ["layer1", "layer2", "layer3", "layer4"]:
        blocks.extend(list(getattr(m, name)))
    blocks.append(nn.Sequential(m.avgpool, nn.Flatten()))
    blocks.append(m.fc)
    return blocks   # indices 0..11, cut k means stage0=[:k], stage1=[k:]


# ---------------------------------------------------------------- number formats
def encode(x, dtype):
    """Turn an activation tensor into bytes in the chosen format."""
    x = x.contiguous()
    if dtype == "fp32":
        return x.numpy().tobytes(), 1.0
    if dtype == "fp16":
        return x.half().numpy().tobytes(), 1.0
    # int8: symmetric, one scale for the whole tensor
    scale = max(x.abs().max().item(), 1e-8) / 127.0
    q = (x / scale).round().clamp(-127, 127).to(torch.int8)
    return q.numpy().tobytes(), scale


def decode(buf, shape, dtype, scale):
    arr = np.frombuffer(buf, dtype=NP_DTYPE[dtype]).reshape(shape).copy()
    t = torch.from_numpy(arr)
    return t.float() * scale if dtype == "int8" else t.float()


# ---------------------------------------------------------------- socket helpers
def send_msg(sock, obj):
    data = pickle.dumps(obj)
    sock.sendall(struct.pack("!I", len(data)) + data)


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock):
    (n,) = struct.unpack("!I", recv_exact(sock, 4))
    return pickle.loads(recv_exact(sock, n))


def throttle(elapsed, slow):
    """Pretend the device is `slow` times slower than this laptop."""
    if slow > 1.0:
        time.sleep(elapsed * (slow - 1.0))


# ---------------------------------------------------------------- stage 1 server
def run_server(a):
    stage = nn.Sequential(*load_blocks()[a.cut:])
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", PORT))
    srv.listen(1)
    conn, _ = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    inbox = queue.Queue()

    def receiver():
        # Reads messages and holds each one until its simulated arrival time
        while True:
            msg = recv_msg(conn)
            if msg.get("stop"):
                inbox.put(None)
                return
            wait = msg["sent_at"] + a.delay_ms / 1000 - time.time()
            if wait > 0:
                time.sleep(wait)
            inbox.put(msg)

    threading.Thread(target=receiver, daemon=True).start()
    with torch.no_grad():
        while True:
            msg = inbox.get()
            if msg is None:
                break
            t0 = time.perf_counter()
            x = decode(msg["payload"], msg["shape"], msg["dtype"], msg["scale"])
            pred = int(stage(x).argmax(1).item())
            throttle(time.perf_counter() - t0, a.slow)
            send_msg(conn, {"id": msg["id"], "pred": pred, "sent_at": time.time()})
    conn.close()


# ---------------------------------------------------------------- stage 0 client
def run_stream(sock, stage0, imgs, a, pipelined):
    """Send len(imgs) images through the pipeline.
    pipelined=False: wait for each answer before starting the next image (latency).
    pipelined=True : keep feeding images without waiting (throughput).
    Returns (latencies_sec, total_sec, preds, bytes_per_image)."""
    n = len(imgs)
    send_q = queue.Queue()
    done = [threading.Event() for _ in range(n)]
    t_start, t_end, preds = {}, {}, {}

    def sender():
        # The "network card": transmitting takes bytes*8/bandwidth seconds
        while True:
            item = send_q.get()
            if item is None:
                return
            if a.mbps > 0:
                time.sleep(len(item["payload"]) * 8 / (a.mbps * 1e6))
            item["sent_at"] = time.time()
            send_msg(sock, item)

    def receiver():
        for _ in range(n):
            r = recv_msg(sock)
            wait = r["sent_at"] + a.delay_ms / 1000 - time.time()
            if wait > 0:
                time.sleep(wait)
            t_end[r["id"]] = time.perf_counter()
            preds[r["id"]] = r["pred"]
            done[r["id"]].set()

    ts = threading.Thread(target=sender)
    tr = threading.Thread(target=receiver)
    ts.start()
    tr.start()
    nbytes = 0
    t_all = time.perf_counter()
    with torch.no_grad():
        for i in range(n):
            t_start[i] = time.perf_counter()
            t0 = time.perf_counter()
            out = stage0(imgs[i:i + 1])
            payload, scale = encode(out, a.dtype)
            throttle(time.perf_counter() - t0, a.slow)
            nbytes = len(payload)
            send_q.put({"id": i, "payload": payload, "shape": tuple(out.shape),
                        "dtype": a.dtype, "scale": scale})
            if not pipelined:
                done[i].wait()
    send_q.put(None)
    for e in done:
        e.wait()
    total = time.perf_counter() - t_all
    ts.join()
    tr.join()
    lats = [t_end[i] - t_start[i] for i in range(n)]
    return lats, total, [preds[i] for i in range(n)], nbytes


def run_local(model, imgs, a):
    """Baseline: whole model in this one process, one image at a time."""
    lats, preds = [], []
    t_all = time.perf_counter()
    with torch.no_grad():
        for i in range(len(imgs)):
            t0 = time.perf_counter()
            preds.append(int(model(imgs[i:i + 1]).argmax(1).item()))
            throttle(time.perf_counter() - t0, a.slow)
            lats.append(time.perf_counter() - t0)
    return lats, time.perf_counter() - t_all, preds


# ---------------------------------------------------------------- experiment
def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["baseline", "split", "server"])
    p.add_argument("--cut", type=int, default=8)
    p.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "int8"])
    p.add_argument("--mbps", type=float, default=0, help="0 = no bandwidth limit")
    p.add_argument("--delay-ms", type=float, default=0)
    p.add_argument("--slow", type=float, default=1.0)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--label", default="run")
    a = p.parse_args()

    if a.mode == "server":
        run_server(a)
        return

    d = torch.load("data.pt")
    X, Y = d["x"], d["y"]
    blocks = load_blocks()
    server = None
    if a.mode == "baseline":
        model = nn.Sequential(*blocks)
        run_local(model, X[:10], a)   # warm-up, thrown away
        lats, _, _ = run_local(model, X[:30], a)
        tputs = []
        for _ in range(a.trials):
            _, tot, _ = run_local(model, X[:200], a)
            tputs.append(200 / tot)
        _, _, preds = run_local(model, X[:300], a)
        nbytes = X[0:1].numel() * 4
    else:
        server = subprocess.Popen(
            [sys.executable, __file__, "server", "--cut", str(a.cut),
             "--delay-ms", str(a.delay_ms), "--slow", str(a.slow)])
        sock = None
        for _ in range(100):
            try:
                sock = socket.create_connection(("127.0.0.1", PORT))
                break
            except ConnectionRefusedError:
                time.sleep(0.3)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        stage0 = nn.Sequential(*blocks[:a.cut])
        run_stream(sock, stage0, X[:10], a, pipelined=False)   # warm-up
        lats, _, _, nbytes = run_stream(sock, stage0, X[:30], a, pipelined=False)
        tputs = []
        for _ in range(a.trials):
            _, tot, _, _ = run_stream(sock, stage0, X[:200], a, pipelined=True)
            tputs.append(200 / tot)
        _, _, preds, _ = run_stream(sock, stage0, X[:300], a, pipelined=True)
        send_msg(sock, {"stop": True})
        server.wait(timeout=20)

    acc = 100 * sum(int(p_ == y) for p_, y in zip(preds, Y[:300].tolist())) / 300
    lat_ms = [x * 1000 for x in lats]
    row = {
        "label": a.label, "mode": a.mode, "cut": a.cut if a.mode == "split" else "-",
        "dtype": a.dtype, "mbps": a.mbps, "delay_ms": a.delay_ms,
        "bytes_per_img": nbytes,
        "lat_mean_ms": round(st.mean(lat_ms), 1), "lat_std_ms": round(st.stdev(lat_ms), 1),
        "tput_mean": round(st.mean(tputs), 2),
        "tput_std": round(st.stdev(tputs), 2) if len(tputs) > 1 else 0,
        "top1_pct": round(acc, 1),
    }
    print(row)
    new = not os.path.exists("part2_results.csv")
    with open("part2_results.csv", "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if new:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    main()
