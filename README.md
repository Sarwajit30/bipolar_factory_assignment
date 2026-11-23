# The Drawer of Old Phones: can old phones do the work of one GPU?

Take-home for the Research Associate Intern role. I profile ResNet-18, split it across processes that talk over sockets, build a scheduler for the phone fleet, and write up what I found.

**Short answer:** the drawer beats the Raspberry Pi alone by about 6.5x in throughput, but it does not improve latency and it is nowhere near a GPU. Details are in `research_note.md`.

I used an AI assistant for code scaffolding and explanations. I ran every experiment myself, and I can explain the code and every number below.

## Reproduce (Linux, CPU only)

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install matplotlib

python profile_resnet18.py                    # Part 1, writes part1_profile.csv (about 2 min)
python get_data.py                            # downloads Imagenette, builds data.pt (300 images)
./run_part2.sh                                # Part 2, writes part2_results.csv (10 to 15 min)
python scheduler.py --validate                # cost model vs Part 2 measurements
python scheduler.py | tee part3_output.txt    # Part 3, also writes part3_dropout.png
```

Machine: HP Victus 15 laptop, CPU only, one thread per process. Close other apps and keep the charger plugged in. An early Part 1 run gave 71 ms for the whole model, probably because of a power-saving CPU state. After plugging in, it gave 36 ms and agreed with the Part 2 baseline (38 ms), so the table below uses the second run.

## Files

| File | What it is |
|---|---|
| `profile_resnet18.py` | Part 1 profiler |
| `get_data.py` | builds the 300-image Imagenette test set |
| `split.py`, `run_part2.sh` | Part 2 split pipeline (two processes, sockets) and all experiments |
| `scheduler.py`, `fleet.csv` | Part 3 scheduler, simulator and dropout test |
| `part1_profile.csv`, `part2_results.csv`, `part3_output.txt`, `part3_dropout.png` | raw results |
| `research_note.md` | Part 4, note for the CTO |

## Part 1: ResNet-18 profile

Model: torchvision ResNet-18, pretrained, one 3x224x224 image, one CPU thread. Per block: 10 warm-up runs thrown away, then 50 timed runs, mean and standard deviation. The model is cut into 13 rows: the input, the stem, the maxpool, eight residual blocks, the pooling and the final classifier.

| Block | MMACs | Params KB | Output shape | FP32 KB | INT8 KB | Latency ms (mean ± std) |
|---|---|---|---|---|---|---|
| input image | 0 | 0 | 1x3x224x224 | 588.0 | 147.0 | |
| stem (conv1+bn+relu) | 118.0 | 37.2 | 1x64x112x112 | 3136.0 | 784.0 | 2.03 ± 0.03 |
| maxpool | 0.0 | 0 | 1x64x56x56 | 784.0 | 196.0 | 4.11 ± 0.04 |
| layer1.0 | 231.2 | 289.0 | 1x64x56x56 | 784.0 | 196.0 | 3.55 ± 0.01 |
| layer1.1 | 231.2 | 289.0 | 1x64x56x56 | 784.0 | 196.0 | 3.58 ± 0.03 |
| layer2.0 | 179.8 | 899.0 | 1x128x28x28 | 392.0 | 98.0 | 2.99 ± 0.04 |
| layer2.1 | 231.2 | 1154.0 | 1x128x28x28 | 392.0 | 98.0 | 3.54 ± 0.04 |
| layer3.0 | 179.8 | 3590.0 | 1x256x14x14 | 196.0 | 49.0 | 3.04 ± 0.04 |
| layer3.1 | 231.2 | 4612.0 | 1x256x14x14 | 196.0 | 49.0 | 3.72 ± 0.06 |
| layer4.0 | 179.8 | 14348.0 | 1x512x7x7 | 98.0 | 24.5 | 4.01 ± 0.08 |
| layer4.1 | 231.2 | 18440.0 | 1x512x7x7 | 98.0 | 24.5 | 4.84 ± 0.08 |
| avgpool+flatten | 0.0 | 0 | 1x512 | 2.0 | 0.5 | 0.01 ± 0.00 |
| fc | 0.5 | 2003.9 | 1x1000 | 3.9 | 1.0 | 0.03 ± 0.00 |

Total about 1,814 MMACs. Sum of block times 35.46 ms; whole model in one call 36.06 ± 0.15 ms (the 0.6 ms gap is per-call overhead).

*How the columns are computed:* MMACs = millions of multiply-accumulate operations, counted with hooks on every conv and linear layer. Params KB = number of weights x 4 bytes. Activation KB = number of values in the block's output x 4 bytes (FP32) or x 1 byte (INT8), which is why INT8 is exactly 4x smaller.

**Observations**
- Latency is not proportional to MACs. Maxpool does no multiplications but takes 4.11 ms, more than the stem (118 MMACs, 2.03 ms), because it moves a large tensor and is limited by memory speed, not arithmetic.
- Compute is spread fairly evenly: every residual block costs 3 to 5 ms and there is no single heavy block. So the choice of cut depends mostly on activation size.
- **Natural cut points:** after layer3.1 (196 KB FP32 / 49 KB INT8, with 75% of compute time already done) and after layer2.1 (392 KB, 56% done). Cuts in the stem, maxpool or layer1 are bad because the output (784 to 3136 KB) is as large as or larger than the 588 KB input image, so it would be cheaper to send the raw image.

## Part 2: split pipeline

Stage 0 (the camera-host side) and stage 1 run as separate processes connected by a TCP socket. Cut `k` means stage 0 runs blocks 0 to k-1 and stage 1 runs the rest. Threads let the stages overlap, so while image 5 is being sent, image 6 is already being computed (pipelining).

The network limit is simulated in my own code: sending N bytes takes N x 8 / bandwidth seconds (the sender waits, because the link is busy), then the message arrives `delay` later. Replies also pay the delay.

- **Latency:** 30 images, one at a time, after 10 warm-up images.
- **Throughput:** 200 images pipelined, 3 trials.
- **Accuracy:** top-1 on 300 Imagenette validation images, using the full 1000-class ImageNet output.
- **INT8:** symmetric, one scale per activation tensor, activations only (weights stay FP32).

| Config | Bytes per image | Latency ms (mean ± std) | Throughput img/s (mean ± std) | vs baseline | Top-1 % |
|---|---|---|---|---|---|
| baseline, 1 process | 602112 | 38.2 ± 2.0 | 25.47 ± 0.25 | 1.00x | 79.7 |
| k8 FP32, no limit | 200704 | 40.7 ± 1.6 | 34.15 ± 0.16 | 1.34x | 79.7 |
| k6 FP32, no limit | 401408 | 47.5 ± 5.2 | 40.08 ± 2.59 | 1.57x | 79.7 |
| k8 FP32, 20 Mbps + 30 ms | 200704 | 188.2 ± 3.3 | 12.32 ± 0.00 | 0.48x | 79.7 |
| k8 FP16, 20 Mbps + 30 ms | 100352 | 150.2 ± 3.5 | 24.45 ± 0.00 | 0.96x | 79.7 |
| k8 INT8, 20 Mbps + 30 ms | 50176 | 128.1 ± 3.2 | 30.02 ± 0.95 | 1.18x | 79.7 |
| k6 FP32, 20 Mbps + 30 ms | 401408 | 268.8 ± 3.7 | 6.19 ± 0.00 | 0.24x | 79.7 |
| k6 FP16, 20 Mbps + 30 ms | 200704 | 187.1 ± 3.4 | 12.32 ± 0.00 | 0.48x | 79.7 |
| k6 INT8, 20 Mbps + 30 ms | 100352 | 202.7 ± 9.5 | 23.02 ± 0.43 | 0.90x | 79.7 |

**Observations**
- Splitting never reduced single-image latency. Pipelining raises throughput only: each image still needs all the compute plus the transfer.
- With no network limit, k6 beats k8 because compute is more evenly balanced. At 20 Mbps the order flips, because k6 sends twice the bytes. The best cut depends on the network.
- At 20 Mbps, FP32 loses to the baseline (0.48x). A 196 KB message takes 80 ms on that link, which is longer than the compute. INT8 sends 4x fewer bytes and brings k8 back to 1.18x.
- Accuracy was identical in every format (79.7%). With 300 images one image is 0.33 points, so the honest claim is "no measurable drop at this test size", not "INT8 is lossless".
- A throughput std of 0.00 under the network limit is not a perfect measurement. The link is a deterministic sleep, so there is almost no variance by construction.
- The baseline uses one core and the split uses two, which favours the split slightly. I report it as is.

## Part 3: scheduler

**Goal: maximise throughput**, ties broken by lower latency. A camera feed is a stream, so frames per second matters more than the delay of one frame. Latency is reported as well.

**Cost model**
- Block time on a device = my laptop's measured block time x (laptop GFLOPS / device GFLOPS). The laptop's effective speed from Part 1 is 102 GFLOPS (2 x total MACs / total time). I scale measured times rather than computing FLOPs / GFLOPS directly, because some blocks (maxpool) have no math but still take time.
- Throughput of a plan = 1 / (the slowest of any device's compute time and any link's sending time). This is the assembly-line rule: the slowest stage sets the pace.
- A hop between two devices uses the lower of their two link speeds and adds both link latencies.
- Activations are sent as INT8 (no accuracy loss in Part 2).
- The Pi starts every image, as the task states. It may hold zero blocks and simply forward the raw image.
- Search: brute force over every device order and every set of cut points. This is fine for 6 devices.

**Battery rule (duty cap).** Battery 50% or more, or mains: may be busy 100% of the time. 25% to 50%: busy at most 50% of the time. Below 25%: not used. *Reasoning:* the energy a phone spends is roughly proportional to how long it computes, so a lower battery gets a lower busy limit, and a nearly empty phone is not trusted with a stage at all. This excludes the OnePlus 6 (20%) and halves the Galaxy A50 (35%). The 50% and 25% thresholds are my choices, not measured drain.

**Chosen assignment (battery rule on)**

| Device | Blocks | Compute ms | Busy % |
|---|---|---|---|
| Raspberry Pi 4 | forwards raw image | 0 | 0 |
| Redmi Note 7 | stem .. layer1.0 | 110.2 | 98 |
| Galaxy A50 | layer1.1 .. layer2.0 | 56.1 | 50 |
| Pixel 3a | layer2.1 .. layer4.0 | 104.6 | 93 |
| Moto G6 | layer4.1 .. fc | 83.3 | 74 |

**Does the drawer beat the Pi alone?** Yes, on throughput.

| | Throughput img/s | Latency ms |
|---|---|---|
| Raspberry Pi alone | 1.38 | 726 |
| Drawer, battery rule on | 8.92 (6.5x) | 658 |
| Drawer, battery rule off | 14.32 (10.4x) | 656 |

The battery rule costs 38% of throughput. The binding limit in the chosen plan is the Galaxy A50's duty cap: 56 ms of work allowed only 50% of the time gives 112 ms per image. The Pi gets no layers because at 5 GFLOPS it is too slow to help; excluding it from compute would not change the plan.

**Where the time goes at 5 fps (mean latency 658 ms):** compute 54%, transfer 21%, fixed network delay 25%, queue waiting 0%.

**Dropout test.** One device dies at t = 20 s. The system notices after 1.0 s (assumed), re-runs the scheduler on the remaining devices (7 ms) and replays unfinished frames from the Pi, which keeps every frame until its answer returns.

| Device lost | New img/s | Peak latency ms | First output after drop (s) | Back to normal after drop (s) |
|---|---|---|---|---|
| Redmi Note 7 | 7.96 | 2378 | 1.8 | 4.3 |
| Galaxy A50 | 7.96 | 2200 | 1.6 | 4.2 |
| Pixel 3a | 6.59 | 2396 | 1.8 | 6.5 |
| Moto G6 | 7.96 | 2123 | 1.5 | 4.2 |

See `part3_dropout.png`. Latency jumps to about 2.4 s, then drains as the backlog clears. It settles at a higher steady value (788 ms vs 658 ms) because fewer or weaker devices share the work. Most of the first second is my assumed detection time, not the scheduler. Losing the Pixel 3a hurts most because it is the strongest device in the plan.

**Cost model check** (`python scheduler.py --validate`): where the network is the bottleneck, predicted throughput is within 1 to 2% of the Part 2 measurements. Where compute is the bottleneck it is 10 to 25% optimistic, because the real pipeline has pickling, socket and thread overhead that the model ignores. Predicted latency is 5 to 13% low for the same reason. One run (k6 INT8) is far off, and it also had a high std of 9.5 ms, so it was probably noisy.

## Assumptions

- Phones are simulated by scaling laptop times. No real phones were used.
- Links do not interfere with each other. A real shared Wi-Fi would be worse.
- Camera rate is 5 frames per second (changeable with `--fps`).
- Failure detection takes 1.0 s (assumed). A dropped device is gone for good.
- The result sent back to the Pi is tiny, so it only pays link delay.
- The Pi is the camera host and may be given layers but is not forced to take any.

## What I did not finish, and what I would do about it

- **No real phone or TFLite run.** Next step: run one stage on a real phone and replace the table's GFLOPS with measured ones.
- **No spatial tiling (DeepThings style) and no energy measurement.** Tiling could cut the fixed network delay that dominates latency.
- **Brute-force scheduler.** Fine for 6 devices but will not scale. A dynamic-programming version over contiguous splits would.
- **Network simulated with delays in code, not `tc`.** Shared Wi-Fi contention is not modelled.
- **Part 3 is a simulation.** I did not run the throttled pipeline across processes at the fleet's speeds.
- **One model, 300 test images.** INT8 quantises activations only, with one scale per tensor.
- **Dropout:** one failure at a time, and no device rejoining.
