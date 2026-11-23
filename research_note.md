# Can a drawer of old phones replace a GPU?

**To:** CTO  **Re:** image classifier for the Coimbatore customer
**Model:** ResNet-18 (pretrained, ImageNet). **Important:** every number about the fleet is *simulated* from the speeds in the fleet table. I did not run on real phones.

## Answer

**No, not as a GPU replacement. Yes, as a cheap way to handle one or two slow camera feeds.**

- **Throughput:** the best plan runs at 8.9 images/s. The Raspberry Pi alone manages 1.4 images/s, so the drawer is 6.5x faster. A 5 fps camera is handled with headroom; the Pi alone cannot keep up.
- **Latency:** almost no gain, 658 ms against 726 ms for the Pi alone. Splitting a model only helps throughput. Each image still passes through all the compute and every network hop.
- **Compared with a GPU:** one CPU core of my laptop runs the same model at about 28 images/s, roughly three times the whole drawer. I did not measure a GPU, so I give no GPU number. But a drawer whose combined usable compute is less than one laptop core is not in the same class as a GPU.
- **Reliability:** the plan uses four phones, and the chance that at least one drops in a given hour is about 29%. A drop causes a latency spike of about 2.4 s and takes 4 to 6.5 s to recover. Throughput then falls to 6.6 to 8.0 images/s, still above 5 fps.

If the customer needs a few frames per second and can accept about 0.7 s per frame and occasional hiccups, the drawer works. If they need more feeds or low latency, it does not.

## Where the time goes

Latency and throughput have different limits, so there are two answers.

**Latency (658 ms per image):** compute 54%, data transfer 21%, fixed network delay 25%, queue waiting about 0% (at 5 fps the pipeline is never backed up). The fixed delay is the sum of link latencies over a chain of five stages. Compression or faster phones cannot reduce it; only fewer hops can.

**Throughput:** here compute is the limit, and specifically my battery rule. The Galaxy A50 has 35% battery, so I allow it to be busy only half the time. Its 56 ms of work then behaves like 112 ms per image and caps the whole fleet at 8.9 images/s. The Redmi Note 7 is close behind at 98% busy; the busiest link is 57% busy. With the battery rule switched off, the fleet reaches 14.3 images/s, so protecting two low batteries costs 38% of throughput.

This is the opposite of Part 2 on my laptop, where a 20 Mbps link made the *network* the limit. Sending 196 KB at FP32 took 80 ms and cut throughput to 0.48x of the single process. Sending INT8 (4x fewer bytes) brought it to 1.18x with no change in accuracy on 300 images (79.7% both ways). The phone fleet has faster links (15 to 90 Mbps) and much slower devices, so compute dominates. The same model and cut points can flip from network-bound to compute-bound depending on the fleet.

I checked my cost model against the Part 2 measurements. For network-limited runs it predicted throughput within 1 to 2%. For compute-limited runs it was 10 to 25% too optimistic, because it ignores socket and pickling overhead. So the fleet numbers above are probably a little high.

## Relation to prior work

I read **Neurosurgeon** (Kang et al., ASPLOS 2017) closely. It splits a neural network at layer boundaries between a mobile device and a cloud server. It predicts each layer's latency and energy on both sides using regression models built per layer type, then chooses the cut that is best for the current bandwidth and server load, either for lowest latency or lowest energy. The paper reports large average gains in latency and energy over running everything on the phone or everything on the cloud.

My work shares its core idea: a good cut is cheap to send and comes after heavy compute, and the best cut moves when bandwidth changes (my k6 versus k8 result in Part 2). It differs in four ways:

1. I chain several weak devices instead of one phone plus one server.
2. I optimise throughput for a stream, with battery as a constraint, while Neurosurgeon optimises latency or energy per input.
3. I scale measured laptop block times by each device's GFLOPS instead of using trained per-layer predictors.
4. Neurosurgeon's gains rely on a much faster machine at the end of the link. My drawer has no fast machine, which is why my latency does not improve.

I have not read **MoDNN** (Mao et al., 2017) or **DeepThings** (Zhao et al., 2018) in depth, so I make no claims about their results. I know from their abstracts that both target groups of small devices. DeepThings splits *inside* a layer (spatial tiling). That could reduce the fixed network delay that dominates my latency, and I did not try it.

## Next experiment (two more weeks)

Run one stage on a real phone with TFLite, then put three phones on one shared Wi-Fi network and measure the real speed of a real pipeline. This tests the two assumptions my results rest on most: that the table's GFLOPS are real, and that links do not interfere with each other.

**What would change my mind:** I assumed a 5 fps camera, and the simulated 8.9 images/s is 1.8x that. If the real pipeline gives less than about 5 images/s (a drop of about 44%), the drawer fails the customer, and I would recommend buying a small device with a proper accelerator. If it holds above about 8 images/s, I would recommend a pilot with one real camera feed.

## Limits of this study

All phone behaviour is simulated. One model, 300 test images, one fleet. Dropout assumes a 1.0 s detection time and one failure at a time. Throughput variance under the simulated network is zero by construction, so those error bars say nothing about real Wi-Fi. The battery thresholds (50%, 25%) are my own choices, not measured drain.
