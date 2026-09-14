# ZeroGraph benchmark record

This page records what was measured and, equally importantly, what was not.

## 1.54B Qwen2.5 / three-seed four-L4 replication

The stable 0.4.0 method is clean cached teacher-boundary regression at
`sigma=0`. Each seed used four-GPU FSDP2 ordinary SFT as the conventional
baseline and four independent single-L4 block jobs as the treatment.

| Seed | FSDP2 CE | Cached CE | FSDP2 critical s | Cached critical s | Ratio | Gates |
|---:|---:|---:|---:|---:|---:|---:|
| 20260910 | 1.43768 | **1.27789** | 5,004.66 | 505.95 | 9.89x | Both pass |
| 20260911 | 1.43825 | **1.27619** | 4,870.20 | 511.13 | 9.53x | Both pass |
| 20260912 | 1.43262 | **1.27572** | 4,924.68 | 511.83 | 9.62x | Both pass |

Across seeds, FSDP2 CE was 1.43618 +/- 0.00310 and cached ZeroGraph CE was
1.27660 +/- 0.00114. The mean local-critical-path ratio was 9.68x. Peak
allocated HBM was effectively equal (8.024 GB versus 7.997 GB); peak reserved
HBM was 12.012 GB versus 8.997 GB, a 1.335x ratio. Strict ZeroGraph recorded
zero inter-block gradient bytes. All three ZeroGraph seeds passed finite-value,
held-out quality, coherence, and context-use gates, including positive context
effects on all four fixed prompts.

The critical-path comparison excludes teacher-cache construction, persistent
cache storage, checkpoint I/O, assembly, and evaluation. It measures the
online local optimization phase, not first-use end-to-end economics.

## 1.1B TinyLlama / four-L4 primary

| Field | Value |
|---|---:|
| Base checkpoint parameters | 1,100,048,384 |
| Dataset | revision-pinned UltraChat 200k |
| Training examples | 8,192 first-turn exchanges |
| Validation examples | 512 disjoint exchanges |
| Sequence length | 256 |
| Blocks | 4: [0,6), [6,12), [12,17), [17,22) |
| Primary updates | 2,048 per block |
| Aggregate tokens per arm | 2,097,152 |
| Hardware | four disjoint NVIDIA L4 GPUs |
| Ordinary DDP critical compute | 1,936.41 s |
| Slowest cached ZeroGraph block | 395.20 s |
| Critical-path ratio | 4.90× shorter |
| Ordinary peak allocated HBM | 12.32 GiB/GPU |
| Maximum cached local peak | 2.67 GiB/GPU |
| HBM ratio | 4.62× lower |
| Frozen base held-out CE | 1.6813 |
| Frozen Chat teacher CE | 1.3256 |
| Ordinary DDP SFT CE | 1.2791 |
| Assembled ZeroGraph CE | 1.3096 |
| Inter-block gradient bytes | 0 |

The accepted method was teacher-anchored conversion: each block learned cached
teacher boundary targets plus a clean initialization anchor; the final block
also used assistant-token CE. The unanchored pilot failed generation coherence
and was not escalated.

The cached replay reproduced all four online-reference checkpoint SHA-256
digests exactly and reproduced held-out evaluation metrics. The one-time 40-GiB
BF16 boundary cache is persistent storage and is excluded from the per-epoch
critical compute comparison.

## TinyLlama objective and systems ablations

| Arm | Objective / topology | Held-out CE | Teacher-gen CE | Critical s | Coherent | Gate |
|---|---|---:|---:|---:|---:|---:|
| A | ordinary FSDP2 | 1.46428 | 1.44737 | 3,564.41 | yes | pass |
| C | cached clean boundary, B=4 | **1.30592** | **1.04395** | 536.75 | yes | **pass** |
| D | cached full-noise, B=4 | 1.31729 | 3.45313 | 531.29 | no | fail |
| E | online full-noise, B=4 | 1.31729 | 3.45313 | 578.89 | no | fail |
| F | full-noise B=2, FSDP2/block | 1.33664 | 3.87109 | 2,184.58 | no | fail |

The matching D/E metrics show that caching did not create the failure. Clean
boundary regression preserved coherent generation; the tested full-noise
objective did not. The earlier unanchored GPT-2 Large run was retained as the
negative Arm-B control because its local objectives improved while its language
retention and context-use gates failed.

## 4.027B logical synthetic fixture

Ten independently optimized 402.7M-parameter blocks produced:

- 9.85× lower optimized local peak allocation than the full resident control;
- 59.6% lower assembled denoising MSE than the fixed-noise input;
- 10,000 aggregate real optimizer updates;
- zero inter-block gradient bytes.

This fixture validates mechanics and memory accounting. It is not a language
quality or foundation-model convergence result.

## What is not established

- universal compatibility with arbitrary PyTorch graphs;
- from-scratch or unanchored foundation-model pretraining;
- parity with ordinary end-to-end gradients;
- a universal speedup across models, interconnects, or block counts;
- quality scaling beyond the tested model, dataset, and token budget.
- a cache-inclusive 9.68x end-to-end speedup.

Every new benchmark should report local loss windows, assembled quality,
ordinary and method HBM, critical compute, total accelerator-hours, block
topology, communication bytes, and failed preregistered gates.
