# ZeroGraph benchmark record

This page records what was measured and, equally importantly, what was not.

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

Every new benchmark should report local loss windows, assembled quality,
ordinary and method HBM, critical compute, total accelerator-hours, block
topology, communication bytes, and failed preregistered gates.
