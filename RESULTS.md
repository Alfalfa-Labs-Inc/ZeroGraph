# ZeroGraph 0.4.0 result card

## Stable product

ZeroGraph 0.4.0 trains residual depth partitions against clean cached teacher
boundary states (`sigma=0`). After the cache is materialized, each block owns
its optimizer and backward graph; the live teacher and neighboring trainable
blocks are absent. Strict mode records zero inter-block gradient bytes.

## Qwen2.5-1.5B replication

| Seed | FSDP2 CE | ZeroGraph CE | Local critical-path ratio | Result |
|---:|---:|---:|---:|---:|
| 20260910 | 1.43768 | **1.27789** | 9.89x | Pass |
| 20260911 | 1.43825 | **1.27619** | 9.53x | Pass |
| 20260912 | 1.43262 | **1.27572** | 9.62x | Pass |

- Mean CE: **1.27660 +/- 0.00114** for ZeroGraph versus **1.43618 +/- 0.00310** for FSDP2.
- Mean online local-critical-path ratio: **9.68x shorter**.
- Peak allocated HBM: 7.997 GB versus 8.024 GB; effectively equal.
- Peak reserved HBM: 8.997 GB versus 12.012 GB; **1.335x lower**.
- ZeroGraph gates: **3/3 passed** for quality, coherence, context use, and finite values.
- Strict inter-block gradient traffic: **0 bytes**.

## TinyLlama 1.1B confirmation

- Cached conversion CE: **1.30959**.
- Ordinary SFT CE: **1.27910**.
- Frozen teacher CE: **1.32563**.
- Cached slowest-block compute: **395.20 s** versus **1,936.41 s** ordinary critical compute.
- Cached peak allocated HBM: **2.865 GB** versus **13.227 GB** ordinary.
- Online/cached checkpoint hashes: **4/4 exact matches**.
- Strict inter-block gradient traffic: **0 bytes**.

## Negative results retained

- Unanchored GPT-2 Large reduced local losses but failed context and generation retention.
- TinyLlama cached full-noise, online full-noise, and hierarchical full-noise arms failed coherence.
- Therefore full-noise denoising remains experimental and is not the 0.4.0 stable product.

## Claim boundary

The 9.68x result is the local optimization critical path on disjoint L4 GPUs.
It excludes cache construction, storage, checkpoint I/O, assembly, and final
evaluation. ZeroGraph 0.4.0 is a pretrained-model conversion/distillation
system, not established from-scratch pretraining or ordinary-gradient parity.

See [BENCHMARKS.md](BENCHMARKS.md) for protocols and the full comparison.
