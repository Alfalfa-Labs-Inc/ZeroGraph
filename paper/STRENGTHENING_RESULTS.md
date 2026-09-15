# ZeroGraph Paper Strengthening Campaign — Final Results

Status: **VERIFIED**. All nine phases completed; no verifier failures. Total recorded spend: **$33.01** of the $50 ceiling.

## Direct answers in the requested order

1. **Objective-matched comparison:** Clean cached and online teacher-boundary jobs used the same supervision and controls. Cache integrity was bitwise exact (maximum MSE 0). Cached jobs averaged 99.13 s after materialization versus 116.39 s online across 16 paired trials, but the 618.3 s cache made cache-inclusive amortized time 137.77 s at K=16 versus 116.39 s online. **Materialization changed the dependency graph and shortened recurring target acquisition, but did not reach wall-time break-even by K=16.**
2. **Cache-inclusive accounting:** First-use ZeroGraph was 1169.5 s; reuse was 551.1 s. FSDP2 SFT was 5034.3 s; ZeRO-3 was 3496.5 s; globally connected intermediate distillation was 5275.3 s. These objectives are reported separately rather than mislabeled as parity.
3. **Compositionality:** Clean-boundary CE for B=2/4/8 was 1.2937/1.2779/1.2742. Boundary interpolation showed increasing boundary error toward assembled predecessors, while clean-boundary language quality remained robust in the measured range. The separate noisy objective still failed free-running coherence, confirming that local trainability alone is insufficient.
4. **Real-model scale:** Qwen2.5-3B (3,085,938,688 parameters) trained as four independent blocks for 2,048 updates each. Assembled CE was 1.3002 versus base 1.2309 and teacher 1.3600; context delta was +0.509 nat, all gates passed, peak allocated HBM was 13.59 GiB, and inter-block gradient bytes were zero. This validates real pretrained 3B conversion/distillation—not from-scratch pretraining.
5. **Failure/restart:** Block 1 was terminated with SIGTERM halfway, restored independently with optimizer/RNG/sample-order state, and finished bitwise-identically to an uninterrupted deterministic control (hash prefix `238f1f1d7e3e`). Blocks 0, 2, and 3 were not retrained; no global rollback occurred.
6. **Scientific framing:** Supported: **blockwise trainability does not imply compositional trainability**. A coherent boundary trajectory supplies the semantic contract.
7. **Causal result:** Caching itself does not repair a bad objective; cached and online noisy targets fail alike. Clean boundary regression works. Materialization relocates an otherwise identical signal from online computation to persistent data.

## Primary systems table

| Arm | Objective | First use (s) | Reuse (s) | Peak alloc. (GiB) | Held-out CE |
|---|---|---:|---:|---:|---:|
| FSDP2 | ordinary SFT | 5034.3 | 5034.3 | 7.47 | 1.4377 |
| ZeRO-3 | ordinary SFT | 3496.5 | 3496.5 | 11.52 | 1.4399 |
| Global distillation | teacher boundaries online in one connected job | 5275.3 | 4657.0 | 10.71 | 1.3093 |
| ZeroGraph | cached clean teacher boundaries, independent jobs | 1169.5 | 551.1 | 7.45 | 1.2779 |

## Reuse accounting

| K | Cached amortized s/run | Online mean s/run | Favorable? |
|---:|---:|---:|---|
| 1 | 723.98 | 116.35 | no |
| 2 | 412.89 | 117.17 | no |
| 4 | 256.59 | 117.05 | no |
| 8 | 176.47 | 117.17 | no |
| 16 | 137.77 | 116.39 | no |

The authoritative machine-readable source is `artifacts/zerograph_systems_campaign/final-evidence/VERIFIED-SUMMARY.json`.
