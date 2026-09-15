<div align="center">

# ZeroGraph

### Materialize the trajectory. Train the blocks independently.

Clean teacher-boundary supervision for independently schedulable PyTorch jobs.

[![CI](https://github.com/Alfalfa-Labs-Inc/ZeroGraph/actions/workflows/ci.yml/badge.svg)](https://github.com/Alfalfa-Labs-Inc/ZeroGraph/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

</div>

ZeroGraph moves cross-depth information out of the synchronous backward path.
A frozen teacher is run ahead of optimization, its clean partition-boundary
states are stored in an integrity-checked cache, and each student block then
trains as an independent job. The teacher body and neighboring trainable blocks
are absent from the local training loop.

This is the configuration that passed the language-quality gates. The older
full-noise local objective remains experimental because its TinyLlama arms did
not preserve coherent generation.

## Measured results

Three-seed Qwen2.5-1.5B/UltraChat comparison on four NVIDIA L4 GPUs:

| Metric | FSDP2 ordinary SFT | Cached ZeroGraph | Result |
|---|---:|---:|---:|
| Mean held-out CE | 1.4362 +/- 0.0031 | **1.2766 +/- 0.0011** | -0.1596 nat |
| Local critical path | 4,933 s mean | 510 s mean | **9.68x shorter** |
| Peak allocated HBM | 8.024 GB | 7.997 GB | effectively equal |
| Peak reserved HBM | 12.012 GB | 8.997 GB | **1.335x lower** |
| Inter-block gradient bytes | global backward | **0** | strict independence |
| Seeds passing all gates | 3/3 | **3/3** | replicated |

The 9.68x number is the measured **local optimization critical path**. It
excludes one-time teacher materialization, persistent cache storage, checkpoint
I/O, assembly, and evaluation. It is not a universal end-to-end speed claim.

The completed evidence-first campaign adds ZeRO-3, objective-matched
online/cached pairs, cache-inclusive reuse through K=16, B=2/4/8, boundary
robustness, independent restart, and Qwen2.5-3B. The 3.086B conversion passed
all quality gates at CE 1.3002 with zero inter-block gradient bytes. Cached
target acquisition was faster after materialization, but did **not** reach
end-to-end break-even by K=16. See the [paper and evidence](paper/README.md).

TinyLlama 1.1B independently confirmed the method: cached conversion reached CE
1.3096 versus 1.2791 for ordinary SFT and 1.3256 for the teacher, with zero
inter-block gradient bytes. See [BENCHMARKS.md](BENCHMARKS.md) for the complete
positive and negative record.

## Why ZeroGraph

- One optimizer and one backward graph per depth partition.
- Zero strict-mode inter-block gradient communication.
- A frozen, reusable cache replaces a live teacher during local optimization.
- Cache shards and manifests are SHA-256 verified before use.
- Attention masks, positional inputs, and final task losses are supported through
  explicit callbacks instead of architecture guesses.
- FSDP, tensor parallelism, context parallelism, and checkpointing remain usable
  inside an oversized block.

## Install

```bash
git clone https://github.com/Alfalfa-Labs-Inc/ExactGraph.git
python -m pip install ./ExactGraph

git clone https://github.com/Alfalfa-Labs-Inc/ZeroGraph.git
python -m pip install -e './ZeroGraph[dev]'
```

The distribution name remains `diffusionblocks-independent`. Version 0.5.0
depends on `diffusionblocks-v5==0.17.0` for the guarded partition/compiler
runtime. Both `zerograph` and `diffusionblocks-independent` invoke the CLI.

## Python quick start

The extractor defines the teacher trajectory. It may return a sequence of clean
boundaries or `MaterializedBatch(boundaries, context)` when the student needs
tensor metadata such as an attention mask.

```python
import torch
from diffusionblocks_independent import (
    MaterializedBatch,
    MaterializedBoundaryCache,
    materialize_boundaries,
    train_materialized_block,
)

def extract_boundaries(teacher, batch):
    hidden = batch["inputs"]
    boundaries = [hidden]
    for partition in teacher.partitions:
        hidden = partition(hidden, attention_mask=batch["attention_mask"])
        boundaries.append(hidden)
    return MaterializedBatch(
        boundaries,
        context={"attention_mask": batch["attention_mask"]},
    )

materialize_boundaries(
    teacher,
    training_batches,
    extract_boundaries,
    "boundary-cache",
    teacher_id="org/model",
    teacher_revision="full-commit-sha",
    data_id="sha256:tokenized-training-stream",
)

cache = MaterializedBoundaryCache("boundary-cache")
block = build_student_block(block_id=0).cuda()
optimizer = torch.optim.AdamW(block.parameters(), lr=2e-5)

report = train_materialized_block(
    block,
    cache,
    block_id=0,
    optimizer=optimizer,
    steps=2048,
    device="cuda",
    forward=lambda module, source, context: module(
        source, attention_mask=context["attention_mask"]
    ),
)

assert report["inter_block_gradient_bytes"] == 0
assert report["live_teacher_parameters"] == 0
```

Run the same call for every block on a separate device or distributed group.
The package never silently synchronizes parameters shared between blocks;
freeze them, make them block-private, or account for synchronization explicitly.

## CLI workflow

Factories and callbacks use `module.path:name` descriptors:

```bash
zerograph materialize \
  --teacher-factory my_project.zero:teacher \
  --batch-factory my_project.zero:batches \
  --boundary-extractor my_project.zero:extract \
  --teacher-id org/model \
  --teacher-revision FULL_COMMIT_SHA \
  --data-id sha256:TOKEN_STREAM_DIGEST \
  --output-dir boundary-cache

zerograph train-materialized \
  --block-factory my_project.zero:block \
  --cache boundary-cache \
  --block-id 0 \
  --steps 2048 \
  --device cuda \
  --learning-rate 2e-5 \
  --forward my_project.zero:forward_block \
  --loss my_project.zero:boundary_and_token_loss \
  --checkpoint runs/block-00.pt \
  --output runs/block-00.json
```

Launch one `train-materialized` command per block. Each worker loads its block
and cache records; it does not load the live teacher.

## Legacy experimental diffusion mode

The original compiler and noise-interval runner remain available for research:

```python
from diffusionblocks_independent import compile_checkpoint

program = compile_checkpoint(**model_payload, checkpoint=None, blocks=4)
program.save_pretrained("experimental-noise-program")
```

This mode changes the learning objective and has not passed the same language
generation gates as clean materialized supervision. Do not treat decreasing
local denoising loss as proof of assembled-model quality.

## Contract

```python
from diffusionblocks_independent import IndependentContract
print(IndependentContract())
```

The stable 0.4.0 contract declares:

- `objective = clean_teacher_boundary_regression_sigma_zero`
- `teacher_trajectory_required = True`
- `teacher_live_during_local_training = False`
- `inter_block_gradient_bytes = 0`
- `ordinary_gradient_parity = False`
- `bitwise_ordinary_training_parity = False`

ZeroGraph changes the learning signal. [ExactGraph](https://github.com/Alfalfa-Labs-Inc/ExactGraph)
is the separate objective-preserving product that retains exact downstream
adjoints and certifies ordinary-update parity or rejects execution.

## What is not claimed

- From-scratch independent foundation-model pretraining.
- Ordinary-gradient or bitwise-training parity.
- Universal support for arbitrary PyTorch graphs.
- Universal total-HBM reduction.
- A 9.68x cache-inclusive, end-to-end speedup.
- That full-noise DiffusionBlocks preserves language quality.

## Project

- [Complete benchmark record](BENCHMARKS.md)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Code of conduct](CODE_OF_CONDUCT.md)

ZeroGraph is an Alfalfa Labs, Inc. project under the [Apache-2.0 license](LICENSE).
The DiffusionBlocks method originates with Shing, Koyama, and Akiba; see
[NOTICE](NOTICE) for attribution.
