# ZeroGraph 0.3.0

ZeroGraph is the deliberately non-parity, independently parallel DiffusionBlocks
package from Alfalfa Labs and the companion to ExactGraph 0.17.0. It reuses the
core automatic checkpoint converter,
but changes the training objective: every residual block learns its assigned
denoising interval with its own optimizer and no inter-block gradient traffic.

The contract is explicit:

- no bitwise or gradient parity with ordinary end-to-end training;
- one independently loadable and trainable block per job;
- zero inter-block gradient bytes in frozen/private-interface mode;
- smaller active model-state working set, subject to codec/shared-interface and
  runtime overhead;
- concurrent wall time can improve when blocks have disjoint hardware, but
  measured speedup and assembled quality are required for every model.

The tested distribution name remains `diffusionblocks-independent`, and it
depends on `diffusionblocks-v5==0.17.0` for the shared compiler/runtime. Installing
this package does not replace or mutate ExactGraph.

Version 0.2.1 incorporates issues found by a real 4.0269B-parameter training
run: CUDA RNG tensors are restored through CPU byte tensors, deserialized resume
payloads are released before working-set measurement, and an explicit
`--override-resume-learning-rate` option supports audited optimizer calibration.
The 4B run measured a 9.85x optimized local-training memory reduction and a
59.59% assembled-MSE reduction from its fixed synthetic-noise baseline. It did
not establish language quality, convergence, multi-GPU scaling, or a general
speedup.

Version 0.3.0 adds an explicitly asynchronous single-device scheduling API.
`enqueue_local_step` leaves its scalar loss on the device instead of forcing a
host synchronization, and `run_cuda_block_jobs` can place persistent block jobs
on independent CUDA streams. This is primarily a validation and opportunistic
overlap mode: separate streams on one GPU still contend for the same SMs and
memory bandwidth, so it is not a substitute for disjoint devices.

## Persistent single-device concurrency

```python
from diffusionblocks_independent import run_cuda_block_jobs

report = run_cuda_block_jobs(
    programs=loaded_local_programs,
    batches=per_block_batch_schedules,
    optimizers=per_block_optimizers,
    sigmas=per_block_sigma_schedules,
    generators=per_block_cuda_generators,
    concurrent=True,
)
```

All programs must be independently loaded on one CUDA device. The result
contains shared-epoch CUDA event intervals, their exact union and overlap,
wall time, peak allocated memory, and an explicit zero inter-block-gradient
field. `concurrent=False` uses one CUDA stream and provides the matched
persistent sequential control. Callers must use identical initial block states,
batch/sigma schedules, and per-block generators when comparing schedules.

The shipped 4.0269B fixture exercised ten 402.7M-parameter blocks for two
warmup and 20 measured updates per block. Concurrent and sequential schedules
produced identical SHA-256 digests for every final block. The persistent
sequential control took 5.4339 seconds and ten streams took 5.2327 seconds, a
measured 1.038x speedup on one GB10, while peak allocated memory rose from
16.157 GB to 18.108 GB. Nsight independently captured 8,150 kernels across ten
streams; 27.52 ms of kernel-active time contained kernels from two streams.
No NCCL/c10d collective kernel appeared. This proves concurrent schedulability
on that single GPU, not multi-GPU speedup or a general throughput advantage.

## Python conversion

```python
from diffusionblocks_independent import compile_checkpoint

program = compile_checkpoint(
    model,
    checkpoint="model.safetensors",
    optimizer=optimizer,
    step_fn=step_fn,
    example_args=example_args,
    example_batches=example_batches,
    blocks=4,
)
program.save_pretrained("compiled-independent")
```

Only compiler-accepted residual tensor programs are converted. A checkpoint
does not contain its architecture, loss, optimizer, or guarded examples, so the
caller must supply them through the same executable factory contract as the
core compiler.

## Inspect and launch

```bash
diffusionblocks-independent inspect \
  --program compiled-independent --output independent-plan.json

diffusionblocks-independent launch \
  --program compiled-independent \
  --batch-factory my_project.data:make_batches \
  --steps 1000 --devices 0,1,2,3 \
  --output-dir runs/independent
```

With four blocks and four GPUs, the launcher starts four separate block jobs.
With fewer device groups it schedules waves. Use `--dry-run` to emit the exact
commands without executing them. Multi-device blocks use
`--processes-per-block` and the core FSDP/TP/CP trainer.

The launch report separates elapsed wall time from aggregate block-job time.
Neither the theoretical state ratio nor concurrency is labeled a speedup.

Oversized blocks can compose three inner mesh axes. Their product must equal
the process count assigned to that block:

```bash
diffusionblocks-independent launch \
  --program compiled-independent \
  --batch-factory my_project.data:make_batches \
  --steps 1000 --devices 0,1,2,3,4,5,6,7 \
  --processes-per-block 8 \
  --fsdp-degree 2 --tensor-parallel-degree 2 --context-parallel-degree 2 \
  --output-dir runs/hierarchical
```

The converter and launcher reject inconsistent meshes instead of silently
changing a requested degree. Tensor parallel execution remains subject to the
core package's generated-graph plan/certificate requirements.

## Authentication and deterministic resume

Program signatures are checked against a caller-pinned public key. Single-file
block checkpoints receive detached signatures; distributed DCP checkpoints
receive a signed canonical manifest covering `latest.json` and every regular
file in the referenced checkpoint tree. A signed resume verifies the prior
snapshot before starting another process group.

```bash
diffusionblocks-independent launch \
  --program compiled-independent \
  --batch-factory my_project.data:make_batches \
  --steps 1000 --devices 0,1,2,3 \
  --trusted-public-key release-public.pem --require-signature \
  --checkpoint-signing-private-key run-private.pem \
  --trusted-checkpoint-public-key run-public.pem \
  --require-checkpoint-signature \
  --output-dir runs/authenticated
```

When a resumed optimizer must deliberately use a newly calibrated learning
rate, pass both `--resume` and `--override-resume-learning-rate`. The override is
recorded in every worker report and currently applies only to single-process
block workers. Without it, the optimizer state retains its saved learning rate.

Private keys are paths supplied by the operator; their contents are never
embedded in a launch plan. Unencrypted keys are supported by this noninteractive
CLI. Use an access-controlled run key rather than a long-lived release key.

## Leakage and local-capacity diagnostics

```bash
diffusionblocks-independent diagnose \
  --program compiled-independent \
  --batch-factory my_project.data:validation_batches \
  --probe-batches 4 --sigma-points 3 \
  --output diagnostics.json
```

The probe streams one block at a time. For batch-aligned context it compares
full and batch-permuted context with identical noise, avoiding the invalid
assumption that token ID zero means "no context." When no meaningful
permutation exists it reports the leakage measurement as unavailable. Capacity
ratios are descriptive diagnostics based on measured loss difficulty, not a
guarantee that a chosen block count will preserve assembled quality.

## Three-mode benchmark

```bash
diffusionblocks-independent benchmark \
  --model-factory my_project.model:make_training_payload \
  --batch-factory my_project.data:make_batches \
  --blocks 4 --steps 20 --devices 0,1,2,3 \
  --output-dir evidence/three-mode
```

Use `--independent-steps-per-block N` when constructing an aggregate-work or
accelerator-time-matched arm. `--steps` always controls ordinary and Exact;
without the override, Independent receives the same number of examples per
block. The report independently marks per-block sample matching, aggregate
block-example matching, and aggregate-seconds matching within a fixed 20% band.
Matching one view generally breaks another, so preserve both arms.

The model factory returns the same keyword payload accepted by
`compile_checkpoint`. The report keeps ordinary task loss, Exact's guarded
bitwise benchmark, and Independent assembled denoising quality separate. It
reports both elapsed and aggregate accelerator time and states whether sample
or accelerator work is actually matched. A small benchmark is evidence only
for that model, hardware, step count, and batch shape.

Load the independently trained checkpoints only when it is time to evaluate the
assembled model:

```python
from diffusionblocks_independent import load_assembled

program = load_assembled(
    "compiled-independent",
    [
        "runs/independent/block_00/checkpoint.pt",
        "runs/independent/block_01/checkpoint.pt",
    ],
)
result = program.assembled_reverse(initial_noise, context)
```

Checkpoint digests, block order, and the originating compiled-program digest
are verified before weights are assembled.

If a program directory already contains trained PT2 block weights, load it
directly after Independent-contract and optional signature verification:

```python
from diffusionblocks_independent import load_program

program = load_program(
    "trained-independent-program",
    trusted_public_key_path="release-public.pem",
    require_signature=True,
)
samples = program.assembled_reverse(initial_noise, context)
```
