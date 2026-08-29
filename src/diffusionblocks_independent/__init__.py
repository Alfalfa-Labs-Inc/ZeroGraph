"""Approximate independently parallel companion to DiffusionBlocks Exact."""

from .api import (
    INDEPENDENT_CLAIM_SCOPE,
    IndependentContract,
    compile_checkpoint,
    inspect_program,
    launch_plan,
    load_assembled,
    load_program,
    sign_distributed_checkpoint,
    verify_distributed_checkpoint,
)
from .benchmark import benchmark_modes
from .concurrency import (
    StreamInterval,
    enqueue_local_step,
    run_cuda_block_jobs,
    summarize_intervals,
)
from .diagnostics import diagnose_program

__version__ = "0.3.0"

__all__ = [
    "INDEPENDENT_CLAIM_SCOPE",
    "IndependentContract",
    "StreamInterval",
    "benchmark_modes",
    "compile_checkpoint",
    "diagnose_program",
    "enqueue_local_step",
    "inspect_program",
    "launch_plan",
    "load_assembled",
    "load_program",
    "run_cuda_block_jobs",
    "sign_distributed_checkpoint",
    "summarize_intervals",
    "verify_distributed_checkpoint",
]
