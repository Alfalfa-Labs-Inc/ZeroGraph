"""ZeroGraph: materialized supervision for independent block training."""

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
from .materialized import (
    MATERIALIZED_CACHE_KIND,
    MaterializedBatch,
    MaterializedBoundaryCache,
    materialize_boundaries,
    train_materialized_block,
)

__version__ = "0.4.0"

__all__ = [
    "INDEPENDENT_CLAIM_SCOPE",
    "MATERIALIZED_CACHE_KIND",
    "IndependentContract",
    "MaterializedBatch",
    "MaterializedBoundaryCache",
    "StreamInterval",
    "benchmark_modes",
    "compile_checkpoint",
    "diagnose_program",
    "enqueue_local_step",
    "inspect_program",
    "launch_plan",
    "load_assembled",
    "load_program",
    "materialize_boundaries",
    "run_cuda_block_jobs",
    "sign_distributed_checkpoint",
    "summarize_intervals",
    "train_materialized_block",
    "verify_distributed_checkpoint",
]
