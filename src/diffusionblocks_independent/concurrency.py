"""Single-device concurrency primitives for independent local block jobs.

The ordinary worker intentionally materializes scalar metrics after every step.
That is convenient for a standalone process but synchronizes CUDA.  The helpers
here keep losses device-resident until a complete block job has been enqueued,
which permits independent jobs to occupy distinct CUDA streams.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StreamInterval:
    """One measured device interval relative to a shared CUDA epoch."""

    block_id: int
    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


def summarize_intervals(intervals: Sequence[StreamInterval]) -> dict[str, Any]:
    """Return exact interval-union and concurrency statistics."""

    if not intervals:
        raise ValueError("at least one interval is required")
    ordered = sorted(intervals, key=lambda item: (item.start_seconds, item.end_seconds))
    for item in ordered:
        if (
            item.block_id < 0
            or not math.isfinite(item.start_seconds)
            or not math.isfinite(item.end_seconds)
            or item.start_seconds < 0
            or item.end_seconds < item.start_seconds
        ):
            raise ValueError("intervals must be finite, nonnegative, and ordered")
    # End events sort before starts at the same timestamp so touching jobs do
    # not count as overlapping.
    events = (
        sorted(
            ((item.start_seconds, 1) for item in ordered),
            key=lambda event: (event[0], -event[1]),
        )
        + []
    )
    events.extend((item.end_seconds, -1) for item in ordered)
    events.sort(key=lambda event: (event[0], event[1]))
    active = 0
    previous = events[0][0]
    union = 0.0
    overlapped = 0.0
    maximum = 0
    concurrency_integral = 0.0
    for timestamp, delta in events:
        span = timestamp - previous
        if active > 0:
            union += span
            concurrency_integral += span * active
        if active > 1:
            overlapped += span
        active += delta
        maximum = max(maximum, active)
        previous = timestamp
    summed = sum(item.duration_seconds for item in ordered)
    return {
        "interval_count": len(ordered),
        "span_seconds": max(item.end_seconds for item in ordered)
        - min(item.start_seconds for item in ordered),
        "union_seconds": union,
        "summed_interval_seconds": summed,
        "overlapped_union_seconds": overlapped,
        "overlap_fraction_of_union": overlapped / union if union else 0.0,
        "average_active_intervals": concurrency_integral / union if union else 0.0,
        "maximum_active_intervals": maximum,
    }


def enqueue_local_step(
    program,
    batch,
    optimizer,
    *,
    sigma: float,
    generator,
):
    """Enqueue one local step without synchronizing a CUDA scalar to Python."""

    import torch

    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    target_index = int(program.manifest["target_batch_index"])
    raw_target = batch[target_index]
    target = program.target_codec.encode(raw_target).detach()
    noise = torch.randn(
        target.shape,
        device=target.device,
        dtype=target.dtype,
        generator=generator,
    )
    context_names = tuple(program.manifest.get("context_names", ()))
    context_indices = tuple(program.manifest.get("context_batch_indices", ()))
    context = (
        {
            name: batch[index]
            for name, index in zip(context_names, context_indices, strict=True)
        }
        if context_names
        else None
    )
    optimizer.zero_grad(set_to_none=True)
    prediction = program.block(
        target + float(sigma) * noise,
        torch.tensor(float(sigma), device=target.device),
        context,
    )
    loss = program.target_codec.loss(prediction, raw_target)
    loss.backward()
    optimizer.step()
    return loss.detach()


def run_cuda_block_jobs(
    *,
    programs: Sequence[Any],
    batches: Sequence[Sequence[Any]],
    optimizers: Sequence[Any],
    sigmas: Sequence[Sequence[float]],
    generators: Sequence[Any],
    concurrent: bool,
    nvtx_range_name: str | None = None,
    cuda_profiler_range: bool = False,
) -> dict[str, Any]:
    """Measure persistent local jobs serially or on independent CUDA streams.

    Each block executes all of its listed steps as one persistent job.  A
    shared epoch event makes event timestamps comparable across streams.
    """

    import torch

    count = len(programs)
    if not count or not (
        len(batches) == len(optimizers) == len(sigmas) == len(generators) == count
    ):
        raise ValueError("all job collections must have the same nonzero length")
    parameters = [next(program.block.parameters()) for program in programs]
    devices = {parameter.device for parameter in parameters}
    if len(devices) != 1 or next(iter(devices)).type != "cuda":
        raise ValueError("all block jobs must reside on one CUDA device")
    device = next(iter(devices))
    if any(not schedule for schedule in sigmas):
        raise ValueError("every block job needs at least one sigma")
    if any(len(batch_schedule) == 0 for batch_schedule in batches):
        raise ValueError("every block job needs at least one batch")

    streams = (
        [torch.cuda.Stream(device=device) for _ in range(count)]
        if concurrent
        else [torch.cuda.Stream(device=device)] * count
    )
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
    epoch = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    if cuda_profiler_range:
        torch.cuda.cudart().cudaProfilerStart()
    if nvtx_range_name:
        torch.cuda.nvtx.range_push(nvtx_range_name)
    try:
        epoch.record(torch.cuda.current_stream(device))
        wall_started = time.perf_counter()
        final_losses = []
        for block_id, (program, optimizer, schedule, generator, stream) in enumerate(
            zip(programs, optimizers, sigmas, generators, streams, strict=True)
        ):
            stream.wait_event(epoch)
            with torch.cuda.stream(stream):
                starts[block_id].record(stream)
                loss = None
                for step, sigma in enumerate(schedule):
                    loss = enqueue_local_step(
                        program,
                        batches[block_id][step % len(batches[block_id])],
                        optimizer,
                        sigma=sigma,
                        generator=generator,
                    )
                ends[block_id].record(stream)
                final_losses.append(loss)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - wall_started
    finally:
        if nvtx_range_name:
            torch.cuda.nvtx.range_pop()
        if cuda_profiler_range:
            torch.cuda.cudart().cudaProfilerStop()
    intervals = [
        StreamInterval(
            block_id=block_id,
            start_seconds=epoch.elapsed_time(starts[block_id]) / 1000.0,
            end_seconds=epoch.elapsed_time(ends[block_id]) / 1000.0,
        )
        for block_id in range(count)
    ]
    summary = summarize_intervals(intervals)
    return {
        "mode": "concurrent_cuda_streams" if concurrent else "sequential_cuda_stream",
        "block_count": count,
        "steps_per_block": [len(schedule) for schedule in sigmas],
        "total_block_updates": sum(len(schedule) for schedule in sigmas),
        "wall_seconds": elapsed,
        "updates_per_second": sum(len(schedule) for schedule in sigmas) / elapsed,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "final_losses": [float(loss.cpu()) for loss in final_losses],
        "intervals": [
            {
                "block_id": item.block_id,
                "start_seconds": item.start_seconds,
                "end_seconds": item.end_seconds,
                "duration_seconds": item.duration_seconds,
            }
            for item in intervals
        ],
        "interval_summary": summary,
        "torch_distributed_initialized": torch.distributed.is_initialized(),
        "inter_block_gradient_bytes": 0,
    }
