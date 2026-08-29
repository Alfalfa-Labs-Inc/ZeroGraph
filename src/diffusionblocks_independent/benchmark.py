"""Matched evidence harness for ordinary, exact, and independent training modes."""

from __future__ import annotations

import gc
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .api import compile_checkpoint, inspect_program, load_assembled


def _callable(descriptor: str):
    module_name, separator, callable_name = descriptor.partition(":")
    if not separator or not module_name or not callable_name:
        raise ValueError("factory must use module.path:callable syntax")
    value = getattr(importlib.import_module(module_name), callable_name)
    if not callable(value):
        raise TypeError("factory descriptor does not resolve to a callable")
    return value


def _batches(descriptor: str):
    produced = _callable(descriptor)()
    if isinstance(produced, dict):
        produced = produced.get("batches")
    result = tuple(produced)
    if not result:
        raise ValueError("batch factory produced no batches")
    return result


def _batch_size(batch) -> int:
    import torch

    for value in batch:
        if isinstance(value, torch.Tensor) and value.ndim:
            return int(value.shape[0])
    raise ValueError("benchmark batch has no batched tensor")


def _tensor_bytes(value) -> int:
    import torch

    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _move(value, device):
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _ordinary_run(payload: dict, batches, steps: int) -> dict:
    import torch

    model = payload["model"]
    optimizer = payload["optimizer"]
    step_fn = payload["step_fn"]
    device_parameter = next(model.parameters(), None)
    cuda_device = (
        device_parameter.device
        if device_parameter is not None and device_parameter.is_cuda
        else None
    )
    active_batches = tuple(
        _move(batch, cuda_device or torch.device("cpu")) for batch in batches
    )
    if cuda_device is not None:
        torch.cuda.reset_peak_memory_stats(cuda_device)
        torch.cuda.synchronize(cuda_device)
    started = time.perf_counter()
    final_loss = None
    for index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        value = step_fn(model, active_batches[index % len(active_batches)])
        loss = value[0] if isinstance(value, tuple) else value
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
    elapsed = time.perf_counter() - started
    with torch.no_grad():
        value = step_fn(model, active_batches[0])
        evaluation_loss = value[0] if isinstance(value, tuple) else value
    return {
        "status": "measured",
        "steps": steps,
        "elapsed_wall_seconds": elapsed,
        "aggregate_accelerator_seconds": elapsed,
        "steps_per_second": steps / elapsed,
        "final_training_loss": final_loss,
        "evaluation_task_loss": float(evaluation_loss.detach()),
        "model_state_bytes": _tensor_bytes(model.state_dict()),
        "optimizer_state_bytes": _tensor_bytes(optimizer.state_dict()),
        "optimizer_class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(cuda_device)
            if cuda_device is not None
            else None
        ),
        "external_gradient_communication_bytes": 0,
    }


def benchmark_modes(
    *,
    model_factory: str,
    batch_factory: str,
    blocks: int,
    steps: int,
    independent_steps_per_block: int | None = None,
    output_directory: str | Path,
    devices: tuple[str, ...],
    precision: str = "fp32",
    learning_rate: float = 1e-4,
    optimizer: str = "adamw",
    exact_warmup_steps: int = 0,
) -> dict:
    """Run one explicit small-scale three-mode benchmark and retain raw artifacts."""

    import torch

    from diffusionblocks import benchmark_exact_rematerialization
    from diffusionblocks.parity import compile as compile_exact

    local_steps = (
        steps if independent_steps_per_block is None else independent_steps_per_block
    )
    if blocks < 2 or steps <= 0 or local_steps <= 0 or not devices:
        raise ValueError(
            "benchmark requires at least two blocks, positive steps/devices"
        )
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError("benchmark output already exists; archive it first")
    output.mkdir(parents=True)
    batches = _batches(batch_factory)
    batch_size = _batch_size(batches[0])

    torch.manual_seed(3101)
    ordinary = _ordinary_run(_callable(model_factory)(), batches, steps)
    ordinary_optimizer_name = ordinary["optimizer_class"].rsplit(".", 1)[-1].lower()
    if ordinary_optimizer_name != optimizer.lower():
        raise ValueError(
            "independent optimizer must match the model factory optimizer; "
            f"factory={ordinary_optimizer_name}, requested={optimizer.lower()}"
        )
    (output / "ordinary.json").write_text(
        json.dumps(ordinary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    torch.manual_seed(3101)
    exact_payload = _callable(model_factory)()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        exact = compile_exact(
            exact_payload["model"],
            optimizer=exact_payload["optimizer"],
            step_fn=exact_payload["step_fn"],
            example_args=exact_payload["example_args"],
            example_kwargs=exact_payload.get("example_kwargs"),
            example_batches=exact_payload["example_batches"],
            dynamic_shapes=exact_payload.get("dynamic_shapes"),
            blocks=blocks,
            exact_rematerialization=True,
        )
        exact_report = benchmark_exact_rematerialization(
            exact,
            _move(batches[0], next(exact.model.parameters()).device),
            warmup_steps=exact_warmup_steps,
            measured_steps=steps,
        )
        (output / "exact-rematerialization.json").write_text(
            json.dumps(exact_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    finally:
        torch.use_deterministic_algorithms(
            previous_deterministic, warn_only=previous_warn_only
        )

    del exact, exact_payload
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    torch.manual_seed(3101)
    independent_payload = _callable(model_factory)()
    program = compile_checkpoint(**independent_payload, checkpoint=None, blocks=blocks)
    program_directory = output / "program"
    program.save_pretrained(program_directory)
    del program, independent_payload
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    run_directory = output / "independent-run"
    command = [
        sys.executable,
        "-m",
        "diffusionblocks_independent.cli",
        "launch",
        "--program",
        str(program_directory),
        "--batch-factory",
        batch_factory,
        "--steps",
        str(local_steps),
        "--output-dir",
        str(run_directory),
        "--devices",
        ",".join(devices),
        "--precision",
        precision,
        "--learning-rate",
        repr(learning_rate),
        "--optimizer",
        optimizer,
    ]
    environment = os.environ.copy()
    completed = subprocess.run(
        command,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    if completed.returncode:
        raise RuntimeError(
            "independent launcher benchmark failed: "
            + (completed.stderr or completed.stdout)[-4000:]
        )
    launch = json.loads(
        (run_directory / "launch-report.json").read_text(encoding="utf-8")
    )
    checkpoints = [
        run_directory / f"block_{block_id:02d}" / "checkpoint.pt"
        for block_id in range(blocks)
    ]
    assembled = load_assembled(program_directory, checkpoints)
    assembled_parameter = next(
        parameter for block in assembled.blocks for parameter in block.parameters()
    )
    batch = _move(batches[0], assembled_parameter.device)
    target_index = int(assembled.manifest["target_batch_index"])
    raw_target = batch[target_index]
    target = assembled.target_codec.encode(raw_target).detach()
    context_names = tuple(assembled.manifest.get("context_names", ()))
    context_indices = tuple(assembled.manifest.get("context_batch_indices", ()))
    context = {
        name: batch[index]
        for name, index in zip(context_names, context_indices, strict=True)
    }
    generator = torch.Generator(device=target.device.type).manual_seed(3102)
    initial_noise = torch.randn(
        target.shape,
        dtype=target.dtype,
        device=target.device,
        generator=generator,
    )
    with torch.no_grad():
        from_noise = assembled.assembled_reverse(initial_noise, context or None)
        zero_sigma = assembled.zero_sigma_assembled(target, context or None)
        from_noise_loss = float(assembled.target_codec.loss(from_noise, raw_target))
        zero_sigma_loss = float(assembled.target_codec.loss(zero_sigma, raw_target))
    inspection = inspect_program(program_directory)
    local_reports = [
        json.loads(
            (run_directory / f"block_{block_id:02d}.json").read_text(encoding="utf-8")
        )
        for block_id in range(blocks)
    ]
    independent = {
        "status": "measured",
        "steps_per_block": local_steps,
        "aggregate_block_updates": local_steps * blocks,
        "elapsed_wall_seconds": launch["elapsed_wall_seconds"],
        "aggregate_accelerator_seconds": launch["aggregate_block_job_seconds"],
        "observed_concurrency_ratio": launch["observed_concurrency_ratio"],
        "assembled_from_noise_loss": from_noise_loss,
        "assembled_zero_sigma_loss": zero_sigma_loss,
        "peak_cuda_allocated_bytes_by_block": [
            row["peak_cuda_allocated_bytes"] for row in local_reports
        ],
        "inter_block_gradient_bytes": 0,
        "serialized_state_working_set": inspection["state_working_set"],
    }
    report = {
        "format_version": 1,
        "status": "measured",
        "model_factory": model_factory,
        "batch_factory": batch_factory,
        "blocks": blocks,
        "optimizer": optimizer,
        "batch_size": batch_size,
        "ordinary": ordinary,
        "exact_rematerialization": exact_report,
        "independent": independent,
        "sample_token_view": {
            "ordinary_examples": steps * batch_size,
            "exact_examples": steps * batch_size,
            "independent_examples_per_block": local_steps * batch_size,
            "independent_aggregate_block_examples": (local_steps * batch_size * blocks),
            "per_block_matched": local_steps == steps,
            "aggregate_block_examples_matched": local_steps * blocks == steps,
            "matched_scope": ("per-block examples" if local_steps == steps else "none"),
        },
        "accelerator_time_view": {
            "ordinary_aggregate_seconds": ordinary["aggregate_accelerator_seconds"],
            "independent_aggregate_seconds": independent[
                "aggregate_accelerator_seconds"
            ],
            "independent_elapsed_wall_seconds": independent["elapsed_wall_seconds"],
            "aggregate_seconds_ratio_independent_over_ordinary": (
                independent["aggregate_accelerator_seconds"]
                / ordinary["aggregate_accelerator_seconds"]
            ),
            "matched_within_20_percent": (
                0.8
                <= independent["aggregate_accelerator_seconds"]
                / ordinary["aggregate_accelerator_seconds"]
                <= 1.2
            ),
            "reason": (
                "aggregate measured worker time; elapsed concurrent wall time is "
                "reported separately"
            ),
        },
        "quality_metric_scope": {
            "ordinary": "original task loss",
            "exact": "bitwise equality to ordinary at every benchmark step",
            "independent": "assembled target-representation denoising loss",
            "directly_comparable": False,
        },
        "claim_scope": (
            "measured small-scale benchmark; no general speed, memory, quality, or "
            "scaling claim"
        ),
    }
    destination = output / "benchmark.json"
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**report, "artifact": str(destination)}
