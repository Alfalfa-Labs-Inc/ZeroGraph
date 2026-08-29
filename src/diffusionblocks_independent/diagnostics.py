"""Measured leakage and local-capacity probes for serialized independent blocks."""

from __future__ import annotations

import gc
import importlib
import math
from pathlib import Path

from .api import inspect_program


def _load_callable(descriptor: str):
    module_name, separator, callable_name = descriptor.partition(":")
    if not separator or not module_name or not callable_name:
        raise ValueError("batch factory must use module.path:callable syntax")
    value = getattr(importlib.import_module(module_name), callable_name)
    if not callable(value):
        raise TypeError("batch factory descriptor does not resolve to a callable")
    return value


def _move(value, *, device):
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, tuple):
        return tuple(_move(item, device=device) for item in value)
    if isinstance(value, list):
        return [_move(item, device=device) for item in value]
    if isinstance(value, dict):
        return {key: _move(item, device=device) for key, item in value.items()}
    return value


def _permuted_context(context, *, batch_size: int):
    import torch

    changed = []
    permuted = {}
    for name, value in context.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim > 0
            and value.shape[0] == batch_size
            and batch_size > 1
        ):
            permuted[name] = torch.roll(value, shifts=1, dims=0)
            changed.append(name)
        else:
            permuted[name] = value
    return permuted, changed


def diagnose_program(
    directory: str | Path,
    *,
    batch_factory: str,
    probe_batches: int = 2,
    sigma_points: int = 3,
    device: str = "cpu",
    seed: int = 2041,
    trusted_public_key_path: str | Path | None = None,
    require_signature: bool = False,
) -> dict:
    """Probe every block while keeping only one serialized block resident."""

    import torch

    from diffusionblocks import load_pretrained_block
    from diffusionblocks.diagnostics import local_capacity_ratio

    if probe_batches <= 0 or sigma_points < 2 or seed < 0:
        raise ValueError("probe_batches must be positive and sigma_points at least two")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diagnostics requested but CUDA is unavailable")
    produced = _load_callable(batch_factory)()
    if isinstance(produced, dict):
        produced = produced.get("batches")
    batches = tuple(produced)
    if not batches:
        raise ValueError("batch factory produced no batches")
    batches = tuple(_move(batch, device=target_device) for batch in batches)
    inspection = inspect_program(
        directory,
        trusted_public_key_path=trusted_public_key_path,
        require_signature=require_signature,
    )
    reports = []
    with torch.no_grad():
        for block_id in range(inspection["block_count"]):
            local = load_pretrained_block(
                directory,
                block_id,
                trusted_public_key_path=trusted_public_key_path,
                require_signature=require_signature,
            )
            local.block.to(target_device)
            local.target_codec.to(target_device)
            target_index = int(local.manifest["target_batch_index"])
            context_names = tuple(local.manifest.get("context_names", ()))
            context_indices = tuple(local.manifest.get("context_batch_indices", ()))
            interval = local.interval
            log_low, log_high = (
                math.log(interval.sigma_low),
                math.log(interval.sigma_high),
            )
            fractions = [
                0.1 + 0.8 * index / (sigma_points - 1) for index in range(sigma_points)
            ]
            generator = torch.Generator(device=target_device.type).manual_seed(
                seed + block_id
            )
            probes = []
            for fraction in fractions:
                sigma = math.exp(log_low + fraction * (log_high - log_low))
                full_total = 0.0
                permuted_total = 0.0
                ablated_batches = 0
                changed_names = set()
                for batch_index in range(probe_batches):
                    batch = batches[batch_index % len(batches)]
                    raw_target = batch[target_index]
                    target = local.target_codec.encode(raw_target).detach()
                    noise = torch.randn(
                        target.shape,
                        dtype=target.dtype,
                        device=target.device,
                        generator=generator,
                    )
                    noisy = target + sigma * noise
                    context = {
                        name: batch[index]
                        for name, index in zip(
                            context_names, context_indices, strict=True
                        )
                    }
                    prediction = local.block(
                        noisy,
                        torch.tensor(sigma, device=target.device),
                        context or None,
                    )
                    full_total += float(local.target_codec.loss(prediction, raw_target))
                    permuted, changed = _permuted_context(
                        context, batch_size=int(target.shape[0])
                    )
                    if changed:
                        changed_names.update(changed)
                        without = local.block(
                            noisy,
                            torch.tensor(sigma, device=target.device),
                            permuted,
                        )
                        permuted_total += float(
                            local.target_codec.loss(without, raw_target)
                        )
                        ablated_batches += 1
                full_loss = full_total / probe_batches
                measured = ablated_batches == probe_batches
                permuted_loss = (
                    permuted_total / ablated_batches if ablated_batches else None
                )
                delta = permuted_loss - full_loss if measured else None
                probes.append(
                    {
                        "sigma": sigma,
                        "full_context_loss": full_loss,
                        "permuted_context_loss": permuted_loss,
                        "context_delta": delta,
                        "context_delta_fraction": (
                            delta / max(full_loss, 1e-12) if measured else None
                        ),
                        "leakage_warning": (
                            delta <= 0.01 * full_loss if measured else None
                        ),
                        "ablation_measured": measured,
                        "permuted_context_names": sorted(changed_names),
                    }
                )
            full_loss = sum(row["full_context_loss"] for row in probes) / len(probes)
            reports.append(
                {
                    "block_id": block_id,
                    "parameters": sum(
                        parameter.numel()
                        for parameter in local.block.parameters()
                        if parameter.requires_grad
                    ),
                    "full_context_loss": full_loss,
                    "sigma_probes": probes,
                    "leakage_warning": probes[0]["leakage_warning"],
                    "leakage_status": (
                        "measured_context_permutation"
                        if probes[0]["ablation_measured"]
                        else "not_measurable_no_batch_aligned_context"
                    ),
                }
            )
            del local
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    median_loss = sorted(row["full_context_loss"] for row in reports)[len(reports) // 2]
    for row in reports:
        difficulty = max(row["full_context_loss"] / max(median_loss, 1e-12), 1e-12)
        row["normalized_difficulty"] = difficulty
        row["capacity_ratio"] = local_capacity_ratio(row["parameters"], difficulty)
    return {
        "format_version": 1,
        "status": "measured",
        "program_manifest_sha256": inspection["manifest_sha256"],
        "block_count": inspection["block_count"],
        "blocks": reports,
        "probe_batches": probe_batches,
        "sigma_points_per_block": sigma_points,
        "leakage_threshold_fraction": 0.01,
        "leakage_protocol": "paired_same_noise_context_batch_permutation_v1",
        "capacity_protocol": "parameters_over_median_normalized_full_loss_v1",
        "program_authenticated": inspection["authenticity"]["authenticated"],
        "peak_resident_diffusion_blocks": 1,
        "claim_scope": (
            "measured block-local diagnostic; permutation tests context association, "
            "not complete absence of every possible information channel"
        ),
    }
