"""Single-device block-only worker used by the concurrent launcher."""

from __future__ import annotations

import argparse
import importlib
import json
import time
from hashlib import sha256
from pathlib import Path


def _load_callable(descriptor: str):
    module_name, separator, callable_name = descriptor.partition(":")
    if not separator or not module_name or not callable_name:
        raise ValueError("batch factory must use module.path:callable syntax")
    value = getattr(importlib.import_module(module_name), callable_name)
    if not callable(value):
        raise TypeError("batch factory descriptor does not resolve to a callable")
    return value


def _move(value, *, device, dtype):
    import torch

    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            return value.to(device=device, dtype=dtype)
        return value.to(device=device)
    if isinstance(value, tuple):
        return tuple(_move(item, device=device, dtype=dtype) for item in value)
    if isinstance(value, list):
        return [_move(item, device=device, dtype=dtype) for item in value]
    if isinstance(value, dict):
        return {
            key: _move(item, device=device, dtype=dtype) for key, item in value.items()
        }
    return value


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="diffusionblocks-independent-worker")
    parser.add_argument("--program", required=True)
    parser.add_argument("--batch-factory", required=True)
    parser.add_argument("--block-id", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--optimizer", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--seed", type=int, default=2040)
    state = parser.add_mutually_exclusive_group()
    state.add_argument("--resume", action="store_true")
    state.add_argument("--overwrite", action="store_true")
    parser.add_argument("--override-resume-learning-rate", action="store_true")
    parser.add_argument("--trusted-public-key")
    parser.add_argument("--require-signature", action="store_true")
    parser.add_argument("--checkpoint-signing-private-key")
    parser.add_argument("--trusted-checkpoint-public-key")
    parser.add_argument("--require-checkpoint-signature", action="store_true")
    args = parser.parse_args(argv)

    import torch

    from diffusionblocks import load_pretrained_block

    from .api import inspect_program

    if args.steps <= 0 or args.learning_rate <= 0 or args.seed < 0:
        raise ValueError("steps and learning rate must be positive")
    if args.override_resume_learning_rate and not args.resume:
        raise ValueError("overriding the resume learning rate requires --resume")
    if args.require_checkpoint_signature and not args.trusted_checkpoint_public_key:
        raise ValueError(
            "requiring a checkpoint signature requires a trusted checkpoint public key"
        )
    if args.require_checkpoint_signature and not args.checkpoint_signing_private_key:
        raise ValueError(
            "requiring an authenticated output checkpoint requires a signing private key"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[args.precision]
    produced = _load_callable(args.batch_factory)()
    if isinstance(produced, dict):
        produced = produced.get("batches")
    batches = tuple(produced)
    if not batches:
        raise ValueError("batch factory produced no batches")
    inspection = inspect_program(
        args.program,
        trusted_public_key_path=args.trusted_public_key,
        require_signature=args.require_signature,
    )
    program = load_pretrained_block(
        args.program,
        args.block_id,
        trusted_public_key_path=args.trusted_public_key,
        require_signature=args.require_signature,
    )
    program.block.to(device=device, dtype=dtype)
    program.target_codec.to(device=device, dtype=dtype)
    batches = tuple(_move(batch, device=device, dtype=dtype) for batch in batches)
    optimizer = program.make_optimizer(
        learning_rate=args.learning_rate,
        optimizer_name=args.optimizer,
    )
    checkpoint = Path(args.checkpoint).resolve()
    metadata_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
    if checkpoint.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(
            "checkpoint already exists; pass --resume or --overwrite explicitly"
        )
    start_step = 0
    if args.resume:
        if not checkpoint.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                "resume requested but checkpoint pair is incomplete"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("sha256") != _sha256(checkpoint):
            raise ValueError("resume checkpoint digest mismatch")
        checkpoint_authentication = None
        if args.trusted_checkpoint_public_key:
            from diffusionblocks import verify_artifact_signature

            checkpoint_authentication = verify_artifact_signature(
                checkpoint, args.trusted_checkpoint_public_key
            )
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        if (
            payload.get("kind")
            != "diffusionblocks_serialized_local_training_checkpoint"
            or payload.get("block_id") != args.block_id
            or payload.get("program_manifest_sha256") != inspection["manifest_sha256"]
        ):
            raise ValueError("resume checkpoint provenance does not match this job")
        optimizer_class = f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
        if payload.get("optimizer_class") != optimizer_class:
            raise ValueError("resume optimizer class mismatch")
        program.block.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        if args.override_resume_learning_rate:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = args.learning_rate
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        cuda_states = payload.get("cuda_rng_state_all", [])
        if cuda_states and torch.cuda.is_available():
            # ``map_location=device`` also moves serialized RNG byte tensors.
            # CUDA's restore API deliberately accepts CPU ByteTensors only.
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])
        start_step = int(payload["step"])
        if start_step >= args.steps:
            raise ValueError("resume target steps must exceed the saved step")
        # ``load_state_dict`` has copied checkpoint tensors into the resident
        # block/optimizer.  Release the deserialized payload before measuring
        # the resumed training working set.
        del payload
        import gc

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        checkpoint_authentication = None
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    metrics = None
    for step in range(start_step, args.steps):
        metrics = program.train_local_step(
            batches[step % len(batches)],
            optimizer,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_name(f".{checkpoint.name}.tmp")
    torch.save(
        {
            "format_version": 1,
            "kind": "diffusionblocks_serialized_local_training_checkpoint",
            "claim_scope": "independently_trained_local_denoising_block",
            "program_manifest_sha256": inspection["manifest_sha256"],
            "block_id": args.block_id,
            "step": args.steps,
            "model": program.block.state_dict(),
            "optimizer": optimizer.state_dict(),
            "optimizer_class": (
                f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
            ),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
        },
        temporary,
    )
    temporary.replace(checkpoint)
    checkpoint_authentication = None
    checkpoint_signature = Path(str(checkpoint) + ".signature.json")
    checkpoint_signature.unlink(missing_ok=True)
    if args.checkpoint_signing_private_key:
        from diffusionblocks import sign_artifact

        sign_artifact(checkpoint, args.checkpoint_signing_private_key)
        if args.trusted_checkpoint_public_key:
            from diffusionblocks import verify_artifact_signature

            checkpoint_authentication = verify_artifact_signature(
                checkpoint, args.trusted_checkpoint_public_key
            )
    checkpoint_metadata = {
        "format_version": 1,
        "checkpoint": checkpoint.name,
        "sha256": _sha256(checkpoint),
        "block_id": args.block_id,
        "step": args.steps,
        "program_manifest_sha256": inspection["manifest_sha256"],
        "signature": (
            checkpoint_signature.name if checkpoint_signature.is_file() else None
        ),
        "authenticated": bool(
            checkpoint_authentication
            and checkpoint_authentication.get("authenticity_established")
        ),
    }
    _write_json(
        checkpoint.with_suffix(checkpoint.suffix + ".json"),
        checkpoint_metadata,
    )
    report = {
        "format_version": 1,
        "status": "trained",
        "claim_scope": (
            "single independently loaded local-denoising block; no bitwise ordinary-"
            "training parity or assembled-quality claim"
        ),
        "block_id": args.block_id,
        "start_step": start_step,
        "end_step": args.steps,
        "steps_executed": args.steps - start_step,
        "learning_rate": args.learning_rate,
        "resume_learning_rate_overridden": args.override_resume_learning_rate,
        "seconds": elapsed,
        "steps_per_second": (args.steps - start_step) / elapsed,
        "device": str(device),
        "precision": args.precision,
        "parameter_count": sum(
            parameter.numel() for parameter in program.block.parameters()
        ),
        "full_original_model_loaded": False,
        "inter_block_gradient_bytes": 0,
        "program_authenticated": inspection["authenticity"]["authenticated"],
        "program_signature_required": args.require_signature,
        "checkpoint_authenticated": bool(
            checkpoint_authentication
            and checkpoint_authentication.get("authenticity_established")
        ),
        "checkpoint_signature_required": args.require_checkpoint_signature,
        "final_metrics": metrics,
        "checkpoint": str(checkpoint),
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }
    _write_json(Path(args.report).resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
