"""Console interface for inspecting and concurrently launching independent blocks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .api import (
    inspect_program,
    launch_plan,
    sign_distributed_checkpoint,
    verify_distributed_checkpoint,
)
from .benchmark import benchmark_modes
from .diagnostics import diagnose_program


def _write_json(path: str | Path, payload) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def _devices(value: str) -> tuple[str, ...]:
    rows = tuple(item.strip() for item in value.split(",") if item.strip())
    if not rows:
        raise argparse.ArgumentTypeError("devices cannot be empty")
    return rows


def _run_group(group_id: int, jobs: list[dict], output: Path) -> list[dict]:
    results = []
    for job in jobs:
        log = output / f"block_{job['block_id']:02d}.log"
        environment = os.environ.copy()
        if job["cuda_visible_devices"] is not None:
            environment["CUDA_VISIBLE_DEVICES"] = job["cuda_visible_devices"]
        authentication = None
        authentication_error = None
        distributed = len(job["devices"]) > 1
        try:
            if (
                distributed
                and "--resume" in job["command"]
                and job["trusted_checkpoint_public_key"] is not None
            ):
                authentication = verify_distributed_checkpoint(
                    job["checkpoint_directory"],
                    trusted_public_key_path=job["trusted_checkpoint_public_key"],
                )
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            authentication_error = f"{type(exc).__name__}: {exc}"
        started = time.perf_counter()
        if authentication_error is not None:
            return [
                {
                    "block_id": job["block_id"],
                    "group_id": group_id,
                    "returncode": 3,
                    "seconds": 0.0,
                    "log": str(log),
                    "report": job["report"],
                    "checkpoint_authentication_error": authentication_error,
                }
            ]
        with log.open("wb") as handle:
            completed = subprocess.run(
                job["command"],
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode == 0 and distributed:
            try:
                if job["checkpoint_signing_private_key"] is not None:
                    authentication = sign_distributed_checkpoint(
                        job["checkpoint_directory"],
                        job["checkpoint_signing_private_key"],
                        trusted_public_key_path=job["trusted_checkpoint_public_key"],
                    )
                elif job["require_checkpoint_signature"]:
                    raise RuntimeError(
                        "distributed checkpoint authentication was required but no "
                        "signing key was configured"
                    )
            except (OSError, TypeError, ValueError, RuntimeError) as exc:
                authentication_error = f"{type(exc).__name__}: {exc}"
        returncode = completed.returncode if authentication_error is None else 3
        results.append(
            {
                "block_id": job["block_id"],
                "group_id": group_id,
                "returncode": returncode,
                "seconds": time.perf_counter() - started,
                "log": str(log),
                "report": job["report"],
                "checkpoint_authentication": authentication,
                "checkpoint_authentication_error": authentication_error,
            }
        )
        if returncode:
            break
    return results


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="diffusionblocks-independent")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--program", required=True)
    inspect.add_argument("--output", required=True)
    inspect.add_argument("--skip-block-byte-verification", action="store_true")

    diagnose = subparsers.add_parser("diagnose")
    diagnose.add_argument("--program", required=True)
    diagnose.add_argument("--batch-factory", required=True)
    diagnose.add_argument("--output", required=True)
    diagnose.add_argument("--probe-batches", type=int, default=2)
    diagnose.add_argument("--sigma-points", type=int, default=3)
    diagnose.add_argument("--device", default="cpu")
    diagnose.add_argument("--seed", type=int, default=2041)
    diagnose.add_argument("--trusted-public-key")
    diagnose.add_argument("--require-signature", action="store_true")

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--model-factory", required=True)
    benchmark.add_argument("--batch-factory", required=True)
    benchmark.add_argument("--blocks", type=int, required=True)
    benchmark.add_argument("--steps", type=int, required=True)
    benchmark.add_argument("--independent-steps-per-block", type=int)
    benchmark.add_argument("--output-dir", required=True)
    benchmark.add_argument("--devices", required=True, type=_devices)
    benchmark.add_argument(
        "--precision", choices=("fp32", "bf16", "fp16"), default="fp32"
    )
    benchmark.add_argument("--learning-rate", type=float, default=1e-4)
    benchmark.add_argument("--optimizer", default="adamw")
    benchmark.add_argument("--exact-warmup-steps", type=int, default=0)

    launch = subparsers.add_parser("launch")
    launch.add_argument("--program", required=True)
    launch.add_argument("--batch-factory", required=True)
    launch.add_argument("--steps", required=True, type=int)
    launch.add_argument("--output-dir", required=True)
    launch.add_argument("--devices", required=True, type=_devices)
    launch.add_argument("--processes-per-block", type=int, default=1)
    launch.add_argument("--fsdp-degree", type=int)
    launch.add_argument("--tensor-parallel-degree", type=int, default=1)
    launch.add_argument("--context-parallel-degree", type=int, default=1)
    launch.add_argument("--tensor-parallel-adapter")
    launch.add_argument("--tensor-parallel-adapter-certificate")
    launch.add_argument("--tensor-parallel-adapter-certificate-signature")
    launch.add_argument("--tensor-parallel-adapter-certificate-trusted-public-key")
    launch.add_argument(
        "--require-tensor-parallel-adapter-certificate-signature",
        action="store_true",
    )
    launch.add_argument("--sequence-dimension", type=int, default=-2)
    launch.add_argument("--raw-target-sequence-dimension", type=int)
    launch.add_argument("--context-sequence-dimensions", type=json.loads)
    launch.add_argument("--timeout-seconds", type=int, default=1800)
    launch.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    launch.add_argument("--learning-rate", type=float, default=1e-4)
    launch.add_argument("--optimizer", default="adamw")
    launch.add_argument("--seed", type=int, default=2040)
    state = launch.add_mutually_exclusive_group()
    state.add_argument("--resume", action="store_true")
    state.add_argument("--overwrite", action="store_true")
    launch.add_argument("--override-resume-learning-rate", action="store_true")
    launch.add_argument("--trusted-public-key")
    launch.add_argument("--require-signature", action="store_true")
    launch.add_argument("--checkpoint-signing-private-key")
    launch.add_argument("--trusted-checkpoint-public-key")
    launch.add_argument("--require-checkpoint-signature", action="store_true")
    launch.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "inspect":
        report = inspect_program(
            args.program,
            verify_block_bytes=not args.skip_block_byte_verification,
        )
        _write_json(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if args.command == "diagnose":
        report = diagnose_program(
            args.program,
            batch_factory=args.batch_factory,
            probe_batches=args.probe_batches,
            sigma_points=args.sigma_points,
            device=args.device,
            seed=args.seed,
            trusted_public_key_path=args.trusted_public_key,
            require_signature=args.require_signature,
        )
        _write_json(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if args.command == "benchmark":
        report = benchmark_modes(
            model_factory=args.model_factory,
            batch_factory=args.batch_factory,
            blocks=args.blocks,
            steps=args.steps,
            independent_steps_per_block=args.independent_steps_per_block,
            output_directory=args.output_dir,
            devices=args.devices,
            precision=args.precision,
            learning_rate=args.learning_rate,
            optimizer=args.optimizer,
            exact_warmup_steps=args.exact_warmup_steps,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    plan = launch_plan(
        args.program,
        batch_factory=args.batch_factory,
        steps=args.steps,
        output_directory=args.output_dir,
        devices=args.devices,
        processes_per_block=args.processes_per_block,
        fsdp_degree=args.fsdp_degree,
        tensor_parallel_degree=args.tensor_parallel_degree,
        context_parallel_degree=args.context_parallel_degree,
        tensor_parallel_adapter=args.tensor_parallel_adapter,
        tensor_parallel_adapter_certificate_path=(
            args.tensor_parallel_adapter_certificate
        ),
        tensor_parallel_adapter_certificate_signature_path=(
            args.tensor_parallel_adapter_certificate_signature
        ),
        tensor_parallel_adapter_certificate_trusted_public_key_path=(
            args.tensor_parallel_adapter_certificate_trusted_public_key
        ),
        require_tensor_parallel_adapter_certificate_signature=(
            args.require_tensor_parallel_adapter_certificate_signature
        ),
        sequence_dimension=args.sequence_dimension,
        raw_target_sequence_dimension=args.raw_target_sequence_dimension,
        context_sequence_dimensions=args.context_sequence_dimensions,
        timeout_seconds=args.timeout_seconds,
        precision=args.precision,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        seed=args.seed,
        resume=args.resume,
        override_resume_learning_rate=args.override_resume_learning_rate,
        overwrite=args.overwrite,
        trusted_public_key_path=args.trusted_public_key,
        require_signature=args.require_signature,
        checkpoint_signing_private_key_path=args.checkpoint_signing_private_key,
        trusted_checkpoint_public_key_path=args.trusted_checkpoint_public_key,
        require_checkpoint_signature=args.require_checkpoint_signature,
    )
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "launch-plan.json", plan)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    grouped = {
        group_id: [job for job in plan["jobs"] if job["group_id"] == group_id]
        for group_id in range(len(plan["execution"]["groups"]))
    }
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(grouped)) as executor:
        futures = [
            executor.submit(_run_group, group_id, jobs, output)
            for group_id, jobs in grouped.items()
        ]
        results = [row for future in futures for row in future.result()]
    elapsed = time.perf_counter() - started
    aggregate = sum(row["seconds"] for row in results)
    report = {
        "format_version": 1,
        "status": (
            "completed"
            if len(results) == len(plan["jobs"])
            and all(row["returncode"] == 0 for row in results)
            else "failed"
        ),
        "plan": str(output / "launch-plan.json"),
        "elapsed_wall_seconds": elapsed,
        "aggregate_block_job_seconds": aggregate,
        "observed_concurrency_ratio": aggregate / elapsed if elapsed else None,
        "results": sorted(results, key=lambda row: row["block_id"]),
        "claim_scope": (
            "measured launcher wall time; speedup requires a matched end-to-end "
            "baseline and assembled-quality gate"
        ),
    }
    _write_json(output / "launch-report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main(sys.argv[1:])
