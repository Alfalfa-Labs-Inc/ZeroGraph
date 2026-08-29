"""Public contract and launch planning for independent DiffusionBlocks."""

from __future__ import annotations

import json
import math
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

INDEPENDENT_CLAIM_SCOPE = (
    "independent_tensor_regression_diffusion_mode_not_bitwise_training_parity"
)
_CORE_VERSION = "0.17.0"
_PACKAGE_VERSION = "0.3.0"
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_DISTRIBUTED_CHECKPOINT_MANIFEST = "independent-checkpoint-manifest.json"


@dataclass(frozen=True)
class IndependentContract:
    """Machine-readable separation from the exact-rematerialization product."""

    package_mode: str = "approximate_independent_diffusion_training"
    exact_release_preserved: str = "diffusionblocks-v5==0.17.0"
    bitwise_ordinary_training_parity: bool = False
    ordinary_gradient_parity: bool = False
    corrector_training_required: bool = False
    inter_block_gradient_bytes: int = 0
    objective: str = "block_local_noise_interval_denoising"
    execution: str = "one_independent_job_per_block"
    speed_claim: str = "must_be_measured_on_disjoint_hardware"
    quality_claim: str = "must_be_measured_after_assembled_inference"


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, *, maximum_bytes: int | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular non-symlink file: {path}")
    if maximum_bytes is not None and path.stat().st_size > maximum_bytes:
        raise ValueError(f"file exceeds the declared safety bound: {path}")
    return path


def _safe_member(root: Path, relative: str) -> Path:
    member = Path(relative)
    if member.is_absolute() or ".." in member.parts:
        raise ValueError(f"unsafe program member path: {relative!r}")
    resolved = (root / member).resolve()
    if resolved.parent != root:
        raise ValueError(f"program member escapes its directory: {relative!r}")
    return _regular_file(resolved)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _distributed_checkpoint_files(directory: str | Path) -> tuple[Path, dict, list]:
    root = Path(directory).resolve()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("distributed checkpoint must be a real directory")
    latest_path = _regular_file(root / "latest.json", maximum_bytes=1024 * 1024)
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    checkpoint_name = latest.get("checkpoint")
    if not isinstance(checkpoint_name, str):
        raise TypeError("distributed checkpoint marker has no checkpoint name")
    checkpoint_member = Path(checkpoint_name)
    if checkpoint_member.is_absolute() or ".." in checkpoint_member.parts:
        raise ValueError("distributed checkpoint marker has an unsafe path")
    checkpoint = (root / checkpoint_member).resolve()
    if checkpoint.parent != root or checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("distributed checkpoint marker does not name a safe directory")
    files = [latest_path]
    for candidate in sorted(checkpoint.rglob("*")):
        if candidate.is_symlink():
            raise ValueError("distributed checkpoint contains a symbolic link")
        if candidate.is_file():
            files.append(candidate)
        elif not candidate.is_dir():
            raise ValueError("distributed checkpoint contains an unsupported member")
    if len(files) == 1:
        raise ValueError("distributed checkpoint snapshot contains no payload files")
    return root, latest, files


def sign_distributed_checkpoint(
    directory: str | Path,
    private_key_path: str | Path,
    *,
    trusted_public_key_path: str | Path | None = None,
) -> dict[str, Any]:
    """Authenticate the exact files referenced by a DCP latest marker."""

    from diffusionblocks import sign_artifact

    root, latest, files = _distributed_checkpoint_files(directory)
    rows = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]
    manifest = {
        "format_version": 1,
        "kind": "diffusionblocks_independent_distributed_checkpoint_snapshot",
        "scope": "latest marker and every regular file in its referenced DCP tree",
        "block_id": latest.get("block_id"),
        "step": latest.get("step"),
        "program_sha256": latest.get("program_sha256"),
        "checkpoint": latest.get("checkpoint"),
        "files": rows,
    }
    manifest_path = root / _DISTRIBUTED_CHECKPOINT_MANIFEST
    _atomic_json(manifest_path, manifest)
    sign_artifact(manifest_path, private_key_path)
    verification = None
    if trusted_public_key_path is not None:
        verification = verify_distributed_checkpoint(
            root, trusted_public_key_path=trusted_public_key_path
        )
    return {
        "status": "signed",
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "file_count": len(rows),
        "authenticated": bool(verification and verification["authenticated"]),
    }


def verify_distributed_checkpoint(
    directory: str | Path,
    *,
    trusted_public_key_path: str | Path,
) -> dict[str, Any]:
    """Verify a signed DCP snapshot and every file it covers."""

    from diffusionblocks import verify_artifact_signature

    root, latest, current_files = _distributed_checkpoint_files(directory)
    manifest_path = _regular_file(
        root / _DISTRIBUTED_CHECKPOINT_MANIFEST, maximum_bytes=16 * 1024 * 1024
    )
    authentication = verify_artifact_signature(manifest_path, trusted_public_key_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("format_version") != 1
        or manifest.get("kind")
        != "diffusionblocks_independent_distributed_checkpoint_snapshot"
        or manifest.get("checkpoint") != latest.get("checkpoint")
        or manifest.get("block_id") != latest.get("block_id")
        or manifest.get("step") != latest.get("step")
        or manifest.get("program_sha256") != latest.get("program_sha256")
    ):
        raise ValueError("distributed checkpoint snapshot provenance mismatch")
    rows = manifest.get("files")
    current_relative = {str(path.relative_to(root)) for path in current_files}
    if (
        not isinstance(rows, list)
        or {row.get("path") for row in rows} != current_relative
    ):
        raise ValueError("distributed checkpoint snapshot file set mismatch")
    for row in rows:
        path = (root / str(row["path"])).resolve()
        if root not in path.parents and path != root:
            raise ValueError("distributed checkpoint snapshot contains an unsafe path")
        _regular_file(path)
        if path.stat().st_size != row.get("bytes") or _sha256(path) != row.get(
            "sha256"
        ):
            raise ValueError("distributed checkpoint snapshot file digest mismatch")
    return {
        "status": "verified",
        "authenticated": True,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "file_count": len(rows),
        "signature": authentication,
    }


def _verified_manifest(directory: str | Path) -> tuple[Path, dict[str, Any], str]:
    root = Path(directory).resolve()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("program must be a real directory")
    manifest_path = _regular_file(
        root / "diffusionblocks.json", maximum_bytes=16 * 1024 * 1024
    )
    digest_path = _regular_file(root / "diffusionblocks.sha256", maximum_bytes=256)
    expected = digest_path.read_text(encoding="ascii").strip()
    if not _HEX_64.fullmatch(expected):
        raise ValueError("program digest sidecar is malformed")
    actual = _sha256(manifest_path)
    if actual != expected:
        raise ValueError("program manifest digest does not match its sidecar")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("claim_scope") != INDEPENDENT_CLAIM_SCOPE:
        raise ValueError(
            "program is not an independent approximate DiffusionBlocks artifact"
        )
    return root, manifest, actual


def compile_checkpoint(*args, **kwargs):
    """Compile through the core converter while making the broken-parity mode explicit."""

    import diffusionblocks
    from diffusionblocks import convert_checkpoint

    if diffusionblocks.__version__ != _CORE_VERSION:
        raise RuntimeError(
            f"diffusionblocks-independent {_PACKAGE_VERSION} requires core "
            f"{_CORE_VERSION}, "
            f"found {diffusionblocks.__version__}"
        )
    program = convert_checkpoint(*args, **kwargs)
    program.compatibility.inferred["independent_package_contract"] = asdict(
        IndependentContract()
    )
    return program


def inspect_program(
    directory: str | Path,
    *,
    verify_block_bytes: bool = True,
    trusted_public_key_path: str | Path | None = None,
    require_signature: bool = False,
) -> dict[str, Any]:
    """Verify a serialized program without loading the full model into memory."""

    if require_signature and trusted_public_key_path is None:
        raise ValueError("requiring a program signature requires a trusted public key")
    root, manifest, manifest_sha256 = _verified_manifest(directory)
    signature = None
    if trusted_public_key_path is not None:
        from diffusionblocks import verify_pretrained_signature

        signature = verify_pretrained_signature(root, trusted_public_key_path)
        if signature.get("artifact_sha256") != manifest_sha256:
            raise ValueError("program signature authenticates a different manifest")
    rows = manifest.get("blocks")
    if not isinstance(rows, list) or not rows:
        raise ValueError("program contains no block artifacts")
    verified = []
    expected_ids = list(range(len(rows)))
    actual_ids = [row.get("block_id") for row in rows]
    if actual_ids != expected_ids:
        raise ValueError("block identifiers must be contiguous and ordered")
    for row in rows:
        path = _safe_member(root, str(row["path"]))
        expected = str(row["sha256"])
        if not _HEX_64.fullmatch(expected):
            raise ValueError(f"block {row['block_id']} has a malformed digest")
        actual = _sha256(path) if verify_block_bytes else None
        if actual is not None and actual != expected:
            raise ValueError(f"block {row['block_id']} digest mismatch")
        verified.append(
            {
                "block_id": int(row["block_id"]),
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": expected,
                "digest_verified": verify_block_bytes,
                "parameter_count": int(row["parameter_count"]),
                "trainable_parameter_count": int(row["trainable_parameter_count"]),
                "unique_state_bytes": int(row["unique_state_bytes"]),
            }
        )
    state_bytes = [row["unique_state_bytes"] for row in verified]
    parameter_counts = [row["parameter_count"] for row in verified]
    return {
        "format_version": 1,
        "status": "verified",
        "program": str(root),
        "manifest_sha256": manifest_sha256,
        "blocks": verified,
        "block_count": len(verified),
        "authenticity": {
            "signature_required": require_signature,
            "trusted_public_key": (
                str(Path(trusted_public_key_path).resolve())
                if trusted_public_key_path is not None
                else None
            ),
            "signature": signature,
            "authenticated": bool(
                signature and signature.get("authenticity_established")
            ),
        },
        "contract": asdict(IndependentContract()),
        "state_working_set": {
            "sum_serialized_block_state_bytes": sum(state_bytes),
            "largest_active_block_state_bytes": max(state_bytes),
            "sum_over_largest_active_block": sum(state_bytes) / max(state_bytes),
            "scope": (
                "serialized block state only; excludes optimizer, activations, codec, "
                "allocator, and framework overhead"
            ),
        },
        "logical_parameters": {
            "sum_across_blocks": sum(parameter_counts),
            "largest_active_block": max(parameter_counts),
        },
    }


def launch_plan(
    directory: str | Path,
    *,
    batch_factory: str,
    steps: int,
    output_directory: str | Path,
    devices: Sequence[str],
    processes_per_block: int = 1,
    fsdp_degree: int | None = None,
    tensor_parallel_degree: int = 1,
    context_parallel_degree: int = 1,
    tensor_parallel_adapter: str | None = None,
    tensor_parallel_adapter_certificate_path: str | Path | None = None,
    tensor_parallel_adapter_certificate_signature_path: str | Path | None = None,
    tensor_parallel_adapter_certificate_trusted_public_key_path: str
    | Path
    | None = None,
    require_tensor_parallel_adapter_certificate_signature: bool = False,
    sequence_dimension: int = -2,
    raw_target_sequence_dimension: int | None = None,
    context_sequence_dimensions: dict[str, int] | None = None,
    timeout_seconds: int = 1800,
    precision: str = "bf16",
    learning_rate: float = 1e-4,
    optimizer: str = "adamw",
    python_executable: str = sys.executable,
    seed: int = 2040,
    resume: bool = False,
    override_resume_learning_rate: bool = False,
    overwrite: bool = False,
    trusted_public_key_path: str | Path | None = None,
    require_signature: bool = False,
    checkpoint_signing_private_key_path: str | Path | None = None,
    trusted_checkpoint_public_key_path: str | Path | None = None,
    require_checkpoint_signature: bool = False,
) -> dict[str, Any]:
    """Create independent block commands and a wave schedule."""

    if (
        steps <= 0
        or processes_per_block <= 0
        or tensor_parallel_degree <= 0
        or context_parallel_degree <= 0
        or learning_rate <= 0
        or timeout_seconds <= 0
        or seed < 0
    ):
        raise ValueError("steps, parallel degrees, and learning rate must be positive")
    model_parallel_degree = tensor_parallel_degree * context_parallel_degree
    if processes_per_block % model_parallel_degree:
        raise ValueError(
            "processes_per_block must be divisible by tensor_parallel_degree times "
            "context_parallel_degree"
        )
    inferred_fsdp_degree = processes_per_block // model_parallel_degree
    if fsdp_degree is None:
        fsdp_degree = inferred_fsdp_degree
    if fsdp_degree <= 0 or fsdp_degree != inferred_fsdp_degree:
        raise ValueError(
            "processes_per_block must equal fsdp_degree times "
            "tensor_parallel_degree times context_parallel_degree"
        )
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("precision must be fp32, bf16, or fp16")
    if (
        require_tensor_parallel_adapter_certificate_signature
        and tensor_parallel_adapter_certificate_trusted_public_key_path is None
    ):
        raise ValueError(
            "requiring a TP adapter certificate signature requires its trusted key"
        )
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    if override_resume_learning_rate and not resume:
        raise ValueError("overriding the resume learning rate requires resume=True")
    if override_resume_learning_rate and processes_per_block != 1:
        raise ValueError(
            "resume learning-rate override is currently supported only by the "
            "single-process independent worker"
        )
    if require_checkpoint_signature and trusted_checkpoint_public_key_path is None:
        raise ValueError(
            "requiring checkpoint signatures requires a trusted checkpoint public key"
        )
    if require_checkpoint_signature and checkpoint_signing_private_key_path is None:
        raise ValueError(
            "requiring authenticated output checkpoints requires a signing private key"
        )
    if not devices:
        raise ValueError("at least one CPU slot or CUDA device is required")
    cpu = all(device == "cpu" for device in devices)
    if not cpu and any(device == "cpu" for device in devices):
        raise ValueError("CPU and CUDA slots cannot be mixed in one launch")
    if cpu:
        cpu_slots = tuple(f"cpu:{index}" for index in range(len(devices)))
        if len(cpu_slots) % processes_per_block:
            raise ValueError("CPU slot count must divide into complete block groups")
        groups = [
            tuple(cpu_slots[index : index + processes_per_block])
            for index in range(0, len(cpu_slots), processes_per_block)
        ]
        runtime_device = "cpu"
    else:
        if len(devices) % processes_per_block:
            raise ValueError("CUDA device count must divide into complete block groups")
        groups = [
            tuple(devices[index : index + processes_per_block])
            for index in range(0, len(devices), processes_per_block)
        ]
        runtime_device = "cuda"
    inspection = inspect_program(
        directory,
        trusted_public_key_path=trusted_public_key_path,
        require_signature=require_signature,
    )
    output = Path(output_directory).resolve()
    jobs = []
    for block_id in range(inspection["block_count"]):
        group_id = block_id % len(groups)
        checkpoint = output / f"block_{block_id:02d}"
        report = output / f"block_{block_id:02d}.json"
        if processes_per_block == 1:
            command = [
                python_executable,
                "-m",
                "diffusionblocks_independent.worker",
                "--program",
                str(Path(directory).resolve()),
                "--batch-factory",
                batch_factory,
                "--block-id",
                str(block_id),
                "--steps",
                str(steps),
                "--learning-rate",
                repr(learning_rate),
                "--optimizer",
                optimizer,
                "--checkpoint",
                str(checkpoint / "checkpoint.pt"),
                "--device",
                runtime_device,
                "--precision",
                precision,
                "--report",
                str(report),
                "--seed",
                str(seed + block_id),
            ]
            if resume:
                command.append("--resume")
            if override_resume_learning_rate:
                command.append("--override-resume-learning-rate")
            if overwrite:
                command.append("--overwrite")
            if trusted_public_key_path is not None:
                command.extend(
                    [
                        "--trusted-public-key",
                        str(Path(trusted_public_key_path).resolve()),
                    ]
                )
            if require_signature:
                command.append("--require-signature")
            if checkpoint_signing_private_key_path is not None:
                command.extend(
                    [
                        "--checkpoint-signing-private-key",
                        str(Path(checkpoint_signing_private_key_path).resolve()),
                    ]
                )
            if trusted_checkpoint_public_key_path is not None:
                command.extend(
                    [
                        "--trusted-checkpoint-public-key",
                        str(Path(trusted_checkpoint_public_key_path).resolve()),
                    ]
                )
            if require_checkpoint_signature:
                command.append("--require-checkpoint-signature")
        else:
            command = [
                python_executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={processes_per_block}",
                "-m",
                "diffusionblocks.cli",
                "automatic-train-block-distributed",
                "--program",
                str(Path(directory).resolve()),
                "--batch-factory",
                batch_factory,
                "--block-id",
                str(block_id),
                "--steps",
                str(steps),
                "--learning-rate",
                repr(learning_rate),
                "--optimizer",
                optimizer,
                "--checkpoint-dir",
                str(checkpoint),
                "--device",
                runtime_device,
                "--precision",
                precision,
                "--fsdp-degree",
                str(fsdp_degree),
                "--tensor-parallel-degree",
                str(tensor_parallel_degree),
                "--context-parallel-degree",
                str(context_parallel_degree),
                "--sequence-dimension",
                str(sequence_dimension),
                "--timeout-seconds",
                str(timeout_seconds),
                "--report",
                str(report),
            ]
            if raw_target_sequence_dimension is not None:
                command.extend(
                    [
                        "--raw-target-sequence-dimension",
                        str(raw_target_sequence_dimension),
                    ]
                )
            if context_sequence_dimensions is not None:
                command.extend(
                    [
                        "--context-sequence-dimensions",
                        json.dumps(context_sequence_dimensions, sort_keys=True),
                    ]
                )
            if tensor_parallel_adapter is not None:
                command.extend(["--tensor-parallel-adapter", tensor_parallel_adapter])
            if tensor_parallel_adapter_certificate_path is not None:
                command.extend(
                    [
                        "--tensor-parallel-adapter-certificate",
                        str(Path(tensor_parallel_adapter_certificate_path).resolve()),
                    ]
                )
            if tensor_parallel_adapter_certificate_signature_path is not None:
                command.extend(
                    [
                        "--tensor-parallel-adapter-certificate-signature",
                        str(
                            Path(
                                tensor_parallel_adapter_certificate_signature_path
                            ).resolve()
                        ),
                    ]
                )
            if tensor_parallel_adapter_certificate_trusted_public_key_path is not None:
                command.extend(
                    [
                        "--tensor-parallel-adapter-certificate-trusted-public-key",
                        str(
                            Path(
                                tensor_parallel_adapter_certificate_trusted_public_key_path
                            ).resolve()
                        ),
                    ]
                )
            if require_tensor_parallel_adapter_certificate_signature:
                command.append(
                    "--require-tensor-parallel-adapter-certificate-signature"
                )
            if resume:
                command.append("--resume")
            if trusted_public_key_path is not None:
                command.extend(
                    [
                        "--trusted-public-key",
                        str(Path(trusted_public_key_path).resolve()),
                    ]
                )
            if require_signature:
                command.append("--require-signature")
        jobs.append(
            {
                "block_id": block_id,
                "group_id": group_id,
                "devices": list(groups[group_id]),
                "cuda_visible_devices": (
                    ",".join(groups[group_id]) if runtime_device == "cuda" else None
                ),
                "command": command,
                "checkpoint_directory": str(checkpoint),
                "report": str(report),
                "checkpoint_signing_private_key": (
                    str(Path(checkpoint_signing_private_key_path).resolve())
                    if checkpoint_signing_private_key_path is not None
                    else None
                ),
                "trusted_checkpoint_public_key": (
                    str(Path(trusted_checkpoint_public_key_path).resolve())
                    if trusted_checkpoint_public_key_path is not None
                    else None
                ),
                "require_checkpoint_signature": require_checkpoint_signature,
            }
        )
    waves = math.ceil(len(jobs) / len(groups))
    return {
        "format_version": 1,
        "status": "planned",
        "program": inspection,
        "contract": asdict(IndependentContract()),
        "execution": {
            "device_kind": runtime_device,
            "groups": [list(group) for group in groups],
            "concurrent_block_jobs": len(groups),
            "waves": waves,
            "processes_per_block": processes_per_block,
            "fsdp_degree": fsdp_degree,
            "tensor_parallel_degree": tensor_parallel_degree,
            "context_parallel_degree": context_parallel_degree,
            "sequence_dimension": sequence_dimension,
            "raw_target_sequence_dimension": raw_target_sequence_dimension,
            "context_sequence_dimensions": context_sequence_dimensions,
            "inter_block_gradient_bytes": 0,
            "resume": resume,
            "overwrite": overwrite,
            "seed_base": seed,
        },
        "jobs": jobs,
        "claim_scope": (
            "executable independent-job schedule; concurrency and state ratio are not "
            "measured wall-time, HBM, or task-quality results"
        ),
    }


def load_assembled(
    directory: str | Path,
    checkpoints: Sequence[str | Path],
    *,
    trusted_program_public_key_path: str | Path | None = None,
    require_program_signature: bool = False,
    trusted_checkpoint_public_key_path: str | Path | None = None,
    require_checkpoint_signatures: bool = False,
):
    """Load every independently trained block for assembled evaluation/inference."""

    import torch

    from diffusionblocks import load_pretrained

    if require_checkpoint_signatures and trusted_checkpoint_public_key_path is None:
        raise ValueError(
            "requiring checkpoint signatures requires a trusted checkpoint public key"
        )
    inspection = inspect_program(
        directory,
        trusted_public_key_path=trusted_program_public_key_path,
        require_signature=require_program_signature,
    )
    if len(checkpoints) != inspection["block_count"]:
        raise ValueError("assembled loading requires exactly one checkpoint per block")
    program = load_pretrained(
        directory,
        trusted_public_key_path=trusted_program_public_key_path,
        require_signature=require_program_signature,
    )
    seen = set()
    for expected_block_id, value in enumerate(checkpoints):
        source = _regular_file(Path(value).resolve())
        metadata_path = _regular_file(
            source.with_suffix(source.suffix + ".json"), maximum_bytes=1024 * 1024
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("sha256") != _sha256(source):
            raise ValueError(f"block {expected_block_id} checkpoint digest mismatch")
        if trusted_checkpoint_public_key_path is not None:
            from diffusionblocks import verify_artifact_signature

            verify_artifact_signature(source, trusted_checkpoint_public_key_path)
        payload = torch.load(source, map_location="cpu", weights_only=True)
        if (
            not isinstance(payload, dict)
            or payload.get("format_version") != 1
            or payload.get("kind")
            != "diffusionblocks_serialized_local_training_checkpoint"
        ):
            raise ValueError("unsupported independent checkpoint schema")
        block_id = int(payload.get("block_id", -1))
        if block_id != expected_block_id or block_id in seen:
            raise ValueError(
                "independent checkpoints are missing, duplicated, or reordered"
            )
        if payload.get("program_manifest_sha256") != inspection["manifest_sha256"]:
            raise ValueError("checkpoint belongs to a different compiled program")
        program.blocks[block_id].load_state_dict(payload["model"], strict=True)
        seen.add(block_id)
    return program


def load_program(
    directory: str | Path,
    *,
    trusted_public_key_path: str | Path | None = None,
    require_signature: bool = False,
):
    """Load already-trained independent PT2 blocks after contract verification."""

    from diffusionblocks import load_pretrained

    inspect_program(
        directory,
        trusted_public_key_path=trusted_public_key_path,
        require_signature=require_signature,
    )
    return load_pretrained(
        directory,
        trusted_public_key_path=trusted_public_key_path,
        require_signature=require_signature,
    )
