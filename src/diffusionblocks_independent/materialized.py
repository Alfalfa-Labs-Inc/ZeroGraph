"""Clean materialized-boundary supervision for independent block training.

The cache is deliberately model-agnostic: callers provide a boundary extractor
for their teacher and optional forward/loss callbacks for their student block.
Only tensors and JSON metadata cross the cache boundary.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

MATERIALIZED_CACHE_KIND = "zerograph_clean_teacher_boundaries_v1"


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _safe_root(path: str | Path, *, create: bool = False) -> Path:
    unresolved = Path(path).absolute()
    if unresolved.is_symlink():
        raise ValueError("boundary cache root cannot be a symbolic link")
    root = unresolved.resolve()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise ValueError("boundary cache root must be a directory")
    return root


def _tensor_metadata(tensor) -> dict[str, Any]:
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}


@dataclass(frozen=True)
class MaterializedBatch:
    """One teacher trajectory and tensor-only context for a training batch."""

    boundaries: Sequence[Any]
    context: Mapping[str, Any] | None = None


def _normalize_record(value: Any) -> MaterializedBatch:
    if isinstance(value, MaterializedBatch):
        record = value
    elif isinstance(value, Mapping) and "boundaries" in value:
        record = MaterializedBatch(value["boundaries"], value.get("context"))
    else:
        record = MaterializedBatch(value)
    import torch

    boundaries = tuple(record.boundaries)
    if len(boundaries) < 2 or not all(
        isinstance(item, torch.Tensor) for item in boundaries
    ):
        raise TypeError("boundary extractor must return at least two tensors")
    context = dict(record.context or {})
    if not all(
        isinstance(key, str) and isinstance(item, torch.Tensor)
        for key, item in context.items()
    ):
        raise TypeError("materialized context must be a string-to-tensor mapping")
    return MaterializedBatch(boundaries, context)


def materialize_boundaries(
    teacher,
    batches: Iterable[Any],
    boundary_extractor: Callable[[Any, Any], Any],
    output_directory: str | Path,
    *,
    teacher_id: str,
    teacher_revision: str,
    data_id: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run a frozen teacher once and persist clean (`sigma=0`) trajectories.

    ``boundary_extractor(teacher, batch)`` returns either a sequence
    ``[h_0, ..., h_B]`` or ``MaterializedBatch(boundaries, context)``. Context
    is an optional mapping of tensor inputs such as attention masks.
    """

    import torch

    unresolved_root = Path(output_directory).absolute()
    if unresolved_root.is_symlink():
        raise ValueError("boundary cache root cannot be a symbolic link")
    root = unresolved_root.resolve()
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"boundary cache already exists: {root}")
        if root.is_symlink() or not root.is_dir():
            raise ValueError("refusing to overwrite a non-directory cache path")
        existing_manifest = root / "manifest.json"
        if existing_manifest.is_symlink() or not existing_manifest.is_file():
            raise ValueError("overwrite is restricted to an existing ZeroGraph cache")
        try:
            existing_kind = json.loads(
                existing_manifest.read_text(encoding="utf-8")
            ).get("kind")
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError("existing cache manifest is unreadable") from exc
        if existing_kind != MATERIALIZED_CACHE_KIND:
            raise ValueError("overwrite target is not a ZeroGraph boundary cache")
        shutil.rmtree(root)
    root.mkdir(parents=True)
    teacher.eval()
    teacher.requires_grad_(False)
    rows: list[dict[str, Any]] = []
    block_count = None
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for index, batch in enumerate(batches):
                record = _normalize_record(boundary_extractor(teacher, batch))
                current_blocks = len(record.boundaries) - 1
                if block_count is None:
                    block_count = current_blocks
                elif current_blocks != block_count:
                    raise ValueError("boundary count changed between cache records")
                payload = {
                    "boundaries": [
                        item.detach().cpu().contiguous() for item in record.boundaries
                    ],
                    "context": {
                        key: item.detach().cpu().contiguous()
                        for key, item in (record.context or {}).items()
                    },
                }
                path = root / f"batch_{index:08d}.pt"
                temporary = path.with_name(f".{path.name}.tmp")
                torch.save(payload, temporary)
                os.replace(temporary, path)
                rows.append(
                    {
                        "index": index,
                        "path": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                        "boundaries": [
                            _tensor_metadata(item) for item in payload["boundaries"]
                        ],
                        "context": {
                            key: _tensor_metadata(item)
                            for key, item in payload["context"].items()
                        },
                    }
                )
        if not rows:
            raise ValueError("cannot materialize an empty batch stream")
        manifest = {
            "format_version": 1,
            "kind": MATERIALIZED_CACHE_KIND,
            "status": "completed",
            "objective": "clean_teacher_boundary_regression",
            "sigma": 0.0,
            "teacher": {"id": teacher_id, "revision": teacher_revision},
            "data_id": data_id,
            "block_count": block_count,
            "batch_count": len(rows),
            "seconds": time.perf_counter() - started,
            "files": rows,
        }
        _atomic_json(root / "manifest.json", manifest)
        _atomic_json(
            root / "manifest.sha256.json",
            {"manifest": "manifest.json", "sha256": _sha256(root / "manifest.json")},
        )
        return manifest
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise


class MaterializedBoundaryCache:
    """Verified read-only access to clean teacher boundary pairs."""

    def __init__(self, directory: str | Path, *, verify: bool = True) -> None:
        self.root = _safe_root(directory)
        manifest_path = self.root / "manifest.json"
        digest_path = self.root / "manifest.sha256.json"
        if manifest_path.is_symlink() or digest_path.is_symlink():
            raise ValueError("boundary cache metadata cannot be symbolic links")
        digest = json.loads(digest_path.read_text(encoding="utf-8"))
        if digest.get("sha256") != _sha256(manifest_path):
            raise ValueError("boundary cache manifest digest mismatch")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("kind") != MATERIALIZED_CACHE_KIND:
            raise ValueError("unsupported boundary cache kind")
        if self.manifest.get("sigma") != 0.0:
            raise ValueError("stable materialized mode requires sigma=0")
        self.files = tuple(self.manifest.get("files", ()))
        if not self.files or self.manifest.get("batch_count") != len(self.files):
            raise ValueError("boundary cache file table is incomplete")
        if verify:
            for row in self.files:
                path = self._member(row["path"])
                if (
                    path.stat().st_size != row["bytes"]
                    or _sha256(path) != row["sha256"]
                ):
                    raise ValueError(
                        f"boundary cache shard digest mismatch: {path.name}"
                    )

    @property
    def block_count(self) -> int:
        return int(self.manifest["block_count"])

    def __len__(self) -> int:
        return len(self.files)

    def _member(self, name: str) -> Path:
        member = Path(name)
        if member.is_absolute() or ".." in member.parts:
            raise ValueError("unsafe boundary cache member")
        unresolved = self.root / member
        if unresolved.is_symlink():
            raise ValueError("boundary cache member cannot be a symbolic link")
        path = unresolved.resolve()
        if path.parent != self.root or not path.is_file():
            raise ValueError("invalid boundary cache member")
        return path

    def load(
        self, index: int, block_id: int, *, device=None
    ) -> tuple[Any, Any, dict[str, Any]]:
        import torch

        if not 0 <= index < len(self.files):
            raise IndexError(index)
        if not 0 <= block_id < self.block_count:
            raise IndexError(block_id)
        row = self.files[index]
        path = self._member(row["path"])
        if path.stat().st_size != row["bytes"]:
            raise ValueError(f"boundary cache shard size mismatch: {path.name}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        boundaries = payload["boundaries"]
        if len(boundaries) != self.block_count + 1:
            raise ValueError("boundary cache shard has the wrong trajectory length")
        move = lambda value: value.to(device) if device is not None else value
        return (
            move(boundaries[block_id]),
            move(boundaries[block_id + 1]),
            {key: move(value) for key, value in payload.get("context", {}).items()},
        )


def train_materialized_block(
    block,
    cache: MaterializedBoundaryCache | str | Path,
    block_id: int,
    optimizer,
    *,
    steps: int,
    device=None,
    forward: Callable[[Any, Any, Mapping[str, Any]], Any] | None = None,
    loss: Callable[[Any, Any, Mapping[str, Any], int], Any] | None = None,
    max_grad_norm: float | None = None,
) -> dict[str, Any]:
    """Optimize one block against cached clean boundaries only.

    The default objective is MSE. ``forward`` and ``loss`` allow architectures
    to supply attention masks, positional state, or a final task loss without
    introducing a live teacher or a neighboring trainable block.
    """

    import torch

    if steps <= 0:
        raise ValueError("steps must be positive")
    cache = (
        cache
        if isinstance(cache, MaterializedBoundaryCache)
        else MaterializedBoundaryCache(cache)
    )
    if not 0 <= block_id < cache.block_count:
        raise IndexError(block_id)
    if device is None:
        device = next(block.parameters()).device
    # Context is intentionally opt-in: generic tensor metadata such as masks or
    # labels cannot safely be guessed as keyword arguments for an arbitrary
    # module. Architectures that consume it provide an explicit callback.
    default_forward = lambda module, source, context: module(source)
    default_loss = lambda prediction, target, context, active_block: (
        torch.nn.functional.mse_loss(prediction, target)
    )
    forward = forward or default_forward
    loss = loss or default_loss
    block.train()
    first: list[float] = []
    last: deque[float] = deque(maxlen=32)
    started = time.perf_counter()
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(steps):
        source, target, context = cache.load(step % len(cache), block_id, device=device)
        optimizer.zero_grad(set_to_none=True)
        prediction = forward(block, source, context)
        objective = loss(prediction, target, context, block_id)
        if objective.ndim != 0 or not torch.isfinite(objective):
            raise RuntimeError("materialized objective must be a finite scalar")
        objective.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(block.parameters(), max_grad_norm)
        optimizer.step()
        value = float(objective.detach())
        if len(first) < 32:
            first.append(value)
        last.append(value)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "format_version": 1,
        "status": "completed",
        "objective": "clean_teacher_boundary_regression",
        "sigma": 0.0,
        "block_id": block_id,
        "steps": steps,
        "first_window_mean_loss": sum(first) / len(first),
        "last_window_mean_loss": sum(last) / len(last),
        "compute_seconds": elapsed,
        "inter_block_gradient_bytes": 0,
        "live_teacher_parameters": 0,
        "cache_manifest_sha256": _sha256(cache.root / "manifest.json"),
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device)
            if torch.device(device).type == "cuda"
            else None
        ),
    }
