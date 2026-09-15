# Changelog

All notable changes are documented here. This project follows semantic
versioning once the public API is declared stable.

## 0.5.0 - 2026-09-15

- Added the verified nine-phase Qwen2.5 campaign and IEEE paper.
- Validated real 3.086B conversion with four independent block jobs.
- Added objective-matched cache reuse through K=16; no break-even was observed.
- Added ZeRO-3/FSDP2 references, block/boundary sweeps, downstream tasks, and
  bitwise independent restart.

## 0.4.0 - 2026-09-14

- Promoted clean cached teacher-boundary regression (`sigma=0`) to the stable
  ZeroGraph training contract.
- Added `materialize_boundaries`, `MaterializedBoundaryCache`, and
  `train_materialized_block` public APIs.
- Added `zerograph materialize` and `zerograph train-materialized` commands.
- Added SHA-256 cache-manifest and shard verification, safe path handling, and
  explicit teacher/data provenance.
- Added callback interfaces for architecture-specific attention inputs and
  final task losses without loading a live teacher in a local worker.
- Added the three-seed Qwen2.5-1.5B replication and the complete TinyLlama
  positive/negative comparison to the documented evidence.
- Reclassified full-noise local denoising as an experimental legacy path after
  its language coherence gates failed.

## 0.3.0 - 2026-08-29

- Added persistent CUDA-stream scheduling and exact interval telemetry.
- Added schedule-invariant final-state digest checks.
- Added architecture-matrix and Hugging Face compatibility tests.
- Added measured 1.1B cached teacher-anchored conversion evidence.

## 0.2.1

- Hardened CUDA RNG restore and resume-memory measurement.
- Added explicit resumed-learning-rate override support.

## 0.2.0

- Added independent program loading, launch planning, diagnostics, checkpoint
  authentication, and assembled inference APIs.
