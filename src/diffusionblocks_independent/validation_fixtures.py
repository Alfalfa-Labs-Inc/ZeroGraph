"""Data-only fixtures for reproducible package validation; not user benchmarks."""


def billion_transformer_batch_factory():
    """CPU batch matching the core 1.0067B CUDA Transformer factory."""

    import torch

    generator = torch.Generator(device="cpu").manual_seed(20260829)
    inputs = torch.randn(1, 8, 4096, dtype=torch.bfloat16, generator=generator)
    targets = torch.randn(1, 8, 4096, dtype=torch.bfloat16, generator=generator)
    return ((inputs, targets),)


def mnist_diagnostic_batch_factory():
    """Small real-data batch for the frozen MNIST program diagnostic."""

    import os
    from pathlib import Path

    import torch
    from torchvision.datasets import MNIST

    data_root = Path(
        os.environ.get("DIFFUSIONBLOCKS_MNIST_DATA", "artifacts/mnist_data")
    )
    test = (
        MNIST(data_root, train=False, download=False)
        .data[:32]
        .float()
        .div_(255)
        .reshape(32, -1)
    )
    generator = torch.Generator().manual_seed(20260829)
    context = (
        test + 0.30 * torch.randn(test.shape, generator=generator)
    ).clamp(0, 1)
    return ((torch.zeros_like(test), context, test),)
