import json
import tempfile
import unittest
from pathlib import Path

import torch

from diffusionblocks_independent import (
    MaterializedBatch,
    MaterializedBoundaryCache,
    materialize_boundaries,
    train_materialized_block,
)


class Teacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(4, 4, bias=False)
        self.second = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.first.weight.copy_(torch.eye(4) * 2)
            self.second.weight.copy_(torch.eye(4) * 3)


def extract(teacher, batch):
    first = teacher.first(batch)
    second = teacher.second(first)
    return MaterializedBatch((batch, first, second), {"scale": torch.ones(())})


class MaterializedTests(unittest.TestCase):
    def test_materialize_verify_and_train_independent_block(self):
        batches = [
            torch.arange(8, dtype=torch.float32).view(2, 4) + index
            for index in range(4)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            manifest = materialize_boundaries(
                Teacher(),
                batches,
                extract,
                root,
                teacher_id="tests/teacher",
                teacher_revision="0123456789abcdef",
                data_id="fixed-test-stream",
            )
            self.assertEqual(manifest["objective"], "clean_teacher_boundary_regression")
            self.assertEqual(manifest["sigma"], 0.0)
            cache = MaterializedBoundaryCache(root)
            self.assertEqual(cache.block_count, 2)
            source, target, context = cache.load(0, 0)
            self.assertTrue(torch.equal(source, batches[0]))
            self.assertTrue(torch.equal(target, batches[0] * 2))
            self.assertEqual(float(context["scale"]), 1.0)

            block = torch.nn.Linear(4, 4, bias=False)
            torch.nn.init.zeros_(block.weight)
            optimizer = torch.optim.SGD(block.parameters(), lr=0.001)
            report = train_materialized_block(
                block, cache, 0, optimizer, steps=64, device="cpu"
            )
            self.assertEqual(report["inter_block_gradient_bytes"], 0)
            self.assertEqual(report["live_teacher_parameters"], 0)
            self.assertLess(
                report["last_window_mean_loss"], report["first_window_mean_loss"]
            )

    def test_tampered_shard_rejects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            materialize_boundaries(
                Teacher(),
                [torch.ones(2, 4)],
                extract,
                root,
                teacher_id="tests/teacher",
                teacher_revision="rev",
                data_id="data",
            )
            shard = root / "batch_00000000.pt"
            shard.write_bytes(shard.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "shard digest mismatch"):
                MaterializedBoundaryCache(root)

    def test_manifest_declares_teacher_and_data_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            materialize_boundaries(
                Teacher(),
                [torch.ones(1, 4)],
                extract,
                root,
                teacher_id="org/model",
                teacher_revision="commit",
                data_id="sha256:data",
            )
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(manifest["teacher"]["revision"], "commit")
            self.assertEqual(manifest["data_id"], "sha256:data")

    def test_refuses_implicit_cache_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            kwargs = {
                "teacher": Teacher(),
                "batches": [torch.ones(1, 4)],
                "boundary_extractor": extract,
                "output_directory": root,
                "teacher_id": "teacher",
                "teacher_revision": "rev",
                "data_id": "data",
            }
            materialize_boundaries(**kwargs)
            with self.assertRaises(FileExistsError):
                materialize_boundaries(**kwargs)

    def test_overwrite_rejects_unrelated_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "not-a-cache"
            root.mkdir()
            (root / "valuable.txt").write_text("preserve me")
            with self.assertRaisesRegex(ValueError, "existing ZeroGraph cache"):
                materialize_boundaries(
                    Teacher(),
                    [torch.ones(1, 4)],
                    extract,
                    root,
                    teacher_id="teacher",
                    teacher_revision="rev",
                    data_id="data",
                    overwrite=True,
                )
            self.assertEqual((root / "valuable.txt").read_text(), "preserve me")


if __name__ == "__main__":
    unittest.main()
