import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path


def _equal(left, right):
    import torch

    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)):
        return (
            isinstance(right, type(left))
            and len(left) == len(right)
            and all(_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


class WorkerResumeTests(unittest.TestCase):
    def test_resume_learning_rate_override_is_explicit_and_persisted(self):
        import torch
        from diffusionblocks_independent import compile_checkpoint
        from diffusionblocks_independent.worker import main as worker_main

        from diffusionblocks.parity_examples import residual_factory

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            program = compile_checkpoint(
                **residual_factory(), checkpoint=None, blocks=2
            )
            program.save_pretrained(root / "program")
            checkpoint = root / "checkpoint.pt"
            common = [
                "--program",
                str(root / "program"),
                "--batch-factory",
                "diffusionblocks.parity_examples:residual_batch_factory",
                "--block-id",
                "0",
                "--optimizer",
                "sgd",
                "--checkpoint",
                str(checkpoint),
                "--device",
                "cpu",
                "--precision",
                "fp32",
                "--seed",
                "992",
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                worker_main(
                    [
                        *common,
                        "--steps",
                        "1",
                        "--learning-rate",
                        "0.001",
                        "--report",
                        str(root / "partial.json"),
                    ]
                )
                worker_main(
                    [
                        *common,
                        "--steps",
                        "2",
                        "--learning-rate",
                        "0.02",
                        "--report",
                        str(root / "resumed.json"),
                        "--resume",
                        "--override-resume-learning-rate",
                    ]
                )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            report = json.loads((root / "resumed.json").read_text())
            self.assertEqual(payload["optimizer"]["param_groups"][0]["lr"], 0.02)
            self.assertTrue(report["resume_learning_rate_overridden"])
            self.assertEqual(report["learning_rate"], 0.02)

    def test_resume_matches_uninterrupted_local_training_exactly(self):
        import torch
        from diffusionblocks_independent import compile_checkpoint
        from diffusionblocks_independent.worker import main as worker_main

        from diffusionblocks.parity_examples import residual_factory

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = residual_factory()
            program = compile_checkpoint(**payload, checkpoint=None, blocks=2)
            program.save_pretrained(root / "program")

            common = [
                "--program",
                str(root / "program"),
                "--batch-factory",
                "diffusionblocks.parity_examples:residual_batch_factory",
                "--block-id",
                "0",
                "--learning-rate",
                "0.001",
                "--optimizer",
                "adamw",
                "--device",
                "cpu",
                "--precision",
                "fp32",
                "--seed",
                "991",
            ]
            continuous = root / "continuous.pt"
            with contextlib.redirect_stdout(io.StringIO()):
                worker_main(
                    [
                        *common,
                        "--steps",
                        "4",
                        "--checkpoint",
                        str(continuous),
                        "--report",
                        str(root / "continuous.json"),
                    ]
                )
            resumed = root / "resumed.pt"
            with contextlib.redirect_stdout(io.StringIO()):
                worker_main(
                    [
                        *common,
                        "--steps",
                        "2",
                        "--checkpoint",
                        str(resumed),
                        "--report",
                        str(root / "partial.json"),
                    ]
                )
                worker_main(
                    [
                        *common,
                        "--steps",
                        "4",
                        "--checkpoint",
                        str(resumed),
                        "--report",
                        str(root / "resumed.json"),
                        "--resume",
                    ]
                )
            uninterrupted = torch.load(
                continuous, map_location="cpu", weights_only=True
            )
            restarted = torch.load(resumed, map_location="cpu", weights_only=True)
            for key in (
                "model",
                "optimizer",
                "torch_rng_state",
                "cuda_rng_state_all",
            ):
                self.assertTrue(_equal(uninterrupted[key], restarted[key]), key)
            self.assertEqual(uninterrupted["step"], restarted["step"])

    def test_signed_checkpoints_are_authenticated_during_assembly(self):
        from diffusionblocks_independent import compile_checkpoint, load_assembled
        from diffusionblocks_independent.worker import main as worker_main

        import diffusionblocks as db
        from diffusionblocks.parity_examples import residual_factory

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = residual_factory()
            inputs = payload["example_args"][0]
            program = compile_checkpoint(**payload, checkpoint=None, blocks=2)
            program.save_pretrained(root / "program")
            private_key = root / "private.pem"
            public_key = root / "public.pem"
            db.generate_signing_keypair(private_key, public_key)
            db.sign_pretrained(root / "program", private_key)
            checkpoints = []
            for block_id in range(2):
                checkpoint = root / f"block_{block_id}.pt"
                with contextlib.redirect_stdout(io.StringIO()):
                    worker_main(
                        [
                            "--program",
                            str(root / "program"),
                            "--batch-factory",
                            "diffusionblocks.parity_examples:residual_batch_factory",
                            "--block-id",
                            str(block_id),
                            "--steps",
                            "1",
                            "--learning-rate",
                            "0.001",
                            "--optimizer",
                            "adamw",
                            "--checkpoint",
                            str(checkpoint),
                            "--device",
                            "cpu",
                            "--precision",
                            "fp32",
                            "--report",
                            str(root / f"block_{block_id}.json"),
                            "--trusted-public-key",
                            str(public_key),
                            "--require-signature",
                            "--checkpoint-signing-private-key",
                            str(private_key),
                            "--trusted-checkpoint-public-key",
                            str(public_key),
                            "--require-checkpoint-signature",
                        ]
                    )
                self.assertTrue(Path(str(checkpoint) + ".signature.json").is_file())
                checkpoints.append(checkpoint)
            assembled = load_assembled(
                root / "program",
                checkpoints,
                trusted_program_public_key_path=public_key,
                require_program_signature=True,
                trusted_checkpoint_public_key_path=public_key,
                require_checkpoint_signatures=True,
            )
            result = assembled.zero_sigma_assembled(inputs)
            self.assertEqual(result.shape, inputs.shape)
            with checkpoint.open("ab") as handle:
                handle.write(b"tamper")
            metadata_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(db.AutomaticConversionError):
                load_assembled(
                    root / "program",
                    checkpoints,
                    trusted_program_public_key_path=public_key,
                    require_program_signature=True,
                    trusted_checkpoint_public_key_path=public_key,
                    require_checkpoint_signatures=True,
                )


if __name__ == "__main__":
    unittest.main()
