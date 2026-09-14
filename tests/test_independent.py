import json
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

from diffusionblocks_independent import (
    INDEPENDENT_CLAIM_SCOPE,
    IndependentContract,
    inspect_program,
    launch_plan,
    load_program,
    sign_distributed_checkpoint,
    verify_distributed_checkpoint,
)


class IndependentPackageTests(unittest.TestCase):
    def _program(self, root: Path, *, claim_scope=INDEPENDENT_CLAIM_SCOPE) -> Path:
        blocks = []
        for block_id, size in enumerate((11, 13, 17, 19)):
            path = root / f"block_{block_id:02d}.pt2"
            path.write_bytes(bytes([block_id + 1]) * size)
            blocks.append(
                {
                    "block_id": block_id,
                    "path": path.name,
                    "sha256": sha256(path.read_bytes()).hexdigest(),
                    "parameter_count": (block_id + 1) * 100,
                    "trainable_parameter_count": (block_id + 1) * 100,
                    "trainable_parameter_bytes": (block_id + 1) * 400,
                    "unique_state_bytes": (block_id + 1) * 400,
                }
            )
        manifest = {
            "format_version": 2,
            "kind": "automatic_diffusionblocks_program",
            "claim_scope": claim_scope,
            "blocks": blocks,
        }
        encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        (root / "diffusionblocks.json").write_bytes(encoded)
        (root / "diffusionblocks.sha256").write_text(
            sha256(encoded).hexdigest() + "\n", encoding="ascii"
        )
        return root

    def test_contract_explicitly_breaks_parity(self):
        contract = IndependentContract()
        self.assertFalse(contract.bitwise_ordinary_training_parity)
        self.assertFalse(contract.ordinary_gradient_parity)
        self.assertEqual(contract.inter_block_gradient_bytes, 0)
        self.assertEqual(contract.exact_release_preserved, "diffusionblocks-v5==0.17.0")
        self.assertEqual(
            contract.objective, "clean_teacher_boundary_regression_sigma_zero"
        )
        self.assertTrue(contract.teacher_trajectory_required)
        self.assertFalse(contract.teacher_live_during_local_training)

    def test_inspection_streams_and_verifies_each_block(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = inspect_program(self._program(Path(temporary)))
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["block_count"], 4)
        self.assertTrue(all(row["digest_verified"] for row in report["blocks"]))
        self.assertEqual(
            report["state_working_set"]["sum_over_largest_active_block"], 2.5
        )

    def test_corrupt_block_rejects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._program(Path(temporary))
            (root / "block_02.pt2").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                inspect_program(root)

    def test_exact_or_unknown_artifact_rejects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._program(Path(temporary), claim_scope="bitwise_exact")
            with self.assertRaisesRegex(ValueError, "not an independent approximate"):
                inspect_program(root)

    def test_signature_required_without_trusted_key_rejects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._program(Path(temporary))
            with self.assertRaisesRegex(ValueError, "requires a trusted public key"):
                inspect_program(root, require_signature=True)

    def test_real_compiled_program_authenticates_with_pinned_ed25519_key(self):
        from diffusionblocks_independent import compile_checkpoint

        import diffusionblocks as db
        from diffusionblocks.parity_examples import residual_factory

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            program = compile_checkpoint(
                **residual_factory(), checkpoint=None, blocks=2
            )
            program_directory = root / "program"
            program.save_pretrained(program_directory)
            private_key = root / "private.pem"
            public_key = root / "public.pem"
            db.generate_signing_keypair(private_key, public_key)
            db.sign_pretrained(program_directory, private_key)
            report = inspect_program(
                program_directory,
                trusted_public_key_path=public_key,
                require_signature=True,
            )
        self.assertTrue(report["authenticity"]["authenticated"])
        self.assertEqual(
            report["authenticity"]["signature"]["artifact_sha256"],
            report["manifest_sha256"],
        )

    def test_load_program_preserves_serialized_independent_contract(self):
        from diffusionblocks_independent import compile_checkpoint

        from diffusionblocks.parity_examples import residual_factory

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = residual_factory()
            inputs = payload["example_args"][0]
            program = compile_checkpoint(**payload, checkpoint=None, blocks=2)
            program.save_pretrained(root)
            loaded = load_program(root)
            self.assertEqual(len(loaded.blocks), 2)
            self.assertEqual(loaded.zero_sigma_assembled(inputs).shape, inputs.shape)

    def test_launch_plan_uses_disjoint_groups_and_waves(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._program(Path(temporary))
            plan = launch_plan(
                root,
                batch_factory="example:factory",
                steps=10,
                output_directory=Path(temporary) / "run",
                devices=("0", "1"),
                precision="bf16",
            )
        self.assertEqual(plan["execution"]["concurrent_block_jobs"], 2)
        self.assertEqual(plan["execution"]["waves"], 2)
        self.assertEqual(
            [job["cuda_visible_devices"] for job in plan["jobs"]],
            ["0", "1", "0", "1"],
        )
        self.assertTrue(
            all(
                "diffusionblocks_independent.worker" in job["command"]
                for job in plan["jobs"]
            )
        )
        self.assertTrue(all("--checkpoint" in job["command"] for job in plan["jobs"]))

    def test_resume_and_overwrite_are_explicit_and_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._program(Path(temporary))
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                launch_plan(
                    root,
                    batch_factory="example:factory",
                    steps=10,
                    output_directory=Path(temporary) / "run",
                    devices=("cpu",),
                    resume=True,
                    overwrite=True,
                )
            plan = launch_plan(
                root,
                batch_factory="example:factory",
                steps=10,
                output_directory=Path(temporary) / "run",
                devices=("cpu",),
                resume=True,
                override_resume_learning_rate=True,
            )
        self.assertIn("--resume", plan["jobs"][0]["command"])
        self.assertIn(
            "--override-resume-learning-rate", plan["jobs"][0]["command"]
        )
        self.assertFalse(plan["execution"]["overwrite"])

    def test_multi_device_block_plan_uses_core_distributed_trainer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._program(Path(temporary))
            plan = launch_plan(
                root,
                batch_factory="example:factory",
                steps=10,
                output_directory=Path(temporary) / "run",
                devices=("0", "1", "2", "3"),
                processes_per_block=2,
            )
        self.assertEqual(plan["execution"]["concurrent_block_jobs"], 2)
        self.assertTrue(
            all(
                "automatic-train-block-distributed" in job["command"]
                and "--fsdp-degree" in job["command"]
                for job in plan["jobs"]
            )
        )

    def test_hierarchical_parallel_degrees_compose_to_block_world_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._program(Path(temporary))
            plan = launch_plan(
                root,
                batch_factory="example:factory",
                steps=10,
                output_directory=Path(temporary) / "run",
                devices=tuple(str(index) for index in range(8)),
                processes_per_block=8,
                fsdp_degree=2,
                tensor_parallel_degree=2,
                context_parallel_degree=2,
            )
            command = plan["jobs"][0]["command"]
            self.assertEqual(plan["execution"]["fsdp_degree"], 2)
            self.assertEqual(plan["execution"]["tensor_parallel_degree"], 2)
            self.assertEqual(plan["execution"]["context_parallel_degree"], 2)
            self.assertIn("--tensor-parallel-degree", command)
            self.assertIn("--context-parallel-degree", command)
            with self.assertRaisesRegex(ValueError, "must equal"):
                launch_plan(
                    root,
                    batch_factory="example:factory",
                    steps=10,
                    output_directory=Path(temporary) / "bad",
                    devices=tuple(str(index) for index in range(8)),
                    processes_per_block=8,
                    fsdp_degree=4,
                    tensor_parallel_degree=2,
                    context_parallel_degree=2,
                )

    def test_cpu_slots_can_form_a_multi_process_block_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._program(Path(temporary))
            plan = launch_plan(
                root,
                batch_factory="example:factory",
                steps=1,
                output_directory=Path(temporary) / "run",
                devices=("cpu", "cpu"),
                processes_per_block=2,
                fsdp_degree=2,
            )
        self.assertEqual(plan["execution"]["groups"], [["cpu:0", "cpu:1"]])
        self.assertEqual(plan["execution"]["concurrent_block_jobs"], 1)
        self.assertIsNone(plan["jobs"][0]["cuda_visible_devices"])

    def test_distributed_checkpoint_tree_is_signed_and_tamper_evident(self):
        import diffusionblocks as db

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "step_00000001"
            checkpoint.mkdir()
            (checkpoint / ".metadata").write_bytes(b"metadata")
            (checkpoint / "__0_0.distcp").write_bytes(b"rank zero")
            (root / "latest.json").write_text(
                json.dumps(
                    {
                        "checkpoint": checkpoint.name,
                        "block_id": 0,
                        "step": 1,
                        "program_sha256": "a" * 64,
                    }
                ),
                encoding="utf-8",
            )
            private_key = root / "private.pem"
            public_key = root / "public.pem"
            db.generate_signing_keypair(private_key, public_key)
            signed = sign_distributed_checkpoint(
                root, private_key, trusted_public_key_path=public_key
            )
            self.assertTrue(signed["authenticated"])
            verified = verify_distributed_checkpoint(
                root, trusted_public_key_path=public_key
            )
            self.assertTrue(verified["authenticated"])
            (checkpoint / "__0_0.distcp").write_bytes(b"forged payload")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                verify_distributed_checkpoint(root, trusted_public_key_path=public_key)

    def test_exact_release_archives_remain_frozen_when_present(self):
        repository = Path(__file__).resolve().parents[3]
        release = repository / "dist" / "exact-0.17.0"
        if not release.is_dir():
            self.skipTest("monorepo exact release is not present")
        expected = {
            "diffusionblocks_v5-0.17.0-py3-none-any.whl": "73cd9710be2de7bf0f54c1b4b5e3e98e1b02a55f9129d7636a794d9eef92ba2d",
            "diffusionblocks_v5-0.17.0.tar.gz": "a278a168f758d344e6e72dc77b2074c41ad018cef5a630828d4983b218352367",
        }
        for name, digest in expected.items():
            self.assertEqual(sha256((release / name).read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
