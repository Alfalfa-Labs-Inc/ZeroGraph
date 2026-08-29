import tempfile
import unittest
from pathlib import Path

from diffusionblocks_independent import benchmark_modes


class MatchedBenchmarkTests(unittest.TestCase):
    def test_three_mode_report_preserves_claim_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = benchmark_modes(
                model_factory="diffusionblocks.parity_examples:residual_factory",
                batch_factory=(
                    "diffusionblocks.parity_examples:residual_batch_factory"
                ),
                blocks=2,
                steps=1,
                output_directory=Path(temporary) / "evidence",
                devices=("cpu", "cpu"),
                precision="fp32",
                learning_rate=1e-3,
                optimizer="adamw",
            )
        self.assertEqual(report["status"], "measured")
        self.assertEqual(report["exact_rematerialization"]["bitwise_steps_compared"], 1)
        self.assertEqual(report["independent"]["inter_block_gradient_bytes"], 0)
        self.assertGreater(
            report["independent"]["serialized_state_working_set"][
                "sum_over_largest_active_block"
            ],
            1.0,
        )
        self.assertFalse(report["accelerator_time_view"]["matched_within_20_percent"])
        self.assertFalse(report["quality_metric_scope"]["directly_comparable"])


if __name__ == "__main__":
    unittest.main()
