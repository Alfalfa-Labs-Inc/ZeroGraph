import tempfile
import unittest

import torch
from diffusionblocks_independent import (
    IndependentContract,
    compile_checkpoint,
    diagnose_program,
)


def regression_step(model, batch):
    inputs, targets = batch
    output = model(inputs)
    return torch.nn.functional.mse_loss(output, targets), output


def context_regression_step(model, batch):
    inputs, context, targets = batch
    output = model(inputs, context)
    return torch.nn.functional.mse_loss(output, targets), output


def causal_lm_step(model, batch):
    input_ids, attention_mask, labels = batch
    output = model(input_ids, attention_mask=attention_mask)
    logits = output if isinstance(output, torch.Tensor) else output[0]
    return (
        torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
        ),
        logits,
    )


class ResidualMLP(torch.nn.Module):
    def __init__(self, width=8, depth=3):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(width, width) for _ in range(depth)
        )

    def forward(self, value):
        for layer in self.layers:
            value = value + torch.nn.functional.gelu(layer(value))
        return value


class ResidualCNN(torch.nn.Module):
    def __init__(self, channels=4, depth=3):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            torch.nn.Conv2d(channels, channels, 3, padding=1) for _ in range(depth)
        )

    def forward(self, value):
        for layer in self.layers:
            value = value + torch.nn.functional.silu(layer(value))
        return value


class MultiBranchResidual(torch.nn.Module):
    def __init__(self, width=8, depth=2):
        super().__init__()
        self.left = torch.nn.ModuleList(
            torch.nn.Linear(width, width) for _ in range(depth)
        )
        self.right = torch.nn.ModuleList(
            torch.nn.Linear(width, width) for _ in range(depth)
        )

    def forward(self, value):
        for left, right in zip(self.left, self.right, strict=True):
            update = torch.nn.functional.gelu(left(value)) * torch.sigmoid(right(value))
            value = value + update
        return value


class DiTConditionedLayer(torch.nn.Module):
    def __init__(self, width=16, heads=4):
        super().__init__()
        self.width = width
        self.heads = heads
        self.norm = torch.nn.LayerNorm(width)
        self.modulation = torch.nn.Linear(width, 2 * width)
        self.qkv = torch.nn.Linear(width, 3 * width)
        self.projection = torch.nn.Linear(width, width)
        self.mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(width),
            torch.nn.Linear(width, 4 * width),
            torch.nn.GELU(),
            torch.nn.Linear(4 * width, width),
        )

    def forward(self, state, timestep_embedding):
        scale, shift = self.modulation(timestep_embedding).chunk(2, dim=-1)
        normalized = self.norm(state)
        normalized = normalized * (1 + scale[:, None, :]) + shift[:, None, :]
        batch, tokens, _ = state.shape
        qkv = self.qkv(normalized).view(
            batch, tokens, 3, self.heads, self.width // self.heads
        )
        query, key, value = qkv.unbind(2)
        attended = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
        )
        state = state + self.projection(
            attended.transpose(1, 2).reshape(batch, tokens, self.width)
        )
        return state + self.mlp(state)


class DiTConditionedResidual(torch.nn.Module):
    def __init__(self, depth=3):
        super().__init__()
        self.layers = torch.nn.ModuleList(DiTConditionedLayer() for _ in range(depth))

    def forward(self, state, timestep_embedding):
        for layer in self.layers:
            state = layer(state, timestep_embedding)
        return state


class NonResidualRecurrent(torch.nn.Module):
    def __init__(self, width=8, depth=3):
        super().__init__()
        self.cells = torch.nn.ModuleList(
            torch.nn.GRUCell(width, width) for _ in range(depth)
        )

    def forward(self, value):
        state = torch.zeros_like(value)
        for cell in self.cells:
            state = cell(value, state)
        return state


class ArchitectureMatrixTests(unittest.TestCase):
    def _compile(self, model, inputs, targets, *, blocks, context=None):
        if context is None:
            step_fn = regression_step
            example_args = (inputs,)
            example_batches = ((inputs, targets),)
            example_kwargs = None
        else:
            step_fn = context_regression_step
            example_args = (inputs, context)
            example_batches = ((inputs, context, targets),)
            example_kwargs = None
        return compile_checkpoint(
            model,
            checkpoint=None,
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
            step_fn=step_fn,
            example_args=example_args,
            example_kwargs=example_kwargs,
            example_batches=example_batches,
            blocks=blocks,
        )

    def _assert_independent_program(self, model, program, inputs, batch, context=None):
        contract = program.compatibility.inferred["independent_package_contract"]
        self.assertEqual(contract["package_mode"], IndependentContract().package_mode)
        self.assertFalse(contract["bitwise_ordinary_training_parity"])
        if context is None:
            expected = model(inputs)
            assembled = program.zero_sigma_assembled(inputs)
        else:
            expected = model(inputs, context)
            assembled = program.zero_sigma_assembled(
                inputs, {program.context_names[0]: context}
            )
        self.assertTrue(torch.equal(expected, assembled))
        self.assertTrue(program.verify_local_isolation(batch)["passed"])
        optimizer = program.make_optimizers()[0]
        metrics = program.train_local_step(0, batch, optimizer, sigma=1.0)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))

    def test_residual_mlp(self):
        torch.manual_seed(101)
        model = ResidualMLP()
        inputs = torch.randn(2, 8)
        targets = torch.randn_like(inputs)
        program = self._compile(model, inputs, targets, blocks=3)
        self._assert_independent_program(model, program, inputs, (inputs, targets))

    def test_channel_first_residual_cnn(self):
        torch.manual_seed(102)
        model = ResidualCNN()
        inputs = torch.randn(2, 4, 8, 8)
        targets = torch.randn_like(inputs)
        program = self._compile(model, inputs, targets, blocks=3)
        self.assertEqual(program.compatibility.inferred["feature_axis"], 1)
        self._assert_independent_program(model, program, inputs, (inputs, targets))

    def test_multibranch_residual_dataflow(self):
        torch.manual_seed(103)
        model = MultiBranchResidual()
        inputs = torch.randn(2, 8)
        targets = torch.randn_like(inputs)
        program = self._compile(model, inputs, targets, blocks=2)
        self._assert_independent_program(model, program, inputs, (inputs, targets))

    def test_dit_style_conditioning_attention(self):
        torch.manual_seed(104)
        model = DiTConditionedResidual()
        latent = torch.randn(2, 8, 16)
        context = torch.randn(2, 16)
        target = torch.randn_like(latent)
        program = self._compile(model, latent, target, blocks=3, context=context)
        self.assertEqual(
            program.compatibility.inferred["context_roles"]["timestep_embedding"],
            "conditioning",
        )
        self._assert_independent_program(
            model,
            program,
            latent,
            (latent, context, target),
            context=context,
        )

    def test_diagnostics_stream_blocks_and_measure_context_permutation(self):
        torch.manual_seed(107)
        model = DiTConditionedResidual()
        latent = torch.randn(2, 8, 16)
        context = torch.randn(2, 16)
        target = torch.randn_like(latent)
        program = self._compile(model, latent, target, blocks=3, context=context)
        with tempfile.TemporaryDirectory() as temporary:
            program.save_pretrained(temporary)
            report = diagnose_program(
                temporary,
                batch_factory=(
                    "diffusionblocks.parity_examples:attention_sequence_batch_factory"
                ),
                probe_batches=1,
                sigma_points=2,
                seed=108,
            )
        self.assertEqual(report["block_count"], 3)
        self.assertEqual(report["peak_resident_diffusion_blocks"], 1)
        self.assertTrue(
            all(
                row["leakage_status"] == "measured_context_permutation"
                and row["capacity_ratio"] > 0
                and len(row["sigma_probes"]) == 2
                for row in report["blocks"]
            )
        )

    def test_nonresidual_recurrent_graph_is_rejected_actionably(self):
        from diffusionblocks import AutomaticConversionError

        torch.manual_seed(105)
        model = NonResidualRecurrent()
        inputs = torch.randn(2, 8)
        targets = torch.randn_like(inputs)
        with self.assertRaises(AutomaticConversionError) as captured:
            self._compile(model, inputs, targets, blocks=2)
        self.assertTrue(str(captured.exception).strip())

    def test_real_huggingface_causal_lm_families(self):
        try:
            import transformers
        except ImportError:
            self.skipTest("Hugging Face extra is not installed")

        configurations = (
            (
                transformers.GPT2LMHeadModel,
                transformers.GPT2Config(
                    n_layer=2,
                    n_head=2,
                    n_embd=16,
                    n_positions=16,
                    vocab_size=32,
                    bos_token_id=0,
                    eos_token_id=1,
                    pad_token_id=2,
                    use_cache=False,
                ),
            ),
            (
                transformers.LlamaForCausalLM,
                transformers.LlamaConfig(
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    hidden_size=16,
                    intermediate_size=32,
                    vocab_size=32,
                    max_position_embeddings=16,
                    bos_token_id=0,
                    eos_token_id=1,
                    pad_token_id=2,
                    use_cache=False,
                ),
            ),
            (
                transformers.MistralForCausalLM,
                transformers.MistralConfig(
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    hidden_size=16,
                    intermediate_size=32,
                    vocab_size=32,
                    max_position_embeddings=16,
                    sliding_window=16,
                    bos_token_id=0,
                    eos_token_id=1,
                    pad_token_id=2,
                    use_cache=False,
                ),
            ),
            (
                transformers.Qwen2ForCausalLM,
                transformers.Qwen2Config(
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    hidden_size=16,
                    intermediate_size=32,
                    vocab_size=32,
                    max_position_embeddings=16,
                    bos_token_id=0,
                    eos_token_id=1,
                    pad_token_id=2,
                    use_cache=False,
                ),
            ),
            (
                transformers.GemmaForCausalLM,
                transformers.GemmaConfig(
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    head_dim=8,
                    hidden_size=16,
                    intermediate_size=32,
                    vocab_size=32,
                    max_position_embeddings=16,
                    bos_token_id=0,
                    eos_token_id=1,
                    pad_token_id=2,
                    use_cache=False,
                ),
            ),
            (
                transformers.Phi3ForCausalLM,
                transformers.Phi3Config(
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    hidden_size=16,
                    intermediate_size=32,
                    vocab_size=32,
                    max_position_embeddings=16,
                    original_max_position_embeddings=16,
                    bos_token_id=0,
                    eos_token_id=1,
                    pad_token_id=2,
                    use_cache=False,
                ),
            ),
        )
        input_ids = torch.randint(0, 32, (2, 5))
        attention_mask = torch.ones_like(input_ids)
        labels = torch.randint(0, 32, (2, 5))
        for model_type, config in configurations:
            with self.subTest(model=model_type.__name__):
                torch.manual_seed(106)
                model = model_type(config)
                program = compile_checkpoint(
                    model,
                    checkpoint=None,
                    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
                    step_fn=causal_lm_step,
                    example_args=(input_ids,),
                    example_kwargs={"attention_mask": attention_mask},
                    example_batches=((input_ids, attention_mask, labels),),
                    blocks=2,
                )
                self.assertTrue(
                    program.compatibility.inferred[
                        "automatically_synthesized_causal_lm_view"
                    ]
                )
                self.assertTrue(
                    program.compatibility.inferred["independent_package_contract"]
                )
                self.assertTrue(
                    program.verify_local_isolation((input_ids, attention_mask, labels))[
                        "passed"
                    ]
                )


if __name__ == "__main__":
    unittest.main()
