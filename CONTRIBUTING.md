# Contributing to ZeroGraph

Thank you for helping make independent blockwise training practical and honest.

## Start here

Install ExactGraph first, then ZeroGraph:

```bash
python -m pip install 'git+https://github.com/Alfalfa-Labs-Inc/ExactGraph.git'
python -m pip install -e '.[dev]'
pytest -q
```

## Pull-request standard

Every pull request should:

1. state which independent-training contract it changes;
2. add focused tests for isolation, resume, and assembled behavior;
3. preserve zero inter-block gradients in strict-independence mode;
4. report quality as well as systems metrics for new performance claims;
5. update documentation and the changelog when behavior changes;
6. keep parity claims explicitly separate from ExactGraph.

## Benchmark claims

Report model/data revisions, block allocation, seeds, tokens, hardware topology,
peak HBM, critical path, total accelerator-hours, communication bytes, local
loss windows, and assembled quality. A logical model shape is not a trained
quality result.

## Contributions and licensing

By submitting a contribution, you agree that it may be distributed under the
project's Apache License 2.0 and that you have the right to submit it. Please
use commits attributable to an identity you control.

For vulnerabilities, follow [SECURITY.md](SECURITY.md) instead of opening a
public issue.
