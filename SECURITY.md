# Security policy

ZeroGraph handles model checkpoints, executable factories, serialized block
programs, optimizer state, signatures, and distributed checkpoint trees. Treat
untrusted artifacts as hostile.

## Supported versions

| Version | Supported |
|---|---|
| 0.3.x | Yes |
| Earlier experimental versions | No |

## Report a vulnerability

Use GitHub's private vulnerability reporting for this repository. Do not open a
public issue and do not attach private model weights, credentials, signing keys,
or proprietary training data.

Include the affected version, minimal reproduction, impact, and any proposed
mitigation. Alfalfa Labs will acknowledge a complete report as soon as
practical, investigate privately, and coordinate disclosure after a fix is
available.

## Security boundaries

- Program hashes establish integrity, not publisher identity.
- Signatures are meaningful only against an independently trusted public key.
- Pickle-based checkpoints and executable factories can run code.
- A block checkpoint must match its program, block identifier, optimizer, and
  data/RNG provenance before resume.
- Zero inter-block gradient traffic does not imply zero checkpoint, evaluation,
  or intra-block distributed communication.
