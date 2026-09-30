# Reproducibility code snapshot

This directory is the anonymous source snapshot used by the accompanying
paper. It contains the training, evaluation, baseline, ablation, parsing, and
test modules together with the configuration templates required by the test
suite. See `../README.md` for the numerical audit, environment setup, and full
pipeline commands.

The snapshot excludes model weights, adapters, benchmark examples, raw model
generations, and machine-specific paths. Use `../reproduce/prepare_workspace.py` to resolve the recorded path markers into an empty rerun workspace; see `../REPRODUCTION.md` for commands.
