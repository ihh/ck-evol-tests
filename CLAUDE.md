# CLAUDE.md

## Project overview

Chapman-Kolmogorov tests for sequence evolvers using TKF92/TKF91 indel models. Single-file Python implementation with a LaTeX paper.

## Key files

- `sample_intermediates.py` — All implementation code (models, HMMs, sampler, CLI)
- `test_sample_intermediates.py` — pytest test suite
- `main.tex` — LaTeX paper describing the math

## Commands

```bash
# Run tests
python -m pytest

# Run built-in integration tests
python sample_intermediates.py --test-all

# Example run
python sample_intermediates.py --model jc --A ACGT --C ACGT --N 10
```

## Dependencies

- Python 3.8+, numpy, scipy

## Architecture

The code is organized in a single file with these sections:
1. **Substitution models** — JC, F81, K2P, LG, file-based rate matrices
2. **TKF92 parameters** — Indel model coefficients (alpha, beta, gamma, kappa)
3. **Pair HMM** — 5-state conditional pair HMM (S, M, I, D, E)
4. **Triad HMM** — 13-state HMM marginalizing over intermediate sequence B
5. **Null state elimination** — Removes null states for efficient forward algorithm
6. **Forward algorithm** — Log-space forward with stochastic traceback
7. **Sampler** — `sample_intermediates()` is the main entry point
8. **CLI** — argparse-based command-line interface

## Conventions

- All likelihood computations use log-space to avoid underflow
- Rate matrices Q are normalised so mean substitution rate = 1
- State indices: S=0, M=1, I=2, D=3, E=4 (pair HMM)
