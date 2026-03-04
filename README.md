# ck-evol-tests

Chapman-Kolmogorov tests for sequence evolvers. Samples intermediate sequences from the posterior distribution under the TKF92/TKF91 indel model with pluggable substitution models.

Given an evolutionary trajectory A → B → C, this tool samples intermediate sequences B from the exact posterior p(B|A,C) implied by the TKF92 pair HMM, using a triad HMM with stochastic traceback.

## Installation

Requires Python 3.8+ with NumPy and SciPy:

```bash
pip install numpy scipy
```

For running tests:

```bash
pip install pytest
```

## Usage

```bash
# Jukes-Cantor model, short DNA sequences
python sample_intermediates.py \
    --model jc --lam 0.1 --mu 0.2 \
    --ell1 0.5 --ell2 0.5 --N 10 \
    --A ACGT --C ACGT

# Le-Gascuel amino acid model with JSON output
python sample_intermediates.py \
    --model lg --N 10000 --top 5 --json \
    --A ELVIS --C LIVES

# Kimura 2-parameter, sequences from FASTA
python sample_intermediates.py \
    --model k2p --k2p-kappa 2.0 \
    --lam 0.1 --mu 0.2 --r 0.1 \
    --ell1 0.3 --ell2 0.7 --N 20 \
    --fasta seqs.fa

# Custom rate matrix from file
python sample_intermediates.py \
    --model file --model-file mymodel.txt \
    --lam 0.1 --mu 0.2 \
    --A ACGT --C ACGT
```

### Substitution models

| Flag | Model | Description |
|------|-------|-------------|
| `--model jc` | Jukes-Cantor | DNA, uniform rates and frequencies |
| `--model f81` | F81 | DNA, user-supplied `--pi` frequencies |
| `--model k2p` | Kimura 2-parameter | DNA, `--k2p-kappa` transition/transversion ratio |
| `--model lg` | Le & Gascuel 2008 | 20-state amino acid model |
| `--model file` | Custom | Rate matrix from `--model-file` |

### Key parameters

- `--lam` / `--mu` — TKF insertion/deletion rates (requires lam < mu)
- `--r` — Fragment extension probability (0 = TKF91, >0 = TKF92)
- `--ell1` / `--ell2` — Branch lengths for A→B and B→C
- `--N` — Number of posterior samples
- `--seed` — Random seed for reproducibility

Run `python sample_intermediates.py --help` for full options.

## Testing

```bash
# pytest suite
python -m pytest

# Built-in integration tests
python sample_intermediates.py --test-all
```

## Reference

See `main.tex` for the mathematical framework and algorithm descriptions.
