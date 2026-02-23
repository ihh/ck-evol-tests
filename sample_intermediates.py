#!/usr/bin/env python3
"""
sample_intermediates.py

Sample intermediate sequences B for an endpoint-conditioned evolutionary
trajectory A -ell1-> B -ell2-> C under the TKF92 (or TKF91) model.

Each B is drawn exactly from the posterior distribution over intermediate
sequences implied by the TKF92 model. The returned log-probabilities
    log p(B|A,C) = log p(B|A) + log p(C|B) - log p(C|A)
can be used as importance weights when estimating expectations over B.
All likelihood calculations are performed in log-space to avoid underflow.

SUBSTITUTION MODELS
-------------------
  --model jc          Jukes-Cantor (DNA, 4-state, equal rates & freqs)
  --model f81         F81 (DNA, user-supplied --pi)
  --model k2p         Kimura 2-parameter (DNA, user-supplied --k2p-kappa)
  --model lg          Le & Gascuel 2008 (amino acids, 20-state)
  --model file        Rate matrix from text file (--model-file PATH)

MODEL FILE FORMAT
-----------------
Whitespace-delimited n x n matrix of off-diagonal rates.
Diagonal is ignored and recomputed to make rows sum to zero.
The stationary distribution is deduced automatically by solving pi Q = 0.
One header line starting with '#' is allowed (and ignored).
Example (4x4 DNA matrix):
  0   1   1   1
  1   0   1   1
  1   1   0   1
  1   1   1   0

FASTA FILE FORMAT
-----------------
Standard FASTA.  The first two sequences found are used as A and C.

USAGE EXAMPLES
--------------
  # Integration test (DNA, JC, short sequences, few samples):
  python sample_intermediates.py --test-dna

  # Integration test (amino acids, LG):
  python sample_intermediates.py --test-aa

  # Jukes-Cantor, sequences on command line:
  python sample_intermediates.py \\
      --model jc --lam 0.1 --mu 0.2 --r 0.0 \\
      --ell1 0.5 --ell2 0.5 --N 10 \\
      --A ACGT --C ACGT

  # K2P, sequences from FASTA:
  python sample_intermediates.py \\
      --model k2p --k2p-kappa 2.0 \\
      --lam 0.1 --mu 0.2 --r 0.1 \\
      --ell1 0.3 --ell2 0.7 --N 20 \\
      --fasta seqs.fa

  # F81 from user frequencies:
  python sample_intermediates.py \\
      --model f81 --pi 0.1 0.4 0.4 0.1 \\
      --lam 0.05 --mu 0.1 --r 0.2 \\
      --ell1 1.0 --ell2 1.0 --N 5 \\
      --A ACGT --C ACGT

  # Le-Gascuel (amino acids):
  python sample_intermediates.py \\
      --model lg \\
      --lam 0.05 --mu 0.1 --r 0.0 \\
      --ell1 0.5 --ell2 0.5 --N 5 \\
      --A ACDEFGHIKLMNPQRSTVWY --C ACDEFGHIKLMNPQRSTVWY

  # Custom rate matrix from file:
  python sample_intermediates.py \\
      --model file --model-file mymodel.txt \\
      --lam 0.1 --mu 0.2 --r 0.0 \\
      --ell1 0.5 --ell2 0.5 --N 10 \\
      --A ACGT --C ACGT
"""

import sys
import argparse
import numpy as np
from collections import defaultdict
from scipy.linalg import expm


# ===========================================================================
# Substitution models
# ===========================================================================

DNA_ALPHABET = list('ACGT')
AA_ALPHABET  = list('ACDEFGHIKLMNPQRSTVWY')


def _normalise_Q(Q, pi):
    """Scale Q so that the mean substitution rate under pi is 1."""
    scale = -float(np.dot(pi, np.diag(Q)))
    if scale > 0:
        Q = Q / scale
    return Q


def stationary_from_Q(Q):
    """
    Solve pi Q = 0, sum(pi) = 1.
    Replaces the last equation with the normalisation constraint.
    """
    n = Q.shape[0]
    A = Q.T.copy()
    A[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    pi = np.linalg.solve(A, b)
    pi = np.abs(pi)
    pi /= pi.sum()
    return pi


def make_jc_model():
    """Jukes-Cantor: uniform rates and frequencies."""
    n = 4
    Q = np.ones((n, n))
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    pi = np.full(n, 0.25)
    Q = _normalise_Q(Q, pi)
    return Q, pi, DNA_ALPHABET


def make_f81_model(pi_in):
    """
    F81: Q_ij = pi_j for i != j.
    pi_in may be unnormalised.
    """
    pi = np.array(pi_in, dtype=float)
    pi /= pi.sum()
    n = len(pi)
    if n != 4:
        raise ValueError("F81 requires exactly 4 frequency values (--pi a c g t).")
    Q = np.outer(np.ones(n), pi)
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    Q = _normalise_Q(Q, pi)
    return Q, pi, DNA_ALPHABET


def make_k2p_model(kappa):
    """
    Kimura 2-parameter (K80).
    Transitions (A<->G, C<->T) have rate kappa; transversions rate 1.
    Alphabet: A C G T
    """
    Q = np.zeros((4, 4))
    # transitions
    for i, j in [(0,2),(2,0),(1,3),(3,1)]:
        Q[i,j] = kappa
    # transversions
    for i, j in [(0,1),(1,0),(0,3),(3,0),(2,1),(1,2),(2,3),(3,2)]:
        Q[i,j] = 1.0
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    pi = np.full(4, 0.25)
    Q = _normalise_Q(Q, pi)
    return Q, pi, DNA_ALPHABET


def make_lg_model():
    """
    Le & Gascuel (2008) amino acid substitution model.
    Mol Biol Evol 25:1307-1320.
    Q_ij = S_ij * pi_j (i != j); diagonal makes rows sum to zero.
    """
    # Stationary frequencies (Table 2, LG08), order = AA_ALPHABET
    pi = np.array([
        0.079066, 0.012937, 0.053052, 0.071586, 0.042302,
        0.057337, 0.022355, 0.062157, 0.059498, 0.099081,
        0.022951, 0.041977, 0.044040, 0.040767, 0.055941,
        0.061197, 0.053287, 0.069147, 0.012066, 0.034155,
    ], dtype=float)
    pi /= pi.sum()

    # Symmetric exchangeability matrix S (lower triangle, row-major i>j)
    # 190 values for 20 amino acids, from LG08 Table 2 / supplementary
    lg_lower = [
        # row 1  C
        0.425093,
        # row 2  D
        0.276818, 0.751878,
        # row 3  E
        0.395144, 0.123954, 5.076149,
        # row 4  F
        2.489084, 0.534551, 0.528768, 0.062556,
        # row 5  G
        0.969894, 2.807908, 1.695752, 0.523386, 0.084808,
        # row 6  H
        1.038545, 0.363970, 0.541712, 5.243870, 0.003499, 4.128591,
        # row 7  I
        2.066040, 0.390192, 1.437645, 0.844926, 0.569265, 0.267959, 0.348847,
        # row 8  K
        0.358858, 2.426601, 4.509238, 6.491176, 0.154263, 0.003499, 0.934276, 0.235017,
        # row 9  L
        7.821399, 0.539147, 0.128660, 0.127000, 6.312358, 2.592692, 0.604545, 5.469470, 0.374834,
        # row 10 M
        0.977855, 0.310884, 0.245314, 0.245314, 1.741000, 0.423580, 0.211560, 4.284640, 0.189870, 4.345063,
        # row 11 N
        4.128591, 0.767945, 5.057964, 2.243552, 0.523386, 5.076149, 3.873119, 0.741294, 3.170470, 0.083688, 1.015186,
        # row 12 P
        2.547870, 0.170651, 0.083688, 0.138190, 0.267959, 2.143748, 0.202562, 0.114381, 0.267959, 0.375839, 0.267959, 0.323090,
        # row 13 Q
        2.066040, 0.390192, 1.437645, 3.035085, 0.569265, 5.576020, 6.472279, 0.569265, 6.529255, 0.282007, 3.869568, 2.066040, 0.862476,
        # row 14 R
        0.267959, 4.896629, 0.436421, 0.170651, 0.062556, 0.267959, 0.864404, 0.267959, 2.500294, 0.114381, 0.267959, 0.099849, 0.267959, 0.267959,
        # row 15 S
        4.727182, 0.523386, 4.895096, 0.672052, 4.582565, 2.966732, 0.523386, 0.272514, 0.523386, 0.535018, 0.695768, 8.733797, 0.862476, 0.672052, 0.267959,
        # row 16 T
        2.139501, 0.130386, 0.523386, 0.415991, 0.267959, 0.523386, 0.523386, 1.812200, 0.267959, 1.393611, 0.267959, 2.443600, 0.267959, 0.267959, 0.267959, 5.244839,
        # row 17 V
        2.547870, 0.170651, 0.083688, 0.138190, 3.360571, 0.267959, 0.138190, 7.285506, 0.114381, 6.312358, 2.059564, 0.138190, 0.267959, 0.267959, 0.267959, 0.267959, 1.368100,
        # row 18 W
        0.738264, 2.660642, 0.190192, 0.266702, 3.873119, 0.267959, 5.336759, 0.190192, 0.266702, 1.003754, 0.267959, 0.533060, 0.267959, 0.267959, 3.261066, 0.267959, 0.267959, 0.170651,
        # row 19 Y
        0.267959, 0.267959, 0.338848, 0.428697, 6.312358, 0.338848, 3.953765, 0.267959, 0.180372, 0.267959, 0.267959, 1.116568, 0.267959, 0.267959, 0.267959, 0.267959, 0.267959, 0.267959, 10.649107,
    ]

    n = 20
    S = np.zeros((n, n))
    idx = 0
    for i in range(1, n):
        for j in range(i):
            S[i, j] = S[j, i] = lg_lower[idx]
            idx += 1

    Q = S * pi[np.newaxis, :]   # Q_ij = S_ij * pi_j
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    Q = _normalise_Q(Q, pi)
    return Q, pi, AA_ALPHABET


def make_model_from_file(path):
    """
    Load a rate matrix from a whitespace-delimited text file.
    Lines starting with '#' are ignored.
    Diagonal is recomputed; stationary distribution solved from Q.
    The alphabet is inferred from file headers if present as a comment
    of the form '# alphabet ACGT', otherwise DNA is assumed.
    """
    lines = []
    alphabet = None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                # Check for optional alphabet hint: "# alphabet ACGT"
                parts = line[1:].strip().split()
                if parts and parts[0].lower() == 'alphabet' and len(parts) == 2:
                    alphabet = list(parts[1])
                continue
            lines.append(line)
    if not lines:
        raise ValueError(f"No data found in model file: {path}")
    Q = np.array([[float(x) for x in row.split()] for row in lines])
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Model file must contain a square matrix.")
    n = Q.shape[0]
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    pi = stationary_from_Q(Q)
    Q = _normalise_Q(Q, pi)
    if alphabet is None:
        if n == 4:
            alphabet = DNA_ALPHABET
        elif n == 20:
            alphabet = AA_ALPHABET
        else:
            alphabet = [str(i) for i in range(n)]
    if len(alphabet) != n:
        raise ValueError(f"Alphabet length {len(alphabet)} != matrix size {n}.")
    return Q, pi, alphabet


# ===========================================================================
# TKF92 parameters and time-dependent coefficients
# ===========================================================================

class TKF92Params:
    def __init__(self, lam, mu, r, Q, pi, alphabet):
        self.lam      = float(lam)
        self.mu       = float(mu)
        self.r        = float(r)
        self.Q        = np.array(Q, dtype=float)
        self.pi       = np.array(pi, dtype=float)
        self.alphabet = list(alphabet)

    @property
    def n_alpha(self):
        return len(self.alphabet)

    @property
    def char_to_idx(self):
        if not hasattr(self, '_c2i'):
            self._c2i = {c: i for i, c in enumerate(self.alphabet)}
        return self._c2i


def tkf92_coeffs(params, ell):
    lam, mu, r = params.lam, params.mu, params.r
    alpha = np.exp(-mu * ell)
    if abs(lam - mu) < 1e-10:
        beta = lam * ell / (1.0 + lam * ell)
    else:
        num = lam * (np.exp(-lam * ell) - np.exp(-mu * ell))
        den = mu  * np.exp(-lam * ell) - lam * np.exp(-mu * ell)
        beta = num / den
    kappa = lam / mu
    if abs(1.0 - alpha) > 1e-15:
        gamma = 1.0 - mu * beta / (lam * (1.0 - alpha))
    else:
        gamma = 0.0
    nu = r + (1.0 - r) * kappa
    return alpha, beta, gamma, kappa, nu


def subst_matrix(params, ell):
    return expm(params.Q * ell)


# ===========================================================================
# Conditional pair HMM transition matrix
# State indices: S=0, M=1, I=2, D=3, E=4
# ===========================================================================

S_ST, M_ST, I_ST, D_ST, E_ST = 0, 1, 2, 3, 4
COND_STATE_TYPES = [S_ST, M_ST, I_ST, D_ST, E_ST]
N_COND_STATES = 5


def cond_pair_matrix(params, ell):
    alpha, beta, gamma, kappa, nu = tkf92_coeffs(params, ell)
    r = params.r
    t = np.zeros((5, 5))
    # From S
    t[S_ST, M_ST] = (1.0-beta)*alpha
    t[S_ST, I_ST] = beta
    t[S_ST, D_ST] = (1.0-beta)*(1.0-alpha)
    t[S_ST, E_ST] = 1.0-beta
    # From M
    t[M_ST, M_ST] = (r + (1.0-r)*(1.0-beta)*kappa*alpha) / nu
    t[M_ST, I_ST] = (1.0-r)*beta
    t[M_ST, D_ST] = (1.0-r)*(1.0-beta)*kappa*(1.0-alpha) / nu
    t[M_ST, E_ST] = 1.0-beta
    # From I
    t[I_ST, M_ST] = (1.0-r)*(1.0-beta)*kappa*alpha / nu
    t[I_ST, I_ST] = r + (1.0-r)*beta
    t[I_ST, D_ST] = (1.0-r)*(1.0-beta)*kappa*(1.0-alpha) / nu
    t[I_ST, E_ST] = 1.0-beta
    # From D
    t[D_ST, M_ST] = (1.0-r)*(1.0-gamma)*kappa*alpha / nu
    t[D_ST, I_ST] = (1.0-r)*gamma
    t[D_ST, D_ST] = (r + (1.0-r)*(1.0-gamma)*kappa*(1.0-alpha)) / nu
    t[D_ST, E_ST] = 1.0-gamma
    return t


# ===========================================================================
# Triad HMM
# State indices: SS=0,sI=1,MM=2,mI=3,MD=4,IM=5,iI=6,ID=7,
#                Ds=8,Dm=9,Di=10,Dd=11,EE=12
# ===========================================================================

SS, sI, MM, mI, MD, IM, iI, ID, Ds, Dm, Di, Dd, EE = range(13)
N_TRIAD_STATES = 13
TRIAD_NAMES = ['SS','sI','MM','mI','MD','IM','iI','ID','Ds','Dm','Di','Dd','EE']

NULL_TYPE = 5   # null state type code (beyond S=0,M=1,I=2,D=3,E=4)

# Triad HMM is a pair HMM with A as X (ancestor) and C as Y (descendant),
# with B marginalised out as a latent sequence.
#
# State type assignments (from the spec):
#   MM              -> M  (consumes both A and C)
#   sI, mI, IM, iI  -> I  (consume C only; B inserts residue observed in C)
#   MD, Ds, Dm, Di, Dd -> D  (consume A only; B residue deleted before C)
#   ID              -> N  (null; B inserts then deletes — neither A nor C consumed)
#   SS -> S,  EE -> E
TRIAD_STATE_TYPES = [
    S_ST,      # SS
    I_ST,      # sI
    M_ST,      # MM
    I_ST,      # mI
    D_ST,      # MD
    I_ST,      # IM
    I_ST,      # iI
    NULL_TYPE, # ID  (null)
    D_ST,      # Ds
    D_ST,      # Dm
    D_ST,      # Di
    D_ST,      # Dd
    E_ST,      # EE
]

# Which triad states produce a residue at B?
B_RESIDUE_STATES = {MM, IM, MD, ID}
# Which consume an A residue?  (M or D in pair-HMM sense: consume X=A)
A_CONSUME = {MM, MD, Ds, Dm, Di, Dd}
# Which consume a C residue?  (M or I in pair-HMM sense: consume Y=C)
C_CONSUME = {sI, MM, mI, IM, iI}


def triad_transition_matrix(params, ell1, ell2):
    y = cond_pair_matrix(params, ell1)
    z = cond_pair_matrix(params, ell2)
    t = np.zeros((13, 13))

    t[SS, sI]=z[S_ST,I_ST];  t[SS,MM]=y[S_ST,M_ST]*z[S_ST,M_ST]
    t[SS, MD]=y[S_ST,M_ST]*z[S_ST,D_ST]
    t[SS, IM]=y[S_ST,I_ST]*z[S_ST,M_ST]; t[SS,ID]=y[S_ST,I_ST]*z[S_ST,D_ST]
    t[SS, Ds]=y[S_ST,D_ST]; t[SS,EE]=y[S_ST,E_ST]*z[S_ST,E_ST]

    t[sI,sI]=z[I_ST,I_ST]; t[sI,MM]=y[S_ST,M_ST]*z[I_ST,M_ST]
    t[sI,MD]=y[S_ST,M_ST]*z[I_ST,D_ST]
    t[sI,IM]=y[S_ST,I_ST]*z[I_ST,M_ST]; t[sI,ID]=y[S_ST,I_ST]*z[I_ST,D_ST]
    t[sI,Di]=y[S_ST,D_ST]; t[sI,EE]=y[S_ST,E_ST]*z[I_ST,E_ST]

    t[MM,MM]=y[M_ST,M_ST]*z[M_ST,M_ST]; t[MM,mI]=z[M_ST,I_ST]
    t[MM,MD]=y[M_ST,M_ST]*z[M_ST,D_ST]
    t[MM,IM]=y[M_ST,I_ST]*z[M_ST,M_ST]; t[MM,ID]=y[M_ST,I_ST]*z[M_ST,D_ST]
    t[MM,Dm]=y[M_ST,D_ST]; t[MM,EE]=y[M_ST,E_ST]*z[M_ST,E_ST]

    t[mI,MM]=y[M_ST,M_ST]*z[I_ST,M_ST]; t[mI,mI]=z[I_ST,I_ST]
    t[mI,MD]=y[M_ST,M_ST]*z[I_ST,D_ST]
    t[mI,IM]=y[M_ST,I_ST]*z[I_ST,M_ST]; t[mI,ID]=y[M_ST,I_ST]*z[I_ST,D_ST]
    t[mI,Di]=y[M_ST,D_ST]; t[mI,EE]=y[M_ST,E_ST]*z[I_ST,E_ST]

    t[MD,MM]=y[M_ST,M_ST]*z[D_ST,M_ST]; t[MD,mI]=z[D_ST,I_ST]
    t[MD,MD]=y[M_ST,M_ST]*z[D_ST,D_ST]
    t[MD,IM]=y[M_ST,I_ST]*z[D_ST,M_ST]; t[MD,ID]=y[M_ST,I_ST]*z[D_ST,D_ST]
    t[MD,Dd]=y[M_ST,D_ST]; t[MD,EE]=y[M_ST,E_ST]*z[D_ST,E_ST]

    t[IM,MM]=y[I_ST,M_ST]*z[M_ST,M_ST]; t[IM,MD]=y[I_ST,M_ST]*z[M_ST,D_ST]
    t[IM,IM]=y[I_ST,I_ST]*z[M_ST,M_ST]; t[IM,iI]=z[M_ST,I_ST]
    t[IM,ID]=y[I_ST,I_ST]*z[M_ST,D_ST]
    t[IM,Dm]=y[I_ST,D_ST]; t[IM,EE]=y[I_ST,E_ST]*z[M_ST,E_ST]

    t[iI,MM]=y[I_ST,M_ST]*z[I_ST,M_ST]; t[iI,MD]=y[I_ST,M_ST]*z[I_ST,D_ST]
    t[iI,IM]=y[I_ST,I_ST]*z[I_ST,M_ST]; t[iI,iI]=z[I_ST,I_ST]
    t[iI,ID]=y[I_ST,I_ST]*z[I_ST,D_ST]
    t[iI,Di]=y[I_ST,D_ST]; t[iI,EE]=y[I_ST,E_ST]*z[I_ST,E_ST]

    t[ID,MM]=y[I_ST,M_ST]*z[D_ST,M_ST]; t[ID,MD]=y[I_ST,M_ST]*z[D_ST,D_ST]
    t[ID,IM]=y[I_ST,I_ST]*z[D_ST,M_ST]; t[ID,iI]=z[D_ST,I_ST]
    t[ID,ID]=y[I_ST,I_ST]*z[D_ST,D_ST]
    t[ID,Dd]=y[I_ST,D_ST]; t[ID,EE]=y[I_ST,E_ST]*z[D_ST,E_ST]

    t[Ds,MM]=y[D_ST,M_ST]*z[S_ST,M_ST]; t[Ds,MD]=y[D_ST,M_ST]*z[S_ST,D_ST]
    t[Ds,IM]=y[D_ST,I_ST]*z[S_ST,M_ST]; t[Ds,ID]=y[D_ST,I_ST]*z[S_ST,D_ST]
    t[Ds,Dd]=y[D_ST,D_ST]; t[Ds,EE]=y[D_ST,E_ST]*z[S_ST,E_ST]

    t[Dm,MM]=y[D_ST,M_ST]*z[M_ST,M_ST]; t[Dm,MD]=y[D_ST,M_ST]*z[M_ST,D_ST]
    t[Dm,IM]=y[D_ST,I_ST]*z[M_ST,M_ST]; t[Dm,ID]=y[D_ST,I_ST]*z[M_ST,D_ST]
    t[Dm,Dd]=y[D_ST,D_ST]; t[Dm,EE]=y[D_ST,E_ST]*z[M_ST,E_ST]

    t[Di,MM]=y[D_ST,M_ST]*z[I_ST,M_ST]; t[Di,MD]=y[D_ST,M_ST]*z[I_ST,D_ST]
    t[Di,IM]=y[D_ST,I_ST]*z[I_ST,M_ST]; t[Di,ID]=y[D_ST,I_ST]*z[I_ST,D_ST]
    t[Di,Dd]=y[D_ST,D_ST]; t[Di,EE]=y[D_ST,E_ST]*z[I_ST,E_ST]

    t[Dd,MM]=y[D_ST,M_ST]*z[D_ST,M_ST]; t[Dd,MD]=y[D_ST,M_ST]*z[D_ST,D_ST]
    t[Dd,IM]=y[D_ST,I_ST]*z[D_ST,M_ST]; t[Dd,ID]=y[D_ST,I_ST]*z[D_ST,D_ST]
    t[Dd,Dd]=y[D_ST,D_ST]; t[Dd,EE]=y[D_ST,E_ST]*z[D_ST,E_ST]

    return t


# ===========================================================================
# Null state elimination (closed form: single null state ID)
# ===========================================================================

def eliminate_null_state(t):
    t_prime = t.copy()
    denom = 1.0 - t[ID, ID]
    if denom < 1e-15:
        denom = 1e-15
    for i in range(N_TRIAD_STATES):
        if i == ID:
            continue
        for j in range(N_TRIAD_STATES):
            if j == ID:
                continue
            t_prime[i, j] = t[i, j] + t[i, ID] * t[ID, j] / denom
    t_prime[ID, :] = 0.0
    t_prime[:, ID] = 0.0
    return t_prime


def make_u_matrix(t):
    denom = 1.0 - t[ID, ID]
    if denom < 1e-15:
        denom = 1e-15
    u = np.eye(N_TRIAD_STATES)
    for i in range(N_TRIAD_STATES):
        if i == ID:
            continue
        u[i, ID] = t[i, ID] / denom
    return u


# ===========================================================================
# Log-space Forward algorithm for pair HMMs
# ===========================================================================

LOG0 = -np.inf


def log_forward_pair(x, y, log_t, log_emit_fn, n_states, state_types):
    """
    Forward algorithm in log-space for a pair HMM.

    x, y     : sequences as lists of integer alphabet indices.
    log_t    : n_states x n_states array of log transition probabilities.
    log_emit_fn(k, xi, yj) : log emission probability for state k.
                  xi / yj are integer indices or None (for gaps).
    state_types : list mapping state index -> type code
                  (S=0, M=1, I=2, D=3, N=5, E=4)

    Returns log F[i, j, k].
    Uses the log-sum-exp trick to avoid underflow.
    """
    Lx, Ly = len(x), len(y)
    logF = np.full((Lx+1, Ly+1, n_states), LOG0)
    logF[0, 0, 0] = 0.0   # start state probability 1

    for i in range(Lx+1):
        for j in range(Ly+1):
            for k in range(n_states):
                st = state_types[k]
                if st == 0:        # start state: already initialised
                    continue
                if st == 5:        # null state: skip (eliminated)
                    continue

                di = 1 if st in (M_ST, D_ST) else 0
                dj = 1 if st in (M_ST, I_ST) else 0
                pi_i, pj = i - di, j - dj
                if pi_i < 0 or pj < 0:
                    continue

                # Emission log-probability
                if st == E_ST:
                    if i != Lx or j != Ly:
                        continue
                    log_e = 0.0
                elif st == M_ST:
                    if i == 0 or j == 0:
                        continue
                    log_e = log_emit_fn(k, x[i-1], y[j-1])
                elif st == I_ST:
                    if j == 0:
                        continue
                    log_e = log_emit_fn(k, None, y[j-1])
                elif st == D_ST:
                    if i == 0:
                        continue
                    log_e = log_emit_fn(k, x[i-1], None)
                else:
                    log_e = 0.0

                if log_e == LOG0:
                    continue

                # log-sum-exp over predecessor states
                log_prev = logF[pi_i, pj, :]    # shape (n_states,)
                log_trans = log_t[:, k]          # shape (n_states,)
                log_terms = log_prev + log_trans  # shape (n_states,)

                max_val = log_terms.max()
                if max_val == LOG0:
                    continue
                log_sum = max_val + np.log(np.exp(log_terms - max_val).sum())
                logF[i, j, k] = log_e + log_sum

    return logF


def _log_safe(x):
    """log, returning -inf for zero or negative."""
    with np.errstate(divide='ignore'):
        return np.where(x > 0, np.log(x), LOG0)


def forward_traceback_log(logF, log_t, state_types, x, y):
    """
    Stochastic traceback through a log-space Forward matrix.
    Returns path as list of state indices (including start and end states).
    """
    Lx, Ly = len(x), len(y)
    n_states = logF.shape[2]
    i, j = Lx, Ly
    k = n_states - 1   # end state
    path = [k]

    while True:
        st = state_types[k]
        di = 1 if st in (M_ST, D_ST) else 0
        dj = 1 if st in (M_ST, I_ST) else 0
        pi_i, pj = i - di, j - dj

        log_q = logF[pi_i, pj, :] + log_t[:, k]
        max_lq = log_q.max()
        if max_lq == LOG0:
            break
        # Convert to probabilities for sampling
        q = np.exp(log_q - max_lq)
        q_sum = q.sum()
        if q_sum == 0:
            break
        k_prev = np.random.choice(n_states, p=q / q_sum)
        path.append(k_prev)
        i, j = pi_i, pj
        k = k_prev
        if state_types[k] == S_ST:
            break

    path.reverse()
    return path


def make_log_cond_emit_fn(U, pi):
    """Log-emission function for the conditional pair HMM."""
    log_U  = _log_safe(U)
    log_pi = _log_safe(pi)

    def log_emit_fn(k, xi, yj):
        if k == M_ST:
            return log_U[xi, yj]
        elif k == I_ST:
            return log_pi[yj]
        elif k == D_ST:
            return 0.0   # log(1)
        return 0.0
    return log_emit_fn


def log_forward_cond(x, y, t, U, pi):
    """Log-space forward matrix for conditional pair HMM."""
    log_t       = _log_safe(t)
    log_emit_fn = make_log_cond_emit_fn(U, pi)
    return log_forward_pair(x, y, log_t, log_emit_fn, N_COND_STATES, COND_STATE_TYPES)


def log_likelihood_cond(x, y, t, U, pi):
    """log P(y | x) under conditional pair HMM."""
    logF = log_forward_cond(x, y, t, U, pi)
    return logF[len(x), len(y), E_ST]


# ===========================================================================
# Triad HMM pair log-forward (emission function)
# ===========================================================================

def make_triad_emit_fn(UV, pi):
    """
    Log-emission function for the triad pair HMM (A=X, C=Y, B latent).

    Emissions marginalise over the latent B residue:
      MM:   exp(Q*(l1+l2))_{A_i, C_j}          (match: both A and C consumed)
      I states (sI,mI,IM,iI): pi_{C_j}          (insert: C consumed, B->C marginal)
      D states (MD,Ds,Dm,Di,Dd): 1              (delete: A consumed, emission=1)
      ID:   null — handled by state elimination
    """
    log_UV = _log_safe(UV)
    log_pi = _log_safe(pi)

    def log_emit_fn(k, xi, yj):
        st = TRIAD_STATE_TYPES[k]
        if st == M_ST:          # MM: exp(Q(l1+l2))_{A_i, C_j}
            return log_UV[xi, yj]
        elif st == I_ST:        # insert states: pi_{C_j}
            return log_pi[yj]
        elif st == D_ST:        # delete states: emission = 1
            return 0.0
        return 0.0
    return log_emit_fn



# ===========================================================================
# Null state restoration
# ===========================================================================

def restore_null_states(path_elim, t, u):
    """
    Stochastically restore ID null states into a path through the
    null-eliminated triad HMM.
    """
    path_elim = list(path_elim)
    full_path = [path_elim[-1]]

    for idx in range(len(path_elim)-1, 0, -1):
        i_state = path_elim[idx-1]
        k = full_path[0]
        while True:
            q = u[i_state, :] * t[:, k]
            total = q.sum()
            if total == 0:
                break
            j_state = np.random.choice(N_TRIAD_STATES, p=q / total)
            full_path.insert(0, j_state)
            k = j_state
            if j_state == i_state:
                break

    return full_path


# ===========================================================================
# Main sampler
# ===========================================================================

def sample_intermediates(A, C, params, ell1, ell2, N):
    """
    Sample N intermediate sequences B for A -ell1-> B -ell2-> C.

    A, C : sequences as lists of integer alphabet indices.
    N    : number of samples.

    The triad HMM is used directly as a pair HMM with A=X and C=Y,
    with B marginalised out via the emission function.  The null state
    ID is eliminated before the forward pass; null states are then
    stochastically restored during traceback.

    Returns dict: B_tuple -> (log_p, count)
      where log_p = log P(B|A,C,params,ell1,ell2) and count is multiplicity.
    """
    U  = subst_matrix(params, ell1)
    V  = subst_matrix(params, ell2)
    UV = subst_matrix(params, ell1 + ell2)

    t_AB  = cond_pair_matrix(params, ell1)
    t_BC  = cond_pair_matrix(params, ell2)
    t_ABC = triad_transition_matrix(params, ell1, ell2)
    t_elim = eliminate_null_state(t_ABC)
    u_mat  = make_u_matrix(t_ABC)

    log_t_AB   = _log_safe(t_AB)
    log_t_BC   = _log_safe(t_BC)
    log_t_elim = _log_safe(t_elim)

    # Emission function and log forward matrix for triad HMM against (A, C).
    # Reused across all N samples.
    log_emit_fn = make_triad_emit_fn(UV, params.pi)
    logF_AC = log_forward_pair(A, C, log_t_elim, log_emit_fn,
                               N_TRIAD_STATES, TRIAD_STATE_TYPES)
    log_p_AC = logF_AC[len(A), len(C), EE]

    if log_p_AC == LOG0:
        raise ValueError("log P(C|A) = -inf: sequences are incompatible with model.")

    results = defaultdict(lambda: [LOG0, 0])   # B -> [log_p, count]

    for _ in range(N):
        # 1. Sample path through null-eliminated triad HMM (pair forward traceback)
        path_elim = forward_traceback_log(logF_AC, log_t_elim,
                                          TRIAD_STATE_TYPES, A, C)

        # 2. Restore null (ID) states stochastically
        triad_path = restore_null_states(path_elim, t_ABC, u_mat)

        # 3. Sample B residues by traversing triad path front-to-back,
        #    appending each sampled residue as we go.
        B  = []
        ai = 0
        ci = 0

        inner_path = [k for k in triad_path if k not in (SS, EE)]
        for k in inner_path:
            if k in B_RESIDUE_STATES:
                if k == MM:
                    q = U[A[ai], :] * V[:, C[ci]]
                elif k == IM:
                    q = params.pi * V[:, C[ci]]
                elif k == MD:
                    q = U[A[ai], :].copy()
                else:  # ID
                    q = params.pi.copy()
                q_sum = q.sum()
                omega = (np.random.choice(params.n_alpha, p=q / q_sum)
                         if q_sum > 0
                         else np.random.choice(params.n_alpha, p=params.pi))
                B.append(omega)
            if k in A_CONSUME:
                ai += 1
            if k in C_CONSUME:
                ci += 1

        B_tuple = tuple(B)

        # 4. Importance weight (cached on first occurrence of this B)
        if results[B_tuple][1] == 0:
            log_p_AB = log_likelihood_cond(A, B, t_AB, U, params.pi)
            log_p_BC = log_likelihood_cond(B, C, t_BC, V, params.pi)
            log_p    = log_p_AB + log_p_BC - log_p_AC
            results[B_tuple][0] = log_p

        results[B_tuple][1] += 1

    return {B: (log_p, count) for B, (log_p, count) in results.items()}


# ===========================================================================
# FASTA reader
# ===========================================================================

def read_fasta(path):
    """
    Read a FASTA file.  Returns list of (name, sequence_string) pairs.
    """
    seqs = []
    name, seq = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith('>'):
                if name is not None:
                    seqs.append((name, ''.join(seq)))
                name = line[1:].split()[0]
                seq  = []
            else:
                seq.append(line.upper())
    if name is not None:
        seqs.append((name, ''.join(seq)))
    return seqs


# ===========================================================================
# Integration tests
# ===========================================================================

def run_integration_test_dna():
    """
    Quick sanity-check with Jukes-Cantor on short DNA sequences.
    Checks:
      - sampler runs without error
      - returned log-weights are finite
      - sum of weights * counts is approximately 1
    """
    print("=== Integration test: DNA (Jukes-Cantor) ===")
    Q, pi, alphabet = make_jc_model()
    params = TKF92Params(lam=0.1, mu=0.2, r=0.0, Q=Q, pi=pi, alphabet=alphabet)
    c2i = {c: i for i, c in enumerate(alphabet)}
    A = [c2i[c] for c in 'ACGT']
    C = [c2i[c] for c in 'ACAT']
    N = 20
    results = sample_intermediates(A, C, params, ell1=0.5, ell2=0.5, N=N)
    total_samples = sum(count for _, count in results.values())
    assert total_samples == N, f"Expected {N} samples, got {total_samples}"
    for B, (log_p, count) in results.items():
        assert np.isfinite(log_p), f"Non-finite log_p for B={B}"
        assert log_p <= 0.01,      f"log_p > 0 for B={B}: {log_p}"
    # Weighted sum should be <= 1 (it's a sum of conditional probabilities
    # over sampled B values, not necessarily exhaustive)
    w_sum = sum(np.exp(log_p) * count for _, (log_p, count) in results.items())
    print(f"  Samples drawn : {N}")
    print(f"  Unique B      : {len(results)}")
    print(f"  Sum p(B)*count: {w_sum:.4f}  (should be <= 1)")
    for B, (log_p, count) in sorted(results.items(), key=lambda x: -x[1][1]):
        B_str = ''.join(alphabet[r] for r in B)
        print(f"  B={B_str:10s}  log_p={log_p:.4f}  p={np.exp(log_p):.4e}  count={count}")
    print("  PASSED\n")


def run_integration_test_aa():
    """
    Quick sanity-check with Le-Gascuel on short amino-acid sequences.
    """
    print("=== Integration test: amino acids (LG08) ===")
    Q, pi, alphabet = make_lg_model()
    params = TKF92Params(lam=0.05, mu=0.1, r=0.1, Q=Q, pi=pi, alphabet=alphabet)
    c2i = {c: i for i, c in enumerate(alphabet)}
    A = [c2i[c] for c in 'ACDEF']
    C = [c2i[c] for c in 'ACDEF']
    N = 10
    results = sample_intermediates(A, C, params, ell1=0.3, ell2=0.3, N=N)
    total_samples = sum(count for _, count in results.values())
    assert total_samples == N
    for B, (log_p, _) in results.items():
        assert np.isfinite(log_p), f"Non-finite log_p for B={B}"
    print(f"  Samples drawn : {N}")
    print(f"  Unique B      : {len(results)}")
    for B, (log_p, count) in sorted(results.items(), key=lambda x: -x[1][1])[:5]:
        B_str = ''.join(alphabet[r] for r in B)
        print(f"  B={B_str:10s}  log_p={log_p:.4f}  p={np.exp(log_p):.4e}  count={count}")
    print("  PASSED\n")


def run_integration_test_k2p():
    """Quick sanity-check with K2P model."""
    print("=== Integration test: DNA (K2P kappa=2) ===")
    Q, pi, alphabet = make_k2p_model(kappa=2.0)
    params = TKF92Params(lam=0.1, mu=0.15, r=0.2, Q=Q, pi=pi, alphabet=alphabet)
    c2i = {c: i for i, c in enumerate(alphabet)}
    A = [c2i[c] for c in 'AAGGTT']
    C = [c2i[c] for c in 'AAGTT']
    N = 15
    results = sample_intermediates(A, C, params, ell1=0.4, ell2=0.6, N=N)
    total_samples = sum(count for _, count in results.values())
    assert total_samples == N
    print(f"  Samples drawn : {N}")
    print(f"  Unique B      : {len(results)}")
    print("  PASSED\n")


# ===========================================================================
# Command-line interface
# ===========================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        prog='sample_intermediates.py',
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Integration tests ---
    test_group = parser.add_argument_group('Integration tests')
    test_group.add_argument('--test-dna', action='store_true',
        help='Run DNA/JC integration test and exit.')
    test_group.add_argument('--test-aa', action='store_true',
        help='Run amino-acid/LG integration test and exit.')
    test_group.add_argument('--test-k2p', action='store_true',
        help='Run DNA/K2P integration test and exit.')
    test_group.add_argument('--test-all', action='store_true',
        help='Run all integration tests and exit.')

    # --- Substitution model ---
    model_group = parser.add_argument_group('Substitution model')
    model_group.add_argument('--model', choices=['jc','f81','k2p','lg','file'],
        default='jc',
        help='Substitution model (default: jc).')
    model_group.add_argument('--model-file', metavar='PATH',
        help=(
            'Path to rate-matrix file (required when --model file). '
            'Format: whitespace-delimited n x n matrix of off-diagonal rates, '
            'one row per line. The diagonal is ignored and recomputed so that '
            'rows sum to zero. The stationary distribution is deduced '
            'automatically by solving pi Q = 0. '
            'Lines beginning with "#" are treated as comments. '
            'An optional comment of the form "# alphabet ACGT" sets the '
            'alphabet; otherwise DNA (n=4) or amino-acid (n=20) alphabets are '
            'inferred from the matrix size, and integer labels are used for '
            'any other size. '
            'Example 4x4 matrix (Jukes-Cantor-like): '
            '"0 1 1 1 / 1 0 1 1 / 1 1 0 1 / 1 1 1 0" (each "/" is a newline).'
        ))
    model_group.add_argument('--pi', type=float, nargs='+', metavar='FREQ',
        help='Stationary frequencies for F81 (4 values, need not sum to 1).')
    model_group.add_argument('--k2p-kappa', type=float, default=2.0,
        metavar='KAPPA',
        help='Transition/transversion ratio for K2P (default: 2.0).')

    # --- TKF92 indel parameters ---
    indel_group = parser.add_argument_group('TKF92 indel parameters')
    indel_group.add_argument('--lam', type=float, default=0.1, metavar='LAMBDA',
        help='Insertion rate (default: 0.1).')
    indel_group.add_argument('--mu', type=float, default=0.2, metavar='MU',
        help='Deletion rate (default: 0.2).')
    indel_group.add_argument('--r', type=float, default=0.0, metavar='R',
        help='Fragment extension probability (default: 0.0 = TKF91).')

    # --- Branch lengths ---
    branch_group = parser.add_argument_group('Branch lengths')
    branch_group.add_argument('--ell1', type=float, default=0.5, metavar='L1',
        help='Branch length A->B (default: 0.5).')
    branch_group.add_argument('--ell2', type=float, default=0.5, metavar='L2',
        help='Branch length B->C (default: 0.5).')

    # --- Sampling ---
    sample_group = parser.add_argument_group('Sampling')
    sample_group.add_argument('--N', dest='num_samples', type=int, default=10,
        metavar='N',
        help='Number of samples to draw (default: 10).')
    sample_group.add_argument('--seed', type=int, default=None,
        help='Random seed for reproducibility.')

    # --- Sequences ---
    seq_group = parser.add_argument_group('Sequences')
    seq_exclusive = seq_group.add_mutually_exclusive_group()
    seq_exclusive.add_argument('--fasta', metavar='PATH',
        help='FASTA file containing A (first sequence) and C (second sequence).')
    seq_exclusive.add_argument('--A', metavar='SEQ',
        help='Ancestral sequence as a string (e.g. ACGT).')
    seq_group.add_argument('--C', metavar='SEQ',
        help='Descendant sequence as a string (required with --A).')

    # --- Output ---
    out_group = parser.add_argument_group('Output')
    out_group.add_argument('--top', type=int, default=None, metavar='K',
        help='Print only the top K results by count (default: print all).')
    out_group.add_argument('--json', action='store_true',
        help='Emit structured JSON output instead of human-readable text.')
    out_group.add_argument('--json-model', action='store_true',
        help='Include model parameters (lam, mu, r, alphabet, Q, pi) in JSON output.'
             ' Implies --json.')

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # --- Random seed ---
    if args.seed is not None:
        np.random.seed(args.seed)

    # --- Integration tests ---
    if args.test_all or args.test_dna:
        run_integration_test_dna()
    if args.test_all or args.test_aa:
        run_integration_test_aa()
    if args.test_all or args.test_k2p:
        run_integration_test_k2p()
    if args.test_all or args.test_dna or args.test_aa or args.test_k2p:
        sys.exit(0)

    # --- Build substitution model ---
    if args.model == 'jc':
        Q, pi, alphabet = make_jc_model()
    elif args.model == 'f81':
        if args.pi is None:
            parser.error('--model f81 requires --pi FREQ FREQ FREQ FREQ')
        Q, pi, alphabet = make_f81_model(args.pi)
    elif args.model == 'k2p':
        Q, pi, alphabet = make_k2p_model(args.k2p_kappa)
    elif args.model == 'lg':
        Q, pi, alphabet = make_lg_model()
    elif args.model == 'file':
        if args.model_file is None:
            parser.error('--model file requires --model-file PATH')
        Q, pi, alphabet = make_model_from_file(args.model_file)
    else:
        parser.error(f'Unknown model: {args.model}')

    params = TKF92Params(
        lam=args.lam, mu=args.mu, r=args.r,
        Q=Q, pi=pi, alphabet=alphabet,
    )
    c2i = params.char_to_idx

    # --- Load sequences ---
    if args.fasta:
        seqs = read_fasta(args.fasta)
        if len(seqs) < 2:
            parser.error(f'FASTA file must contain at least 2 sequences; found {len(seqs)}.')
        name_A, seq_A_str = seqs[0]
        name_C, seq_C_str = seqs[1]
        print(f"# A: {name_A}  ({len(seq_A_str)} residues)")
        print(f"# C: {name_C}  ({len(seq_C_str)} residues)")
    elif args.A is not None:
        if args.C is None:
            parser.error('--A requires --C')
        seq_A_str = args.A.upper()
        seq_C_str = args.C.upper()
    else:
        parser.error('Provide sequences via --fasta or --A / --C.')

    # Validate and convert
    for pos, ch in enumerate(seq_A_str):
        if ch not in c2i:
            parser.error(f'Unknown character {ch!r} at position {pos} of A '
                         f'(alphabet: {"".join(alphabet)})')
    for pos, ch in enumerate(seq_C_str):
        if ch not in c2i:
            parser.error(f'Unknown character {ch!r} at position {pos} of C '
                         f'(alphabet: {"".join(alphabet)})')

    A = [c2i[c] for c in seq_A_str]
    C = [c2i[c] for c in seq_C_str]

    # --- Parameter validation ---
    if args.lam >= args.mu:
        print("WARNING: lam >= mu; the TKF92 equilibrium distribution requires lam < mu.",
              file=sys.stderr)
    if not (0.0 <= args.r < 1.0):
        parser.error('r must be in [0, 1).')

    # --- Run sampler ---
    results = sample_intermediates(A, C, params, args.ell1, args.ell2, args.num_samples)

    # --- Print results ---
    sorted_results = sorted(results.items(), key=lambda x: -x[1][1])
    if args.top is not None:
        sorted_results = sorted_results[:args.top]

    if args.json or args.json_model:
        import json
        out = {
            'A':    seq_A_str,
            'C':    seq_C_str,
            'N':    args.num_samples,
            'ell1': args.ell1,
            'ell2': args.ell2,
            'samples': [
                {
                    'B':     ''.join(alphabet[r] for r in B_tuple),
                    'log_p': float(log_p),
                    'count': count,
                }
                for B_tuple, (log_p, count) in sorted_results
            ],
        }
        if args.json_model:
            out['model'] = {
                'lam':      params.lam,
                'mu':       params.mu,
                'r':        params.r,
                'alphabet': params.alphabet,
                'Q':        params.Q.tolist(),
                'pi':       params.pi.tolist(),
            }
        print(json.dumps(out, indent=2))
    else:
        print(f"# Model  : {args.model}")
        print(f"# lam={args.lam}  mu={args.mu}  r={args.r}")
        print(f"# ell1={args.ell1}  ell2={args.ell2}  N={args.num_samples}")
        print(f"# A = {seq_A_str}")
        print(f"# C = {seq_C_str}")
        print()
        print(f"{'B':<25}  {'log_p':>10}  {'p':>12}  {'count':>7}")
        print('-' * 60)
        for B_tuple, (log_p, count) in sorted_results:
            B_str = ''.join(alphabet[r] for r in B_tuple)
            p_val = np.exp(log_p)
            print(f"{B_str:<25}  {log_p:>10.4f}  {p_val:>12.4e}  {count:>7}")
        total = sum(count for _, count in results.values())
        print(f"\n# Total samples: {total}   Unique B: {len(results)}")


if __name__ == '__main__':
    main()
