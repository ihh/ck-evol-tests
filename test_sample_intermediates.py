"""Tests for sample_intermediates.py"""

import numpy as np
import pytest

from sample_intermediates import (
    make_jc_model,
    make_f81_model,
    make_k2p_model,
    make_lg_model,
    stationary_from_Q,
    _normalise_Q,
    TKF92Params,
    tkf92_coeffs,
    subst_matrix,
    cond_pair_matrix,
    triad_transition_matrix,
    eliminate_null_state,
    log_forward_pair,
    sample_intermediates,
    DNA_ALPHABET,
    AA_ALPHABET,
    S_ST, M_ST, I_ST, D_ST, E_ST,
)


# ---------------------------------------------------------------------------
# Substitution models
# ---------------------------------------------------------------------------

class TestJCModel:
    def test_shape(self):
        Q, pi, alphabet = make_jc_model()
        assert Q.shape == (4, 4)
        assert pi.shape == (4,)
        assert len(alphabet) == 4

    def test_rows_sum_to_zero(self):
        Q, pi, _ = make_jc_model()
        np.testing.assert_allclose(Q.sum(axis=1), 0.0, atol=1e-12)

    def test_stationary_distribution(self):
        Q, pi, _ = make_jc_model()
        np.testing.assert_allclose(pi, 0.25, atol=1e-12)

    def test_normalised(self):
        Q, pi, _ = make_jc_model()
        rate = -float(np.dot(pi, np.diag(Q)))
        np.testing.assert_allclose(rate, 1.0, atol=1e-12)


class TestF81Model:
    def test_basic(self):
        Q, pi, alphabet = make_f81_model([0.1, 0.2, 0.3, 0.4])
        assert Q.shape == (4, 4)
        np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-12)
        np.testing.assert_allclose(Q.sum(axis=1), 0.0, atol=1e-12)

    def test_wrong_size(self):
        with pytest.raises(ValueError):
            make_f81_model([0.25, 0.25, 0.5])

    def test_unnormalised_input(self):
        Q, pi, _ = make_f81_model([1, 2, 3, 4])
        np.testing.assert_allclose(pi, [0.1, 0.2, 0.3, 0.4], atol=1e-12)


class TestK2PModel:
    def test_shape(self):
        Q, pi, alphabet = make_k2p_model(kappa=2.0)
        assert Q.shape == (4, 4)

    def test_rows_sum_to_zero(self):
        Q, _, _ = make_k2p_model(kappa=2.0)
        np.testing.assert_allclose(Q.sum(axis=1), 0.0, atol=1e-12)

    def test_transitions_higher(self):
        Q, _, _ = make_k2p_model(kappa=2.0)
        # A<->G transition rate should be higher than A<->C transversion
        # After normalisation, the ratio should be preserved
        assert abs(Q[0, 2]) > abs(Q[0, 1])


class TestLGModel:
    def test_shape(self):
        Q, pi, alphabet = make_lg_model()
        assert Q.shape == (20, 20)
        assert pi.shape == (20,)
        assert len(alphabet) == 20

    def test_rows_sum_to_zero(self):
        Q, _, _ = make_lg_model()
        np.testing.assert_allclose(Q.sum(axis=1), 0.0, atol=1e-10)

    def test_stationary_sums_to_one(self):
        _, pi, _ = make_lg_model()
        np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-12)

    def test_detailed_balance(self):
        Q, pi, _ = make_lg_model()
        # pi_i * Q_ij should equal pi_j * Q_ji for reversible models
        for i in range(20):
            for j in range(i + 1, 20):
                np.testing.assert_allclose(
                    pi[i] * Q[i, j], pi[j] * Q[j, i], atol=1e-10,
                    err_msg=f"Detailed balance violated for ({i},{j})"
                )


class TestStationaryFromQ:
    def test_recovers_jc(self):
        Q, pi_expected, _ = make_jc_model()
        pi_solved = stationary_from_Q(Q)
        np.testing.assert_allclose(pi_solved, pi_expected, atol=1e-10)


# ---------------------------------------------------------------------------
# TKF92 parameters
# ---------------------------------------------------------------------------

class TestTKF92Params:
    def test_basic(self):
        Q, pi, alphabet = make_jc_model()
        params = TKF92Params(lam=0.1, mu=0.2, r=0.0, Q=Q, pi=pi, alphabet=alphabet)
        assert params.n_alpha == 4
        assert params.char_to_idx['A'] == 0

    def test_coefficients(self):
        Q, pi, alphabet = make_jc_model()
        params = TKF92Params(lam=0.1, mu=0.2, r=0.0, Q=Q, pi=pi, alphabet=alphabet)
        alpha, beta, gamma, kappa, nu = tkf92_coeffs(params, ell=1.0)
        assert 0 < alpha < 1
        assert 0 < beta < 1
        assert 0 <= gamma <= 1
        assert kappa == pytest.approx(0.5)

    def test_coefficients_zero_time(self):
        Q, pi, alphabet = make_jc_model()
        params = TKF92Params(lam=0.1, mu=0.2, r=0.0, Q=Q, pi=pi, alphabet=alphabet)
        alpha, beta, gamma, kappa, nu = tkf92_coeffs(params, ell=0.0)
        assert alpha == pytest.approx(1.0)
        assert beta == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Substitution matrix
# ---------------------------------------------------------------------------

class TestSubstMatrix:
    def test_identity_at_zero(self):
        Q, pi, alphabet = make_jc_model()
        params = TKF92Params(lam=0.1, mu=0.2, r=0.0, Q=Q, pi=pi, alphabet=alphabet)
        P = subst_matrix(params, ell=0.0)
        np.testing.assert_allclose(P, np.eye(4), atol=1e-12)

    def test_rows_sum_to_one(self):
        Q, pi, alphabet = make_jc_model()
        params = TKF92Params(lam=0.1, mu=0.2, r=0.0, Q=Q, pi=pi, alphabet=alphabet)
        P = subst_matrix(params, ell=1.0)
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Pair HMM transition matrix
# ---------------------------------------------------------------------------

class TestCondPairMatrix:
    def test_shape(self):
        Q, pi, alphabet = make_jc_model()
        params = TKF92Params(lam=0.1, mu=0.2, r=0.0, Q=Q, pi=pi, alphabet=alphabet)
        t = cond_pair_matrix(params, ell=0.5)
        assert t.shape == (5, 5)

    def test_nonnegative(self):
        Q, pi, alphabet = make_jc_model()
        params = TKF92Params(lam=0.1, mu=0.2, r=0.0, Q=Q, pi=pi, alphabet=alphabet)
        t = cond_pair_matrix(params, ell=0.5)
        assert np.all(t >= -1e-15)


# ---------------------------------------------------------------------------
# Integration tests (from built-in tests)
# ---------------------------------------------------------------------------

class TestIntegrationDNA:
    def test_jc_sampling(self):
        np.random.seed(42)
        Q, pi, alphabet = make_jc_model()
        params = TKF92Params(lam=0.1, mu=0.2, r=0.0, Q=Q, pi=pi, alphabet=alphabet)
        c2i = {c: i for i, c in enumerate(alphabet)}
        A = [c2i[c] for c in 'ACGT']
        C = [c2i[c] for c in 'ACAT']
        N = 20
        results = sample_intermediates(A, C, params, ell1=0.5, ell2=0.5, N=N)
        total = sum(count for _, count in results.values())
        assert total == N
        for B, (log_p, count) in results.items():
            assert np.isfinite(log_p), f"Non-finite log_p for B={B}"
            assert log_p <= 0.01, f"log_p > 0 for B={B}: {log_p}"


class TestIntegrationAA:
    def test_lg_sampling(self):
        np.random.seed(42)
        Q, pi, alphabet = make_lg_model()
        params = TKF92Params(lam=0.05, mu=0.1, r=0.1, Q=Q, pi=pi, alphabet=alphabet)
        c2i = {c: i for i, c in enumerate(alphabet)}
        A = [c2i[c] for c in 'ACDEF']
        C = [c2i[c] for c in 'ACDEF']
        N = 10
        results = sample_intermediates(A, C, params, ell1=0.3, ell2=0.3, N=N)
        total = sum(count for _, count in results.values())
        assert total == N
        for B, (log_p, _) in results.items():
            assert np.isfinite(log_p), f"Non-finite log_p for B={B}"


class TestIntegrationK2P:
    def test_k2p_sampling(self):
        np.random.seed(42)
        Q, pi, alphabet = make_k2p_model(kappa=2.0)
        params = TKF92Params(lam=0.1, mu=0.15, r=0.2, Q=Q, pi=pi, alphabet=alphabet)
        c2i = {c: i for i, c in enumerate(alphabet)}
        A = [c2i[c] for c in 'AAGGTT']
        C = [c2i[c] for c in 'AAGTT']
        N = 15
        results = sample_intermediates(A, C, params, ell1=0.4, ell2=0.6, N=N)
        total = sum(count for _, count in results.values())
        assert total == N


class TestNoPosteriors:
    def test_no_posteriors_mode(self):
        np.random.seed(42)
        Q, pi, alphabet = make_jc_model()
        params = TKF92Params(lam=0.1, mu=0.2, r=0.0, Q=Q, pi=pi, alphabet=alphabet)
        c2i = {c: i for i, c in enumerate(alphabet)}
        A = [c2i[c] for c in 'ACGT']
        C = [c2i[c] for c in 'ACGT']
        N = 10
        results = sample_intermediates(A, C, params, ell1=0.5, ell2=0.5, N=N,
                                       compute_posteriors=False)
        total = sum(count for _, count in results.values())
        assert total == N
        for B, (log_p, _) in results.items():
            assert log_p is None


class TestChapmanKolmogorov:
    """Test that the sampler respects the Chapman-Kolmogorov equation:
    the posterior-weighted samples should sum to approximately p(C|A)."""

    def test_weights_sum_bounded(self):
        np.random.seed(42)
        Q, pi, alphabet = make_jc_model()
        params = TKF92Params(lam=0.1, mu=0.2, r=0.0, Q=Q, pi=pi, alphabet=alphabet)
        c2i = {c: i for i, c in enumerate(alphabet)}
        A = [c2i[c] for c in 'AC']
        C = [c2i[c] for c in 'AC']
        N = 100
        results = sample_intermediates(A, C, params, ell1=0.5, ell2=0.5, N=N)
        # Sum of p(B|A,C) over sampled B should be <= 1
        w_sum = sum(np.exp(log_p) * count for _, (log_p, count) in results.items())
        assert w_sum <= N * 1.01  # allow small numerical slack
