"""Unit tests for the delta-hedging simulator (vg_engine/hedging.py)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vg_engine"))

from hedging import (simulate_gbm_paths, simulate_vg_paths,        # noqa: E402
                     build_bsm_delta_table, run_hedge)
from vg_fft import vg_omega                                         # noqa: E402
from bsm import bsm_price                                           # noqa: E402

S0, R, Q = 23700.0, 0.066, 0.012
SIGMA, THETA, NU = 0.133, -0.150, 0.114
T, N_STEPS = 30 / 252, 30


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(7)


class TestPathSimulation:
    def test_gbm_martingale(self, rng):
        """discounted GBM forward: E[S_T] ~ S0 * exp((r-q)T)."""
        paths = simulate_gbm_paths(S0, 0.15, T, N_STEPS, 40000, R, Q, rng)
        assert paths.shape == (40000, N_STEPS + 1)
        expected = S0 * np.exp((R - Q) * T)
        assert abs(paths[:, -1].mean() - expected) / expected < 0.005

    def test_vg_martingale(self, rng):
        """VG risk-neutral drift must also make E[S_T] = S0 exp((r-q)T)."""
        paths = simulate_vg_paths(S0, SIGMA, THETA, NU, T, N_STEPS, 60000,
                                  R, Q, rng)
        expected = S0 * np.exp((R - Q) * T)
        assert abs(paths[:, -1].mean() - expected) / expected < 0.01

    def test_vg_has_fatter_tails_than_gbm(self, rng):
        """Excess kurtosis of VG log-returns must exceed GBM's (~0)."""
        from scipy.stats import kurtosis
        gbm = simulate_gbm_paths(S0, 0.15, T, N_STEPS, 20000, R, Q, rng)
        vgp = simulate_vg_paths(S0, SIGMA, THETA, NU, T, N_STEPS, 20000,
                                R, Q, rng)
        k_gbm = kurtosis(np.log(gbm[:, 1] / S0))
        k_vg = kurtosis(np.log(vgp[:, 1] / S0))
        assert k_vg > k_gbm + 0.5

    def test_paths_start_at_s0(self, rng):
        paths = simulate_vg_paths(S0, SIGMA, THETA, NU, T, N_STEPS, 10,
                                  R, Q, rng)
        assert np.allclose(paths[:, 0], S0)

    def test_omega_admissible(self):
        assert np.isfinite(vg_omega(SIGMA, THETA, NU))


class TestHedging:
    def test_bsm_hedge_on_gbm_replicates(self, rng):
        """BSM delta on GBM paths at the same vol must replicate well:
        mean hedge error ~ 0 and error std << premium (the classic result)."""
        vol = 0.15
        K = S0
        premium = bsm_price(K, S0, T, R, Q, vol, "C")
        paths = simulate_gbm_paths(S0, vol, T, N_STEPS, 4000, R, Q, rng)
        table = build_bsm_delta_table(K, S0, T, N_STEPS, R, Q, vol)
        res = run_hedge(paths, table, K, premium, T, R, "test", tc_bps=0.0)
        assert abs(res.hedge_error.mean()) < 0.10 * premium
        assert res.hedge_error.std() < 0.45 * premium

    def test_costs_reduce_pnl(self, rng):
        """Adding transaction costs must lower mean hedge error."""
        vol = 0.15
        K = S0
        premium = bsm_price(K, S0, T, R, Q, vol, "C")
        paths = simulate_gbm_paths(S0, vol, T, N_STEPS, 1500, R, Q, rng)
        table = build_bsm_delta_table(K, S0, T, N_STEPS, R, Q, vol)
        free = run_hedge(paths, table, K, premium, T, R, "free", tc_bps=0.0)
        paid = run_hedge(paths, table, K, premium, T, R, "paid", tc_bps=2.0)
        assert paid.hedge_error.mean() < free.hedge_error.mean()
        assert np.all(paid.costs > 0)

    def test_delta_paths_bounded(self, rng):
        vol = 0.15
        K = S0
        premium = bsm_price(K, S0, T, R, Q, vol, "C")
        paths = simulate_gbm_paths(S0, vol, T, N_STEPS, 200, R, Q, rng)
        table = build_bsm_delta_table(K, S0, T, N_STEPS, R, Q, vol)
        res = run_hedge(paths, table, K, premium, T, R, "test")
        assert np.all(res.delta_paths >= 0) and np.all(res.delta_paths <= 1)

    def test_result_shapes(self, rng):
        vol = 0.15
        K = S0
        premium = bsm_price(K, S0, T, R, Q, vol, "C")
        paths = simulate_gbm_paths(S0, vol, T, N_STEPS, 50, R, Q, rng)
        table = build_bsm_delta_table(K, S0, T, N_STEPS, R, Q, vol)
        res = run_hedge(paths, table, K, premium, T, R, "test")
        assert res.hedge_error.shape == (50,)
        assert res.pnl_paths.shape == (50, N_STEPS + 1)
        assert res.delta_paths.shape == (50, N_STEPS)
        assert np.isfinite(res.mae) and np.isfinite(res.sharpe)
