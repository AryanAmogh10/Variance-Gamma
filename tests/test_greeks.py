"""Unit tests for the VG Greeks engine (vg_engine/greeks.py)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vg_engine"))

from greeks import greeks_chain, greeks_surface          # noqa: E402
from bsm import bsm_price                                 # noqa: E402

# A representative NIFTY-like setup (params near multi-day calibrated means)
S0, T = 23700.0, 0.10
R, Q = 0.066, 0.012
SIGMA, THETA, NU = 0.133, -0.150, 0.114
STRIKES = np.array([21000., 22500., 23700., 25000., 26500.])


@pytest.fixture(scope="module")
def call_greeks():
    return greeks_chain(STRIKES, "C", S0, T, R, Q, SIGMA, THETA, NU)


@pytest.fixture(scope="module")
def put_greeks():
    return greeks_chain(STRIKES, "P", S0, T, R, Q, SIGMA, THETA, NU)


# --------------------------------------------------------------------------- #
# Sign conventions
# --------------------------------------------------------------------------- #
class TestSigns:
    def test_call_delta_positive_and_bounded(self, call_greeks):
        d = call_greeks["delta"]
        assert np.all(d > 0) and np.all(d < 1)

    def test_put_delta_negative_and_bounded(self, put_greeks):
        d = put_greeks["delta"]
        assert np.all(d < 0) and np.all(d > -1)

    def test_delta_monotone_decreasing_in_strike(self, call_greeks):
        assert np.all(np.diff(call_greeks["delta"]) < 0)

    def test_vega_positive(self, call_greeks, put_greeks):
        assert np.all(call_greeks["vega"] > 0)
        assert np.all(put_greeks["vega"] > 0)

    def test_theta_negative_for_long_vanillas(self, call_greeks):
        # ATM/OTM long options decay; deep ITM with rates can be ambiguous,
        # so check ATM and OTM strikes only (indices 2..4 are >= ATM).
        assert np.all(call_greeks["theta"][2:] < 0)

    def test_itm_call_delta_above_otm(self, call_greeks):
        assert call_greeks["delta"][0] > 0.7        # deep ITM
        assert call_greeks["delta"][-1] < 0.35      # OTM


# --------------------------------------------------------------------------- #
# Parity identities
# --------------------------------------------------------------------------- #
class TestParity:
    def test_delta_parity_dividend_adjusted(self, call_greeks, put_greeks):
        """delta_call - delta_put = exp(-qT) (reduces to 1 when q=0)."""
        gap = call_greeks["delta"] - put_greeks["delta"]
        assert np.allclose(gap, np.exp(-Q * T), atol=2e-3)

    def test_delta_parity_no_dividend(self):
        c = greeks_chain(STRIKES, "C", S0, T, R, 0.0, SIGMA, THETA, NU)
        p = greeks_chain(STRIKES, "P", S0, T, R, 0.0, SIGMA, THETA, NU)
        assert np.allclose(c["delta"] - p["delta"], 1.0, atol=2e-3)

    def test_vega_parity(self, call_greeks, put_greeks):
        """Calls and puts have identical vega (parity is sigma-independent)."""
        assert np.allclose(call_greeks["vega"], put_greeks["vega"],
                           rtol=1e-3, atol=1e-2)

    def test_volga_parity(self, call_greeks, put_greeks):
        assert np.allclose(call_greeks["volga"], put_greeks["volga"],
                           rtol=5e-3, atol=1.0)

    def test_vanna_parity(self, call_greeks, put_greeks):
        assert np.allclose(call_greeks["vanna"], put_greeks["vanna"],
                           rtol=5e-3, atol=0.05)


# --------------------------------------------------------------------------- #
# Finite-difference consistency
# --------------------------------------------------------------------------- #
class TestFDConsistency:
    def test_delta_matches_manual_fd(self, call_greeks):
        """Engine delta == an independently computed central difference."""
        from vg_fft import vg_call_price
        h = 0.001 * S0
        manual = (np.array(vg_call_price(STRIKES, S0 + h, T, R, Q, SIGMA, THETA, NU))
                  - np.array(vg_call_price(STRIKES, S0 - h, T, R, Q, SIGMA, THETA, NU))
                  ) / (2 * h)
        assert np.allclose(call_greeks["delta"], manual, atol=1e-6)

    def test_bump_refinement_stability(self):
        """Halving the spot bump changes delta by < 0.1% -- h is in the
        convergent region of the central difference."""
        import config
        base = greeks_chain(STRIKES, "C", S0, T, R, Q, SIGMA, THETA, NU)["delta"]
        old = config.BUMPS["spot_rel"]
        try:
            config.BUMPS["spot_rel"] = old / 2
            fine = greeks_chain(STRIKES, "C", S0, T, R, Q, SIGMA, THETA, NU)["delta"]
        finally:
            config.BUMPS["spot_rel"] = old
        assert np.allclose(base, fine, rtol=1e-3)

    def test_vg_delta_near_bsm_in_brownian_limit(self):
        """As nu -> 0, theta -> 0 the VG delta must approach the BSM delta."""
        from scipy.stats import norm
        g = greeks_chain(STRIKES, "C", S0, T, R, Q, 0.15, 0.0, 1e-5)
        d1 = (np.log(S0 / STRIKES) + (R - Q + 0.5 * 0.15 ** 2) * T) / (0.15 * np.sqrt(T))
        bsm_delta = np.exp(-Q * T) * norm.cdf(d1)
        assert np.allclose(g["delta"], bsm_delta, atol=2e-3)

    def test_theta_matches_price_decay(self):
        """theta ~ (V(T - 1day) - V(T)) / (1 day) for ATM call."""
        from vg_fft import vg_call_price
        K = np.array([23700.0])
        g = greeks_chain(K, "C", S0, T, R, Q, SIGMA, THETA, NU)
        dt = 1.0 / 252
        v_now = vg_call_price(K, S0, T, R, Q, SIGMA, THETA, NU)
        v_later = vg_call_price(K, S0, T - dt, R, Q, SIGMA, THETA, NU)
        realized = (v_later - v_now) / dt          # dV/dt over one day
        assert np.allclose(g["theta"], realized, rtol=0.15)


# --------------------------------------------------------------------------- #
# Surface
# --------------------------------------------------------------------------- #
class TestSurface:
    def test_surface_shapes(self):
        ks = np.linspace(21000, 26500, 12)
        ts = np.array([0.05, 0.10, 0.20])
        surf = greeks_surface(ks, ts, "C", S0, r=R, q=Q,
                              sigma=SIGMA, theta=THETA, nu=NU)
        for name in ["price", "delta", "vega", "volga", "vanna", "theta"]:
            assert surf[name].shape == (3, 12), name

    def test_surface_rows_match_chain(self):
        ks = np.linspace(22000, 25000, 7)
        ts = np.array([0.05, 0.15])
        surf = greeks_surface(ks, ts, "C", S0, r=R, q=Q,
                              sigma=SIGMA, theta=THETA, nu=NU)
        row1 = greeks_chain(ks, "C", S0, 0.15, R, Q, SIGMA, THETA, NU)
        assert np.allclose(surf["delta"][1], row1["delta"])
