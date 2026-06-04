"""Sanity tests for the VG FFT pricer and BSM utilities."""
import numpy as np
from vg_fft import (vg_call_price, vg_call_price_direct, vg_put_price,
                    vg_char_fn, vg_omega)
from bsm import bsm_price, implied_vol

S0, T, r, q = 23700.0, 0.08, 0.066, 0.012
sigma, theta, nu = 0.15, -0.12, 0.25

print("=== 1. VG martingale check: E[S_T] should equal S0*e^{(r-q)T} ===")
# char fn at u=-i gives E[S_T]
ES = vg_char_fn(-1j, S0, T, r, q, sigma, theta, nu).real
print(f"  E[S_T] (char fn @ -i) = {ES:.4f}")
print(f"  S0*e^(r-q)T           = {S0*np.exp((r-q)*T):.4f}")
print(f"  relative error        = {abs(ES - S0*np.exp((r-q)*T))/S0:.2e}")

print("\n=== 2a. FFT vs direct quad integration (internal consistency) ===")
for K in [22000, 23000, 23700, 24500, 25500]:
    c_fft = vg_call_price(K, S0, T, r, q, sigma, theta, nu)
    c_dir = vg_call_price_direct(K, S0, T, r, q, sigma, theta, nu)
    print(f"  K={K}:  FFT={c_fft:10.4f}   direct={c_dir:10.4f}   diff={abs(c_fft-c_dir):.2e}")

print("\n=== 2b. DEFINITIVE: nu->0, theta->0  =>  VG must equal Black-Scholes ===")
for K in [22000, 23000, 23700, 24500, 25500]:
    c_fft = vg_call_price(K, S0, T, r, q, sigma, 0.0, 1e-5)
    c_bsm = bsm_price(K, S0, T, r, q, sigma, "C")
    print(f"  K={K}:  VG_FFT={c_fft:10.4f}   BSM={c_bsm:10.4f}   diff={abs(c_fft-c_bsm):.2e}")

print("\n=== 3. Put-call parity:  C - P = S0 e^{-qT} - K e^{-rT} ===")
for K in [22000, 23700, 25500]:
    c = vg_call_price(K, S0, T, r, q, sigma, theta, nu)
    p = vg_put_price(K, S0, T, r, q, sigma, theta, nu)
    lhs = c - p
    rhs = S0*np.exp(-q*T) - K*np.exp(-r*T)
    print(f"  K={K}:  C-P={lhs:12.4f}   parity={rhs:12.4f}   diff={abs(lhs-rhs):.2e}")

print("\n=== 4. BSM implied-vol round trip ===")
for K in [22000, 23700, 25500]:
    price = bsm_price(K, S0, T, r, q, 0.18, "C")
    iv = implied_vol(price, K, S0, T, r, q, "C")
    print(f"  K={K}:  priced@0.18  recovered IV={iv:.6f}")

print("\nAll checks complete.")
