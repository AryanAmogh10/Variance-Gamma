"""
run_analysis.py
---------------
Phase-3 driver: generate all Greeks-surface and hedging figures into /figures
and print the summary statistics used in research_memo.md.

    python run_analysis.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from greeks import plot_greeks_surface
from hedging import run_experiment, summarize, HedgeResult
from config import HEDGE, MARKET, VG_PARAMS_DEFAULT

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "figures")
S0 = 23700.0


def make_greeks_figures() -> None:
    """Greeks surface heatmaps for calls and puts."""
    strikes = np.linspace(0.85 * S0, 1.15 * S0, 41)
    tenors = np.linspace(0.02, 0.5, 25)
    for typ, name in [("C", "call"), ("P", "put")]:
        out = os.path.join(FIG_DIR, f"greeks_surface_{name}.png")
        plot_greeks_surface(strikes, tenors, typ, S0, out_path=out)
        print(f"  saved {out}")


def make_hedging_figures(results: dict[str, HedgeResult]) -> None:
    """Hedging-error distributions, P&L bands, delta convergence."""
    # --- 1. hedging-error distributions: VG vs BSM on each path engine ---
    for engine, pair in [("gbm", ("gbm_vg", "gbm_bsm")),
                         ("vg", ("vg_vg", "vg_bsm"))]:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for key, color in zip(pair, ["#1f77b4", "#d62728"]):
            res = results[key]
            ax.hist(res.hedge_error, bins=60, alpha=0.55, color=color,
                    label=f"{res.label}  (MAE={res.mae:.1f})", density=True)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("terminal hedging error (index points)")
        ax.set_ylabel("density")
        ax.set_title(f"Hedging-error distribution on "
                     f"{'GBM' if engine == 'gbm' else 'VG'} paths "
                     f"(short ATM call, daily rebalance, 2 bps costs)")
        ax.legend()
        fig.tight_layout()
        out = os.path.join(FIG_DIR, f"hedge_error_dist_{engine}_paths.png")
        fig.savefig(out, dpi=130); plt.close(fig)
        print(f"  saved {out}")

    # --- 2. P&L paths with confidence bands ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, key in zip(axes, ["vg_vg", "vg_bsm"]):
        res = results[key]
        t = np.arange(res.pnl_paths.shape[1])
        med = np.median(res.pnl_paths, axis=0)
        lo, hi = (np.percentile(res.pnl_paths, p, axis=0) for p in (5, 95))
        ax.fill_between(t, lo, hi, alpha=0.25, label="5-95% band")
        ax.plot(t, med, lw=2, label="median")
        for row in res.pnl_paths[:25]:
            ax.plot(t, row, lw=0.4, alpha=0.35, color="grey")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(res.label)
        ax.set_xlabel("trading day")
        ax.legend()
    axes[0].set_ylabel("cumulative hedged P&L (index points)")
    fig.suptitle("Hedged P&L paths on VG underlying (25 sample paths shown)")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "pnl_paths_confidence_bands.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  saved {out}")

    # --- 3. delta convergence / divergence over time ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    dvg = results["vg_vg"].delta_paths
    dbs = results["vg_bsm"].delta_paths
    t = np.arange(dvg.shape[1])
    gap = dvg - dbs
    med = np.median(gap, axis=0)
    lo, hi = (np.percentile(gap, p, axis=0) for p in (10, 90))
    ax.fill_between(t, lo, hi, alpha=0.3, label="10-90% band")
    ax.plot(t, med, lw=2, color="#1f77b4", label="median VG - BSM delta")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("trading day")
    ax.set_ylabel("delta difference (VG - BSM)")
    ax.set_title("Hedge-ratio gap over the option life (same VG paths)")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "delta_gap_over_time.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  saved {out}")

    # --- 4. mean |delta| trajectory (convergence of the hedge) ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for key, color in [("vg_vg", "#1f77b4"), ("vg_bsm", "#d62728")]:
        res = results[key]
        ax.plot(np.mean(res.delta_paths, axis=0), color=color, lw=2,
                label=f"mean delta - {res.label}")
    ax.set_xlabel("trading day")
    ax.set_ylabel("mean hedge delta")
    ax.set_title("Average hedge ratio over time (ATM short call, VG paths)")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "delta_convergence.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  saved {out}")


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=== Greeks surfaces ===")
    make_greeks_figures()

    print("\n=== Hedging experiment ===")
    results = run_experiment(S0=S0)
    print("\n" + summarize(results))

    print("\n=== Hedging figures ===")
    make_hedging_figures(results)

    # dump stats for the memo
    stats = {key: {"label": r.label, "mae": r.mae, "rmse": r.rmse,
                   "mean": float(r.hedge_error.mean()),
                   "std": float(r.hedge_error.std()),
                   "sharpe": r.sharpe,
                   "avg_cost": float(r.costs.mean()),
                   "premium": r.premium}
             for key, r in results.items()}
    stats["config"] = {**HEDGE, **MARKET, **VG_PARAMS_DEFAULT, "S0": S0}
    out = os.path.join(FIG_DIR, "hedging_stats.json")
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats dumped to {out}")


if __name__ == "__main__":
    main()
