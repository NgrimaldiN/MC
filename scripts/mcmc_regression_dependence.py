"""Regression control variates on dependent MCMC output.

This script studies the impact of MCMC dependence on the zero-variance
control-variate regression.  It runs several Metropolis-Hastings chains,
computes the usual posterior mean and the zero-variance regression estimator
for each chain, and compares variability across independent runs.

The regression rows are MCMC draws, not iid observations.  We therefore
compare the naive same-chain OLS fit with three simple fixes:

1. fit beta on the first half of the chain and estimate the mean on the second
   half;
2. thin the chain before fitting OLS;
3. average the chain into blocks before fitting OLS.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import garch_cv as g  # noqa: E402


FIG_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


TRUE_OMEGA = np.array([0.2, 0.2, 0.5])
N_OBS = 600
N_CHAINS = 10
N_STEPS = 5000
BURN_IN = 1500
THIN_FOR_OLS = 8
BLOCK_SIZE = 80
SEED_BASE = 20260504


def lag1_acf(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return np.nan
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return np.nan
    return float(np.dot(x[:-1], x[1:]) / denom)


def block_means(values: np.ndarray, block_size: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n_blocks = values.shape[0] // block_size
    if n_blocks < 2:
        raise ValueError("not enough observations for block means")
    trimmed = values[: n_blocks * block_size]
    return trimmed.reshape(n_blocks, block_size, *values.shape[1:]).mean(axis=1)


def fit_beta(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    controls = np.asarray(controls, dtype=float)
    x_centered = controls - controls.mean(axis=0)
    y_centered = y - y.mean()
    beta, *_ = np.linalg.lstsq(x_centered, y_centered, rcond=None)
    return beta


def intercept_estimate(y: np.ndarray, controls: np.ndarray) -> tuple[float, np.ndarray]:
    beta = fit_beta(y, controls)
    controlled = y - controls @ beta
    return float(controlled.mean()), controlled


def split_chain_estimate(y: np.ndarray, controls: np.ndarray) -> tuple[float, np.ndarray]:
    midpoint = y.size // 2
    beta = fit_beta(y[:midpoint], controls[:midpoint])
    controlled_eval = y[midpoint:] - controls[midpoint:] @ beta
    return float(controlled_eval.mean()), controlled_eval


def make_mcmc_regression_estimates(theta: np.ndarray, omega: np.ndarray, returns: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    grad = g.grad_log_posterior_theta_batch(returns, theta)
    controls, _, _ = g.zv_controls(theta, grad, degree=1)
    control_blocks = block_means(controls, BLOCK_SIZE)

    rows = []
    acf_rows = []
    for j, name in enumerate(g.PARAMETER_NAMES):
        y = omega[:, j]
        y_blocks = block_means(y, BLOCK_SIZE)

        method_values = {
            "MC": (float(y.mean()), y),
            "ZV naive OLS": intercept_estimate(y, controls),
            "ZV split chain": split_chain_estimate(y, controls),
            "ZV thinned OLS": intercept_estimate(y[::THIN_FOR_OLS], controls[::THIN_FOR_OLS]),
            "ZV block OLS": intercept_estimate(y_blocks, control_blocks),
        }
        for method, value in method_values.items():
            estimate, controlled_values = value
            rows.append(
                {
                    "parameter": name,
                    "method": method,
                    "estimate": estimate,
                    "batch_means_se": g.batch_means_se(
                        controlled_values,
                        batch_size=max(10, min(80, controlled_values.size // 6)),
                    ),
                    "n_regression_rows": int(controlled_values.size),
                }
            )

        acf_rows.extend(
            [
                {"parameter": name, "series": "raw chain", "lag1_acf": lag1_acf(y)},
                {"parameter": name, "series": f"thinned every {THIN_FOR_OLS}", "lag1_acf": lag1_acf(y[::THIN_FOR_OLS])},
                {"parameter": name, "series": f"block means size {BLOCK_SIZE}", "lag1_acf": lag1_acf(y_blocks)},
            ]
        )

    return pd.DataFrame(rows), pd.DataFrame(acf_rows)


def write_latex_table(df: pd.DataFrame, path: Path, digits: int = 4) -> None:
    table = df.copy()
    for col in table.columns:
        if pd.api.types.is_numeric_dtype(table[col]):
            table[col] = table[col].map(lambda x: f"{x:.2e}" if abs(x) < 1e-3 and x != 0 else f"{x:.{digits}f}")
    path.write_text(table.to_latex(index=False, escape=True), encoding="utf-8")


def main() -> None:
    started = time.time()
    returns, _ = g.simulate_garch(N_OBS, TRUE_OMEGA, seed=SEED_BASE)
    init_theta = g.omega_to_theta(TRUE_OMEGA)

    tuned_diag, diag_rates = g.tune_rwm_scale(
        returns,
        init_theta,
        np.array([0.12, 0.12, 0.12]),
        seed=SEED_BASE + 1,
        n_rounds=4,
        round_steps=500,
    )
    proposal_cov, cov_rates, pilot_acceptance = g.tune_rwm_covariance(
        returns,
        init_theta,
        tuned_diag,
        seed=SEED_BASE + 2,
        pilot_steps=1600,
        pilot_burn_in=400,
        n_rounds=3,
        round_steps=500,
    )

    all_estimates = []
    all_acf = []
    acceptance_rates = []
    representative = None
    for chain_id in range(1, N_CHAINS + 1):
        result = g.run_rwm_theta(
            returns,
            init_theta,
            proposal_cov,
            n_steps=N_STEPS,
            burn_in=BURN_IN,
            thin=1,
            seed=SEED_BASE + 100 * chain_id,
        )
        if representative is None:
            representative = result
        acceptance_rates.append(result.acceptance_rate)
        estimates, acf = make_mcmc_regression_estimates(result.theta, result.omega, returns)
        estimates.insert(0, "chain", chain_id)
        acf.insert(0, "chain", chain_id)
        all_estimates.append(estimates)
        all_acf.append(acf)
        print(f"chain {chain_id:02d}/{N_CHAINS}, acceptance={result.acceptance_rate:.3f}")

    estimates = pd.concat(all_estimates, ignore_index=True)
    acf_table = pd.concat(all_acf, ignore_index=True)
    estimates.to_csv(RESULT_DIR / "mcmc_regression_estimates_by_chain.csv", index=False)
    acf_table.to_csv(RESULT_DIR / "mcmc_regression_acf_by_chain.csv", index=False)

    summary = (
        estimates.groupby(["parameter", "method"], as_index=False)
        .agg(
            mean_estimate=("estimate", "mean"),
            variance_across_chains=("estimate", "var"),
            sd_across_chains=("estimate", "std"),
            mean_batch_means_se=("batch_means_se", "mean"),
            mean_regression_rows=("n_regression_rows", "mean"),
        )
    )
    mc_var = summary[summary["method"] == "MC"][["parameter", "variance_across_chains"]].rename(
        columns={"variance_across_chains": "mc_variance"}
    )
    summary = summary.merge(mc_var, on="parameter", how="left")
    summary["variance_ratio_vs_mc"] = summary["mc_variance"] / summary["variance_across_chains"]
    summary.loc[summary["method"] == "MC", "variance_ratio_vs_mc"] = 1.0
    summary.to_csv(RESULT_DIR / "mcmc_regression_summary.csv", index=False)

    acf_summary = (
        acf_table.groupby(["parameter", "series"], as_index=False)
        .agg(mean_lag1_acf=("lag1_acf", "mean"), sd_lag1_acf=("lag1_acf", "std"))
    )
    acf_summary.to_csv(RESULT_DIR / "mcmc_regression_acf_summary.csv", index=False)

    write_latex_table(summary, RESULT_DIR / "mcmc_regression_summary.tex")
    write_latex_table(acf_summary, RESULT_DIR / "mcmc_regression_acf_summary.tex")

    metadata = {
        "true_omega": TRUE_OMEGA.tolist(),
        "n_obs": N_OBS,
        "n_chains": N_CHAINS,
        "n_steps": N_STEPS,
        "burn_in": BURN_IN,
        "thin_for_ols": THIN_FOR_OLS,
        "block_size": BLOCK_SIZE,
        "diagonal_tuning_acceptance": diag_rates,
        "covariance_pilot_acceptance": pilot_acceptance,
        "covariance_tuning_acceptance": cov_rates,
        "chain_acceptance_mean": float(np.mean(acceptance_rates)),
        "elapsed_seconds": time.time() - started,
    }
    (RESULT_DIR / "mcmc_regression_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    methods = ["MC", "ZV naive OLS", "ZV split chain", "ZV thinned OLS", "ZV block OLS"]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1), constrained_layout=True)
    for j, param in enumerate(g.PARAMETER_NAMES):
        data = [
            estimates[(estimates["parameter"] == param) & (estimates["method"] == method)]["estimate"].to_numpy()
            for method in methods
        ]
        bp = axes[j].boxplot(data, tick_labels=methods, patch_artist=True, showmeans=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.62)
        axes[j].axhline(TRUE_OMEGA[j], color="#222222", linestyle="--", linewidth=1.0, label="true value")
        axes[j].set_title(param)
        axes[j].tick_params(axis="x", rotation=25)
        if j == 0:
            axes[j].legend(frameon=False, fontsize=8)
    fig.suptitle("Repeated-chain estimates under dependent MCMC output", y=1.02)
    fig.savefig(FIG_DIR / "mcmc_regression_boxplots.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "mcmc_regression_boxplots.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), constrained_layout=True)
    for j, param in enumerate(g.PARAMETER_NAMES):
        subset = summary[summary["parameter"] == param].set_index("method").loc[methods]
        axes[j].bar(np.arange(len(methods)), subset["variance_ratio_vs_mc"], color=colors, alpha=0.75)
        axes[j].axhline(1.0, color="#222222", linewidth=0.9)
        axes[j].set_yscale("log")
        axes[j].set_title(param)
        axes[j].set_xticks(np.arange(len(methods)))
        axes[j].set_xticklabels(methods, rotation=25, ha="right")
        axes[j].set_ylabel("variance ratio vs MC")
    fig.savefig(FIG_DIR / "mcmc_regression_variance_ratios.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "mcmc_regression_variance_ratios.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), constrained_layout=True)
    series_order = ["raw chain", f"thinned every {THIN_FOR_OLS}", f"block means size {BLOCK_SIZE}"]
    for j, param in enumerate(g.PARAMETER_NAMES):
        subset = acf_summary[acf_summary["parameter"] == param].set_index("series").loc[series_order]
        axes[j].bar(np.arange(len(series_order)), subset["mean_lag1_acf"], color=["#4C78A8", "#F58518", "#54A24B"], alpha=0.75)
        axes[j].axhline(0.0, color="#222222", linewidth=0.9)
        axes[j].set_title(param)
        axes[j].set_xticks(np.arange(len(series_order)))
        axes[j].set_xticklabels(["raw", "thinned", "blocks"], rotation=0)
        axes[j].set_ylabel("mean lag-1 autocorrelation")
    fig.savefig(FIG_DIR / "mcmc_regression_acf_reduction.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "mcmc_regression_acf_reduction.png", bbox_inches="tight")
    plt.close(fig)

    if representative is not None:
        fig, axes = plt.subplots(3, 1, figsize=(9.5, 6.2), constrained_layout=True)
        x = np.arange(representative.omega.shape[0]) + BURN_IN
        for j, param in enumerate(g.PARAMETER_NAMES):
            y = representative.omega[:, j]
            running = np.cumsum(y) / np.arange(1, y.size + 1)
            axes[j].plot(x, running, color="#F58518", linewidth=1.1)
            axes[j].axhline(TRUE_OMEGA[j], color="#222222", linestyle="--", linewidth=1.0)
            axes[j].set_title(f"running mean for {param}")
            axes[j].set_ylabel("mean")
        axes[-1].set_xlabel("MH iteration after burn-in")
        fig.savefig(FIG_DIR / "mcmc_regression_running_mean.pdf", bbox_inches="tight")
        fig.savefig(FIG_DIR / "mcmc_regression_running_mean.png", bbox_inches="tight")
        plt.close(fig)

    print(summary)
    print(f"Finished in {metadata['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
