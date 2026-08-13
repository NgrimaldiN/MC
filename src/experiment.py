"""Experiment orchestration for the GARCH control-variate study.

These functions wire together the sampler, control-variate construction, and
estimation routines from :mod:`garch_cv` into reusable building blocks used by
the analysis notebook.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import garch_cv as g


def make_cv_estimates(
    theta_samples: np.ndarray,
    omega_samples: np.ndarray,
    returns: np.ndarray,
    seed: int,
    degree_large: int = 5,
) -> pd.DataFrame:
    """Build MC / ZV-OLS-1 / ZV-OLS-2 / ZV-OLS-q / ZV-Lasso-q estimates.

    Parameters
    ----------
    theta_samples
        Posterior draws in the unconstrained parameterisation, shape
        ``(n_samples, 3)``.
    omega_samples
        Posterior draws in the original GARCH parameters, shape
        ``(n_samples, 3)``.
    returns
        The return series the posterior is conditioned on.
    seed
        Random seed forwarded to the Lasso screening step.
    degree_large
        Maximum polynomial degree for the large dictionary (``q``).
    """
    grad = g.grad_log_posterior_theta_batch(returns, theta_samples)
    H1, labels1, _ = g.zv_controls(theta_samples, grad, degree=1)
    H2, labels2, _ = g.zv_controls(theta_samples, grad, degree=2)
    Hlarge, labels_large, _ = g.zv_controls(theta_samples, grad, degree=degree_large)
    controls = {
        "MC": None,
        "ZV-OLS-1": H1,
        "ZV-OLS-2": H2,
        f"ZV-OLS-{degree_large}": Hlarge,
        f"ZV-Lasso-{degree_large}": Hlarge,
    }

    rows = []
    for j, name in enumerate(g.PARAMETER_NAMES):
        y = omega_samples[:, j]
        for method, H in controls.items():
            if method == "MC":
                est = g.ControlEstimate(
                    float(y.mean()), np.zeros(0), np.array([], dtype=int), np.nan, y.copy()
                )
            elif "Lasso" in method:
                est = g.post_lasso_control_estimate(
                    y, H, seed=seed + 100 * (j + 1), max_train=600
                )
            else:
                est = g.controlled_estimate_ols(y, H)
            rows.append(
                {
                    "parameter": name,
                    "method": method,
                    "estimate": est.estimate,
                    "batch_se": g.batch_means_se(est.controlled_values, batch_size=80),
                    "selected_controls": int(est.selected.size),
                    "condition_number": est.condition_number,
                }
            )
    return pd.DataFrame(rows)


def run_dataset(
    dataset: str,
    returns: np.ndarray,
    init_omega: np.ndarray,
    proposal_scale: np.ndarray,
    n_rep: int,
    n_steps: int,
    burn_in: int,
    thin: int,
    seed_base: int,
    degree_large: int = 5,
) -> tuple[pd.DataFrame, g.MCMCResult]:
    """Run *n_rep* independent RWM chains and collect control-variate estimates.

    Returns
    -------
    (estimates_df, representative)
        ``estimates_df`` has one row per (rep, parameter, method).  The
        ``representative`` is the ``MCMCResult`` from the first repetition.
    """
    init_theta = g.omega_to_theta(init_omega)
    all_rows = []
    representative = None
    started = time.time()
    for rep in range(n_rep):
        seed = seed_base + 1000 * (rep + 1) + (0 if dataset == "simulated" else 500)
        result = g.run_rwm_theta(
            returns,
            init_theta,
            proposal_scale,
            n_steps=n_steps,
            burn_in=burn_in,
            thin=thin,
            seed=seed,
        )
        if representative is None:
            representative = result
        estimates = make_cv_estimates(
            result.theta, result.omega, returns, seed=seed, degree_large=degree_large
        )
        estimates.insert(0, "rep", rep + 1)
        estimates.insert(0, "acceptance_rate", result.acceptance_rate)
        estimates.insert(0, "dataset", dataset)
        all_rows.append(estimates)
    elapsed = time.time() - started
    print(f"{dataset}: {n_rep} chains in {elapsed:.1f}s")
    return pd.concat(all_rows, ignore_index=True), representative
