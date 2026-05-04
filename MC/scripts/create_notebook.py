"""Create the project notebook."""

from __future__ import annotations

import pathlib
import textwrap

import nbformat as nbf


ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "control_variates_garch.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }

    cells = [
        md(
            r"""
            # Control variates for a GARCH model

            This notebook implements the project requested in the TD statement:

            1. a random-walk Metropolis sampler targeting a Bayesian GARCH(1,1) posterior;
            2. zero-variance control variates of Mira, Solgi and Imparato (2013);
            3. a larger polynomial control set and a Lasso screening step inspired by Leluc, Portier and Segers (2021);
            4. repeated-run boxplots and empirical standard deviations to assess numerical error.

            The real-data example uses EUR/USD reference rates downloaded from Frankfurter and stored locally in `data/`.
            """
        ),
        md(
            r"""
            ## Sources used

            - Mira, A., Solgi, R. and Imparato, D. (2013), *Zero Variance Markov Chain Monte Carlo for Bayesian Estimators*, Statistics and Computing 23, 653--662. DOI: `10.1007/s11222-012-9344-6`.
            - Leluc, R., Portier, F. and Segers, J. (2021), *Control variate selection for Monte Carlo integration*, Statistics and Computing 31, article 50. DOI: `10.1007/s11222-021-10011-z`.
            - Frankfurter exchange-rate API documentation, used for the EUR/USD daily rates.
            """
        ),
        md(
            r"""
            ## Model and posterior

            We use the Gaussian GARCH(1,1) recursion

            \[
            r_t\mid \mathcal F_{t-1}\sim \mathcal N(0,h_t),\qquad
            h_t=\omega_1+\omega_2 r_{t-1}^2+\omega_3 h_{t-1}.
            \]

            The posterior is proportional to the Gaussian quasi-likelihood times independent centered normal priors on
            \((\omega_1,\omega_2,\omega_3)\), restricted to
            \(\omega_1>0,\omega_2>0,\omega_3>0,\omega_2+\omega_3<0.995\).

            To remove boundary issues in the zero-variance formula and to avoid rejected negative proposals, the chain is run on an unconstrained vector \(\theta\), transformed to \(\omega\).  The target density in \(\theta\) includes the Jacobian of this transformation.  Posterior summaries are still reported in the original GARCH parameters.
            """
        ),
        code(
            r"""
            import json
            import pathlib
            import sys
            import time
            import warnings

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd

            ROOT = pathlib.Path.cwd()
            SRC = ROOT / "src"
            sys.path.insert(0, str(SRC))

            import garch_cv as g

            warnings.filterwarnings("ignore", category=UserWarning)
            np.set_printoptions(precision=4, suppress=True)

            FIG_DIR = ROOT / "figures"
            RESULT_DIR = ROOT / "results"
            FIG_DIR.mkdir(exist_ok=True)
            RESULT_DIR.mkdir(exist_ok=True)

            plt.rcParams.update({
                "figure.dpi": 130,
                "savefig.dpi": 180,
                "axes.grid": True,
                "grid.alpha": 0.25,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "font.size": 10,
            })
            """
        ),
        code(
            r"""
            # Reproducible experiment settings.
            TRUE_OMEGA = np.array([0.04, 0.08, 0.88])
            N_SIM = 600
            N_REP = 10
            N_STEPS = 8000
            BURN_IN = 2000
            THIN = 3
            LARGE_DEGREE = 5
            SEED_BASE = 20260504
            USE_CACHED_RESULTS = True
            CACHE_SIGNATURE = {
                "n_rep": N_REP,
                "n_steps": N_STEPS,
                "burn_in": BURN_IN,
                "thin": THIN,
                "large_degree": LARGE_DEGREE,
                "seed_base": SEED_BASE,
                "n_sim": N_SIM,
                "proposal": "pilot_full_covariance_v1",
            }

            simulated_returns, simulated_h = g.simulate_garch(N_SIM, TRUE_OMEGA, seed=SEED_BASE)

            fx = pd.read_csv(ROOT / "data" / "eurusd_frankfurter_2023_2024.csv", parse_dates=["date"])
            real_returns = g.prepare_returns_from_prices(fx["eur_usd"].to_numpy())
            real_dates = fx["date"].iloc[1:].to_numpy()

            print(f"Simulated sample: {len(simulated_returns)} returns")
            print(f"EUR/USD sample: {len(real_returns)} daily log-returns from {fx['date'].min().date()} to {fx['date'].max().date()}")
            print(f"EUR/USD return sd, in percent: {real_returns.std(ddof=1):.3f}")
            """
        ),
        code(
            r"""
            fig, ax = plt.subplots(2, 2, figsize=(10.5, 5.8), constrained_layout=True)
            ax[0, 0].plot(simulated_returns, lw=0.9, color="#355C7D")
            ax[0, 0].set_title("Simulated GARCH returns")
            ax[0, 0].set_ylabel("percent")
            ax[0, 1].plot(simulated_h, lw=0.9, color="#C06C84")
            ax[0, 1].set_title("Simulated conditional variance")
            ax[1, 0].plot(fx["date"], fx["eur_usd"], lw=0.9, color="#355C7D")
            ax[1, 0].set_title("EUR/USD reference rate")
            ax[1, 1].plot(real_dates, real_returns, lw=0.9, color="#C06C84")
            ax[1, 1].set_title("EUR/USD daily log-returns")
            ax[1, 1].set_ylabel("percent")
            for axis in ax.ravel():
                axis.tick_params(axis="x", labelrotation=25)
            fig.savefig(FIG_DIR / "data_overview.pdf", bbox_inches="tight")
            fig.savefig(FIG_DIR / "data_overview.png", bbox_inches="tight")
            plt.show()
            """
        ),
        md(
            r"""
            ## Additional data plots

            The next plots are useful for the oral presentation because they show why a conditional-variance model is reasonable.  The histograms summarize the marginal returns; the autocorrelation panels compare the raw returns with squared returns.  In a GARCH-type series, raw returns often have weak autocorrelation, while squared returns retain visible dependence.
            """
        ),
        code(
            r"""
            def autocorrelation(x, max_lag=35):
                x = np.asarray(x, dtype=float)
                x = x - x.mean()
                denom = np.dot(x, x)
                if denom <= 0:
                    return np.zeros(max_lag + 1)
                return np.array([1.0] + [np.dot(x[:-lag], x[lag:]) / denom for lag in range(1, max_lag + 1)])

            def plot_data_diagnostics():
                datasets = {
                    "simulated": simulated_returns,
                    "eurusd": real_returns,
                }
                fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.2), constrained_layout=True)
                for row, (dataset, values) in enumerate(datasets.items()):
                    axes[row, 0].hist(values, bins=35, density=True, color="#4C78A8", alpha=0.75)
                    axes[row, 0].axvline(values.mean(), color="#222222", lw=1.0, ls="--")
                    axes[row, 0].set_title(f"{dataset}: return histogram")
                    axes[row, 0].set_xlabel("percent log-return")

                    acf_returns = autocorrelation(values, max_lag=35)
                    acf_squared = autocorrelation(values**2, max_lag=35)
                    lags = np.arange(acf_returns.size)
                    axes[row, 1].bar(lags, acf_returns, color="#F58518", alpha=0.75)
                    axes[row, 1].axhline(0.0, color="#222222", lw=0.8)
                    axes[row, 1].set_title(f"{dataset}: ACF of returns")
                    axes[row, 1].set_xlabel("lag")

                    axes[row, 2].bar(lags, acf_squared, color="#54A24B", alpha=0.75)
                    axes[row, 2].axhline(0.0, color="#222222", lw=0.8)
                    axes[row, 2].set_title(f"{dataset}: ACF of squared returns")
                    axes[row, 2].set_xlabel("lag")
                fig.savefig(FIG_DIR / "data_diagnostics.pdf", bbox_inches="tight")
                fig.savefig(FIG_DIR / "data_diagnostics.png", bbox_inches="tight")
                plt.show()

            plot_data_diagnostics()
            """
        ),
        md(
            r"""
            ## Random-walk Metropolis sampler

            The proposal is \(\theta'=\theta+\epsilon\).  We first run a diagonal random-walk pilot, then use the empirical covariance of that pilot to build a full-covariance Gaussian proposal.  This follows the TD guidance that a random-walk proposal should be scaled to the shape of the target, and it matters here because \(\omega_1\) and \(\omega_3\) are strongly correlated.  The repeated experiment below also uses longer chains than the first draft: \(8000\) iterations, burn-in \(2000\), and thinning by \(3\).  The table below checks the analytic gradient used by the control variates against a central finite difference.
            """
        ),
        code(
            r"""
            def initial_omega_for_returns(returns, alpha=0.08, beta=0.88):
                unconditional_var = np.var(returns, ddof=1)
                omega_1 = max(unconditional_var * (1.0 - alpha - beta), 1e-4)
                return np.array([omega_1, alpha, beta])

            init = {
                "simulated": TRUE_OMEGA,
                "eurusd": initial_omega_for_returns(real_returns),
            }
            returns_by_dataset = {
                "simulated": simulated_returns,
                "eurusd": real_returns,
            }
            proposal_start = {
                "simulated": np.array([0.16, 0.10, 0.10]),
                "eurusd": np.array([0.16, 0.10, 0.10]),
            }

            tuning_rows = []
            tuned_scales = {}
            for dataset, returns in returns_by_dataset.items():
                init_theta = g.omega_to_theta(init[dataset])
                diagonal_scale, diagonal_rates = g.tune_rwm_scale(
                    returns,
                    init_theta,
                    proposal_start[dataset],
                    seed=SEED_BASE + len(dataset),
                    n_rounds=5,
                    round_steps=650,
                )
                proposal_cov, cov_rates, pilot_acceptance = g.tune_rwm_covariance(
                    returns,
                    init_theta,
                    diagonal_scale,
                    seed=SEED_BASE + 97 + len(dataset),
                    pilot_steps=2500,
                    pilot_burn_in=500,
                    n_rounds=4,
                    round_steps=700,
                )
                tuned_scales[dataset] = proposal_cov
                grad_a, grad_fd, grad_diff = g.check_theta_gradient(returns, init_theta)
                tuning_rows.append({
                    "dataset": dataset,
                    "initial_omega": np.array2string(init[dataset], precision=4),
                    "diagonal_scale": np.array2string(diagonal_scale, precision=4),
                    "proposal_cov_diag": np.array2string(np.diag(proposal_cov), precision=4),
                    "diag_pilot_acceptance": ", ".join(f"{x:.2f}" for x in diagonal_rates),
                    "cov_pilot_acceptance": f"{pilot_acceptance:.2f}; " + ", ".join(f"{x:.2f}" for x in cov_rates),
                    "max_gradient_error": grad_diff,
                })

            tuning_table = pd.DataFrame(tuning_rows)
            tuning_table
            """
        ),
        md(
            r"""
            ## Gradient of the GARCH likelihood

            The zero-variance controls require \(\nabla \log \pi\).  For the likelihood part, write
            \[
            \ell(\omega)=-\frac12\sum_{t=1}^T\left\{\log h_t+\frac{r_t^2}{h_t}\right\},
            \qquad
            h_t=\omega_1+\omega_2 r_{t-1}^2+\omega_3 h_{t-1}.
            \]
            Let \(D_{t,k}=\partial h_t/\partial\omega_k\).  With fixed initial variance \(h_1\), \(D_{1,k}=0\), and for \(t\ge2\)
            \[
            D_{t,1}=1+\omega_3D_{t-1,1},\qquad
            D_{t,2}=r_{t-1}^2+\omega_3D_{t-1,2},\qquad
            D_{t,3}=h_{t-1}+\omega_3D_{t-1,3}.
            \]
            Therefore
            \[
            \frac{\partial\ell}{\partial\omega_k}
            =\frac12\sum_{t=1}^T
            \left(\frac{r_t^2}{h_t^2}-\frac1{h_t}\right)D_{t,k}.
            \]
            The prior contributes \(-\omega_k/\sigma_k^2\).  Since the sampler runs in \(\theta\), the code finally applies the chain rule
            \[
            \nabla_\theta \log\pi_\theta(\theta)
            =
            J_{\omega}(\theta)^\top\nabla_\omega \log\pi_\omega(\omega(\theta))
            +\nabla_\theta \log \left|\det J_{\omega}(\theta)\right|.
            \]
            """
        ),
        md(
            r"""
            ## Zero-variance controls

            Mira, Solgi and Imparato use

            \[
            \tilde f(x)= f(x)-\frac12\Delta P(x)+\nabla P(x)^\top z(x),
            \qquad
            z(x)=-\frac12 \nabla\log\pi(x).
            \]

            For a first-order polynomial \(P(x)=a^\top x\), this reduces to \(f(x)+a^\top z(x)\).
            The coefficients can be estimated by the regression of \(f\) on the control vector.  In the notebook code, `MC` is the raw posterior mean, `ZV-OLS-1` is the first-order version, `ZV-OLS-2` uses all controls up to degree 2, `ZV-OLS-5` uses all controls up to degree 5, and `ZV-Lasso-5` first screens the degree-5 dictionary by Lasso and then refits OLS on the selected controls.
            """
        ),
        code(
            r"""
            def latex_float(x, digits=4):
                if pd.isna(x):
                    return ""
                if abs(x) >= 1000 or (abs(x) < 1e-3 and x != 0):
                    return f"{x:.2e}"
                return f"{x:.{digits}f}"

            def make_cv_estimates(theta_samples, omega_samples, returns, seed, degree_large=LARGE_DEGREE):
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
                            est = g.ControlEstimate(float(y.mean()), np.zeros(0), np.array([], dtype=int), np.nan, y.copy())
                        elif "Lasso" in method:
                            est = g.post_lasso_control_estimate(y, H, seed=seed + 100 * (j + 1), max_train=600)
                        else:
                            est = g.controlled_estimate_ols(y, H)
                        rows.append({
                            "parameter": name,
                            "method": method,
                            "estimate": est.estimate,
                            "batch_se": g.batch_means_se(est.controlled_values, batch_size=80),
                            "selected_controls": int(est.selected.size),
                            "condition_number": est.condition_number,
                        })
                return pd.DataFrame(rows)

            def run_dataset(dataset, returns, init_omega, proposal_scale):
                init_theta = g.omega_to_theta(init_omega)
                all_rows = []
                representative = None
                started = time.time()
                for rep in range(N_REP):
                    seed = SEED_BASE + 1000 * (rep + 1) + (0 if dataset == "simulated" else 500)
                    result = g.run_rwm_theta(
                        returns,
                        init_theta,
                        proposal_scale,
                        n_steps=N_STEPS,
                        burn_in=BURN_IN,
                        thin=THIN,
                        seed=seed,
                    )
                    if representative is None:
                        representative = result
                    estimates = make_cv_estimates(result.theta, result.omega, returns, seed=seed)
                    estimates.insert(0, "rep", rep + 1)
                    estimates.insert(0, "acceptance_rate", result.acceptance_rate)
                    estimates.insert(0, "dataset", dataset)
                    all_rows.append(estimates)
                elapsed = time.time() - started
                print(f"{dataset}: {N_REP} chains in {elapsed:.1f}s")
                return pd.concat(all_rows, ignore_index=True), representative

            cache_path = RESULT_DIR / "repeated_estimates.csv"
            cache_meta_path = RESULT_DIR / "repeated_estimates_meta.json"
            representatives = {}
            cache_is_current = False
            if cache_path.exists() and cache_meta_path.exists():
                cache_is_current = json.loads(cache_meta_path.read_text(encoding="utf-8")) == CACHE_SIGNATURE

            if USE_CACHED_RESULTS and cache_is_current:
                estimates = pd.read_csv(cache_path)
                print(f"Loaded cached repeated estimates from {cache_path}")
                for dataset, returns in returns_by_dataset.items():
                    representatives[dataset] = g.run_rwm_theta(
                        returns,
                        g.omega_to_theta(init[dataset]),
                        tuned_scales[dataset],
                        n_steps=N_STEPS,
                        burn_in=BURN_IN,
                        thin=THIN,
                        seed=SEED_BASE + (11 if dataset == "simulated" else 22),
                    )
            else:
                results = []
                for dataset, returns in returns_by_dataset.items():
                    df, representative = run_dataset(dataset, returns, init[dataset], tuned_scales[dataset])
                    results.append(df)
                    representatives[dataset] = representative
                estimates = pd.concat(results, ignore_index=True)
                estimates.to_csv(cache_path, index=False)
                cache_meta_path.write_text(json.dumps(CACHE_SIGNATURE, indent=2), encoding="utf-8")
            estimates.head()
            """
        ),
        code(
            r"""
            # Save raw repeated-run estimates for auditability.
            estimates.to_csv(RESULT_DIR / "repeated_estimates.csv", index=False)

            summary = (
                estimates
                .groupby(["dataset", "parameter", "method"], as_index=False)
                .agg(
                    mean_estimate=("estimate", "mean"),
                    numerical_sd=("estimate", "std"),
                    mean_batch_se=("batch_se", "mean"),
                    mean_selected=("selected_controls", "mean"),
                    median_condition=("condition_number", "median"),
                    mean_acceptance=("acceptance_rate", "mean"),
                )
            )
            mc_sd = summary[summary["method"] == "MC"][["dataset", "parameter", "numerical_sd"]].rename(columns={"numerical_sd": "mc_sd"})
            summary = summary.merge(mc_sd, on=["dataset", "parameter"], how="left")
            summary["variance_reduction"] = (summary["mc_sd"] / summary["numerical_sd"]) ** 2
            summary.loc[summary["method"] == "MC", "variance_reduction"] = 1.0

            # Add reference columns: data-generating value for simulated data, pooled raw MC for EUR/USD.
            reference = {}
            for j, param in enumerate(g.PARAMETER_NAMES):
                reference[("simulated", param)] = TRUE_OMEGA[j]
                reference[("eurusd", param)] = estimates.query("dataset == 'eurusd' and method == 'MC' and parameter == @param")["estimate"].mean()
            summary["reference"] = [reference[(row.dataset, row.parameter)] for row in summary.itertuples()]
            summary["rmse_to_reference"] = np.nan
            for idx, row in summary.iterrows():
                vals = estimates[
                    (estimates["dataset"] == row["dataset"]) &
                    (estimates["parameter"] == row["parameter"]) &
                    (estimates["method"] == row["method"])
                ]["estimate"].to_numpy()
                summary.loc[idx, "rmse_to_reference"] = np.sqrt(np.mean((vals - row["reference"]) ** 2))

            summary.to_csv(RESULT_DIR / "summary.csv", index=False)
            summary
            """
        ),
        code(
            r"""
            def write_latex_table(df, path, caption=None):
                table = df.copy()
                rename = {
                    "dataset": "data",
                    "parameter": "param.",
                    "method": "method",
                    "mean_estimate": "mean",
                    "numerical_sd": "sd across runs",
                    "rmse_to_reference": "rmse/ref.",
                    "variance_reduction": "var. red.",
                    "mean_selected": "avg. selected",
                }
                cols = list(rename)
                table = table[cols].rename(columns=rename)
                for col in ["mean", "sd across runs", "rmse/ref.", "var. red.", "avg. selected"]:
                    table[col] = table[col].map(lambda x: latex_float(x, 4))
                latex = table.to_latex(index=False, escape=True)
                pathlib.Path(path).write_text(latex, encoding="utf-8")

            sim_table = summary[summary["dataset"] == "simulated"]
            real_table = summary[summary["dataset"] == "eurusd"]
            write_latex_table(sim_table, RESULT_DIR / "summary_simulated.tex")
            write_latex_table(real_table, RESULT_DIR / "summary_eurusd.tex")

            controls_growth = pd.DataFrame({
                "degree": list(range(1, 8)),
                "number_of_controls": [g.count_polynomial_controls(3, d) for d in range(1, 8)],
            })
            controls_growth.to_csv(RESULT_DIR / "controls_growth.csv", index=False)

            def write_compact_latex_table(df, path):
                table = df.copy()
                for col in table.columns:
                    if pd.api.types.is_numeric_dtype(table[col]):
                        table[col] = table[col].map(lambda x: latex_float(x, 4))
                pathlib.Path(path).write_text(table.to_latex(index=False, escape=True), encoding="utf-8")

            variance_summary = summary.copy()
            variance_summary["empirical_variance"] = variance_summary["numerical_sd"] ** 2

            first_order_variance = variance_summary[
                variance_summary["method"].isin(["MC", "ZV-OLS-1"])
            ][["dataset", "parameter", "method", "empirical_variance", "variance_reduction"]]
            first_order_variance.to_csv(RESULT_DIR / "first_order_variance.csv", index=False)
            write_compact_latex_table(first_order_variance, RESULT_DIR / "first_order_variance.tex")

            step3_variance = variance_summary[
                variance_summary["method"].isin(["ZV-OLS-1", f"ZV-OLS-{LARGE_DEGREE}", f"ZV-Lasso-{LARGE_DEGREE}"])
            ][["dataset", "parameter", "method", "empirical_variance", "variance_reduction", "mean_selected", "median_condition"]]
            step3_variance.to_csv(RESULT_DIR / "step3_variance.csv", index=False)
            write_compact_latex_table(step3_variance, RESULT_DIR / "step3_variance.tex")

            controls_growth
            """
        ),
        code(
            r"""
            method_order = ["MC", "ZV-OLS-1", "ZV-OLS-2", f"ZV-OLS-{LARGE_DEGREE}", f"ZV-Lasso-{LARGE_DEGREE}"]
            colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]

            def plot_boxplots(dataset, path_base):
                fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
                for j, param in enumerate(g.PARAMETER_NAMES):
                    ax = axes[j]
                    data = [
                        estimates[
                            (estimates["dataset"] == dataset) &
                            (estimates["parameter"] == param) &
                            (estimates["method"] == method)
                        ]["estimate"].to_numpy()
                        for method in method_order
                    ]
                    bp = ax.boxplot(data, labels=method_order, patch_artist=True, showmeans=True)
                    for patch, color in zip(bp["boxes"], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.55)
                    ref = reference[(dataset, param)]
                    ax.axhline(ref, color="#222222", lw=1.0, ls="--", label="reference")
                    ax.set_title(param)
                    ax.tick_params(axis="x", rotation=25)
                fig.suptitle(f"Repeated-run posterior mean estimates: {dataset}", y=1.02)
                fig.savefig(FIG_DIR / f"{path_base}.pdf", bbox_inches="tight")
                fig.savefig(FIG_DIR / f"{path_base}.png", bbox_inches="tight")
                plt.show()

            plot_boxplots("simulated", "boxplots_simulated")
            plot_boxplots("eurusd", "boxplots_eurusd")
            """
        ),
        code(
            r"""
            fig, axes = plt.subplots(1, 2, figsize=(11.2, 4), constrained_layout=True)
            for ax, dataset in zip(axes, ["simulated", "eurusd"]):
                subset = summary[summary["dataset"] == dataset].copy()
                for j, param in enumerate(g.PARAMETER_NAMES):
                    vals = [
                        subset[
                            (subset["parameter"] == param) &
                            (subset["method"] == method)
                        ]["variance_reduction"].iloc[0]
                        for method in method_order
                    ]
                    x = np.arange(len(method_order)) + 0.24 * (j - 1)
                    ax.bar(x, vals, width=0.22, label=param)
                ax.axhline(1.0, color="#222222", lw=0.8)
                ax.set_yscale("log")
                ax.set_xticks(np.arange(len(method_order)))
                ax.set_xticklabels(method_order, rotation=25, ha="right")
                ax.set_title(dataset)
                ax.set_ylabel("variance reduction vs MC")
                ax.legend(frameon=False, fontsize=8)
            fig.savefig(FIG_DIR / "variance_reduction.pdf", bbox_inches="tight")
            fig.savefig(FIG_DIR / "variance_reduction.png", bbox_inches="tight")
            plt.show()
            """
        ),
        code(
            r"""
            def plot_variance_subset(methods, filename, title):
                fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), constrained_layout=True)
                for ax, dataset in zip(axes, ["simulated", "eurusd"]):
                    subset = summary[(summary["dataset"] == dataset) & (summary["method"].isin(methods))].copy()
                    for j, param in enumerate(g.PARAMETER_NAMES):
                        vals = [
                            subset[
                                (subset["parameter"] == param) &
                                (subset["method"] == method)
                            ]["numerical_sd"].iloc[0] ** 2
                            for method in methods
                        ]
                        x = np.arange(len(methods)) + 0.24 * (j - 1)
                        ax.bar(x, vals, width=0.22, label=param)
                    ax.set_yscale("log")
                    ax.set_xticks(np.arange(len(methods)))
                    ax.set_xticklabels(methods, rotation=25, ha="right")
                    ax.set_title(dataset)
                    ax.set_ylabel("empirical variance across runs")
                    ax.legend(frameon=False, fontsize=8)
                fig.suptitle(title, y=1.03)
                fig.savefig(FIG_DIR / f"{filename}.pdf", bbox_inches="tight")
                fig.savefig(FIG_DIR / f"{filename}.png", bbox_inches="tight")
                plt.show()

            plot_variance_subset(["MC", "ZV-OLS-1"], "first_order_variance_only", "Variance comparison: first-order controls only")
            plot_variance_subset(["ZV-OLS-1", f"ZV-OLS-{LARGE_DEGREE}", f"ZV-Lasso-{LARGE_DEGREE}"], "step3_variance_comparison", "Step 3 variance comparison: large dictionary vs Lasso")
            """
        ),
        code(
            r"""
            fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.7), constrained_layout=True)
            ax[0].plot(controls_growth["degree"], controls_growth["number_of_controls"], marker="o", color="#4C78A8")
            ax[0].set_title("Polynomial dictionary size, d=3")
            ax[0].set_xlabel("maximum polynomial degree")
            ax[0].set_ylabel("number of controls")

            cond_subset = summary[summary["method"].isin(["ZV-OLS-1", "ZV-OLS-2", f"ZV-OLS-{LARGE_DEGREE}"])]
            labels = ["ZV-OLS-1", "ZV-OLS-2", f"ZV-OLS-{LARGE_DEGREE}"]
            cond_vals = [
                cond_subset[
                    (cond_subset["dataset"] == "simulated") &
                    (cond_subset["parameter"] == "omega_1") &
                    (cond_subset["method"] == method)
                ]["median_condition"].iloc[0]
                for method in labels
            ]
            ax[1].bar(labels, cond_vals, color=["#4C78A8", "#F58518", "#B279A2"], alpha=0.75)
            ax[1].set_yscale("log")
            ax[1].set_title("Design conditioning grows quickly")
            ax[1].set_ylabel("median condition number")
            ax[1].tick_params(axis="x", rotation=20)
            fig.savefig(FIG_DIR / "control_dictionary_cost.pdf", bbox_inches="tight")
            fig.savefig(FIG_DIR / "control_dictionary_cost.png", bbox_inches="tight")
            plt.show()
            """
        ),
        code(
            r"""
            def chain_acf(x, max_lag=60):
                x = np.asarray(x, dtype=float)
                x = x - x.mean()
                denom = np.dot(x, x)
                if denom <= 0:
                    return np.zeros(max_lag + 1)
                return np.array([1.0] + [np.dot(x[:-lag], x[lag:]) / denom for lag in range(1, max_lag + 1)])

            def plot_representative_chain(dataset, path_base):
                result = representatives[dataset]
                fig, axes = plt.subplots(3, 3, figsize=(12.5, 7.0), constrained_layout=True)
                x_axis = np.arange(result.omega.shape[0]) * THIN + BURN_IN
                for j, param in enumerate(g.PARAMETER_NAMES):
                    series = result.omega[:, j]
                    running_mean = np.cumsum(series) / np.arange(1, series.size + 1)
                    acf = chain_acf(series, max_lag=60)
                    axes[j, 0].plot(x_axis, series, lw=0.65, color="#4C78A8")
                    axes[j, 0].set_ylabel(param)
                    axes[j, 0].set_title(f"{param} trace")
                    axes[j, 1].plot(x_axis, running_mean, lw=1.0, color="#F58518")
                    axes[j, 1].axhline(running_mean[-1], color="#222222", lw=0.8, ls="--")
                    axes[j, 1].set_title(f"{param} running mean")
                    axes[j, 2].bar(np.arange(acf.size), acf, color="#54A24B", alpha=0.75)
                    axes[j, 2].axhline(0.0, color="#222222", lw=0.8)
                    axes[j, 2].set_title(f"{param} chain ACF")
                fig.suptitle(f"Representative chain diagnostics after burn-in: {dataset}", y=1.02)
                fig.savefig(FIG_DIR / f"{path_base}.pdf", bbox_inches="tight")
                fig.savefig(FIG_DIR / f"{path_base}.png", bbox_inches="tight")
                plt.show()

            plot_representative_chain("simulated", "chain_simulated")
            plot_representative_chain("eurusd", "chain_eurusd")
            """
        ),
        md(
            r"""
            The traces are random-walk Metropolis paths after burn-in.  Short-range upward or downward runs are expected because the samples are autocorrelated.  The relevant diagnostic is whether the running mean stabilizes and whether repeated independent chains give similar final averages; this is why the notebook uses longer chains and reports repeated-run boxplots.
            """
        ),
        code(
            r"""
            def plot_posterior_pairs(dataset, path_base):
                result = representatives[dataset]
                omega = result.omega
                step = max(1, omega.shape[0] // 1200)
                sampled = omega[::step]
                fig, axes = plt.subplots(3, 3, figsize=(8.3, 8.0), constrained_layout=True)
                for i, row_param in enumerate(g.PARAMETER_NAMES):
                    for j, col_param in enumerate(g.PARAMETER_NAMES):
                        ax = axes[i, j]
                        if i == j:
                            ax.hist(sampled[:, j], bins=35, color="#4C78A8", alpha=0.75)
                        else:
                            ax.scatter(sampled[:, j], sampled[:, i], s=6, alpha=0.25, color="#4C78A8", edgecolors="none")
                        if i == 2:
                            ax.set_xlabel(col_param)
                        if j == 0:
                            ax.set_ylabel(row_param)
                fig.suptitle(f"Posterior pair plots: {dataset}", y=1.02)
                fig.savefig(FIG_DIR / f"{path_base}.pdf", bbox_inches="tight")
                fig.savefig(FIG_DIR / f"{path_base}.png", bbox_inches="tight")
                plt.show()

            plot_posterior_pairs("simulated", "posterior_pairs_simulated")
            plot_posterior_pairs("eurusd", "posterior_pairs_eurusd")
            """
        ),
        md(
            r"""
            ## Numerical error assessment

            The following tables summarize the repeated-run distribution.  The column `sd across runs` is the empirical standard deviation of the final estimate over the independent chains.  The boxplots above visualize the same numerical error directly.  `var. red.` is the empirical variance ratio relative to raw Monte Carlo, for the same parameter and dataset.
            """
        ),
        code(
            r"""
            display_cols = ["dataset", "parameter", "method", "mean_estimate", "numerical_sd", "rmse_to_reference", "variance_reduction", "mean_selected", "mean_acceptance"]
            summary[display_cols].sort_values(["dataset", "parameter", "method"])
            """
        ),
        md(
            r"""
            ## Why Lasso is useful here

            For \(d=3\), all non-constant monomials up to total degree \(q\) already produce
            \[
            \binom{d+q}{d}-1
            \]
            candidate controls.  This is moderate for \(q=2\), but the growth is combinatorial in dimension and degree.  Ordinary least squares on all controls can become expensive and unstable because the design matrix becomes wide and highly collinear.  The post-Lasso approach screens a large dictionary, keeps a smaller active set, and then performs the usual OLS control-variate regression on that active set.
            """
        ),
        code(
            r"""
            controls_growth
            """
        ),
        md(
            r"""
            ## Bonus: regression on MCMC output

            Ordinary linear regression formulas are derived for independent observations.  MCMC samples are dependent, so the usual regression standard errors are not valid.  The control-variate estimator itself is still meaningful under the same type of ergodic-average conditions used for MCMC central limit theorems, but its Monte Carlo error must be assessed with MCMC-aware methods.

            Practical fixes:

            - estimate the control coefficients on a pilot chain and evaluate the controlled estimator on a second independent chain;
            - thin or subsample the chain when fitting the regression, which reduces dependence at the cost of fewer fitting points;
            - use batch means or spectral variance estimators for the final numerical standard error;
            - use block cross-validation for Lasso, so adjacent dependent observations do not appear in both training and validation folds.

            The notebook reports batch-means standard errors in the tables and, more importantly, independent repeated-chain boxplots.
            """
        ),
        code(
            r"""
            bonus_rows = []
            dataset = "simulated"
            returns = simulated_returns
            result = representatives[dataset]
            grad = g.grad_log_posterior_theta_batch(returns, result.theta)
            Hlarge, _, _ = g.zv_controls(result.theta, grad, degree=LARGE_DEGREE)
            for j, param in enumerate(g.PARAMETER_NAMES):
                y = result.omega[:, j]
                full = g.controlled_estimate_ols(y, Hlarge)
                thinned = g.controlled_estimate_ols(y[::5], Hlarge[::5])
                bonus_rows.append({
                    "parameter": param,
                    "full_sample_estimate": full.estimate,
                    "full_batch_se": g.batch_means_se(full.controlled_values, batch_size=80),
                    "thin5_estimate": thinned.estimate,
                    "thin5_batch_se": g.batch_means_se(thinned.controlled_values, batch_size=30),
                })

            bonus_table = pd.DataFrame(bonus_rows)
            bonus_table.to_csv(RESULT_DIR / "bonus_thinning.csv", index=False)
            bonus_table.to_latex(RESULT_DIR / "bonus_thinning.tex", index=False, escape=True)
            bonus_table
            """
        ),
        md(
            r"""
            ## Main takeaways

            - The random-walk Metropolis sampler targets the transformed GARCH posterior and gives interpretable posterior summaries in \(\omega\).
            - First-order zero-variance controls are easy to implement once \(\nabla\log\pi\) is available.
            - Larger polynomial dictionaries can reduce variance further, but the OLS design quickly becomes ill-conditioned.
            - Lasso screening is a practical way to use a large candidate dictionary without relying on the full unstable regression.
            - Numerical error should be reported from repeated chains or MCMC-aware standard errors, not from iid regression formulas.
            """
        ),
    ]

    nb["cells"] = cells
    NOTEBOOK.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"Wrote {NOTEBOOK}")


if __name__ == "__main__":
    main()
