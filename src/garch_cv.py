"""Random-walk Metropolis and zero-variance controls for a GARCH(1,1) model.

The implementation uses the GARCH(1,1) recursion

    r_t | F_{t-1} ~ N(0, h_t),
    h_t = omega_1 + omega_2 r_{t-1}^2 + omega_3 h_{t-1}.

Sampling is performed on an unconstrained variable theta.  The map
theta -> omega enforces omega_1 > 0, omega_2 > 0, omega_3 > 0 and
omega_2 + omega_3 < rho.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import comb
from typing import Iterable

import numpy as np


PARAMETER_NAMES = ("omega_1", "omega_2", "omega_3")
THETA_NAMES = ("theta_1", "theta_2", "theta_3")
DEFAULT_PRIOR_SD = np.array([1.0, 1.0, 1.0])
DEFAULT_RHO = 0.995


@dataclass
class MCMCResult:
    theta: np.ndarray
    omega: np.ndarray
    logpost: np.ndarray
    acceptance_rate: float
    proposal_scale: np.ndarray


@dataclass
class ControlEstimate:
    estimate: float
    beta: np.ndarray
    selected: np.ndarray
    condition_number: float
    controlled_values: np.ndarray


def simulate_garch(
    n_obs: int,
    omega: Iterable[float],
    seed: int = 123,
    burn_in: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a stationary GARCH(1,1) path."""
    omega = np.asarray(omega, dtype=float)
    omega_1, omega_2, omega_3 = omega
    if omega_1 <= 0 or omega_2 < 0 or omega_3 < 0 or omega_2 + omega_3 >= 1:
        raise ValueError("simulate_garch requires omega_1>0, omega_2>=0, omega_3>=0, omega_2+omega_3<1")

    rng = np.random.default_rng(seed)
    total = n_obs + burn_in
    eps = rng.normal(size=total)
    r = np.zeros(total)
    h = np.zeros(total)
    h[0] = omega_1 / (1.0 - omega_2 - omega_3)
    r[0] = np.sqrt(h[0]) * eps[0]

    for t in range(1, total):
        h[t] = omega_1 + omega_2 * r[t - 1] ** 2 + omega_3 * h[t - 1]
        r[t] = np.sqrt(h[t]) * eps[t]

    return r[burn_in:], h[burn_in:]


def theta_to_omega(theta: np.ndarray, rho: float = DEFAULT_RHO) -> np.ndarray:
    """Map unconstrained theta to constrained GARCH parameters."""
    theta = np.asarray(theta, dtype=float)
    single = theta.ndim == 1
    theta_2d = theta.reshape(1, -1) if single else theta

    u = np.exp(np.clip(theta_2d[:, 1], -700, 700))
    v = np.exp(np.clip(theta_2d[:, 2], -700, 700))
    s = 1.0 + u + v
    omega = np.empty_like(theta_2d)
    omega[:, 0] = np.exp(np.clip(theta_2d[:, 0], -700, 700))
    omega[:, 1] = rho * u / s
    omega[:, 2] = rho * v / s
    return omega[0] if single else omega


def omega_to_theta(omega: Iterable[float], rho: float = DEFAULT_RHO) -> np.ndarray:
    """Inverse map from constrained omega to unconstrained theta."""
    omega = np.asarray(omega, dtype=float)
    residual = rho - omega[1] - omega[2]
    if omega[0] <= 0 or omega[1] <= 0 or omega[2] <= 0 or residual <= 0:
        raise ValueError("omega is outside the transformed parameter domain")
    return np.array(
        [
            np.log(omega[0]),
            np.log(omega[1] / residual),
            np.log(omega[2] / residual),
        ],
        dtype=float,
    )


def log_jacobian_theta_to_omega(theta: Iterable[float], rho: float = DEFAULT_RHO) -> float:
    """Log determinant of the theta -> omega transformation."""
    theta = np.asarray(theta, dtype=float)
    a, b = theta[1], theta[2]
    max_ab = max(0.0, a, b)
    log_s = max_ab + np.log(np.exp(-max_ab) + np.exp(a - max_ab) + np.exp(b - max_ab))
    return theta[0] + 2.0 * np.log(rho) + a + b - 3.0 * log_s


def jacobian_theta_to_omega(theta: Iterable[float], rho: float = DEFAULT_RHO) -> np.ndarray:
    """Jacobian matrix with entries d omega_i / d theta_j."""
    omega = theta_to_omega(np.asarray(theta, dtype=float), rho=rho)
    omega_1, alpha, beta = omega
    jac = np.zeros((3, 3), dtype=float)
    jac[0, 0] = omega_1
    jac[1, 1] = alpha * (1.0 - alpha / rho)
    jac[1, 2] = -alpha * beta / rho
    jac[2, 1] = -alpha * beta / rho
    jac[2, 2] = beta * (1.0 - beta / rho)
    return jac


def grad_log_jacobian_theta_to_omega(theta: Iterable[float], rho: float = DEFAULT_RHO) -> np.ndarray:
    """Gradient of the log-Jacobian term."""
    omega = theta_to_omega(np.asarray(theta, dtype=float), rho=rho)
    return np.array([1.0, 1.0 - 3.0 * omega[1] / rho, 1.0 - 3.0 * omega[2] / rho])


def garch_filter_and_derivatives(
    returns: np.ndarray,
    omega: Iterable[float],
    h0: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return conditional variances and their omega-derivatives."""
    r = np.asarray(returns, dtype=float)
    omega = np.asarray(omega, dtype=float)
    omega_1, omega_2, omega_3 = omega
    n_obs = len(r)
    if h0 is None:
        h0 = float(np.var(r, ddof=1))
    h0 = max(h0, 1e-10)

    h = np.empty(n_obs, dtype=float)
    dh = np.zeros((n_obs, 3), dtype=float)
    h[0] = h0
    for t in range(1, n_obs):
        h[t] = omega_1 + omega_2 * r[t - 1] ** 2 + omega_3 * h[t - 1]
        if h[t] <= 0 or not np.isfinite(h[t]):
            h[t] = np.nan
            break
        dh[t, 0] = 1.0 + omega_3 * dh[t - 1, 0]
        dh[t, 1] = r[t - 1] ** 2 + omega_3 * dh[t - 1, 1]
        dh[t, 2] = h[t - 1] + omega_3 * dh[t - 1, 2]

    return h, dh


def log_prior_omega(omega: Iterable[float], prior_sd: np.ndarray = DEFAULT_PRIOR_SD) -> float:
    """Independent centered normal prior on omega, restricted by the transform."""
    omega = np.asarray(omega, dtype=float)
    if np.any(omega <= 0) or omega[1] + omega[2] >= DEFAULT_RHO:
        return -np.inf
    return float(-0.5 * np.sum((omega / prior_sd) ** 2))


def log_likelihood_omega(returns: np.ndarray, omega: Iterable[float]) -> float:
    """Gaussian quasi-log-likelihood for GARCH(1,1), up to a constant."""
    h, _ = garch_filter_and_derivatives(returns, omega)
    if np.any(~np.isfinite(h)) or np.any(h <= 0):
        return -np.inf
    r = np.asarray(returns, dtype=float)
    return float(-0.5 * np.sum(np.log(h) + (r**2) / h))


def log_posterior_omega(
    returns: np.ndarray,
    omega: Iterable[float],
    prior_sd: np.ndarray = DEFAULT_PRIOR_SD,
) -> float:
    lp = log_prior_omega(omega, prior_sd=prior_sd)
    if not np.isfinite(lp):
        return -np.inf
    return log_likelihood_omega(returns, omega) + lp


def grad_log_posterior_omega(
    returns: np.ndarray,
    omega: Iterable[float],
    prior_sd: np.ndarray = DEFAULT_PRIOR_SD,
) -> np.ndarray:
    """Gradient of log posterior with respect to omega."""
    omega = np.asarray(omega, dtype=float)
    h, dh = garch_filter_and_derivatives(returns, omega)
    if np.any(~np.isfinite(h)) or np.any(h <= 0):
        return np.full(3, np.nan)
    r = np.asarray(returns, dtype=float)
    weight = 0.5 * ((r**2) / (h**2) - 1.0 / h)
    grad_ll = dh.T @ weight
    grad_prior = -(omega / (prior_sd**2))
    return grad_ll + grad_prior


def log_posterior_theta(
    returns: np.ndarray,
    theta: Iterable[float],
    prior_sd: np.ndarray = DEFAULT_PRIOR_SD,
    rho: float = DEFAULT_RHO,
) -> float:
    omega = theta_to_omega(np.asarray(theta, dtype=float), rho=rho)
    lp_omega = log_posterior_omega(returns, omega, prior_sd=prior_sd)
    if not np.isfinite(lp_omega):
        return -np.inf
    return lp_omega + log_jacobian_theta_to_omega(theta, rho=rho)


def grad_log_posterior_theta(
    returns: np.ndarray,
    theta: Iterable[float],
    prior_sd: np.ndarray = DEFAULT_PRIOR_SD,
    rho: float = DEFAULT_RHO,
) -> np.ndarray:
    """Gradient of transformed log posterior with respect to theta."""
    theta = np.asarray(theta, dtype=float)
    omega = theta_to_omega(theta, rho=rho)
    grad_omega = grad_log_posterior_omega(returns, omega, prior_sd=prior_sd)
    return jacobian_theta_to_omega(theta, rho=rho).T @ grad_omega + grad_log_jacobian_theta_to_omega(theta, rho=rho)


def grad_log_posterior_theta_batch(
    returns: np.ndarray,
    theta_samples: np.ndarray,
    prior_sd: np.ndarray = DEFAULT_PRIOR_SD,
    rho: float = DEFAULT_RHO,
) -> np.ndarray:
    """Batch version of grad_log_posterior_theta."""
    return np.vstack([grad_log_posterior_theta(returns, theta, prior_sd=prior_sd, rho=rho) for theta in theta_samples])


def run_rwm_theta(
    returns: np.ndarray,
    init_theta: Iterable[float],
    proposal_scale: Iterable[float],
    n_steps: int,
    burn_in: int,
    thin: int = 1,
    seed: int = 123,
    prior_sd: np.ndarray = DEFAULT_PRIOR_SD,
    rho: float = DEFAULT_RHO,
) -> MCMCResult:
    """Random-walk Metropolis sampler on theta.

    ``proposal_scale`` may be either a vector of marginal proposal standard
    deviations or a full proposal covariance matrix.
    """
    rng = np.random.default_rng(seed)
    theta = np.asarray(init_theta, dtype=float).copy()
    proposal_scale = np.asarray(proposal_scale, dtype=float)
    if proposal_scale.ndim == 1:
        proposal_chol = None
    elif proposal_scale.ndim == 2:
        proposal_chol = np.linalg.cholesky(proposal_scale)
    else:
        raise ValueError("proposal_scale must be a vector or covariance matrix")
    current_lp = log_posterior_theta(returns, theta, prior_sd=prior_sd, rho=rho)
    if not np.isfinite(current_lp):
        raise ValueError("Initial theta has non-finite log posterior")

    kept_theta = []
    kept_logpost = []
    accepted = 0
    for step in range(n_steps):
        if proposal_chol is None:
            proposal = theta + rng.normal(scale=proposal_scale, size=theta.shape)
        else:
            proposal = theta + proposal_chol @ rng.normal(size=theta.shape)
        proposal_lp = log_posterior_theta(returns, proposal, prior_sd=prior_sd, rho=rho)
        if np.isfinite(proposal_lp) and np.log(rng.uniform()) < proposal_lp - current_lp:
            theta = proposal
            current_lp = proposal_lp
            accepted += 1
        if step >= burn_in and ((step - burn_in) % thin == 0):
            kept_theta.append(theta.copy())
            kept_logpost.append(current_lp)

    kept_theta = np.asarray(kept_theta)
    return MCMCResult(
        theta=kept_theta,
        omega=theta_to_omega(kept_theta, rho=rho),
        logpost=np.asarray(kept_logpost),
        acceptance_rate=accepted / n_steps,
        proposal_scale=proposal_scale,
    )


def tune_rwm_covariance(
    returns: np.ndarray,
    init_theta: Iterable[float],
    diagonal_scale: Iterable[float],
    seed: int = 123,
    pilot_steps: int = 2500,
    pilot_burn_in: int = 500,
    n_rounds: int = 4,
    round_steps: int = 700,
    target_low: float = 0.18,
    target_high: float = 0.35,
    prior_sd: np.ndarray = DEFAULT_PRIOR_SD,
    rho: float = DEFAULT_RHO,
) -> tuple[np.ndarray, list[float], float]:
    """Tune a full-covariance RWM proposal from a diagonal pilot chain."""
    init_theta = np.asarray(init_theta, dtype=float)
    diagonal_scale = np.asarray(diagonal_scale, dtype=float)
    pilot = run_rwm_theta(
        returns,
        init_theta,
        diagonal_scale,
        n_steps=pilot_steps,
        burn_in=pilot_burn_in,
        thin=1,
        seed=seed,
        prior_sd=prior_sd,
        rho=rho,
    )

    dim = init_theta.size
    empirical_cov = np.cov(pilot.theta.T)
    jitter = 1e-6 * np.eye(dim)
    base_cov = (2.38**2 / dim) * (empirical_cov + jitter)
    multiplier = 1.0
    theta = pilot.theta[-1]
    rates: list[float] = []
    for j in range(n_rounds):
        cov = (multiplier**2) * base_cov
        result = run_rwm_theta(
            returns,
            theta,
            cov,
            n_steps=round_steps,
            burn_in=round_steps - 1,
            thin=1,
            seed=seed + 37 * (j + 1),
            prior_sd=prior_sd,
            rho=rho,
        )
        theta = result.theta[-1]
        acc = result.acceptance_rate
        rates.append(acc)
        if acc < target_low:
            multiplier *= 0.8
        elif acc > target_high:
            multiplier *= 1.2
    return (multiplier**2) * base_cov, rates, pilot.acceptance_rate


def tune_rwm_scale(
    returns: np.ndarray,
    init_theta: Iterable[float],
    initial_scale: Iterable[float],
    seed: int = 123,
    n_rounds: int = 5,
    round_steps: int = 700,
    target_low: float = 0.18,
    target_high: float = 0.35,
    prior_sd: np.ndarray = DEFAULT_PRIOR_SD,
    rho: float = DEFAULT_RHO,
) -> tuple[np.ndarray, list[float]]:
    """Simple global pilot tuning for the diagonal random-walk scale."""
    scale = np.asarray(initial_scale, dtype=float).copy()
    theta = np.asarray(init_theta, dtype=float).copy()
    rates: list[float] = []
    for j in range(n_rounds):
        result = run_rwm_theta(
            returns,
            theta,
            scale,
            n_steps=round_steps,
            burn_in=round_steps - 1,
            thin=1,
            seed=seed + 19 * j,
            prior_sd=prior_sd,
            rho=rho,
        )
        theta = result.theta[-1]
        acc = result.acceptance_rate
        rates.append(acc)
        if acc < target_low:
            scale *= 0.75
        elif acc > target_high:
            scale *= 1.25
    return scale, rates


def check_theta_gradient(
    returns: np.ndarray,
    theta: Iterable[float],
    eps: float = 1e-5,
    prior_sd: np.ndarray = DEFAULT_PRIOR_SD,
    rho: float = DEFAULT_RHO,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compare analytic and central finite-difference theta gradients."""
    theta = np.asarray(theta, dtype=float)
    analytic = grad_log_posterior_theta(returns, theta, prior_sd=prior_sd, rho=rho)
    finite = np.zeros_like(theta)
    for j in range(theta.size):
        shift = np.zeros_like(theta)
        shift[j] = eps
        finite[j] = (
            log_posterior_theta(returns, theta + shift, prior_sd=prior_sd, rho=rho)
            - log_posterior_theta(returns, theta - shift, prior_sd=prior_sd, rho=rho)
        ) / (2.0 * eps)
    max_abs_diff = float(np.max(np.abs(analytic - finite)))
    return analytic, finite, max_abs_diff


def _multi_indices_exact(dim: int, degree: int) -> list[tuple[int, ...]]:
    return [alpha for alpha in product(range(degree + 1), repeat=dim) if sum(alpha) == degree]


def multi_indices_upto(dim: int, degree: int) -> list[tuple[int, ...]]:
    """All non-constant monomial exponents up to total degree."""
    indices: list[tuple[int, ...]] = []
    for k in range(1, degree + 1):
        indices.extend(_multi_indices_exact(dim, k))
    return indices


def count_polynomial_controls(dim: int, degree: int) -> int:
    """Number of non-constant monomial controls up to a total degree."""
    return comb(dim + degree, degree) - 1


def monomial(x: np.ndarray, alpha: tuple[int, ...]) -> np.ndarray:
    out = np.ones(x.shape[0], dtype=float)
    for j, power in enumerate(alpha):
        if power:
            out *= x[:, j] ** power
    return out


def zv_controls(
    theta_samples: np.ndarray,
    grad_logpost: np.ndarray,
    degree: int = 1,
) -> tuple[np.ndarray, list[str], list[tuple[int, ...]]]:
    """Zero-variance controls generated by monomials up to a total degree.

    For a monomial P_alpha(theta), the control is

        grad P_alpha(theta)' z(theta) - 0.5 Delta P_alpha(theta),

    with z(theta) = -0.5 grad log pi(theta).
    """
    theta_samples = np.asarray(theta_samples, dtype=float)
    z = -0.5 * np.asarray(grad_logpost, dtype=float)
    dim = theta_samples.shape[1]
    indices = multi_indices_upto(dim, degree)
    controls = np.empty((theta_samples.shape[0], len(indices)), dtype=float)
    labels: list[str] = []

    for col, alpha in enumerate(indices):
        value = np.zeros(theta_samples.shape[0], dtype=float)
        alpha_arr = np.array(alpha)
        for j, power in enumerate(alpha):
            if power >= 1:
                alpha_minus = tuple((alpha_arr - np.eye(dim, dtype=int)[j]).tolist())
                value += power * monomial(theta_samples, alpha_minus) * z[:, j]
            if power >= 2:
                alpha_minus_2 = alpha_arr.copy()
                alpha_minus_2[j] -= 2
                value -= 0.5 * power * (power - 1) * monomial(theta_samples, tuple(alpha_minus_2))
        controls[:, col] = value
        labels.append(" ".join(f"{name}^{power}" for name, power in zip(THETA_NAMES, alpha) if power) or "1")
    return controls, labels, indices


def controlled_estimate_ols(y: np.ndarray, controls: np.ndarray, selected: np.ndarray | None = None) -> ControlEstimate:
    """Regression-based control variate estimate using all supplied controls."""
    y = np.asarray(y, dtype=float)
    controls = np.asarray(controls, dtype=float)
    if controls.ndim == 1:
        controls = controls[:, None]
    if controls.size == 0 or controls.shape[1] == 0:
        return ControlEstimate(float(np.mean(y)), np.zeros(0), np.array([], dtype=int), np.nan, y.copy())

    h_mean = controls.mean(axis=0)
    y_mean = float(y.mean())
    h_centered = controls - h_mean
    y_centered = y - y_mean
    beta, *_ = np.linalg.lstsq(h_centered, y_centered, rcond=None)
    estimate = float(y_mean - h_mean @ beta)
    controlled = y - controls @ beta
    try:
        condition = float(np.linalg.cond(h_centered))
    except np.linalg.LinAlgError:
        condition = np.inf
    if selected is None:
        selected = np.arange(controls.shape[1])
    return ControlEstimate(estimate, beta, np.asarray(selected, dtype=int), condition, controlled)


def post_lasso_control_estimate(
    y: np.ndarray,
    controls: np.ndarray,
    seed: int = 123,
    cv: int = 3,
    max_train: int = 800,
    selection_tol: float = 1e-10,
    n_alphas: int = 30,
    max_iter: int = 250,
) -> ControlEstimate:
    """Lasso screening followed by OLS on selected controls.

    This uses a compact NumPy coordinate-descent path and chooses the penalty
    by BIC.  It avoids a heavy dependency while retaining the relevant
    regularization step from the Lasso control-variate paper.
    """

    y = np.asarray(y, dtype=float)
    controls = np.asarray(controls, dtype=float)
    n_obs, n_controls = controls.shape
    if n_controls == 0:
        return controlled_estimate_ols(y, controls)

    n_train = min(max_train, n_obs)
    rng = np.random.default_rng(seed)
    train_idx = np.sort(rng.choice(n_obs, size=n_train, replace=False))
    h_train = controls[train_idx]
    y_train = y[train_idx]

    h_mean = h_train.mean(axis=0)
    h_scale = h_train.std(axis=0)
    h_scale[h_scale < 1e-12] = 1.0
    x_train = (h_train - h_mean) / h_scale
    y_centered = y_train - y_train.mean()

    del cv
    x_norm = np.mean(x_train * x_train, axis=0)
    alpha_max = float(np.max(np.abs(x_train.T @ y_centered)) / n_train)
    if alpha_max <= 0 or not np.isfinite(alpha_max):
        return ControlEstimate(float(y.mean()), np.zeros(0), np.array([], dtype=int), np.nan, y.copy())

    alphas = alpha_max * np.geomspace(1.0, 1e-3, n_alphas)
    coef = np.zeros(n_controls, dtype=float)
    best_bic = np.inf
    best_coef = coef.copy()

    for alpha in alphas:
        residual = y_centered - x_train @ coef
        for _ in range(max_iter):
            max_change = 0.0
            for j in range(n_controls):
                if x_norm[j] < 1e-14:
                    continue
                residual += x_train[:, j] * coef[j]
                rho_j = float((x_train[:, j] @ residual) / n_train)
                new_coef = np.sign(rho_j) * max(abs(rho_j) - alpha, 0.0) / x_norm[j]
                residual -= x_train[:, j] * new_coef
                max_change = max(max_change, abs(new_coef - coef[j]))
                coef[j] = new_coef
            if max_change < 1e-7:
                break

        rss = float(np.sum((y_centered - x_train @ coef) ** 2))
        df = int(np.sum(np.abs(coef) > selection_tol))
        bic = n_train * np.log(max(rss / n_train, 1e-16)) + df * np.log(n_train)
        if bic < best_bic:
            best_bic = bic
            best_coef = coef.copy()

    raw_coef = best_coef / h_scale
    selected = np.flatnonzero(np.abs(raw_coef) > selection_tol)
    if selected.size == 0:
        return ControlEstimate(float(y.mean()), np.zeros(0), selected, np.nan, y.copy())

    return controlled_estimate_ols(y, controls[:, selected], selected=selected)


def batch_means_se(values: np.ndarray, batch_size: int = 100) -> float:
    """Batch-means standard error for a correlated MCMC output series."""
    values = np.asarray(values, dtype=float)
    n_batches = values.size // batch_size
    if n_batches < 2:
        return float(np.std(values, ddof=1) / np.sqrt(values.size))
    trimmed = values[: n_batches * batch_size]
    means = trimmed.reshape(n_batches, batch_size).mean(axis=1)
    return float(np.std(means, ddof=1) / np.sqrt(n_batches))


def prepare_returns_from_prices(prices: np.ndarray) -> np.ndarray:
    """Compute percentage log-returns from a price series."""
    prices = np.asarray(prices, dtype=float)
    return 100.0 * np.diff(np.log(prices))


def initial_omega_for_returns(
    returns: np.ndarray, alpha: float = 0.08, beta: float = 0.88
) -> np.ndarray:
    """Moment-based GARCH starting values from the unconditional variance."""
    returns = np.asarray(returns, dtype=float)
    unconditional_var = np.var(returns, ddof=1)
    omega_1 = max(unconditional_var * (1.0 - alpha - beta), 1e-4)
    return np.array([omega_1, alpha, beta])


def autocorrelation(x: np.ndarray, max_lag: int = 35) -> np.ndarray:
    """Sample autocorrelation function up to *max_lag*."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom <= 0:
        return np.zeros(max_lag + 1)
    return np.array(
        [1.0]
        + [np.dot(x[:-lag], x[lag:]) / denom for lag in range(1, max_lag + 1)]
    )
