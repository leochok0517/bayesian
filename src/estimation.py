import numpy as np
from scipy.optimize import minimize

from src.model import solve_sir


def least_squares_objective(params, t, I_obs, S0, I0, R0):
    beta, gamma = params

    # 파라미터가 음수가 되면 큰 패널티
    if beta <= 0 or gamma <= 0:
        return 1e10

    _, I_model, _ = solve_sir(beta, gamma, S0, I0, R0, t)

    residuals = I_obs - I_model
    sse = np.sum(residuals**2)

    return sse


def estimate_parameters_ls(t, I_obs, S0, I0, R0, beta_init=0.2, gamma_init=0.2):
    initial_guess = np.array([beta_init, gamma_init])

    bounds = [(1e-6, 2.0), (1e-6, 2.0)]

    result = minimize(
        least_squares_objective,
        x0=initial_guess,
        args=(t, I_obs, S0, I0, R0),
        method="L-BFGS-B",
        bounds=bounds
    )

    beta_hat, gamma_hat = result.x

    return {
        "beta_hat": beta_hat,
        "gamma_hat": gamma_hat,
        "success": result.success,
        "message": result.message,
        "fun": result.fun,
        "result": result
    }

def log_likelihood_gaussian(beta, gamma, t, I_obs, S0, I0, R0, sigma=10.0):
    if beta <= 0 or gamma <= 0:
        return -np.inf

    _, I_model, _ = solve_sir(beta, gamma, S0, I0, R0, t)
    residuals = I_obs - I_model

    ll = -0.5 * np.sum((residuals / sigma) ** 2 + np.log(2 * np.pi * sigma**2))
    return ll


def negative_log_likelihood(params, t, I_obs, S0, I0, R0, sigma=10.0):
    beta, gamma = params
    ll = log_likelihood_gaussian(beta, gamma, t, I_obs, S0, I0, R0, sigma)
    return -ll if np.isfinite(ll) else 1e10


def estimate_parameters_mle(t, I_obs, S0, I0, R0, sigma=10.0,
                            beta_init=0.2, gamma_init=0.2):

    initial_guess = np.array([beta_init, gamma_init])
    bounds = [(1e-6, 2.0), (1e-6, 2.0)]

    result = minimize(
        negative_log_likelihood,
        x0=initial_guess,
        args=(t, I_obs, S0, I0, R0, sigma),
        method="L-BFGS-B",
        bounds=bounds
    )

    beta_hat, gamma_hat = result.x

    return {
        "beta_hat": beta_hat,
        "gamma_hat": gamma_hat,
        "success": result.success,
        "fun": result.fun,
        "result": result
    }


def log_prior_uniform(beta, gamma, beta_min=0.0, beta_max=1.0, gamma_min=0.0, gamma_max=1.0):
    if beta_min <= beta <= beta_max and gamma_min <= gamma <= gamma_max:
        return 0.0
    return -np.inf


def log_posterior(
    beta, gamma, t, I_obs, S0, I0, R0, sigma=10.0,
    beta_min=0.0, beta_max=1.0, gamma_min=0.0, gamma_max=1.0
):
    lp = log_prior_uniform(
        beta, gamma,
        beta_min=beta_min, beta_max=beta_max,
        gamma_min=gamma_min, gamma_max=gamma_max
    )

    if not np.isfinite(lp):
        return -np.inf

    ll = log_likelihood_gaussian(beta, gamma, t, I_obs, S0, I0, R0, sigma=sigma)
    return lp + ll


def compute_posterior_grid(
    beta_grid, gamma_grid, t, I_obs, S0, I0, R0, sigma=10.0,
    beta_min=0.0, beta_max=1.0, gamma_min=0.0, gamma_max=1.0
):
    log_post = np.empty((len(gamma_grid), len(beta_grid)))

    for i, gamma in enumerate(gamma_grid):
        for j, beta in enumerate(beta_grid):
            log_post[i, j] = log_posterior(
                beta, gamma, t, I_obs, S0, I0, R0, sigma=sigma,
                beta_min=beta_min, beta_max=beta_max,
                gamma_min=gamma_min, gamma_max=gamma_max
            )

    max_log_post = np.max(log_post)
    post_unnormalized = np.exp(log_post - max_log_post)
    posterior = post_unnormalized / np.sum(post_unnormalized)

    return posterior, log_post


def posterior_mean_from_grid(beta_grid, gamma_grid, posterior):
    beta_mean = np.sum(beta_grid[np.newaxis, :] * posterior)
    gamma_mean = np.sum(gamma_grid[:, np.newaxis] * posterior)
    return beta_mean, gamma_mean


def map_estimate_from_grid(beta_grid, gamma_grid, posterior):
    idx = np.unravel_index(np.argmax(posterior), posterior.shape)
    gamma_map = gamma_grid[idx[0]]
    beta_map = beta_grid[idx[1]]
    return beta_map, gamma_map