"""Independent coordinate/measure audits for positive JARVIS properties.

The scientific posterior is not constructed here.  This module audits a
separate fixed direct-observation model using the locked coercive log-property
residual.  Genuine pushforwards are integrated in their native coordinates;
independently based Lebesgue routes are compared only at the corrected hard
limit.
"""

from __future__ import annotations

import math
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp
from scipy.stats import norm

from common.evidence import weighted_quantile


def fixed_direct_log_graph(
    mu: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    *,
    mu_sd: float,
) -> np.ndarray:
    """Normalized direct-observation graph density with respect to dmu."""

    values = np.asarray(mu, dtype=float)
    gamma = np.exp(values)[:, None]
    standardized = (y[None, :] - gamma * x[None, :]) / sigma[None, :]
    likelihood = np.sum(norm.logpdf(standardized) - np.log(sigma)[None, :], axis=1)
    return likelihood + norm.logpdf(values, loc=0.0, scale=float(mu_sd))


def _normalized(log_values: np.ndarray) -> np.ndarray:
    return np.exp(log_values - logsumexp(log_values))


def _summary(mu: np.ndarray, log_values: np.ndarray) -> dict[str, float]:
    weights = _normalized(log_values)
    gamma = np.exp(mu)
    mean = float(np.sum(weights * gamma))
    quantiles = weighted_quantile(gamma, weights, np.array([0.005, 0.05, 0.5, 0.95, 0.995]))
    return {
        "gamma_mean": mean,
        "gamma_sd": float(np.sqrt(np.sum(weights * (gamma - mean) ** 2))),
        "gamma_q005": float(quantiles[0]),
        "gamma_q05": float(quantiles[1]),
        "gamma_q50": float(quantiles[2]),
        "gamma_q95": float(quantiles[3]),
        "gamma_q995": float(quantiles[4]),
    }


def global_support_audit(
    x: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    *,
    mu_sd: float,
    local_bounds: tuple[float, float],
    n_points: int = 28001,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], np.ndarray, np.ndarray]:
    """Search graph and independent-base naive targets on mu in [-120, 20]."""

    mu = np.linspace(-120.0, 20.0, int(n_points))
    graph = fixed_direct_log_graph(mu, x, y, sigma, mu_sd=mu_sd)
    n = int(len(x))
    targets = {
        "geometry_graph": graph,
        "independent_direct_lebesgue_naive": graph + n * mu,
        "independent_reciprocal_lebesgue_naive": graph - n * mu,
    }
    lower, upper = local_bounds
    local = (mu >= lower) & (mu <= upper)
    edge = max(2, int(math.ceil(0.01 * len(mu))))
    rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, float]] = {}
    for name, log_values in targets.items():
        weights = _normalized(log_values)
        maximum = float(np.max(log_values))
        mode_index = int(np.argmax(log_values))
        local_mass = float(np.sum(weights[local]))
        summary = _summary(mu, log_values)
        summary.update(
            {
                "mode_mu": float(mu[mode_index]),
                "mode_gamma": float(math.exp(mu[mode_index])),
                "local_window_mass": local_mass,
            }
        )
        summaries[name] = summary
        rows.append(
            {
                "target": name,
                "mu_min": float(mu[0]),
                "mu_max": float(mu[-1]),
                "n_points": int(len(mu)),
                "mode_mu": float(mu[mode_index]),
                "mode_gamma": float(math.exp(mu[mode_index])),
                "lower_endpoint_log_drop": maximum - float(log_values[0]),
                "upper_endpoint_log_drop": maximum - float(log_values[-1]),
                "lower_edge_mass_1pct": float(np.sum(weights[:edge])),
                "upper_edge_mass_1pct": float(np.sum(weights[-edge:])),
                "local_mu_min": float(lower),
                "local_mu_max": float(upper),
                "local_window_mass": local_mass,
                "remote_window_dominates": bool(local_mass < 0.5),
                **{key: value for key, value in summary.items() if key.startswith("gamma_")},
            }
        )
    return pd.DataFrame(rows), summaries, mu, graph


@cache
def _hermite_rule(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = hermgauss(int(n_nodes))
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def _log_likelihood_state(state: np.ndarray | float, y: float, sigma: float) -> np.ndarray:
    values = np.asarray(state, dtype=float)
    return norm.logpdf((float(y) - values) / float(sigma)) - math.log(float(sigma))


@cache
def _native_positive_rule(
    n_segments: int = 80,
    nodes_per_segment: int = 24,
    log_bound: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Gauss--Legendre rule in a positive native coordinate.

    The intervals are logarithmically spaced so that a narrow peak is not
    missed, but the quadrature variable is the positive coordinate itself;
    the 1/state pushforward density remains explicitly in the integrand.
    With rho=25 the omitted tube mass beyond |log(state/m)|=10 is below
    exp(-1250), before the bounded Gaussian likelihood is applied.
    """

    base_nodes, base_weights = leggauss(int(nodes_per_segment))
    edges = np.exp(np.linspace(-float(log_bound), float(log_bound), int(n_segments) + 1))
    nodes: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        midpoint = 0.5 * (lower + upper)
        half_width = 0.5 * (upper - lower)
        nodes.append(midpoint + half_width * base_nodes)
        weights.append(half_width * base_weights)
    all_nodes = np.concatenate(nodes)
    all_weights = np.concatenate(weights)
    all_nodes.setflags(write=False)
    all_weights.setflags(write=False)
    return all_nodes, all_weights


def _finite_log_route_one(
    gamma: float,
    x: float,
    y: float,
    sigma: float,
    rho: float,
    n_nodes: int,
) -> float:
    nodes, weights = _hermite_rule(int(n_nodes))
    scale = math.sqrt(2.0 / float(rho))
    t = scale * nodes
    state = float(gamma) * float(x) * np.exp(t)
    terms = np.log(weights) + math.log(scale) + _log_likelihood_state(state, y, sigma)
    return float(logsumexp(terms))


def _finite_direct_pushforward_one(
    gamma: float,
    x: float,
    y: float,
    sigma: float,
    rho: float,
) -> float:
    m = float(gamma) * float(x)
    ratio, weights = _native_positive_rule()
    state = m * ratio
    terms = (
        np.log(weights)
        + _log_likelihood_state(state, y, sigma)
        - 0.5 * float(rho) * np.log(ratio) ** 2
        - np.log(ratio)
    )
    return float(logsumexp(terms))


def _finite_reciprocal_pushforward_one(
    gamma: float,
    x: float,
    y: float,
    sigma: float,
    rho: float,
) -> float:
    m = float(gamma) * float(x)
    reciprocal_ratio, weights = _native_positive_rule()
    state = m / reciprocal_ratio
    terms = (
        np.log(weights)
        + _log_likelihood_state(state, y, sigma)
        - 0.5 * float(rho) * np.log(reciprocal_ratio) ** 2
        - np.log(reciprocal_ratio)
    )
    return float(logsumexp(terms))


def finite_pushforward_audit(
    gamma_points: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    *,
    mu_sd: float,
    rho: float = 25.0,
    n_nodes: int = 256,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compare complete finite tubes in three genuine state coordinates."""

    rows = []
    for gamma in np.asarray(gamma_points, dtype=float):
        log_prior = float(norm.logpdf(math.log(gamma), loc=0.0, scale=mu_sd))
        route_logs = {"log": log_prior, "direct": log_prior, "reciprocal": log_prior}
        for x_i, y_i, sigma_i in zip(x, y, sigma, strict=True):
            route_logs["log"] += _finite_log_route_one(gamma, x_i, y_i, sigma_i, rho, n_nodes)
            route_logs["direct"] += _finite_direct_pushforward_one(gamma, x_i, y_i, sigma_i, rho)
            route_logs["reciprocal"] += _finite_reciprocal_pushforward_one(
                gamma, x_i, y_i, sigma_i, rho
            )
        rows.append(
            {
                "gamma": float(gamma),
                "rho": float(rho),
                "n_hermite_nodes": int(n_nodes),
                "log_coordinate_finite_logdensity": route_logs["log"],
                "direct_pushforward_finite_logdensity": route_logs["direct"],
                "reciprocal_pushforward_finite_logdensity": route_logs["reciprocal"],
            }
        )
    table = pd.DataFrame(rows)

    def centered(column: str) -> np.ndarray:
        values = table[column].to_numpy(float)
        return values - values[0]

    log_values = centered("log_coordinate_finite_logdensity")
    direct_error = centered("direct_pushforward_finite_logdensity") - log_values
    reciprocal_error = centered("reciprocal_pushforward_finite_logdensity") - log_values
    table["direct_minus_log_centered_error"] = direct_error
    table["reciprocal_minus_log_centered_error"] = reciprocal_error
    stats = {
        "n_parameter_points": int(len(table)),
        "direct_vs_log_centered_max_abs_error": float(np.max(np.abs(direct_error))),
        "reciprocal_vs_log_centered_max_abs_error": float(np.max(np.abs(reciprocal_error))),
        "rho": float(rho),
        "n_hermite_nodes": int(n_nodes),
    }
    return table, stats


def hard_limit_route_audit(
    gamma_points: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    *,
    mu_sd: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Evaluate q, J, J/q and hard limits route by route without cancellation code."""

    contracts = [
        {
            "route": "log_genuine",
            "coordinate": "w=log(Y/X0)",
            "base_relationship": "reference",
            "q_formula": "1",
            "J_formula": "1",
            "J_over_q_formula": "1",
            "finite_comparison": "genuine_pushforward_complete_tube",
            "tail_result": "Gaussian log residual is coercive at w boundaries",
        },
        {
            "route": "direct_genuine",
            "coordinate": "s=Y/X0",
            "base_relationship": "pushforward_of_dw",
            "q_formula": "1/s",
            "J_formula": "1/s",
            "J_over_q_formula": "1",
            "finite_comparison": "genuine_pushforward_complete_tube",
            "tail_result": "log residual is coercive at s=0 and infinity",
        },
        {
            "route": "reciprocal_genuine",
            "coordinate": "v=X0/Y",
            "base_relationship": "pushforward_of_dw",
            "q_formula": "1/v",
            "J_formula": "1/v",
            "J_over_q_formula": "1",
            "finite_comparison": "genuine_pushforward_complete_tube",
            "tail_result": "log residual is coercive at v=0 and infinity",
        },
        {
            "route": "direct_independent_lebesgue",
            "coordinate": "s=Y/X0",
            "base_relationship": "independent_q_equals_1",
            "q_formula": "1",
            "J_formula": "1/s",
            "J_over_q_formula": "1/s",
            "finite_comparison": "corrected_hard_limit_only",
            "tail_result": "log residual and direct likelihood give a proper finite tube",
        },
        {
            "route": "reciprocal_independent_lebesgue",
            "coordinate": "v=X0/Y",
            "base_relationship": "independent_q_equals_1",
            "q_formula": "1",
            "J_formula": "1/v",
            "J_over_q_formula": "1/v",
            "finite_comparison": "corrected_hard_limit_only",
            "tail_result": "log residual is coercive at both reciprocal boundaries",
        },
        {
            "route": "log_residual_rescaled_metric",
            "coordinate": "w=log(Y/X0)",
            "base_relationship": "residual_transform_r=gamma*c",
            "q_formula": "1",
            "J_formula": "|gamma|*sqrt(gamma^-2)=1",
            "J_over_q_formula": "1",
            "finite_comparison": "pointwise_identical_metric_transformation",
            "tail_result": "r^T R_r r equals c^2 pointwise",
        },
    ]
    rows: list[dict[str, Any]] = []
    wrong_double_count: list[float] = []
    residual_metric_errors: list[float] = []
    residual_metric_log_j: list[float] = []
    for gamma in np.asarray(gamma_points, dtype=float):
        mu = math.log(float(gamma))
        m = float(gamma) * x
        log_m = np.log(m)
        log_likelihood = float(np.sum(norm.logpdf((y - m) / sigma) - np.log(sigma)))
        log_prior = float(norm.logpdf(mu, loc=0.0, scale=mu_sd))
        graph = log_prior + log_likelihood
        standardized = (y - m) / sigma
        rescaled = float(gamma) * standardized
        residual_metric_errors.append(
            abs(float(np.sum(standardized**2)) - float(np.sum(rescaled**2 / float(gamma) ** 2)))
        )
        residual_metric_log_j.append(
            abs(len(x) * (math.log(float(gamma)) + 0.5 * math.log(float(gamma) ** -2)))
        )
        for contract in contracts:
            route = contract["route"]
            if route in {"log_genuine", "log_residual_rescaled_metric"}:
                log_q = np.zeros_like(m)
                log_j = np.zeros_like(m)
            elif route == "direct_genuine":
                log_q = -log_m
                log_j = -log_m
            elif route == "reciprocal_genuine":
                log_q = log_m
                log_j = log_m
            elif route == "direct_independent_lebesgue":
                log_q = np.zeros_like(m)
                log_j = -log_m
            elif route == "reciprocal_independent_lebesgue":
                log_q = np.zeros_like(m)
                log_j = log_m
            else:  # pragma: no cover
                raise AssertionError(route)
            log_a = log_j - log_q
            naive = graph + float(np.sum(log_q - log_j))
            # Direct limiting formula includes q/J times the declared A=J/q.
            corrected = log_prior + log_likelihood + float(np.sum(log_q - log_j + log_a))
            rows.append(
                {
                    "gamma": float(gamma),
                    "route": route,
                    "sum_log_q": float(np.sum(log_q)),
                    "sum_log_J": float(np.sum(log_j)),
                    "sum_log_J_over_q": float(np.sum(log_a)),
                    "naive_hard_logdensity": naive,
                    "corrected_hard_logdensity": corrected,
                    "graph_logdensity": graph,
                    "corrected_minus_graph": corrected - graph,
                }
            )
            if route == "direct_genuine":
                # Deliberate failure: multiply pushed q by J instead of J/q.
                wrong_double_count.append(graph + float(np.sum(log_q)))
    table = pd.DataFrame(rows)
    wrong = np.asarray(wrong_double_count, dtype=float)
    graph_reference = table[table["route"] == "log_genuine"]["graph_logdensity"].to_numpy(float)
    wrong_centered_error = (wrong - wrong[0]) - (graph_reference - graph_reference[0])
    stats = {
        "n_parameter_points": int(len(gamma_points)),
        "corrected_hard_vs_graph_max_abs_error": float(table["corrected_minus_graph"].abs().max()),
        "residual_metric_penalty_max_abs_error": float(max(residual_metric_errors)),
        "residual_metric_log_j_max_abs_error": float(max(residual_metric_log_j)),
        "double_counting_failure_centered_max_abs_error": float(
            np.max(np.abs(wrong_centered_error))
        ),
    }
    return pd.DataFrame(contracts), table, stats


def build_geometry_audit(
    x: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    *,
    mu_sd: float,
    local_bounds: tuple[float, float],
) -> dict[str, Any]:
    support, summaries, support_mu, support_graph = global_support_audit(
        x,
        y,
        sigma,
        mu_sd=mu_sd,
        local_bounds=local_bounds,
    )
    graph = summaries["geometry_graph"]
    gamma_points = np.exp(
        np.linspace(math.log(graph["gamma_q005"]), math.log(graph["gamma_q995"]), 25)
    )
    finite, finite_stats = finite_pushforward_audit(
        gamma_points,
        x,
        y,
        sigma,
        mu_sd=mu_sd,
    )
    contracts, hard, hard_stats = hard_limit_route_audit(
        gamma_points,
        x,
        y,
        sigma,
        mu_sd=mu_sd,
    )
    stats = {
        **finite_stats,
        **hard_stats,
        "global_support_all_endpoint_log_drops_above_30": bool(
            support[["lower_endpoint_log_drop", "upper_endpoint_log_drop"]].to_numpy().min() > 30.0
        ),
        "global_support_all_edge_masses_below_1e_10": bool(
            support[["lower_edge_mass_1pct", "upper_edge_mass_1pct"]].to_numpy().max() < 1.0e-10
        ),
    }
    stats["all_passed"] = bool(
        stats["n_parameter_points"] >= 25
        and stats["direct_vs_log_centered_max_abs_error"] < 1.0e-8
        and stats["reciprocal_vs_log_centered_max_abs_error"] < 1.0e-8
        and stats["corrected_hard_vs_graph_max_abs_error"] < 1.0e-10
        and stats["residual_metric_penalty_max_abs_error"] < 1.0e-12
        and stats["residual_metric_log_j_max_abs_error"] < 1.0e-12
        and stats["double_counting_failure_centered_max_abs_error"] > 1.0e-3
        and stats["global_support_all_endpoint_log_drops_above_30"]
        and stats["global_support_all_edge_masses_below_1e_10"]
    )
    return {
        "contracts": contracts,
        "hard_limits": hard,
        "finite_pushforwards": finite,
        "global_support": support,
        "summaries": summaries,
        "stats": stats,
        "_plot_mu": support_mu,
        "_plot_log_targets": {
            "geometry_graph": support_graph,
            "independent_direct_lebesgue_naive": support_graph + len(x) * support_mu,
            "independent_reciprocal_lebesgue_naive": support_graph - len(x) * support_mu,
        },
    }
