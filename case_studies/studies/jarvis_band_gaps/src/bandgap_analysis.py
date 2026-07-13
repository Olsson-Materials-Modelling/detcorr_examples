"""Registered reanalysis of the pinned JARVIS band-gap comparison table.

The natural scientific target is a common multiplicative calibration under a
normalized relative-error likelihood.  The determinant-omitted target is kept
only as a measure-error diagnostic; it is never offered as a competing model.
All predictive rows are produced by leave-one-formula-group-out refits.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp
from scipy.stats import norm, spearmanr
from scipy.stats import t as student_t

from common.evidence import strict_json_dump
from common.jarvis_geometry import build_geometry_audit
from common.jarvis_protocol import (
    PROTOCOL_VERSION,
    RESAMPLING_SEED,
    add_claim_provenance,
    canonical_hash,
    file_hash,
    logo_role_manifest,
    predictive_grid_differences,
)
from common.predictive_mixture import (
    MixtureQuantileResult,
    posterior_mixture_logratio_quantiles,
)

EXPECTED_SHA256 = "4b7a5f437b1a41030bb65c31d56eaa9524cc91598bd09ff2825c02df60528631"
SOURCE_URL = "https://jarvis-materials-design.github.io/dbdocs/jarvisdft/"
REQUIRED = (
    "material_label",
    "formula",
    "jvasp_id",
    "bandgap_opt_ev",
    "bandgap_expt_ev",
)


@dataclass(frozen=True)
class GridSpec:
    mu: np.ndarray
    eta: np.ndarray
    label: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fixture(path: Path, *, require_pinned_checksum: bool = True) -> pd.DataFrame:
    if require_pinned_checksum and sha256(path) != EXPECTED_SHA256:
        raise ValueError("band-gap fixture checksum mismatch")
    frame = pd.read_csv(path)
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    for column in ("material_label", "formula", "jvasp_id"):
        if frame[column].isna().any():
            raise ValueError(f"{column} is required and cannot be blank")
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"{column} is required and cannot be blank")
    for column in ("formula", "jvasp_id"):
        if frame[column].str.contains(r"\s", regex=True).any():
            raise ValueError(f"{column} cannot contain whitespace after normalization")
    if frame["jvasp_id"].duplicated().any():
        raise ValueError("jvasp_id must be unique after whitespace normalization")
    for column in ("bandgap_opt_ev", "bandgap_expt_ev"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.all(np.isfinite(frame[column])) or np.any(frame[column] <= 0.0):
            raise ValueError(f"{column} must be finite and positive")
    return frame.reset_index(drop=True)


def detcorr_api_audit(gamma: float, opt_gap_ev: np.ndarray) -> dict[str, Any]:
    """Check the release API against the independently derived diagonal result."""

    import detcorr

    version = str(detcorr.__version__)
    numeric = tuple(int(part) for part in version.split("+")[0].split(".")[:2])
    if numeric < (0, 10):
        raise RuntimeError(f"detcorr>=0.10 is required, found {version}")
    # For e=E/E0, c=log(e/(gamma*E_OPT/E0)),
    # |D_e c|=1/(gamma*E_OPT/E0) on the graph (E0=1 eV here).
    diagonal = 1.0 / (gamma * np.asarray(opt_gap_ev, dtype=float))
    result = detcorr.log_correction(jacobian=diagonal, structure="diagonal")
    analytic = float(np.sum(np.log(diagonal)))
    return {
        "analytic_logabsdet": analytic,
        "api_logabsdet": float(result.value),
        "detcorr_version": version,
        "method": str(result.method),
        "structure": "diagonal",
        "absolute_error": abs(float(result.value) - analytic),
    }


def make_grid(mode: str) -> GridSpec:
    sizes = {
        "quick": (181, 101),
        "standard": (321, 181),
        "full": (641, 321),
    }
    n_mu, n_eta = sizes[mode]
    return GridSpec(
        mu=np.linspace(-1.60, 1.60, n_mu),
        eta=np.linspace(math.log(0.03), math.log(2.50), n_eta),
        label=mode,
    )


def _log_prior(grid: GridSpec, *, mu_sd: float = 0.75) -> np.ndarray:
    # Prior is declared with respect to dmu deta.  eta=log(tau).
    mu = grid.mu[:, None]
    eta = grid.eta[None, :]
    eta_center = math.log(0.45)
    eta_sd = 0.75
    return -0.5 * (mu / mu_sd) ** 2 - 0.5 * ((eta - eta_center) / eta_sd) ** 2


def log_posterior_grid(
    log_ratio: np.ndarray,
    grid: GridSpec,
    *,
    family: str = "normal",
    mu_sd: float = 0.75,
) -> np.ndarray:
    values = np.asarray(log_ratio, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("log_ratio must contain at least two finite values")
    mu = grid.mu[:, None]
    tau = np.exp(grid.eta)[None, :]
    if family == "normal":
        centered = values[:, None] - grid.mu[None, :]
        sse = np.sum(centered * centered, axis=0)[:, None]
        log_like = -values.size * np.log(tau) - 0.5 * sse / (tau * tau)
    elif family == "student_t4":
        nu = 4.0
        z = (values[None, None, :] - mu[:, :, None]) / tau[:, :, None]
        constant = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0)
        constant -= 0.5 * math.log(nu * math.pi)
        log_like = np.sum(
            constant - np.log(tau[:, :, None]) - 0.5 * (nu + 1.0) * np.log1p((z * z) / nu),
            axis=2,
        )
    else:
        raise ValueError(f"unknown family: {family}")
    return log_like + _log_prior(grid, mu_sd=mu_sd)


def direct_fixed_log_posterior(
    x: np.ndarray,
    y: np.ndarray,
    mu: np.ndarray,
    *,
    floor_ev: float,
    fractional: float,
    family: str,
    mu_sd: float = 0.75,
) -> np.ndarray:
    gamma = np.exp(mu)[:, None]
    residual = y[None, :] - gamma * x[None, :]
    sigma = np.sqrt(floor_ev**2 + (fractional * x) ** 2)[None, :]
    if family == "normal":
        ll = np.sum(norm.logpdf(residual / sigma) - np.log(sigma), axis=1)
    elif family == "student_t4":
        ll = np.sum(student_t.logpdf(residual / sigma, df=4) - np.log(sigma), axis=1)
    else:
        raise ValueError(f"unknown family: {family}")
    return ll - 0.5 * (mu / mu_sd) ** 2


def normalize_log_weights(log_weight: np.ndarray) -> np.ndarray:
    log_weight = np.asarray(log_weight, dtype=float)
    return np.exp(log_weight - logsumexp(log_weight))


def _trapezoid_axis_weights(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    differences = np.diff(values)
    if values.ndim != 1 or values.size < 2 or np.any(differences <= 0.0):
        raise ValueError("quadrature points must be a strictly increasing vector")
    weights = np.empty_like(values)
    weights[0] = 0.5 * differences[0]
    weights[-1] = 0.5 * differences[-1]
    weights[1:-1] = 0.5 * (differences[:-1] + differences[1:])
    return weights


def _grid_log_weights(grid: GridSpec, log_post: np.ndarray) -> np.ndarray:
    measure = np.outer(
        _trapezoid_axis_weights(grid.mu),
        _trapezoid_axis_weights(grid.eta),
    )
    values = np.asarray(log_post, dtype=float) + np.log(measure)
    return values - logsumexp(values)


def _eta_log_weights(grid: GridSpec, log_post_eta: np.ndarray) -> np.ndarray:
    values = np.asarray(log_post_eta, dtype=float) + np.log(_trapezoid_axis_weights(grid.eta))
    return values - logsumexp(values)


def _continuous_mu_quantiles(
    grid: GridSpec,
    log_density: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    shifted = np.exp(log_density - np.max(log_density))
    marginal = shifted @ _trapezoid_axis_weights(grid.eta)
    increments = 0.5 * (marginal[:-1] + marginal[1:]) * np.diff(grid.mu)
    cdf = np.concatenate(([0.0], np.cumsum(increments)))
    if not np.isfinite(cdf[-1]) or cdf[-1] <= 0.0:
        raise RuntimeError("posterior mu marginal could not be normalized")
    cdf /= cdf[-1]
    keep = np.concatenate(([True], np.diff(cdf) > 0.0))
    return np.interp(probabilities, cdf[keep], grid.mu[keep])


def posterior_summary(
    grid: GridSpec, log_post: np.ndarray, *, tilt: float = 0.0
) -> dict[str, float]:
    tilted = log_post + tilt * grid.mu[:, None]
    weights = np.exp(_grid_log_weights(grid, tilted).ravel())
    gamma = np.repeat(np.exp(grid.mu), grid.eta.size)
    mean = float(np.sum(weights * gamma))
    variance = float(np.sum(weights * (gamma - mean) ** 2))
    mu_quantiles = _continuous_mu_quantiles(
        grid,
        tilted,
        np.array([0.05, 0.5, 0.95]),
    )
    q05, q50, q95 = np.exp(mu_quantiles)
    tau = np.tile(np.exp(grid.eta), grid.mu.size)
    matrix_weights = weights.reshape(grid.mu.size, grid.eta.size)
    edge_cells = max(2, int(math.ceil(0.02 * grid.mu.size)))
    return {
        "gamma_mean": mean,
        "gamma_sd": math.sqrt(max(variance, 0.0)),
        "gamma_q05": float(q05),
        "gamma_q50": float(q50),
        "gamma_q95": float(q95),
        "tau_log_ratio_mean": float(np.sum(weights * tau)),
        "lower_mu_edge_mass_2pct": float(np.sum(matrix_weights[:edge_cells, :])),
        "upper_mu_edge_mass_2pct": float(np.sum(matrix_weights[-edge_cells:, :])),
    }


def load_data_manifest(
    path: Path,
    digest: str,
    *,
    data_path: Path,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "dataset_id",
        "version",
        "study_role",
        "evidence_timing_class",
        "source_name",
        "source_url_or_accession",
        "source_citation",
        "retrieval_timestamp_utc",
        "retrieval_method",
        "raw_filename",
        "raw_sha256",
        "raw_bytes",
        "schema_version",
        "source_url",
        "retrieval_date",
        "license_status",
        "preparation",
        "sha256",
        "row_count",
        "source_row_key",
        "dependent_observation_columns",
        "reference_input_columns",
        "units",
        "selection_rule",
        "deduplication_rule",
        "grouping_rule",
        "known_source_dependence",
        "redistribution_decision",
        "transform_code_commit",
        "processed_filename",
        "processed_sha256",
        "processed_bytes",
        "protocol_version",
        "protocol_timing",
        "row_identity_key",
        "grouping_key",
        "n_rows",
        "n_formula_groups",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"data manifest missing fields: {missing}")
    blank = sorted(key for key in required if manifest[key] in (None, "", [], {}))
    if blank:
        raise ValueError(f"data manifest has blank fields: {blank}")
    if manifest["sha256"] != digest:
        raise ValueError("data manifest SHA256 does not match selected input")
    if manifest["processed_sha256"] != digest:
        raise ValueError("processed data SHA256 does not match selected input")
    if manifest["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError(f"manifest protocol_version must be {PROTOCOL_VERSION}")
    if manifest["row_identity_key"] != "jvasp_id" or manifest["grouping_key"] != "formula":
        raise ValueError("manifest must declare jvasp_id identity and formula grouping")
    expected_dependent = ["bandgap_expt_ev"]
    expected_reference = ["bandgap_opt_ev"]
    expected_units = {
        "bandgap_opt_ev": "eV",
        "bandgap_expt_ev": "eV",
    }
    checks = {
        "processed_filename": manifest["processed_filename"] == data_path.name,
        "processed_bytes": manifest["processed_bytes"] == data_path.stat().st_size,
        "row_count": manifest["row_count"] == len(frame),
        "n_rows": manifest["n_rows"] == len(frame),
        "n_formula_groups": manifest["n_formula_groups"] == frame["formula"].nunique(),
        "dependent_observation_columns": (
            manifest["dependent_observation_columns"] == expected_dependent
        ),
        "reference_input_columns": manifest["reference_input_columns"] == expected_reference,
        "units": manifest["units"] == expected_units,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"data manifest declarations do not match selected input: {failed}")
    return manifest


def summary_one_dim(mu: np.ndarray, log_post: np.ndarray) -> dict[str, float]:
    axis_weights = _trapezoid_axis_weights(mu)
    weights = normalize_log_weights(log_post + np.log(axis_weights))
    gamma = np.exp(mu)
    mean = float(np.sum(weights * gamma))
    density = np.exp(log_post - np.max(log_post))
    increments = 0.5 * (density[:-1] + density[1:]) * np.diff(mu)
    cdf = np.concatenate(([0.0], np.cumsum(increments)))
    cdf /= cdf[-1]
    keep = np.concatenate(([True], np.diff(cdf) > 0.0))
    q = np.exp(np.interp(np.array([0.05, 0.5, 0.95]), cdf[keep], mu[keep]))
    return {
        "gamma_mean": mean,
        "gamma_sd": float(np.sqrt(np.sum(weights * (gamma - mean) ** 2))),
        "gamma_q05": float(q[0]),
        "gamma_q50": float(q[1]),
        "gamma_q95": float(q[2]),
    }


PREDICTIVE_QUANTILE_PROBABILITIES = np.array([0.05, 0.25, 0.50, 0.75, 0.95])


def predictive_record(
    x: float,
    y: float,
    grid: GridSpec,
    log_post: np.ndarray,
    *,
    family: str = "student_t4",
    quantile_result: MixtureQuantileResult | None = None,
) -> dict[str, Any]:
    posterior_log_weights = _grid_log_weights(grid, log_post).ravel()
    weights = np.exp(posterior_log_weights)
    mu = np.repeat(grid.mu, grid.eta.size)
    tau = np.tile(np.exp(grid.eta), grid.mu.size)
    log_ratio = math.log(y / x)
    z_observed = (log_ratio - mu) / tau
    if family == "student_t4":
        component_logpdf = student_t.logpdf(z_observed, df=4) - np.log(tau) - math.log(y)
    elif family == "normal":
        component_logpdf = norm.logpdf(z_observed) - np.log(tau) - math.log(y)
    else:
        raise ValueError(f"unknown family: {family}")
    log_score = float(logsumexp(posterior_log_weights + component_logpdf))
    if quantile_result is None:
        quantile_result = posterior_mixture_logratio_quantiles(
            mu,
            tau,
            weights,
            PREDICTIVE_QUANTILE_PROBABILITIES,
            family=family,
        )
    q05, q25, q50, q75, q95 = x * np.exp(quantile_result.quantiles)
    diagnostics = quantile_result.diagnostics
    return {
        "coverage50": float(q25 <= y <= q75),
        "coverage90": float(q05 <= y <= q95),
        "interval50_width_ev": float(q75 - q25),
        "interval90_width_ev": float(q95 - q05),
        "log_predictive_density_per_ev": log_score,
        "prediction_point_ev": float(q50),
        "prediction_median_ev": float(q50),
        "prediction_q05_ev": float(q05),
        "prediction_q25_ev": float(q25),
        "prediction_q75_ev": float(q75),
        "prediction_q95_ev": float(q95),
        "quantile_cdf_method": diagnostics.method,
        "quantile_input_component_count": diagnostics.input_component_count,
        "quantile_evaluated_component_count": diagnostics.evaluated_component_count,
        "quantile_max_abs_cdf_root_residual": diagnostics.max_abs_cdf_root_residual,
    }


def identity_posterior(
    log_ratio: np.ndarray, grid: GridSpec, *, family: str = "student_t4"
) -> np.ndarray:
    tau = np.exp(grid.eta)
    eta_prior = -0.5 * ((grid.eta - math.log(0.45)) / 0.75) ** 2
    z = log_ratio[None, :] / tau[:, None]
    if family == "student_t4":
        ll = np.sum(student_t.logpdf(z, df=4) - np.log(tau[:, None]), axis=1)
    elif family == "normal":
        ll = np.sum(norm.logpdf(z) - np.log(tau[:, None]), axis=1)
    else:
        raise ValueError(f"unknown family: {family}")
    return ll + eta_prior


def identity_predictive_record(
    x: float,
    y: float,
    grid: GridSpec,
    log_post_eta: np.ndarray,
    *,
    family: str = "student_t4",
    quantile_result: MixtureQuantileResult | None = None,
) -> dict[str, Any]:
    log_weight = _eta_log_weights(grid, log_post_eta)
    weights = np.exp(log_weight)
    tau = np.exp(grid.eta)
    r = math.log(y / x)
    standardized = r / tau
    component = (
        student_t.logpdf(standardized, df=4)
        if family == "student_t4"
        else norm.logpdf(standardized)
    )
    score = float(logsumexp(log_weight + component - np.log(tau)) - math.log(y))
    if quantile_result is None:
        quantile_result = posterior_mixture_logratio_quantiles(
            np.zeros_like(tau),
            tau,
            weights,
            PREDICTIVE_QUANTILE_PROBABILITIES,
            family=family,
        )
    q05, q25, q50, q75, q95 = x * np.exp(quantile_result.quantiles)
    diagnostics = quantile_result.diagnostics
    return {
        "coverage50": float(q25 <= y <= q75),
        "coverage90": float(q05 <= y <= q95),
        "interval50_width_ev": float(q75 - q25),
        "interval90_width_ev": float(q95 - q05),
        "log_predictive_density_per_ev": score,
        "prediction_point_ev": float(q50),
        "prediction_median_ev": float(q50),
        "prediction_q05_ev": float(q05),
        "prediction_q25_ev": float(q25),
        "prediction_q75_ev": float(q75),
        "prediction_q95_ev": float(q95),
        "quantile_cdf_method": diagnostics.method,
        "quantile_input_component_count": diagnostics.input_component_count,
        "quantile_evaluated_component_count": diagnostics.evaluated_component_count,
        "quantile_max_abs_cdf_root_residual": diagnostics.max_abs_cdf_root_residual,
    }


def grouped_log_scale_predictions(frame: pd.DataFrame, grid: GridSpec) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    x_all = frame["bandgap_opt_ev"].to_numpy(float)
    y_all = frame["bandgap_expt_ev"].to_numpy(float)
    groups = frame["formula"].astype(str).to_numpy()
    for fold_index, held_group in enumerate(sorted(set(groups))):
        test = groups == held_group
        train = ~test
        train_ratio = np.log(y_all[train] / x_all[train])
        graph = log_posterior_grid(train_ratio, grid, family="student_t4")
        identity = identity_posterior(train_ratio, grid)
        graph_quantiles = posterior_mixture_logratio_quantiles(
            np.repeat(grid.mu, grid.eta.size),
            np.tile(np.exp(grid.eta), grid.mu.size),
            np.exp(_grid_log_weights(grid, graph).ravel()),
            PREDICTIVE_QUANTILE_PROBABILITIES,
            family="student_t4",
        )
        identity_quantiles = posterior_mixture_logratio_quantiles(
            np.zeros_like(grid.eta),
            np.exp(grid.eta),
            np.exp(_eta_log_weights(grid, identity)),
            PREDICTIVE_QUANTILE_PROBABILITIES,
            family="student_t4",
        )
        for row_index in np.flatnonzero(test):
            base = {
                "fold": fold_index,
                "heldout_formula": held_group,
                "jvasp_id": frame.loc[row_index, "jvasp_id"],
                "material_label": frame.loc[row_index, "material_label"],
                "n_train": int(np.sum(train)),
                "observed_ev": float(y_all[row_index]),
                "opt_ev": float(x_all[row_index]),
            }
            for model, record in (
                (
                    "identity",
                    identity_predictive_record(
                        x_all[row_index],
                        y_all[row_index],
                        grid,
                        identity,
                        quantile_result=identity_quantiles,
                    ),
                ),
                (
                    "graph_primary",
                    predictive_record(
                        x_all[row_index],
                        y_all[row_index],
                        grid,
                        graph,
                        quantile_result=graph_quantiles,
                    ),
                ),
            ):
                rows.append({**base, "model": model, **record})
    return pd.DataFrame(rows)


def physical_density_jacobian_audit(frame: pd.DataFrame, grid: GridSpec) -> dict[str, float]:
    """Verify that rescaling physical eV changes log density by -log(scale)."""

    x = frame["bandgap_opt_ev"].to_numpy(float)
    y = frame["bandgap_expt_ev"].to_numpy(float)
    groups = frame["formula"].astype(str).to_numpy()
    train = groups != sorted(set(groups))[0]
    ratio = np.log(y[train] / x[train])
    graph = log_posterior_grid(ratio, grid, family="student_t4")
    identity = identity_posterior(ratio, grid)
    index = int(np.flatnonzero(~train)[0])
    scale = 7.25
    graph_quantiles = posterior_mixture_logratio_quantiles(
        np.repeat(grid.mu, grid.eta.size),
        np.tile(np.exp(grid.eta), grid.mu.size),
        np.exp(_grid_log_weights(grid, graph).ravel()),
        PREDICTIVE_QUANTILE_PROBABILITIES,
        family="student_t4",
    )
    identity_quantiles = posterior_mixture_logratio_quantiles(
        np.zeros_like(grid.eta),
        np.exp(grid.eta),
        np.exp(_eta_log_weights(grid, identity)),
        PREDICTIVE_QUANTILE_PROBABILITIES,
        family="student_t4",
    )
    graph_native = predictive_record(
        x[index], y[index], grid, graph, quantile_result=graph_quantiles
    )
    graph_scaled = predictive_record(
        scale * x[index],
        scale * y[index],
        grid,
        graph,
        quantile_result=graph_quantiles,
    )
    identity_native = identity_predictive_record(
        x[index], y[index], grid, identity, quantile_result=identity_quantiles
    )
    identity_scaled = identity_predictive_record(
        scale * x[index],
        scale * y[index],
        grid,
        identity,
        quantile_result=identity_quantiles,
    )
    graph_error = abs(
        graph_scaled["log_predictive_density_per_ev"]
        - graph_native["log_predictive_density_per_ev"]
        + math.log(scale)
    )
    identity_error = abs(
        identity_scaled["log_predictive_density_per_ev"]
        - identity_native["log_predictive_density_per_ev"]
        + math.log(scale)
    )
    return {
        "physical_scale_factor": scale,
        "expected_log_density_shift": -math.log(scale),
        "graph_absolute_error": float(graph_error),
        "identity_absolute_error": float(identity_error),
        "all_passed": bool(max(graph_error, identity_error) < 1.0e-10),
    }


def heldout_outcome_leakage_sentinel(frame: pd.DataFrame, grid: GridSpec) -> dict[str, Any]:
    """Perturb every held-out response and verify fit/predictive objects are unchanged."""

    _ = grid
    audit_grid = make_grid("quick")
    x = frame["bandgap_opt_ev"].to_numpy(float)
    y = frame["bandgap_expt_ev"].to_numpy(float)
    groups = frame["formula"].astype(str).to_numpy()
    max_graph_fit_error = 0.0
    max_identity_fit_error = 0.0
    max_predictive_quantile_error = 0.0
    checked = 0
    quantile_keys = (
        "prediction_q05_ev",
        "prediction_q25_ev",
        "prediction_median_ev",
        "prediction_q75_ev",
        "prediction_q95_ev",
    )
    graph_locations = np.repeat(audit_grid.mu, audit_grid.eta.size)
    graph_scales = np.tile(np.exp(audit_grid.eta), audit_grid.mu.size)
    identity_locations = np.zeros_like(audit_grid.eta)
    identity_scales = np.exp(audit_grid.eta)
    for held_formula in sorted(set(groups)):
        held = groups == held_formula
        train = ~held
        perturbed = y.copy()
        perturbed[held] = 3.0 * perturbed[held] + 1.7
        ratio = np.log(y[train] / x[train])
        perturbed_ratio = np.log(perturbed[train] / x[train])
        graph = log_posterior_grid(ratio, audit_grid, family="student_t4")
        graph_perturbed = log_posterior_grid(perturbed_ratio, audit_grid, family="student_t4")
        identity = identity_posterior(ratio, audit_grid)
        identity_perturbed = identity_posterior(perturbed_ratio, audit_grid)
        graph_quantiles = posterior_mixture_logratio_quantiles(
            graph_locations,
            graph_scales,
            np.exp(_grid_log_weights(audit_grid, graph).ravel()),
            PREDICTIVE_QUANTILE_PROBABILITIES,
            family="student_t4",
        )
        identity_quantiles = posterior_mixture_logratio_quantiles(
            identity_locations,
            identity_scales,
            np.exp(_eta_log_weights(audit_grid, identity)),
            PREDICTIVE_QUANTILE_PROBABILITIES,
            family="student_t4",
        )
        max_graph_fit_error = max(
            max_graph_fit_error, float(np.max(np.abs(graph - graph_perturbed)))
        )
        max_identity_fit_error = max(
            max_identity_fit_error,
            float(np.max(np.abs(identity - identity_perturbed))),
        )
        for index in np.flatnonzero(held):
            for original, changed in (
                (
                    predictive_record(
                        x[index],
                        y[index],
                        audit_grid,
                        graph,
                        quantile_result=graph_quantiles,
                    ),
                    predictive_record(
                        x[index],
                        perturbed[index],
                        audit_grid,
                        graph_perturbed,
                        quantile_result=graph_quantiles,
                    ),
                ),
                (
                    identity_predictive_record(
                        x[index],
                        y[index],
                        audit_grid,
                        identity,
                        quantile_result=identity_quantiles,
                    ),
                    identity_predictive_record(
                        x[index],
                        perturbed[index],
                        audit_grid,
                        identity_perturbed,
                        quantile_result=identity_quantiles,
                    ),
                ),
            ):
                max_predictive_quantile_error = max(
                    max_predictive_quantile_error,
                    max(abs(original[key] - changed[key]) for key in quantile_keys),
                )
            checked += 1
    result = {
        "n_heldout_rows_perturbed": int(checked),
        "graph_fit_max_abs_error": max_graph_fit_error,
        "identity_fit_max_abs_error": max_identity_fit_error,
        "predictive_quantile_max_abs_error": max_predictive_quantile_error,
    }
    result["all_passed"] = bool(
        checked == len(frame)
        and max(max_graph_fit_error, max_identity_fit_error, max_predictive_quantile_error)
        < 1.0e-12
    )
    return result


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, part in predictions.groupby("model", sort=True):
        error = part["prediction_point_ev"].to_numpy() - part["observed_ev"].to_numpy()
        rows.append(
            {
                "model": model,
                "n_heldout": int(len(part)),
                "mean_log_predictive_density_per_ev": float(
                    part["log_predictive_density_per_ev"].mean()
                ),
                "rmse_point_prediction_ev": float(np.sqrt(np.mean(error * error))),
                "mae_point_prediction_ev": float(np.mean(np.abs(error))),
                "coverage50": float(part["coverage50"].mean()),
                "coverage90": float(part["coverage90"].mean()),
                "mean_interval50_width_ev": float(part["interval50_width_ev"].mean()),
                "mean_interval90_width_ev": float(part["interval90_width_ev"].mean()),
            }
        )
    return pd.DataFrame(rows)


def predictive_group_evidence(
    predictions: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int = RESAMPLING_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return paired formula ΔELPD and its registered cluster bootstrap."""

    index = ["heldout_formula", "jvasp_id"]
    pivot = predictions.pivot(index=index, columns="model", values="log_predictive_density_per_ev")
    required = {"graph_primary", "identity"}
    if not required.issubset(pivot.columns):
        raise ValueError("graph and identity predictions are required")
    row_delta = pivot["graph_primary"] - pivot["identity"]
    contribution = row_delta.groupby(level="heldout_formula").sum().sort_index()
    values = contribution.to_numpy(float)
    total_absolute = float(np.sum(np.abs(values)))
    absolute_fraction = (
        np.abs(values) / total_absolute if total_absolute > 0.0 else np.zeros_like(values)
    )
    table = contribution.rename("delta_elpd_graph_minus_identity").reset_index()
    table["absolute_delta_elpd"] = np.abs(values)
    table["fraction_total_absolute_delta_elpd"] = absolute_fraction
    largest_index = int(np.argmax(np.abs(values)))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(int(bootstrap_replicates), len(values)))
    bootstrap = np.sum(values[indices], axis=1)
    bootstrap_table = pd.DataFrame(
        {
            "replicate": np.arange(int(bootstrap_replicates)),
            "formula_cluster_bootstrap_delta_elpd": bootstrap,
        }
    )
    summary = {
        "aggregate_delta_elpd": float(np.sum(values)),
        "mean_formula_delta_elpd": float(np.mean(values)),
        "median_formula_delta_elpd": float(np.median(values)),
        "bootstrap_q05_delta_elpd": float(np.quantile(bootstrap, 0.05)),
        "bootstrap_q95_delta_elpd": float(np.quantile(bootstrap, 0.95)),
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_seed": int(seed),
        "n_formula_clusters": int(len(values)),
        "total_absolute_formula_delta_elpd": total_absolute,
        "maximum_absolute_formula_contribution_fraction": float(absolute_fraction[largest_index]),
        "maximum_absolute_formula_contribution_formula": str(contribution.index[largest_index]),
    }
    return table, bootstrap_table, summary


def primary_support_audit(
    frame: pd.DataFrame,
    local_grid: GridSpec,
    *,
    minimum_opt_gap_ev: float = 0.0,
) -> pd.DataFrame:
    """Audit graph and identity full/LOGO posteriors on a broad grid."""

    broad = GridSpec(
        mu=np.linspace(-4.0, 4.0, 401),
        eta=np.linspace(math.log(0.003), math.log(10.0), 181),
        label="broad_support_audit",
    )
    x = frame["bandgap_opt_ev"].to_numpy(float)
    y = frame["bandgap_expt_ev"].to_numpy(float)
    groups = frame["formula"].astype(str).to_numpy()
    cases: list[tuple[str, np.ndarray]] = [("full", np.ones(len(frame), dtype=bool))]
    cases.extend((f"omit:{group}", groups != group) for group in sorted(set(groups)))
    inside_mu = (broad.mu >= local_grid.mu[0]) & (broad.mu <= local_grid.mu[-1])
    inside_eta = (broad.eta >= local_grid.eta[0]) & (broad.eta <= local_grid.eta[-1])
    inside = inside_mu[:, None] & inside_eta[None, :]
    edge_mu = max(2, int(math.ceil(0.01 * broad.mu.size)))
    edge_eta = max(2, int(math.ceil(0.01 * broad.eta.size)))
    edge_mask = np.zeros((broad.mu.size, broad.eta.size), dtype=bool)
    edge_mask[:edge_mu, :] = True
    edge_mask[-edge_mu:, :] = True
    edge_mask[:, :edge_eta] = True
    edge_mask[:, -edge_eta:] = True
    rows = []
    for case, selected in cases:
        ratio = np.log(y[selected] / x[selected])
        log_post = log_posterior_grid(ratio, broad, family="student_t4")
        weights = np.exp(_grid_log_weights(broad, log_post))
        maximum = float(np.max(log_post))
        mode = np.unravel_index(int(np.argmax(log_post)), log_post.shape)
        rows.append(
            {
                "minimum_opt_gap_ev": float(minimum_opt_gap_ev),
                "case": case,
                "model": "graph_primary",
                "n_train": int(np.sum(selected)),
                "numeric_mu_min": float(broad.mu[0]),
                "numeric_mu_max": float(broad.mu[-1]),
                "numeric_eta_min": float(broad.eta[0]),
                "numeric_eta_max": float(broad.eta[-1]),
                "analytic_mu_tail_coercive": True,
                "analytic_eta_tail_coercive": True,
                "tail_certificate": (
                    "Gaussian priors in mu and eta dominate the Student-t4 "
                    "log-likelihood's at-most logarithmic/linear tail growth"
                ),
                "broad_mass_outside_local_grid": float(
                    max(0.0, 1.0 - float(np.sum(weights[inside])))
                ),
                "broad_edge_mass": float(np.sum(weights[edge_mask])),
                "minimum_broad_boundary_profile_log_drop": float(
                    min(
                        maximum - np.max(log_post[0, :]),
                        maximum - np.max(log_post[-1, :]),
                        maximum - np.max(log_post[:, 0]),
                        maximum - np.max(log_post[:, -1]),
                    )
                ),
                "mode_mu": float(broad.mu[mode[0]]),
                "mode_eta": float(broad.eta[mode[1]]),
                "mode_inside_local_grid": bool(inside[mode]),
            }
        )

        identity_log_post = identity_posterior(ratio, broad, family="student_t4")
        identity_weights = np.exp(_eta_log_weights(broad, identity_log_post))
        identity_maximum = float(np.max(identity_log_post))
        identity_mode = int(np.argmax(identity_log_post))
        rows.append(
            {
                "minimum_opt_gap_ev": float(minimum_opt_gap_ev),
                "case": case,
                "model": "identity",
                "n_train": int(np.sum(selected)),
                "numeric_mu_min": 0.0,
                "numeric_mu_max": 0.0,
                "numeric_eta_min": float(broad.eta[0]),
                "numeric_eta_max": float(broad.eta[-1]),
                "analytic_mu_tail_coercive": True,
                "analytic_eta_tail_coercive": True,
                "tail_certificate": (
                    "identity fixes mu=0; the Gaussian eta prior dominates the "
                    "Student-t4 log-likelihood's at-most linear eta-tail growth"
                ),
                "broad_mass_outside_local_grid": float(
                    max(0.0, 1.0 - float(np.sum(identity_weights[inside_eta])))
                ),
                "broad_edge_mass": float(
                    np.sum(identity_weights[:edge_eta]) + np.sum(identity_weights[-edge_eta:])
                ),
                "minimum_broad_boundary_profile_log_drop": float(
                    min(
                        identity_maximum - identity_log_post[0],
                        identity_maximum - identity_log_post[-1],
                    )
                ),
                "mode_mu": 0.0,
                "mode_eta": float(broad.eta[identity_mode]),
                "mode_inside_local_grid": bool(inside_eta[identity_mode]),
            }
        )
    return pd.DataFrame(rows)


def threshold_support_audit(frame: pd.DataFrame, local_grid: GridSpec) -> pd.DataFrame:
    """Audit every graph/identity fit used by each fixed threshold analysis."""

    tables = []
    for threshold in (0.0, 0.25, 0.50, 1.00):
        subset = frame[frame["bandgap_opt_ev"] >= threshold].reset_index(drop=True)
        tables.append(
            primary_support_audit(
                subset,
                local_grid,
                minimum_opt_gap_ev=threshold,
            )
        )
    return pd.concat(tables, ignore_index=True)


def direct_model_ladder(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Formula-grouped classical Gaussian prediction audit in physical units.

    Student-t prediction intervals propagate the fitted direct-error variance
    and coefficient uncertainty.  This is a model-adequacy screen, separate
    from the determinant calculation.
    """

    x_all = frame["bandgap_opt_ev"].to_numpy(float)
    y_all = frame["bandgap_expt_ev"].to_numpy(float)
    groups = frame["formula"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for held_group in sorted(set(groups)):
        test = groups == held_group
        train = ~test
        x, y = x_all[train], y_all[train]
        for model in ("identity", "scale_direct", "affine_direct"):
            if model == "identity":
                fitted = x
                dfree = len(x)
                s2 = float(np.sum((y - fitted) ** 2) / dfree)
                beta = np.array([1.0])
                covariance_core = None
            elif model == "scale_direct":
                design = x[:, None]
                inverse = np.linalg.inv(design.T @ design)
                beta = inverse @ design.T @ y
                fitted = design @ beta
                dfree = len(x) - 1
                s2 = float(np.sum((y - fitted) ** 2) / dfree)
                covariance_core = inverse
            else:
                design = np.column_stack([np.ones(len(x)), x])
                inverse = np.linalg.inv(design.T @ design)
                beta = inverse @ design.T @ y
                fitted = design @ beta
                dfree = len(x) - 2
                s2 = float(np.sum((y - fitted) ** 2) / dfree)
                covariance_core = inverse
            for index in np.flatnonzero(test):
                if model == "identity":
                    location = x_all[index]
                    leverage = 0.0
                elif model == "scale_direct":
                    vector = np.array([x_all[index]])
                    location = float(vector @ beta)
                    leverage = float(vector @ covariance_core @ vector)
                else:
                    vector = np.array([1.0, x_all[index]])
                    location = float(vector @ beta)
                    leverage = float(vector @ covariance_core @ vector)
                scale = math.sqrt(s2 * (1.0 + leverage))
                q05, q25, q75, q95 = student_t.ppf(
                    [0.05, 0.25, 0.75, 0.95], df=dfree, loc=location, scale=scale
                )
                rows.append(
                    {
                        "heldout_formula": held_group,
                        "jvasp_id": frame.loc[index, "jvasp_id"],
                        "model": model,
                        "observed_ev": y_all[index],
                        "prediction_ev": location,
                        "prediction_q05_ev": q05,
                        "prediction_q25_ev": q25,
                        "prediction_q75_ev": q75,
                        "prediction_q95_ev": q95,
                        "coverage50": float(q25 <= y_all[index] <= q75),
                        "coverage90": float(q05 <= y_all[index] <= q95),
                        "log_predictive_density_per_ev": float(
                            student_t.logpdf(y_all[index], df=dfree, loc=location, scale=scale)
                        ),
                    }
                )
    predictions = pd.DataFrame(rows)
    summaries = []
    for model, part in predictions.groupby("model", sort=True):
        error = part["prediction_ev"] - part["observed_ev"]
        summaries.append(
            {
                "model": model,
                "n_heldout": int(len(part)),
                "rmse_ev": float(np.sqrt(np.mean(error * error))),
                "mae_ev": float(np.mean(np.abs(error))),
                "mean_log_predictive_density_per_ev": float(
                    part["log_predictive_density_per_ev"].mean()
                ),
                "coverage50": float(part["coverage50"].mean()),
                "coverage90": float(part["coverage90"].mean()),
            }
        )
    return predictions, pd.DataFrame(summaries)


def influence_table(frame: pd.DataFrame, grid: GridSpec, full: dict[str, float]) -> pd.DataFrame:
    x = frame["bandgap_opt_ev"].to_numpy(float)
    y = frame["bandgap_expt_ev"].to_numpy(float)
    groups = frame["formula"].astype(str).to_numpy()
    log_ratio = np.log(y / x)
    full_post = log_posterior_grid(log_ratio, grid, family="student_t4")
    weights = np.exp(_grid_log_weights(grid, full_post).ravel())
    mu_mean = float(np.sum(weights * np.repeat(grid.mu, grid.eta.size)))
    tau_mean = float(np.sum(weights * np.tile(np.exp(grid.eta), grid.mu.size)))
    shifts: dict[str, float] = {}
    for group in sorted(set(groups)):
        subset = groups != group
        omitted = posterior_summary(
            grid, log_posterior_grid(log_ratio[subset], grid, family="student_t4")
        )
        shifts[group] = (omitted["gamma_mean"] - full["gamma_mean"]) / full["gamma_sd"]
    result = frame.loc[
        :, ["material_label", "formula", "jvasp_id", "bandgap_opt_ev", "bandgap_expt_ev"]
    ].copy()
    result["log_expt_over_opt"] = log_ratio
    result["fitted_median_ev"] = x * math.exp(mu_mean)
    result["log_residual"] = log_ratio - mu_mean
    result["standardized_log_residual"] = result["log_residual"] / tau_mean
    result["omit_formula_shift_graph_sd"] = [shifts[g] for g in groups]
    return result


def permutation_control(
    frame: pd.DataFrame, permutations: int, seed: int = 20260711
) -> tuple[pd.DataFrame, dict[str, float]]:
    # Group means make each formula the exchangeable unit; this is a diagnostic,
    # not a formal randomization test for heterogeneous materials.
    grouped = frame.groupby("formula", sort=True).agg(
        x=("bandgap_opt_ev", lambda z: float(np.exp(np.mean(np.log(z))))),
        y=("bandgap_expt_ev", lambda z: float(np.exp(np.mean(np.log(z))))),
    )
    lx = np.log(grouped["x"].to_numpy())
    ly = np.log(grouped["y"].to_numpy())

    def logo_rmse(response: np.ndarray) -> float:
        predictions = np.empty_like(response)
        for index in range(len(response)):
            train = np.arange(len(response)) != index
            mu_hat = float(np.mean(response[train] - lx[train]))
            predictions[index] = lx[index] + mu_hat
        return float(np.sqrt(np.mean((predictions - response) ** 2)))

    observed = logo_rmse(ly)
    rng = np.random.default_rng(seed)
    values = np.array([logo_rmse(rng.permutation(ly)) for _ in range(permutations)])
    table = pd.DataFrame(
        {
            "permutation_index": np.arange(permutations),
            "logo_log_rmse": values,
            "observed_logo_log_rmse": observed,
            "is_at_or_below_observed": values <= observed,
            "seed": int(seed),
        }
    )
    summary = {
        "n_formula_groups": int(len(grouped)),
        "n_permutations": int(permutations),
        "seed": int(seed),
        "observed_logo_log_rmse": observed,
        "permutation_q05": float(np.quantile(values, 0.05)),
        "permutation_median": float(np.median(values)),
        "permutation_q95": float(np.quantile(values, 0.95)),
        "lower_tail_rank_fraction_with_plus_one": float(
            (1 + np.sum(values <= observed)) / (permutations + 1)
        ),
    }
    return table, summary


def rejected_reciprocal_diagnostic(frame: pd.DataFrame) -> dict[str, Any]:
    x = frame["bandgap_opt_ev"].to_numpy(float)
    y = frame["bandgap_expt_ev"].to_numpy(float)
    mu = np.linspace(-120.0, 20.0, 28001)
    graph = direct_fixed_log_posterior(
        x,
        y,
        mu,
        floor_ev=0.12,
        fractional=0.15,
        family="normal",
        mu_sd=0.75,
    )
    # For c=1/u-gamma*x, |D_u c|=(gamma*x)^2 on the graph.
    # The independently based Lebesgue-du naive hard target is therefore
    # graph/J, or graph-2*n*mu up to a mu-independent constant.
    formal_naive_hard = graph - 2.0 * len(x) * mu
    weights = normalize_log_weights(formal_naive_hard)
    mode_index = int(np.argmax(formal_naive_hard))
    maximum = float(np.max(formal_naive_hard))
    local = (mu >= -1.6) & (mu <= 1.6)
    edge = max(2, int(math.ceil(0.01 * len(mu))))
    remote_minus_local = float(np.max(formal_naive_hard[~local]) - np.max(formal_naive_hard[local]))
    return {
        "status": "rejected_not_an_active_target",
        "construction": "u=E0/E; c=1/u-gamma*x under Lebesgue du",
        "finite_penalty_integral": (
            "diverges because as u tends to infinity, E0/u tends to zero, the "
            "OPT-conditioned direct likelihood tends to a positive finite value, "
            "and c tends to -gamma*x; the integrand therefore tends to a positive "
            "constant for every finite gamma and rho"
        ),
        "finite_tube_proper": False,
        "hard_limit_mass_behavior": (
            "remote under the extension geometry likelihood/prior; this formal "
            "hard-target location is model-dependent and cannot rescue a route "
            "whose finite-rho defining integral is improper"
        ),
        "formal_naive_hard_logdensity_formula": (
            "geometry_graph_logdensity - 2*n*mu + constant independent of mu"
        ),
        "formal_naive_hard_global_mu_min": float(mu[0]),
        "formal_naive_hard_global_mu_max": float(mu[-1]),
        "formal_naive_hard_n_grid_points": int(len(mu)),
        "formal_naive_hard_mode_mu": float(mu[mode_index]),
        "formal_naive_hard_mode_gamma": float(math.exp(mu[mode_index])),
        "formal_naive_hard_local_window_mass": float(np.sum(weights[local])),
        "formal_naive_hard_remote_minus_local_max_logdensity": remote_minus_local,
        "formal_naive_hard_lower_endpoint_log_drop": maximum - float(formal_naive_hard[0]),
        "formal_naive_hard_upper_endpoint_log_drop": maximum - float(formal_naive_hard[-1]),
        "formal_naive_hard_lower_edge_mass_1pct": float(np.sum(weights[:edge])),
        "formal_naive_hard_upper_edge_mass_1pct": float(np.sum(weights[-edge:])),
        "formal_naive_hard_computed_globally": True,
        "n_positive_reference_inputs": int(np.sum(x > 0.0)),
        "all_reference_inputs_finite_positive": bool(np.all(np.isfinite(x)) and np.all(x > 0.0)),
    }


def sensitivity_table(frame: pd.DataFrame, grid: GridSpec) -> pd.DataFrame:
    x = frame["bandgap_opt_ev"].to_numpy(float)
    y = frame["bandgap_expt_ev"].to_numpy(float)
    ratio = np.log(y / x)
    rows = []
    for name, family, mu_sd in (
        ("primary_robust_log_student_t4", "student_t4", 0.75),
        ("gaussian_log_sensitivity", "normal", 0.75),
        ("narrower_mu_prior", "student_t4", 0.40),
        ("wider_mu_prior", "student_t4", 1.00),
    ):
        result = posterior_summary(
            grid, log_posterior_grid(ratio, grid, family=family, mu_sd=mu_sd)
        )
        rows.append(
            {
                "analysis": name,
                "likelihood_coordinate": "log ratio",
                "likelihood_family": family,
                "mu_prior_sd": mu_sd,
                "physical_sigma_floor": float("nan"),
                "physical_sigma_fraction": float("nan"),
                **result,
            }
        )
    for name, family, floor, fraction in (
        ("direct_floor012_frac015", "normal", 0.12, 0.15),
        ("direct_robust_floor012_frac015", "student_t4", 0.12, 0.15),
        ("direct_floor025_frac020", "normal", 0.25, 0.20),
    ):
        log_post = direct_fixed_log_posterior(
            x, y, grid.mu, floor_ev=floor, fractional=fraction, family=family
        )
        rows.append(
            {
                "analysis": name,
                "likelihood_coordinate": "eV; sigma depends on OPT only",
                "likelihood_family": family,
                "mu_prior_sd": 0.75,
                "physical_sigma_floor": floor,
                "physical_sigma_fraction": fraction,
                "tau_log_ratio_mean": float("nan"),
                **summary_one_dim(grid.mu, log_post),
            }
        )
    return pd.DataFrame(rows)


def domain_sensitivity_table(frame: pd.DataFrame, grid: GridSpec) -> pd.DataFrame:
    """Protocol-locked retrospective OPT-gap threshold sensitivity subsets."""

    rows = []
    for threshold in (0.0, 0.25, 0.50, 1.00):
        subset = frame[frame["bandgap_opt_ev"] >= threshold]
        ratio = np.log(subset["bandgap_expt_ev"].to_numpy() / subset["bandgap_opt_ev"].to_numpy())
        result = posterior_summary(grid, log_posterior_grid(ratio, grid, family="student_t4"))
        rows.append(
            {
                "minimum_opt_gap_ev": threshold,
                "n_rows": int(len(subset)),
                "n_formula_groups": int(subset["formula"].nunique()),
                **result,
            }
        )
    return pd.DataFrame(rows)


def domain_predictive_sensitivity(
    frame: pd.DataFrame,
    grid: GridSpec,
    *,
    bootstrap_replicates: int,
    full_predictions: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the complete graph-vs-identity LOGO protocol at each fixed threshold."""

    prediction_tables = []
    score_tables = []
    contribution_tables = []
    bootstrap_tables = []
    evidence_rows = []
    for threshold in (0.0, 0.25, 0.50, 1.00):
        subset = frame[frame["bandgap_opt_ev"] >= threshold].reset_index(drop=True)
        if threshold == 0.0 and full_predictions is not None:
            predictions = full_predictions.copy()
        else:
            predictions = grouped_log_scale_predictions(subset, grid)
        scores = summarize_predictions(predictions)
        contributions, bootstrap, evidence = predictive_group_evidence(
            predictions,
            bootstrap_replicates=bootstrap_replicates,
            seed=RESAMPLING_SEED,
        )
        predictions.insert(0, "minimum_opt_gap_ev", threshold)
        scores.insert(0, "minimum_opt_gap_ev", threshold)
        contributions.insert(0, "minimum_opt_gap_ev", threshold)
        bootstrap.insert(0, "minimum_opt_gap_ev", threshold)
        prediction_tables.append(predictions)
        score_tables.append(scores)
        contribution_tables.append(contributions)
        bootstrap_tables.append(bootstrap)
        evidence_rows.append(
            {
                "minimum_opt_gap_ev": threshold,
                "n_rows": int(len(subset)),
                "n_formula_groups": int(subset["formula"].nunique()),
                **evidence,
            }
        )
    return {
        "predictions": pd.concat(prediction_tables, ignore_index=True),
        "scores": pd.concat(score_tables, ignore_index=True),
        "contributions": pd.concat(contribution_tables, ignore_index=True),
        "bootstrap": pd.concat(bootstrap_tables, ignore_index=True),
        "evidence": pd.DataFrame(evidence_rows),
    }


def threshold_role_audit(
    frame: pd.DataFrame,
    *,
    protocol_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Emit row-level train/held-out roles for every fixed threshold."""

    manifests = []
    validations = []
    for threshold in (0.0, 0.25, 0.50, 1.00):
        subset = frame[frame["bandgap_opt_ev"] >= threshold].reset_index(drop=True)
        threshold_config = {
            **protocol_config,
            "role_manifest_scope": "threshold_specific_formula_logo",
            "minimum_opt_gap_ev": threshold,
        }
        manifest, validation = logo_role_manifest(
            subset,
            protocol_config=threshold_config,
        )
        manifest.insert(0, "minimum_opt_gap_ev", threshold)
        manifests.append(manifest)
        validations.append(
            {
                "minimum_opt_gap_ev": threshold,
                **validation,
            }
        )
    return pd.concat(manifests, ignore_index=True), pd.DataFrame(validations)


def threshold_leakage_audit(frame: pd.DataFrame, grid: GridSpec) -> pd.DataFrame:
    """Run the held-out-outcome sentinel in every nested threshold subset."""

    rows = []
    for threshold in (0.0, 0.25, 0.50, 1.00):
        subset = frame[frame["bandgap_opt_ev"] >= threshold].reset_index(drop=True)
        rows.append(
            {
                "minimum_opt_gap_ev": threshold,
                "n_rows": int(len(subset)),
                "n_formula_groups": int(subset["formula"].nunique()),
                **heldout_outcome_leakage_sentinel(subset, grid),
            }
        )
    return pd.DataFrame(rows)


def grid_resolution_audit(
    frame: pd.DataFrame,
    active_grid: GridSpec,
    active_predictions: pd.DataFrame,
    active_graph: dict[str, float],
    active_domain_sensitivity: pd.DataFrame,
    active_domain_predictive: dict[str, pd.DataFrame],
    *,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    """Compare primary and threshold endpoints on adjacent grid resolutions."""

    comparison_mode = "standard" if active_grid.label in {"quick", "full"} else "quick"
    comparison_grid = make_grid(comparison_mode)
    comparison_predictions = grouped_log_scale_predictions(frame, comparison_grid)
    x = frame["bandgap_opt_ev"].to_numpy(float)
    y = frame["bandgap_expt_ev"].to_numpy(float)
    comparison_graph = posterior_summary(
        comparison_grid,
        log_posterior_grid(np.log(y / x), comparison_grid, family="student_t4"),
    )
    comparison_domain_sensitivity = domain_sensitivity_table(frame, comparison_grid)
    comparison_domain_predictive = domain_predictive_sensitivity(
        frame,
        comparison_grid,
        bootstrap_replicates=bootstrap_replicates,
        full_predictions=comparison_predictions,
    )
    quantile_columns = (
        "prediction_q05_ev",
        "prediction_q25_ev",
        "prediction_median_ev",
        "prediction_q75_ev",
        "prediction_q95_ev",
    )
    primary_details, primary_endpoints = predictive_grid_differences(
        active_predictions,
        comparison_predictions,
        key_columns=("fold", "heldout_formula", "jvasp_id", "model"),
        input_column="opt_ev",
        density_column="log_predictive_density_per_ev",
        quantile_columns=quantile_columns,
    )
    threshold_details, threshold_endpoints = predictive_grid_differences(
        active_domain_predictive["predictions"],
        comparison_domain_predictive["predictions"],
        key_columns=(
            "minimum_opt_gap_ev",
            "fold",
            "heldout_formula",
            "jvasp_id",
            "model",
        ),
        input_column="opt_ev",
        density_column="log_predictive_density_per_ev",
        quantile_columns=quantile_columns,
    )

    def aggregate_delta(predictions: pd.DataFrame) -> float:
        pivot = predictions.pivot(
            index=["heldout_formula", "jvasp_id"],
            columns="model",
            values="log_predictive_density_per_ev",
        )
        return float(np.sum(pivot["graph_primary"] - pivot["identity"]))

    active_threshold_evidence = active_domain_predictive["evidence"].set_index("minimum_opt_gap_ev")
    comparison_threshold_evidence = comparison_domain_predictive["evidence"].set_index(
        "minimum_opt_gap_ev"
    )
    active_threshold_posterior = active_domain_sensitivity.set_index("minimum_opt_gap_ev")
    comparison_threshold_posterior = comparison_domain_sensitivity.set_index("minimum_opt_gap_ev")
    stats: dict[str, Any] = {
        "active_grid_mode": active_grid.label,
        "active_grid_shape": [int(active_grid.mu.size), int(active_grid.eta.size)],
        "comparison_grid_mode": comparison_mode,
        "comparison_grid_shape": [
            int(comparison_grid.mu.size),
            int(comparison_grid.eta.size),
        ],
        "posterior_gamma_mean_abs_difference": abs(
            active_graph["gamma_mean"] - comparison_graph["gamma_mean"]
        ),
        "posterior_tau_mean_abs_difference": abs(
            active_graph["tau_log_ratio_mean"] - comparison_graph["tau_log_ratio_mean"]
        ),
        "posterior_gamma_quantile_max_abs_difference": max(
            abs(active_graph[key] - comparison_graph[key])
            for key in ("gamma_q05", "gamma_q50", "gamma_q95")
        ),
        "posterior_log_gamma_quantile_max_abs_difference": max(
            abs(math.log(active_graph[key]) - math.log(comparison_graph[key]))
            for key in ("gamma_q05", "gamma_q50", "gamma_q95")
        ),
        "aggregate_delta_elpd_abs_difference": abs(
            aggregate_delta(active_predictions) - aggregate_delta(comparison_predictions)
        ),
        "threshold_gamma_mean_max_abs_difference": float(
            np.max(
                np.abs(
                    active_threshold_posterior["gamma_mean"]
                    - comparison_threshold_posterior["gamma_mean"]
                )
            )
        ),
        "threshold_gamma_quantile_max_abs_difference": float(
            max(
                np.max(
                    np.abs(active_threshold_posterior[key] - comparison_threshold_posterior[key])
                )
                for key in ("gamma_q05", "gamma_q50", "gamma_q95")
            )
        ),
        "threshold_log_gamma_quantile_max_abs_difference": float(
            max(
                np.max(
                    np.abs(
                        np.log(active_threshold_posterior[key])
                        - np.log(comparison_threshold_posterior[key])
                    )
                )
                for key in ("gamma_q05", "gamma_q50", "gamma_q95")
            )
        ),
        "threshold_aggregate_delta_elpd_max_abs_difference": float(
            np.max(
                np.abs(
                    active_threshold_evidence["aggregate_delta_elpd"]
                    - comparison_threshold_evidence["aggregate_delta_elpd"]
                )
            )
        ),
        "primary_endpoints": primary_endpoints,
        "threshold_endpoints": threshold_endpoints,
    }
    posterior_quantile_tolerance = 0.001 if active_grid.label == "full" else 0.003
    stats["posterior_log_gamma_quantile_tolerance"] = posterior_quantile_tolerance
    gates = {
        "posterior_gamma_mean_abs_difference_below_1e_minus_10": bool(
            stats["posterior_gamma_mean_abs_difference"] < 1.0e-10
        ),
        "posterior_tau_mean_abs_difference_below_1e_minus_10": bool(
            stats["posterior_tau_mean_abs_difference"] < 1.0e-10
        ),
        "posterior_log_gamma_quantile_refinement_within_declared_tolerance": bool(
            stats["posterior_log_gamma_quantile_max_abs_difference"] < posterior_quantile_tolerance
        ),
        "aggregate_delta_elpd_abs_difference_below_1e_minus_10": bool(
            stats["aggregate_delta_elpd_abs_difference"] < 1.0e-10
        ),
        "primary_row_log_density_abs_difference_below_1e_minus_10": bool(
            primary_endpoints["max_abs_log_predictive_density_difference"] < 1.0e-10
        ),
        "primary_logratio_quantile_abs_difference_below_1e_minus_10": bool(
            primary_endpoints["max_abs_logratio_quantile_difference"] < 1.0e-10
        ),
        "primary_coverage_decisions_identical": bool(
            primary_endpoints["coverage_decision_mismatch_count"] == 0
        ),
        "threshold_gamma_mean_max_abs_difference_below_1e_minus_10": bool(
            stats["threshold_gamma_mean_max_abs_difference"] < 1.0e-10
        ),
        "threshold_log_gamma_quantile_refinement_within_declared_tolerance": bool(
            stats["threshold_log_gamma_quantile_max_abs_difference"] < posterior_quantile_tolerance
        ),
        "threshold_delta_elpd_max_abs_difference_below_1e_minus_10": bool(
            stats["threshold_aggregate_delta_elpd_max_abs_difference"] < 1.0e-10
        ),
        "threshold_row_log_density_abs_difference_below_1e_minus_10": bool(
            threshold_endpoints["max_abs_log_predictive_density_difference"] < 1.0e-10
        ),
        "threshold_logratio_quantile_abs_difference_below_1e_minus_10": bool(
            threshold_endpoints["max_abs_logratio_quantile_difference"] < 1.0e-10
        ),
        "threshold_coverage_decisions_identical": bool(
            threshold_endpoints["coverage_decision_mismatch_count"] == 0
        ),
        "all_mixture_cdf_routes_are_exact": bool(
            primary_endpoints["all_quantile_routes_exact"]
            and threshold_endpoints["all_quantile_routes_exact"]
        ),
        "mixture_cdf_root_residual_below_1e_10": bool(
            max(
                primary_endpoints["max_abs_cdf_root_residual"],
                threshold_endpoints["max_abs_cdf_root_residual"],
            )
            < 1.0e-10
        ),
    }
    stats["gates"] = gates
    stats["all_passed"] = bool(all(gates.values()))
    gate_table = pd.DataFrame([{"gate": name, "passed": passed} for name, passed in gates.items()])
    return {
        "primary_details": primary_details,
        "threshold_details": threshold_details,
        "gate_table": gate_table,
        "stats": stats,
    }


def evaluate_claims(
    predictive_scores: pd.DataFrame,
    predictive_evidence: dict[str, Any],
    direct_scores: pd.DataFrame,
    domain_sensitivity: pd.DataFrame,
    domain_predictive: dict[str, pd.DataFrame],
    influence: pd.DataFrame,
    permutation_summary: dict[str, float],
    support_audit: pd.DataFrame,
    threshold_support: pd.DataFrame,
    geometry_audit: dict[str, Any],
    api_audit: dict[str, Any],
    rejected_reciprocal: dict[str, Any],
    role_validation: dict[str, Any],
    threshold_role_validation: pd.DataFrame,
    density_jacobian: dict[str, float],
    leakage_sentinel: dict[str, Any],
    threshold_leakage: pd.DataFrame,
    grid_refinement: dict[str, Any],
) -> list[dict[str, Any]]:
    scores = predictive_scores.set_index("model")
    graph = scores.loc["graph_primary"]
    identity = scores.loc["identity"]
    direct = direct_scores.set_index("model")
    predictive_gates = {
        "median_point_rmse_improves": bool(
            graph["rmse_point_prediction_ev"] < identity["rmse_point_prediction_ev"]
        ),
        "mean_log_score_improves": bool(
            graph["mean_log_predictive_density_per_ev"]
            > identity["mean_log_predictive_density_per_ev"]
        ),
        "aggregate_delta_elpd_positive": predictive_evidence["aggregate_delta_elpd"] > 0.0,
        "registered_formula_bootstrap_q05_delta_elpd_positive": (
            predictive_evidence["bootstrap_q05_delta_elpd"] > 0.0
        ),
        "median_formula_delta_elpd_positive": predictive_evidence["median_formula_delta_elpd"]
        > 0.0,
        "median_point_mae_improves": bool(
            graph["mae_point_prediction_ev"] < identity["mae_point_prediction_ev"]
        ),
        "coverage_not_degraded_by_more_than_0p05": bool(
            graph["coverage90"] >= identity["coverage90"] - 0.05
        ),
        "coverage50_90_and_widths_are_emitted": bool(
            np.isfinite(
                graph[
                    [
                        "coverage50",
                        "coverage90",
                        "mean_interval50_width_ev",
                        "mean_interval90_width_ev",
                    ]
                ]
            ).all()
            and np.isfinite(
                identity[
                    [
                        "coverage50",
                        "coverage90",
                        "mean_interval50_width_ev",
                        "mean_interval90_width_ev",
                    ]
                ]
            ).all()
        ),
        "maximum_group_influence_below_one_graph_sd": bool(
            influence["omit_formula_shift_graph_sd"].abs().max() < 1.0
        ),
        "pairing_negative_control_lower_tail_rank_at_most_0p05": bool(
            permutation_summary["lower_tail_rank_fraction_with_plus_one"] <= 0.05
        ),
        "affine_screen_does_not_beat_scale": bool(
            direct.loc["affine_direct", "rmse_ev"] >= direct.loc["scale_direct", "rmse_ev"]
        ),
        "direct_scale_screen_rmse_improves_identity": bool(
            direct.loc["scale_direct", "rmse_ev"] < direct.loc["identity", "rmse_ev"]
        ),
        "all_primary_full_and_logo_support_audits_pass": bool(
            support_audit["mode_inside_local_grid"].all()
            and support_audit["analytic_mu_tail_coercive"].all()
            and support_audit["analytic_eta_tail_coercive"].all()
            and support_audit["broad_mass_outside_local_grid"].max() < 1.0e-4
            and support_audit["broad_edge_mass"].max() < 1.0e-10
            and support_audit["minimum_broad_boundary_profile_log_drop"].min() > 30.0
        ),
        "logo_role_manifest_has_no_leakage": bool(role_validation["all_passed"]),
        "heldout_outcome_leakage_sentinel_passes": bool(leakage_sentinel["all_passed"]),
        "physical_density_jacobian_passes": bool(density_jacobian["all_passed"]),
        "predictive_grid_refinement_passes": bool(grid_refinement["stats"]["all_passed"]),
    }
    predictive_core = all(
        predictive_gates[key]
        for key in (
            "median_point_rmse_improves",
            "median_point_mae_improves",
            "mean_log_score_improves",
            "aggregate_delta_elpd_positive",
            "registered_formula_bootstrap_q05_delta_elpd_positive",
            "median_formula_delta_elpd_positive",
            "direct_scale_screen_rmse_improves_identity",
            "coverage_not_degraded_by_more_than_0p05",
            "maximum_group_influence_below_one_graph_sd",
            "pairing_negative_control_lower_tail_rank_at_most_0p05",
            "all_primary_full_and_logo_support_audits_pass",
            "logo_role_manifest_has_no_leakage",
            "heldout_outcome_leakage_sentinel_passes",
            "coverage50_90_and_widths_are_emitted",
            "physical_density_jacobian_passes",
            "predictive_grid_refinement_passes",
        )
    )
    predictive_implementation_ready = all(
        predictive_gates[key]
        for key in (
            "coverage50_90_and_widths_are_emitted",
            "all_primary_full_and_logo_support_audits_pass",
            "logo_role_manifest_has_no_leakage",
            "heldout_outcome_leakage_sentinel_passes",
            "physical_density_jacobian_passes",
            "predictive_grid_refinement_passes",
        )
    )
    geometry_stats = geometry_audit["stats"]
    geometry_gates = {
        "complete_finite_pushforwards_agree_at_25_points": bool(
            geometry_stats["n_parameter_points"] >= 25
            and geometry_stats["direct_vs_log_centered_max_abs_error"] < 1.0e-8
            and geometry_stats["reciprocal_vs_log_centered_max_abs_error"] < 1.0e-8
        ),
        "all_corrected_hard_routes_match_graph": bool(
            geometry_stats["corrected_hard_vs_graph_max_abs_error"] < 1.0e-10
        ),
        "residual_metric_transform_is_pointwise_invariant": bool(
            geometry_stats["residual_metric_penalty_max_abs_error"] < 1.0e-12
            and geometry_stats["residual_metric_log_j_max_abs_error"] < 1.0e-12
        ),
        "deliberate_double_counting_control_fails": bool(
            geometry_stats["double_counting_failure_centered_max_abs_error"] > 1.0e-3
        ),
        "global_support_and_tail_gates_pass": bool(
            geometry_stats["global_support_all_endpoint_log_drops_above_30"]
            and geometry_stats["global_support_all_edge_masses_below_1e_10"]
        ),
        "detcorr_api_absolute_error_below_1e_10": api_audit["absolute_error"] < 1.0e-10,
        "rejected_reciprocal_finite_tube_is_identified_as_divergent": (
            rejected_reciprocal["status"] == "rejected_not_an_active_target"
            and rejected_reciprocal["finite_tube_proper"] is False
        ),
        "rejected_reciprocal_formal_hard_target_is_computed_not_asserted": bool(
            rejected_reciprocal["formal_naive_hard_computed_globally"]
            and rejected_reciprocal["formal_naive_hard_n_grid_points"] >= 28001
            and math.isfinite(rejected_reciprocal["formal_naive_hard_mode_mu"])
            and math.isfinite(rejected_reciprocal["formal_naive_hard_mode_gamma"])
            and math.isfinite(rejected_reciprocal["formal_naive_hard_local_window_mass"])
            and math.isfinite(
                rejected_reciprocal["formal_naive_hard_remote_minus_local_max_logdensity"]
            )
            and rejected_reciprocal["formal_naive_hard_lower_endpoint_log_drop"] > 30.0
            and rejected_reciprocal["formal_naive_hard_upper_endpoint_log_drop"] > 30.0
            and rejected_reciprocal["formal_naive_hard_lower_edge_mass_1pct"] < 1.0e-10
            and rejected_reciprocal["formal_naive_hard_upper_edge_mass_1pct"] < 1.0e-10
        ),
        "rejected_reciprocal_inputs_are_finite_positive": bool(
            rejected_reciprocal["all_reference_inputs_finite_positive"]
        ),
    }
    gamma_range = float(
        domain_sensitivity["gamma_mean"].max() - domain_sensitivity["gamma_mean"].min()
    )
    domain_scores = domain_predictive["scores"]
    domain_evidence = domain_predictive["evidence"]
    thresholds = [0.0, 0.25, 0.5, 1.0]
    delta = domain_evidence.set_index("minimum_opt_gap_ev")["aggregate_delta_elpd"]
    full_sign = bool(delta.loc[0.0] > 0.0)
    ranking_consistent = bool(np.all((delta > 0.0).to_numpy() == full_sign))
    if full_sign:
        bootstrap_sign_supported = bool((domain_evidence["bootstrap_q05_delta_elpd"] > 0.0).all())
    else:
        bootstrap_sign_supported = bool((domain_evidence["bootstrap_q95_delta_elpd"] < 0.0).all())
    domain_gates = {
        "exact_protocol_locked_thresholds_evaluated": bool(
            sorted(domain_evidence["minimum_opt_gap_ev"].tolist()) == thresholds
        ),
        "each_threshold_has_graph_and_identity_scores": bool(
            domain_scores.groupby("minimum_opt_gap_ev")["model"]
            .apply(set)
            .apply(lambda value: value == {"graph_primary", "identity"})
            .all()
        ),
        "each_threshold_reports_locked_bootstrap": bool(
            domain_evidence["bootstrap_seed"].eq(RESAMPLING_SEED).all()
            and domain_evidence["bootstrap_replicates"]
            .eq(predictive_evidence["bootstrap_replicates"])
            .all()
        ),
        "each_threshold_reports_rmse_mae_coverage50_90_and_widths": bool(
            np.isfinite(
                domain_scores[
                    [
                        "rmse_point_prediction_ev",
                        "mae_point_prediction_ev",
                        "coverage50",
                        "coverage90",
                        "mean_interval50_width_ev",
                        "mean_interval90_width_ev",
                    ]
                ].to_numpy(float)
            ).all()
        ),
        "threshold_posterior_scale_range_is_reported": bool(np.isfinite(gamma_range)),
        "every_threshold_graph_and_identity_support_audit_passes": bool(
            threshold_support["mode_inside_local_grid"].all()
            and threshold_support["analytic_mu_tail_coercive"].all()
            and threshold_support["analytic_eta_tail_coercive"].all()
            and threshold_support["broad_mass_outside_local_grid"].max() < 1.0e-4
            and threshold_support["broad_edge_mass"].max() < 1.0e-10
            and threshold_support["minimum_broad_boundary_profile_log_drop"].min() > 30.0
        ),
        "every_threshold_role_manifest_passes": bool(
            len(threshold_role_validation) == 4 and threshold_role_validation["all_passed"].all()
        ),
        "every_threshold_heldout_outcome_leakage_sentinel_passes": bool(
            len(threshold_leakage) == 4 and threshold_leakage["all_passed"].all()
        ),
        "physical_density_jacobian_passes": bool(density_jacobian["all_passed"]),
        "primary_and_threshold_grid_refinement_passes": bool(
            grid_refinement["stats"]["all_passed"]
        ),
        "predictive_ranking_consistent_with_full_fixture": ranking_consistent,
        "every_threshold_bootstrap_interval_supports_full_fixture_sign": (bootstrap_sign_supported),
    }
    domain_structural = all(
        value
        for key, value in domain_gates.items()
        if key
        not in {
            "predictive_ranking_consistent_with_full_fixture",
            "every_threshold_bootstrap_interval_supports_full_fixture_sign",
        }
    )
    substantial_case_concentration = bool(
        predictive_evidence["maximum_absolute_formula_contribution_fraction"] > 0.5
    )
    if not predictive_implementation_ready:
        predictive_status = "BLOCKED"
        predictive_qualification = "Implementation or support gates did not all pass."
    elif predictive_evidence["aggregate_delta_elpd"] <= 0.0:
        predictive_status = "NOT_SUPPORTED"
        predictive_qualification = "Aggregate formula-LOGO delta ELPD is non-positive."
    elif predictive_core and predictive_gates["affine_screen_does_not_beat_scale"]:
        predictive_status = "SUPPORTED"
        predictive_qualification = "All declared predictive and model-screen gates passed."
    else:
        predictive_status = "MIXED"
        predictive_qualification = (
            "Aggregate delta ELPD is positive, but the affine screen or another "
            "declared scientific gate qualifies the common-scale interpretation."
        )
    if predictive_status == "SUPPORTED" and substantial_case_concentration:
        predictive_qualification += " Supported with substantial case concentration."
    predictive_claim = (
        "A robust common scale improves formula-grouped positive-gap "
        "prediction over identity on this fixture."
    )
    if predictive_status == "SUPPORTED" and substantial_case_concentration:
        predictive_claim = (
            predictive_claim.removesuffix(".") + ", with substantial case concentration."
        )
    affine_screen_phrase = (
        "does not" if predictive_gates["affine_screen_does_not_beat_scale"] else "does"
    )
    if not domain_structural:
        domain_status = "BLOCKED"
        domain_claim = "Threshold-specific predictive stability could not be evaluated."
        domain_qualification = "One or more required threshold outputs or gates is missing."
    elif ranking_consistent and bootstrap_sign_supported:
        domain_status = "SUPPORTED"
        domain_claim = (
            "The graph-versus-identity formula-LOGO ranking is stable across four "
            "protocol-locked retrospective nested calculated-gap subsets, with "
            "bootstrap intervals supporting the same direction."
        )
        domain_qualification = (
            "All four nested subsets preserve the full-fixture aggregate-score sign, "
            "and each 90% bootstrap interval excludes zero in that direction; this "
            "is within-fixture subset sensitivity, not independent-domain replication."
        )
    elif not ranking_consistent:
        domain_status = "MIXED"
        domain_claim = (
            "The graph-versus-identity formula-LOGO ranking changes across the "
            "four retrospective nested calculated-gap subsets on this fixture."
        )
        domain_qualification = (
            "At least one threshold reverses the full-fixture aggregate-score sign, "
            "so the predictive interpretation is low-gap-threshold dependent."
        )
    else:
        domain_status = "MIXED"
        domain_claim = (
            "The point graph-versus-identity formula-LOGO ranking has a consistent "
            "direction across the four fixed calculated-gap thresholds, but its "
            "locked bootstrap support is not stable."
        )
        domain_qualification = (
            "At least one threshold's locked 90% bootstrap interval includes zero, "
            "so point-sign consistency alone is insufficient for a stability claim."
        )
    return [
        {
            "claim_id": "JG-PRED",
            "status": predictive_status,
            "claim": predictive_claim,
            "qualification": predictive_qualification,
            "primary_result": {
                "aggregate_delta_elpd_graph_minus_identity": predictive_evidence[
                    "aggregate_delta_elpd"
                ],
                "formula_bootstrap_q05": predictive_evidence["bootstrap_q05_delta_elpd"],
                "formula_bootstrap_q95": predictive_evidence["bootstrap_q95_delta_elpd"],
                "maximum_absolute_formula_contribution_fraction": predictive_evidence[
                    "maximum_absolute_formula_contribution_fraction"
                ],
                "maximum_absolute_formula_contribution_formula": predictive_evidence[
                    "maximum_absolute_formula_contribution_formula"
                ],
            },
            "plain_language_interpretation": (
                "The aggregate held-out graph-minus-identity delta ELPD is "
                f"{predictive_evidence['aggregate_delta_elpd']:.6g}; the affine "
                f"screen {affine_screen_phrase} beat the direct scale screen."
            ),
            "plain_language_non_claim": (
                "This does not establish a universal DFT band-gap correction, broad "
                "transport across formula populations, or coverage of metals and "
                "zero-gap materials. The direct Gaussian screen has nonphysical "
                "negative-gap tails and is diagnostic only."
            ),
            "gates": predictive_gates,
        },
        {
            "claim_id": "JG-GEOM",
            "status": "SUPPORTED" if all(geometry_gates.values()) else "BLOCKED",
            "claim": (
                "Genuine log/direct/reciprocal pushforwards preserve the complete "
                "finite tube, and independently based corrected hard limits recover "
                "the geometry graph."
            ),
            "qualification": (
                "The geometry audit uses a fixed direct-energy likelihood separate "
                "from the robust scientific primary."
            ),
            "primary_result": geometry_stats,
            "plain_language_interpretation": (
                "The independently evaluated coordinate and measure routes satisfy "
                "the declared finite-tube and hard-limit identities."
            ),
            "plain_language_non_claim": (
                "Measure consistency is not a materials mechanism or evidence that "
                "the common-scale model is predictively adequate."
            ),
            "gates": geometry_gates,
        },
        {
            "claim_id": "JG-THRESHOLD",
            "status": domain_status,
            "claim": domain_claim,
            "qualification": domain_qualification,
            "primary_result": {
                "posterior_gamma_mean_range": gamma_range,
                "aggregate_delta_elpd_by_threshold": {
                    str(key): float(value) for key, value in delta.items()
                },
                "bootstrap_q05_delta_elpd_by_threshold": {
                    str(row.minimum_opt_gap_ev): float(row.bootstrap_q05_delta_elpd)
                    for row in domain_evidence.itertuples()
                },
                "bootstrap_q95_delta_elpd_by_threshold": {
                    str(row.minimum_opt_gap_ev): float(row.bootstrap_q95_delta_elpd)
                    for row in domain_evidence.itertuples()
                },
            },
            "plain_language_interpretation": (
                "The reported status follows the sign of the complete threshold-"
                "specific held-out comparison, not posterior movement alone."
            ),
            "plain_language_non_claim": (
                "The nested thresholds do not define an optimal materials domain, "
                "were not used to select the primary dataset, and are not "
                "independent-domain replications."
            ),
            "gates": domain_gates,
        },
    ]


def write_outputs(
    frame: pd.DataFrame,
    output_root: Path,
    mode: str,
    permutations: int,
    *,
    input_sha256: str,
    pinned_fixture: bool,
    data_manifest: dict[str, Any],
) -> dict[str, Any]:
    grid = make_grid(mode)
    x = frame["bandgap_opt_ev"].to_numpy(float)
    y = frame["bandgap_expt_ev"].to_numpy(float)
    ratio = np.log(y / x)
    log_post = log_posterior_grid(ratio, grid, family="student_t4")
    graph = posterior_summary(grid, log_post)
    sigma_geometry = np.sqrt(0.12**2 + (0.15 * x) ** 2)
    geometry_audit = build_geometry_audit(
        x,
        y,
        sigma_geometry,
        mu_sd=0.75,
        local_bounds=(-1.6, 1.6),
    )
    geometry_graph_summary = geometry_audit["summaries"]["geometry_graph"]
    api_audit = detcorr_api_audit(float(geometry_graph_summary["gamma_mean"]), x)
    rejected_reciprocal = rejected_reciprocal_diagnostic(frame)
    posterior_targets = pd.DataFrame(
        [{"target": "graph_primary", "scientific_status": "sole scientific target", **graph}]
    )
    bootstrap_replicates = {"quick": 199, "standard": 999, "full": 2000}[mode]
    protocol_config = {
        "protocol_version": PROTOCOL_VERSION,
        "study": "jarvis_band_gaps",
        "primary_model": "student_t4_log_ratio_common_scale",
        "comparison_model": "identity_with_student_t4_log_ratio_discrepancy",
        "grouping": "leave_one_formula_group_out",
        "grid_mode": mode,
        "grid_shape": [int(grid.mu.size), int(grid.eta.size)],
        "bootstrap_replicates": bootstrap_replicates,
        "permutation_replicates": int(permutations),
        "resampling_seed": RESAMPLING_SEED,
        "coverage_levels": [0.5, 0.9],
        "opt_gap_thresholds_ev": [0.0, 0.25, 0.5, 1.0],
        "priors": {
            "mu": "Normal(0,0.75^2) with respect to dmu",
            "eta": "Normal(log(0.45),0.75^2) with respect to deta",
        },
        "predictive_quantiles": {
            "method": "exact_finite_mixture_cdf_inversion_in_log_ratio",
            "probabilities": PREDICTIVE_QUANTILE_PROBABILITIES.tolist(),
            "component_limit": None,
            "brent_xtol": 1.0e-12,
            "brent_rtol": float(8.0 * np.finfo(float).eps),
        },
        "grid_refinement": {
            "quick_comparison": "standard",
            "standard_comparison": "quick",
            "full_comparison": "standard",
            "density_quantile_and_elpd_tolerance": 1.0e-10,
            "posterior_log_gamma_quantile_tolerance_full": 0.001,
            "posterior_log_gamma_quantile_tolerance_nonclaim": 0.003,
            "coverage_mismatches_allowed": 0,
        },
        "claim_rules": {
            "point_error_requirement": "both_median_rmse_and_mae_improve",
            "substantial_case_concentration_fraction": 0.5,
            "coverage90_maximum_degradation": 0.05,
            "maximum_formula_influence_graph_sd": 1.0,
            "pairing_lower_tail_rank_maximum": 0.05,
            "direct_scale_rmse_must_improve_identity": True,
            "affine_rmse_must_not_beat_scale_for_supported_primary": True,
            "threshold_point_sign_must_match_full_fixture": True,
            "threshold_bootstrap90_must_exclude_zero_in_full_fixture_direction": True,
        },
    }
    role_manifest, role_validation = logo_role_manifest(frame, protocol_config=protocol_config)
    predictions = grouped_log_scale_predictions(frame, grid)
    predictive_scores = summarize_predictions(predictions)
    direct_predictions, direct_scores = direct_model_ladder(frame)
    group_elpd, group_bootstrap, predictive_evidence = predictive_group_evidence(
        predictions,
        bootstrap_replicates=bootstrap_replicates,
        seed=RESAMPLING_SEED,
    )
    support_audit = primary_support_audit(frame, grid)
    influence = influence_table(frame, grid, graph)
    permutation_values, permutation_summary = permutation_control(
        frame, permutations, seed=RESAMPLING_SEED
    )
    sensitivity = sensitivity_table(frame, grid)
    domain_sensitivity = domain_sensitivity_table(frame, grid)
    domain_predictive = domain_predictive_sensitivity(
        frame,
        grid,
        bootstrap_replicates=bootstrap_replicates,
        full_predictions=predictions,
    )
    grid_refinement = grid_resolution_audit(
        frame,
        grid,
        predictions,
        graph,
        domain_sensitivity,
        domain_predictive,
        bootstrap_replicates=bootstrap_replicates,
    )
    threshold_support = threshold_support_audit(frame, grid)
    threshold_roles, threshold_role_validation = threshold_role_audit(
        frame,
        protocol_config=protocol_config,
    )
    threshold_leakage = threshold_leakage_audit(frame, grid)
    rho, _ = spearmanr(x, y)
    residual_rho, _ = spearmanr(x, influence["log_residual"])
    density_jacobian = physical_density_jacobian_audit(frame, grid)
    leakage_sentinel = heldout_outcome_leakage_sentinel(frame, grid)
    claim_evaluations = evaluate_claims(
        predictive_scores,
        predictive_evidence,
        direct_scores,
        domain_sensitivity,
        domain_predictive,
        influence,
        permutation_summary,
        support_audit,
        threshold_support,
        geometry_audit,
        api_audit,
        rejected_reciprocal,
        role_validation,
        threshold_role_validation,
        density_jacobian,
        leakage_sentinel,
        threshold_leakage,
        grid_refinement,
    )
    claim_bearing_run = bool(
        mode == "full" and pinned_fixture and bootstrap_replicates == 2000 and permutations == 2000
    )
    if not claim_bearing_run:
        for claim in claim_evaluations:
            claim["diagnostic_gate_status"] = claim["status"]
            claim["status"] = "EXPLORATORY"
            claim["qualification"] += (
                " This quick or custom-input run is diagnostic only; only a full "
                "run on the pinned fixture receives a terminal claim status."
            )

    tables = output_root / "tables"
    figures = output_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    table_map = {
        "posterior_targets.csv": posterior_targets,
        "geometry_route_contract.csv": geometry_audit["contracts"],
        "geometry_hard_limit_audit.csv": geometry_audit["hard_limits"],
        "geometry_finite_pushforward_audit.csv": geometry_audit["finite_pushforwards"],
        "geometry_global_support.csv": geometry_audit["global_support"],
        "rejected_reciprocal_diagnostic.csv": pd.DataFrame([rejected_reciprocal]),
        "detcorr_api_audit.csv": pd.DataFrame([api_audit]),
        "logo_role_manifest.csv": role_manifest,
        "logo_role_validation.csv": pd.DataFrame([role_validation]),
        "heldout_outcome_leakage_audit.csv": pd.DataFrame([leakage_sentinel]),
        "physical_density_jacobian_audit.csv": pd.DataFrame([density_jacobian]),
        "predictive_grid_refinement_gates.csv": grid_refinement["gate_table"],
        "predictive_grid_refinement_details.csv": grid_refinement["primary_details"],
        "opt_gap_threshold_grid_refinement_details.csv": grid_refinement["threshold_details"],
        "grouped_predictions.csv": predictions,
        "predictive_scores.csv": predictive_scores,
        "predictive_group_elpd_contributions.csv": group_elpd,
        "predictive_formula_cluster_bootstrap.csv": group_bootstrap,
        "primary_grid_support_audit.csv": support_audit,
        "direct_model_ladder_predictions.csv": direct_predictions,
        "direct_model_ladder_scores.csv": direct_scores,
        "influence_and_residuals.csv": influence,
        "pairing_permutation_distribution.csv": permutation_values,
        "sensitivity.csv": sensitivity,
        "opt_gap_domain_sensitivity.csv": domain_sensitivity,
        "opt_gap_threshold_logo_predictions.csv": domain_predictive["predictions"],
        "opt_gap_threshold_logo_scores.csv": domain_predictive["scores"],
        "opt_gap_threshold_formula_contributions.csv": domain_predictive["contributions"],
        "opt_gap_threshold_formula_bootstrap.csv": domain_predictive["bootstrap"],
        "opt_gap_threshold_predictive_evidence.csv": domain_predictive["evidence"],
        "opt_gap_threshold_grid_support_audit.csv": threshold_support,
        "opt_gap_threshold_logo_role_manifest.csv": threshold_roles,
        "opt_gap_threshold_role_validation.csv": threshold_role_validation,
        "opt_gap_threshold_heldout_leakage_audit.csv": threshold_leakage,
    }
    for name, table in table_map.items():
        table.to_csv(tables / name, index=False)

    claim_evaluations = add_claim_provenance(
        claim_evaluations,
        model_config=protocol_config,
        input_sha256=input_sha256,
        claim_bearing_full_run=claim_bearing_run,
        timing_class=(
            "registered_reanalysis"
            if claim_bearing_run
            else "verification"
            if pinned_fixture
            else "exploratory"
        ),
        evidence_class=(
            "fixture_conditional_inference" if claim_bearing_run else "implementation_diagnostic"
        ),
        evidence_level="E1" if claim_bearing_run else "E0",
        source_paths={
            "study_analysis": Path(__file__),
            "common_geometry": Path(__file__).resolve().parents[3]
            / "common"
            / "jarvis_geometry.py",
            "common_protocol": Path(__file__).resolve().parents[3]
            / "common"
            / "jarvis_protocol.py",
            "common_evidence": Path(__file__).resolve().parents[3] / "common" / "evidence.py",
            "common_predictive_mixture": Path(__file__).resolve().parents[3]
            / "common"
            / "predictive_mixture.py",
            "protocol_lock": Path(__file__).resolve().parents[3] / "PROTOCOL_LOCK.json",
        },
        tables_root=tables,
        evidence_files={
            "JG-PRED": (
                "grouped_predictions.csv",
                "predictive_scores.csv",
                "predictive_group_elpd_contributions.csv",
                "predictive_formula_cluster_bootstrap.csv",
                "direct_model_ladder_scores.csv",
                "direct_model_ladder_predictions.csv",
                "influence_and_residuals.csv",
                "pairing_permutation_distribution.csv",
                "primary_grid_support_audit.csv",
                "logo_role_manifest.csv",
                "logo_role_validation.csv",
                "heldout_outcome_leakage_audit.csv",
                "physical_density_jacobian_audit.csv",
                "predictive_grid_refinement_gates.csv",
                "predictive_grid_refinement_details.csv",
            ),
            "JG-GEOM": (
                "geometry_route_contract.csv",
                "geometry_hard_limit_audit.csv",
                "geometry_finite_pushforward_audit.csv",
                "geometry_global_support.csv",
                "rejected_reciprocal_diagnostic.csv",
                "detcorr_api_audit.csv",
            ),
            "JG-THRESHOLD": (
                "opt_gap_domain_sensitivity.csv",
                "opt_gap_threshold_logo_predictions.csv",
                "opt_gap_threshold_logo_scores.csv",
                "opt_gap_threshold_formula_contributions.csv",
                "opt_gap_threshold_formula_bootstrap.csv",
                "opt_gap_threshold_predictive_evidence.csv",
                "opt_gap_threshold_grid_support_audit.csv",
                "opt_gap_threshold_logo_role_manifest.csv",
                "opt_gap_threshold_role_validation.csv",
                "opt_gap_threshold_heldout_leakage_audit.csv",
                "physical_density_jacobian_audit.csv",
                "predictive_grid_refinement_gates.csv",
                "predictive_grid_refinement_details.csv",
                "opt_gap_threshold_grid_refinement_details.csv",
            ),
        },
    )
    claim_table = pd.DataFrame(
        [
            {
                "claim_id": row["claim_id"],
                "status": row["status"],
                "claim": row["claim"],
                "qualification": row["qualification"],
                "protocol_version": row["protocol_version"],
                "evidence_level": row["evidence_level"],
                "model_config_sha256": row["model_config_sha256"],
                "n_gates_passed": int(sum(row["gates"].values())),
                "n_gates": int(len(row["gates"])),
            }
            for row in claim_evaluations
        ]
    )
    claim_table.to_csv(tables / "claim_evaluations.csv", index=False)

    support_mu = geometry_audit["_plot_mu"]
    support_targets = geometry_audit["_plot_log_targets"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(
        support_mu,
        normalize_log_weights(support_targets["geometry_graph"]),
        label="geometry graph",
        lw=2.2,
    )
    ax.plot(
        support_mu,
        normalize_log_weights(support_targets["independent_direct_lebesgue_naive"]),
        label="independent direct-base naive hard limit",
        lw=1.6,
    )
    ax.plot(
        support_mu,
        normalize_log_weights(support_targets["independent_reciprocal_lebesgue_naive"]),
        label="independent reciprocal-base naive hard limit",
        lw=1.6,
    )
    display_lower = min(math.log(row["gamma_q005"]) for row in geometry_audit["summaries"].values())
    display_upper = max(math.log(row["gamma_q995"]) for row in geometry_audit["summaries"].values())
    display_padding = max(0.05, 0.15 * (display_upper - display_lower))
    ax.set_xlim(display_lower - display_padding, display_upper + display_padding)
    ax.axvline(0.0, color="black", lw=1, ls="--", label="identity")
    ax.set(
        xlabel="log common scale, mu=log(gamma)",
        ylabel="normalized mass on the uniform global mu grid",
        title="Band-gap global coordinate/measure support audit",
    )
    ax.text(
        0.01,
        0.01,
        "normalized/audited on mu in [-120, 20]; display cropped to joint 99% ranges",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "posterior_targets.png", dpi=180)
    plt.close(fig)

    graph_pred = predictions[predictions["model"] == "graph_primary"].copy()
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.errorbar(
        graph_pred["observed_ev"],
        graph_pred["prediction_median_ev"],
        yerr=[
            graph_pred["prediction_median_ev"] - graph_pred["prediction_q05_ev"],
            graph_pred["prediction_q95_ev"] - graph_pred["prediction_median_ev"],
        ],
        fmt="o",
        ms=4,
        alpha=0.75,
    )
    bounds = [
        0.0,
        1.05 * max(graph_pred["observed_ev"].max(), graph_pred["prediction_q95_ev"].max()),
    ]
    ax.plot(bounds, bounds, "k--", lw=1)
    ax.set(
        xlabel="held-out experimental gap (eV)",
        ylabel="LOGO posterior prediction (eV)",
        title="Formula-grouped held-out predictions",
    )
    fig.tight_layout()
    fig.savefig(figures / "grouped_predictions.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.hist(permutation_values["logo_log_rmse"], bins=24, color="#6c8ebf", alpha=0.8)
    ax.axvline(
        permutation_summary["observed_logo_log_rmse"],
        color="#b2182b",
        lw=2,
        label="observed pairing",
    )
    ax.set(
        xlabel="formula-LOGO RMSE in log gap",
        ylabel="permutations",
        title="Pairing-permutation negative control",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "pairing_permutation_control.png", dpi=180)
    plt.close(fig)

    claim_figures = {
        "JG-PRED": ("grouped_predictions.png", "pairing_permutation_control.png"),
        "JG-GEOM": ("posterior_targets.png",),
        "JG-THRESHOLD": (),
    }
    for claim in claim_evaluations:
        claim["evidence_artifacts"].extend(
            {
                "path": f"figures/{name}",
                "sha256": file_hash(figures / name),
                "artifact_type": (
                    "claim_bearing_figure" if claim_bearing_run else "diagnostic_figure"
                ),
            }
            for name in claim_figures[claim["claim_id"]]
        )

    support_passed = bool(
        support_audit["mode_inside_local_grid"].all()
        and support_audit["analytic_mu_tail_coercive"].all()
        and support_audit["analytic_eta_tail_coercive"].all()
        and support_audit["broad_mass_outside_local_grid"].max() < 1.0e-4
        and support_audit["broad_edge_mass"].max() < 1.0e-10
        and support_audit["minimum_broad_boundary_profile_log_drop"].min() > 30.0
    )
    claim_status = {
        row["claim_id"]: row.get("diagnostic_gate_status", row["status"])
        for row in claim_evaluations
    }
    affine_beats_scale = bool(
        direct_scores.set_index("model").loc["affine_direct", "rmse_ev"]
        < direct_scores.set_index("model").loc["scale_direct", "rmse_ev"]
    )
    implementation_checks = {
        "sole_scientific_posterior_is_graph_primary": bool(
            len(posterior_targets) == 1 and posterior_targets.iloc[0]["target"] == "graph_primary"
        ),
        "formula_logo_role_manifest_all_passed": bool(role_validation["all_passed"]),
        "primary_full_and_logo_support_all_passed": support_passed,
        "geometry_contract_all_passed": bool(geometry_audit["stats"]["all_passed"]),
        "physical_density_minus_log_y_jacobian_all_passed": bool(density_jacobian["all_passed"]),
        "heldout_outcome_leakage_sentinel_all_passed": bool(leakage_sentinel["all_passed"]),
        "detcorr_api_matches_analytic_result": bool(api_audit["absolute_error"] < 1.0e-10),
        "locked_resampling_seed_used": bool(
            predictive_evidence["bootstrap_seed"] == RESAMPLING_SEED
            and permutation_summary["seed"] == RESAMPLING_SEED
            and domain_predictive["evidence"]["bootstrap_seed"].eq(RESAMPLING_SEED).all()
        ),
        "claim_bearing_full_run_uses_2000_bootstraps_and_permutations": bool(
            mode != "full"
            or not pinned_fixture
            or (
                predictive_evidence["bootstrap_replicates"] == 2000
                and permutation_summary["n_permutations"] == 2000
                and domain_predictive["evidence"]["bootstrap_replicates"].eq(2000).all()
            )
        ),
        "positive_gap_delta_elpd_with_better_affine_screen_maps_to_mixed": bool(
            predictive_evidence["aggregate_delta_elpd"] <= 0.0
            or not affine_beats_scale
            or claim_status["JG-PRED"] == "MIXED"
        ),
        "threshold_domain_outputs_are_complete": bool(
            len(domain_predictive["evidence"]) == 4
            and len(domain_predictive["scores"]) == 8
            and len(threshold_role_validation) == 4
            and len(threshold_leakage) == 4
            and set(threshold_support["model"]) == {"graph_primary", "identity"}
        ),
        "threshold_formula_logo_role_manifests_all_passed": bool(
            threshold_role_validation["all_passed"].all()
        ),
        "threshold_heldout_outcome_leakage_sentinels_all_passed": bool(
            threshold_leakage["all_passed"].all()
        ),
        "primary_and_threshold_predictive_grid_refinement_all_passed": bool(
            grid_refinement["stats"]["all_passed"]
        ),
        "all_claim_gates_are_boolean": bool(
            all(
                isinstance(value, (bool, np.bool_))
                for claim in claim_evaluations
                for value in claim["gates"].values()
            )
        ),
    }
    implementation_checks["all_passed"] = bool(all(implementation_checks.values()))

    summary = {
        "analysis_status": (
            "registered reanalysis of a pre-existing public fixture; not prospective confirmation"
        ),
        "claim_bearing_full_run": claim_bearing_run,
        "claim_ceiling": (
            "fixture-conditional predictive calibration and model-adequacy "
            "evidence; no universal DFT correction"
        ),
        "data": {
            "input_sha256": input_sha256,
            "is_bundled_pinned_fixture": pinned_fixture,
            "bundled_fixture_expected_sha256": EXPECTED_SHA256,
            "n_formula_groups": int(frame["formula"].nunique()),
            "n_rows": int(len(frame)),
            "source_url": SOURCE_URL,
            "upstream_redistribution_terms": data_manifest["license_status"],
            "manifest": data_manifest,
        },
        "grid": {"mode": mode, "n_mu": int(grid.mu.size), "n_eta": int(grid.eta.size)},
        "primary_estimand": (
            "common positive scale gamma under a normalized Student-t(4) log-ratio "
            "discrepancy, with full-support Gaussian priors and density with "
            "respect to dlog(gamma) dlog(tau); local quadrature is verified by a "
            "broad-grid audit"
        ),
        "protocol": {
            "version": PROTOCOL_VERSION,
            "timing_class": (
                "registered_reanalysis"
                if claim_bearing_run
                else "verification"
                if pinned_fixture
                else "exploratory"
            ),
            "protocol_timing": "post_outcome_retrospective",
            "evidence_level": "E1" if claim_bearing_run else "E0",
            "model_config_sha256": canonical_hash(protocol_config),
            "configuration": protocol_config,
        },
        "posterior": {"primary_graph": graph},
        "geometry_diagnostic": {
            "likelihood": "fixed direct-energy Gaussian; sigma=sqrt(0.12^2+(0.15 E_OPT)^2)",
            "scientific_role": "separate coordinate/measure audit; not the primary model",
            "graph": geometry_graph_summary,
            "independent_direct_lebesgue_naive": geometry_audit["summaries"][
                "independent_direct_lebesgue_naive"
            ],
            "independent_reciprocal_lebesgue_naive": geometry_audit["summaries"][
                "independent_reciprocal_lebesgue_naive"
            ],
            "computed_gate_statistics": geometry_audit["stats"],
            "rejected_reciprocal": rejected_reciprocal,
        },
        "detcorr_api_audit": api_audit,
        "predictive_protocol": (
            "leave one complete formula group out; every prior and discrepancy "
            "choice fixed in code before evaluating that group"
        ),
        "predictive_scores": predictive_scores.to_dict(orient="records"),
        "predictive_evidence": predictive_evidence,
        "logo_role_manifest_validation": role_validation,
        "physical_density_jacobian_audit": density_jacobian,
        "heldout_outcome_leakage_sentinel": leakage_sentinel,
        "threshold_predictive_evidence": domain_predictive["evidence"].to_dict(orient="records"),
        "threshold_predictive_scores": domain_predictive["scores"].to_dict(orient="records"),
        "threshold_role_manifest_validation": threshold_role_validation.to_dict(orient="records"),
        "threshold_heldout_outcome_leakage_sentinels": threshold_leakage.to_dict(orient="records"),
        "predictive_grid_refinement": grid_refinement["stats"],
        "direct_model_ladder_scores": direct_scores.to_dict(orient="records"),
        "direct_model_ladder_negative_q05_counts": {
            str(model): int((part["prediction_q05_ev"] < 0.0).sum())
            for model, part in direct_predictions.groupby("model", sort=True)
        },
        "negative_control": permutation_summary,
        "diagnostics": {
            "spearman_opt_vs_expt": float(rho),
            "spearman_opt_vs_primary_log_residual": float(residual_rho),
            "maximum_absolute_omit_group_shift_graph_sd": float(
                influence["omit_formula_shift_graph_sd"].abs().max()
            ),
        },
        "limitations": [
            "The 54-row positive-gap documentation fixture is small and heterogeneous.",
            "Metallic and zero-gap records are outside this positive-domain calibration.",
            (
                "The source labels the response as an experimental gap but does not "
                "resolve fundamental-versus-optical measurement type row by row."
            ),
            (
                "Experimental uncertainty, temperature, phase, and "
                "measurement-protocol metadata are unavailable in the fixture."
            ),
            (
                "The primary discrepancy is a model for aggregate calibration "
                "error, not a measured experimental error law."
            ),
            (
                "The direct Gaussian identity/scale/affine ladder is an adequacy "
                "screen with unbounded physical-eV support and nonphysical "
                "negative-gap tails, not a positive-gap generative model."
            ),
            (
                "LOGO evaluates interpolation across represented material types, "
                "not transport to a new database or functional."
            ),
            (
                "The pairing permutation is a negative-control diagnostic; "
                "heterogeneous formula groups are not asserted exchangeable for "
                "formal hypothesis testing."
            ),
        ],
        "implementation_checks": implementation_checks,
        "claim_evaluations": claim_evaluations,
    }
    strict_json_dump(summary, output_root / "summary.json")
    return summary


def run(
    data_path: Path,
    output_root: Path,
    mode: str,
    permutations: int,
    *,
    require_pinned_checksum: bool = True,
    data_manifest_path: Path | None = None,
) -> dict[str, Any]:
    digest = sha256(data_path)
    if data_manifest_path is None:
        raise ValueError("a provenance manifest is required for every input table")
    frame = load_fixture(data_path, require_pinned_checksum=require_pinned_checksum)
    manifest = load_data_manifest(
        data_manifest_path,
        digest,
        data_path=data_path,
        frame=frame,
    )
    return write_outputs(
        frame,
        output_root,
        mode,
        permutations,
        input_sha256=digest,
        pinned_fixture=require_pinned_checksum and digest == EXPECTED_SHA256,
        data_manifest=manifest,
    )
