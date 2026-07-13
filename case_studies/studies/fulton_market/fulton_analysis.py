"""Core calculations for the registered Fulton temporal reanalysis.

An iid/native-price construction provides an independent numerical
verification route. The primary scientific analysis fits a log-price
reference, calibrates a multiplicative scale on the immediately preceding
observations, and scores the next observation, which has not entered either
fit or calibration.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
from scipy import linalg, stats  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

from common.evidence import density_distances, strict_json_dump, trapezoid

STUDY_ROOT = Path(__file__).resolve().parent
DATA_PATH = STUDY_ROOT / "data" / "fish.csv"
DATA_MANIFEST_PATH = STUDY_ROOT / "data" / "FIXTURE_MANIFEST.json"
REGISTERED_DATA_DISPLAY_PATH = "studies/fulton_market/data/fish.csv"
PROTOCOL_PATH = STUDY_ROOT / "REGISTERED_PROTOCOL.json"
DATA_URL = (
    "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/wooldridge/fish.csv"
)
DATA_SHA256 = "fa51db236cb3f9e7040cdc50b002542bcbb3d51077132848d1960a996ec9b3c9"

DESIGN_COLUMNS = ("wave2", "speed2", "mon", "tues", "wed", "thurs", "t")
REQUIRED_COLUMNS = ("rownames", "avgprc", *DESIGN_COLUMNS)

REQUIRED_DATA_MANIFEST_FIELDS = {
    "license_identifier",
    "redistribution_decision",
    "rights_basis",
    "row_count",
    "selection_rule",
    "sha256",
    "source_name",
    "source_row_key",
    "source_url_or_accession",
    "units",
}

VERIFICATION_TRAIN_SIZE = 64
FIRST_TEST_POSITION = 64  # zero based: source t=65
CALIBRATION_WINDOW = 16
PLACEBO_LAG = 4
PRIOR_ETA_SD = 0.5
PHI_LIMIT = 0.90
PHI_CLIP_SENSITIVITY_LIMITS = (0.50, 0.75, PHI_LIMIT, 0.99)
PHI_CLIP_SENSITIVITY_TIMING = "post_outcome_nonselecting"
INTERVAL_LEVEL = 0.90
BOOTSTRAP_SEED = 20260711
BOOTSTRAP_LOWER_QUANTILE = 0.05
BOOTSTRAP_UPPER_QUANTILE = 0.95
PENALTY_RHO = 5.0e7
GRID_MIN = 0.25
GRID_MAX = 2.0
PRIMARY_GEOMETRY_RHO = 1.0e6
PRIMARY_GEOMETRY_ETA_POINTS = 25
PRIMARY_GEOMETRY_CENTRAL_PROBABILITY = 0.99
PRIMARY_GEOMETRY_QUICK_SOBOL_POWER = 11
PRIMARY_GEOMETRY_FULL_SOBOL_POWER = 14
PRIMARY_GEOMETRY_QUICK_SCRAMBLES = 2
PRIMARY_GEOMETRY_FULL_SCRAMBLES = 4
PRIMARY_GEOMETRY_MAX_ERROR_FRACTION = 0.01

PRIMARY_MODEL = "ar1_graph"
MODEL_ORDER = (
    PRIMARY_MODEL,
    "ar1_identity_reference_baseline",
    "iid_graph_sensitivity",
    "iid_identity_reference_baseline",
    "ar1_beta_uncertainty_sensitivity",
    "ar1_beta_uncertainty_identity_baseline",
    "ar1_naive_volume_omission_counterfactual",
    "ar1_lag4_reference_placebo",
    "last_price_persistence_baseline",
)

MATCHED_CALIBRATION_COMPARISONS = (
    (
        "ar1_plugin",
        "ar1_graph",
        "ar1_identity_reference_baseline",
    ),
    (
        "iid",
        "iid_graph_sensitivity",
        "iid_identity_reference_baseline",
    ),
    (
        "ar1_beta_uncertainty",
        "ar1_beta_uncertainty_sensitivity",
        "ar1_beta_uncertainty_identity_baseline",
    ),
)


@dataclass(frozen=True)
class ReferenceFit:
    """Training-only reduced-form fit in either native or log-price units."""

    response: str
    coefficients: np.ndarray
    coefficient_covariance: np.ndarray
    residuals: np.ndarray
    residual_sd: float
    phi_raw: float
    phi_used: float
    r2: float
    n_train: int
    n_parameters: int


@dataclass(frozen=True)
class EtaPosterior:
    """Exact normal posterior for eta=log(lambda)."""

    mean: float
    variance: float
    graph_mean: float
    graph_variance: float
    target: str

    @property
    def sd(self) -> float:
        return math.sqrt(self.variance)


@dataclass(frozen=True)
class LognormalPrediction:
    """One-step predictive distribution in native price units."""

    log_mean: float
    log_sd: float

    @property
    def mean(self) -> float:
        return math.exp(self.log_mean + 0.5 * self.log_sd**2)

    @property
    def median(self) -> float:
        return math.exp(self.log_mean)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data_manifest(path: Path, *, data_path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    """Validate and return the provenance record bound to an input table."""

    manifest = json.loads(path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_DATA_MANIFEST_FIELDS - set(manifest))
    if missing:
        raise ValueError(f"data manifest missing fields: {missing}")
    blank = sorted(
        key for key in REQUIRED_DATA_MANIFEST_FIELDS if manifest[key] in (None, "", [], {})
    )
    if blank:
        raise ValueError(f"data manifest has blank fields: {blank}")
    digest = sha256_file(data_path)
    if manifest["sha256"] != digest:
        raise ValueError("data manifest SHA256 does not match selected input")
    if int(manifest["row_count"]) != len(frame):
        raise ValueError("data manifest row_count does not match selected input")
    if manifest["source_row_key"] != "rownames":
        raise ValueError("data manifest source_row_key must be 'rownames'")
    if not isinstance(manifest["units"], dict) or "avgprc" not in manifest["units"]:
        raise ValueError("data manifest units must declare avgprc")
    return manifest


def canonical_json_sha256(payload: Any) -> str:
    """Hash a JSON-compatible payload using a stable canonical encoding."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_and_validate_protocol(path: Path = PROTOCOL_PATH) -> tuple[dict[str, Any], str]:
    """Load the protocol and reject any claim-bearing semantic drift."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    expected: dict[str, Any] = {
        "analysis_class": "registered_retrospective_reanalysis",
        "baselines": [
            "ar1_identity_reference_baseline",
            "iid_identity_reference_baseline",
            "ar1_beta_uncertainty_identity_baseline",
            "last_price_persistence_baseline",
        ],
        "calibration_window": CALIBRATION_WINDOW,
        "claim_boundary": (
            "conditional one-step price calibration; no causal, structural, elasticity, "
            "supply, demand, or market-power claim"
        ),
        "dependence": {
            "innovation_sd": ("derived as stationary_marginal_residual_sd * sqrt(1-phi_used^2)"),
            "phi_bound": PHI_LIMIT,
            "phi_bound_sensitivity": {
                "absolute_limits": list(PHI_CLIP_SENSITIVITY_LIMITS),
                "primary_absolute_limit": PHI_LIMIT,
                "selection_role": "robustness disclosure only; never selects the primary model",
                "timing": PHI_CLIP_SENSITIVITY_TIMING,
            },
            "phi_inference": "training-only plug-in estimate; no phi posterior",
            "primary": "stationary_ar1_training_only_plugin",
            "residual_sd": "training-only plug-in stationary marginal residual SD",
            "sensitivity": "iid",
        },
        "first_test_position_zero_based": FIRST_TEST_POSITION,
        "first_test_t_registered_fixture": 65,
        "forecast_horizon": 1,
        "geometry_audit": {
            "central_graph_probability": PRIMARY_GEOMETRY_CENTRAL_PROBABILITY,
            "direct_state_base_density": "q_u=1 relative to du",
            "eta_points": PRIMARY_GEOMETRY_ETA_POINTS,
            "full_sobol_power": PRIMARY_GEOMETRY_FULL_SOBOL_POWER,
            "full_sobol_scrambles": PRIMARY_GEOMETRY_FULL_SCRAMBLES,
            "log_state_base_density": "q_v=exp(sum(v)) relative to dv",
            "max_error_fraction_of_determinant_variation": (PRIMARY_GEOMETRY_MAX_ERROR_FRACTION),
            "primary_origin_t": 65,
            "quick_sobol_power": PRIMARY_GEOMETRY_QUICK_SOBOL_POWER,
            "quick_sobol_scrambles": PRIMARY_GEOMETRY_QUICK_SCRAMBLES,
            "rho": PRIMARY_GEOMETRY_RHO,
            "seed": BOOTSTRAP_SEED,
            "state_scale": "geometric mean price of reference-training rows",
        },
        "matched_identity_models": {
            "ar1_beta_uncertainty": "ar1_beta_uncertainty_identity_baseline",
            "ar1_plugin": "ar1_identity_reference_baseline",
            "iid": "iid_identity_reference_baseline",
        },
        "negative_control": {
            "lag_observations": PLACEBO_LAG,
            "name": "causal_lagged_reference",
            "rule": (
                "replace each contemporaneous reference design row by its row four "
                "observations earlier without reordering responses"
            ),
        },
        "observation_coordinate": "log_native_average_price",
        "parameter": (
            "eta=log(lambda), where lambda is a dimensionless multiplicative "
            "reference-to-state scale"
        ),
        "primary_model": PRIMARY_MODEL,
        "registered_claim_ids": ["FU-PRED", "FU-DEP", "FU-GEOM"],
        "prior": (
            "eta ~ Normal(0, 0.5^2), equivalently a lognormal density for lambda "
            "with respect to d lambda"
        ),
        "prior_parameters": {
            "eta_mean": 0.0,
            "eta_sd": PRIOR_ETA_SD,
            "family": "normal_on_eta",
            "lambda_density_measure": "d_lambda",
        },
        "reference_fit": {
            "columns": ["const", *DESIGN_COLUMNS],
            "response": "log(avgprc)",
            "rule": "for test position j, fit only rows strictly before j-16",
        },
        "score_uncertainty": {
            "block_length": 4,
            "full_replicates": 5000,
            "interval_quantiles": [
                BOOTSTRAP_LOWER_QUANTILE,
                BOOTSTRAP_UPPER_QUANTILE,
            ],
            "method": ("circular moving-block bootstrap of paired origin-level score contrasts"),
            "quick_replicates": 500,
            "seed": BOOTSTRAP_SEED,
        },
        "scores": [
            "native-price log predictive density",
            "native-price CRPS",
            "predictive-mean RMSE",
            "predictive-median MAE",
            "90-percent interval coverage and width",
        ],
        "test_information": {
            "contemporaneous_covariates_used": True,
            "price_outcome_used_before_scoring": False,
            "scope": (
                "conditional one-step prediction, not an unconditional forecast when "
                "contemporaneous weather is unavailable"
            ),
        },
        "sensitivities": [
            "iid log-price errors",
            "Gaussian OLS coefficient covariance propagated analytically",
            "state-volume determinant omission counterfactual",
            "one-at-a-time symmetric AR(1) coefficient clipping bounds",
        ],
        "version": 2,
    }

    def require_equal(actual: Any, required: Any, location: str) -> None:
        if isinstance(required, dict):
            if not isinstance(actual, dict):
                raise ValueError(f"REGISTERED_PROTOCOL.json mismatch for {location}")
            if set(actual) != set(required):
                raise ValueError(
                    f"REGISTERED_PROTOCOL.json keys mismatch for {location}: "
                    f"expected {sorted(required)}, found {sorted(actual)}"
                )
            for key in required:
                require_equal(actual[key], required[key], f"{location}.{key}")
            return
        if isinstance(required, list):
            if not isinstance(actual, list) or len(actual) != len(required):
                raise ValueError(f"REGISTERED_PROTOCOL.json mismatch for {location}")
            for index, (actual_value, required_value) in enumerate(
                zip(actual, required, strict=True)
            ):
                require_equal(actual_value, required_value, f"{location}[{index}]")
            return
        if actual != required or type(actual) is not type(required):
            raise ValueError(
                f"REGISTERED_PROTOCOL.json mismatch for {location}: "
                f"expected {required!r}, found {actual!r}"
            )

    require_equal(payload, expected, "root")
    return payload, sha256_file(path)


def load_fish(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load a registered or schema-compatible Fulton-format series.

    Rows are never sorted or silently dropped.  A failure therefore exposes a
    provenance or preprocessing change instead of changing the time design.
    """

    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"input is missing required columns: {missing}")
    if len(frame) < FIRST_TEST_POSITION + 1:
        raise ValueError("at least 65 chronologically ordered rows are required")
    required = frame.loc[:, REQUIRED_COLUMNS]
    if required.isna().any().any():
        raise ValueError("required analysis columns contain missing values")
    numeric = required.to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("required analysis columns contain non-finite values")
    if np.any(frame["avgprc"].to_numpy(dtype=float) <= 0.0):
        raise ValueError("avgprc must be strictly positive")
    times = frame["t"].to_numpy(dtype=float)
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("rows must already be in strictly increasing t order")
    if frame["rownames"].duplicated().any():
        raise ValueError("rownames must be unique")
    return frame.reset_index(drop=True).copy()


def design_matrix(frame: pd.DataFrame) -> np.ndarray:
    values = frame.loc[:, DESIGN_COLUMNS].to_numpy(dtype=float)
    return np.column_stack([np.ones(len(frame), dtype=float), values])


def build_role_manifest(
    frame: pd.DataFrame,
    *,
    first_test_position: int = FIRST_TEST_POSITION,
    calibration_window: int = CALIBRATION_WINDOW,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the deterministic per-origin train/calibration/test assignment."""

    split_algorithm = "expanding_train_fixed16_calibration_horizon1_v1"
    split_salt = "fulton-temporal-v1-no-randomization"
    rows: list[dict[str, Any]] = []
    for test_position in range(first_test_position, len(frame)):
        calibration_start = test_position - calibration_window
        origin_t = int(frame.iloc[test_position]["t"])
        for position in range(test_position + 1):
            if position < calibration_start:
                role = "train"
            elif position < test_position:
                role = "calibration"
            else:
                role = "test"
            rows.append(
                {
                    "origin_test_t": origin_t,
                    "source_row": int(frame.iloc[position]["rownames"]),
                    "source_t": int(frame.iloc[position]["t"]),
                    "role": role,
                    "split_algorithm": split_algorithm,
                    "split_salt": split_salt,
                }
            )
    manifest = pd.DataFrame(rows)
    grouped = manifest.groupby("origin_test_t", sort=True)
    test_counts = grouped["role"].apply(lambda values: (values == "test").sum())
    one_test_per_origin = bool((test_counts == 1).all())
    calibration_counts = grouped["role"].apply(lambda values: (values == "calibration").sum())
    tests = manifest.loc[manifest["role"] == "test", "source_t"].to_numpy(dtype=int)
    assertions = {
        "one_test_row_per_origin": one_test_per_origin,
        "fixed_calibration_count": bool(np.all(calibration_counts == calibration_window)),
        "test_rows_are_65_through_end_once": bool(
            np.array_equal(tests, frame.iloc[first_test_position:]["t"].to_numpy(dtype=int))
        ),
        "roles_are_disjoint_within_origin": bool(
            not manifest.duplicated(["origin_test_t", "source_row"]).any()
        ),
        "split_algorithm": split_algorithm,
        "split_salt": split_salt,
    }
    if not all(value for key, value in assertions.items() if isinstance(value, bool)):
        raise AssertionError("role manifest assertions failed")
    return manifest, assertions


def _validated_phi_limit(phi_limit: float) -> float:
    value = float(phi_limit)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("phi_limit must be finite and strictly between zero and one")
    return value


def fit_reference(
    frame: pd.DataFrame,
    *,
    response: str = "log",
    phi_limit: float = PHI_LIMIT,
) -> ReferenceFit:
    """Fit the predeclared reduced-form reference using training rows only."""

    if response not in {"log", "native"}:
        raise ValueError("response must be 'log' or 'native'")
    phi_limit = _validated_phi_limit(phi_limit)
    x = design_matrix(frame)
    raw = frame["avgprc"].to_numpy(dtype=float)
    y = np.log(raw) if response == "log" else raw
    if len(frame) <= x.shape[1] + 2:
        raise ValueError("reference training block has insufficient residual degrees of freedom")
    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if int(rank) != x.shape[1]:
        raise ValueError("reference design matrix is rank deficient")
    fitted = x @ coefficients
    residuals = y - fitted
    dof = len(y) - x.shape[1]
    residual_variance = float(residuals @ residuals) / dof
    if not math.isfinite(residual_variance) or residual_variance <= 0.0:
        raise ValueError("reference residual variance must be positive and finite")
    covariance = residual_variance * np.linalg.pinv(x.T @ x, hermitian=True)
    centered = residuals - float(np.mean(residuals))
    denominator = float(centered[:-1] @ centered[:-1])
    phi_raw = float(centered[1:] @ centered[:-1]) / denominator if denominator > 0.0 else 0.0
    phi_used = float(np.clip(phi_raw, -phi_limit, phi_limit))
    centered_y = y - float(np.mean(y))
    total = float(centered_y @ centered_y)
    r2 = 1.0 - float(residuals @ residuals) / total if total > 0.0 else 0.0
    return ReferenceFit(
        response=response,
        coefficients=np.asarray(coefficients, dtype=float),
        coefficient_covariance=np.asarray(covariance, dtype=float),
        residuals=np.asarray(residuals, dtype=float),
        residual_sd=math.sqrt(residual_variance),
        phi_raw=phi_raw,
        phi_used=phi_used,
        r2=float(r2),
        n_train=len(frame),
        n_parameters=x.shape[1],
    )


def _lognormal_prior_lambda(lambda_grid: np.ndarray, log_sd: float = PRIOR_ETA_SD) -> np.ndarray:
    values = np.asarray(lambda_grid, dtype=float)
    result = np.full_like(values, -np.inf)
    valid = values > 0.0
    eta = np.log(values[valid])
    result[valid] = (
        -np.log(values[valid] * log_sd * math.sqrt(2.0 * math.pi)) - 0.5 * (eta / log_sd) ** 2
    )
    return result


def _normalize_grid(grid: np.ndarray, logdensity: np.ndarray) -> np.ndarray:
    shifted = np.asarray(logdensity, dtype=float) - float(np.max(logdensity))
    density = np.exp(shifted)
    normalizer = float(trapezoid(density, grid))
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("grid density could not be normalized")
    return density / normalizer


def _grid_summary(grid: np.ndarray, density: np.ndarray) -> dict[str, float]:
    mean = float(trapezoid(density * grid, grid))
    variance = float(trapezoid(density * (grid - mean) ** 2, grid))
    cdf = np.zeros_like(grid)
    cdf[1:] = np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))
    cdf /= cdf[-1]
    return {
        "mean": mean,
        "sd": math.sqrt(max(variance, 0.0)),
        "q05": float(np.interp(0.05, cdf, grid)),
        "q50": float(np.interp(0.50, cdf, grid)),
        "q95": float(np.interp(0.95, cdf, grid)),
    }


def exact_verification_spine(
    frame: pd.DataFrame,
    *,
    n_grid: int,
    rho: float = PENALTY_RHO,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate the 64/remaining iid/native-price verification route.

    The full-state target is integrated analytically and independently of the
    reduced graph expression.  It is a mathematical check, not the scientific
    time-series model used for rolling predictions.
    """

    train = frame.iloc[:VERIFICATION_TRAIN_SIZE]
    heldout = frame.iloc[VERIFICATION_TRAIN_SIZE:]
    fit = fit_reference(train, response="native")
    x_heldout = design_matrix(heldout)
    reference_native = x_heldout @ fit.coefficients
    if np.any(reference_native <= 0.0):
        raise ValueError("verification held-out reference must be positive")
    scale = float(train["avgprc"].mean())
    observed = heldout["avgprc"].to_numpy(dtype=float) / scale
    reference = reference_native / scale
    sigma = fit.residual_sd / scale
    n_state = len(heldout)
    grid = np.linspace(GRID_MIN, GRID_MAX, n_grid)
    state = reference[None, :] / grid[:, None]
    standardized = (observed[None, :] - state) / sigma
    graph_log = _lognormal_prior_lambda(grid) - 0.5 * np.sum(standardized**2, axis=1)
    graph_log -= n_state * math.log(sigma * math.sqrt(2.0 * math.pi))

    sigma2 = sigma**2
    tube_variance = 1.0 / (rho * grid**2)
    total_variance = sigma2 + tube_variance
    difference = observed[None, :] - reference[None, :] / grid[:, None]
    finite_naive_log = _lognormal_prior_lambda(grid)
    finite_naive_log -= 0.5 * n_state * np.log1p(rho * grid**2 * sigma2)
    finite_naive_log -= 0.5 * np.sum(difference**2 / total_variance[:, None], axis=1)
    finite_corrected_log = finite_naive_log + n_state * np.log(grid)

    log_targets = {
        "graph_reduced": graph_log,
        "finite_naive": finite_naive_log,
        "finite_corrected": finite_corrected_log,
    }
    densities = {name: _normalize_grid(grid, values) for name, values in log_targets.items()}
    rows = []
    for name in ("graph_reduced", "finite_naive", "finite_corrected"):
        rows.append({"target": name, **_grid_summary(grid, densities[name])})
    table = pd.DataFrame(rows)
    indexed = table.set_index("target")
    graph_sd = float(indexed.loc["graph_reduced", "sd"])
    distances_naive = density_distances(grid, densities["graph_reduced"], densities["finite_naive"])
    distances_corrected = density_distances(
        grid, densities["graph_reduced"], densities["finite_corrected"]
    )
    summary = {
        "n_train": len(train),
        "n_calibration_states": n_state,
        "rho": rho,
        "graph_mean": float(indexed.loc["graph_reduced", "mean"]),
        "graph_sd": graph_sd,
        "naive_mean": float(indexed.loc["finite_naive", "mean"]),
        "corrected_mean": float(indexed.loc["finite_corrected", "mean"]),
        "naive_shift_graph_sd": (
            float(indexed.loc["finite_naive", "mean"]) - float(indexed.loc["graph_reduced", "mean"])
        )
        / graph_sd,
        "naive_total_variation_vs_graph": distances_naive["total_variation"],
        "corrected_total_variation_vs_graph": distances_corrected["total_variation"],
        "route": "independent exact product of native-price Gaussian state integrals",
    }
    arrays = {"grid": grid, **densities}
    return table, summary, arrays


def residual_pushforward_audit(n_state: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Audit c=a-lambda*u against r=c/lambda with its pushed metric.

    Under r=c/lambda the state Jacobian is -I.  The same penalty geometry is
    obtained with R_r=lambda^2 I, whose half log-determinant restores the
    n*log(lambda) contribution.  Leaving R_r=I would define a different tube,
    and is reported explicitly rather than called an invariance result.
    """

    lambdas = np.array([0.55, 0.75, 1.0, 1.25, 1.55], dtype=float)
    a = np.linspace(0.45, 1.35, n_state)
    direction = np.sin(np.linspace(0.2, 2.8, n_state)) * 0.03
    rows: list[dict[str, float]] = []
    for value in lambdas:
        u = a / value + direction
        c = a - value * u
        normalized = c / value
        original_correction = n_state * math.log(value)
        normalized_jacobian = 0.0
        pushed_metric_half_logdet = n_state * math.log(value)
        pushed_correction = normalized_jacobian + pushed_metric_half_logdet
        original_penalty = float(c @ c)
        pushed_penalty = float(value**2 * (normalized @ normalized))
        rows.append(
            {
                "lambda": value,
                "original_state_logdet": original_correction,
                "normalized_state_logdet": normalized_jacobian,
                "pushed_metric_half_logdet": pushed_metric_half_logdet,
                "pushed_total_correction": pushed_correction,
                "pushed_minus_original_correction": pushed_correction - original_correction,
                "original_penalty_quadratic": original_penalty,
                "pushed_penalty_quadratic": pushed_penalty,
                "pushed_minus_original_penalty": pushed_penalty - original_penalty,
                "unpushed_metric_changes_log_correction_by": -original_correction,
            }
        )
    table = pd.DataFrame(rows)
    summary = {
        "max_abs_correction_error": float(
            np.max(np.abs(table["pushed_minus_original_correction"]))
        ),
        "max_abs_penalty_error": float(np.max(np.abs(table["pushed_minus_original_penalty"]))),
    }
    return table, summary


def parameter_pushforward_audit(frame: pd.DataFrame, *, n_grid: int) -> dict[str, float]:
    """Check the lambda-to-eta density pushforward on the verification target."""

    train = frame.iloc[:VERIFICATION_TRAIN_SIZE]
    heldout = frame.iloc[VERIFICATION_TRAIN_SIZE:]
    fit = fit_reference(train, response="native")
    scale = float(train["avgprc"].mean())
    observed = heldout["avgprc"].to_numpy(dtype=float) / scale
    reference = design_matrix(heldout) @ fit.coefficients / scale
    sigma = fit.residual_sd / scale
    lambdas = np.linspace(GRID_MIN, GRID_MAX, n_grid)
    eta = np.log(lambdas)
    standardized = (observed[None, :] - reference[None, :] / lambdas[:, None]) / sigma
    log_lambda_density = _lognormal_prior_lambda(lambdas)
    log_lambda_density -= 0.5 * np.sum(standardized**2, axis=1)
    lambda_density = _normalize_grid(lambdas, log_lambda_density)
    eta_density = lambda_density * lambdas
    eta_density /= trapezoid(eta_density, eta)
    returned_lambda_density = eta_density / lambdas
    returned_lambda_density /= trapezoid(returned_lambda_density, lambdas)
    return {
        "lambda_to_eta_to_lambda_total_variation": density_distances(
            lambdas, lambda_density, returned_lambda_density
        )["total_variation"],
        "lambda_mean_direct": float(trapezoid(lambda_density * lambdas, lambdas)),
        "lambda_mean_via_eta": float(trapezoid(eta_density * np.exp(eta), eta)),
    }


def _multivariate_log_likelihood(
    observed_log: np.ndarray,
    state_log: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """Evaluate log N(observed_log | state_log, covariance) row-wise."""

    observed = np.asarray(observed_log, dtype=float)
    states = np.asarray(state_log, dtype=float)
    if states.ndim == 1:
        states = states[None, :]
    cho = linalg.cho_factor(covariance, lower=True, check_finite=True)
    difference = observed[None, :] - states
    solved = linalg.cho_solve(cho, difference.T).T
    quadratic = np.einsum("ij,ij->i", difference, solved)
    logdet = 2.0 * float(np.sum(np.log(np.diag(cho[0]))))
    normalization = observed.size * math.log(2.0 * math.pi) + logdet
    return -0.5 * (quadratic + normalization)


def primary_model_geometry_audit(
    frame: pd.DataFrame,
    *,
    full: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit J/q and the finite tube for the primary positive-state model.

    The first registered rolling origin is used for the numerical integration;
    the coordinate identities are algebraic and apply to every origin.  State
    prices are divided by the geometric mean of the origin's reference-training
    prices, making `u`, `a`, and `c=a-lambda*u` dimensionless.

    With direct-state base density q_u=1, J_u=lambda^n.  Under the genuine
    pushforward v=log(u), q_v=exp(sum(v)) and J_v=prod(lambda*exp(v)); evaluated
    on the graph, J_v/q_v=lambda^n=J_u/q_u.

    The normalized corrected finite tube is independently evaluated through
    w=sqrt(rho)*(a-lambda*u):

        E[L(y | (a-W/sqrt(rho))/lambda) 1(W < sqrt(rho)*a)], W~N(0,I).

    This is the direct positive-state integral after its lambda^n correction,
    not a calculation assembled from the graph likelihood.
    """

    test_position = FIRST_TEST_POSITION
    calibration_start = test_position - CALIBRATION_WINDOW
    train = frame.iloc[:calibration_start]
    calibration = frame.iloc[calibration_start:test_position]
    fit = fit_reference(train, response="log")
    scale_log = float(np.mean(np.log(train["avgprc"].to_numpy(dtype=float))))
    x_cal = design_matrix(calibration)
    reference_log_dimensionless = x_cal @ fit.coefficients - scale_log
    reference_dimensionless = np.exp(reference_log_dimensionless)
    observed_log_dimensionless = np.log(calibration["avgprc"].to_numpy(dtype=float)) - scale_log
    covariance = ar1_covariance(CALIBRATION_WINDOW, fit.residual_sd, fit.phi_used)
    posterior = eta_posterior(
        reference_log_dimensionless,
        observed_log_dimensionless,
        covariance,
    )
    tail_probability = 0.5 * (1.0 - PRIMARY_GEOMETRY_CENTRAL_PROBABILITY)
    bound = float(stats.norm.ppf(1.0 - tail_probability))
    eta_grid = np.linspace(
        posterior.mean - bound * posterior.sd,
        posterior.mean + bound * posterior.sd,
        PRIMARY_GEOMETRY_ETA_POINTS,
    )
    n_state = CALIBRATION_WINDOW
    log_reference_sum = float(np.sum(reference_log_dimensionless))
    graph_log_likelihood = np.array(
        [
            _multivariate_log_likelihood(
                observed_log_dimensionless,
                reference_log_dimensionless - eta,
                covariance,
            )[0]
            for eta in eta_grid
        ]
    )
    log_prior_eta = -0.5 * (eta_grid / PRIOR_ETA_SD) ** 2
    graph_log_target = graph_log_likelihood + log_prior_eta

    sobol_power = PRIMARY_GEOMETRY_FULL_SOBOL_POWER if full else PRIMARY_GEOMETRY_QUICK_SOBOL_POWER
    scrambles = PRIMARY_GEOMETRY_FULL_SCRAMBLES if full else PRIMARY_GEOMETRY_QUICK_SCRAMBLES
    replicate_log_targets: list[np.ndarray] = []
    invalid_fractions: list[float] = []
    sqrt_rho = math.sqrt(PRIMARY_GEOMETRY_RHO)
    for scramble in range(scrambles):
        engine = stats.qmc.Sobol(
            d=n_state,
            scramble=True,
            seed=BOOTSTRAP_SEED + scramble,
        )
        uniforms = engine.random_base2(sobol_power)
        normal_draws = stats.norm.ppf(np.clip(uniforms, 1.0e-12, 1.0 - 1.0e-12))
        positive_numerator = reference_dimensionless[None, :] - normal_draws / sqrt_rho
        valid = np.all(positive_numerator > 0.0, axis=1)
        invalid_fractions.append(1.0 - float(np.mean(valid)))
        finite_values: list[float] = []
        for eta, prior_value in zip(eta_grid, log_prior_eta, strict=True):
            state_log = np.log(positive_numerator[valid]) - eta
            log_likelihood = _multivariate_log_likelihood(
                observed_log_dimensionless,
                state_log,
                covariance,
            )
            log_expectation = float(logsumexp(log_likelihood) - math.log(len(uniforms)))
            finite_values.append(log_expectation + float(prior_value))
        replicate_log_targets.append(np.asarray(finite_values, dtype=float))

    replicate_array = np.vstack(replicate_log_targets)
    finite_log_target = logsumexp(replicate_array, axis=0) - math.log(scrambles)
    center = len(eta_grid) // 2
    graph_centered = graph_log_target - graph_log_target[center]
    finite_centered = finite_log_target - finite_log_target[center]
    replicate_centered = replicate_array - replicate_array[:, [center]]
    replicate_centered_sd = np.std(replicate_centered, axis=0, ddof=1)
    centered_error = finite_centered - graph_centered

    direct_log_q = np.zeros_like(eta_grid)
    direct_log_j = n_state * eta_grid
    log_coordinate_log_q = log_reference_sum - n_state * eta_grid
    log_coordinate_log_j = np.full_like(eta_grid, log_reference_sum)
    direct_log_ratio = direct_log_j - direct_log_q
    log_coordinate_log_ratio = log_coordinate_log_j - log_coordinate_log_q
    determinant_variation = float(np.ptp(direct_log_ratio))
    max_centered_error = float(np.max(np.abs(centered_error)))
    max_qmc_centered_sd = float(np.max(replicate_centered_sd))
    error_fraction = max_centered_error / determinant_variation
    qmc_fraction = max_qmc_centered_sd / determinant_variation
    max_ratio_error = float(np.max(np.abs(direct_log_ratio - log_coordinate_log_ratio)))

    table = pd.DataFrame(
        {
            "eta": eta_grid,
            "lambda": np.exp(eta_grid),
            "direct_log_q_on_graph": direct_log_q,
            "direct_log_J_on_graph": direct_log_j,
            "direct_log_J_over_q": direct_log_ratio,
            "log_state_log_q_on_graph": log_coordinate_log_q,
            "log_state_log_J_on_graph": log_coordinate_log_j,
            "log_state_log_J_over_q": log_coordinate_log_ratio,
            "graph_log_target_centered": graph_centered,
            "corrected_finite_tube_log_target_centered": finite_centered,
            "finite_minus_graph_centered": centered_error,
            "qmc_replicate_centered_log_sd": replicate_centered_sd,
        }
    )
    support_certificate = {
        "state_domain": "u in (0, infinity)^n",
        "u_to_zero": (
            "the Gaussian log-price likelihood decays faster than every power of u, "
            "so it is integrable at the positive boundary"
        ),
        "u_to_infinity": ("for every lambda>0, exp[-rho*||a-lambda*u||^2/2] is Gaussian-coercive"),
        "lambda_to_zero_and_infinity": (
            "the proper normal prior on eta=log(lambda) dominates polynomial tube-volume "
            "factors; the corrected hard limit is the proper conjugate eta posterior"
        ),
        "direct_to_log_pushforward": (
            "du=exp(sum(v))dv is included as q_v, so no state Jacobian is double counted"
        ),
        "finite_tube_integrable_for_every_lambda_positive": True,
        "global_eta_target_normalizable": True,
        "positive_boundary_respected": max(invalid_fractions) < 1.0e-6,
    }
    passed = bool(
        max_ratio_error < 1.0e-12
        and error_fraction < PRIMARY_GEOMETRY_MAX_ERROR_FRACTION
        and qmc_fraction < PRIMARY_GEOMETRY_MAX_ERROR_FRACTION
        and all(value for value in support_certificate.values() if isinstance(value, bool))
    )
    summary = {
        "claim_bearing_model": "rolling primary AR(1) log-price graph model",
        "origin_test_t": int(frame.iloc[test_position]["t"]),
        "n_state": n_state,
        "state_scale_native": math.exp(scale_log),
        "state_scale_source": "geometric mean price of reference-training rows 1-48",
        "direct_state_coordinate": "u=native_price/state_scale in (0,infinity)",
        "log_state_coordinate": "v=log(u)",
        "direct_state_base_density_q": "q_u(u)=1 relative to du",
        "pushed_log_state_base_density_q": "q_v(v)=exp(sum(v)) relative to dv",
        "direct_normal_factor_J": "J_u=lambda^n",
        "log_state_normal_factor_J": "J_v=prod(lambda*exp(v))",
        "required_multiplier": "J/q=lambda^n in both coordinates on the graph",
        "rho": PRIMARY_GEOMETRY_RHO,
        "rho_unit": "inverse squared dimensionless residual",
        "eta_points": len(eta_grid),
        "central_graph_probability": PRIMARY_GEOMETRY_CENTRAL_PROBABILITY,
        "sobol_power": sobol_power,
        "sobol_samples_per_scramble": 2**sobol_power,
        "sobol_scrambles": scrambles,
        "sobol_seed_sequence": [BOOTSTRAP_SEED + index for index in range(scrambles)],
        "max_invalid_positive_state_fraction": max(invalid_fractions),
        "max_abs_log_J_over_q_coordinate_error": max_ratio_error,
        "determinant_log_variation_central_region": determinant_variation,
        "max_abs_centered_finite_vs_graph_log_error": max_centered_error,
        "max_qmc_replicate_centered_log_sd": max_qmc_centered_sd,
        "finite_vs_graph_error_fraction_of_determinant_variation": error_fraction,
        "qmc_error_fraction_of_determinant_variation": qmc_fraction,
        "registered_max_error_fraction": PRIMARY_GEOMETRY_MAX_ERROR_FRACTION,
        "support_certificate": support_certificate,
        "passed": passed,
        "exact_gaussian_verification_used_as_claim_evidence": False,
    }
    return table, summary


def ar1_covariance(n: int, sigma: float, phi: float) -> np.ndarray:
    if n < 1 or sigma <= 0.0 or abs(phi) >= 1.0:
        raise ValueError("invalid stationary AR(1) covariance arguments")
    indices = np.arange(n)
    return sigma**2 * phi ** np.abs(indices[:, None] - indices[None, :])


def eta_posterior(
    reference_log: np.ndarray,
    observed_log: np.ndarray,
    covariance: np.ndarray,
    *,
    prior_sd: float = PRIOR_ETA_SD,
    omit_state_volume: bool = False,
) -> EtaPosterior:
    """Return the exact normal posterior for eta=log(lambda).

    q=log(a)-log(y) is Normal(eta*1, covariance).  The graph posterior is
    conjugate.  Omitting |det D_u c|=lambda^n tilts the eta density by
    exp(-n*eta), shifting its mean by -n times the graph variance.
    """

    a = np.asarray(reference_log, dtype=float)
    y = np.asarray(observed_log, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if a.ndim != 1 or y.shape != a.shape or covariance.shape != (a.size, a.size):
        raise ValueError("reference, observations, and covariance dimensions must match")
    ones = np.ones(a.size)
    cho = linalg.cho_factor(covariance, lower=True, check_finite=True)
    solved_ones = linalg.cho_solve(cho, ones)
    solved_q = linalg.cho_solve(cho, a - y)
    precision = 1.0 / prior_sd**2 + float(ones @ solved_ones)
    graph_variance = 1.0 / precision
    graph_mean = graph_variance * float(ones @ solved_q)
    mean = graph_mean - a.size * graph_variance if omit_state_volume else graph_mean
    return EtaPosterior(
        mean=mean,
        variance=graph_variance,
        graph_mean=graph_mean,
        graph_variance=graph_variance,
        target="naive_volume_omission" if omit_state_volume else "graph",
    )


def _joint_log_covariance(
    fit: ReferenceFit,
    x_reference: np.ndarray,
    *,
    phi: float,
    include_beta_uncertainty: bool,
) -> np.ndarray:
    covariance = ar1_covariance(len(x_reference), fit.residual_sd, phi)
    if include_beta_uncertainty:
        covariance = covariance + x_reference @ fit.coefficient_covariance @ x_reference.T
    return np.asarray(covariance, dtype=float)


def posterior_predictive_logprice(
    fit: ReferenceFit,
    x_calibration: np.ndarray,
    observed_calibration_log: np.ndarray,
    x_test: np.ndarray,
    *,
    phi: float,
    include_beta_uncertainty: bool = False,
    omit_state_volume: bool = False,
) -> tuple[EtaPosterior, LognormalPrediction]:
    """Integrate eta exactly and condition the Gaussian log-error process."""

    if fit.response != "log":
        raise ValueError("scientific prediction requires the log-price reference fit")
    x_test = np.asarray(x_test, dtype=float).reshape(1, -1)
    x_calibration = np.asarray(x_calibration, dtype=float)
    x_all = np.vstack([x_calibration, x_test])
    reference_all = x_all @ fit.coefficients
    covariance_all = _joint_log_covariance(
        fit,
        x_all,
        phi=phi,
        include_beta_uncertainty=include_beta_uncertainty,
    )
    n_cal = len(x_calibration)
    covariance_cal = covariance_all[:n_cal, :n_cal]
    posterior = eta_posterior(
        reference_all[:n_cal],
        observed_calibration_log,
        covariance_cal,
        omit_state_volume=omit_state_volume,
    )
    cross = covariance_all[n_cal, :n_cal]
    cho = linalg.cho_factor(covariance_cal, lower=True, check_finite=True)
    gain = linalg.cho_solve(cho, cross).reshape(-1)
    conditional_variance = float(
        covariance_all[n_cal, n_cal] - cross @ linalg.cho_solve(cho, cross)
    )
    if conditional_variance <= 0.0:
        raise ValueError("conditional predictive variance is not positive")
    calibration_reference = reference_all[:n_cal]
    base = float(reference_all[n_cal] + gain @ (observed_calibration_log - calibration_reference))
    eta_coefficient = -1.0 + float(gain @ np.ones(n_cal))
    predictive_mean = base + eta_coefficient * posterior.mean
    predictive_variance = conditional_variance + eta_coefficient**2 * posterior.variance
    return posterior, LognormalPrediction(
        log_mean=float(predictive_mean),
        log_sd=math.sqrt(float(predictive_variance)),
    )


def fixed_eta_predictive_logprice(
    fit: ReferenceFit,
    x_calibration: np.ndarray,
    observed_calibration_log: np.ndarray,
    x_test: np.ndarray,
    *,
    phi: float,
    eta: float = 0.0,
    include_beta_uncertainty: bool = False,
) -> LognormalPrediction:
    """Dynamic identity-reference baseline with eta fixed rather than inferred."""

    x_test = np.asarray(x_test, dtype=float).reshape(1, -1)
    x_all = np.vstack([x_calibration, x_test])
    reference = x_all @ fit.coefficients
    covariance = _joint_log_covariance(
        fit,
        x_all,
        phi=phi,
        include_beta_uncertainty=include_beta_uncertainty,
    )
    n_cal = len(x_calibration)
    covariance_cal = covariance[:n_cal, :n_cal]
    cross = covariance[n_cal, :n_cal]
    cho = linalg.cho_factor(covariance_cal, lower=True, check_finite=True)
    gain = linalg.cho_solve(cho, cross).reshape(-1)
    residual_cal = observed_calibration_log - (reference[:n_cal] - eta)
    conditional_mean = float(reference[n_cal] - eta + gain @ residual_cal)
    conditional_variance = float(covariance[n_cal, n_cal] - cross @ linalg.cho_solve(cho, cross))
    return LognormalPrediction(conditional_mean, math.sqrt(conditional_variance))


def lognormal_crps(observed: float, prediction: LognormalPrediction) -> float:
    """Exact CRPS for a lognormal predictive distribution."""

    if observed <= 0.0 or prediction.log_sd <= 0.0:
        raise ValueError("CRPS requires positive observation and predictive SD")
    mu = prediction.log_mean
    sigma = prediction.log_sd
    mean = prediction.mean
    z = (math.log(observed) - mu) / sigma
    truncated_z = (math.log(observed) - mu - sigma**2) / sigma
    expected_abs = mean * (1.0 - 2.0 * stats.norm.cdf(truncated_z))
    expected_abs += observed * (2.0 * stats.norm.cdf(z) - 1.0)
    half_pairwise_abs = mean * (2.0 * stats.norm.cdf(sigma / math.sqrt(2.0)) - 1.0)
    result = float(expected_abs - half_pairwise_abs)
    if result < -1.0e-12:
        raise ArithmeticError("lognormal CRPS formula produced a negative value")
    return max(result, 0.0)


def score_prediction(
    observed: float,
    prediction: LognormalPrediction,
    *,
    interval_level: float = INTERVAL_LEVEL,
) -> dict[str, float]:
    alpha = 1.0 - interval_level
    lower_z = float(stats.norm.ppf(alpha / 2.0))
    upper_z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    lower = math.exp(prediction.log_mean + lower_z * prediction.log_sd)
    upper = math.exp(prediction.log_mean + upper_z * prediction.log_sd)
    log_score = float(
        stats.norm.logpdf(math.log(observed), loc=prediction.log_mean, scale=prediction.log_sd)
        - math.log(observed)
    )
    return {
        "predictive_log_mean": prediction.log_mean,
        "predictive_log_sd": prediction.log_sd,
        "predictive_mean": prediction.mean,
        "predictive_median": prediction.median,
        "interval_lower": lower,
        "interval_upper": upper,
        "interval_width": upper - lower,
        "covered": float(lower <= observed <= upper),
        "log_score": log_score,
        "crps": lognormal_crps(observed, prediction),
        "absolute_error_median": abs(prediction.median - observed),
        "squared_error_mean": (prediction.mean - observed) ** 2,
    }


def _model_prediction_rows(
    frame: pd.DataFrame,
    *,
    calibration_window: int = CALIBRATION_WINDOW,
    first_test_position: int = FIRST_TEST_POSITION,
    phi_limit: float = PHI_LIMIT,
    prescore_path: Path | None = None,
) -> pd.DataFrame:
    phi_limit = _validated_phi_limit(phi_limit)
    if calibration_window <= PLACEBO_LAG + 2:
        raise ValueError("calibration window is too short for the registered placebo")
    if first_test_position - calibration_window <= len(DESIGN_COLUMNS) + 3:
        raise ValueError("initial reference block has insufficient degrees of freedom")
    rows: list[dict[str, Any]] = []
    prescore_rows: list[dict[str, Any]] = []
    x_full = design_matrix(frame)

    for test_position in range(first_test_position, len(frame)):
        calibration_start = test_position - calibration_window
        train = frame.iloc[:calibration_start]
        fit = fit_reference(train, response="log", phi_limit=phi_limit)
        x_cal = x_full[calibration_start:test_position]
        x_test = x_full[test_position]
        observed_cal = np.log(
            frame.iloc[calibration_start:test_position]["avgprc"].to_numpy(dtype=float)
        )

        configurations = (
            (PRIMARY_MODEL, fit.phi_used, False, False, x_cal, x_test),
            ("iid_graph_sensitivity", 0.0, False, False, x_cal, x_test),
            (
                "ar1_beta_uncertainty_sensitivity",
                fit.phi_used,
                True,
                False,
                x_cal,
                x_test,
            ),
            (
                "ar1_naive_volume_omission_counterfactual",
                fit.phi_used,
                False,
                True,
                x_cal,
                x_test,
            ),
            (
                "ar1_lag4_reference_placebo",
                fit.phi_used,
                False,
                False,
                x_full[calibration_start - PLACEBO_LAG : test_position - PLACEBO_LAG],
                x_full[test_position - PLACEBO_LAG],
            ),
        )
        origin_predictions: list[tuple[str, float, EtaPosterior | None, LognormalPrediction]] = []
        for (
            name,
            phi,
            beta_uncertainty,
            omit_volume,
            model_x_cal,
            model_x_test,
        ) in configurations:
            posterior, prediction = posterior_predictive_logprice(
                fit,
                model_x_cal,
                observed_cal,
                model_x_test,
                phi=phi,
                include_beta_uncertainty=beta_uncertainty,
                omit_state_volume=omit_volume,
            )
            origin_predictions.append((name, phi, posterior, prediction))

        matched_identities = (
            (
                "ar1_identity_reference_baseline",
                fit.phi_used,
                False,
            ),
            (
                "iid_identity_reference_baseline",
                0.0,
                False,
            ),
            (
                "ar1_beta_uncertainty_identity_baseline",
                fit.phi_used,
                True,
            ),
        )
        for name, phi, beta_uncertainty in matched_identities:
            prediction = fixed_eta_predictive_logprice(
                fit,
                x_cal,
                observed_cal,
                x_test,
                phi=phi,
                eta=0.0,
                include_beta_uncertainty=beta_uncertainty,
            )
            origin_predictions.append((name, phi, None, prediction))

        history_log = np.log(frame.iloc[:test_position]["avgprc"].to_numpy(dtype=float))
        difference_sd = float(np.std(np.diff(history_log), ddof=1))
        persistence = LognormalPrediction(
            log_mean=float(history_log[-1]),
            log_sd=max(difference_sd, 1.0e-8),
        )
        origin_predictions.append(("last_price_persistence_baseline", 0.0, None, persistence))

        origin_prescore: list[dict[str, Any]] = []
        for name, phi, posterior, prediction in origin_predictions:
            record = _prediction_metadata(
                frame,
                test_position,
                calibration_start,
                fit,
                name,
                prediction,
                posterior=posterior,
                phi=phi,
                calibration_window=calibration_window,
            )
            record["predictive_log_mean"] = prediction.log_mean
            record["predictive_log_sd"] = prediction.log_sd
            record["prediction_sha256"] = canonical_json_sha256(record)
            origin_prescore.append(record)
        prescore_rows.extend(origin_prescore)
        if prescore_path is not None:
            _write_csv(pd.DataFrame(prescore_rows), prescore_path)

        observed_test = float(frame.iloc[test_position]["avgprc"])
        for (name, phi, posterior, prediction), prescore in zip(
            origin_predictions, origin_prescore, strict=True
        ):
            rows.append(
                _prediction_row(
                    frame,
                    test_position,
                    calibration_start,
                    fit,
                    name,
                    observed_test,
                    prediction,
                    posterior=posterior,
                    phi=phi,
                    calibration_window=calibration_window,
                    prediction_sha256=str(prescore["prediction_sha256"]),
                )
            )
    result = pd.DataFrame(rows)
    result["model"] = pd.Categorical(result["model"], categories=MODEL_ORDER, ordered=True)
    return result.sort_values(["test_t", "model"], kind="stable").reset_index(drop=True)


def _prediction_row(
    frame: pd.DataFrame,
    test_position: int,
    calibration_start: int,
    fit: ReferenceFit,
    model: str,
    observed: float,
    prediction: LognormalPrediction,
    *,
    posterior: EtaPosterior | None,
    phi: float,
    calibration_window: int,
    prediction_sha256: str,
) -> dict[str, Any]:
    score = score_prediction(observed, prediction)
    return {
        **_prediction_metadata(
            frame,
            test_position,
            calibration_start,
            fit,
            model,
            prediction,
            posterior=posterior,
            phi=phi,
            calibration_window=calibration_window,
        ),
        "prediction_sha256_before_test_score": prediction_sha256,
        "observed_avgprc": observed,
        **score,
    }


def _prediction_metadata(
    frame: pd.DataFrame,
    test_position: int,
    calibration_start: int,
    fit: ReferenceFit,
    model: str,
    prediction: LognormalPrediction,
    *,
    posterior: EtaPosterior | None,
    phi: float,
    calibration_window: int,
) -> dict[str, Any]:
    innovation_sd = fit.residual_sd * math.sqrt(max(1.0 - phi**2, 0.0))
    return {
        "model": model,
        "test_position_zero_based": test_position,
        "source_row": int(frame.iloc[test_position]["rownames"]),
        "test_t": int(frame.iloc[test_position]["t"]),
        "reference_train_t_start": int(frame.iloc[0]["t"]),
        "reference_train_t_end": int(frame.iloc[calibration_start - 1]["t"]),
        "n_reference_train": calibration_start,
        "calibration_t_start": int(frame.iloc[calibration_start]["t"]),
        "calibration_t_end": int(frame.iloc[test_position - 1]["t"]),
        "n_calibration": calibration_window,
        "stationary_marginal_residual_sd_log_plugin": fit.residual_sd,
        "derived_innovation_sd_log_plugin": innovation_sd,
        "ar1_phi_raw_training": fit.phi_raw,
        "ar1_phi_used": phi,
        "eta_posterior_mean": None if posterior is None else posterior.mean,
        "eta_posterior_sd": None if posterior is None else posterior.sd,
        "lambda_posterior_median": None if posterior is None else math.exp(posterior.mean),
        "predictive_log_mean": prediction.log_mean,
        "predictive_log_sd": prediction.log_sd,
    }


def aggregate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        subset = predictions.loc[predictions["model"] == model]
        if subset.empty:
            continue
        rows.append(
            {
                "model": model,
                "n_forecasts": len(subset),
                "mean_log_score": float(subset["log_score"].mean()),
                "total_log_score": float(subset["log_score"].sum()),
                "mean_crps": float(subset["crps"].mean()),
                "mae_predictive_median": float(subset["absolute_error_median"].mean()),
                "rmse_predictive_mean": math.sqrt(float(subset["squared_error_mean"].mean())),
                "interval_level": INTERVAL_LEVEL,
                "interval_coverage": float(subset["covered"].mean()),
                "mean_interval_width": float(subset["interval_width"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _moving_block_means(
    values: np.ndarray,
    *,
    replicates: int,
    block_length: int,
    seed: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    result = np.empty(replicates, dtype=float)
    offsets = np.arange(block_length)
    blocks_needed = math.ceil(values.size / block_length)
    for index in range(replicates):
        starts = rng.integers(0, values.size, size=blocks_needed)
        sampled = np.concatenate([(start + offsets) % values.size for start in starts])
        result[index] = float(np.mean(values[sampled[: values.size]]))
    return result


def paired_score_uncertainty(
    predictions: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    block_length: int = 4,
) -> pd.DataFrame:
    primary = predictions.loc[predictions["model"] == PRIMARY_MODEL].set_index("test_t")
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        if model == PRIMARY_MODEL:
            continue
        comparator = predictions.loc[predictions["model"] == model].set_index("test_t")
        if not primary.index.equals(comparator.index):
            raise ValueError("paired comparison forecast origins differ")
        contrasts = {
            "log_score_primary_minus_comparator": (
                primary["log_score"] - comparator["log_score"]
            ).to_numpy(dtype=float),
            "crps_comparator_minus_primary": (comparator["crps"] - primary["crps"]).to_numpy(
                dtype=float
            ),
        }
        for metric, values in contrasts.items():
            boot = _moving_block_means(
                values,
                replicates=bootstrap_replicates,
                block_length=block_length,
                seed=BOOTSTRAP_SEED,
            )
            rows.append(
                {
                    "comparator": model,
                    "contrast": metric,
                    "positive_favors_primary": True,
                    "mean_contrast": float(np.mean(values)),
                    "moving_block_bootstrap_ci05": float(
                        np.quantile(boot, BOOTSTRAP_LOWER_QUANTILE)
                    ),
                    "moving_block_bootstrap_ci95": float(
                        np.quantile(boot, BOOTSTRAP_UPPER_QUANTILE)
                    ),
                    "n_origins": len(values),
                    "block_length": block_length,
                    "bootstrap_replicates": bootstrap_replicates,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "interval_level": 0.90,
                }
            )
    return pd.DataFrame(rows)


def matched_calibration_effects(
    predictions: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    block_length: int = 4,
) -> pd.DataFrame:
    """Compare each graph calibration with identity under matched assumptions."""

    rows: list[dict[str, Any]] = []
    for family, graph_model, identity_model in MATCHED_CALIBRATION_COMPARISONS:
        graph = predictions.loc[predictions["model"] == graph_model].set_index("test_t")
        identity = predictions.loc[predictions["model"] == identity_model].set_index("test_t")
        if not graph.index.equals(identity.index):
            raise ValueError(f"matched origins differ for {family}")
        delta_elpd = (graph["log_score"] - identity["log_score"]).to_numpy(dtype=float)
        bootstrap = _moving_block_means(
            delta_elpd,
            replicates=bootstrap_replicates,
            block_length=block_length,
            seed=BOOTSTRAP_SEED,
        )
        rows.append(
            {
                "assumption_family": family,
                "graph_model": graph_model,
                "matched_identity_model": identity_model,
                "n_origins": len(delta_elpd),
                "cumulative_delta_elpd_graph_minus_identity": float(np.sum(delta_elpd)),
                "mean_delta_elpd_graph_minus_identity": float(np.mean(delta_elpd)),
                "moving_block_90pct_mean_delta_elpd_lower": float(
                    np.quantile(bootstrap, BOOTSTRAP_LOWER_QUANTILE)
                ),
                "moving_block_90pct_mean_delta_elpd_upper": float(
                    np.quantile(bootstrap, BOOTSTRAP_UPPER_QUANTILE)
                ),
                "graph_mean_crps": float(graph["crps"].mean()),
                "identity_mean_crps": float(identity["crps"].mean()),
                "crps_identity_minus_graph": float(identity["crps"].mean() - graph["crps"].mean()),
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_replicates": bootstrap_replicates,
                "block_length": block_length,
                "interval_level": 0.90,
            }
        )
    return pd.DataFrame(rows)


def phi_clip_sensitivity(
    frame: pd.DataFrame,
    primary_predictions: pd.DataFrame,
    primary_matched_effects: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    limits: tuple[float, ...] = PHI_CLIP_SENSITIVITY_LIMITS,
    block_length: int = 4,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate the declared AR(1) clip bounds one axis at a time.

    This is a post-outcome robustness sensitivity. It never chooses or replaces
    the primary ``PHI_LIMIT`` analysis.  Every alternative reuses the same
    observations, roles, prior, covariance convention, scores, bootstrap seed,
    and matched identity construction; only the symmetric absolute clipping
    bound applied to the training-only plug-in ``phi`` estimate changes.
    """

    validated_limits = tuple(_validated_phi_limit(value) for value in limits)
    if len(set(validated_limits)) != len(validated_limits):
        raise ValueError("phi clip sensitivity limits must be unique")
    if PHI_LIMIT not in validated_limits:
        raise ValueError("phi clip sensitivity must include the primary bound")
    if bootstrap_replicates < 1 or block_length < 1:
        raise ValueError("sensitivity bootstrap settings must be positive")

    primary_graph = primary_predictions.loc[
        primary_predictions["model"] == PRIMARY_MODEL
    ].set_index("test_t")
    primary_eta = primary_graph["eta_posterior_mean"].to_numpy(dtype=float)
    primary_eta_sd = primary_graph["eta_posterior_sd"].to_numpy(dtype=float)
    if np.any(primary_eta_sd <= 0.0):
        raise ValueError("primary eta posterior SD must be positive")

    rows: list[dict[str, Any]] = []
    for limit in validated_limits:
        if limit == PHI_LIMIT:
            predictions = primary_predictions
            matched_effects = primary_matched_effects
        else:
            predictions = _model_prediction_rows(frame, phi_limit=limit)
            matched_effects = matched_calibration_effects(
                predictions,
                bootstrap_replicates=bootstrap_replicates,
                block_length=block_length,
            )

        matched = matched_effects.set_index("assumption_family")
        ar1 = matched.loc["ar1_plugin"]
        iid = matched.loc["iid"]
        graph = predictions.loc[predictions["model"] == PRIMARY_MODEL].set_index("test_t")
        identity = predictions.loc[
            predictions["model"] == "ar1_identity_reference_baseline"
        ].set_index("test_t")
        placebo = predictions.loc[predictions["model"] == "ar1_lag4_reference_placebo"].set_index(
            "test_t"
        )
        naive = predictions.loc[
            predictions["model"] == "ar1_naive_volume_omission_counterfactual"
        ].set_index("test_t")
        if not (
            graph.index.equals(identity.index)
            and graph.index.equals(placebo.index)
            and graph.index.equals(naive.index)
            and graph.index.equals(primary_graph.index)
        ):
            raise ValueError("phi clip sensitivity forecast origins differ")

        raw_phi = graph["ar1_phi_raw_training"].to_numpy(dtype=float)
        used_phi = graph["ar1_phi_used"].to_numpy(dtype=float)
        eta = graph["eta_posterior_mean"].to_numpy(dtype=float)
        eta_sd = graph["eta_posterior_sd"].to_numpy(dtype=float)
        naive_eta = naive["eta_posterior_mean"].to_numpy(dtype=float)
        graph_naive_delta = graph["log_score"].to_numpy(dtype=float) - naive["log_score"].to_numpy(
            dtype=float
        )
        graph_naive_bootstrap = _moving_block_means(
            graph_naive_delta,
            replicates=bootstrap_replicates,
            block_length=block_length,
            seed=BOOTSTRAP_SEED,
        )

        cumulative_positive = bool(ar1["cumulative_delta_elpd_graph_minus_identity"] > 0.0)
        interval_lower_positive = bool(ar1["moving_block_90pct_mean_delta_elpd_lower"] > 0.0)
        crps_no_worse = bool(ar1["crps_identity_minus_graph"] >= 0.0)
        reference_beats_placebo = bool(graph["log_score"].sum() > placebo["log_score"].sum())
        predictive_gate_passed = bool(
            cumulative_positive
            and interval_lower_positive
            and crps_no_worse
            and reference_beats_placebo
        )
        direction_reversal = bool(
            np.sign(float(ar1["mean_delta_elpd_graph_minus_identity"]))
            != np.sign(float(iid["mean_delta_elpd_graph_minus_identity"]))
        )
        interval_gate_change = bool(
            (ar1["moving_block_90pct_mean_delta_elpd_lower"] > 0.0)
            != (iid["moving_block_90pct_mean_delta_elpd_lower"] > 0.0)
        )
        volume_shift = (naive_eta - eta) / eta_sd
        eta_change = (eta - primary_eta) / primary_eta_sd
        rows.append(
            {
                "phi_absolute_limit": limit,
                "analysis_role": (
                    "primary"
                    if limit == PHI_LIMIT
                    else ("tighter_stress" if limit < PHI_LIMIT else "looser_boundary")
                ),
                "post_outcome_nonselecting_addition": True,
                "non_claim_selecting": True,
                "n_origins": len(graph),
                "n_origins_clipped": int(np.count_nonzero(np.abs(raw_phi) > limit)),
                "min_phi_raw": float(np.min(raw_phi)),
                "max_phi_raw": float(np.max(raw_phi)),
                "max_abs_phi_raw": float(np.max(np.abs(raw_phi))),
                "min_phi_used": float(np.min(used_phi)),
                "max_phi_used": float(np.max(used_phi)),
                "cumulative_delta_elpd_graph_minus_identity": float(
                    ar1["cumulative_delta_elpd_graph_minus_identity"]
                ),
                "mean_delta_elpd_graph_minus_identity": float(
                    ar1["mean_delta_elpd_graph_minus_identity"]
                ),
                "moving_block_90pct_mean_delta_elpd_lower": float(
                    ar1["moving_block_90pct_mean_delta_elpd_lower"]
                ),
                "moving_block_90pct_mean_delta_elpd_upper": float(
                    ar1["moving_block_90pct_mean_delta_elpd_upper"]
                ),
                "crps_identity_minus_graph": float(ar1["crps_identity_minus_graph"]),
                "cumulative_delta_elpd_graph_minus_lag4": float(
                    graph["log_score"].sum() - placebo["log_score"].sum()
                ),
                "predictive_gate_cumulative_delta_elpd_positive": cumulative_positive,
                "predictive_gate_interval_lower_positive": interval_lower_positive,
                "predictive_gate_graph_crps_no_worse": crps_no_worse,
                "predictive_gate_reference_beats_lag4": reference_beats_placebo,
                "predictive_gate_passed": predictive_gate_passed,
                "ar1_iid_direction_reversal": direction_reversal,
                "ar1_iid_interval_gate_change": interval_gate_change,
                "dependence_sensitive": bool(direction_reversal or interval_gate_change),
                "cumulative_delta_elpd_graph_minus_naive": float(np.sum(graph_naive_delta)),
                "moving_block_90pct_mean_delta_elpd_graph_minus_naive_lower": float(
                    np.quantile(graph_naive_bootstrap, BOOTSTRAP_LOWER_QUANTILE)
                ),
                "moving_block_90pct_mean_delta_elpd_graph_minus_naive_upper": float(
                    np.quantile(graph_naive_bootstrap, BOOTSTRAP_UPPER_QUANTILE)
                ),
                "mean_eta_volume_shift_in_graph_sd": float(np.mean(volume_shift)),
                "max_abs_eta_volume_shift_in_graph_sd": float(np.max(np.abs(volume_shift))),
                "max_abs_eta_change_vs_primary_in_primary_sd": float(np.max(np.abs(eta_change))),
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_replicates": bootstrap_replicates,
                "block_length": block_length,
            }
        )

    table = (
        pd.DataFrame(rows).sort_values("phi_absolute_limit", kind="stable").reset_index(drop=True)
    )
    primary_row = table.loc[table["phi_absolute_limit"] == PHI_LIMIT].iloc[0]
    numeric = table.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    summary = {
        "role": (
            "post-outcome one-at-a-time robustness sensitivity; alternatives do not select "
            "or replace the primary analysis"
        ),
        "timing": PHI_CLIP_SENSITIVITY_TIMING,
        "outcomes_known_when_added": True,
        "primary_absolute_limit": PHI_LIMIT,
        "absolute_limits": list(validated_limits),
        "single_changed_axis": "symmetric absolute clipping bound for training-only phi",
        "held_fixed": (
            "data, temporal roles, reference fit, stationary marginal residual SD, prior, "
            "calibration window, scores, matched comparators, bootstrap seed and replicates"
        ),
        "primary_clipping_active": bool(primary_row["n_origins_clipped"] > 0),
        "primary_n_origins_clipped": int(primary_row["n_origins_clipped"]),
        "raw_phi_range": [
            float(primary_row["min_phi_raw"]),
            float(primary_row["max_phi_raw"]),
        ],
        "all_predictive_gate_decisions_match_primary": bool(
            table["predictive_gate_passed"].nunique() == 1
        ),
        "all_dependence_sensitivity_categories_match_primary": bool(
            table["dependence_sensitive"].nunique() == 1
        ),
        "all_geometry_consequences_finite": bool(
            np.all(
                np.isfinite(
                    table[
                        [
                            "cumulative_delta_elpd_graph_minus_naive",
                            "mean_eta_volume_shift_in_graph_sd",
                            "max_abs_eta_volume_shift_in_graph_sd",
                        ]
                    ].to_numpy(dtype=float)
                )
            )
        ),
        "maximum_abs_eta_change_vs_primary_in_primary_sd": float(
            table["max_abs_eta_change_vs_primary_in_primary_sd"].max()
        ),
        "all_numeric_outputs_finite": bool(np.all(np.isfinite(numeric))),
        "scientific_outcomes_are_not_implementation_pass_fail_checks": True,
        "table": "tables/phi_clip_sensitivity.csv",
    }
    return table, summary


def evaluate_claims(
    predictions: pd.DataFrame,
    matched_effects: pd.DataFrame,
    *,
    primary_geometry: dict[str, Any],
    geometry_checks_passed: bool,
    bootstrap_replicates: int,
    full: bool,
    registered_fixture: bool,
) -> list[dict[str, Any]]:
    """Evaluate every registered Fulton claim without outcome-driven wording."""

    matched = matched_effects.set_index("assumption_family")
    ar1 = matched.loc["ar1_plugin"]
    iid = matched.loc["iid"]
    beta = matched.loc["ar1_beta_uncertainty"]
    primary = predictions.loc[predictions["model"] == PRIMARY_MODEL].set_index("test_t")
    placebo = predictions.loc[predictions["model"] == "ar1_lag4_reference_placebo"].set_index(
        "test_t"
    )
    persistence = predictions.loc[
        predictions["model"] == "last_price_persistence_baseline"
    ].set_index("test_t")
    naive = predictions.loc[
        predictions["model"] == "ar1_naive_volume_omission_counterfactual"
    ].set_index("test_t")
    iid_graph = predictions.loc[predictions["model"] == "iid_graph_sensitivity"].set_index("test_t")
    beta_graph = predictions.loc[
        predictions["model"] == "ar1_beta_uncertainty_sensitivity"
    ].set_index("test_t")

    pred_gate = {
        "cumulative_delta_elpd_positive": bool(
            ar1["cumulative_delta_elpd_graph_minus_identity"] > 0.0
        ),
        "moving_block_90pct_lower_positive": bool(
            ar1["moving_block_90pct_mean_delta_elpd_lower"] > 0.0
        ),
        "graph_crps_no_worse_than_identity": bool(ar1["crps_identity_minus_graph"] >= 0.0),
        "contemporaneous_reference_beats_lag4_log_score": bool(
            primary["log_score"].sum() > placebo["log_score"].sum()
        ),
    }
    pred_gate_passed = all(pred_gate.values())
    if not registered_fixture:
        pred_status = "EXPLORATORY"
    elif not pred_gate_passed:
        pred_status = "NOT_SUPPORTED"
    elif not full:
        pred_status = "MIXED"
    else:
        pred_status = "SUPPORTED"

    ar1_direction = float(ar1["mean_delta_elpd_graph_minus_identity"])
    iid_direction = float(iid["mean_delta_elpd_graph_minus_identity"])
    beta_direction = float(beta["mean_delta_elpd_graph_minus_identity"])
    direction_reversal = bool(np.sign(ar1_direction) != np.sign(iid_direction))
    matched_gate_change = bool(
        (ar1["moving_block_90pct_mean_delta_elpd_lower"] > 0.0)
        != (iid["moving_block_90pct_mean_delta_elpd_lower"] > 0.0)
    )
    dependence_sensitive = direction_reversal or matched_gate_change
    dep_status = "EXPLORATORY" if not registered_fixture else ("SUPPORTED" if full else "MIXED")
    dep_conclusion = (
        "DEPENDENCE_SENSITIVE"
        if dependence_sensitive
        else "STABLE_DIRECTION_UNDER_REGISTERED_IID_AR1_CHECK"
    )

    graph_naive_delta = primary["log_score"] - naive["log_score"]
    graph_naive_bootstrap = _moving_block_means(
        graph_naive_delta.to_numpy(dtype=float),
        replicates=bootstrap_replicates,
        block_length=4,
        seed=BOOTSTRAP_SEED,
    )
    graph_eta = primary["eta_posterior_mean"].to_numpy(dtype=float)
    naive_eta = naive["eta_posterior_mean"].to_numpy(dtype=float)
    graph_eta_sd = primary["eta_posterior_sd"].to_numpy(dtype=float)
    standardized_shift = (naive_eta - graph_eta) / graph_eta_sd
    model_sensitivity_envelope = max(
        float(
            np.max(
                np.abs(
                    (iid_graph["eta_posterior_mean"].to_numpy(dtype=float) - graph_eta)
                    / graph_eta_sd
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    (beta_graph["eta_posterior_mean"].to_numpy(dtype=float) - graph_eta)
                    / graph_eta_sd
                )
            )
        ),
    )
    geometry_shift_max = float(np.max(np.abs(standardized_shift)))
    if not registered_fixture:
        geom_status = "EXPLORATORY"
    elif not geometry_checks_passed:
        geom_status = "BLOCKED"
    else:
        geom_status = "SUPPORTED" if full else "MIXED"
    return [
        {
            "claim_id": "FU-PRED",
            "status": pred_status,
            "conclusion": (
                "PREDICTIVE_IMPROVEMENT_ESTABLISHED"
                if pred_status == "SUPPORTED"
                else "PREDICTIVE_IMPROVEMENT_NOT_ESTABLISHED"
            ),
            "evidence_level": "E2_conditional_registered_fixture",
            "claim_bearing_full_run": bool(full and registered_fixture),
            "gate_passed": pred_gate_passed,
            "gate_components": pred_gate,
            "matched_assumption_family": "ar1_plugin",
            "cumulative_delta_elpd_graph_minus_identity": float(
                ar1["cumulative_delta_elpd_graph_minus_identity"]
            ),
            "moving_block_90pct_mean_delta_elpd": [
                float(ar1["moving_block_90pct_mean_delta_elpd_lower"]),
                float(ar1["moving_block_90pct_mean_delta_elpd_upper"]),
            ],
            "crps_identity_minus_graph": float(ar1["crps_identity_minus_graph"]),
            "cumulative_delta_elpd_graph_minus_lag4": float(
                primary["log_score"].sum() - placebo["log_score"].sum()
            ),
            "cumulative_delta_elpd_graph_minus_persistence_context": float(
                primary["log_score"].sum() - persistence["log_score"].sum()
            ),
            "interpretation": (
                "The registered graph calibration did not satisfy every predictive-improvement "
                "gate on the current short retrospective series."
                if pred_status == "NOT_SUPPORTED"
                else "The status follows the locked predictive gate."
            ),
            "non_claim": (
                "Failure to establish improvement does not establish that scale one is true, "
                "nor does any result identify market structure."
            ),
        },
        {
            "claim_id": "FU-DEP",
            "status": dep_status,
            "conclusion": dep_conclusion,
            "evidence_level": "E2_matched_plugin_sensitivity",
            "claim_bearing_full_run": bool(full and registered_fixture),
            "matched_mean_delta_elpd": {
                "ar1_plugin": ar1_direction,
                "iid": iid_direction,
                "ar1_beta_uncertainty": beta_direction,
            },
            "ar1_iid_direction_reversal": direction_reversal,
            "ar1_iid_interval_gate_change": matched_gate_change,
            "phi_is_plugin_not_posterior": True,
            "interpretation": (
                "The estimated calibration contribution changes direction between matched "
                "AR(1) and iid comparisons."
                if direction_reversal
                else (
                    "The matched AR(1) and iid comparisons retain the same point direction, "
                    "but their registered interval-gate decisions differ."
                    if matched_gate_change
                    else (
                        "The matched AR(1) and iid calibration effects retain the same "
                        "direction and interval-gate decision."
                    )
                )
            ),
            "non_claim": (
                "This comparison does not establish that stationary AR(1) errors are true or "
                "that phi has a posterior distribution."
            ),
        },
        {
            "claim_id": "FU-GEOM",
            "status": geom_status,
            "conclusion": ("QUANTIFIED" if geom_status in {"SUPPORTED", "MIXED"} else "BLOCKED"),
            "evidence_level": "E2_analytic_and_callable_geometry_audit",
            "claim_bearing_full_run": bool(full and registered_fixture),
            "state_base_measure": "product Lebesgue du over direct positive price states",
            "state_base_density_q": "q(u)=1 in the declared direct-price coordinate",
            "normal_factor_J": "J(lambda)=lambda^n",
            "required_graph_multiplier": "J/q=lambda^n",
            "primary_model_coordinate_audit": {
                "origin_test_t": primary_geometry["origin_test_t"],
                "max_abs_log_J_over_q_coordinate_error": primary_geometry[
                    "max_abs_log_J_over_q_coordinate_error"
                ],
                "finite_vs_graph_error_fraction_of_determinant_variation": (
                    primary_geometry["finite_vs_graph_error_fraction_of_determinant_variation"]
                ),
                "qmc_error_fraction_of_determinant_variation": primary_geometry[
                    "qmc_error_fraction_of_determinant_variation"
                ],
                "support_certificate": primary_geometry["support_certificate"],
                "passed": primary_geometry["passed"],
            },
            "cumulative_delta_elpd_graph_minus_naive": float(graph_naive_delta.sum()),
            "moving_block_90pct_mean_delta_elpd_graph_minus_naive": [
                float(np.quantile(graph_naive_bootstrap, BOOTSTRAP_LOWER_QUANTILE)),
                float(np.quantile(graph_naive_bootstrap, BOOTSTRAP_UPPER_QUANTILE)),
            ],
            "crps_naive_minus_graph": float(naive["crps"].mean() - primary["crps"].mean()),
            "mean_eta_shift_in_graph_sd": float(np.mean(standardized_shift)),
            "max_abs_eta_shift_in_graph_sd": geometry_shift_max,
            "model_sensitivity_envelope_in_graph_sd": model_sensitivity_envelope,
            "interpretation": (
                "The omitted J/q factor changes the declared rolling target and its predictive "
                "distribution by the reported amounts."
            ),
            "non_claim": (
                "The determinant is a measure correction, not an economic mechanism or causal "
                "price effect."
            ),
        },
    ]


def residual_diagnostics(
    frame: pd.DataFrame, *, max_lag: int = 12
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Report, but do not select models from, initial training residual diagnostics."""

    fit = fit_reference(frame, response="log")
    residual = fit.residuals
    centered = residual - float(np.mean(residual))
    denominator = float(centered @ centered)
    max_lag = min(max_lag, len(residual) - 2)
    acf_rows = []
    for lag in range(max_lag + 1):
        correlation = 1.0 if lag == 0 else float(centered[lag:] @ centered[:-lag]) / denominator
        acf_rows.append({"lag": lag, "sample_acf": correlation})
    acf = pd.DataFrame(acf_rows)
    positive_lags = acf.loc[acf["lag"] > 0]
    q = (
        len(residual)
        * (len(residual) + 2.0)
        * float(
            np.sum(
                positive_lags["sample_acf"].to_numpy() ** 2
                / (len(residual) - positive_lags["lag"].to_numpy())
            )
        )
    )
    q_pvalue = float(stats.chi2.sf(q, max_lag))
    fitted = design_matrix(frame) @ fit.coefficients
    time = frame["t"].to_numpy(dtype=float)
    squared = residual**2
    spearman_fitted = stats.spearmanr(fitted, squared)
    spearman_time = stats.spearmanr(time, squared)
    durbin_watson = float(np.sum(np.diff(residual) ** 2) / np.sum(residual**2))
    innovation_sd = fit.residual_sd * math.sqrt(1.0 - fit.phi_used**2)
    diagnostics = pd.DataFrame(
        [
            {"diagnostic": "n_initial_reference_train", "value": len(frame)},
            {
                "diagnostic": "stationary_marginal_log_residual_sd_plugin",
                "value": fit.residual_sd,
            },
            {
                "diagnostic": "derived_log_innovation_sd_plugin",
                "value": innovation_sd,
            },
            {"diagnostic": "reference_r2_log", "value": fit.r2},
            {"diagnostic": "residual_mean", "value": float(np.mean(residual))},
            {"diagnostic": "residual_skew", "value": float(stats.skew(residual))},
            {
                "diagnostic": "residual_excess_kurtosis",
                "value": float(stats.kurtosis(residual)),
            },
            {"diagnostic": "durbin_watson", "value": durbin_watson},
            {"diagnostic": "ar1_phi_raw", "value": fit.phi_raw},
            {"diagnostic": "ar1_phi_used", "value": fit.phi_used},
            {"diagnostic": f"ljung_box_q_lag_{max_lag}", "value": q},
            {"diagnostic": f"ljung_box_p_lag_{max_lag}", "value": q_pvalue},
            {
                "diagnostic": "spearman_squared_residual_vs_fitted",
                "value": float(spearman_fitted.statistic),
            },
            {
                "diagnostic": "spearman_squared_residual_vs_fitted_p",
                "value": float(spearman_fitted.pvalue),
            },
            {
                "diagnostic": "spearman_squared_residual_vs_time",
                "value": float(spearman_time.statistic),
            },
            {
                "diagnostic": "spearman_squared_residual_vs_time_p",
                "value": float(spearman_time.pvalue),
            },
        ]
    )
    summary = {
        "phi_raw": fit.phi_raw,
        "phi_used": fit.phi_used,
        "stationary_marginal_log_residual_sd_plugin": fit.residual_sd,
        "derived_log_innovation_sd_plugin": innovation_sd,
        "phi_is_training_only_plugin_not_posterior": True,
        "durbin_watson": durbin_watson,
        "ljung_box_q": q,
        "ljung_box_p_approximate": q_pvalue,
        "max_lag": max_lag,
    }
    return diagnostics, acf, summary


def detcorr_callable_audit(lambda_scale: float = 0.79, n_state: int = 7) -> dict[str, Any]:
    """Cross-check detcorr>=0.10 against the analytic diagonal state Jacobian."""

    try:
        import detcorr
        from detcorr import evaluate_corrected_finite_penalty_target
    except Exception as exc:  # pragma: no cover - environment-dependent failure surface
        return {"available": False, "reason": repr(exc)}

    version = str(getattr(detcorr, "__version__", "unknown"))
    version_match = re.match(r"^(\d+)\.(\d+)", version)
    version_compatible = bool(
        version_match and (int(version_match.group(1)), int(version_match.group(2))) >= (0, 10)
    )
    reference = np.linspace(0.55, 1.25, n_state)
    projected = reference / lambda_scale
    state = projected + 1.0e-5 * np.cos(np.linspace(0.1, 1.7, n_state))

    def residual(theta: float, state_value: np.ndarray) -> np.ndarray:
        return reference - float(theta) * np.asarray(state_value, dtype=float)

    evaluation = evaluate_corrected_finite_penalty_target(
        residual,
        float(lambda_scale),
        state,
        rho=2.0e5,
        target_measure="graph",
        base_logdensity=0.0,
        projected_state=projected,
        log_correction_kwargs={
            "backend": "scipy",
            "finite_difference_scheme": "central",
            "singular_policy": "raise",
        },
        metadata={
            "state_coordinate": "dimensionless_price",
            "residual_coordinate": "dimensionless_price",
            "analytic_state_jacobian": "-lambda*I",
        },
    )
    expected = n_state * math.log(lambda_scale)
    observed = float(evaluation.primary_correction)
    return {
        "available": True,
        "detcorr_version": version,
        "version_is_at_least_0_10": version_compatible,
        "route": "detcorr callable residual with scipy central finite differences",
        "lambda": lambda_scale,
        "n_state": n_state,
        "expected_state_logdet": expected,
        "observed_state_logdet": observed,
        "absolute_error": abs(observed - expected),
        "target_measure": evaluation.target_measure,
    }


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def _plot_residual_acf(path: Path, acf: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.vlines(acf["lag"], 0.0, acf["sample_acf"], color="tab:blue", linewidth=1.6)
    ax.scatter(acf["lag"], acf["sample_acf"], color="tab:blue", s=24)
    bound = 1.96 / math.sqrt(48.0)
    ax.axhline(bound, color="tab:red", linestyle=":", linewidth=1.0)
    ax.axhline(-bound, color="tab:red", linestyle=":", linewidth=1.0)
    ax.set_xlabel("lag (market observations)")
    ax.set_ylabel("sample residual ACF")
    ax.set_title("Initial training-only log-reference residuals")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, metadata={"Software": "detcorr scientific case studies"})
    plt.close(fig)


def _plot_rolling_predictions(path: Path, predictions: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    primary = predictions.loc[predictions["model"] == PRIMARY_MODEL].sort_values("test_t")
    identity = predictions.loc[
        predictions["model"] == "ar1_identity_reference_baseline"
    ].sort_values("test_t")
    persistence = predictions.loc[
        predictions["model"] == "last_price_persistence_baseline"
    ].sort_values("test_t")
    x = primary["test_t"].to_numpy(dtype=float)
    ax.fill_between(
        x,
        primary["interval_lower"].to_numpy(dtype=float),
        primary["interval_upper"].to_numpy(dtype=float),
        color="tab:blue",
        alpha=0.16,
        label="primary 90% predictive interval",
    )
    ax.plot(
        x,
        primary["observed_avgprc"],
        color="black",
        marker="o",
        markersize=3,
        label="observed",
    )
    ax.plot(
        x,
        primary["predictive_median"],
        color="tab:blue",
        linewidth=1.5,
        label="AR(1) graph median",
    )
    ax.plot(
        x,
        identity["predictive_median"],
        color="tab:orange",
        linestyle="--",
        linewidth=1.1,
        label="identity dynamic baseline",
    )
    ax.plot(
        x,
        persistence["predictive_median"],
        color="0.5",
        linestyle=":",
        linewidth=1.1,
        label="last-price baseline",
    )
    ax.set_xlabel("chronological observation t (each point untouched at calibration)")
    ax.set_ylabel("average price (native dataset units)")
    ax.set_title("Rolling one-step train--calibrate--test predictions")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, metadata={"Software": "detcorr scientific case studies"})
    plt.close(fig)


def _plot_cumulative_scores(path: Path, predictions: pd.DataFrame) -> None:
    primary = predictions.loc[predictions["model"] == PRIMARY_MODEL].set_index("test_t")
    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    for model in MODEL_ORDER:
        if model == PRIMARY_MODEL:
            continue
        comparator = predictions.loc[predictions["model"] == model].set_index("test_t")
        delta = (primary["log_score"] - comparator["log_score"]).cumsum()
        ax.plot(delta.index, delta.to_numpy(), linewidth=1.1, label=model)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("test observation t")
    ax.set_ylabel("cumulative log score: primary minus comparator")
    ax.set_title("Sequential score contrasts (positive favours primary)")
    ax.legend(frameon=False, fontsize=6.7, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, metadata={"Software": "detcorr scientific case studies"})
    plt.close(fig)


def _plot_eta_shift(path: Path, predictions: pd.DataFrame) -> None:
    graph = predictions.loc[predictions["model"] == PRIMARY_MODEL].sort_values("test_t")
    naive = predictions.loc[
        predictions["model"] == "ar1_naive_volume_omission_counterfactual"
    ].sort_values("test_t")
    fig, ax = plt.subplots(figsize=(7.5, 4.1))
    ax.plot(
        graph["test_t"],
        graph["eta_posterior_mean"],
        marker="o",
        markersize=3,
        label="graph eta mean",
    )
    ax.plot(
        naive["test_t"],
        naive["eta_posterior_mean"],
        marker="s",
        markersize=2.5,
        label="volume omitted",
    )
    ax.axhline(0.0, color="0.3", linestyle=":")
    ax.set_xlabel("test observation t")
    ax.set_ylabel("calibration posterior mean eta = log(lambda)")
    ax.set_title("Determinant-omission consequence across rolling origins")
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, metadata={"Software": "detcorr scientific case studies"})
    plt.close(fig)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            str(key): (value.item() if isinstance(value, np.generic) else value)
            for key, value in row.items()
        }
        for row in frame.to_dict(orient="records")
    ]


def run_analysis(
    output_root: Path,
    *,
    data_path: Path = DATA_PATH,
    data_manifest_path: Path = DATA_MANIFEST_PATH,
    full: bool = False,
) -> dict[str, Any]:
    """Run the registered analysis and write its deterministic output surface."""

    output_root = Path(output_root)
    tables = output_root / "tables"
    figures = output_root / "figures"
    claims_root = output_root / "claims"
    output_root.mkdir(parents=True, exist_ok=True)
    protocol, protocol_sha = load_and_validate_protocol()
    frame = load_fish(data_path)
    input_sha = sha256_file(data_path)
    data_manifest = load_data_manifest(data_manifest_path, data_path=data_path, frame=frame)
    registered_fixture = (
        input_sha == DATA_SHA256
        and len(frame) == 97
        and data_manifest_path.resolve() == DATA_MANIFEST_PATH.resolve()
    )
    n_grid = 4001 if full else 1601
    bootstrap_replicates = 5000 if full else 500
    configuration = {
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_interval_quantiles": [
            BOOTSTRAP_LOWER_QUANTILE,
            BOOTSTRAP_UPPER_QUANTILE,
        ],
        "calibration_window": CALIBRATION_WINDOW,
        "first_test_position_zero_based": FIRST_TEST_POSITION,
        "grid_points_verification": n_grid,
        "interval_level": INTERVAL_LEVEL,
        "mode": "full" if full else "quick",
        "phi_clip": [-PHI_LIMIT, PHI_LIMIT],
        "phi_clip_sensitivity": {
            "absolute_limits": list(PHI_CLIP_SENSITIVITY_LIMITS),
            "primary_absolute_limit": PHI_LIMIT,
            "selection_role": "robustness disclosure only; never selects the primary model",
            "timing": PHI_CLIP_SENSITIVITY_TIMING,
        },
        "placebo_lag": PLACEBO_LAG,
        "primary_geometry": {
            "central_graph_probability": PRIMARY_GEOMETRY_CENTRAL_PROBABILITY,
            "eta_points": PRIMARY_GEOMETRY_ETA_POINTS,
            "max_error_fraction": PRIMARY_GEOMETRY_MAX_ERROR_FRACTION,
            "rho": PRIMARY_GEOMETRY_RHO,
            "sobol_power": (
                PRIMARY_GEOMETRY_FULL_SOBOL_POWER if full else PRIMARY_GEOMETRY_QUICK_SOBOL_POWER
            ),
            "sobol_scrambles": (
                PRIMARY_GEOMETRY_FULL_SCRAMBLES if full else PRIMARY_GEOMETRY_QUICK_SCRAMBLES
            ),
        },
        "prior_eta_sd": PRIOR_ETA_SD,
    }
    configuration_sha = canonical_json_sha256(configuration)

    verification_table, verification, _verification_arrays = exact_verification_spine(
        frame, n_grid=n_grid
    )
    pushforward_table, pushforward = residual_pushforward_audit(
        verification["n_calibration_states"]
    )
    parameter_pushforward = parameter_pushforward_audit(frame, n_grid=n_grid)
    primary_geometry_table, primary_geometry = primary_model_geometry_audit(frame, full=full)
    initial_train = frame.iloc[: FIRST_TEST_POSITION - CALIBRATION_WINDOW]
    diagnostics_table, acf_table, diagnostics = residual_diagnostics(initial_train)
    role_manifest, role_assertions = build_role_manifest(frame)
    _write_csv(role_manifest, tables / "role_manifest.csv")
    role_manifest_sha = sha256_file(tables / "role_manifest.csv")
    prescore_path = tables / "pre_score_predictions.csv"
    predictions = _model_prediction_rows(frame, prescore_path=prescore_path)
    metrics = aggregate_metrics(predictions)
    uncertainty = paired_score_uncertainty(
        predictions,
        bootstrap_replicates=bootstrap_replicates,
        block_length=4,
    )
    matched_effects = matched_calibration_effects(
        predictions,
        bootstrap_replicates=bootstrap_replicates,
        block_length=4,
    )
    phi_clip_table, phi_clip_summary = phi_clip_sensitivity(
        frame,
        predictions,
        matched_effects,
        bootstrap_replicates=bootstrap_replicates,
        limits=PHI_CLIP_SENSITIVITY_LIMITS,
        block_length=4,
    )

    _write_csv(verification_table, tables / "verification_spine.csv")
    _write_csv(pushforward_table, tables / "residual_pushforward_audit.csv")
    _write_csv(
        primary_geometry_table,
        tables / "primary_model_geometry_audit.csv",
    )
    _write_csv(diagnostics_table, tables / "initial_reference_residual_diagnostics.csv")
    _write_csv(acf_table, tables / "initial_reference_residual_acf.csv")
    _write_csv(predictions, tables / "rolling_predictions.csv")
    _write_csv(metrics, tables / "aggregate_predictive_metrics.csv")
    _write_csv(uncertainty, tables / "paired_score_uncertainty.csv")
    _write_csv(matched_effects, tables / "matched_calibration_effects.csv")
    _write_csv(phi_clip_table, tables / "phi_clip_sensitivity.csv")

    _plot_residual_acf(figures / "initial_reference_residual_acf.png", acf_table)
    _plot_rolling_predictions(figures / "rolling_predictions.png", predictions)
    _plot_cumulative_scores(figures / "cumulative_log_score_contrasts.png", predictions)
    _plot_eta_shift(figures / "rolling_eta_volume_effect.png", predictions)

    api_audit = detcorr_callable_audit()
    metric_index = metrics.set_index("model")
    primary_metrics = {
        key: float(value)
        for key, value in metric_index.loc[PRIMARY_MODEL].items()
        if key != "n_forecasts"
    }
    primary_metrics["n_forecasts"] = int(metric_index.loc[PRIMARY_MODEL, "n_forecasts"])
    prescore = pd.read_csv(prescore_path)
    prescore_hashes = sorted(prescore["prediction_sha256"].astype(str))
    scored_hashes = sorted(predictions["prediction_sha256_before_test_score"].astype(str))
    implementation_checks = {
        "registered_protocol_loaded_and_validated": bool(protocol and protocol_sha),
        "configuration_hash_recorded": len(configuration_sha) == 64,
        "input_schema_and_time_order_valid": True,
        "role_manifest_assertions_pass": all(
            value for value in role_assertions.values() if isinstance(value, bool)
        ),
        "reference_calibration_test_roles_disjoint": bool(
            np.all(
                predictions["reference_train_t_end"].to_numpy()
                < predictions["calibration_t_start"].to_numpy()
            )
            and np.all(
                predictions["calibration_t_end"].to_numpy() < predictions["test_t"].to_numpy()
            )
        ),
        "every_model_scores_same_origins": bool(
            predictions.groupby("model", observed=True)["test_t"].nunique().nunique() == 1
        ),
        "pre_score_predictions_hashed_before_outcome_score": (
            prescore_hashes == scored_hashes and len(prescore_hashes) == len(predictions)
        ),
        "all_matched_identity_comparisons_present": (
            set(matched_effects["assumption_family"])
            == {family for family, _, _ in MATCHED_CALIBRATION_COMPARISONS}
        ),
        "locked_bootstrap_seed_and_90pct_interval_used": bool(
            np.all(uncertainty["bootstrap_seed"] == BOOTSTRAP_SEED)
            and np.all(uncertainty["interval_level"] == 0.90)
            and "moving_block_bootstrap_ci05" in uncertainty
            and "moving_block_bootstrap_ci95" in uncertainty
        ),
        "phi_clip_sensitivity_contract_complete": bool(
            tuple(phi_clip_table["phi_absolute_limit"].to_numpy(dtype=float))
            == PHI_CLIP_SENSITIVITY_LIMITS
            and set(phi_clip_table["bootstrap_seed"]) == {BOOTSTRAP_SEED}
            and set(phi_clip_table["bootstrap_replicates"]) == {bootstrap_replicates}
            and bool(phi_clip_table["non_claim_selecting"].all())
            and phi_clip_summary["all_numeric_outputs_finite"]
        ),
        "primary_phi_clip_row_reproduces_claim_analysis": bool(
            phi_clip_table.loc[
                phi_clip_table["phi_absolute_limit"] == PHI_LIMIT,
                "cumulative_delta_elpd_graph_minus_identity",
            ].iloc[0]
            == matched_effects.set_index("assumption_family").loc[
                "ar1_plugin", "cumulative_delta_elpd_graph_minus_identity"
            ]
        ),
        "verification_corrected_tube_matches_graph": (
            verification["corrected_total_variation_vs_graph"] < 2.0e-5
        ),
        "verification_naive_tube_differs_from_graph": (
            verification["naive_total_variation_vs_graph"] > 0.25
        ),
        "primary_model_direct_log_J_over_q_and_finite_tube_gate": bool(primary_geometry["passed"]),
        "residual_metric_pushforward_exact": (
            pushforward["max_abs_correction_error"] < 1.0e-12
            and pushforward["max_abs_penalty_error"] < 1.0e-12
        ),
        "parameter_density_pushforward_exact": (
            parameter_pushforward["lambda_to_eta_to_lambda_total_variation"] < 1.0e-10
        ),
        "detcorr_callable_matches_analytic_jacobian": bool(
            api_audit.get("available")
            and api_audit.get("version_is_at_least_0_10")
            and api_audit.get("absolute_error", 1.0) < 2.0e-6
        ),
        "predictive_outputs_finite": bool(
            np.all(
                np.isfinite(
                    predictions[
                        [
                            "predictive_log_mean",
                            "predictive_log_sd",
                            "log_score",
                            "crps",
                        ]
                    ].to_numpy(dtype=float)
                )
            )
        ),
    }
    geometry_checks_passed = all(
        implementation_checks[name]
        for name in (
            "primary_model_direct_log_J_over_q_and_finite_tube_gate",
            "residual_metric_pushforward_exact",
            "parameter_density_pushforward_exact",
            "detcorr_callable_matches_analytic_jacobian",
        )
    )
    claim_evaluations = evaluate_claims(
        predictions,
        matched_effects,
        primary_geometry=primary_geometry,
        geometry_checks_passed=geometry_checks_passed,
        bootstrap_replicates=bootstrap_replicates,
        full=full,
        registered_fixture=registered_fixture,
    )
    source_hashes = {
        "fulton_analysis.py": sha256_file(Path(__file__)),
        "run_analysis.py": sha256_file(STUDY_ROOT / "run_analysis.py"),
        "REGISTERED_PROTOCOL.json": protocol_sha,
        "common/evidence.py": sha256_file(STUDY_ROOT.parents[1] / "common" / "evidence.py"),
    }
    evidence_paths = {
        "FU-PRED": (
            "tables/rolling_predictions.csv",
            "tables/aggregate_predictive_metrics.csv",
            "tables/matched_calibration_effects.csv",
            "tables/paired_score_uncertainty.csv",
            "tables/role_manifest.csv",
            "tables/pre_score_predictions.csv",
            "tables/phi_clip_sensitivity.csv",
            "figures/rolling_predictions.png",
            "figures/cumulative_log_score_contrasts.png",
        ),
        "FU-DEP": (
            "tables/matched_calibration_effects.csv",
            "tables/paired_score_uncertainty.csv",
            "tables/initial_reference_residual_diagnostics.csv",
            "tables/initial_reference_residual_acf.csv",
            "tables/phi_clip_sensitivity.csv",
            "figures/initial_reference_residual_acf.png",
        ),
        "FU-GEOM": (
            "tables/primary_model_geometry_audit.csv",
            "tables/residual_pushforward_audit.csv",
            "tables/rolling_predictions.csv",
            "tables/matched_calibration_effects.csv",
            "tables/phi_clip_sensitivity.csv",
            "figures/rolling_eta_volume_effect.png",
        ),
    }
    for claim in claim_evaluations:
        claim_bearing = bool(claim["claim_bearing_full_run"])
        if claim["claim_id"] == "FU-PRED":
            phi_clip_claim_evidence = {
                "all_gate_decisions_match_primary": phi_clip_summary[
                    "all_predictive_gate_decisions_match_primary"
                ],
                "cumulative_delta_elpd_range": [
                    float(phi_clip_table["cumulative_delta_elpd_graph_minus_identity"].min()),
                    float(phi_clip_table["cumulative_delta_elpd_graph_minus_identity"].max()),
                ],
                "moving_block_lower_endpoint_range": [
                    float(phi_clip_table["moving_block_90pct_mean_delta_elpd_lower"].min()),
                    float(phi_clip_table["moving_block_90pct_mean_delta_elpd_lower"].max()),
                ],
            }
        elif claim["claim_id"] == "FU-DEP":
            phi_clip_claim_evidence = {
                "all_dependence_categories_match_primary": phi_clip_summary[
                    "all_dependence_sensitivity_categories_match_primary"
                ],
                "dependence_sensitive_by_limit": {
                    str(row.phi_absolute_limit): bool(row.dependence_sensitive)
                    for row in phi_clip_table.itertuples(index=False)
                },
            }
        else:
            phi_clip_claim_evidence = {
                "all_geometry_consequences_finite": phi_clip_summary[
                    "all_geometry_consequences_finite"
                ],
                "graph_minus_naive_delta_elpd_range": [
                    float(phi_clip_table["cumulative_delta_elpd_graph_minus_naive"].min()),
                    float(phi_clip_table["cumulative_delta_elpd_graph_minus_naive"].max()),
                ],
                "mean_eta_volume_shift_graph_sd_range": [
                    float(phi_clip_table["mean_eta_volume_shift_in_graph_sd"].min()),
                    float(phi_clip_table["mean_eta_volume_shift_in_graph_sd"].max()),
                ],
            }
        phi_clip_claim_evidence.update(
            {
                "absolute_limits": list(PHI_CLIP_SENSITIVITY_LIMITS),
                "outcomes_known_when_added": True,
                "primary_absolute_limit": PHI_LIMIT,
                "primary_n_origins_clipped": phi_clip_summary["primary_n_origins_clipped"],
                "role": phi_clip_summary["role"],
                "table": phi_clip_summary["table"],
                "timing": PHI_CLIP_SENSITIVITY_TIMING,
            }
        )
        claim.update(
            {
                "protocol_version": str(protocol["version"]),
                "timing_class": ("registered_reanalysis" if registered_fixture else "exploratory"),
                "protocol_timing": PHI_CLIP_SENSITIVITY_TIMING,
                "evidence_class": (
                    "registered_retrospective_evaluation"
                    if registered_fixture
                    else "schema_compatible_external_exploration"
                ),
                "model_config_sha256": configuration_sha,
                "data_sha256": input_sha,
                "source_hashes": source_hashes,
                "phi_clip_sensitivity": phi_clip_claim_evidence,
                "evidence_artifacts": [
                    {
                        "path": relative,
                        "sha256": sha256_file(output_root / relative),
                        "artifact_type": ("figure" if relative.startswith("figures/") else "table"),
                    }
                    for relative in evidence_paths[claim["claim_id"]]
                ],
            }
        )
        if not claim_bearing:
            claim["evidence_level"] = (
                "E0_nonclaim_quick" if registered_fixture else "ungraded_exploratory"
            )
        if claim["claim_id"] == "FU-GEOM":
            claim["geometry_gate_evidence"] = {
                "primary_model_finite_tube": primary_geometry,
                "residual_metric_pushforward": pushforward,
                "parameter_density_pushforward": parameter_pushforward,
                "detcorr_callable": api_audit,
                "all_geometry_checks_passed": bool(geometry_checks_passed),
            }
    claim_table = pd.DataFrame(
        [
            {
                "claim_id": claim["claim_id"],
                "status": claim["status"],
                "conclusion": claim["conclusion"],
                "evidence_level": claim["evidence_level"],
                "timing_class": claim["timing_class"],
                "claim_bearing_full_run": claim["claim_bearing_full_run"],
                "interpretation": claim["interpretation"],
                "non_claim": claim["non_claim"],
            }
            for claim in claim_evaluations
        ]
    )
    _write_csv(claim_table, tables / "claim_evaluations.csv")
    strict_json_dump(
        {
            "protocol_sha256": protocol_sha,
            "configuration_sha256": configuration_sha,
            "claims": claim_evaluations,
        },
        claims_root / "claim_evaluations.json",
    )
    implementation_checks["all_registered_claims_evaluated"] = all(
        claim["status"] != "UNTESTED" for claim in claim_evaluations
    )
    implementation_checks["all_passed"] = all(implementation_checks.values())
    environment = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "detcorr": api_audit.get("detcorr_version", "unavailable"),
        "byteorder": sys.byteorder,
    }

    summary: dict[str, Any] = {
        "schema_version": "1.1",
        "study": "fulton_market",
        "analysis_status": (
            "registered_retrospective_reanalysis"
            if registered_fixture
            else "exploratory_schema_compatible_external_input"
        ),
        "mode": "full" if full else "quick",
        "claim_bearing_full_run": bool(full and registered_fixture),
        "reproducibility": {
            "registered_protocol_path": "REGISTERED_PROTOCOL.json",
            "registered_protocol_sha256": protocol_sha,
            "registered_protocol_version": protocol["version"],
            "configuration_sha256": configuration_sha,
            "source_hashes": source_hashes,
            "data_sha256": input_sha,
            "role_manifest_path": "tables/role_manifest.csv",
            "role_manifest_sha256": role_manifest_sha,
            "role_manifest_assertions": role_assertions,
            "pre_score_prediction_manifest_path": "tables/pre_score_predictions.csv",
            "pre_score_prediction_manifest_sha256": sha256_file(prescore_path),
            "random_seed": BOOTSTRAP_SEED,
            "randomness_scope": "moving-block bootstrap resampling only",
            "environment": environment,
        },
        "input": {
            "path": (
                REGISTERED_DATA_DISPLAY_PATH
                if registered_fixture
                else str(Path(data_path).resolve())
            ),
            "sha256": input_sha,
            "manifest_path": (
                "studies/fulton_market/data/FIXTURE_MANIFEST.json"
                if registered_fixture
                else str(Path(data_manifest_path).resolve())
            ),
            "manifest": data_manifest,
            "registered_fixture_sha256": DATA_SHA256,
            "is_registered_97_row_fixture": registered_fixture,
            "n_rows": len(frame),
            "n_columns": len(frame.columns),
            "t_start": int(frame["t"].iloc[0]),
            "t_end": int(frame["t"].iloc[-1]),
            "source_url": DATA_URL,
            "data_rights_boundary": (
                (
                    "The registered fish.csv is conveyed separately under GPL-3 to the extent "
                    "the wooldridge package licence applies; the Rdatasets underlying-rights "
                    "caveat is retained. See THIRD_PARTY_NOTICES.md and TECHNICAL_DETAILS.md."
                )
                if registered_fixture
                else (
                    "The caller is responsible for the provenance and redistribution terms of "
                    "this external input; no repository licence is inferred."
                )
            ),
        },
        "registered_protocol": {
            "first_test_t": int(frame.iloc[FIRST_TEST_POSITION]["t"]),
            "last_test_t": int(frame.iloc[-1]["t"]),
            "calibration_window": CALIBRATION_WINDOW,
            "forecast_horizon": 1,
            "prior": "eta=log(lambda) ~ Normal(0, 0.5^2), density pushed to d lambda",
            "primary": (
                "training-only log-price reference; training-only plug-in stationary AR(1) "
                "error; graph posterior; one-step native-price predictive score"
            ),
            "sensitivities": [
                "iid log-price error",
                "Gaussian OLS coefficient-covariance propagation",
                "state-volume determinant omission counterfactual",
                (
                    "post-outcome one-at-a-time symmetric AR(1) phi clipping bounds; "
                    "non-claim-selecting"
                ),
            ],
            "negative_control": (
                "four-observation causal lag of the reference design; response order retained "
                "and no future row used"
            ),
            "baselines": [
                "AR(1)-conditioned identity reference matched to the primary graph model",
                "iid identity reference matched to the iid graph sensitivity",
                (
                    "AR(1) plus coefficient-uncertainty identity reference matched to that "
                    "graph sensitivity"
                ),
                "last-price lognormal persistence with past-only difference scale",
            ],
            "role_separation": (
                "for test t, rows before t-16 fit the reference, rows t-16 through t-1 "
                "calibrate eta, and row t is scored once"
            ),
            "test_outcome_used_before_scoring": False,
            "test_covariates_used_for_conditional_prediction": True,
        },
        "verification_spine": verification,
        "residual_and_coordinate_audits": {
            "primary_model_direct_log_and_finite_tube": primary_geometry,
            "residual_pushforward": pushforward,
            "parameter_pushforward": parameter_pushforward,
            "detcorr_callable": api_audit,
        },
        "initial_training_residual_diagnostics": diagnostics,
        "primary_predictive_metrics": primary_metrics,
        "aggregate_predictive_metrics": _records(metrics),
        "matched_calibration_effects": _records(matched_effects),
        "phi_clip_sensitivity": {
            **phi_clip_summary,
            "results": _records(phi_clip_table),
        },
        "claim_evaluations": claim_evaluations,
        "score_uncertainty": {
            "method": (
                "circular moving-block bootstrap of paired origin-level contrasts; block "
                "length fixed at four; descriptive interval, not an iid significance test"
            ),
            "replicates": bootstrap_replicates,
            "seed": BOOTSTRAP_SEED,
            "interval_quantiles": [
                BOOTSTRAP_LOWER_QUANTILE,
                BOOTSTRAP_UPPER_QUANTILE,
            ],
            "table": "tables/paired_score_uncertainty.csv",
        },
        "implementation_checks": implementation_checks,
        "scientific_outcomes_are_not_pass_fail_checks": True,
        "interpretation": (
            "The rolling results quantify conditional one-step calibration and its sensitivity "
            "to dependence, plug-in reference uncertainty, and target-measure omission. They "
            "do not identify market structure or validate determinant correction through "
            "predictive superiority."
        ),
        "limitations": [
            (
                "The 97-row fixture is short and was inspected before this extension was "
                "registered; results are retrospective, not prospective confirmation."
            ),
            (
                "Weather, weekday, and trend regressors form a compact reduced reference, "
                "not a structural supply or demand system."
            ),
            (
                "The AR(1) coefficient and stationary marginal residual SD are plug-in "
                "training estimates, and the innovation SD is derived from them; phi is not "
                "given a posterior. Only the declared coefficient-covariance sensitivity "
                "propagates part of reference-fit uncertainty."
            ),
            (
                "Rolling score contrasts are dependent because training and calibration "
                "windows overlap; moving-block intervals are descriptive with only 33 "
                "registered origins."
            ),
            (
                "The lognormal observation model and stationary AR(1) covariance remain "
                "simplifying assumptions and do not resolve all heteroskedasticity or "
                "regime change."
            ),
            (
                "Each test price is untouched, but its contemporaneous weather and weekday "
                "covariates are treated as known; the result is conditional prediction, not "
                "an unconditional advance forecast when those covariates are unavailable."
            ),
            (
                "Lambda is a conditional dimensionless calibration scale, not an elasticity, "
                "causal effect, market-power coefficient, or universal economic constant."
            ),
        ],
        "claim_ceiling": (
            "A defensible result is conditional to this fixed temporal protocol and model family: "
            "the graph-measure calibration has auditable one-step predictive consequences and "
            "its conclusions can be compared with prespecified dependence, uncertainty, and "
            "negative-control analyses. No causal or structural Fulton-market claim follows."
        ),
        "outputs": {
            "tables": sorted(path.name for path in tables.glob("*.csv")),
            "figures": sorted(path.name for path in figures.glob("*.png")),
            "claims": ["claims/claim_evaluations.json"],
        },
        "configuration": configuration,
    }
    strict_json_dump(summary, output_root / "summary.json")
    return summary


def public_fit_record(fit: ReferenceFit) -> dict[str, Any]:
    """Return a JSON-friendly fit record for debugging or external adapters."""

    record = asdict(fit)
    for key in ("coefficients", "coefficient_covariance", "residuals"):
        record[key] = np.asarray(record[key]).tolist()
    return record
