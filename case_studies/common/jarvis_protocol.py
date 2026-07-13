"""Locked, application-neutral protocol helpers for the JARVIS studies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROTOCOL_VERSION = "0.2-retrospective"
RESAMPLING_SEED = 20260711


def predictive_grid_differences(
    active: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    key_columns: tuple[str, ...],
    input_column: str,
    density_column: str,
    quantile_columns: tuple[str, ...],
    coverage_columns: tuple[str, ...] = ("coverage50", "coverage90"),
) -> tuple[pd.DataFrame, dict[str, float | int | bool]]:
    """Compare held-out endpoints from two posterior-grid resolutions."""

    for label, frame in (("active", active), ("comparison", comparison)):
        required = {
            *key_columns,
            input_column,
            density_column,
            *quantile_columns,
            *coverage_columns,
            "quantile_cdf_method",
            "quantile_max_abs_cdf_root_residual",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{label} predictions lack columns: {missing}")
        if frame.duplicated(list(key_columns)).any():
            raise ValueError(f"{label} prediction keys are not unique")

    merged = active.merge(
        comparison,
        on=list(key_columns),
        how="outer",
        validate="one_to_one",
        suffixes=("_active", "_comparison"),
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("grid refinements do not contain identical prediction keys")
    active_input = merged[f"{input_column}_active"].to_numpy(float)
    comparison_input = merged[f"{input_column}_comparison"].to_numpy(float)
    if (
        not np.all(np.isfinite(active_input))
        or np.any(active_input <= 0.0)
        or not np.array_equal(active_input, comparison_input)
    ):
        raise ValueError("grid refinements must use identical positive reference inputs")

    log_quantile_errors = []
    for column in quantile_columns:
        active_values = merged[f"{column}_active"].to_numpy(float)
        comparison_values = merged[f"{column}_comparison"].to_numpy(float)
        if (
            not np.all(np.isfinite(active_values))
            or not np.all(np.isfinite(comparison_values))
            or np.any(active_values <= 0.0)
            or np.any(comparison_values <= 0.0)
        ):
            raise ValueError("predictive quantiles must be finite and positive")
        log_quantile_errors.append(
            np.abs(
                np.log(active_values / active_input) - np.log(comparison_values / comparison_input)
            )
        )
    quantile_error = np.max(np.column_stack(log_quantile_errors), axis=1)
    density_error = np.abs(
        merged[f"{density_column}_active"].to_numpy(float)
        - merged[f"{density_column}_comparison"].to_numpy(float)
    )
    coverage_mismatches = np.zeros(len(merged), dtype=int)
    for column in coverage_columns:
        coverage_mismatches += merged[f"{column}_active"].to_numpy(float) != merged[
            f"{column}_comparison"
        ].to_numpy(float)
    exact_method = merged["quantile_cdf_method_active"].eq("exact") & merged[
        "quantile_cdf_method_comparison"
    ].eq("exact")
    root_residual = np.maximum(
        merged["quantile_max_abs_cdf_root_residual_active"].to_numpy(float),
        merged["quantile_max_abs_cdf_root_residual_comparison"].to_numpy(float),
    )
    details = merged.loc[:, list(key_columns)].copy()
    details["abs_log_predictive_density_difference"] = density_error
    details["max_abs_logratio_quantile_difference"] = quantile_error
    details["coverage_decision_mismatch_count"] = coverage_mismatches
    details["both_quantile_routes_exact"] = exact_method.to_numpy(bool)
    details["max_abs_cdf_root_residual"] = root_residual
    summary: dict[str, float | int | bool] = {
        "n_prediction_rows": int(len(details)),
        "max_abs_log_predictive_density_difference": float(np.max(density_error)),
        "max_abs_logratio_quantile_difference": float(np.max(quantile_error)),
        "coverage_decision_mismatch_count": int(np.sum(coverage_mismatches)),
        "all_quantile_routes_exact": bool(np.all(exact_method)),
        "max_abs_cdf_root_residual": float(np.max(root_residual)),
    }
    return details, summary


def canonical_hash(payload: Any) -> str:
    """SHA256 of a canonical JSON representation."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def logo_role_manifest(
    frame: pd.DataFrame,
    *,
    protocol_config: dict[str, Any],
    row_key: str = "jvasp_id",
    group_key: str = "formula",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Emit and validate every training/held-out role in formula LOGO.

    A long table is intentional: readers can audit exact row identities without
    parsing list-valued cells, and leakage checks operate on the emitted object.
    """

    protocol_hash = canonical_hash(protocol_config)
    groups = sorted(frame[group_key].astype(str).unique())
    rows: list[dict[str, Any]] = []
    for fold, held_formula in enumerate(groups):
        for record in frame[[row_key, group_key]].itertuples(index=False, name=None):
            identifier, formula = (str(record[0]), str(record[1]))
            rows.append(
                {
                    "fold": int(fold),
                    "heldout_formula": held_formula,
                    "row_role": "heldout" if formula == held_formula else "training",
                    "jvasp_id": identifier,
                    "row_formula": formula,
                    "protocol_hash": protocol_hash,
                }
            )
    table = pd.DataFrame(rows)

    identifiers = set(frame[row_key].astype(str))
    no_overlap = True
    held_sets_match = True
    train_sets_match = True
    for held_formula, part in table.groupby("heldout_formula", sort=True):
        held = set(part.loc[part["row_role"] == "heldout", "jvasp_id"])
        train = set(part.loc[part["row_role"] == "training", "jvasp_id"])
        expected_held = set(
            frame.loc[frame[group_key].astype(str) == held_formula, row_key].astype(str)
        )
        no_overlap = no_overlap and held.isdisjoint(train)
        held_sets_match = held_sets_match and held == expected_held
        train_sets_match = train_sets_match and train == identifiers - expected_held
    held_counts = table[table["row_role"] == "heldout"].groupby("jvasp_id").size()
    each_row_held_once = set(held_counts.index) == identifiers and bool(held_counts.eq(1).all())
    one_hash = bool(table["protocol_hash"].nunique() == 1)
    validation = {
        "protocol_hash": protocol_hash,
        "n_folds": int(len(groups)),
        "n_source_rows": int(len(frame)),
        "n_manifest_rows": int(len(table)),
        "no_training_heldout_overlap_within_fold": bool(no_overlap),
        "heldout_sets_match_formula_groups": bool(held_sets_match),
        "training_sets_are_exact_complements": bool(train_sets_match),
        "every_row_is_held_out_exactly_once": bool(each_row_held_once),
        "single_protocol_hash": one_hash,
    }
    validation["all_passed"] = bool(
        validation["n_folds"] == frame[group_key].nunique()
        and validation["n_manifest_rows"] == len(frame) * len(groups)
        and all(
            validation[key]
            for key in (
                "no_training_heldout_overlap_within_fold",
                "heldout_sets_match_formula_groups",
                "training_sets_are_exact_complements",
                "every_row_is_held_out_exactly_once",
                "single_protocol_hash",
            )
        )
    )
    return table, validation


def add_claim_provenance(
    claims: list[dict[str, Any]],
    *,
    model_config: dict[str, Any],
    input_sha256: str,
    claim_bearing_full_run: bool,
    timing_class: str,
    evidence_class: str,
    evidence_level: str,
    source_paths: dict[str, Path],
    tables_root: Path,
    evidence_files: dict[str, tuple[str, ...]],
) -> list[dict[str, Any]]:
    """Attach conservative retrospective E1 provenance to each claim."""

    config_hash = canonical_hash(model_config)
    source_hashes = {name: file_hash(path) for name, path in source_paths.items()}
    enriched: list[dict[str, Any]] = []
    for claim in claims:
        artifacts = []
        for name in evidence_files[claim["claim_id"]]:
            path = tables_root / name
            artifacts.append(
                {
                    "path": f"tables/{name}",
                    "sha256": file_hash(path),
                    "artifact_type": "table",
                }
            )
        enriched.append(
            {
                **claim,
                "protocol_version": PROTOCOL_VERSION,
                "timing_class": timing_class,
                "protocol_timing": "post_outcome_retrospective",
                "evidence_class": evidence_class,
                "evidence_level": evidence_level,
                "model_config_sha256": config_hash,
                "data_sha256": input_sha256,
                "claim_bearing_full_run": bool(claim_bearing_full_run),
                "source_hashes": source_hashes,
                "evidence_artifacts": artifacts,
            }
        )
    return enriched
