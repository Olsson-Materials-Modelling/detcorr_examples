#!/usr/bin/env python3
"""Generate the worked 1D elliptic-PDE detcorr demonstrator artifacts."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
COMMON = ROOT.parent / "detcorr_demo_common" / "src"
for path in [str(SRC), str(COMMON)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import elliptic_pde_model as benchmark  # noqa: E402

from detcorr_demo_common import (  # noqa: E402
    density_l1,
    normalize_on_grid,
    reproducibility_provenance,
    write_json,
    write_table,
)


def build_routes() -> tuple[np.ndarray, dict[str, np.ndarray], pd.DataFrame]:
    """Evaluate and normalize the three independently defined theta routes."""

    theta = np.linspace(benchmark.THETA_MIN, benchmark.THETA_MAX, benchmark.GRID_SIZE)
    logdensities = {
        "reduced_graph": np.array(
            [benchmark.reduced_logposterior(float(value)) for value in theta]
        ),
        "naive_full_space": np.array(
            [
                benchmark.integrated_fullspace_logposterior(
                    float(value),
                    corrected=False,
                )
                for value in theta
            ]
        ),
        "detcorr_full_space": np.array(
            [
                benchmark.integrated_fullspace_logposterior(
                    float(value),
                    corrected=True,
                )
                for value in theta
            ]
        ),
    }
    densities: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    graph_density = None
    graph_summary = None
    for name, logdensity in logdensities.items():
        density, _, summary = normalize_on_grid(theta, logdensity)
        densities[name] = density
        if name == "reduced_graph":
            graph_density = density
            graph_summary = summary
        rows.append({"target": name, **asdict(summary)})
    assert graph_density is not None and graph_summary is not None
    for row in rows:
        name = str(row["target"])
        row["density_l1_vs_graph"] = density_l1(
            theta,
            densities[name],
            graph_density,
        )
        row["mean_shift_in_graph_sd"] = (float(row["mean"]) - graph_summary.mean) / graph_summary.sd
    return theta, densities, pd.DataFrame(rows)


def make_figures(
    theta: np.ndarray,
    densities: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    labels = {
        "reduced_graph": "reduced graph",
        "naive_full_space": "naive full-space tube",
        "detcorr_full_space": "detcorr full-space tube",
    }
    for name in ["naive_full_space", "detcorr_full_space", "reduced_graph"]:
        style = {"linestyle": "--", "linewidth": 2.2} if name == "reduced_graph" else {}
        ax.plot(theta, densities[name], label=labels[name], **style)
    ax.set_xlabel(r"log-diffusion parameter $\theta$")
    ax.set_ylabel("posterior density")
    ax.set_title("Elliptic inverse problem: state-volume correction")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "elliptic_posterior_routes.png", dpi=180)
    plt.close(fig)

    x = np.linspace(0.0, benchmark.DOMAIN_LENGTH, benchmark.N_STATE + 2)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for theta_value, label in [
        (benchmark.SYNTHETIC_TRUE_THETA, "declared truth"),
        (0.0, "theta = 0 control"),
        (0.7, "higher diffusion"),
    ]:
        state = np.concatenate([[0.0], benchmark.solve_state(theta_value), [0.0]])
        ax.plot(x, state, marker="o", label=label)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="observation")
    ax.set_xlabel("nondimensional position x")
    ax.set_ylabel("nondimensional state u")
    ax.set_title("Finite-difference state profiles")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "elliptic_state_profiles.png", dpi=180)
    plt.close(fig)


def run_demo(output_dir: Path) -> dict[str, Any]:
    theta, densities, table = build_routes()
    benchmark_summary = benchmark.run_benchmark(verbose=False)
    log_corrections = np.array(
        [benchmark._logdet_spd(benchmark.operator_matrix(float(value))) for value in theta]
    )
    graph_row = table.loc[table["target"] == "reduced_graph"].iloc[0]
    corrected_row = table.loc[table["target"] == "detcorr_full_space"].iloc[0]
    naive_row = table.loc[table["target"] == "naive_full_space"].iloc[0]

    output_dir = Path(output_dir)
    write_table(table, output_dir / "tables" / "posterior_routes.csv")
    make_figures(theta, densities, output_dir)
    summary = {
        "demo": "elliptic_pde_1d",
        "classification": "transparent_synthetic_physics_demonstrator",
        "provenance": reproducibility_provenance(example_root=ROOT),
        "model": benchmark_summary["model"],
        "synthetic_data": benchmark_summary["synthetic_data"],
        "target_measure": "graph posterior over theta with respect to d theta",
        "correction_driver": "exact logdet of the parameter-dependent PDE state operator",
        "correction_route": "dense SPD Cholesky on the six-state CI discretization",
        "correction_log_range_on_grid": float(np.ptp(log_corrections)),
        "posterior": table.to_dict(orient="records"),
        "outcome": {
            "correction_log_range_on_grid": float(np.ptp(log_corrections)),
            "corrected_density_l1_vs_graph": float(corrected_row["density_l1_vs_graph"]),
            "naive_density_l1_vs_graph": float(naive_row["density_l1_vs_graph"]),
            "corrected_mean_shift_in_graph_sd": float(corrected_row["mean_shift_in_graph_sd"]),
            "naive_mean_shift_in_graph_sd": float(naive_row["mean_shift_in_graph_sd"]),
            "naive_minus_corrected_mean_in_graph_sd": (
                float(naive_row["mean"]) - float(corrected_row["mean"])
            )
            / float(graph_row["sd"]),
            "graph_mean": float(graph_row["mean"]),
            "graph_sd": float(graph_row["sd"]),
        },
        "independent_validation": {
            "route": "analytic Gaussian integration over the full discrete state",
            "manufactured_stiffness_check": True,
            "max_reduced_residual_norm": benchmark_summary["audit"]["max_reduced_residual_norm"],
            "detcorr_logdet_abs_error": benchmark_summary["audit"]["max_abs_logdet_error"],
        },
        "interpretation_boundary": (
            "Validates measure correction for a small consistent nondimensional "
            "finite-difference inverse problem; it is not empirical PDE inference."
        ),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = run_demo(args.outputs)
    print(
        "elliptic_pde_1d: "
        f"naive shift={summary['outcome']['naive_mean_shift_in_graph_sd']:.3f} graph SD"
    )


if __name__ == "__main__":
    main()
