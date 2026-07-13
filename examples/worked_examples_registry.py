"""Single source of truth for the active release worked-example set."""

from __future__ import annotations

WORKED_EXAMPLES = (
    {
        "slug": "calorimetry_constant_determinant_control",
        "title": "Calorimetry constant-determinant control",
        "tier": "illustrative",
        "data_basis": "transparent synthetic verification problem",
        "main_figure": "calorimetry_constant_determinant_posteriors.png",
        "corrected_l1_role": "finite-penalty corrected-target agreement with graph target",
        "independent_metric": None,
    },
    {
        "slug": "kinetics_decay_time_stepping",
        "title": "First-order decay with implicit time stepping",
        "tier": "illustrative",
        "data_basis": "transparent synthetic verification problem",
        "main_figure": "eta_posterior_comparison.png",
        "corrected_l1_role": "finite-penalty corrected-target agreement with graph target",
        "independent_metric": "corrected_centered_logdensity_max_abs_diff_vs_graph",
    },
    {
        "slug": "adsorption_langmuir_equilibrium",
        "title": "Nonlinear Langmuir equilibrium",
        "tier": "illustrative",
        "data_basis": "transparent synthetic verification problem",
        "main_figure": "posterior_eta_targets.png",
        "corrected_l1_role": "finite-penalty corrected-target agreement with graph target",
        "independent_metric": "corrected_centered_logdensity_convergence_error",
    },
    {
        "slug": "sensor_weighted_constraint",
        "title": "Sensor gain with a weighted constraint metric",
        "tier": "illustrative",
        "data_basis": "transparent synthetic verification problem",
        "main_figure": "sensor_gain_posteriors.png",
        "corrected_l1_role": "finite-penalty corrected-target agreement with graph target",
        "independent_metric": "finite_vs_hard_tube_centered_max_abs_diff",
    },
    {
        "slug": "elliptic_pde_1d",
        "title": "One-dimensional elliptic inverse problem",
        "tier": "illustrative",
        "data_basis": "transparent synthetic verification problem",
        "main_figure": "elliptic_posterior_routes.png",
        "corrected_l1_role": "finite-penalty corrected-target agreement with graph target",
        "independent_metric": "detcorr_logdet_abs_error",
    },
    {
        "slug": "materials_bulk_modulus_samplers_jarvis",
        "title": "JARVIS bulk-modulus calibration",
        "tier": "open_data",
        "data_basis": "publicly accessible paired JARVIS OPT/experimental fixture",
        "main_figure": "posterior_gamma_bulk_modulus.png",
        "corrected_l1_role": (
            "algebraic hard-limit round-trip consistency; not independent validation"
        ),
        "independent_metric": "finite_vs_hard_tube_centered_max_abs_diff",
    },
    {
        "slug": "econometrics_fulton_market",
        "title": "Held-out Fulton price-scale calibration",
        "tier": "open_data",
        "data_basis": "publicly accessible Wooldridge/Graddy Fulton fish-price fixture",
        "main_figure": "heldout_scale_posteriors.png",
        "corrected_l1_role": "finite-penalty corrected-target agreement with graph target",
        "independent_metric": "finite_vs_hard_tube_centered_max_abs_diff",
    },
    {
        "slug": "materials_bandgap_matbench",
        "title": "Held-out Matbench nonlinear-coordinate stress test",
        "tier": "open_data",
        "data_basis": "publicly accessible Matbench/Matminer experimental-gap fixture",
        "main_figure": "heldout_alpha_posteriors.png",
        "corrected_l1_role": "finite-penalty corrected-target agreement with graph target",
        "independent_metric": "finite_vs_hard_tube_centered_max_abs_diff",
    },
    {
        "slug": "materials_bandgap_samplers_jarvis",
        "title": "JARVIS log-energy-residual gap calibration",
        "tier": "open_data",
        "data_basis": "publicly accessible paired JARVIS OPT/experimental gap fixture",
        "main_figure": "posterior_gamma_log_energy_residual.png",
        "corrected_l1_role": (
            "algebraic hard-limit round-trip consistency; not independent validation"
        ),
        "independent_metric": "finite_vs_hard_tube_centered_max_abs_diff",
    },
    {
        "slug": "materials_screened_lattice_2d3d_jarvis",
        "title": "Paired-JARVIS screened-lattice surrogate",
        "tier": "open_data",
        "data_basis": "publicly accessible paired JARVIS fixture in a spatial surrogate",
        "main_figure": "posterior_spatial_lattice_scale.png",
        "corrected_l1_role": (
            "algebraic hard-limit round-trip consistency; not independent validation"
        ),
        "independent_metric": "finite_vs_hard_tube_centered_max_abs_diff",
    },
)

WORKED_EXAMPLE_SLUGS = tuple(row["slug"] for row in WORKED_EXAMPLES)
WORKED_EXAMPLE_TIERS = ("illustrative", "open_data")


def tier_counts() -> dict[str, int]:
    """Return deterministic counts for every declared release tier."""

    return {
        tier: sum(row["tier"] == tier for row in WORKED_EXAMPLES) for tier in WORKED_EXAMPLE_TIERS
    }
