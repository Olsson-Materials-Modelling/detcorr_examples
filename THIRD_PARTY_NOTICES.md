# Third-party data notices

The repository MIT licence covers original software and documentation. It does
not relicense bundled third-party data or source-bearing reproductions. This
page is the repository-level map of the applicable permissions, attribution,
and reuse actions. The study-specific notices are retained in
[`case_studies/THIRD_PARTY_NOTICES.md`](case_studies/THIRD_PARTY_NOTICES.md).

## At a glance

| Data | Used by | Repository position | Reuse action |
|---|---|---|---|
| Synthetic tutorial designs | Calorimetry, kinetics, Langmuir, weighted sensor, elliptic PDE | Repository-authored inputs under the project licence | Retain the repository licence and do not describe the synthetic systems as experimental validation. |
| JARVIS positive band gaps | Gap demonstrator and scientific gap study | Adapted from Choudhary *et al.*, *Scientific Data* 5, 180082 (2018), Table 3, under CC BY 4.0 | Credit the source, link CC BY 4.0, identify the column selection/normalisation, retain the third-party-material qualification, and imply no endorsement. |
| JARVIS bulk moduli | Bulk demonstrator, screened-lattice demonstrator, and scientific bulk study | Distributed under NIST's public-information permission and disclaimer; not relicensed as MIT or CC BY | Credit NIST/JARVIS, retain the transformation notice, disclaimer, and non-endorsement statement. |
| Fulton `fish` table | Fulton demonstrator and scientific Fulton study | Conveyed separately under GPL-3 to the extent the official `wooldridge` package licence applies, with the Rdatasets underlying-rights caveat retained | Preserve attribution, the GPL-3 text, the package/Rdatasets lineage, and the underlying-rights qualification. |
| Matbench `expt_gap` experimental gaps | Matbench demonstrator | The authoritative Figshare dataset record declares MIT; the fixture is a deterministic transformation of the pinned source payload | Credit Hacking Materials, retain the dataset DOI and MIT licence, and preserve the source/fixture hash chain and selection rule. |

Public accessibility and a SHA256 hash establish access and byte identity, not
permission by themselves. Generated figures and tables do not erase the terms
or provenance of the source values they reproduce.

## `detcorr` software dependency — GPL-3.0-only

The examples install `detcorr` from
[Olsson-Materials-Modelling/detcorr](https://github.com/Olsson-Materials-Modelling/detcorr)
at the commit identified in the README and CI workflows. That dependency is
licensed GPL-3.0-only, is installed separately, and is not included in this
repository's distribution. The repository MIT licence covers the original
examples code and documentation; it does not relicense `detcorr`. Reuse of the
dependency remains subject to its
[upstream licence](https://github.com/Olsson-Materials-Modelling/detcorr/blob/a414f063515359f70952e2717a506b75e69e4b0b/LICENSE).

## JARVIS positive-gap fixture — CC BY 4.0

The identical 54-row positive-gap fixture appears at:

- `examples/materials_bandgap_samplers_jarvis/data/jarvis_bandgap_benchmark_fixture.csv`;
- `case_studies/studies/jarvis_band_gaps/data/jarvis_bandgap_benchmark_fixture.csv`;
  and
- source-bearing rows in `case_studies/reference_results/jarvis_band_gaps/`.

It is adapted from K. Choudhary, Q. Zhang, A. C. E. Reid *et al.*,
“Computational screening of high-performance optoelectronic materials using
OptB88vdW and TB-mBJ formalisms,” *Scientific Data* **5**, 180082 (2018),
Table 3, <https://doi.org/10.1038/sdata.2018.82>. The source article is
licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The OPT and experimental columns were selected and renamed, identifiers and
labels were normalised, provenance fields were added, and only the 54 finite
strictly positive pairs were retained. Values were not fitted or imputed. The
table does not resolve fundamental versus optical measurement type or the
experimental protocol row by row; that remains a scientific limitation.

## JARVIS bulk-modulus fixture — NIST public information

The identical 36-row bulk-modulus fixture appears at:

- `examples/materials_bulk_modulus_samplers_jarvis/data/jarvis_bulk_modulus_benchmark_fixture.csv`;
- `examples/materials_screened_lattice_2d3d_jarvis/data/jarvis_bulk_modulus_paired_fixture.csv`;
- `case_studies/studies/jarvis_bulk_modulus/data/jarvis_bulk_modulus_benchmark_fixture.csv`;
  and
- source-bearing rows in `case_studies/reference_results/jarvis_bulk_modulus/`.

The values were transcribed from the NIST JARVIS-DFT documentation comparison
table. NIST's
[Copyrights & Disclaimers](https://www.nist.gov/copyrights-disclaimers)
policy states that unmarked information on NIST sites is public information
that may be distributed or copied and requests appropriate credit. The table
carries no separate copyright mark. It is therefore distributed under that
public-information permission, not under the repository MIT licence.

Material names and JARVIS identifiers were normalised into explicit columns;
the 36 published OPT/experiment pairs were not altered. The source reports an
MAE rounded to 8.51 GPa; the fixture recomputes 8.5055555556 GPa. No NIST
endorsement is implied. Row-level experimental references, phase/protocol,
uncertainty, temperature, and measurement conditions are not supplied.

## Fulton `fish` fixture — qualified GPL-3 package content

The byte-identical CSV appears at:

- `examples/econometrics_fulton_market/data/fish.csv`; and
- `case_studies/studies/fulton_market/data/fish.csv`.

The official `wooldridge` R data package distributes `fish` as package content
and declares GPL-3. The repository conveys the CSV separately under GPL-3 to
the extent that package licence applies. The complete text is retained at
[`case_studies/third_party_licenses/GPL-3.0.txt`](case_studies/third_party_licenses/GPL-3.0.txt).

Rdatasets exported the package table to CSV and added the `rownames` column;
the repository copied that CSV byte for byte. Rdatasets also states that it
could not establish the underlying rights in every hosted table. That caveat
is retained: GPL-3 is not represented as proof of rights held by the original
researcher, provider, textbook author, or publisher. Attribution remains due
to K. Graddy, the 1995 *RAND Journal of Economics* paper, Joshua Angrist, the
`wooldridge` package, and Rdatasets as recorded in the local fixture manifest
and [technical details](examples/econometrics_fulton_market/TECHNICAL_DETAILS.md).

## Matbench `expt_gap` fixture — MIT

The 100-row fixture in `examples/materials_bandgap_matbench/data/` is derived
from Hacking Materials, *expt_gap* (2019), Figshare dataset DOI
[`10.6084/m9.figshare.9765779.v1`](https://doi.org/10.6084/m9.figshare.9765779.v1).
The authoritative Figshare dataset record declares the dataset under the MIT
licence. The repository retains that attribution and licence basis; it does
not infer the data licence from the separate Matbench software repository.

The bundled CSV contains the first 100 source-ordered, unique, finite,
strictly positive gaps whose formulas match the declared ordinary
unparenthesized notation. That rule uses the positive-state eligibility
criterion but no fitted residual or model-performance ranking; it produces an
alphabetically concentrated convenience fixture, not a
representative Matbench benchmark sample. The source payload URL, Figshare
file ID, source SHA256 and MD5, fixture SHA256, exact selection rule, rebuild
script, citations, and deterministic held-out split are recorded in the local
[`FIXTURE_MANIFEST.json`](examples/materials_bandgap_matbench/data/FIXTURE_MANIFEST.json)
and [technical details](examples/materials_bandgap_matbench/TECHNICAL_DETAILS.md).

## Scientific and modelling boundary

Data permission and model validity are separate. The worked examples remain
simplified demonstrations, and the scientific studies remain conditional on
their pinned fixtures, likelihoods, priors, splits, dependence assumptions,
and model classes. Reuse for a stronger application requires source-specific
provenance, uncertainty and protocol metadata, a justified identification and
discrepancy model, leakage-safe validation, and domain-appropriate sensitivity
analysis. The README files state the supported take-homes; each
`TECHNICAL_DETAILS.md` records the corresponding assumptions and evidence.

The recorded permission statements were checked on 2026-07-12. Upstream terms
can change; downstream redistributors should re-check the linked authoritative
records.
