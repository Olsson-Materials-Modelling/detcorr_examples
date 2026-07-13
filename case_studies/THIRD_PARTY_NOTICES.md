# Third-party data notices

The root MIT licence covers this repository's original software and
documentation. It does **not** cover the three bundled source-derived data
fixtures or source-bearing reproductions of those fixtures. The notices below
record the applicable permission, attribution, changes, and limitations.

## `detcorr` software dependency — GPL-3.0-only

The case studies install `detcorr` from
[Olsson-Materials-Modelling/detcorr](https://github.com/Olsson-Materials-Modelling/detcorr)
at the commit identified in the README and workflows. That separately installed
dependency is GPL-3.0-only and is not bundled here. This subproject's MIT
licence does not relicense it; reuse remains subject to the
[upstream licence](https://github.com/Olsson-Materials-Modelling/detcorr/blob/a414f063515359f70952e2717a506b75e69e4b0b/LICENSE).

## JARVIS band-gap fixture — CC BY 4.0

Applies to:

- `studies/jarvis_band_gaps/data/jarvis_bandgap_benchmark_fixture.csv`; and
- row-level source fields reproduced from that fixture in
  `reference_results/jarvis_band_gaps/`.

Source and attribution:

> Adapted from K. Choudhary, Q. Zhang, A. C. E. Reid *et al.*,
> “Computational screening of high-performance optoelectronic materials using
> OptB88vdW and TB-mBJ formalisms,” *Scientific Data* **5**, 180082 (2018),
> Table 3, <https://doi.org/10.1038/sdata.2018.82>, and its JARVIS-DFT
> documentation reproduction.

The source article is licensed under the
[Creative Commons Attribution 4.0 International licence](https://creativecommons.org/licenses/by/4.0/).
Its rights statement permits use, adaptation, distribution, and reproduction
with appropriate credit, a licence link, and an indication of changes. The
article's qualification for separately credited third-party material remains
applicable.

Changes made here: the OPT and experimental columns were selected and renamed;
identifiers and material labels were normalized; provenance columns were
added; and the registered positive-domain fixture retained the 54 pairs for
which both values are finite and strictly positive. No source value was fitted,
imputed, clipped, or altered. These changes and the resulting analysis have not
been endorsed by the source authors, Springer Nature, or NIST.

The compact table does not identify the experimental source or measurement
protocol row by row. This is a scientific provenance limitation, not a change
to the CC BY attribution obligation.

## JARVIS bulk-modulus fixture — NIST public information

Applies to:

- `studies/jarvis_bulk_modulus/data/jarvis_bulk_modulus_benchmark_fixture.csv`;
  and
- row-level source fields reproduced from that fixture in
  `reference_results/jarvis_bulk_modulus/`.

Source and attribution:

> Data transcribed from the NIST JARVIS-DFT documentation table “Comparison of
> bulk modulus, K_V (GPa), from vdW-DF-optB88 (OPT) and experiments,”
> <https://jarvis-materials-design.github.io/dbdocs/jarvisdft/>. JARVIS-DFT is
> a National Institute of Standards and Technology research database.

NIST's [Copyrights & Disclaimers](https://www.nist.gov/copyrights-disclaimers)
policy states that unmarked information on NIST sites is public information
that may be distributed or copied and requests appropriate credit. The JARVIS
comparison table carries no separate copyright mark. This repository therefore
distributes the transcription under that NIST public-information permission;
it does not relabel the table as CC BY or place it under MIT.

Changes made here: material names and JARVIS identifiers were normalized into
explicit columns, source notes were added, and the 36 published OPT/experiment
pairs were transcribed without fitting or changing their numeric values. The
source reports an MAE rounded to 8.51 GPa; the fixture recomputes
8.5055555556 GPa. No endorsement by NIST is implied.

NIST provides databases without a warranty of data correctness and disclaims
liability for errors or omissions; the full current statement is at the policy
link above. The comparison table does not supply a row-level experimental
reference, phase/protocol record, uncertainty, or measurement conditions. No
ownership or separate licence is asserted here over the underlying experimental
facts.

## Fulton `fish` fixture — GPL-3 package content with retained caveat

Applies to:

- `studies/fulton_market/data/fish.csv`; and
- row-level source fields reproduced from that fixture in
  `reference_results/fulton_market/`.

Source and attribution:

> `fish` from the official `wooldridge` R data package, exported to CSV by
> Rdatasets. The package documentation attributes the observations to K.
> Graddy, “Testing for Imperfect Competition at the Fulton Fish Market,”
> *RAND Journal of Economics* **26** (1995), 75–92, and states that Joshua
> Angrist provided the data to Wooldridge.

The official [`wooldridge` package](https://CRAN.R-project.org/package=wooldridge)
is a data package that distributes `fish` as package content and declares
GPL-3. This repository conveys the CSV separately under GPL-3 to the extent
that the package licence applies. The complete licence is in
[`third_party_licenses/GPL-3.0.txt`](third_party_licenses/GPL-3.0.txt). The CSV
itself is the preferred editable form supplied here; the official package and
the [Rdatasets snapshot](https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/wooldridge/fish.csv)
are the corresponding upstream sources.

Transformation notice: Rdatasets serialized the package table to CSV and added
the `rownames` index column. This repository copied that CSV byte for byte and
made no further change to the observations.

Rdatasets [states explicitly](https://github.com/vincentarelbundock/Rdatasets#license)
that its GPL declaration covers its code and copied documentation, while it
could not definitively establish the underlying rights in every hosted table.
That limitation is retained. We do not represent GPL-3 as proof of rights held
by the original researcher, data provider, textbook author, or publisher, and
we assert no ownership over the factual observations.

## Reference results and downstream redistribution

The study code, statistical models, original narrative, and original aggregate
analysis are MIT-licensed. A reference-result file or figure that reproduces
row-level source data remains subject to the relevant notice above. Preserve
the source attribution when redistributing those artifacts. Generated results
do not erase the source-data obligations or imply source-provider endorsement.

These terms were checked on 2026-07-12. Upstream terms can change; a downstream
redistributor should re-check the linked authoritative records.
