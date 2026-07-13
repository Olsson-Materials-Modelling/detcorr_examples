from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = (
    ROOT / "studies" / "jarvis_bulk_modulus" / "data" / "FIXTURE_MANIFEST.json",
    ROOT / "studies" / "jarvis_band_gaps" / "data" / "FIXTURE_MANIFEST.json",
    ROOT / "studies" / "fulton_market" / "data" / "FIXTURE_MANIFEST.json",
)
REQUIRED_FIELDS = {
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
    "row_count",
    "source_row_key",
    "dependent_observation_columns",
    "reference_input_columns",
    "units",
    "selection_rule",
    "deduplication_rule",
    "grouping_rule",
    "known_source_dependence",
    "license_status",
    "license_identifier",
    "license_source_url",
    "rights_basis",
    "rights_checked_date",
    "repository_license_applies",
    "third_party_notice",
    "redistribution_decision",
    "transform_code_commit",
    "processed_filename",
    "processed_sha256",
    "protocol_version",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def test_fixture_manifests_have_complete_explicit_provenance() -> None:
    for path in MANIFESTS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert set(payload) >= REQUIRED_FIELDS, f"missing fields in {path}"
        assert payload["evidence_timing_class"] == "registered_reanalysis"
        for field in REQUIRED_FIELDS:
            assert payload[field] not in (None, ""), f"blank {field} in {path}"


def test_bundled_fixture_identity_matches_each_manifest() -> None:
    for path in MANIFESTS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        filename = payload.get("bundled_file")
        if filename is None:
            filename = payload["processed_filename"]
        fixture = path.parent / filename
        assert fixture.is_file(), fixture
        assert _sha256(fixture) == payload["sha256"]
        expected_bytes = payload.get("processed_bytes", payload.get("raw_bytes"))
        assert fixture.stat().st_size == expected_bytes
        with fixture.open("r", encoding="utf-8", newline="") as stream:
            row_count = sum(1 for _ in csv.reader(stream)) - 1
        assert row_count == payload["row_count"]


def test_third_party_fixtures_are_not_silently_mit_licensed() -> None:
    for path in MANIFESTS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["repository_license_applies"] is False
        assert "unresolved" not in payload["license_status"].lower()
        notice = (path.parent / payload["third_party_notice"]).resolve()
        assert notice == ROOT / "THIRD_PARTY_NOTICES.md"
        assert notice.is_file()

    root_license = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "does not license third-party datasets" in root_license


def test_fixture_rights_basis_and_notice_files_are_present() -> None:
    payloads = {
        path.parent.parent.name: json.loads(path.read_text(encoding="utf-8")) for path in MANIFESTS
    }
    assert payloads["jarvis_band_gaps"]["license_identifier"] == "CC-BY-4.0"
    assert payloads["jarvis_bulk_modulus"]["license_identifier"] == "NIST-PUBLIC-INFORMATION"
    assert payloads["fulton_market"]["license_identifier"] == "GPL-3.0-only"

    gpl_path = (MANIFESTS[2].parent / payloads["fulton_market"]["license_file"]).resolve()
    assert gpl_path == ROOT / "third_party_licenses" / "GPL-3.0.txt"
    assert "GNU GENERAL PUBLIC LICENSE" in gpl_path.read_text(encoding="utf-8")
    assert (ROOT / "third_party_licenses" / "CC-BY-4.0-NOTICE.md").is_file()
    assert (ROOT / "third_party_licenses" / "NIST-PUBLIC-INFORMATION-NOTICE.md").is_file()
