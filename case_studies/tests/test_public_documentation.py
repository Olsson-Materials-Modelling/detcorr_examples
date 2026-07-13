from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _assert_two_document_pair(root: Path, *, minimum_figures: int) -> None:
    readme = root / "README.md"
    technical = root / "TECHNICAL_DETAILS.md"
    assert readme.is_file()
    assert technical.is_file()

    text = readme.read_text(encoding="utf-8")
    assert "(TECHNICAL_DETAILS.md)" in text
    figures = re.findall(r"!\[[^]]*\]\(([^)]+\.png)\)", text)
    assert len(figures) >= minimum_figures
    for target in figures:
        assert (root / target).resolve().is_file(), target


def test_scientific_projects_follow_the_two_document_contract() -> None:
    _assert_two_document_pair(ROOT, minimum_figures=3)
    registry = json.loads((ROOT / "STUDY_REGISTRY.json").read_text(encoding="utf-8"))
    for row in registry["registered_reanalyses"]:
        _assert_two_document_pair(ROOT / "studies" / row["slug"], minimum_figures=2)


def test_standalone_markdown_surface_contains_only_project_docs_and_legal_notices() -> None:
    registry = json.loads((ROOT / "STUDY_REGISTRY.json").read_text(encoding="utf-8"))
    expected = {
        ROOT / "README.md",
        ROOT / "TECHNICAL_DETAILS.md",
        ROOT / "THIRD_PARTY_NOTICES.md",
        ROOT / "third_party_licenses" / "CC-BY-4.0-NOTICE.md",
        ROOT / "third_party_licenses" / "NIST-PUBLIC-INFORMATION-NOTICE.md",
    }
    expected.update(
        ROOT / "studies" / row["slug"] / filename
        for row in registry["registered_reanalyses"]
        for filename in ("README.md", "TECHNICAL_DETAILS.md")
    )
    observed = {path for path in ROOT.rglob("*.md") if ".pytest_cache" not in path.parts}
    assert observed == expected


def test_standalone_documentation_links_do_not_escape_the_subproject() -> None:
    link_pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    root = ROOT.resolve()
    documents = sorted(path for path in ROOT.rglob("*.md") if ".pytest_cache" not in path.parts)
    for document in documents:
        for raw_target in link_pattern.findall(document.read_text(encoding="utf-8")):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            local_path = target.split("#", maxsplit=1)[0]
            resolved = (document.parent / local_path).resolve()
            assert resolved == root or root in resolved.parents, (
                f"standalone documentation link escapes case_studies/: "
                f"{document.relative_to(ROOT)} -> {target}"
            )
