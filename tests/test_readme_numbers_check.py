"""scripts/readme_numbers_check.py: it must pass on a true README and FAIL on each way a number can go wrong.

The last test runs the checker on the repository's own README and manifest, so a PR that edits README.md or
a report a declared number comes from fails the required `test` job (that is the CI wiring).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import readme_numbers_check as rnc  # noqa: E402

README = (
    "# Title\n"
    "| Judge mean | vscode | **12.73/15 (84.8%)**, n=11, see ADR-0061 and PR #170 on 2026-10-11 and `qwen3:8b` |\n"
    "| Exact component | both | 44/64 = 68.8% |\n"
    "unrelated line with 999 in it\n"
)
REPORT = {"vscode": {"mean": 12.7273, "n": 11, "share": 0.848}, "exact": {"equal": 44, "n": 64}}
MANIFEST = {
    "lines": [
        {
            "id": "judge",
            "line_contains": "Judge mean",
            "numbers": [
                {"text": "12.73", "fmt": "{:.2f}", "source": {"file": "r.json", "path": ["vscode", "mean"]}},
                {"text": "15", "constant": "the judge scale is /15 by rubric"},
                {"text": "84.8", "fmt": "{:.1f}", "source": {"expr": "s * 100", "vars": {"s": {"file": "r.json", "path": ["vscode", "share"]}}}},
                {"text": "11", "fmt": "{:.0f}", "source": {"file": "r.json", "path": ["vscode", "n"]}},
            ],
        },
        {
            "id": "exact",
            "line_contains": "Exact component",
            "numbers": [
                {"text": "44", "fmt": "{:.0f}", "source": {"file": "r.json", "path": ["exact", "equal"]}},
                {"text": "64", "fmt": "{:.0f}", "source": {"file": "r.json", "path": ["exact", "n"]}},
                {"text": "68.8", "fmt": "{:.1f}", "source": {"expr": "a / b * 100", "vars": {"a": {"file": "r.json", "path": ["exact", "equal"]}, "b": {"file": "r.json", "path": ["exact", "n"]}}}},
            ],
        },
    ]
}


def _run(tmp_path: Path, readme: str = README, report: dict | None = None, manifest: dict | None = None) -> list[str]:
    (tmp_path / "README.md").write_text(readme, encoding="utf-8")
    (tmp_path / "r.json").write_text(json.dumps(report or REPORT), encoding="utf-8")
    (tmp_path / "m.json").write_text(json.dumps(manifest or MANIFEST), encoding="utf-8")
    errors, _ = rnc.check(tmp_path / "README.md", tmp_path / "m.json", root=tmp_path)
    return errors


def test_a_true_readme_passes_and_identifiers_are_not_claims(tmp_path: Path) -> None:
    assert _run(tmp_path) == []


def test_altering_one_readme_number_fails(tmp_path: Path) -> None:
    errors = _run(tmp_path, README.replace("68.8%", "69.8%"))
    assert errors and any("68.8" in e for e in errors)


def test_altering_the_source_value_fails(tmp_path: Path) -> None:
    bad = {**REPORT, "vscode": {**REPORT["vscode"], "mean": 12.5}}
    errors = _run(tmp_path, report=bad)
    assert errors and any("12.73" in e and "12.50" in e for e in errors)


def test_an_undeclared_number_on_a_declared_line_fails(tmp_path: Path) -> None:
    errors = _run(tmp_path, README.replace("n=11", "n=11, 53 k8s"))
    assert errors and any("undeclared number '53'" in e for e in errors)


def test_a_number_appearing_twice_needs_two_declarations(tmp_path: Path) -> None:
    errors = _run(tmp_path, README.replace("44/64 = 68.8%", "44/64 = 68.8% (44 exact)"))
    assert errors and any("undeclared number '44'" in e for e in errors)


def test_a_missing_or_ambiguous_line_fails(tmp_path: Path) -> None:
    assert _run(tmp_path, README.replace("Exact component", "Renamed"))
    assert _run(tmp_path, README + "| Judge mean | again |\n")


def test_a_missing_source_file_fails_instead_of_passing(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(README, encoding="utf-8")
    (tmp_path / "m.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    errors, _ = rnc.check(tmp_path / "README.md", tmp_path / "m.json", root=tmp_path)
    assert errors and any("cannot read source" in e for e in errors)


def test_the_repository_readme_matches_its_declared_sources() -> None:
    errors, _ = rnc.check(ROOT / "README.md", ROOT / "docs" / "readme_numbers.json")
    assert errors == [], "\n".join(errors)
