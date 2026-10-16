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
    assert errors and any("failed" in e for e in errors)


def _custom(tmp_path: Path, readme: str, manifest: dict, strict: bool | None = None) -> list[str]:
    (tmp_path / "README.md").write_text(readme, encoding="utf-8")
    (tmp_path / "m.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "src.md").write_text("The adapter reached recall of 39.4 percent on the clean subset.\n", encoding="utf-8")
    return rnc.check(tmp_path / "README.md", tmp_path / "m.json", root=tmp_path, strict=strict)[0]


def _contains(text: str, phrase: str) -> dict:
    return {"id": "c", "line_contains": "recall line", "numbers": [{"text": text, "source": {"file": "src.md", "contains": phrase}}]}


def test_a_quoted_source_phrase_passes_and_fails_when_the_phrase_or_number_changes(tmp_path: Path) -> None:
    ok = {"lines": [_contains("39.4", "recall of 39.4 percent")]}
    assert _custom(tmp_path, "recall line 39.4% here\n", ok) == []
    assert _custom(tmp_path, "recall line 39.5% here\n", ok)  # README number altered
    stale = {"lines": [_contains("39.4", "recall of 41.0 percent")]}
    assert _custom(tmp_path, "recall line 39.4% here\n", stale)  # the source phrase no longer exists


def test_strict_mode_fails_on_any_undeclared_claim_line_but_declared_rules_exempt_identifiers(tmp_path: Path) -> None:
    manifest = {
        "strict": True,
        "ignore_rules": [{"name": "version", "pattern": r"\bv\d+\b", "reason": "a method version label, not a result"}],
        "ignore_lines": [{"name": "shell", "pattern": r"^\$ ", "reason": "command lines carry arguments, not claims"}],
        "lines": [_contains("39.4", "recall of 39.4 percent")],
    }
    readme = "recall line 39.4% under v2\n$ run --epochs 5\n"
    assert _custom(tmp_path, readme, manifest) == []
    errors = _custom(tmp_path, readme + "a new claim of 87% appears\n", manifest)
    assert errors and any("unsourced claim" in e for e in errors)
    assert _custom(tmp_path, readme + "a new claim of 87% appears\n", {**manifest, "strict": False}) == []


def test_unary_minus_and_a_stated_tolerance_for_low_precision_sources(tmp_path: Path) -> None:
    (tmp_path / "r.json").write_text(json.dumps({"gain": -1.91, "ci": 0.7895}), encoding="utf-8")
    manifest = {
        "lines": [
            {
                "id": "n",
                "line_contains": "gain line",
                "numbers": [
                    {"text": "1.91", "fmt": "{:.2f}", "source": {"expr": "-v", "vars": {"v": {"file": "r.json", "path": ["gain"]}}}},
                    {"text": "78.9", "tol": 0.051, "source": {"expr": "v * 100", "vars": {"v": {"file": "r.json", "path": ["ci"]}}}},
                ],
            }
        ]
    }
    (tmp_path / "m.json").write_text(json.dumps(manifest), encoding="utf-8")
    ok = tmp_path / "README.md"
    ok.write_text("gain line 1.91 and 78.9\n", encoding="utf-8")
    assert rnc.check(ok, tmp_path / "m.json", root=tmp_path)[0] == []
    ok.write_text("gain line 1.91 and 79.1\n", encoding="utf-8")
    assert rnc.check(ok, tmp_path / "m.json", root=tmp_path)[0]  # outside the tolerance
    ok.write_text("gain line 1.92 and 78.9\n", encoding="utf-8")
    assert rnc.check(ok, tmp_path / "m.json", root=tmp_path)[0]  # exact format still exact


def test_an_ignore_rule_without_a_reason_is_an_error(tmp_path: Path) -> None:
    manifest = {"ignore_rules": [{"name": "x", "pattern": r"\bv\d+\b", "reason": ""}], "lines": []}
    assert _custom(tmp_path, "nothing 1\n", manifest)


def test_the_repository_readme_matches_its_declared_sources() -> None:
    errors, _ = rnc.check(ROOT / "README.md", ROOT / "docs" / "readme_numbers.json")
    assert errors == [], "\n".join(errors)
