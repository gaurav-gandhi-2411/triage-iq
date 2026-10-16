"""Check that the numbers in README.md still match the report files they came from.

docs/readme_numbers.json declares, per README line, every number on that line and where it comes from:

    {"id": "llm-vscode", "line_contains": "**Current mean, regression detector (cassette",
     "numbers": [
       {"text": "12.73", "fmt": "{:.2f}", "source": {"file": "reports/eval_baseline.json",
                                                      "path": ["per_repo", "microsoft/vscode", "mean"]}},
       {"text": "15", "constant": "the judge scale is /15 by rubric"}]}

A source is one of: {"file", "path"} (JSON field, path is a list of keys), {"file", "regex"} (first group of the
first match in a text file), or {"expr", "vars"} (arithmetic over named sources, + - * / only).
The check, per declared line:
  1. exactly one README line contains `line_contains`;
  2. every declared `text` is a number on that line and the source, re-read NOW and formatted with `fmt`,
     equals it;
  3. COMPLETENESS: the numbers on the line, after removing identifiers that are not claims (backticked code,
     ADR/PR/issue/decision references, dates, link URLs), are exactly the declared ones. A line cannot be
     half-checked, and the same number twice needs two declarations.
A `constant` needs a reason and is listed in the output as a declared constant with no source.

Lines not named in the manifest are not checked; the summary prints how many numeric tokens sit on such lines.
Usage: python scripts/readme_numbers_check.py [--readme README.md] [--manifest docs/readme_numbers.json]
Exit 0 = all declared numbers match, 1 = a mismatch / undeclared number / bad manifest.
"""

from __future__ import annotations

import argparse
import ast
import json
import operator
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
# Identifiers that contain digits but are not claims about results.
NOT_CLAIMS = [
    r"\]\([^)]*\)",  # the URL part of a markdown link; the visible link text stays and is checked
    r"`[^`]*`",  # code spans: model names, hashes, file names, flags
    r"\bADR-\d+(?:\s*[/,]\s*\d+)*",
    r"\bADRs?\s+\d+",
    r"\bPRs?\s*#?\d+",
    r"(?<![\w.])#\d+",
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\bD\d+(?:[-–]D?\d+)?\b",
]
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def _eval(node: ast.AST, env: dict[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body, env)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left, env), _eval(node.right, env))
    raise ValueError(f"unsupported expression element: {ast.dump(node)}")


def resolve(root: Path, src: dict) -> float:
    if "expr" in src:
        env = {name: resolve(root, s) for name, s in src["vars"].items()}
        return _eval(ast.parse(src["expr"], mode="eval"), env)
    text = (root / src["file"]).read_text(encoding="utf-8")
    if "regex" in src:
        m = re.search(src["regex"], text)
        if not m:
            raise ValueError(f"regex {src['regex']!r} not found in {src['file']}")
        return float(m.group(1).replace(",", ""))
    cur = json.loads(text)
    for k in src["path"]:
        cur = cur[k]
    return float(cur)


def claim_numbers(line: str) -> list[str]:
    for pat in NOT_CLAIMS:
        line = re.sub(pat, lambda m: " " * len(m.group(0)), line)
    return NUMBER.findall(line)


def check(readme: Path, manifest: Path, root: Path = ROOT) -> tuple[list[str], dict]:
    errors: list[str] = []
    lines = readme.read_text(encoding="utf-8").splitlines()
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    verified, constants, touched = 0, [], set()
    for entry in spec.get("lines", []):
        eid = entry["id"]
        hits = [i for i, ln in enumerate(lines) if entry["line_contains"] in ln]
        if len(hits) != 1:
            errors.append(f"[{eid}] line_contains matches {len(hits)} README lines (need exactly 1): {entry['line_contains']!r}")
            continue
        idx = hits[0]
        touched.add(idx)
        present = Counter(claim_numbers(lines[idx]))
        declared = Counter(n["text"] for n in entry["numbers"])
        for text in declared - present:
            errors.append(f"[{eid}] README line {idx + 1} no longer has {text!r} as a number (declared {declared[text]}x, found {present[text]}x)")
        for text in present - declared:
            errors.append(f"[{eid}] README line {idx + 1} has undeclared number {text!r} (found {present[text]}x, declared {declared[text]}x); declare it with a source or a constant")
        for num in entry["numbers"]:
            text = num["text"]
            if "constant" in num:
                if not str(num["constant"]).strip():
                    errors.append(f"[{eid}] constant {text!r} has no reason")
                constants.append(f"{eid}: {text} ({num['constant']})")
                continue
            try:
                value = resolve(root, num["source"])
            except Exception as exc:  # noqa: BLE001 - any unreadable source is a failed check; say which one
                errors.append(f"[{eid}] {text!r}: cannot read source {num['source']}: {exc}")
                continue
            shown = num.get("fmt", "{}").format(value)
            if shown != text:
                errors.append(f"[{eid}] README line {idx + 1} says {text!r} but the source gives {shown!r} ({num['source']})")
            else:
                verified += 1
    other = sum(len(NUMBER.findall(ln)) for i, ln in enumerate(lines) if i not in touched)
    info = {"declared_lines": len(touched), "verified_numbers": verified, "constants": constants, "numeric_tokens_on_unchecked_lines": other}
    return errors, info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--readme", default=str(ROOT / "README.md"))
    ap.add_argument("--manifest", default=str(ROOT / "docs" / "readme_numbers.json"))
    a = ap.parse_args()
    errors, info = check(Path(a.readme), Path(a.manifest))
    print(f"lines checked: {info['declared_lines']}; numbers re-read from their source and matching: {info['verified_numbers']}")
    print(f"declared constants (no source): {len(info['constants'])}")
    for c in info["constants"]:
        print(f"  - {c}")
    print(f"numeric tokens on README lines NOT in the manifest (unchecked): {info['numeric_tokens_on_unchecked_lines']}")
    for e in errors:
        print("FAIL", e)
    print("OK" if not errors else f"{len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
