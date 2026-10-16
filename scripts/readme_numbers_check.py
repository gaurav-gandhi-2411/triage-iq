"""Check that the numbers in README.md still match the sources they came from.

docs/readme_numbers.json declares, per README line, every number on that line and where it comes from:

    {"id": "llm-vscode", "line_contains": "**Current mean, regression detector (cassette",
     "numbers": [
       {"text": "12.73", "fmt": "{:.2f}", "source": {"file": "reports/eval_baseline.json",
                                                      "path": ["per_repo", "microsoft/vscode", "mean"]}},
       {"text": "15", "constant": "the judge scale is /15 by rubric"}]}

A source is one of (tier A is a re-computation, tier B an anchored quotation):
  A  {"file", "path"}         a JSON field (path is a list of keys/indexes), formatted with `fmt`
  A  {"expr", "vars"}         arithmetic (+ - * /) over named sources
  A  {"file", "regex"}        first group of the first match in a text file, converted to a number, formatted
  B  {"file", "contains"}     a phrase that must still appear verbatim in the file AND contain the number
A `constant` is for definitions only (a rubric scale, a score level, a config value with no file); it needs a
reason. A MEASURED number is never a constant: it gets a source or the README is fixed.

The check, per declared line:
  1. exactly one README line contains `line_contains`;
  2. every declared `text` is a claim number on that line and its source re-reads to it;
  3. COMPLETENESS: the claim numbers on the line are exactly the declared ones (the same number twice needs two
     declarations). Identifiers are not claims; they are removed by declared rules before counting: the built-in
     rules below plus the manifest's `ignore_rules` (regex + reason) and `ignore_lines` (whole-line regex + reason).
With "strict": true in the manifest (or --strict) a README line that still carries a claim number but is not
declared is also an error, so a new unsourced number cannot enter the README.

Usage: python scripts/readme_numbers_check.py [--readme README.md] [--manifest docs/readme_numbers.json]
                                              [--strict] [--list-uncovered]
Exit 0 = every declared number matches (and, in strict mode, nothing is undeclared); 1 otherwise.
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
# Built-in identifier rules: they contain digits but are not claims about results.
BUILTIN_RULES = [
    ("markdown link URL", r"\]\([^)]*\)"),  # the visible link text stays and is checked
    ("code span", r"`[^`]*`"),  # model names, hashes, file names, flags
    ("ADR reference", r"\bADR-\d+(?:\s*[/,]\s*\d+)*|\bADRs?\s+\d+"),
    ("PR / issue reference", r"\bPRs?\s*#?\d+|(?<![\w.])#\d+"),
    ("ISO date", r"\b\d{4}-\d{2}-\d{2}\b"),
    ("decision reference", r"\bD\d+(?:[-–]D?\d+)?\b"),
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


def source_kind(src: dict) -> str:
    return "contains" if "contains" in src else "expr" if "expr" in src else "regex" if "regex" in src else "json"


def claim_numbers(line: str, extra_rules: list[tuple[str, str]] | None = None, removed: Counter | None = None) -> list[str]:
    for name, pat in BUILTIN_RULES + (extra_rules or []):
        def blank(m: re.Match, name: str = name) -> str:
            if removed is not None and NUMBER.search(m.group(0)):
                removed[name] += 1
            return " " * len(m.group(0))

        line = re.sub(pat, blank, line)
    return NUMBER.findall(line)


def check(readme: Path, manifest: Path, root: Path = ROOT, strict: bool | None = None) -> tuple[list[str], dict]:
    errors: list[str] = []
    lines = readme.read_text(encoding="utf-8").splitlines()
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    strict = spec.get("strict", False) if strict is None else strict
    rules = [(r["name"], r["pattern"]) for r in spec.get("ignore_rules", [])]
    skip = [(r["name"], re.compile(r["pattern"])) for r in spec.get("ignore_lines", [])]
    for r in spec.get("ignore_rules", []) + spec.get("ignore_lines", []):
        if not str(r.get("reason", "")).strip():
            errors.append(f"ignore rule {r.get('name')!r} has no reason")
    removed: Counter = Counter()
    kinds: Counter = Counter()
    verified, constants, touched, claim_tokens = 0, [], set(), 0
    for entry in spec.get("lines", []):
        eid = entry["id"]
        hits = [i for i, ln in enumerate(lines) if entry["line_contains"] in ln]
        if len(hits) != 1:
            errors.append(f"[{eid}] line_contains matches {len(hits)} README lines (need exactly 1): {entry['line_contains']!r}")
            continue
        idx = hits[0]
        touched.add(idx)
        present = Counter(claim_numbers(lines[idx], rules, removed))
        claim_tokens += sum(present.values())
        declared = Counter(n["text"] for n in entry["numbers"])
        for text in declared - present:
            errors.append(f"[{eid}] README line {idx + 1} no longer has {text!r} as a number (declared {declared[text]}x, found {present[text]}x)")
        for text in present - declared:
            errors.append(f"[{eid}] README line {idx + 1} has undeclared number {text!r} (found {present[text]}x, declared {declared[text]}x); declare it with a source")
        for num in entry["numbers"]:
            text = num["text"]
            if "constant" in num:
                if not str(num["constant"]).strip():
                    errors.append(f"[{eid}] constant {text!r} has no reason")
                constants.append(f"{eid}: {text} ({num['constant']})")
                kinds["constant"] += 1
                continue
            src = num["source"]
            kinds[source_kind(src)] += 1
            try:
                if "contains" in src:
                    body = (root / src["file"]).read_text(encoding="utf-8")
                    if src["contains"] not in body:
                        raise ValueError(f"phrase {src['contains']!r} not found in {src['file']}")
                    if text not in src["contains"]:
                        raise ValueError(f"phrase {src['contains']!r} does not contain {text!r}")
                    verified += 1
                    continue
                value = resolve(root, src)
            except Exception as exc:  # noqa: BLE001 - any unreadable source is a failed check; say which one
                errors.append(f"[{eid}] {text!r}: source {src} failed: {exc}")
                continue
            shown = num.get("fmt", "{}").format(value)
            if shown != text:
                errors.append(f"[{eid}] README line {idx + 1} says {text!r} but the source gives {shown!r} ({src})")
            else:
                verified += 1
    uncovered = []
    for i, ln in enumerate(lines):
        if i in touched or any(p.search(ln) for _, p in skip):
            continue
        toks = claim_numbers(ln, rules)
        if toks:
            uncovered.append((i + 1, len(toks), ln.strip()[:100]))
            if strict:
                errors.append(f"README line {i + 1} carries {len(toks)} unsourced claim number(s): {ln.strip()[:80]!r}")
    claim_lines = len(touched) + len(uncovered)
    info = {
        "claim_lines": claim_lines,
        "declared_lines": len(touched),
        "uncovered": uncovered,
        "claim_numbers_declared": claim_tokens,
        "claim_numbers_uncovered": sum(k for _, k, _ in uncovered),
        "verified_numbers": verified,
        "source_kinds": dict(kinds),
        "constants": constants,
        "ignored_by_rule": dict(removed),
        "strict": strict,
    }
    return errors, info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--readme", default=str(ROOT / "README.md"))
    ap.add_argument("--manifest", default=str(ROOT / "docs" / "readme_numbers.json"))
    ap.add_argument("--strict", action="store_true", help="fail on any claim-bearing README line that is not declared")
    ap.add_argument("--list-uncovered", action="store_true")
    a = ap.parse_args()
    errors, info = check(Path(a.readme), Path(a.manifest), strict=True if a.strict else None)
    pct = 100 * info["declared_lines"] / info["claim_lines"] if info["claim_lines"] else 100.0
    print(f"claim-bearing README lines: {info['claim_lines']}; declared: {info['declared_lines']} ({pct:.1f}%); uncovered: {len(info['uncovered'])}")
    print(f"claim numbers: {info['claim_numbers_declared']} declared, {info['claim_numbers_uncovered']} uncovered; verified against a source: {info['verified_numbers']}")
    print(f"source kinds: {info['source_kinds']}  (constants are definitions only; each has a reason)")
    print(f"identifier tokens removed by declared rules: {info['ignored_by_rule']}")
    print(f"strict mode: {info['strict']}")
    if a.list_uncovered:
        for n, k, t in info["uncovered"]:
            print(f"  uncovered line {n} ({k}): {t}")
    for e in errors:
        print("FAIL", e)
    print("OK" if not errors else f"{len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
