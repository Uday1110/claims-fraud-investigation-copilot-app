"""Citation-Resolves Rule: every evidence citation in the docs must resolve to a committed artifact.

Checked citation forms (in docs/*.md and README.md):
* `path/to/file.ext`                  -> file (or directory) exists
* `path/to/file.py::name[.name]`      -> the file defines that function / class / constant
* run_id `<16 hex>` + span_id `<16 hex>` on the same line -> that span exists in traces/phoenix_spans.jsonl and
  belongs to that run
* `logs/<file>.jsonl#L<n>`            -> line n exists (and its run_id matches when one is cited on the same line)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = sorted((ROOT / "docs").glob("*.md")) + [ROOT / "README.md"]
PATH_RE = re.compile(r"`((?:src|mcp_server|scripts|evals|tests|data|logs|traces|reports|docs|evidence_archive)/[\w./\-]*|README\.md|"
                     r"requirements\.txt|\.env\.example|\.gitignore|pytest\.ini)(?:::([\w.]+))?(?:#L(\d+))?`")
RUN_RE = re.compile(r"run_id[`\s:=]*`?([0-9a-f]{16})`?")
SPAN_RE = re.compile(r"span_id[`\s:=]*`?([0-9a-f]{16})`?")


def _spans() -> dict[str, dict]:
    """Spans of the current export plus archived runs (evidence_archive/<run>/traces/phoenix_spans.jsonl)."""
    out = {}
    for p in [ROOT / "traces" / "phoenix_spans.jsonl", *sorted(ROOT.glob("evidence_archive/*/traces/phoenix_spans.jsonl"))]:
        for line in p.read_text("utf-8").splitlines() if p.exists() else []:
            r = json.loads(line)
            out[r["context.span_id"]] = r
    return out


def _defines(path: Path, dotted: str) -> bool:
    text = path.read_text("utf-8", errors="ignore")
    name = dotted.split(".")[-1]
    return bool(re.search(rf"^\s*(async\s+def|def|class)\s+{re.escape(name)}\b|^\s*{re.escape(name)}\s*[:=]", text, re.M))


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_path_and_symbol_citations_resolve(doc):
    if not doc.exists():
        pytest.skip("doc not written yet")
    missing = []
    for line in doc.read_text("utf-8").splitlines():
        for path, symbol, lineno in PATH_RE.findall(line):
            p = ROOT / path
            if not p.exists():
                missing.append(path)
            elif symbol and p.is_file() and not _defines(p, symbol):
                missing.append(f"{path}::{symbol}")
            elif lineno:
                lines = p.read_text("utf-8").splitlines()
                if int(lineno) > len(lines):
                    missing.append(f"{path}#L{lineno}")
                else:
                    runs = RUN_RE.findall(line)
                    rec = json.loads(lines[int(lineno) - 1])
                    if runs and rec.get("run_id") not in runs:
                        missing.append(f"{path}#L{lineno} run_id mismatch")
    assert not missing, f"{doc.name}: unresolvable citations {missing}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_phoenix_span_citations_resolve(doc):
    if not doc.exists():
        pytest.skip("doc not written yet")
    spans = _spans()
    bad = []
    for line in doc.read_text("utf-8").splitlines():
        for sid in SPAN_RE.findall(line):
            s = spans.get(sid)
            runs = RUN_RE.findall(line)
            if s is None:
                bad.append(f"span {sid} not in traces/phoenix_spans.jsonl")
            elif runs and s.get("run_id") not in runs:
                bad.append(f"span {sid} belongs to run {s.get('run_id')}, cited with {runs}")
    assert not bad, f"{doc.name}: {bad}"


def test_failure_analysis_has_three_cited_failures():
    doc = ROOT / "docs" / "failure-analysis.md"
    if not doc.exists():
        pytest.skip("not written yet")
    sections = [s for s in doc.read_text("utf-8").split("\n## ")[1:] if s.lower().startswith("failure")]
    assert len(sections) >= 3
    for s in sections:
        assert SPAN_RE.search(s) or re.search(r"logs/\w+\.jsonl#L\d+", s), s[:80]
        assert "root cause" in s.lower() and "fix" in s.lower()
