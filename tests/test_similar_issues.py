"""Unit tests for src/triage_iq/models/similar_issues.py's LEVER 1 (tokenizer-based corpus
truncation) and LEVER 2 (BGE query instruction prefix) fixes.

Uses a minimal stub tokenizer (same interface as a HF fast tokenizer's .encode()/.decode())
so these tests isolate _build_text()'s truncation logic from any real model load -- fast,
deterministic, no GPU/network dependency.
"""

from __future__ import annotations

import pandas as pd

from triage_iq.models.similar_issues import (
    QUERY_INSTRUCTION_REPO_OVERRIDE,
    QUERY_INSTRUCTIONS,
    SimilarIssueRetriever,
    _build_text,
)


class _StubTokenizer:
    """Whitespace tokenizer: each word is one token id, decode joins with spaces.

    Good enough to test the truncation ARITHMETIC (reserve title tokens, cut body to the
    remaining budget) without depending on BERT's real wordpiece vocabulary.
    """

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> list[int]:
        words = text.split()
        ids = list(range(len(words)))
        self._last_words = words  # noqa: SLF001 -- test-only introspection hook
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        words = self._last_words
        return " ".join(words[i] for i in ids if i < len(words))


def test_build_text_without_tokenizer_uses_legacy_char_truncation() -> None:
    title = pd.Series(["Bug report"])
    body = pd.Series(["x" * 1000])
    out = _build_text(title, body, tokenizer=None, max_body=512)
    assert out == ["Bug report. " + "x" * 512]


def test_build_text_with_tokenizer_does_not_truncate_short_body() -> None:
    title = pd.Series(["Crash on startup"])
    body = pd.Series(["short body text here"])
    out = _build_text(title, body, tokenizer=_StubTokenizer(), max_tokens=512)
    assert out == ["Crash on startup. short body text here"]


def test_build_text_with_tokenizer_truncates_long_body_to_token_budget() -> None:
    title = pd.Series(["T"])
    body_words = [f"word{i}" for i in range(1000)]
    body = pd.Series([" ".join(body_words)])
    out = _build_text(title, body, tokenizer=_StubTokenizer(), max_tokens=20)
    # prefix "T. " = 2 tokens ("T", ".")-ish per the stub's whitespace split -- the exact
    # count depends on the stub, but the key invariant is: body is cut, not passed through
    # whole, and the result stays within a small margin of max_tokens once re-tokenized.
    result_text = out[0]
    n_result_tokens = len(_StubTokenizer().encode(result_text))
    assert n_result_tokens <= 20
    assert "word999" not in result_text  # the tail of the 1000-word body must be gone


def test_build_text_reserves_room_for_longer_titles() -> None:
    """A longer title should leave less budget for the body than a short title, given the
    same max_tokens -- this is what "reserve room for the title" means concretely."""
    body_words = [f"w{i}" for i in range(100)]
    body = pd.Series([" ".join(body_words)])

    short_title = pd.Series(["A"])
    long_title = pd.Series(["A B C D E F G H I J"])

    out_short = _build_text(short_title, body, tokenizer=_StubTokenizer(), max_tokens=30)
    out_long = _build_text(long_title, body, tokenizer=_StubTokenizer(), max_tokens=30)

    n_words_short = len(out_short[0].split(". ", 1)[1].split())
    n_words_long = len(out_long[0].split(". ", 1)[1].split())
    assert n_words_long < n_words_short


def _retriever_stub(model_key: str, repo: str = "some_other_repo") -> SimilarIssueRetriever:
    """Build a SimilarIssueRetriever without loading a real SentenceTransformer -- only
    model_key/repo are needed by _apply_query_instruction()."""
    obj = SimilarIssueRetriever.__new__(SimilarIssueRetriever)
    obj.model_key = model_key
    obj.repo = repo
    return obj


def test_query_instruction_default_applies_for_bge_on_a_repo_with_no_override() -> None:
    r = _retriever_stub("bge")
    out = r._apply_query_instruction("how do I fix X", None)  # noqa: SLF001
    assert out == QUERY_INSTRUCTIONS["bge"] + "how do I fix X"


def test_query_instruction_default_is_noop_for_minilm() -> None:
    r = _retriever_stub("minilm")
    out = r._apply_query_instruction("how do I fix X", None)  # noqa: SLF001
    assert out == "how do I fix X"


def test_query_instruction_explicit_override_true() -> None:
    r = _retriever_stub("bge")
    out = r._apply_query_instruction("q", True)  # noqa: SLF001
    assert out == QUERY_INSTRUCTIONS["bge"] + "q"


def test_query_instruction_explicit_override_false() -> None:
    r = _retriever_stub("bge")
    out = r._apply_query_instruction("q", False)  # noqa: SLF001
    assert out == "q"


def test_query_instruction_repo_override_on_for_k8s() -> None:
    """ADR-0040: k8s gets the instruction by default even without an explicit override."""
    r = _retriever_stub("bge", repo="kubernetes_kubernetes")
    out = r._apply_query_instruction("q", None)  # noqa: SLF001
    assert out == QUERY_INSTRUCTIONS["bge"] + "q"
    assert QUERY_INSTRUCTION_REPO_OVERRIDE["kubernetes_kubernetes"] is True


def test_query_instruction_repo_override_off_for_vscode() -> None:
    """ADR-0040: vscode's per-repo override wins over the model's own bge default -- this is
    the whole point of the override (bge's blanket default would otherwise turn it on here)."""
    r = _retriever_stub("bge", repo="microsoft_vscode")
    out = r._apply_query_instruction("q", None)  # noqa: SLF001
    assert out == "q"
    assert QUERY_INSTRUCTION_REPO_OVERRIDE["microsoft_vscode"] is False


def test_query_instruction_repo_override_beaten_by_explicit_flag() -> None:
    """The explicit True/False override (eval A/B isolation) beats even the repo override."""
    r = _retriever_stub("bge", repo="microsoft_vscode")
    out = r._apply_query_instruction("q", True)  # noqa: SLF001
    assert out == QUERY_INSTRUCTIONS["bge"] + "q"


def test_build_index_never_gets_query_instruction() -> None:
    """The instruction is query-side only (BGE's own asymmetric training convention) --
    _build_text() (used only by build_index()) must never see it."""
    assert (
        "Represent this sentence"
        not in _build_text(pd.Series(["T"]), pd.Series(["B"]), tokenizer=None)[0]
    )
