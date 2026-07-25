"""Tests cho pipeline reuse_rag (phần deterministic — không gọi LLM)."""

from __future__ import annotations

from reuse_rag.indexer import index_source
from reuse_rag.render import context_token_estimate, render_context
from reuse_rag.reuse_scorer import score_reuse
from reuse_rag.tasks import load_repo_tasks

SAMPLE_SRC = '''
def chunked(src, size, count=None):
    """Split *src* into chunks of *size*.

    >>> chunked([1, 2, 3], 2)
    [[1, 2], [3]]
    """
    return [src[i:i + size] for i in range(0, len(src), size)]


def _private(x):
    return x
'''


# --- indexer ---------------------------------------------------------------
def test_index_source_extracts_public_function():
    chunks = index_source(SAMPLE_SRC, "pkg.mod", "pkg/mod.py")
    assert "pkg.mod.chunked" in chunks
    assert "pkg.mod._private" not in chunks  # private bị bỏ
    c = chunks["pkg.mod.chunked"]
    assert c["signature"] == "def chunked(src, size, count=None):"
    assert "Split" in c["docstring"]
    assert ">>>" not in c["docstring"]          # doctest đã bị tách khỏi prose
    assert c["example"].startswith(">>> chunked")
    assert "return [src[i" in c["body"]


# --- render ----------------------------------------------------------------
def _chunk():
    return index_source(SAMPLE_SRC, "pkg.mod", "pkg/mod.py")["pkg.mod.chunked"]


def test_render_levels_are_monotonic_in_tokens():
    c = _chunk()
    toks = [context_token_estimate(render_context(c, lvl)) for lvl in (1, 2, 3, 4)]
    assert toks[0] < toks[1] < toks[2] <= toks[3]


def test_render_level_contents():
    c = _chunk()
    l1 = render_context(c, 1)
    l2 = render_context(c, 2)
    l3 = render_context(c, 3)
    l4 = render_context(c, 4)
    assert "# Project API: pkg.mod.chunked" in l1
    assert "from pkg.mod import chunked" in l1
    assert '"""' not in l1 and '"""' in l2          # docstring chỉ từ L2
    assert "# Example:" not in l2 and "# Example:" in l3
    assert "# Full implementation:" in l4 and "range(0, len(src)" in l4


# --- reuse_scorer ----------------------------------------------------------
def test_reuse_from_import():
    code = "from boltons.iterutils import chunked\ndef w(x, n):\n    return chunked(x, n)"
    r = score_reuse(code, "boltons.iterutils", "chunked")
    assert r.reused and r.imported and not r.self_defined


def test_reuse_submodule_attribute():
    code = "from boltons import iterutils\ndef w(x, n):\n    return iterutils.chunked(x, n)"
    assert score_reuse(code, "boltons.iterutils", "chunked").reused


def test_reuse_full_path():
    code = "import boltons.iterutils\ndef w(x, n):\n    return boltons.iterutils.chunked(x, n)"
    assert score_reuse(code, "boltons.iterutils", "chunked").reused


def test_false_positive_self_defined_not_reuse():
    code = "def chunked(x, n):\n    return [x]\ndef w(x, n):\n    return chunked(x, n)"
    r = score_reuse(code, "boltons.iterutils", "chunked")
    assert r.self_defined and not r.reused


def test_no_reuse_reimplementation():
    code = "def w(x, n):\n    return [x[i:i+n] for i in range(0, len(x), n)]"
    assert not score_reuse(code, "boltons.iterutils", "chunked").reused


def test_syntax_error_is_not_reuse():
    assert not score_reuse("def w(:\n bad", "boltons.iterutils", "chunked").reused


# --- tasks -----------------------------------------------------------------
def test_repo_tasks_load_and_convert():
    tasks = load_repo_tasks("data/repo_tasks.json")
    assert len(tasks) >= 12
    t = tasks[0]
    assert t.target_fqn == "boltons.iterutils.chunked"
    ht = t.to_harness_task()
    assert ht.task_id == t.task_id and ht.test_list == t.test_list
