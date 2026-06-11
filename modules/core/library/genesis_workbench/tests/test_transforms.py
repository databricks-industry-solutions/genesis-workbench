"""Unit tests for the Vortex/ai_canvas transform ops (`_run_transform`) and the
`_dig` path resolver they rely on.

Transforms are the deterministic reshape steps between nodes — extracting a field,
mapping/renaming, selecting top-K, parsing files, SMILES→PDB. A transform that
silently returns null lets a downstream node run on garbage and the whole job
report success with a fake result (see protein_design→extract_field→deepstabp),
so these tests pin both the happy path AND the fail-loud-on-null behaviour.

Run: pytest tests/test_transforms.py   (needs the package deps + py3.11+).
"""
import sys
from pathlib import Path

import pytest

# Import the package from src/ without an install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genesis_workbench import executor as ex  # noqa: E402
from genesis_workbench.executor import _dig, _run_transform  # noqa: E402

# Every op the dispatcher knows — keep in sync; the coverage test asserts each is exercised.
ALL_OPS = {
    "read_text_file", "parse_fasta", "csv_column", "extract_field",
    "field_mapper", "select_top_k", "smiles_to_pdb",
}
_TESTED_OPS: set[str] = set()


def run(op, inputs=None, params=None):
    _TESTED_OPS.add(op)
    return _run_transform(None, op, inputs or {}, params or {})


# ── _dig ─────────────────────────────────────────────────────────────────────
def test_dig_dict_path():
    assert _dig({"a": {"b": 7}}, "a.b") == 7


def test_dig_list_index():
    assert _dig([{"x": 1}, {"x": 2}], "1.x") == 2


def test_dig_single_element_list_unwraps():
    # field path against a 1-element list auto-descends (top-K with k=1)
    assert _dig([{"seq": "ABC"}], "seq") == "ABC"


def test_dig_field_on_string_returns_none():
    # the real bug: `[0].sequence` on a list of PDB *strings*
    assert _dig(["PARENT N/A\nATOM ...", "ATOM ..."], "0.sequence") is None


def test_dig_index_out_of_range_returns_none():
    assert _dig([1, 2], "5") is None


def test_dig_empty_path_returns_obj():
    assert _dig({"a": 1}, "") == {"a": 1}


# ── extract_field ──────────────────────────────────────────────────────────────
def test_extract_field_returns_value():
    out = run("extract_field", {"data": {"results": [{"smiles": "CCO"}]}}, {"path": "results.0.smiles"})
    assert out == {"value": "CCO"}


def test_extract_field_raises_when_path_unresolved():
    # mirrors protein_design.designs ([0].sequence on PDB strings) → None → must fail
    with pytest.raises(RuntimeError, match="did not resolve"):
        run("extract_field", {"data": ["PARENT N/A\nATOM ..."]}, {"path": "[0].sequence"})


# ── field_mapper ───────────────────────────────────────────────────────────────
def test_field_mapper_maps_paths():
    out = run(
        "field_mapper",
        {"data": {"a": {"b": 1}, "c": 2}},
        {"mappings": {"first": "a.b", "second": "c"}},
    )
    assert out == {"mapped": {"first": 1, "second": 2}}


def test_field_mapper_accepts_json_string_mapping():
    out = run("field_mapper", {"data": {"x": 9}}, {"mappings": '{"y": "x"}'})
    assert out == {"mapped": {"y": 9}}


def test_field_mapper_raises_when_all_none():
    with pytest.raises(RuntimeError, match="none of the source paths"):
        run("field_mapper", {"data": {"x": 1}}, {"mappings": {"a": "missing", "b": "also.missing"}})


# ── select_top_k ───────────────────────────────────────────────────────────────
def test_select_top_k_sorts_desc_by_key():
    items = [{"s": 1}, {"s": 9}, {"s": 5}]
    out = run("select_top_k", {"items": items}, {"by": "s", "k": 2})
    assert out == {"top": [{"s": 9}, {"s": 5}]}


def test_select_top_k_ascending():
    items = [{"s": 1}, {"s": 9}, {"s": 5}]
    out = run("select_top_k", {"items": items}, {"by": "s", "k": 2, "order": "asc"})
    assert out == {"top": [{"s": 1}, {"s": 5}]}


def test_select_top_k_without_key_truncates():
    out = run("select_top_k", {"items": [1, 2, 3, 4]}, {"k": 2})
    assert out == {"top": [1, 2]}


# ── file-backed ops (monkeypatch the volume reader) ─────────────────────────────
def test_parse_fasta(monkeypatch):
    fasta = ">a\nMKT\nAYI\n>b\nGGG\n"
    monkeypatch.setattr(ex, "_read_volume_text", lambda w, p: fasta)
    out = run("parse_fasta", {"file": "/Volumes/x/y/z.fasta"})
    assert out == {"sequences": ["MKTAYI", "GGG"]}


def test_csv_column(monkeypatch):
    monkeypatch.setattr(ex, "_read_volume_text", lambda w, p: "name,score\na,1\nb,2\n")
    out = run("csv_column", {"table": "/Volumes/x/y/z.csv"}, {"column": "score"})
    assert out == {"values": [1, 2]}


def test_csv_column_missing_column_returns_empty(monkeypatch):
    monkeypatch.setattr(ex, "_read_volume_text", lambda w, p: "name\na\n")
    out = run("csv_column", {"table": "/Volumes/x/y/z.csv"}, {"column": "nope"})
    assert out == {"values": []}


def test_read_text_file(monkeypatch):
    monkeypatch.setattr(ex, "_read_volume_text", lambda w, p: "hello world")
    out = run("read_text_file", {"file": "/Volumes/x/y/z.txt"})
    assert out == {"text": "hello world"}


# ── smiles_to_pdb (rdkit) ───────────────────────────────────────────────────────
def test_smiles_to_pdb_valid():
    pytest.importorskip("rdkit")
    out = run("smiles_to_pdb", {"smiles": "CCO"})
    assert "ATOM" in out["pdb"] or "HETATM" in out["pdb"]


def test_smiles_to_pdb_invalid_raises():
    pytest.importorskip("rdkit")
    with pytest.raises(RuntimeError, match="Invalid SMILES"):
        run("smiles_to_pdb", {"smiles": "not-a-molecule!!!"})


# ── misc ────────────────────────────────────────────────────────────────────────
def test_unknown_op_raises():
    with pytest.raises(RuntimeError, match="Unknown transform op"):
        _run_transform(None, "bogus_op", {}, {})


def test_zzz_all_ops_covered():
    """Guard: every op the dispatcher handles is exercised above (smiles via importorskip
    still counts — it's invoked). Fails if a new transform is added without a test."""
    missing = ALL_OPS - _TESTED_OPS
    assert not missing, f"transform ops with no test: {missing}"
