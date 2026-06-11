"""Deterministic reshape resolver tests (capability-registry A+B).

Proves that, from each output port's declared SHAPE, the system derives the
correct extraction path to a target dtype — or correctly REJECTS the impossible
ones (the recurring Vortex failures: a sequence cannot be pulled from a PDB
structure map/list). No LLM, no guessing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genesis_workbench.node_catalog import Port, PortType, reshape_path  # noqa: E402


def _ok(src, dst, expected_path):
    path, reason = reshape_path(src, dst)
    assert path == expected_path, f"{src.name}->{dst}: got {path!r} ({reason})"


def _reject(src, dst):
    path, reason = reshape_path(src, dst)
    assert path is None, f"{src.name}->{dst}: expected reject, got path {path!r}"
    assert reason, "reject must carry a reason"


# ── scalar ────────────────────────────────────────────────────────────────────
def test_scalar_direct():
    _ok(Port("pdb", PortType.PDB), "pdb", "")            # esmfold.pdb -> a PDB input
    _reject(Port("pdb", PortType.PDB), "sequence")        # a PDB is not a sequence


# ── list ────────────────────────────────────────────────────────────────────
def test_list_first_element():
    seqs = Port("sequences", PortType.SEQUENCES, shape="list", item="sequence")
    _ok(seqs, "sequence", "0")          # protein_design.sequences -> deepstabp.sequence
    _ok(seqs, "sequences", "0")         # plural target still indexes (sequence~sequences)


def test_list_of_structures_rejects_sequence():
    designs = Port("designs", PortType.JSON, shape="list", item="pdb")
    _reject(designs, "sequence")        # designs are STRUCTURES, no sequence
    _ok(designs, "pdb", "0")            # but the first structure IS a PDB


# ── list_obj ────────────────────────────────────────────────────────────────
def test_list_obj_field_by_dtype():
    top = Port("top_k", PortType.JSON, shape="list_obj",
               fields={"smiles": "smiles", "qed": "score", "reward": "score"})
    _ok(top, "smiles", "0.smiles")      # molecule_optimization.top_k -> admet.smiles
    _ok(top, "score", "0.qed")          # first score-typed field
    _reject(top, "sequence")            # no sequence field


# ── map (the enzyme.candidates case) ────────────────────────────────────────
def test_map_first_value_and_reject():
    cand = Port("candidates", PortType.JSON, shape="map", item="pdb")
    _reject(cand, "sequence")           # THE recurring bug: sequence from a PDB map -> impossible
    _ok(cand, "pdb", "*")               # first structure value of the map


# ── against the real built-in catalog ───────────────────────────────────────
def test_real_catalog_ports():
    from genesis_workbench.builtin_nodes import CURATED_BY_TYPE

    def out(node_type, port):
        return next(p for p in CURATED_BY_TYPE[node_type].outputs if p.name == port)

    # enzyme candidates -> ESMFold/NetSolP/DeepSTABp sequence: deterministically rejected
    _reject(out("enzyme_optimization", "candidates"), "sequence")
    # protein_design.sequences -> a sequence consumer: resolves to first element
    _ok(out("protein_design", "sequences"), "sequence", "0")
    # molecule_optimization.top_k -> admet smiles
    _ok(out("molecule_optimization", "top_k"), "smiles", "0.smiles")
