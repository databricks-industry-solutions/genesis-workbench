"""Orchestrator utilities for the Vaccine Immunogen Design loop.

A vaccine teaches the immune system to recognize a small patch on a pathogen —
the **epitope**. That patch alone is usually floppy and hard to manufacture. This
workflow uses **RFD4-Proteina** to design a brand-new, stable scaffold protein
that holds the epitope locked in its native 3D conformation ("keep the gem fixed,
design the ring around it") — i.e. **motif scaffolding**.

Four concerns live here (mirrors antibody_design/utils.py; the generation
condition_spec and the headline reward axis differ):

1. RFD4-Proteina in-process generator — loads RFD4 once (the nb01
   `load_ckpt_n_configure_inference` path) and generates scaffold candidates that
   graft the epitope motif via a **motif-scaffolding condition_spec**: a single
   chain whose motif residues are held FIXED in coordinates (`C_CRD`) and sequence
   (`C_SEQ`) while the flanks are generated. (Contrast antibody_design, which
   generates a separate binder chain against an antigen with `C_HOT` hotspots.)

2. Epitope-presentation fidelity — `motif_backbone_rmsd_located` folds the
   designed sequence (ESMFold, upstream) and measures backbone RMSD of the folded
   motif region against the input epitope (self-consistency; lower is better). The
   motif is located in the designed sequence by subsequence match (the output is
   renumbered 1..L, so residue-id matching from the antibody loop can't be reused).

3. Endpoint helpers — thin wrappers around the serving endpoints the loop uses
   (ESMFold, ProteinMPNN, NetSolP, DeepSTABp, MHCflurry, HLAIIPred). Reused verbatim
   from the antibody_design utils so the orchestrator and app resolve the same
   endpoints regardless of `dev_user_prefix`.

4. Reward composer + resampling strategy — identical machinery to
   antibody_design/enzyme_optimization (z-score→min-max per axis within a batch,
   weighted sum).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Endpoint name resolution + long-timeout client (reused from antibody utils)
# ---------------------------------------------------------------------------

_AXIS_TO_UC_NAME = {
    "esmfold":     "esmfold",
    "proteinmpnn": "proteinmpnn",
    "netsolp":     "netsolp_v1",
    "deepstabp":   "deepstabp_v1",
    "mhcflurry":   "mhcflurry_v2",
    "hlaiipred":   "hlaiipred_v1",
}

# Every endpoint has scale_to_zero=true; a cold start can take 5-20 min. 1200s
# gives every call a 20-minute ceiling; warm calls still return in 1-2 min.
_DEFAULT_TIMEOUT_SECONDS = 1200
_long_client = WorkspaceClient(config=Config(http_timeout_seconds=_DEFAULT_TIMEOUT_SECONDS))


def endpoint_name(axis: str, dev_user_prefix: Optional[str] = None) -> str:
    uc_name = _AXIS_TO_UC_NAME[axis]
    prefix = dev_user_prefix or os.environ.get("DEV_USER_PREFIX")
    if prefix and prefix.strip().lower() not in ("", "none"):
        return f"gwb_{prefix}_{uc_name}_endpoint"
    return f"gwb_{uc_name}_endpoint"


def _query(axis: str, inputs: Any, dev_user_prefix: Optional[str] = None,
           timeout_seconds: Optional[int] = None) -> Any:
    if timeout_seconds and timeout_seconds != _DEFAULT_TIMEOUT_SECONDS:
        client = WorkspaceClient(config=Config(http_timeout_seconds=timeout_seconds))
    else:
        client = _long_client
    name = endpoint_name(axis, dev_user_prefix=dev_user_prefix)
    return client.serving_endpoints.query(name=name, inputs=inputs)


# ---------------------------------------------------------------------------
# Validation / scoring endpoint helpers (reused from antibody utils)
# ---------------------------------------------------------------------------

def _extract_mean_plddt_from_pdb(pdb_str: str) -> float:
    """ESMFold writes pLDDT into the B-factor column. Mean of CA B-factors."""
    plddts = []
    for line in pdb_str.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                plddts.append(float(line[60:66]))
            except ValueError:
                continue
    return float(np.mean(plddts)) if plddts else 0.0


def call_esmfold(sequence: str, dev_user_prefix: Optional[str] = None) -> Dict[str, Any]:
    resp = _query("esmfold", [sequence], dev_user_prefix=dev_user_prefix)
    out = resp.predictions[0]
    if isinstance(out, dict):
        return {"pdb": out.get("pdb", ""),
                "mean_plddt": float(out.get("mean_plddt", out.get("plddt", 0.0)))}
    return {"pdb": str(out), "mean_plddt": _extract_mean_plddt_from_pdb(str(out))}


def call_proteinmpnn(pdb_str: str,
                     fixed_positions: Optional[Dict[str, List[int]]] = None,
                     dev_user_prefix: Optional[str] = None) -> List[str]:
    """Redesign a backbone's sequence. `fixed_positions` ({chain: [res_nums]})
    keeps those positions' identities — here we FIX the grafted epitope motif and
    redesign the surrounding scaffold (so the presented epitope sequence is never
    touched, only the carrier that holds it)."""
    fp_str = json.dumps(fixed_positions) if fixed_positions else ""
    payload: Any = [{"pdb": pdb_str, "fixed_positions": fp_str}]
    resp = _query("proteinmpnn", payload, dev_user_prefix=dev_user_prefix)
    return [str(s) for s in resp.predictions]


def call_netsolp(sequences: List[str], dev_user_prefix: Optional[str] = None) -> List[float]:
    resp = _query("netsolp", [{"sequence": s} for s in sequences], dev_user_prefix=dev_user_prefix)
    return [float(v) for v in pd.DataFrame(resp.predictions)["predicted_solubility"]]


def call_deepstabp(sequences: List[str], growth_temp: float = 37.0, mt_mode: str = "Cell",
                   dev_user_prefix: Optional[str] = None) -> List[float]:
    payload = [{"sequence": s, "growth_temp": float(growth_temp), "mt_mode": str(mt_mode)} for s in sequences]
    resp = _query("deepstabp", payload, dev_user_prefix=dev_user_prefix)
    return [float(v) for v in pd.DataFrame(resp.predictions)["predicted_tm_celsius"]]


_DEFAULT_MHC_ALLELES = (
    "HLA-A*02:01,HLA-A*01:01,HLA-B*07:02,HLA-B*44:02,HLA-C*07:01,HLA-C*04:01"
)


def call_mhcflurry(sequences: List[str], alleles: Optional[str] = None,
                   dev_user_prefix: Optional[str] = None) -> List[float]:
    a = alleles or _DEFAULT_MHC_ALLELES
    payload = [{"sequence": s, "alleles": a} for s in sequences]
    resp = _query("mhcflurry", payload, dev_user_prefix=dev_user_prefix)
    return [float(v) for v in pd.DataFrame(resp.predictions)["predicted_immuno_burden"]]


# MHC class II (CD4) de-immunization panel — 8 common DRB1 alleles. For a vaccine
# immunogen this is an OPTIONAL "scaffold self-reactivity" signal: minimizing it
# trims unwanted T-helper epitopes in the *carrier* while the B-cell epitope stays
# displayed. Off by default (weight 0) — a vaccine is meant to be immunogenic.
_DEFAULT_MHC2_ALLELES = (
    "DRB1*01:01,DRB1*03:01,DRB1*04:01,DRB1*07:01,"
    "DRB1*08:01,DRB1*11:01,DRB1*13:01,DRB1*15:01"
)


def call_hlaiipred(sequences: List[str], alleles: Optional[str] = None,
                   dev_user_prefix: Optional[str] = None) -> List[float]:
    a = alleles or _DEFAULT_MHC2_ALLELES
    payload = [{"sequence": s, "alleles": a} for s in sequences]
    resp = _query("hlaiipred", payload, dev_user_prefix=dev_user_prefix)
    return [float(v) for v in pd.DataFrame(resp.predictions)["predicted_immuno_burden"]]


def warmup_developability_endpoints(dev_user_prefix: Optional[str] = None,
                                    sample_seq: str = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ") -> Dict[str, str]:
    """One dummy call per developability endpoint so the first scoring round
    doesn't eat a cold start. Per-axis failures are reported, not raised."""
    results: Dict[str, str] = {}
    for axis_name, fn in (
        ("netsolp",   lambda: call_netsolp([sample_seq], dev_user_prefix=dev_user_prefix)),
        ("deepstabp", lambda: call_deepstabp([sample_seq], dev_user_prefix=dev_user_prefix)),
        ("mhcflurry", lambda: call_mhcflurry([sample_seq], dev_user_prefix=dev_user_prefix)),
        ("hlaiipred", lambda: call_hlaiipred([sample_seq], dev_user_prefix=dev_user_prefix)),
    ):
        try:
            fn()
            results[axis_name] = "ok"
        except Exception as e:  # noqa: BLE001
            results[axis_name] = f"FAILED: {type(e).__name__}: {str(e)[:120]}"
        print(f"[warmup] {axis_name}: {results[axis_name]}")
    return results


# ---------------------------------------------------------------------------
# Reward composer + strategy (reused from antibody utils)
# ---------------------------------------------------------------------------

@dataclass
class PredictorAxis:
    name: str
    weight: float
    lower_is_better: bool = False
    pre_normalized: bool = False

    @property
    def enabled(self) -> bool:
        return self.weight > 0


def _zscore_then_minmax(values: List[float], lower_is_better: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    nan_mask = np.isnan(arr)
    if nan_mask.all():
        return np.zeros_like(arr)
    if nan_mask.any():
        arr = arr.copy()
        finite = arr[~nan_mask]
        worst = float(finite.max()) if lower_is_better else float(finite.min())
        arr[nan_mask] = worst
    if lower_is_better:
        arr = -arr
    if arr.std() < 1e-12:
        return np.zeros_like(arr)
    z = (arr - arr.mean()) / arr.std()
    rng = z.max() - z.min()
    if rng < 1e-12:
        return np.zeros_like(arr)
    return (z - z.min()) / rng


def compose_rewards(per_axis_scores: Dict[str, List[float]],
                    axes: List[PredictorAxis]) -> List[float]:
    enabled = [a for a in axes if a.enabled and a.name in per_axis_scores]
    if not enabled:
        return [0.0] * len(next(iter(per_axis_scores.values())))
    K = len(per_axis_scores[enabled[0].name])
    composite = np.zeros(K)
    total_weight = 0.0
    for axis in enabled:
        scores = per_axis_scores[axis.name]
        norm = np.asarray(scores, dtype=float) if axis.pre_normalized \
            else _zscore_then_minmax(scores, axis.lower_is_better)
        composite += axis.weight * norm
        total_weight += axis.weight
    return (composite / total_weight).tolist() if total_weight > 0 else composite.tolist()


# ---------------------------------------------------------------------------
# Sequence-liability scan (rule-based "chemical inertness" — no model/endpoint)
#
# Flags the standard protein developability liabilities straight from the sequence
# (Therapeutic-Antibody-Profiler style) — relevant for a *manufacturable* vaccine
# antigen. `liability_weighted_count` is the reward-loop axis value (lower = more
# inert/developable); `liability_detail` is a human-readable breakdown for the
# result dialog.
# ---------------------------------------------------------------------------

_LIABILITY_RULES = [
    # (name, regex, weight)
    ("deamidation",   r"N[GS]",      1.0),   # NG/NS Asn deamidation hot-spots
    ("isomerization", r"D[GSTH]",    1.0),   # Asp isomerization
    ("fragmentation", r"DP",         1.0),   # Asp-Pro acid-labile cleavage
    ("nglyc_sequon",  r"N[^P][ST]",  2.0),   # N-linked glycosylation sequon (N-X-S/T, X != P)
    ("oxidation",     r"[MW]",       0.3),   # Met/Trp oxidation (surface-dependent -> low weight)
]
_LIABILITY_WEIGHTS = {name: w for name, _pat, w in _LIABILITY_RULES}
_LIABILITY_WEIGHTS["unpaired_cys"] = 2.0


def liability_scan(sequence: str) -> Dict[str, int]:
    """Per-liability hit counts for a sequence (non-overlapping regex matches)."""
    seq = (sequence or "").upper()
    out: Dict[str, int] = {name: len(re.findall(pat, seq)) for name, pat, _w in _LIABILITY_RULES}
    # An odd cysteine count implies at least one free (non-disulfide) Cys — a real liability.
    out["unpaired_cys"] = 1 if (seq.count("C") % 2 == 1) else 0
    return out


def liability_weighted_count(sequence: str) -> float:
    """Weighted sum of sequence liabilities (lower = more inert/developable) — the axis value."""
    return float(sum(_LIABILITY_WEIGHTS.get(k, 1.0) * v for k, v in liability_scan(sequence).items()))


def liability_detail(sequence: str) -> str:
    """Human-readable breakdown for the dialog, e.g. 'deamidation:2, nglyc_sequon:1' (or 'none')."""
    hits = [f"{k}:{v}" for k, v in liability_scan(sequence).items() if v]
    return ", ".join(hits) if hits else "none"


class Strategy:
    name: str = "abstract"

    def propose(self, parents: List[Dict[str, Any]], rewards: List[float],
                length_min: int, length_max: int,
                num_samples_next: int) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class ResampleStrategy(Strategy):
    """Softmax-resample toward high-reward candidates, re-generate at the same
    scaffold-length window."""
    name = "resample"

    def __init__(self, temperature: float = 0.1):
        self.temperature = float(temperature)

    def propose(self, parents, rewards, length_min, length_max, num_samples_next):
        return {"length_min": length_min, "length_max": length_max,
                "num_samples": num_samples_next}


class NoOpStrategy(Strategy):
    name = "noop"

    def propose(self, parents, rewards, length_min, length_max, num_samples_next):
        return None


def make_strategy(name: str, **kwargs: Any) -> Strategy:
    if name == "resample":
        return ResampleStrategy(**{k: v for k, v in kwargs.items() if k in ("temperature",)})
    if name == "noop":
        return NoOpStrategy()
    raise ValueError(f"Unknown strategy '{name}'. Known: resample, noop.")


# ---------------------------------------------------------------------------
# Epitope-presentation fidelity — motif backbone RMSD, located by sequence
#
# The designed scaffold is renumbered 1..L by RFD4/ESMFold, so the antibody loop's
# residue-id matching can't be reused. Instead we locate the fixed epitope motif in
# the designed sequence by exact subsequence match (C_SEQ keeps it verbatim; the
# ProteinMPNN redesign fixes it too), then superimpose the folded motif backbone
# against the input epitope. Lower RMSD = the scaffold presents the epitope in its
# native conformation = a better immunogen.
# ---------------------------------------------------------------------------

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def _chain_res_range(pdb_str: str, chain: str) -> Optional[Tuple[int, int]]:
    """Min/max residue id for `chain` from the motif PDB ATOM/HETATM records (the
    author numbering the CIF preserves). Returns (lo, hi), or None if empty."""
    ids = []
    for line in pdb_str.splitlines():
        if line[:6].strip() in ("ATOM", "HETATM") and len(line) >= 26 and line[21] == chain:
            try:
                ids.append(int(line[22:26]))
            except ValueError:
                pass
    return (min(ids), max(ids)) if ids else None


def motif_sequence(pdb_str: str, chain: str, lo: int, hi: int) -> str:
    """One-letter sequence of the motif residues [lo, hi] on `chain`, ordered by
    residue id (CA records only, so each residue is counted once)."""
    seen: Dict[int, str] = {}
    for line in pdb_str.splitlines():
        if not line.startswith("ATOM") or len(line) < 26:
            continue
        if line[21] != chain or line[12:16].strip() != "CA":
            continue
        try:
            resid = int(line[22:26])
        except ValueError:
            continue
        if lo <= resid <= hi:
            seen.setdefault(resid, _THREE_TO_ONE.get(line[17:20].strip(), "X"))
    return "".join(seen[r] for r in sorted(seen))


def locate_motif_in_sequence(designed_seq: str, motif_seq: str) -> List[int]:
    """1-based positions of the motif within the designed sequence (exact match).
    Returns [] when the motif isn't found verbatim (e.g. an unfixed redesign)."""
    if not designed_seq or not motif_seq:
        return []
    idx = designed_seq.find(motif_seq)
    if idx < 0:
        return []
    return list(range(idx + 1, idx + len(motif_seq) + 1))


def motif_backbone_rmsd_located(input_pdb: str, input_chain: str, lo: int, hi: int,
                                designed_pdb: str, designed_positions: List[int]) -> float:
    """Backbone (N, CA, C) RMSD between the input epitope motif (residues [lo, hi]
    of `input_chain`) and the folded designed structure's motif region.

    Uses **biotite** (not Bio.PDB) — biotite is a core rfproteina dependency that is
    always present in the orchestrator kernel, whereas biopython is not. The design is
    folded by ESMFold, which renumbers its single chain 1..L, so the 1-based
    `designed_positions` ARE the folded structure's residue ids. Input motif residues
    are paired with the designed positions in sorted order (matches `motif_sequence`).

    Returns NaN for empty/missing input or a length mismatch — the reward composer
    treats NaN as "skip this axis" instead of crashing the iteration."""
    if not input_pdb or not designed_pdb or not designed_positions:
        return float("nan")
    import io
    import numpy as _np
    import biotite.structure as struc
    import biotite.structure.io.pdb as _pdb

    try:
        in_arr = _pdb.PDBFile.read(io.StringIO(input_pdb)).get_structure(model=1)
        de_arr = _pdb.PDBFile.read(io.StringIO(designed_pdb)).get_structure(model=1)
    except Exception:
        return float("nan")

    # Input motif residue ids (CA present, in [lo, hi]), sorted — matches motif_sequence().
    ca_mask = ((in_arr.chain_id == input_chain) & (in_arr.atom_name == "CA")
               & (in_arr.res_id >= lo) & (in_arr.res_id <= hi))
    in_resids = sorted({int(r) for r in in_arr.res_id[ca_mask]})
    if len(in_resids) != len(designed_positions):
        return float("nan")

    de_chains = {str(c) for c in _np.unique(de_arr.chain_id)}
    de_chain = "A" if "A" in de_chains else sorted(de_chains)[0]

    def _one(arr, chain_id, rid, name):
        sel = arr[(arr.chain_id == chain_id) & (arr.res_id == rid) & (arr.atom_name == name)]
        return sel[0] if sel.array_length() == 1 else None

    in_atoms, de_atoms = [], []
    for in_rid, de_rid in zip(in_resids, designed_positions):
        for name in ("N", "CA", "C"):
            a = _one(in_arr, input_chain, in_rid, name)
            b = _one(de_arr, de_chain, int(de_rid), name)
            if a is None or b is None:
                continue  # skip atoms missing on either side (keep the pairing aligned)
            in_atoms.append(a)
            de_atoms.append(b)
    if len(in_atoms) < 3:
        return float("nan")
    try:
        in_bb = struc.array(in_atoms)
        de_bb = struc.array(de_atoms)
        fitted, _ = struc.superimpose(in_bb, de_bb)
        return float(struc.rmsd(in_bb, fitted))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# RFD4-Proteina in-process motif-scaffolding generator
#
# Loads RFD4 once (the nb01 load_ckpt_n_configure_inference path) and generates
# scaffolds that present the epitope motif. Imports are deferred into the functions
# so this module imports cleanly on a kernel that hasn't yet installed rfproteina
# (e.g. during py_compile / unit import).
# ---------------------------------------------------------------------------

def load_rfd4(flow_ckpt: str, ae_ckpt: str) -> Dict[str, Any]:
    """Load RFD4-Proteina for warm in-process generation. Returns a context
    dict {model, transform, pkg_dir, device, torch}. Mirrors nb01's
    RFD4ProteinaModel.load_context."""
    import os as _os
    import torch
    import rfproteina
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from rfproteina.generate import load_ckpt_n_configure_inference
    from rfproteina.datapipes.pipeline import build_design_validation_pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    pkg_dir = _os.path.dirname(rfproteina.__file__)
    monomer_cif = _os.path.join(pkg_dir, "data", "benchmarks", "monomer", "monomer-short.cif")
    overrides = [
        "inference_experiment=inference_on_user_inputs",
        "~metrics/refolding_oracles", "metrics_tags=[]", "num_workers=0", "seed=42",
        f"inputs={monomer_cif}",
        f"ckpt_path={_os.path.dirname(flow_ckpt)}", f"ckpt_name={_os.path.basename(flow_ckpt)}",
        f"autoencoder_ckpt_path={ae_ckpt}", "out_dir=/tmp/rfd4_immunogen_out",
    ]
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=_os.path.join(pkg_dir, "configs"), version_base="1.3"):
        cfg = compose(config_name="inference", overrides=overrides)
    model = load_ckpt_n_configure_inference(cfg)
    model.to(device).eval()
    transform = build_design_validation_pipeline(metrics_tags=None)
    print(f"[rfd4] loaded on {device}")
    return {"model": model, "transform": transform, "pkg_dir": pkg_dir,
            "device": device, "torch": torch}


def _flank_segments(length_min: int, length_max: int, motif_len: int) -> Tuple[str, str]:
    """Split the scaffold-length window into two symmetric flexible flanks that
    sandwich the motif, honoring the total-length window [length_min, length_max].

    FIRST-DRAFT placement: the epitope is centered with two EQUAL symmetric flanks,
    so the total length (2*flank + motif) stays within [length_min, length_max]
    (independent per-segment ranges would widen the sum). Returns (`pre`, `post`)
    contig length-range tokens like "20-30". A terminal placement (0-length flank)
    or a discontinuous (multi-segment) motif is a future refinement.
    """
    flank_lo = max(0, int(length_min) - int(motif_len))
    flank_hi = max(flank_lo, int(length_max) - int(motif_len))
    half_lo = max(1, flank_lo // 2)
    half_hi = max(half_lo, (flank_hi + 1) // 2)
    seg = f"{half_lo}-{half_hi}"
    return seg, seg


def _motif_scaffold_condition_spec(motif_cif_path: str, motif_chain: str,
                                   motif_range: Tuple[int, int],
                                   length_min: int, length_max: int, tmpdir: str) -> str:
    """Build the RFD4 condition_spec JSON for epitope motif-scaffolding and return
    its path.

    The designed chain is ONE continuous chain: a flexible flank, the preserved
    epitope motif segment, and another flexible flank. The motif residues are held
    FIXED in both coordinates (`C_CRD`) and sequence (`C_SEQ`); the flanks are
    generated de novo. No chain break (`/0`) and no hotspots (`C_HOT`) — those are
    the binder/antibody path. Syntax follows the proven recipe (see
    [[gwb-rfd4-proteina-inference-recipe]]): a preserved contig segment is
    `<chain><start>-<end>`, and conditions use atomworks `res_id` selects.
    """
    import json as _json
    lo, hi = motif_range
    pre, post = _flank_segments(length_min, length_max, hi - lo + 1)
    contig = f"{pre}, {motif_chain}{lo}-{hi}, {post}"
    motif_sel = f"res_id>={lo} and res_id<={hi}"
    conditions: Dict[str, Any] = {
        "C_CRD": {True: [{"select": motif_sel}]},  # fix the epitope backbone coordinates
        "C_SEQ": {True: [{"select": motif_sel}]},  # fix the epitope sequence
    }
    spec = {"immunogen": {"input": motif_cif_path, "contig": contig, "conditions": conditions}}
    spec_path = os.path.join(tmpdir, "immunogen_spec.json")
    with open(spec_path, "w") as f:
        f.write(_json.dumps(spec))
    return spec_path


def _structure_to_pdb_and_seq(sample: Dict[str, Any]) -> Tuple[str, str]:
    """biotite AtomArray → (pdb_string, sequence). Mirrors nb01."""
    import io
    import numpy as _np
    aa = sample["generated_atom_array"]
    seq = ""
    try:
        from biotite.structure import to_sequence
        seqs, _ = to_sequence(aa)
        seq = "".join(str(s) for s in seqs)
    except Exception:
        pass
    cats, n = set(aa.get_annotation_categories()), aa.array_length()
    if "b_factor" in cats:
        aa.set_annotation("b_factor", _np.zeros(n))
    if "occupancy" in cats:
        aa.set_annotation("occupancy", _np.ones(n))
    if "charge" in cats:
        aa.set_annotation("charge", _np.zeros(n, dtype=int))
    try:
        from biotite.structure.io.pdb import PDBFile
        buf = io.StringIO(); pf = PDBFile(); pf.set_structure(aa); pf.write(buf)
        return buf.getvalue(), seq
    except Exception:
        from biotite.structure.io.pdbx import CIFFile, set_structure
        cf = CIFFile(); set_structure(cf, aa); buf = io.StringIO(); cf.write(buf)
        return buf.getvalue(), seq


def _pdb_to_cif(pdb_path: str, cif_path: str) -> None:
    """RFD4's condition_spec `input` must be CIF/BinaryCIF — it rejects PDB. Convert
    the motif PDB → CIF. Prefer atomworks (the model's own parser, preserves
    bonds/charges); fall back to a plain biotite atom_site CIF if the atomworks
    writer API differs across versions (fine for a standard protein motif)."""
    try:
        from atomworks.io.parser import parse as _aw_parse
        try:
            from atomworks.io.writer import to_cif_file as _to_cif
        except Exception:
            from atomworks.io import to_cif_file as _to_cif  # writer location varies across atomworks versions
        _to_cif(_aw_parse(pdb_path), cif_path)
        return
    except Exception as e:  # noqa: BLE001
        print(f"[motif->cif] atomworks conversion unavailable ({type(e).__name__}: {str(e)[:140]}); using biotite")
    import biotite.structure.io.pdb as _pdb
    import biotite.structure.io.pdbx as _pdbx
    arr = _pdb.PDBFile.read(pdb_path).get_structure(model=1)
    cif = _pdbx.CIFFile()
    _pdbx.set_structure(cif, arr)
    cif.write(cif_path)


def generate_scaffold(ctx: Dict[str, Any], motif_pdb_str: str, motif_chain: str,
                      motif_range: Tuple[int, int], length_min: int, length_max: int,
                      num_samples: int) -> pd.DataFrame:
    """Generate `num_samples` scaffold candidates presenting the epitope motif.
    Returns a DataFrame with columns `designed_pdb`, `designed_sequence`,
    `sample_id` — the same contract the loop expects."""
    from rfproteina.generate import get_contig_or_design_problem_dataset
    from rfproteina.datapipes.collate import collate_batch
    model, transform, torch = ctx["model"], ctx["transform"], ctx["torch"]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        motif_pdb = os.path.join(tmp, "motif.pdb")
        with open(motif_pdb, "w") as f:
            f.write(motif_pdb_str)
        # RFD4's condition_spec `input` must be CIF/BinaryCIF, not PDB — convert first.
        motif_cif = os.path.join(tmp, "motif.cif")
        _pdb_to_cif(motif_pdb, motif_cif)
        spec_path = _motif_scaffold_condition_spec(motif_cif, motif_chain, motif_range,
                                                   length_min, length_max, tmp)
        ds = get_contig_or_design_problem_dataset(spec_path, num_replicates=int(num_samples),
                                                  transform=transform)
        batch = collate_batch([ds[i] for i in range(len(ds))],
                              schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample",
                              fixed_dimensions=None)
        batch = model.transfer_batch_to_device(batch, torch.device(ctx["device"]), 0)
        with torch.no_grad():
            outs = model.predict_step(batch, 0)
    for i, out in enumerate(outs):
        pdb_str, seq = _structure_to_pdb_and_seq(out)
        rows.append({"sample_id": i, "designed_pdb": pdb_str, "designed_sequence": seq})
    return pd.DataFrame(rows)
