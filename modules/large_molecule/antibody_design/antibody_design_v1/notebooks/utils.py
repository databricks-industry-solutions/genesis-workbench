"""Orchestrator utilities for the Antibody (VHH) Design loop.

Four concerns live here:

1. RFD4-Proteina in-process generator — loads RFD4 once (the nb01
   `load_ckpt_n_configure_inference` path) and generates VHH candidates against
   an antigen epitope via a condition_spec (binder-style conditioning on the
   epitope hotspots). **The VHH condition_spec is FIRST-DRAFT** — true
   framework/CDR scaffolding (keep an Ig framework, design only the CDR loops)
   is the deploy-time refinement; see `generate_vhh`.

2. anarcii numbering — annotate a designed sequence as a VHH and extract its
   CDR residue positions (for targeted ProteinMPNN redesign). Best-effort: if
   anarcii isn't importable or numbering fails, we degrade gracefully.

3. Endpoint helpers — thin wrappers around the serving endpoints the loop uses
   for validation + scoring (ESMFold, ProteinMPNN, Boltz, NetSolP, PLTNUM,
   DeepSTABp, MHCflurry). Reused verbatim from the enzyme_optimization utils so
   the orchestrator and app resolve the same endpoints regardless of
   `dev_user_prefix`.

4. Reward composer + resampling strategy — identical machinery to
   enzyme_optimization (z-score→min-max per axis within a batch, weighted sum,
   half-life anchor sigmoid).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Endpoint name resolution + long-timeout client (reused from enzyme utils)
# ---------------------------------------------------------------------------

_AXIS_TO_UC_NAME = {
    "esmfold":     "esmfold",
    "proteinmpnn": "proteinmpnn",
    "boltz":       "boltz",
    "netsolp":     "netsolp_v1",
    "pltnum":      "pltnum_v1",
    "deepstabp":   "deepstabp_v1",
    "mhcflurry":   "mhcflurry_v2",
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
# Validation / scoring endpoint helpers (reused from enzyme utils)
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
    keeps those positions' identities — here we FIX the framework and redesign
    the CDR loops (the inverse of the enzyme loop, which fixed the motif)."""
    fp_str = json.dumps(fixed_positions) if fixed_positions else ""
    payload: Any = [{"pdb": pdb_str, "fixed_positions": fp_str}]
    resp = _query("proteinmpnn", payload, dev_user_prefix=dev_user_prefix)
    return [str(s) for s in resp.predictions]


def call_boltz(boltz_input: str, dev_user_prefix: Optional[str] = None,
               timeout_seconds: int = 900) -> Dict[str, Any]:
    """boltz_input e.g. 'protein_A:<antigen>;protein_B:<vhh>' for a complex.
    Returns dict: { pdb, ipTM?, iLDDT?, ... }."""
    payload = [{"input": boltz_input, "msa": "no_msa", "use_msa_server": "True"}]
    resp = _query("boltz", payload, dev_user_prefix=dev_user_prefix, timeout_seconds=timeout_seconds)
    pred = resp.predictions[0] if resp.predictions else {}
    return {"pdb": pred} if isinstance(pred, str) else pred


def call_netsolp(sequences: List[str], dev_user_prefix: Optional[str] = None) -> List[float]:
    resp = _query("netsolp", [{"sequence": s} for s in sequences], dev_user_prefix=dev_user_prefix)
    return [float(v) for v in pd.DataFrame(resp.predictions)["predicted_solubility"]]


def call_pltnum(sequences: List[str], dev_user_prefix: Optional[str] = None) -> List[float]:
    resp = _query("pltnum", [{"sequence": s} for s in sequences], dev_user_prefix=dev_user_prefix)
    return [float(v) for v in pd.DataFrame(resp.predictions)["predicted_stability"]]


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


def warmup_developability_endpoints(dev_user_prefix: Optional[str] = None,
                                    sample_seq: str = "QVQLVESGGGLVQAGGSLRLSCAASG") -> Dict[str, str]:
    """One dummy call per developability endpoint so the first scoring round
    doesn't eat a cold start. Per-axis failures are reported, not raised."""
    results: Dict[str, str] = {}
    for axis_name, fn in (
        ("netsolp",   lambda: call_netsolp([sample_seq], dev_user_prefix=dev_user_prefix)),
        ("pltnum",    lambda: call_pltnum([sample_seq], dev_user_prefix=dev_user_prefix)),
        ("deepstabp", lambda: call_deepstabp([sample_seq], dev_user_prefix=dev_user_prefix)),
        ("mhcflurry", lambda: call_mhcflurry([sample_seq], dev_user_prefix=dev_user_prefix)),
    ):
        try:
            fn()
            results[axis_name] = "ok"
        except Exception as e:  # noqa: BLE001
            results[axis_name] = f"FAILED: {type(e).__name__}: {str(e)[:120]}"
        print(f"[warmup] {axis_name}: {results[axis_name]}")
    return results


# ---------------------------------------------------------------------------
# Reward composer + strategy (reused from enzyme utils)
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


def half_life_anchor_threshold(reference_pltnum_scores: List[float], margin: float = 0.05) -> float:
    if not reference_pltnum_scores:
        return -math.inf
    return float(min(reference_pltnum_scores)) + float(margin)


def half_life_anchor_rewards(candidate_pltnum_scores: List[float],
                             threshold: float, beta: float = 0.05) -> List[float]:
    return [1.0 / (1.0 + math.exp(-((s - threshold) / max(beta, 1e-6))))
            for s in candidate_pltnum_scores]


# ---------------------------------------------------------------------------
# Sequence-liability scan (rule-based "chemical inertness" — no model/endpoint)
#
# Flags the standard antibody/protein developability liabilities straight from the
# sequence (Therapeutic-Antibody-Profiler style). `liability_weighted_count` is the
# reward-loop axis value (lower = more inert/developable); `liability_detail` is a
# human-readable breakdown for the result dialog. Weights are tunable; oxidation is
# surface-context-dependent so it carries a low weight.
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
    length window. (Length is already tightly constrained to the VHH range.)"""
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
# anarcii — VHH numbering + CDR extraction (best-effort)
# ---------------------------------------------------------------------------

# IMGT CDR residue ranges (1-based IMGT numbering). Used to map anarcii's
# per-residue numbering back to the 1-based positions in the designed sequence.
_IMGT_CDR_RANGES = {"CDR1": (27, 38), "CDR2": (56, 65), "CDR3": (105, 117)}


def number_vhh(sequence: str) -> Dict[str, Any]:
    """Number a designed sequence as a VHH with anarcii and return
    {"is_vhh": bool, "cdr_positions": [1-based ints], "scheme": "imgt"}.

    Best-effort: anarcii ships with the rfproteina install. If it isn't
    importable, or the sequence doesn't number as an antibody V-domain, we
    return is_vhh=False with an empty CDR list and the caller redesigns the
    whole variable region instead of just the CDRs.
    """
    try:
        from anarcii import Anarcii
    except Exception as e:  # noqa: BLE001
        print(f"[anarcii] not importable ({type(e).__name__}: {str(e)[:80]}); skipping CDR numbering")
        return {"is_vhh": False, "cdr_positions": [], "scheme": "imgt"}
    try:
        model = Anarcii(seq_type="antibody", mode="accuracy", scheme="imgt", verbose=False)
        numbered = model.number([("design", sequence)])
        # anarcii returns a per-sequence record; shapes vary across versions, so
        # pull the numbered residue list defensively.
        rec = numbered[0] if isinstance(numbered, (list, tuple)) else numbered
        numbering = rec.get("numbering") if isinstance(rec, dict) else getattr(rec, "numbering", None)
        if not numbering:
            return {"is_vhh": False, "cdr_positions": [], "scheme": "imgt"}
        cdr_positions: List[int] = []
        seq_pos = 0
        for entry in numbering:
            # entry ~ ((imgt_number, insertion_code), amino_acid)
            try:
                (imgt_num, _ins), aa = entry
            except Exception:
                continue
            if aa == "-":
                continue
            seq_pos += 1
            for _cdr, (lo, hi) in _IMGT_CDR_RANGES.items():
                if lo <= int(imgt_num) <= hi:
                    cdr_positions.append(seq_pos)
                    break
        return {"is_vhh": bool(cdr_positions), "cdr_positions": cdr_positions, "scheme": "imgt"}
    except Exception as e:  # noqa: BLE001
        print(f"[anarcii] numbering failed ({type(e).__name__}: {str(e)[:120]}); redesigning whole V-region")
        return {"is_vhh": False, "cdr_positions": [], "scheme": "imgt"}


# ---------------------------------------------------------------------------
# RFD4-Proteina in-process VHH generator
#
# Loads RFD4 once (the nb01 load_ckpt_n_configure_inference path) and generates
# VHH candidates against the antigen epitope. Imports are deferred into the
# functions so this module imports cleanly on a kernel that hasn't yet installed
# rfproteina (e.g. during py_compile / unit import).
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
        f"autoencoder_ckpt_path={ae_ckpt}", "out_dir=/tmp/rfd4_ab_out",
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


def _vhh_condition_spec(antigen_cif_path: str, epitope_residues: List[int],
                        antigen_chain: str, antigen_res_range, length_min: int, length_max: int,
                        tmpdir: str) -> str:
    """Build the RFD4 condition_spec JSON for VHH design against an antigen epitope and return its path.

    FIRST-DRAFT conditioning (same caveat as nb01's binder/motif path): a fresh VHH-length chain is
    generated against the antigen, with the antigen backbone + sequence kept as context (C_CRD/C_SEQ) and
    the epitope residues as hotspots (C_HOT). Does NOT yet scaffold a true Ig/VHH framework — that's the
    intended refinement. Syntax follows the model's own inference examples (rfproteina tests /
    docs/workflow/condition-spec): a preserved contig segment is `<chain><start>-<end>` (NOT just the chain
    letter — that raises "Malformed preserved-segment token"), and conditions use atomworks `res_id` selects
    (NOT `sel(...)`).
    """
    import json as _json
    lo, hi = antigen_res_range if antigen_res_range else (1, 1)
    # contig: fresh VHH chain (length window) + chain break (/0) + the preserved antigen chain, by range.
    contig = f"{int(length_min)}-{int(length_max)}, /0, {antigen_chain}{lo}-{hi}"
    whole_antigen = f"res_id>={lo} and res_id<={hi}"
    conditions: Dict[str, Any] = {
        "C_CRD": {True: [{"select": whole_antigen}]},  # keep the antigen backbone coords as context
        "C_SEQ": {True: [{"select": whole_antigen}]},  # keep the antigen sequence
    }
    if epitope_residues:
        conditions["C_HOT"] = {True: [{"select": f"res_id=={int(r)}"} for r in epitope_residues]}
    spec = {"binder": {"input": antigen_cif_path, "contig": contig, "conditions": conditions}}
    spec_path = os.path.join(tmpdir, "vhh_spec.json")
    with open(spec_path, "w") as f:
        f.write(_json.dumps(spec))
    return spec_path


def _chain_res_range(pdb_str: str, chain: str):
    """Min/max residue id for `chain` from the antigen PDB ATOM/HETATM records (the author numbering the
    CIF preserves). Returns (lo, hi), or None if the chain has no residues."""
    ids = []
    for line in pdb_str.splitlines():
        if line[:6].strip() in ("ATOM", "HETATM") and len(line) >= 26 and line[21] == chain:
            try:
                ids.append(int(line[22:26]))
            except ValueError:
                pass
    return (min(ids), max(ids)) if ids else None


def _structure_to_pdb_and_seq(sample: Dict[str, Any]) -> tuple:
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
    """RFD4's condition_spec `input` must be CIF/BinaryCIF — it rejects PDB ("PDB files don't reliably
    encode the bonds/formal charges the model consumes directly"). Convert the antigen PDB → CIF. Prefer
    atomworks (the model's own parser, as its error recommends; preserves bonds/charges); fall back to a
    plain biotite atom_site CIF if the atomworks writer API differs across versions (fine for a standard
    protein antigen, where bonds are inferred from residue templates)."""
    try:
        from atomworks.io.parser import parse as _aw_parse
        try:
            from atomworks.io.writer import to_cif_file as _to_cif
        except Exception:
            from atomworks.io import to_cif_file as _to_cif  # writer location varies across atomworks versions
        _to_cif(_aw_parse(pdb_path), cif_path)
        return
    except Exception as e:  # noqa: BLE001
        print(f"[antigen->cif] atomworks conversion unavailable ({type(e).__name__}: {str(e)[:140]}); using biotite")
    import biotite.structure.io.pdb as _pdb
    import biotite.structure.io.pdbx as _pdbx
    arr = _pdb.PDBFile.read(pdb_path).get_structure(model=1)
    cif = _pdbx.CIFFile()
    _pdbx.set_structure(cif, arr)
    cif.write(cif_path)


def generate_vhh(ctx: Dict[str, Any], antigen_pdb_str: str, epitope_residues: List[int],
                 antigen_chain: str, length_min: int, length_max: int,
                 num_samples: int) -> pd.DataFrame:
    """Generate `num_samples` VHH candidates against the antigen epitope.
    Returns a DataFrame with columns `designed_pdb`, `designed_sequence`,
    `sample_id` — the same contract the enzyme loop expects from `call_ame`."""
    from rfproteina.generate import get_contig_or_design_problem_dataset
    from rfproteina.datapipes.collate import collate_batch
    model, transform, torch = ctx["model"], ctx["transform"], ctx["torch"]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        antigen_pdb = os.path.join(tmp, "antigen.pdb")
        with open(antigen_pdb, "w") as f:
            f.write(antigen_pdb_str)
        # RFD4's condition_spec `input` must be CIF/BinaryCIF, not PDB — convert the antigen first.
        antigen_cif = os.path.join(tmp, "antigen.cif")
        _pdb_to_cif(antigen_pdb, antigen_cif)
        res_range = _chain_res_range(antigen_pdb_str, antigen_chain)
        spec_path = _vhh_condition_spec(antigen_cif, epitope_residues, antigen_chain,
                                        res_range, length_min, length_max, tmp)
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
