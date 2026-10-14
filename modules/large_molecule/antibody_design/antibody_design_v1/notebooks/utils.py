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
from typing import Any, Dict, List, Optional, Tuple

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
    Returns the dbboltz FLAT result dict for the first diffusion sample:
    { pdb, confidence_score, ptm, iptm, ligand_iptm, protein_iptm } — note the keys
    are lowercase and the confidence values are STRINGS (7-decimal). Use `boltz_iptm`
    to pull the interface pTM."""
    payload = [{"input": boltz_input, "msa": "no_msa", "use_msa_server": "True"}]
    resp = _query("boltz", payload, dev_user_prefix=dev_user_prefix, timeout_seconds=timeout_seconds)
    pred = resp.predictions[0] if resp.predictions else {}
    return {"pdb": pred} if isinstance(pred, str) else pred


def boltz_iptm(out: Dict[str, Any]) -> float:
    """Interface pTM (binding confidence) from a Boltz result. The dbboltz pyfunc emits
    a flat dict with lowercase, STRING-valued keys ('iptm','protein_iptm','ptm',
    'confidence_score','ligand_iptm'); prefer iptm → protein_iptm → confidence_score.
    Returns 0.0 if none parse. (The old `out.get("ipTM", out.get("iLDDT", 0.0))` matched
    NO key — wrong case/name — so the binding axis was silently 0 for every candidate.)"""
    for k in ("iptm", "protein_iptm", "confidence_score", "ptm"):
        v = out.get(k)
        if v not in (None, ""):
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return 0.0


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


# MHC class II (CD4 / anti-drug-antibody) de-immunization panel — 8 common DRB1 alleles.
# This is the *right* immunogenicity signal for antibodies/VHH (ADA risk is MHC-II / CD4-driven),
# complementing MHCflurry's MHC-I (CD8) burden.
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
                                    sample_seq: str = "QVQLVESGGGLVQAGGSLRLSCAASG") -> Dict[str, str]:
    """One dummy call per developability endpoint so the first scoring round
    doesn't eat a cold start. Per-axis failures are reported, not raised."""
    results: Dict[str, str] = {}
    for axis_name, fn in (
        ("netsolp",   lambda: call_netsolp([sample_seq], dev_user_prefix=dev_user_prefix)),
        ("pltnum",    lambda: call_pltnum([sample_seq], dev_user_prefix=dev_user_prefix)),
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


def self_consistency_rmsd(design_pdb: str, refold_pdb: str) -> float:
    """Cα self-consistency RMSD between a designed structure and an independent refold of its
    sequence (RFD4-Proteina's designability signal; lower = the sequence folds back to the design).
    Both are the same chain/length, so Cα correspond by residue order; Kabsch-superposed. Returns
    NaN if either is unparseable/too short."""
    def _ca(pdb: str) -> np.ndarray:
        xs = [(float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
              for ln in pdb.splitlines()
              if ln.startswith("ATOM") and ln[12:16].strip() == "CA"]
        return np.asarray(xs, dtype=float)
    A, B = _ca(design_pdb or ""), _ca(refold_pdb or "")
    n = min(len(A), len(B))
    if n < 4:
        return float("nan")
    A = A[:n] - A[:n].mean(0)
    B = B[:n] - B[:n].mean(0)
    U, _S, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return float(np.sqrt((((R @ A.T).T - B) ** 2).sum(1).mean()))


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

    Best-effort: anarcii is installed explicitly by the orchestrator's install
    cell (it is NOT an rfproteina dependency — do not assume it is present). If it
    isn't importable, or the sequence doesn't number as an antibody V-domain, we
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
# CDR grafting — transplant RFD4-designed CDR loops onto a real VHH framework
#
# RFD4 has no fold/secondary-structure conditioning and C_SEQ can't pin residues on
# freshly-designed backbone (verified against the rfproteina condition_spec), so there is
# no way to MAKE the de-novo chain adopt an Ig fold at generation time. Instead we keep
# RFD4's de-novo design (which reliably finds the right epitope) and graft its CDR loops
# onto a canonical humanized VHH framework, so every deliverable carries a genuine antibody
# framework. The grafted sequence is then re-folded (ESMFold) and re-scored (binding via
# Boltz co-fold), so a graft that loses binding is caught, not hidden.
# ---------------------------------------------------------------------------

# IMGT region boundaries (1-based IMGT numbers), full FR/CDR partition of a V-domain.
_IMGT_REGION_RANGES = [
    ("FR1", 1, 26), ("CDR1", 27, 38), ("FR2", 39, 55), ("CDR2", 56, 65),
    ("FR3", 66, 104), ("CDR3", 105, 117), ("FR4", 118, 128),
]

# Humanized universal VHH acceptor framework (the CDR-graft QC scaffold). FR1/FR2/FR3/FR4 are the
# REAL h-NbBCII10 universal humanized nanobody framework (the scaffold in RFD4's own nanobody
# benchmark — see `_extract_benchmark_framework`), carrying the camelid WFRQ FR2 hallmark that keeps a
# single VH domain soluble without a light chain. The CDR stretches here are placeholders — REPLACED by
# the donor's CDRs at graft time; only the FR residues are used. Overridable via the
# `framework_vhh_sequence` widget. The acceptor is IMGT-numbered at runtime (same scheme as the donor),
# so the exact concatenation boundaries need not be perfect — anarcii re-partitions it. NOTE: the main
# framework-conditioning path no longer uses this sequence — it extracts the real scaffold STRUCTURE
# directly (`default_framework_ctx`); this constant only backs the optional CDR-graft QC step.
CANONICAL_VHH_FRAMEWORK = (
    "QVQLVESGGGLVQPGGSLRLSC"                   # FR1 (real h-NbBCII10)
    "GRTFSSYA"                                 # CDR1 (placeholder)
    "WFRQAPGQGLEAVA"                           # FR2 (real h-NbBCII10, WFRQ hallmark)
    "AISWSGGST"                                # CDR2 (placeholder)
    "YADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYC"    # FR3 (real h-NbBCII10)
    "AADYGSSWYDY"                              # CDR3 (placeholder)
    "WGQGTLVTVS"                               # FR4 (real h-NbBCII10)
)

_ACCEPTOR_REGIONS_CACHE: Dict[str, Optional[Dict[str, str]]] = {}


def _number_regions(sequence: str) -> Optional[Dict[str, str]]:
    """IMGT-number `sequence` with anarcii and partition it into the seven V-domain regions
    {FR1,CDR1,FR2,CDR2,FR3,CDR3,FR4} as amino-acid substrings. Returns None if anarcii is
    unavailable, the sequence doesn't number as a V-domain, or no CDR residues are found.
    Shared by the CDR-graft for both the donor (RFD4 design) and the acceptor framework."""
    try:
        from anarcii import Anarcii
    except Exception as e:  # noqa: BLE001
        print(f"[anarcii] not importable ({type(e).__name__}: {str(e)[:80]}); cannot number for graft")
        return None
    try:
        model = Anarcii(seq_type="antibody", mode="accuracy", scheme="imgt", verbose=False)
        numbered = model.number([("seq", sequence)])
        rec = numbered[0] if isinstance(numbered, (list, tuple)) else numbered
        numbering = rec.get("numbering") if isinstance(rec, dict) else getattr(rec, "numbering", None)
        if not numbering:
            return None
        regions = {name: "" for name, _lo, _hi in _IMGT_REGION_RANGES}
        for entry in numbering:
            try:
                (imgt_num, _ins), aa = entry
            except Exception:  # noqa: BLE001
                continue
            if aa == "-":
                continue
            n = int(imgt_num)
            for name, lo, hi in _IMGT_REGION_RANGES:
                if lo <= n <= hi:
                    regions[name] += aa
                    break
        if not (regions["CDR1"] or regions["CDR2"] or regions["CDR3"]):
            return None
        return regions
    except Exception as e:  # noqa: BLE001
        print(f"[anarcii] region numbering failed ({type(e).__name__}: {str(e)[:120]})")
        return None


def graft_cdrs_onto_framework(donor_sequence: str,
                              acceptor_sequence: str = CANONICAL_VHH_FRAMEWORK) -> Optional[str]:
    """CDR-graft: transplant the donor design's IMGT CDR1/CDR2/CDR3 onto the acceptor VHH
    framework (FR1-FR4), returning a sequence guaranteed to carry a real Ig framework.

    Returns None (a hard VHH gate) when the donor can't be numbered as a V-domain — such a
    design has no cleanly-extractable CDR loops to graft — or when CDR3 (the dominant binding
    loop) is missing, or when the acceptor framework itself doesn't number into FR1-FR4.
    A donor CDR1/CDR2 that anarcii can't resolve falls back to the acceptor's own loop."""
    donor = _number_regions(donor_sequence)
    if donor is None:
        return None
    if acceptor_sequence not in _ACCEPTOR_REGIONS_CACHE:
        _ACCEPTOR_REGIONS_CACHE[acceptor_sequence] = _number_regions(acceptor_sequence)
    acc = _ACCEPTOR_REGIONS_CACHE[acceptor_sequence]
    if acc is None or not all(acc.get(fr) for fr in ("FR1", "FR2", "FR3", "FR4")):
        print("[graft] acceptor framework did not number into FR1-FR4; cannot graft")
        return None
    cdr3 = donor.get("CDR3")
    if not cdr3:
        return None
    cdr1 = donor.get("CDR1") or acc.get("CDR1", "")
    cdr2 = donor.get("CDR2") or acc.get("CDR2", "")
    return acc["FR1"] + cdr1 + acc["FR2"] + cdr2 + acc["FR3"] + cdr3 + acc["FR4"]


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
        # Co-generate the antigen SIDE CHAINS (RFD4-Proteina: fixing target side chains hurts binder
        # success vs letting them adapt); fix only the backbone coords as context. C_SEQ still preserves
        # the antigen sequence, so only the rotamers are free to co-generate at the interface.
        "C_CRD": {True: [{"select": f"({whole_antigen}) and is_protein_backbone"}]},
        "C_SEQ": {True: [{"select": whole_antigen}]},  # keep the antigen sequence
    }
    if epitope_residues:
        conditions["C_HOT"] = {True: [{"select": f"res_id=={int(r)}"} for r in epitope_residues]}
    spec = {"binder": {"input": antigen_cif_path, "contig": contig, "conditions": conditions}}
    spec_path = os.path.join(tmpdir, "vhh_spec.json")
    with open(spec_path, "w") as f:
        f.write(_json.dumps(spec))
    return spec_path


# ---------------------------------------------------------------------------
# Framework-conditioned antibody design (the RFD4-Proteina native "Antibody/TCR
# framework conditioning" task — Table S4 of the paper).
#
# The de-novo binder spec above (_vhh_condition_spec) asks RFD4 for a generic
# epitope binder; it reliably finds the epitope but produces non-antibody folds
# (helical bundles / receptor Ig-like domains), which is why outputs "don't look
# like a regular antibody". The fix is to invoke the model's framework-conditioning
# task: give a REAL VHH framework as a free-placement INTERNAL-coordinate motif
# (C_CRDGRP id ≥1 → the model chooses the framework↔antigen dock itself, no manual
# docking — paper §S6.1.2), pin its sequence (C_SEQ), design ONLY the CDR loops, keep
# the antigen + epitope hotspots, and (optionally) request the immunoglobulin fold
# (C_CATH "2.60.40"). Verified against the rfproteina source (src/rfproteina/
# conditions.py + docs/workflow/condition-spec.md at the pinned release).
# ---------------------------------------------------------------------------

# CDR design windows (min_len, max_len) RFD4 designs between the fixed FRs — nanobody IMGT CDR
# length ranges (CDR3 long/variable); the model picks a length in-range per sample. In the framework
# path the designed VHH length is therefore FR(83) + these windows, NOT the de-novo length widget.
_DEFAULT_CDR_WINDOWS: Dict[str, Tuple[int, int]] = {"CDR1": (5, 8), "CDR2": (3, 8), "CDR3": (7, 18)}

# FR-only fragment sequence of the real h-NbBCII10 universal humanized nanobody scaffold (the
# framework in RFD4's own nanobody benchmark); used only as a post-extraction sanity log.
_H_NBBCII10_FR_SEQ = ("QVQLVESGGGLVQPGGSLRLSC" "WFRQAPGQGLEAVA"
                      "YADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYC" "WGQGTLVTVS")

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _extract_benchmark_framework(nanobody_dir: Optional[str] = None
                                 ) -> Tuple[str, Dict[str, Tuple[int, int]], str]:
    """Extract the REAL h-NbBCII10 VHH framework from RFD4-Proteina's own nanobody-design benchmark,
    which ships inside the installed rfproteina package (pyproject force-includes data/benchmarks →
    rfproteina/data; `load_rfd4` already reads monomer-short.cif the same way).

    Reads a `*h-NbBCII10*cdr123*.cif.zst` condition CIF (every one carries the identical scaffold as
    chain H; the three CDRs are single design placeholders with no coordinates), keeps the framework
    residues (those with a Cα), relabels them to chain 'F' renumbered 1..N, and derives the four FR
    spans from the CDR numbering-gaps. Returns (pdb_str, fr_spans {FR->(lo,hi)}, fr_sequence). NO
    ESMFold and NO anarcii — the coordinates and the FR/CDR partition are known by construction, which
    is why this replaces the hand-built acceptor that failed to fold/number (so framework conditioning
    kept silently falling back to de-novo)."""
    import glob
    import io
    import zstandard
    if nanobody_dir is None:
        import rfproteina
        nanobody_dir = os.path.join(os.path.dirname(rfproteina.__file__),
                                    "data", "benchmarks", "nanobody")
    cands = sorted(glob.glob(os.path.join(nanobody_dir, "*h-NbBCII10*cdr123*.cif.zst")))
    if not cands:
        raise FileNotFoundError(f"no h-NbBCII10 benchmark CIF under {nanobody_dir}")
    buf = io.BytesIO()
    with open(cands[0], "rb") as fh:
        zstandard.ZstdDecompressor().copy_stream(fh, buf)   # stream: frames carry no content size
    lines = buf.getvalue().decode("utf-8", "replace").splitlines()
    cols: List[str] = []
    data_start = None
    for i, ln in enumerate(lines):
        if ln.startswith("_atom_site."):
            cols.append(ln.strip().replace("_atom_site.", ""))
        elif cols:
            data_start = i
            break
    ix = {c: k for k, c in enumerate(cols)}
    byres: Dict[int, list] = {}
    for ln in lines[data_start:]:
        s = ln.strip()
        if s.startswith("ATOM") or s.startswith("HETATM"):
            r = ln.split()
            if len(r) == len(cols) and r[ix["auth_asym_id"]] == "H":
                byres.setdefault(int(r[ix["auth_seq_id"]]), []).append(r)
        elif s:
            break  # end of the _atom_site loop
    fw_ids = [rid for rid in sorted(byres)
              if any(a[ix["label_atom_id"]] == "CA" for a in byres[rid])]
    if not fw_ids:
        raise ValueError("benchmark chain H had no framework (Cα) residues")

    def _finite(*vals: str) -> bool:
        try:
            return all(math.isfinite(float(v)) for v in vals)
        except ValueError:
            return False

    out: List[str] = []
    serial = 0
    seq = ""
    for new, rid in enumerate(fw_ids, start=1):
        comp = byres[rid][0][ix["label_comp_id"]]
        seq += _THREE_TO_ONE.get(comp, "X")
        for a in byres[rid]:
            an = a[ix["label_atom_id"]]
            if an == "UNK":
                continue
            x, y, z = a[ix["Cartn_x"]], a[ix["Cartn_y"]], a[ix["Cartn_z"]]
            if not _finite(x, y, z):
                continue   # drop masked/terminal atoms with no coordinates
            serial += 1
            name = an if len(an) >= 4 else (" " + an)
            out.append(f"ATOM  {serial:>5} {name:<4} {comp:>3} F{new:>4}    "
                       f"{float(x):8.3f}{float(y):8.3f}{float(z):8.3f}  1.00  0.00          "
                       f"{a[ix['type_symbol']]:>2}")
    out.append("TER")
    # FR spans in the renumbered 1..N frame: split at the CDR numbering-gaps in the source ids.
    blocks: List[List[int]] = [[1]]
    for k in range(1, len(fw_ids)):
        (blocks[-1].append(k + 1) if fw_ids[k] - fw_ids[k - 1] == 1 else blocks.append([k + 1]))
    if len(blocks) != 4:
        raise ValueError(f"expected 4 framework (FR) blocks from the benchmark scaffold, got {len(blocks)}")
    fr_spans = {name: (blk[0], blk[-1]) for name, blk in zip(("FR1", "FR2", "FR3", "FR4"), blocks)}
    if seq != _H_NBBCII10_FR_SEQ:
        print(f"[framework] note: extracted scaffold FR sequence differs from the recorded h-NbBCII10 "
              f"({len(seq)} aa) — using it as-is")
    return "\n".join(out) + "\n", fr_spans, seq


def _keep_fr_residues(pdb_str: str, fr_seq_ranges: List[Tuple[int, int]]
                      ) -> Tuple[str, Dict[str, Tuple[int, int]]]:
    """Keep ONLY the framework residues of a folded VHH PDB (drop the CDR residues), relabel the
    survivors to chain 'F' renumbered 1..M, and return (pdb_str, fr_spans {FR->(lo,hi)}) in the new
    frame. `fr_seq_ranges` are 1-based (lo, hi) sequence positions of FR1..FR4 (== residue order in
    the fold). Mirrors the benchmark's FR-only framework motif so the custom-framework path feeds RFD4
    the same shape as the default path."""
    keep = set()
    for lo, hi in fr_seq_ranges:
        keep.update(range(lo, hi + 1))
    out: List[str] = []
    cur_key = None
    src_id = 0
    remap: Dict[int, int] = {}
    for ln in pdb_str.splitlines():
        if ln[:4] != "ATOM" or len(ln) < 27:
            continue
        key = (ln[22:26], ln[26])
        if key != cur_key:
            cur_key = key
            src_id += 1
            if src_id in keep:
                remap[src_id] = len(remap) + 1
        if src_id in keep:
            out.append(ln[:21] + "F" + f"{remap[src_id]:>4}" + " " + ln[27:])
    out.append("TER")
    spans: Dict[str, Tuple[int, int]] = {}
    pos = 0
    for name, (lo, hi) in zip(("FR1", "FR2", "FR3", "FR4"), fr_seq_ranges):
        length = hi - lo + 1
        spans[name] = (pos + 1, pos + length)
        pos += length
    return "\n".join(out) + "\n", spans


def _relabel_renumber_pdb(pdb_str: str, new_chain: str = "F") -> Tuple[str, int]:
    """Rewrite protein ATOM records to chain `new_chain` with residues renumbered 1..N in order
    (dropping HETATM/water/insertion codes). Returns (pdb_text, n_residues). Pure text — keeps the
    framework CIF numbering aligned with the FR spans carried in the framework context."""
    out: List[str] = []
    cur = None
    newid = 0
    for ln in pdb_str.splitlines():
        if ln[:4] != "ATOM" or len(ln) < 27:
            continue
        key = (ln[22:26], ln[26])  # resSeq + iCode
        if key != cur:
            cur = key
            newid += 1
        out.append(ln[:21] + new_chain + f"{newid:>4}" + " " + ln[27:])
    out.append("TER")
    return "\n".join(out) + "\n", newid


def _combine_framework_antigen_cif(framework_pdb_str: str, antigen_pdb_str: str,
                                   antigen_chain: str, tmpdir: str) -> str:
    """Build a single multi-chain CIF (the condition_spec `input` must be one file): framework as
    chain 'F' renumbered 1..N, plus the antigen chain verbatim (original numbering, so epitope
    residue ids stay valid). Returns the CIF path."""
    fw_text, _ = _relabel_renumber_pdb(framework_pdb_str, "F")
    ag_lines = [ln for ln in antigen_pdb_str.splitlines()
                if ln[:4] == "ATOM" and len(ln) > 21 and ln[21] == antigen_chain]
    combined = fw_text + "\n".join(ag_lines) + "\nTER\nEND\n"
    combined_pdb = os.path.join(tmpdir, "framework_antigen.pdb")
    with open(combined_pdb, "w") as f:
        f.write(combined)
    cif_path = os.path.join(tmpdir, "framework_antigen.cif")
    _pdb_to_cif(combined_pdb, cif_path)
    return cif_path


def _framework_condition_spec(combined_cif: str, fr_spans: Dict[str, Tuple[int, int]],
                              cdr_windows: Dict[str, Tuple[int, int]],
                              epitope_residues: List[int], antigen_chain: str, antigen_res_range,
                              tmpdir: str, add_cath: bool = False) -> str:
    """Build the framework-conditioning condition_spec: fixed FR segments (chain F) interleaved with
    designed CDR windows → one antibody chain, then the antigen chain. Framework = internal-coordinate
    motif (C_CRDGRP 1, free placement — the model docks it) + fixed sequence (C_SEQ); antigen backbone
    fixed (C_CRD) with epitope hotspots (C_HOT, chain-qualified). `fr_spans` are (lo, hi) residue ids
    on chain F; `cdr_windows` are (min_len, max_len) design lengths for each loop between them.

    `add_cath` is OFF by default to mirror the validated nanobody benchmark exactly — that task uses
    only internal-coord + sequence + hotspot conditioning (verified in the benchmark condition CIF's
    mask columns), and the FR internal-coord motif already pins the immunoglobulin fold. C_CATH is not
    part of that benchmark, so adding it is an untested condition whose failure would silently drop the
    run back to de-novo; leave it opt-in."""
    import json as _json
    alo, ahi = antigen_res_range if antigen_res_range else (1, 1)
    parts: List[str] = []
    fr_ranges: List[Tuple[int, int]] = []
    for fr, cdr in (("FR1", "CDR1"), ("FR2", "CDR2"), ("FR3", "CDR3")):
        lo, hi = fr_spans[fr]
        parts.append(f"F{lo}-{hi}")                   # fixed framework segment (internal-coord motif)
        fr_ranges.append((lo, hi))
        clo, chi = cdr_windows[cdr]
        parts.append(f"{int(clo)}-{int(chi)}")        # designed CDR loop (fresh length window)
    lo, hi = fr_spans["FR4"]
    parts.append(f"F{lo}-{hi}")
    fr_ranges.append((lo, hi))
    contig = ", ".join(parts) + f", /0, {antigen_chain}{alo}-{ahi}"
    fr_sel = "sel('F/*/[" + ", ".join(f"{lo}-{hi}" for lo, hi in fr_ranges) + "]')"
    conditions: Dict[str, Any] = {
        # Framework as a free-placement internal-coordinate motif — the model docks it to the antigen.
        # Restrict the group to BACKBONE atoms: MaterializeInternalCoords requires every atom in a
        # coordinate group to have finite coords, but the atomworks PDB->CIF round-trip completes each
        # residue to its full template, re-adding NaN placeholders for the side-chain/terminal atoms the
        # benchmark leaves masked (OXT, etc.). A residue-level select would pull those NaN atoms into the
        # group and raise "needs finite live coordinates"; backbone N/CA/C/O are always present + finite
        # and fully define the framework fold (C_SEQ still pins identity so side chains are repacked).
        "C_CRDGRP": {1: [{"select": f"{fr_sel} and is_protein_backbone"}]},
        # Fix sequence identity for BOTH the humanized framework and the known antigen.
        "C_SEQ": {True: [{"select": fr_sel}, {"select": f"sel('{antigen_chain}')"}]},
        # Antigen backbone fixed (sidechains repack) — same convention as the PPI/binder spec.
        "C_CRD": {True: [{"select": f"sel('{antigen_chain}') and is_protein_backbone"}]},
    }
    if epitope_residues:
        conditions["C_HOT"] = {True: [{"select": f"sel('{antigen_chain}/*/{int(r)}')"}
                                      for r in epitope_residues]}
    if add_cath:
        conditions["C_CATH"] = {"2.60.40": [{"select": fr_sel}]}     # immunoglobulin topology
    spec = {"antibody": {"input": combined_cif, "contig": contig, "conditions": conditions}}
    spec_path = os.path.join(tmpdir, "framework_spec.json")
    with open(spec_path, "w") as f:
        f.write(_json.dumps(spec))
    return spec_path


def default_framework_ctx() -> Optional[Dict[str, Any]]:
    """Default framework context: the REAL h-NbBCII10 scaffold extracted from the installed rfproteina
    nanobody benchmark (no ESMFold, no anarcii — coords + FR partition are known by construction).
    Returns None (→ caller falls back to de-novo) only if the benchmark can't be read."""
    try:
        pdb_str, fr_spans, seq = _extract_benchmark_framework()
    except Exception as e:  # noqa: BLE001
        print(f"[framework] could not load the h-NbBCII10 benchmark scaffold "
              f"({type(e).__name__}: {str(e)[:160]}); framework conditioning disabled (de-novo fallback)")
        return None
    print(f"[framework] real h-NbBCII10 scaffold loaded ({len(seq)} aa); "
          + ", ".join(f"{r}={fr_spans[r][0]}-{fr_spans[r][1]}" for r in ("FR1", "FR2", "FR3", "FR4")))
    return {"seq": seq, "pdb": pdb_str, "fr_spans": fr_spans,
            "cdr_windows": dict(_DEFAULT_CDR_WINDOWS), "source": "h-NbBCII10 (rfproteina benchmark)"}


def build_framework_ctx(framework_seq: str, framework_pdb_str: str) -> Optional[Dict[str, Any]]:
    """Framework context for a USER-SUPPLIED acceptor sequence: IMGT-number it (anarcii), keep ONLY its
    FR residues from the folded structure (drop the acceptor's own CDRs — RFD4 designs fresh ones), and
    return the same ctx shape as `default_framework_ctx`. Returns None (→ de-novo fallback) if the
    acceptor can't be numbered into FR1-FR4 + CDR3. The default path uses `default_framework_ctx` and
    does NOT depend on this (so anarcii is only needed when a custom framework is supplied)."""
    regions = _number_regions(framework_seq)
    if regions is None or not all(regions.get(fr) for fr in ("FR1", "FR2", "FR3", "FR4")) \
            or not regions.get("CDR3"):
        print("[framework] acceptor sequence did not number into FR1-FR4 + CDR3; "
              "framework conditioning disabled (falling back to de-novo)")
        return None
    order = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]
    vdom = "".join(regions.get(r, "") for r in order)
    start = framework_seq.find(vdom)
    pos = start if start >= 0 else 0
    seq_ranges: Dict[str, Tuple[int, int]] = {}
    for r in order:
        length = len(regions.get(r, ""))
        seq_ranges[r] = (pos + 1, pos + length)
        pos += length
    fr_seq_ranges = [seq_ranges[fr] for fr in ("FR1", "FR2", "FR3", "FR4")]
    fr_pdb, fr_spans = _keep_fr_residues(framework_pdb_str, fr_seq_ranges)
    fr_seq = "".join(regions[fr] for fr in ("FR1", "FR2", "FR3", "FR4"))
    print("[framework] custom acceptor numbered; FR spans: "
          + ", ".join(f"{r}={fr_spans[r][0]}-{fr_spans[r][1]}" for r in ("FR1", "FR2", "FR3", "FR4")))
    return {"seq": fr_seq, "pdb": fr_pdb, "fr_spans": fr_spans,
            "cdr_windows": dict(_DEFAULT_CDR_WINDOWS), "source": "custom (anarcii-numbered)"}


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


def _structure_to_pdb_and_seq(sample: Dict[str, Any], exclude_chain: Optional[str] = None) -> tuple:
    """biotite AtomArray → (pdb_string, sequence). Mirrors nb01.

    For the binder/VHH path the RFD4 output is a TWO-chain complex — the designed
    binder PLUS the preserved antigen chain (verified on an H100 run: designed chain
    'B', antigen chain 'A'). Pass `exclude_chain=<antigen_chain>` to drop the antigen
    and keep ONLY the designed binder, then relabel it to 'A' so everything downstream
    (anarcii numbering, ProteinMPNN `fixed_positions={"A": ...}`, Boltz `protein_B`,
    developability scorers, the viewer) operates on the isolated VHH — NOT the
    VHH+antigen fusion. (Omitting this made the whole complex the "designed sequence",
    which silently broke is_vhh, Boltz binding, and every per-sequence score.) Relies on
    RFD4 preserving the input antigen's chain id; falls back to the full array if the
    exclude would drop everything."""
    import io
    import numpy as _np
    aa = sample["generated_atom_array"]
    if exclude_chain is not None:
        keep = aa.chain_id != exclude_chain
        if keep.any() and not keep.all():
            aa = aa[keep]
            aa.set_annotation("chain_id", _np.full(aa.array_length(), "A"))
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
                 num_samples: int, framework_ctx: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Generate `num_samples` VHH candidates against the antigen epitope.
    Returns a DataFrame with columns `designed_pdb`, `designed_sequence`,
    `sample_id` — the same contract the enzyme loop expects from `call_ame`.

    When `framework_ctx` is given (from `build_framework_ctx`), uses the native
    FRAMEWORK-CONDITIONING task (keep a real VHH framework, design only the CDR loops →
    genuine antibody folds); otherwise falls back to the de-novo binder spec. Any failure
    in the framework path degrades to de-novo rather than killing the run."""
    from rfproteina.generate import get_contig_or_design_problem_dataset
    from rfproteina.datapipes.collate import collate_batch
    model, transform, torch = ctx["model"], ctx["transform"], ctx["torch"]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        antigen_pdb = os.path.join(tmp, "antigen.pdb")
        with open(antigen_pdb, "w") as f:
            f.write(antigen_pdb_str)
        res_range = _chain_res_range(antigen_pdb_str, antigen_chain)
        spec_path = None
        if framework_ctx is not None:
            try:
                combined_cif = _combine_framework_antigen_cif(
                    framework_ctx["pdb"], antigen_pdb_str, antigen_chain, tmp)
                spec_path = _framework_condition_spec(
                    combined_cif, framework_ctx["fr_spans"], framework_ctx["cdr_windows"],
                    epitope_residues, antigen_chain, res_range, tmp)
                print(f"[framework] using framework-conditioning condition_spec "
                      f"(scaffold: {framework_ctx.get('source', '?')})")
            except Exception as e:  # noqa: BLE001
                print(f"[framework] spec build failed ({type(e).__name__}: {str(e)[:160]}); "
                      "falling back to de-novo binder generation")
                spec_path = None
        if spec_path is None:
            # De-novo binder spec: antigen context + epitope hotspots, fresh VHH-length chain.
            antigen_cif = os.path.join(tmp, "antigen.cif")
            _pdb_to_cif(antigen_pdb, antigen_cif)
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
        # Keep ONLY the designed binder chain — the RFD4 output also carries the
        # preserved antigen chain, and scoring the VHH+antigen fusion silently broke
        # is_vhh / Boltz / developability. Drop the antigen chain here.
        pdb_str, seq = _structure_to_pdb_and_seq(out, exclude_chain=antigen_chain)
        # Also keep the FULL RFD4 complex (antigen chain A + designed VHH chain B) so the
        # binding pose/epitope is persisted — downstream scoring still uses the isolated VHH.
        complex_pdb, _ = _structure_to_pdb_and_seq(out, exclude_chain=None)
        rows.append({"sample_id": i, "designed_pdb": pdb_str, "designed_sequence": seq,
                     "complex_pdb": complex_pdb})
    return pd.DataFrame(rows)
