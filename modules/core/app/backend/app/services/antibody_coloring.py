"""Per-residue coloring for the Antibody Design structure viewer.

Each mode returns a `(res_values, theme, with_surface)` triple consumed by
`molstar.molstar_html_colored`, which encodes the per-residue value into the mmCIF B-factor and
colors by the given Mol* theme. Everything is computed from the PDB + sequence alone — the app
backend has no `anarcii`, so CDR/framework is found by locating the real h-NbBCII10 framework
fragments in the VHH sequence (the framework-conditioned designs are built on that scaffold),
epitope/paratope is 5 Å interface geometry, pLDDT reads the ESMFold B-factor, and liabilities are
the same rule-based sequence scan the orchestrator uses.

Category codes (0-100) are chosen so the continuous "uncertainty" ramp renders them as distinct,
legible colors; pLDDT uses the dedicated "plddt-confidence" palette on a 0-100 scale.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

# Real h-NbBCII10 framework fragments (FR1-FR4). Locating them in a VHH sequence partitions it into
# framework vs CDR without anarcii — exact because framework conditioning pins them verbatim (C_SEQ).
_H_NBBCII10_FR = [
    "QVQLVESGGGLVQPGGSLRLSC",
    "WFRQAPGQGLEAVA",
    "YADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYC",
    "WGQGTLVTVS",
]

# High-signal developability liabilities (name, regex) — mirrors the orchestrator's liability_scan
# (deamidation, isomerization, fragmentation, N-glycosylation sequon). The chemically labile residue
# is the match start. Met/Trp oxidation are omitted here (too common to be a useful highlight).
_LIABILITY_PATTERNS: List[Tuple[str, str]] = [
    ("deamidation", r"N[GS]"),
    ("isomerization", r"D[GSTH]"),
    ("fragmentation", r"DP"),
    ("n_glyc_sequon", r"N[^P][ST]"),
]

# Category codes on the 0-100 "uncertainty" ramp.
_C_NEUTRAL = 12.0     # background (framework / non-interface / non-liable)
_C_EPITOPE = 92.0     # antigen interface
_C_PARATOPE = 50.0    # VHH interface
_C_CDR = {1: 55.0, 2: 75.0, 3: 95.0}
_C_LIABLE = 90.0


def _ca_residues(pdb_str: str, chain: str) -> List[Tuple[int, str]]:
    """[(resid, one-letter), …] for the CA atoms of `chain`, in file order (each residue once)."""
    aa3 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
        "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
        "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    out: List[Tuple[int, str]] = []
    for ln in pdb_str.splitlines():
        if ln[:4] != "ATOM" or len(ln) < 26 or ln[21] != chain or ln[12:16].strip() != "CA":
            continue
        try:
            out.append((int(ln[22:26]), aa3.get(ln[17:20].strip(), "X")))
        except ValueError:
            continue
    return out


def _chains(pdb_str: str) -> List[str]:
    """Chain ids present (ATOM records), in first-seen order."""
    seen: List[str] = []
    for ln in pdb_str.splitlines():
        if ln[:4] == "ATOM" and len(ln) > 21 and ln[21] not in seen:
            seen.append(ln[21])
    return seen


def _resolve_chains(pdb_str: str, antigen_chain: str) -> Tuple[str, Optional[str]]:
    """Return (vhh_chain, antigen_chain_or_None). Single chain → it's the VHH. Two chains → the VHH
    is the one that is not `antigen_chain` (falling back to the shorter chain if that's ambiguous)."""
    chains = _chains(pdb_str)
    if len(chains) <= 1:
        return (chains[0] if chains else "A"), None
    if antigen_chain in chains:
        vhh = next((c for c in chains if c != antigen_chain), chains[0])
        return vhh, antigen_chain
    lengths = {c: len(_ca_residues(pdb_str, c)) for c in chains}
    ag = max(lengths, key=lengths.get)
    vhh = min(lengths, key=lengths.get)
    return vhh, ag


def _interface(pdb_str: str, chain_a: str, chain_b: str, cutoff: float = 5.0
               ) -> Tuple[set, set]:
    """(chain_a interface resids, chain_b interface resids) — residues with any atom within `cutoff`
    of any atom of the other chain."""
    def atoms(ch: str):
        a = []
        for ln in pdb_str.splitlines():
            if ln[:4] != "ATOM" or len(ln) < 54 or ln[21] != ch:
                continue
            try:
                a.append((int(ln[22:26]), float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
            except ValueError:
                continue
        return a
    aa, bb = atoms(chain_a), atoms(chain_b)
    cut2 = cutoff * cutoff
    ia, ib = set(), set()
    for ra, xa, ya, za in aa:
        for rb, xb, yb, zb in bb:
            if (xa - xb) ** 2 + (ya - yb) ** 2 + (za - zb) ** 2 <= cut2:
                ia.add(ra); ib.add(rb)
    return ia, ib


def _fr_cdr_values(pdb_str: str, vhh_chain: str) -> Dict[Tuple[str, int], float]:
    """Code each VHH residue by region: framework = neutral, CDR1/2/3 = distinct codes. Locates the
    h-NbBCII10 FR fragments in the chain's sequence; the stretches between consecutive FRs are CDRs."""
    res = _ca_residues(pdb_str, vhh_chain)
    seq = "".join(aa for _, aa in res)
    resids = [rid for rid, _ in res]
    # Find each FR fragment's span in the sequence (consecutive, left to right).
    fr_spans: List[Tuple[int, int]] = []
    cursor = 0
    for frag in _H_NBBCII10_FR:
        idx = seq.find(frag, cursor)
        if idx < 0:
            return {}  # not a framework-conditioned VHH we can partition without anarcii
        fr_spans.append((idx, idx + len(frag)))
        cursor = idx + len(frag)
    values: Dict[Tuple[str, int], float] = {(vhh_chain, rid): _C_NEUTRAL for rid in resids}
    # CDRs = gaps between FR1/FR2, FR2/FR3, FR3/FR4.
    for cdr_num, (a, b) in enumerate(zip(fr_spans, fr_spans[1:]), start=1):
        for pos in range(a[1], b[0]):
            if pos < len(resids):
                values[(vhh_chain, resids[pos])] = _C_CDR[cdr_num]
    return values


def _plddt_values(pdb_str: str, chain: str) -> Dict[Tuple[str, int], float]:
    """Per-residue pLDDT from the CA B-factor of `chain`, scaled to 0-100 (ESMFold writes 0-1)."""
    out: Dict[Tuple[str, int], float] = {}
    for ln in pdb_str.splitlines():
        if ln[:4] != "ATOM" or len(ln) < 66 or ln[21] != chain or ln[12:16].strip() != "CA":
            continue
        try:
            rid = int(ln[22:26]); b = float(ln[60:66])
        except ValueError:
            continue
        out[(chain, rid)] = b * 100.0 if b <= 1.0 else b
    return out


def _liability_values(pdb_str: str, vhh_chain: str) -> Dict[Tuple[str, int], float]:
    """Code VHH residues that start a sequence-liability motif (deamidation/isomerization/
    fragmentation/N-glyc sequon) + any Cys when the chain has an odd (unpaired) cysteine count."""
    res = _ca_residues(pdb_str, vhh_chain)
    seq = "".join(aa for _, aa in res)
    resids = [rid for rid, _ in res]
    values: Dict[Tuple[str, int], float] = {(vhh_chain, rid): _C_NEUTRAL for rid in resids}
    for _name, pat in _LIABILITY_PATTERNS:
        for m in re.finditer(pat, seq):
            if m.start() < len(resids):
                values[(vhh_chain, resids[m.start()])] = _C_LIABLE
    if seq.count("C") % 2 == 1:  # at least one free (non-disulfide) Cys
        for i, aa in enumerate(seq):
            if aa == "C":
                values[(vhh_chain, resids[i])] = _C_LIABLE
    return values


# color_by -> (needs the complex? , human label)
COLOR_MODES = {
    "chain": "By chain (antibody vs antigen)",
    "epitope_paratope": "Epitope / paratope (binding interface)",
    "cdr_fr": "CDR / framework",
    "plddt": "Confidence (pLDDT)",
    "liability": "Sequence liabilities",
    "mhc_burden": "Immunogenicity (MHC-II)",
}

_MHC_WINDOW = 15  # HLAIIPred's peptide window width (15-mer core)


def mhc_windows(pdb_str: str, antigen_chain: str = "A", k: int = _MHC_WINDOW
                ) -> Tuple[List[str], str, List[int]]:
    """Slide a k-mer window over the VHH chain's sequence for an MHC-II presentation scan. Returns
    (windows, vhh_chain, resids): the k-mer window strings, the resolved VHH chain id, and the chain's
    residue ids in order. Empty windows list if the chain is shorter than k (caller should fall back)."""
    vhh, _ag = _resolve_chains(pdb_str, antigen_chain)
    res = _ca_residues(pdb_str, vhh)
    seq = "".join(aa for _, aa in res)
    resids = [rid for rid, _ in res]
    if len(seq) < k:
        return [], vhh, resids
    windows = [seq[i:i + k] for i in range(0, len(seq) - k + 1)]
    return windows, vhh, resids


def mhc_burden_values(vhh_chain: str, resids: List[int], window_scores: List[float],
                      k: int = _MHC_WINDOW) -> Dict[Tuple[str, int], float]:
    """Aggregate per-window MHC-II presentation scores into a per-residue burden: each residue gets the
    mean score of the windows covering it, scaled to 0-100 for the ramp (high = strong presenter / ADA
    risk → red; low → blue). Residues not covered by any scored window stay neutral."""
    values: Dict[Tuple[str, int], float] = {(vhh_chain, rid): _C_NEUTRAL for rid in resids}
    n = len(resids)
    nwin = len(window_scores)
    for i in range(n):
        lo = max(0, i - k + 1)
        hi = min(i, nwin - 1)
        covering = [window_scores[j] for j in range(lo, hi + 1)] if hi >= lo else []
        if covering:
            score = sum(covering) / len(covering)          # mean presentation over covering windows
            values[(vhh_chain, resids[i])] = max(0.0, min(100.0, score * 100.0))
    return values


def compute_coloring(pdb_str: str, color_by: str, antigen_chain: str = "A"
                     ) -> Tuple[Optional[Dict[Tuple[str, int], float]], str, bool]:
    """Return (res_values, theme, with_surface) for a coloring mode. `res_values` is None for the
    'chain' mode (chain-id theme ignores the B-factor). Unknown/again-the-default modes fall back to
    'chain'. res_values keys are (chain, resid) matching the mmCIF the viewer builds."""
    vhh, ag = _resolve_chains(pdb_str, antigen_chain)

    if color_by == "epitope_paratope" and ag is not None:
        epi, par = _interface(pdb_str, ag, vhh)
        values: Dict[Tuple[str, int], float] = {}
        for rid, _ in _ca_residues(pdb_str, ag):
            values[(ag, rid)] = _C_EPITOPE if rid in epi else _C_NEUTRAL
        for rid, _ in _ca_residues(pdb_str, vhh):
            values[(vhh, rid)] = _C_PARATOPE if rid in par else _C_NEUTRAL
        return values, "uncertainty", False

    if color_by == "cdr_fr":
        values = _fr_cdr_values(pdb_str, vhh)
        if ag is not None:  # keep the antigen neutral in a complex view
            for rid, _ in _ca_residues(pdb_str, ag):
                values[(ag, rid)] = _C_NEUTRAL
        if values:
            return values, "uncertainty", False
        return None, "chain-id", True  # could not partition → fall back to chain view

    if color_by == "plddt":
        values = _plddt_values(pdb_str, vhh)
        if ag is not None:
            values.update(_plddt_values(pdb_str, ag))
        return values, "plddt-confidence", False

    if color_by == "liability":
        values = _liability_values(pdb_str, vhh)
        if ag is not None:
            for rid, _ in _ca_residues(pdb_str, ag):
                values[(ag, rid)] = _C_NEUTRAL
        return values, "uncertainty", False

    # "chain" (default): chain-id theme + surface for the complex.
    return None, "chain-id", True
