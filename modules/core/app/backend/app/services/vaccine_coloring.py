"""Per-residue coloring for the Vaccine Immunogen result-dialog structure viewer.

The vaccine immunogen is a SINGLE designed chain (the epitope motif grafted onto a de-novo
scaffold) — there is no antigen complex, so the antibody `chain` / `epitope_paratope` modes do
not apply, and the MHC-II / ADA burden mode is intentionally absent (a vaccine is MEANT to be
immunogenic). Three modes make sense:

  - "motif"     — the FIXED epitope motif vs the designed scaffold (motif-vs-scaffold, the analog
                  of the antibody CDR/framework split). Driven by the per-candidate motif positions
                  the orchestrator precomputes.
  - "plddt"     — ESMFold per-residue confidence (reads the B-factor).
  - "liability" — the same rule-based sequence-liability scan the orchestrator uses.

Everything is computed CPU-side from the PDB (+ the precomputed motif positions) — the viewer never
calls a serving endpoint. The per-residue encoders are shared verbatim with the antibody viewer
(`antibody_coloring`); only the mode registry + the motif encoder are immunogen-specific.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from app.services.antibody_coloring import (
    _C_NEUTRAL,
    _ca_residues,
    _chains,
    _liability_values,
    _plddt_values,
)

# Fixed epitope motif → the red end of the "uncertainty" ramp (the scaffold stays neutral).
_C_MOTIF = 92.0

# color_by -> human label
COLOR_MODES = {
    "motif": "Epitope motif / scaffold",
    "plddt": "Confidence (pLDDT)",
    "liability": "Sequence liabilities",
}


def _single_chain(pdb_str: str) -> str:
    """The designed chain id (the immunogen is one chain; first ATOM chain, default 'A')."""
    chains = _chains(pdb_str)
    return chains[0] if chains else "A"


def motif_values_from_positions(pdb_str: str, chain: str, positions: Optional[List[int]]
                                ) -> Dict[Tuple[str, int], float]:
    """Color the fixed epitope-motif residues vs the designed scaffold. `positions` are the 1-based
    positions of the located motif in the designed sequence (the orchestrator stores them per
    candidate); the stored structure is renumbered 1..L, so the i-th Cα residue is sequence position i.
    Motif residues → _C_MOTIF, everything else → neutral. An empty/None list colors all scaffold."""
    res = _ca_residues(pdb_str, chain)
    resids = [rid for rid, _ in res]
    values: Dict[Tuple[str, int], float] = {(chain, rid): _C_NEUTRAL for rid in resids}
    for p in positions or []:
        if 1 <= int(p) <= len(resids):
            values[(chain, resids[int(p) - 1])] = _C_MOTIF
    return values


def compute_coloring(pdb_str: str, color_by: str, motif_positions: Optional[List[int]] = None
                     ) -> Tuple[Optional[Dict[Tuple[str, int], float]], str, bool]:
    """Return (res_values, theme, with_surface) for a vaccine coloring mode. Keys are (chain, resid)
    matching the mmCIF the viewer builds. Unknown modes fall back to pLDDT."""
    chain = _single_chain(pdb_str)

    if color_by == "liability":
        return _liability_values(pdb_str, chain), "uncertainty", False

    if color_by == "motif":
        return motif_values_from_positions(pdb_str, chain, motif_positions), "uncertainty", False

    # "plddt" (default): ESMFold confidence palette.
    return _plddt_values(pdb_str, chain), "plddt-confidence", False
