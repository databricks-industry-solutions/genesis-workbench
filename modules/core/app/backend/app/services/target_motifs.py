"""Seed-motif lookup — turn an identified target into small-molecule binding
motifs to seed GenMol's fragment mode.

Given a gene symbol (or a protein sequence to reverse-resolve to its gene via
`gene_sequences`), return the Murcko scaffolds of that target's known ChEMBL
binders (from the `target_binders` table). Those scaffolds are the
pharmacophore "motifs" you feed GenMol `fragment_completion` to grow novel,
target-relevant analogs. Self-contained — only reads Delta tables.
"""
from __future__ import annotations

import logging
import os
import re

from genesis_workbench.workbench import execute_select_query

logger = logging.getLogger(__name__)

_GENE_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,40}$")
_SEQ_RE = re.compile(r"^[A-Za-z]{10,}$")


def _catalog_schema() -> tuple[str, str]:
    return os.environ["CORE_CATALOG_NAME"], os.environ["CORE_SCHEMA_NAME"]


def gene_from_sequence(sequence: str) -> str | None:
    """Reverse-resolve a protein sequence to its gene symbol via gene_sequences
    (exact match — works for sequences surfaced by the Target Resolver)."""
    seq = re.sub(r"\s+", "", (sequence or "")).strip()
    if not _SEQ_RE.match(seq):
        return None
    catalog, schema = _catalog_schema()
    q = (
        f"SELECT gene FROM {catalog}.{schema}.gene_sequences "
        f"WHERE sequence = '{seq}' ORDER BY seq_length DESC LIMIT 1"
    )
    try:
        df = execute_select_query(q)
    except Exception as e:
        logger.warning("gene_from_sequence failed: %s", e)
        return None
    if df is None or df.empty:
        return None
    return str(df.iloc[0]["gene"])


def seed_motifs(gene: str | None = None, sequence: str | None = None, limit: int = 8) -> dict:
    """Return candidate binding motifs for a target. Resolves a sequence to a
    gene if no gene given. Each motif: scaffold SMILES + how many known binders
    share it + best pChEMBL + an example binder."""
    g = (gene or "").strip()
    if not g and sequence:
        g = gene_from_sequence(sequence) or ""
    if not _GENE_RE.match(g):
        return {"gene": None, "motifs": []}

    catalog, schema = _catalog_schema()
    q = (
        f"SELECT murcko_scaffold, count(*) AS n, max(pchembl) AS best, "
        f"min(binder_smiles) AS example "
        f"FROM {catalog}.{schema}.target_binders "
        f"WHERE upper(gene) = upper('{g}') "
        f"AND murcko_scaffold IS NOT NULL AND length(murcko_scaffold) > 0 "
        f"GROUP BY murcko_scaffold ORDER BY n DESC, best DESC LIMIT {int(limit)}"
    )
    try:
        df = execute_select_query(q)
    except Exception as e:
        # target_binders may not be built yet — degrade gracefully.
        logger.warning("seed_motifs lookup failed: %s", e)
        return {"gene": g, "motifs": []}
    if df is None or df.empty:
        return {"gene": g, "motifs": []}

    motifs = [
        {
            "scaffold": str(r["murcko_scaffold"]),
            "count": int(r["n"]),
            "best_pchembl": (None if r["best"] is None else float(r["best"])),
            "example_smiles": str(r["example"]),
        }
        for _, r in df.iterrows()
    ]
    return {"gene": g, "motifs": motifs}
