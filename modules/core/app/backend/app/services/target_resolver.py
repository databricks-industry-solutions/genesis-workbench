"""Target Resolver — gene symbol → canonical protein sequence.

Self-contained: queries the `gene_sequences` Delta table (SwissProt reviewed
human proteins, ingested once by notebooks/ingest_uniprot_genes.py) — no
external API at runtime. Backs the Large Molecule "Resolve from gene symbol"
input so a target surfaced by Single Cell / Genomics (a gene name) becomes a
sequence without a manual UniProt copy-paste.
"""
from __future__ import annotations

import os
import re

from genesis_workbench.workbench import execute_select_query

GENE_TABLE = "gene_sequences"
# Gene symbols are alphanumeric plus a few separators; reject anything else so
# the value is safe to inline into the lookup query.
_GENE_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,40}$")


def resolve_gene(gene: str) -> dict | None:
    """Return the canonical (longest reviewed) human protein for a gene symbol,
    or None if not found / invalid input."""
    g = (gene or "").strip()
    if not _GENE_RE.match(g):
        return None
    catalog = os.environ["CORE_CATALOG_NAME"]
    schema = os.environ["CORE_SCHEMA_NAME"]
    query = (
        f"SELECT gene, accession, protein_name, organism, sequence, seq_length "
        f"FROM {catalog}.{schema}.{GENE_TABLE} "
        f"WHERE upper(gene) = upper('{g}') "
        f"ORDER BY seq_length DESC LIMIT 1"
    )
    df = execute_select_query(query)
    if df is None or df.empty:
        return None
    r = df.iloc[0]
    return {
        "gene": str(r["gene"]),
        "accession": str(r["accession"]),
        "protein_name": str(r["protein_name"]),
        "organism": str(r["organism"]),
        "sequence": str(r["sequence"]),
        "length": int(r["seq_length"]),
    }
