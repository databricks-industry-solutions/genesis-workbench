# Databricks notebook source
# MAGIC %md
# MAGIC # 🧰 Genesis Workbench Protein-Design Toolkit (endpoints-only)
# MAGIC
# MAGIC `%run` this notebook from any protein-design notebook to get a tested set of **direct serving-endpoint
# MAGIC callers** plus parsing/scoring helpers — with **no `genesis_workbench` library import**. It defines a
# MAGIC `WorkspaceClient` (`w`), lazily discovers endpoint names, and exposes one thin function per model.
# MAGIC
# MAGIC ```python
# MAGIC %run ./00_toolkit
# MAGIC ```
# MAGIC
# MAGIC Covers the full GWB protein-design family: binder design, ligand-binder design, motif scaffolding,
# MAGIC monomer primitives (ESMFold / RFdiffusion-inpaint / ProteinMPNN), complex confidence (Boltz-2),
# MAGIC docking (DiffDock), developability (NetSolP/DeepSTABp/PLTNUM/MHCflurry), and small-molecule ADMET.

# COMMAND ----------

# MAGIC %pip install -q "databricks-sdk>=0.50.0" requests py3Dmol
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import pandas as pd
import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from databricks.sdk.service.serving import DataframeSplitInput

# Heavy GPU models (generation, Boltz co-fold) can take minutes — long client timeout.
w = WorkspaceClient(config=Config(http_timeout_seconds=900))

# ── Lazy, library-free endpoint discovery ────────────────────────────────────
_EP_CACHE: dict[str, str] = {}
_ALL_ENDPOINTS: list[str] | None = None

def get_ep(slug: str, override: str = "", exclude=()) -> str:
    """Resolve a GWB serving-endpoint name by model slug (e.g. 'esmfold', 'boltz',
    'proteina_complexa'). Prefers an exact short-name stem `..._<slug>_endpoint`,
    falls back to substring. `exclude` drops sibling models (e.g. the ligand/ame
    Proteina-Complexa variants, or diffdock_esm_embeddings). Cached per session;
    fails loud if nothing matches so a missing/asleep model surfaces immediately."""
    if slug in _EP_CACHE:
        return _EP_CACHE[slug]
    if override.strip():
        _EP_CACHE[slug] = override.strip()
        return _EP_CACHE[slug]
    global _ALL_ENDPOINTS
    if _ALL_ENDPOINTS is None:
        _ALL_ENDPOINTS = [e.name for e in w.serving_endpoints.list()]
    cands = [n for n in _ALL_ENDPOINTS if slug in n and not any(x in n for x in exclude)]
    if not cands:
        raise RuntimeError(
            f"No serving endpoint matches '{slug}'. Available: {sorted(_ALL_ENDPOINTS)}. "
            f"Is the model deployed and awake?"
        )
    def stem(n: str) -> str:
        return n[:-len("_endpoint")] if n.endswith("_endpoint") else n
    exact = [n for n in cands if stem(n).endswith(slug)]
    chosen = (exact or cands)[0]
    _EP_CACHE[slug] = chosen
    return chosen

def _unwrap(preds):
    """Normalize endpoint predictions (some nest under {'predictions': [...]}) to a list."""
    if isinstance(preds, dict) and "predictions" in preds:
        preds = preds["predictions"]
    if isinstance(preds, dict):
        return [preds]
    return list(preds or [])

def to_float(x, default=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generation endpoints (de novo design)

# COMMAND ----------

# All three Proteina-Complexa variants share this column order (dataframe_split).
_PC_COLS = ["target_pdb", "binder_length_min", "binder_length_max",
            "num_samples", "hotspot_residues", "target_chain"]

def _proteina(slug, slot_pdb, length_min, length_max, num_samples, hotspots, chain, exclude=()):
    row = [slot_pdb, int(length_min), int(length_max), int(num_samples), str(hotspots), str(chain)]
    resp = w.serving_endpoints.query(
        name=get_ep(slug, exclude=exclude),
        dataframe_split=DataframeSplitInput(columns=_PC_COLS, data=[row]),
    )
    return [r for r in _unwrap(resp.predictions) if isinstance(r, dict)]

def gen_protein_binders(target_pdb, target_chain="A", hotspots="", length_min=50, length_max=80, num_samples=4):
    """Proteina-Complexa Binder — binders against a PROTEIN target (hotspot-aware).
    Returns dicts: {sample_id, pdb_output (CA backbone), sequence, rewards}."""
    return _proteina("proteina_complexa", target_pdb, length_min, length_max,
                     num_samples, hotspots, target_chain, exclude=("ligand", "ame"))

def gen_ligand_binders(ligand_pdb, length_min=50, length_max=80, num_samples=4):
    """Proteina-Complexa Ligand — protein binders against a SMALL-MOLECULE target.
    `ligand_pdb` is a ligand-only PDB (HETATM w/ CONECT); hotspots/chain are ignored."""
    return _proteina("proteina_complexa_ligand", ligand_pdb, length_min, length_max,
                     num_samples, "", "A")

def scaffold_motif(motif_pdb, motif_chain="B", length_min=50, length_max=80, num_samples=4):
    """Proteina-Complexa AME — scaffold a protein around a functional MOTIF (active site, epitope).
    Returns dicts: {sample_id, pdb_output, sequence, rewards}."""
    return _proteina("proteina_complexa_ame", motif_pdb, length_min, length_max,
                     num_samples, "", motif_chain)

def rfdiffusion_inpaint(pdb: str, start_idx: int, end_idx: int) -> list[str]:
    """RFdiffusion motif inpainting — regenerate the 1-indexed inclusive span
    [start_idx, end_idx] while holding the flanks fixed. Returns backbone PDB string(s)."""
    resp = w.serving_endpoints.query(
        name=get_ep("rfdiffusion"),
        dataframe_records=[{"pdb": pdb, "start_idx": int(start_idx), "end_idx": int(end_idx)}],
    )
    return [str(s) for s in _unwrap(resp.predictions)]

def proteinmpnn(pdb: str, fixed_positions: str = "") -> list[str]:
    """ProteinMPNN inverse folding — design sequences for a backbone PDB.
    `fixed_positions` is JSON {chain:[resnums]} (1-indexed) to keep. Returns sequence strings."""
    resp = w.serving_endpoints.query(
        name=get_ep("proteinmpnn"),
        dataframe_records=[{"pdb": pdb, "fixed_positions": fixed_positions}],
    )
    return [str(s) for s in _unwrap(resp.predictions)]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Structure prediction & scoring

# COMMAND ----------

def esmfold(sequence: str) -> str:
    """ESMFold monomer fold → PDB string (per-residue pLDDT 0-100 is in the B-factor column)."""
    return w.serving_endpoints.query(name=get_ep("esmfold"), inputs=[sequence]).predictions[0]

def boltz_complex(spec: str, msa="no_msa", use_msa_server="True") -> dict:
    """Boltz-2 co-fold. `spec` uses the chain grammar, joined by ';':
      protein_A:<seq>;protein_B:<seq>   (binder:target complex)
      protein_A:<seq>;ligand_B:<SMILES> (protein:ligand)
    Returns {pdb, confidence_score, ptm, iptm, ligand_iptm, protein_iptm, complex_plddt} (strings)."""
    resp = w.serving_endpoints.query(
        name=get_ep("boltz"),
        inputs=[{"input": spec, "msa": msa, "use_msa_server": use_msa_server}],
    )
    out = _unwrap(resp.predictions)
    return out[0] if out else {}

def diffdock(protein_pdb: str, ligand_smiles: str, samples: int = 5):
    """DiffDock blind docking (ESM-2 embed → dock). Returns (best_sdf, confidence) or (None, None)."""
    emb = w.serving_endpoints.query(
        name=get_ep("diffdock_esm_embeddings"),
        dataframe_split=DataframeSplitInput(columns=["protein_pdb"], data=[[protein_pdb]]),
    )
    b64 = (_unwrap(emb.predictions)[0] or {}).get("embeddings_b64", "{}")
    poses = w.serving_endpoints.query(
        name=get_ep("diffdock", exclude=("esm_embeddings",)),
        dataframe_split=DataframeSplitInput(
            columns=["protein_pdb", "ligand_smiles", "samples_per_complex", "esm_embeddings_b64"],
            data=[[protein_pdb, ligand_smiles, int(samples), b64]]),
    )
    best = None
    for p in _unwrap(poses.predictions):
        sdf = str(p.get("ligand_sdf", ""))
        if sdf.startswith("ERROR"):
            continue
        c = to_float(p.get("confidence"), -1e9)
        if best is None or c > best[0]:
            best = (c, sdf)
    return (best[1], best[0]) if best else (None, None)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Developability & ADMET (which designs to carry forward)

# COMMAND ----------

_DEV_SPEC = {
    # slug      -> (payload builder, output column, higher_is_better)
    "netsolp":   (lambda s: {"sequence": s},                                   "predicted_solubility", True),
    "deepstabp": (lambda s: {"sequence": s, "growth_temp": 37.0, "mt_mode": "Cell"}, "predicted_tm_celsius", True),
    "pltnum":    (lambda s: {"sequence": s},                                   "predicted_stability",  True),
    "mhcflurry": (lambda s: {"sequence": s,
                             "alleles": "HLA-A*02:01,HLA-A*01:01,HLA-B*07:02,HLA-B*44:02,HLA-C*07:01,HLA-C*04:01"},
                                                                               "predicted_immuno_burden", False),
}

def developability(sequence: str, which=("netsolp", "deepstabp", "pltnum", "mhcflurry")) -> dict:
    """Run developability predictors over one sequence. Returns {col: value}.
    Units: solubility[0,1]↑, Tm °C↑, stability[0,1]↑ (relative ranker, NOT hours), immuno burden↓."""
    out = {}
    for slug in which:
        build, col, _ = _DEV_SPEC[slug]
        resp = w.serving_endpoints.query(name=get_ep(slug), dataframe_records=[build(sequence)])
        rows = _unwrap(resp.predictions)
        out[col] = to_float(rows[0].get(col)) if rows else float("nan")
    return out

def chemprop_admet(smiles_list, which=("chemprop_admet", "chemprop_bbbp", "chemprop_clintox")):
    """Small-molecule ADMET/toxicity predictors (for ligand-design campaigns). Returns {slug: predictions}."""
    out = {}
    for slug in which:
        resp = w.serving_endpoints.query(name=get_ep(slug), inputs=list(smiles_list))
        out[slug] = _unwrap(resp.predictions)
    return out

# COMMAND ----------

# MAGIC %md
# MAGIC ## Local helpers — PDB/SMILES parsing & composite scoring (no models)

# COMMAND ----------

_AA3TO1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
    "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
    "THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M","SEC":"U","PYL":"O",
}

def fetch_pdb(pdb_id: str) -> str:
    """Download a public structure from RCSB."""
    r = requests.get(f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb", timeout=30)
    r.raise_for_status()
    return r.text

def chain_sequence(pdb_text: str, chain: str = "A") -> str:
    """One-letter sequence of `chain` from CA records, in residue order."""
    seq, seen = [], set()
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA" and line[21] == chain:
            key = line[22:27]
            if key in seen:
                continue
            seen.add(key)
            seq.append(_AA3TO1.get(line[17:20].strip(), "X"))
    return "".join(seq)

def extract_ligand_pdb(pdb_text: str, resname: str) -> str:
    """Pull a ligand's HETATM (+CONECT) block out of a full PDB as a ligand-only PDB
    (input for gen_ligand_binders). `resname` is the 3-letter HET code (e.g. 'ATP')."""
    het = [l for l in pdb_text.splitlines() if l.startswith("HETATM") and l[17:20].strip() == resname]
    serials = {l[6:11].strip() for l in het}
    conect = [l for l in pdb_text.splitlines()
              if l.startswith("CONECT") and l[6:11].strip() in serials]
    return "\n".join(het + conect) + "\nEND\n"

def mean_plddt(pdb_text: str) -> float:
    """Mean CA B-factor. ESMFold writes per-residue pLDDT (0-100) into the B-factor column."""
    vals = [float(l[60:66]) for l in pdb_text.splitlines()
            if l.startswith("ATOM") and l[12:16].strip() == "CA"]
    return float(np.mean(vals)) if vals else float("nan")

def _norm(vals, invert=False):
    arr = np.array([v if v is not None and not np.isnan(v) else np.nan for v in vals], dtype=float)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(lo) or hi == lo:
        base = np.full(arr.shape, 0.5)
    else:
        base = (arr - lo) / (hi - lo)
        base = np.where(np.isnan(base), 0.0, base)
    return (1.0 - base) if invert else base

# GWB Guided-Enzyme-Optimization default reward blend (drop axes you didn't compute).
DEFAULT_WEIGHTS = {"plddt": 1.3, "iptm": 0.5, "solubility": 1.0,
                   "stability": 2.6, "tm": 1.0, "immuno": 1.5}

def composite_reward(df: pd.DataFrame, axis_cols: dict, weights: dict = None) -> pd.Series:
    """Min-max normalize each axis, orient so higher=better, weight, and average.
    `axis_cols` maps weight-key -> (dataframe column, higher_is_better).
    e.g. {'plddt': ('mean_plddt', True), 'iptm': ('iptm', True), 'immuno': ('immuno_burden', False)}."""
    weights = weights or DEFAULT_WEIGHTS
    keys = [k for k in axis_cols if k in weights]
    normed = {k: _norm(df[axis_cols[k][0]], invert=not axis_cols[k][1]) for k in keys}
    total = sum(weights[k] for k in keys) or 1.0
    return pd.Series(sum(weights[k] * normed[k] for k in keys) / total, index=df.index)

print("✅ GWB protein-design toolkit loaded. Endpoints resolve lazily on first use (get_ep).")
