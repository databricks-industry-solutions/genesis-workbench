# Databricks notebook source
# MAGIC %md
# MAGIC # 🐍 De Novo Protein Binders Against Snake-Venom Toxins
# MAGIC ### Recreating *Vázquez Torres et al., Nature 2025* with Genesis Workbench serving endpoints
# MAGIC
# MAGIC **Paper:** "De novo designed proteins neutralize lethal snake venom toxins"
# MAGIC (Nature 639, 2025 — DOI [10.1038/s41586-024-08393-x](https://www.nature.com/articles/s41586-024-08393-x)).
# MAGIC The Baker lab used deep-learning protein design to invent, from scratch, small proteins that
# MAGIC bind and neutralize **three-finger toxins (3FTx)** — the lethal α-neurotoxins and cytotoxins
# MAGIC in elapid (cobra/krait/mamba) venom. With *limited* experimental screening they obtained
# MAGIC binders with sub-nanomolar affinity, high thermal stability, and 100% protection in mouse
# MAGIC lethal-challenge experiments.
# MAGIC
# MAGIC In this workshop you will reproduce the **in-silico design funnel** of that paper against the
# MAGIC paper's headline long-chain target, **α-cobratoxin** (from *Naja kaouthia*), using models that
# MAGIC are already deployed as **Genesis Workbench serving endpoints**. You will *not* import any GWB
# MAGIC library — you call the endpoints directly, exactly as any external consumer would.
# MAGIC
# MAGIC | Paper step | Paper method | GWB endpoint you'll call |
# MAGIC |---|---|---|
# MAGIC | Generate binder backbone **+** sequence | RFdiffusion + ProteinMPNN | **Proteina-Complexa Binder** |
# MAGIC | Does the binder fold? (monomer) | AlphaFold2 monomer | **ESMFold** |
# MAGIC | Fold the binder–toxin **complex** + score the interface | AF2-multimer (pae / ipTM) | **Boltz-2** |
# MAGIC | Developability pre-screen (which to "order") | wet-lab | **NetSolP · DeepSTABp · PLTNUM · MHCflurry** |
# MAGIC | Rank & select hits | funnel + Rosetta | *composite reward, computed here* |
# MAGIC
# MAGIC > **Honest scope.** The paper's *wet-lab* steps — expression yield, SPR/BLI Kd, CD melting Tm,
# MAGIC > crystallography, mouse neutralization — cannot be reproduced in silico. The developability
# MAGIC > endpoints here are *computational proxies* that decide **which designs you would order**, not
# MAGIC > a substitute for assays.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0 · Setup — connect to the Genesis Workbench serving endpoints
# MAGIC
# MAGIC We use only the **Databricks SDK** (`WorkspaceClient.serving_endpoints`). No `genesis_workbench`
# MAGIC import. `requests` is used to pull a public PDB from RCSB; `py3Dmol` (optional) renders structures.

# COMMAND ----------

# MAGIC %pip install -q "databricks-sdk>=0.50.0" requests py3Dmol
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Optional endpoint-name overrides. Leave BLANK to auto-discover from the workspace.
# (Fill these in only if auto-discovery picks the wrong endpoint.)
dbutils.widgets.text("ep_proteina_complexa", "", "Binder endpoint (blank = auto)")
dbutils.widgets.text("ep_esmfold",           "", "ESMFold endpoint (blank = auto)")
dbutils.widgets.text("ep_boltz",             "", "Boltz endpoint (blank = auto)")
dbutils.widgets.text("ep_netsolp",           "", "NetSolP endpoint (blank = auto)")
dbutils.widgets.text("ep_deepstabp",         "", "DeepSTABp endpoint (blank = auto)")
dbutils.widgets.text("ep_pltnum",            "", "PLTNUM endpoint (blank = auto)")
dbutils.widgets.text("ep_mhcflurry",         "", "MHCflurry endpoint (blank = auto)")

from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from databricks.sdk.service.serving import DataframeSplitInput

# Heavy GPU models (binder design, Boltz co-fold) can take minutes — use a long client timeout.
w = WorkspaceClient(config=Config(http_timeout_seconds=900))
print("Connected to:", w.config.host)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0.1 Resolve endpoint names (library-free discovery)
# MAGIC Genesis Workbench names its endpoints `gwb_<prefix>_<model>_endpoint`. Rather than hard-code the
# MAGIC prefix (brittle), we **list the workspace's serving endpoints and match by model slug**. A widget
# MAGIC override wins if set. This cell *fails loud* if any required model is missing or asleep — fix that
# MAGIC before the session (see the facilitator checklist in the write-up).

# COMMAND ----------

_ALL_ENDPOINTS = [e.name for e in w.serving_endpoints.list()]

def resolve_endpoint(slug: str, override: str = "", exclude=()) -> str:
    """Find the serving endpoint whose name carries `slug` (e.g. 'esmfold').
    Prefers an exact short-name stem `..._<slug>_endpoint`; falls back to substring.
    `exclude` drops sibling models that share a prefix (e.g. proteina_complexa_ligand)."""
    if override.strip():
        return override.strip()
    cands = [n for n in _ALL_ENDPOINTS if slug in n and not any(x in n for x in exclude)]
    if not cands:
        raise RuntimeError(
            f"No serving endpoint matches '{slug}'. Available: {sorted(_ALL_ENDPOINTS)}. "
            f"Is the model deployed and awake? Set the widget override if the name differs."
        )
    def stem(n: str) -> str:
        return n[:-len("_endpoint")] if n.endswith("_endpoint") else n
    exact = [n for n in cands if stem(n).endswith(slug)]
    return (exact or cands)[0]

g = dbutils.widgets.get
ENDPOINTS = {
    # the binder generator — exclude the ligand- and motif-scaffolding (AME) siblings
    "proteina_complexa": resolve_endpoint("proteina_complexa", g("ep_proteina_complexa"),
                                          exclude=("ligand", "ame")),
    "esmfold":   resolve_endpoint("esmfold",   g("ep_esmfold")),
    "boltz":     resolve_endpoint("boltz",     g("ep_boltz")),
    "netsolp":   resolve_endpoint("netsolp",   g("ep_netsolp")),
    "deepstabp": resolve_endpoint("deepstabp", g("ep_deepstabp")),
    "pltnum":    resolve_endpoint("pltnum",    g("ep_pltnum")),
    "mhcflurry": resolve_endpoint("mhcflurry", g("ep_mhcflurry")),
}
for k, v in ENDPOINTS.items():
    print(f"  {k:18s} -> {v}")
print("\n✅ All required Genesis Workbench endpoints resolved.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0.2 Thin endpoint callers
# MAGIC One tiny function per endpoint. Each is a *direct* `serving_endpoints.query` — this is the entire
# MAGIC coupling to Genesis Workbench. Note the three Databricks serving payload styles:
# MAGIC `inputs=[...]`, `dataframe_records=[{...}]`, and `dataframe_split=DataframeSplitInput(...)`.

# COMMAND ----------

def _unwrap(preds):
    """GWB endpoints sometimes nest results under {'predictions': [...]}. Normalize to a list."""
    if isinstance(preds, dict) and "predictions" in preds:
        preds = preds["predictions"]
    if isinstance(preds, dict):
        return [preds]
    return list(preds or [])

# Columns the Proteina-Complexa Binder endpoint expects (dataframe_split).
_PC_COLS = ["target_pdb", "binder_length_min", "binder_length_max",
            "num_samples", "hotspot_residues", "target_chain"]

def gen_binders(target_pdb, target_chain="A", hotspots="", length_min=50, length_max=80, num_samples=4):
    """Proteina-Complexa Binder: hotspot-aware de novo binder generation.
    Returns a list of dicts: {sample_id, pdb_output (CA backbone), sequence, rewards}."""
    row = [target_pdb, int(length_min), int(length_max), int(num_samples), str(hotspots), str(target_chain)]
    resp = w.serving_endpoints.query(
        name=ENDPOINTS["proteina_complexa"],
        dataframe_split=DataframeSplitInput(columns=_PC_COLS, data=[row]),
    )
    return [r for r in _unwrap(resp.predictions) if isinstance(r, dict)]

def esmfold(sequence: str) -> str:
    """ESMFold monomer fold. Returns a PDB string; per-residue pLDDT is in the B-factor column."""
    return w.serving_endpoints.query(name=ENDPOINTS["esmfold"], inputs=[sequence]).predictions[0]

def boltz_complex(spec: str, msa="no_msa", use_msa_server="True") -> dict:
    """Boltz-2 co-fold. `spec` uses the chain grammar 'protein_A:<seq>;protein_B:<seq>'.
    Returns a dict with 'pdb' PLUS interface-confidence scalars:
    confidence_score, ptm, iptm, ligand_iptm, protein_iptm, complex_plddt (strings)."""
    resp = w.serving_endpoints.query(
        name=ENDPOINTS["boltz"],
        inputs=[{"input": spec, "msa": msa, "use_msa_server": use_msa_server}],
    )
    out = _unwrap(resp.predictions)
    return out[0] if out else {}

def developability(slug: str, record: dict) -> dict:
    """Call a developability predictor (netsolp/deepstabp/pltnum/mhcflurry) with one sequence record."""
    resp = w.serving_endpoints.query(name=ENDPOINTS[slug], dataframe_records=[record])
    out = _unwrap(resp.predictions)
    return out[0] if out else {}

print("Endpoint callers ready: gen_binders, esmfold, boltz_complex, developability")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0.3 Local helpers (no Genesis Workbench, no modeling — just parsing & scoring)

# COMMAND ----------

import requests
import numpy as np
import pandas as pd

_AA3TO1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
    "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
    "THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M","SEC":"U","PYL":"O",
}

def fetch_pdb(pdb_id: str) -> str:
    """Download a public structure from RCSB (same source the GWB app uses for its examples)."""
    r = requests.get(f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb", timeout=30)
    r.raise_for_status()
    return r.text

def chain_sequence(pdb_text: str, chain: str = "A") -> str:
    """One-letter sequence of `chain` from ATOM/CA records, in residue order."""
    seq, seen = [], set()
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA" and line[21] == chain:
            key = (line[22:27])  # resseq + insertion code
            if key in seen:
                continue
            seen.add(key)
            seq.append(_AA3TO1.get(line[17:20].strip(), "X"))
    return "".join(seq)

def mean_plddt(pdb_text: str) -> float:
    """Mean CA B-factor. ESMFold writes per-residue pLDDT (0-100) into the B-factor column."""
    vals = [float(l[60:66]) for l in pdb_text.splitlines()
            if l.startswith("ATOM") and l[12:16].strip() == "CA"]
    return float(np.mean(vals)) if vals else float("nan")

def to_float(x, default=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default

print("Helpers ready: fetch_pdb, chain_sequence, mean_plddt, to_float")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Load the target — α-cobratoxin (PDB `1CTX`)
# MAGIC α-Cobratoxin is a **long-chain α-neurotoxin** three-finger toxin (71 residues, single chain) that
# MAGIC blocks the nicotinic acetylcholine receptor. It is the paper's headline in-vivo-neutralized target
# MAGIC (their binder **LNG**; they deposited the LNG–α-cobratoxin crystal complex as **9BK5**). We feed its
# MAGIC structure to the binder generator and its sequence to Boltz.

# COMMAND ----------

TARGET_PDB_ID = "1CTX"
TARGET_CHAIN  = "A"

target_pdb_text = fetch_pdb(TARGET_PDB_ID)
toxin_seq = chain_sequence(target_pdb_text, TARGET_CHAIN)
print(f"{TARGET_PDB_ID} chain {TARGET_CHAIN}: {len(toxin_seq)} residues")
print(toxin_seq)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pick the hotspot
# MAGIC The paper neutralizes 3FTx by binding the toxin's **three-finger loops**. Their crystal structure
# MAGIC highlights **Arg33 at the tip of loop II** of α-cobratoxin as a key binder contact. We nominate a
# MAGIC small set of loop-II residues as the design hotspot — a comma list of residue numbers on the target
# MAGIC chain. **Edit this** to explore how the hotspot steers the designs.

# COMMAND ----------

dbutils.widgets.text("hotspot_residues", "30,33,36", "Hotspot residues (loop-II tip)")
HOTSPOTS = dbutils.widgets.get("hotspot_residues")
print("Hotspot residues on chain", TARGET_CHAIN, ":", HOTSPOTS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Generate binders (Proteina-Complexa Binder)
# MAGIC In the paper this is **RFdiffusion** (backbone) **+ ProteinMPNN** (sequence). Genesis Workbench's
# MAGIC **Proteina-Complexa Binder** collapses both into one hotspot-aware call that jointly emits a backbone
# MAGIC (`pdb_output`), a sequence, and a generative-model `rewards` score. We request several designs.
# MAGIC
# MAGIC > Each design is a GPU generation — expect a few minutes. For a half-day session, 6–8 is plenty.

# COMMAND ----------

dbutils.widgets.text("num_designs", "8", "Number of binder designs")
NUM_DESIGNS = int(dbutils.widgets.get("num_designs"))

raw = gen_binders(
    target_pdb=target_pdb_text,
    target_chain=TARGET_CHAIN,
    hotspots=HOTSPOTS,
    length_min=50,
    length_max=80,
    num_samples=NUM_DESIGNS,
)

designs = [{
    "sample_id": str(r.get("sample_id", i)),
    "sequence":  str(r.get("sequence", "")),
    "rewards":   to_float(r.get("rewards", 0.0), 0.0),
    "backbone_pdb": r.get("pdb_output", ""),
} for i, r in enumerate(raw) if str(r.get("sequence", "")).strip()]

print(f"Generated {len(designs)} binder designs.")
display(pd.DataFrame([{k: d[k] for k in ("sample_id", "sequence", "rewards")} for d in designs]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Does each binder fold? (ESMFold monomer, pLDDT filter)
# MAGIC First triage: fold each designed sequence on its own and keep the ones the model is confident about.
# MAGIC This is the cheap gate before the expensive complex co-fold — mirrors the paper's monomer AF2 check.

# COMMAND ----------

PLDDT_CUTOFF = 70.0  # ESMFold pLDDT (0-100). ~70+ = confidently folded monomer.

for d in designs:
    pdb = esmfold(d["sequence"])
    d["monomer_pdb"]  = pdb
    d["mean_plddt"]   = mean_plddt(pdb)

folded = sorted([d for d in designs if d["mean_plddt"] >= PLDDT_CUTOFF],
                key=lambda d: d["mean_plddt"], reverse=True)

print(f"{len(folded)}/{len(designs)} designs passed the pLDDT ≥ {PLDDT_CUTOFF} monomer gate.")
display(pd.DataFrame([{"sample_id": d["sample_id"], "mean_plddt": round(d["mean_plddt"], 1),
                       "rewards": round(d["rewards"], 4), "len": len(d["sequence"])} for d in folded]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Fold the binder–toxin **complex** and score the interface (Boltz-2)
# MAGIC This is the heart of the funnel and the modern analog of the paper's **AF2-multimer filter**.
# MAGIC We co-fold each surviving binder *with* the toxin using Boltz's chain grammar
# MAGIC `protein_A:<binder>;protein_B:<toxin>`, then rank on **ipTM / protein-ipTM** — the interface
# MAGIC confidence (BindCraft-style; higher = the model believes the two chains dock as modeled).
# MAGIC
# MAGIC > Co-folding is the slowest step. We fold only the **top N monomer-passing designs**.

# COMMAND ----------

dbutils.widgets.text("num_cofold", "4", "How many top designs to co-fold")
TOP_N   = int(dbutils.widgets.get("num_cofold"))
IPTM_CUTOFF = 0.50  # interface-confidence gate (BindCraft-style starting point)

shortlist = folded[:TOP_N]
for d in shortlist:
    spec = f"protein_A:{d['sequence']};protein_B:{toxin_seq}"
    res = boltz_complex(spec)
    d["complex_pdb"]   = res.get("pdb", "")
    d["iptm"]          = to_float(res.get("iptm"))
    d["protein_iptm"]  = to_float(res.get("protein_iptm"))
    d["ptm"]           = to_float(res.get("ptm"))
    d["complex_plddt"] = to_float(res.get("complex_plddt"))
    d["boltz_conf"]    = to_float(res.get("confidence_score"))

ranked = sorted(shortlist, key=lambda d: (d["iptm"] if not np.isnan(d["iptm"]) else -1), reverse=True)
passers = [d for d in ranked if (d["iptm"] >= IPTM_CUTOFF)]

print(f"{len(passers)}/{len(shortlist)} co-folded designs passed ipTM ≥ {IPTM_CUTOFF}.")
display(pd.DataFrame([{
    "sample_id": d["sample_id"], "iptm": round(d["iptm"], 3),
    "protein_iptm": round(d["protein_iptm"], 3), "ptm": round(d["ptm"], 3),
    "complex_plddt": round(d["complex_plddt"], 1), "mean_plddt": round(d["mean_plddt"], 1),
} for d in ranked]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### (Optional) Visualize the top binder–toxin complex

# COMMAND ----------

try:
    import py3Dmol
    best = ranked[0]
    view = py3Dmol.view(width=720, height=480)
    view.addModel(best["complex_pdb"], "pdb")
    view.setStyle({"chain": "A"}, {"cartoon": {"color": "cyan"}})     # designed binder
    view.setStyle({"chain": "B"}, {"cartoon": {"color": "magenta"}})  # α-cobratoxin
    view.zoomTo()
    displayHTML(view._make_html())
    print(f"Top design {best['sample_id']}: ipTM={best['iptm']:.3f}  (cyan = binder, magenta = toxin)")
except Exception as e:
    print("py3Dmol unavailable or no complex to show:", e)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Developability pre-screen — "which binders would we actually order?"
# MAGIC The paper's binders succeeded because they were **soluble, thermostable, and manufacturable**. We
# MAGIC run the interface-passing designs through four GWB developability endpoints. **Mind the units:**
# MAGIC
# MAGIC | Endpoint | Output | Meaning |
# MAGIC |---|---|---|
# MAGIC | NetSolP | `predicted_solubility` ∈ [0,1] | higher = more soluble |
# MAGIC | DeepSTABp | `predicted_tm_celsius` (°C) | predicted melting temperature |
# MAGIC | PLTNUM | `predicted_stability` ∈ [0,1] | relative stability ranker — **NOT hours** |
# MAGIC | MHCflurry | `predicted_immuno_burden` | MHC-I epitope burden/residue — **lower = better** |

# COMMAND ----------

MHC_ALLELES = "HLA-A*02:01,HLA-A*01:01,HLA-B*07:02,HLA-B*44:02,HLA-C*07:01,HLA-C*04:01"
candidates = passers or ranked[:TOP_N]  # fall back so the funnel still demonstrates if ipTM was harsh

for d in candidates:
    seq = d["sequence"]
    d["solubility"]   = to_float(developability("netsolp",   {"sequence": seq}).get("predicted_solubility"))
    d["tm_celsius"]   = to_float(developability("deepstabp", {"sequence": seq, "growth_temp": 37.0,
                                                              "mt_mode": "Cell"}).get("predicted_tm_celsius"))
    d["stability"]    = to_float(developability("pltnum",    {"sequence": seq}).get("predicted_stability"))
    d["immuno_burden"]= to_float(developability("mhcflurry", {"sequence": seq,
                                                              "alleles": MHC_ALLELES}).get("predicted_immuno_burden"))

display(pd.DataFrame([{
    "sample_id": d["sample_id"], "iptm": round(d["iptm"], 3),
    "solubility": round(d["solubility"], 3), "tm_C": round(d["tm_celsius"], 1),
    "stability": round(d["stability"], 3), "immuno_burden": round(d["immuno_burden"], 4),
} for d in candidates]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · Composite reward & selection (live "refinement")
# MAGIC No single number decides a binder. We combine the axes into one score, reusing the **weighting GWB's
# MAGIC Guided-Enzyme-Optimization loop uses by default** (so this mirrors the production reward). Each axis is
# MAGIC min-max normalized across the candidates and oriented so **higher = better** (immunogenicity is
# MAGIC inverted). `motif_rmsd` from the enzyme loop doesn't apply to a binder campaign, so we drop it.

# COMMAND ----------

# GWB Guided-Enzyme-Optimization default axis weights (the production reward blend).
WEIGHTS = {"plddt": 1.3, "iptm": 0.5, "solubility": 1.0, "stability": 2.6, "tm": 1.0, "immuno": 1.5}

def _norm(vals, invert=False):
    arr = np.array([v if v is not None and not np.isnan(v) else np.nan for v in vals], dtype=float)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(lo) or hi == lo:
        base = np.full(arr.shape, 0.5)   # degenerate axis → neutral 0.5
    else:
        base = (arr - lo) / (hi - lo)
        base = np.where(np.isnan(base), 0.0, base)
    return (1.0 - base) if invert else base

df = pd.DataFrame(candidates)
axes = {
    "plddt":      _norm(df["mean_plddt"]),
    "iptm":       _norm(df["iptm"]),
    "solubility": _norm(df["solubility"]),
    "stability":  _norm(df["stability"]),
    "tm":         _norm(df["tm_celsius"]),
    "immuno":     _norm(df["immuno_burden"], invert=True),  # lower burden is better
}
df["composite_reward"] = sum(WEIGHTS[a] * axes[a] for a in WEIGHTS) / sum(WEIGHTS.values())
df = df.sort_values("composite_reward", ascending=False).reset_index(drop=True)

print("🏆 Ranked finalists (the designs you'd carry forward to the wet lab):")
display(df[["sample_id", "composite_reward", "iptm", "mean_plddt",
            "solubility", "tm_celsius", "stability", "immuno_burden", "sequence"]])

winner = df.iloc[0]
print(f"\nTop pick: design {winner['sample_id']}  |  composite={winner['composite_reward']:.3f}  "
      f"ipTM={winner['iptm']:.3f}  Tm≈{winner['tm_celsius']:.0f}°C")

# COMMAND ----------

# MAGIC %md
# MAGIC ### (Optional) Log the campaign to MLflow for reproducibility
# MAGIC MLflow is a standard Databricks tool (not a GWB library) — handy for capturing the funnel.

# COMMAND ----------

try:
    import mlflow
    with mlflow.start_run(run_name=f"denovo_binder_{TARGET_PDB_ID}"):
        mlflow.log_params({"target": TARGET_PDB_ID, "chain": TARGET_CHAIN, "hotspots": HOTSPOTS,
                           "num_designs": NUM_DESIGNS, "plddt_cutoff": PLDDT_CUTOFF, "iptm_cutoff": IPTM_CUTOFF})
        mlflow.log_metric("n_generated", len(designs))
        mlflow.log_metric("n_monomer_pass", len(folded))
        mlflow.log_metric("n_interface_pass", len(passers))
        mlflow.log_metric("best_composite_reward", float(winner["composite_reward"]))
        mlflow.log_metric("best_iptm", float(winner["iptm"]))
        df.drop(columns=[c for c in ("backbone_pdb", "monomer_pdb", "complex_pdb") if c in df.columns]) \
          .to_csv("/tmp/binder_funnel.csv", index=False)
        mlflow.log_artifact("/tmp/binder_funnel.csv")
    print("Logged funnel to MLflow.")
except Exception as e:
    print("MLflow logging skipped:", e)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · Compare to the real paper
# MAGIC The authors deposited **9BK5** — their LNG binder in complex with α-cobratoxin — and reported a
# MAGIC design-vs-crystal backbone RMSD of **0.42 Å** for LNG (near-atomic agreement), Kd **1.9 nM**, and
# MAGIC Tm **> 95 °C**. Pull the real complex and compare it, qualitatively, to the one you just designed.

# COMMAND ----------

try:
    import py3Dmol
    ref = fetch_pdb("9BK5")
    view = py3Dmol.view(width=720, height=480)
    view.addModel(ref, "pdb")
    view.setStyle({"cartoon": {"colorscheme": "chainHetatm"}})
    view.zoomTo()
    displayHTML(view._make_html())
    print("9BK5 — the Baker-lab LNG binder bound to α-cobratoxin (the structure your funnel is chasing).")
except Exception as e:
    print("Could not render 9BK5:", e)

# COMMAND ----------

# MAGIC %md
# MAGIC **Reflection.** What the in-silico funnel *can* tell you: a plausible binder sequence+fold, interface
# MAGIC confidence, and a developability triage to prioritize what to order. What it *cannot*: real binding
# MAGIC affinity, measured stability, off-target behavior, or in-vivo neutralization — those still need the
# MAGIC wet lab. The value is **enrichment**: the paper reached potent binders with *limited* experimental
# MAGIC screening precisely because the computational funnel concentrated the hits.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Appendix A · Refinement by re-sampling (endpoints-only)
# MAGIC The paper improved weak initial hits with **partial diffusion** (re-sampling around a promising
# MAGIC design). The simplest endpoints-only analog: **re-run the binder generator** focused on your winning
# MAGIC design's length and hotspot, drawing more samples, then re-run steps 3–6. Try tightening the length
# MAGIC window around the winner and increasing `num_samples`.

# COMMAND ----------

# Example: resample around the winning design's length (uncomment to run — another GPU batch).
# win_len = len(winner["sequence"])
# refined = gen_binders(target_pdb=target_pdb_text, target_chain=TARGET_CHAIN, hotspots=HOTSPOTS,
#                       length_min=max(40, win_len - 5), length_max=win_len + 5, num_samples=8)
# print(f"Re-sampled {len(refined)} designs near length {win_len}; feed these back through steps 3-6.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Appendix B · Guided Enzyme Optimization job (production refinement loop) — *curious attendees*
# MAGIC Genesis Workbench ships a **reward-weighted optimization job** (`run_enzyme_optimization_gwb`) that
# MAGIC automates generate → redesign → fold → multi-axis-score → resample. It is **built for enzyme / motif
# MAGIC scaffolding** (it drives the Proteina-Complexa **AME** variant and needs a motif PDB + substrate
# MAGIC SMILES), so it is **not** a drop-in binder-vs-toxin campaign — but it's the best illustration of the
# MAGIC closed optimization loop and the same 7 reward axes we scored by hand above.
# MAGIC
# MAGIC It is a **long-running job (up to hours)** — dispatch it async via the Jobs API and inspect its MLflow
# MAGIC `reward_trajectory.csv` / `topK_pdbs/` later. (This uses the Databricks Jobs SDK against a GWB-deployed
# MAGIC job — still no `genesis_workbench` library import.)

# COMMAND ----------

# Reference only — dispatch the deployed GWB optimization job (fill motif_pdb_path + substrate_smiles).
#
# jobs = list(w.jobs.list(name="run_enzyme_optimization_gwb"))
# if not jobs:
#     print("Enzyme-optimization job is not deployed in this workspace.")
# else:
#     run = w.jobs.run_now(job_id=jobs[0].job_id, job_parameters={
#         "catalog": "<catalog>", "schema": "<schema>", "cache_dir": "enzyme_optimization",
#         "sql_warehouse_id": "<warehouse_id>", "user_email": "<you@org>",
#         "mlflow_experiment": "<experiment>", "mlflow_run_name": "workshop_enzyme_opt",
#         "motif_pdb_path": "/Volumes/<catalog>/<schema>/.../motif.pdb",
#         "motif_residues_csv": "", "target_chain": "B",
#         "scaffold_length_min": "80", "scaffold_length_max": "120",
#         "num_samples": "8", "num_iterations": "10",
#         "substrate_smiles": "<SMILES>", "references_json": "[]",
#         # 7 reward axes (same blend used in Step 6): motif_rmsd, plddt, boltz, solubility, half_life, thermostab, immuno
#         "weights_json": '{"motif_rmsd":1.0,"plddt":1.3,"boltz":0.5,"solubility":1.0,"half_life":2.6,"thermostab":1.0,"immuno":1.5}',
#         "resampling_temperature": "0.1", "strategy": "resample", "run_proteinmpnn": "true",
#     })
#     print("Dispatched enzyme-optimization run:", run.run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Appendix C · Same pipeline via the Genesis Workbench **MCP server**
# MAGIC Everything above is also exposed as **MCP tools** by the GWB MCP app, so an LLM agent can run the
# MAGIC funnel conversationally. The tool names mirror the endpoints/workflows:
# MAGIC
# MAGIC - `endpoint_proteina_complexa` — generate binders
# MAGIC - `endpoint_esmfold` — fold a sequence (monomer)
# MAGIC - `endpoint_boltz` — co-fold a complex (`protein_A:…;protein_B:…`) + confidence
# MAGIC - `endpoint_netsolp_v1`, `endpoint_deepstabp_v1`, `endpoint_pltnum_v1`, `endpoint_mhcflurry_v2` — developability
# MAGIC - `workflow_protein_binder_design` — the generate→fold→validate chain in one call
# MAGIC
# MAGIC Point your MCP client at the deployed GWB MCP app URL and ask, e.g.:
# MAGIC *"Design 6 binders against the α-cobratoxin PDB on hotspots 30,33,36; co-fold each with the toxin and
# MAGIC rank by ipTM; then screen the top 3 for solubility, Tm, and immunogenicity."*
