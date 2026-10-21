# Databricks notebook source
# MAGIC %md
# MAGIC # TEMPLATE · Protein Design From a Paper (Genesis Workbench, endpoints-only)
# MAGIC
# MAGIC **How to use this template.** This is the skeleton the `genesis-workbench-protein-design-from-paper`
# MAGIC skill fills in. Replace every `TODO` / `<...>` with your problem's specifics, keep the design-variant
# MAGIC block that matches your task (delete the others), and run top to bottom. All model access is via
# MAGIC Genesis Workbench **serving endpoints** through the shared toolkit — no `genesis_workbench` import.
# MAGIC
# MAGIC > Fill in from your paper: **what you're designing** (binder vs protein / binder vs ligand / motif
# MAGIC > scaffold / de novo monomer / optimize an existing design), the **target** (PDB id, sequence, or
# MAGIC > SMILES), and any **constraints** (length, hotspots, chains, motif residues).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paper & objective
# MAGIC - **Paper:** `<citation / DOI / URL>`
# MAGIC - **Goal (one line):** `<e.g. "design a small protein binder against <TARGET> that blocks <FUNCTION>">`
# MAGIC - **Design task:** `<binder_vs_protein | binder_vs_ligand | motif_scaffold | denovo_monomer | optimize>`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0 · Load the shared toolkit (endpoint callers + helpers)

# COMMAND ----------

# MAGIC %run ./00_toolkit

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Define the problem
# MAGIC Set your target and constraints. Use ONE of the target sources below.

# COMMAND ----------

# ---- Target source (pick one; comment out the rest) ----
TARGET_PDB_ID = "<e.g. 1CTX>"        # fetch a public structure from RCSB
TARGET_CHAIN  = "A"
target_pdb_text = fetch_pdb(TARGET_PDB_ID)
target_seq = chain_sequence(target_pdb_text, TARGET_CHAIN)

# target_seq = "<PASTE SEQUENCE>"; target_pdb_text = esmfold(target_seq)   # fold a sequence target
# LIGAND_SMILES = "<SMILES>"                                               # small-molecule target

# ---- Constraints (edit to your problem) ----
HOTSPOTS      = "<comma residue numbers on TARGET_CHAIN, or ''>"   # e.g. "30,33,36"
LENGTH_MIN, LENGTH_MAX = 50, 80
NUM_DESIGNS   = 8
print(f"Target {TARGET_PDB_ID} chain {TARGET_CHAIN}: {len(target_seq)} aa  hotspots={HOTSPOTS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Generate designs — KEEP THE BLOCK FOR YOUR TASK, DELETE THE OTHERS

# COMMAND ----------

# ===== TASK A — binder vs a PROTEIN target =====================================
raw = gen_protein_binders(target_pdb=target_pdb_text, target_chain=TARGET_CHAIN,
                          hotspots=HOTSPOTS, length_min=LENGTH_MIN, length_max=LENGTH_MAX,
                          num_samples=NUM_DESIGNS)

# ===== TASK B — binder vs a SMALL MOLECULE (ligand) ============================
# ligand_pdb = extract_ligand_pdb(fetch_pdb("<PDB_WITH_LIGAND>"), "<HET_CODE>")
# raw = gen_ligand_binders(ligand_pdb, length_min=LENGTH_MIN, length_max=LENGTH_MAX, num_samples=NUM_DESIGNS)

# ===== TASK C — scaffold a functional MOTIF ====================================
# motif_pdb = fetch_pdb("<MOTIF_PDB>")   # or a /Volumes/... path content
# raw = scaffold_motif(motif_pdb, motif_chain="B", length_min=LENGTH_MIN, length_max=LENGTH_MAX, num_samples=NUM_DESIGNS)

# ===== TASK D — de novo monomer / diversify a fold =============================
# bb = rfdiffusion_inpaint(target_pdb_text, start_idx=<i>, end_idx=<j>)[0]   # regenerate a span
# raw = [{"sample_id": k, "sequence": s, "rewards": 0.0, "pdb_output": bb} for k, s in enumerate(proteinmpnn(bb))]

# ===== TASK E — optimize an EXISTING design ====================================
# See the skill's "optimize" recipe: re-sample around a seed (endpoints-only) or dispatch the
# run_enzyme_optimization_gwb job (motif/enzyme capstone) via the Jobs API. Start from TASK A/C output.

designs = [{"sample_id": str(r.get("sample_id", i)), "sequence": str(r.get("sequence", "")),
            "rewards": to_float(r.get("rewards", 0.0), 0.0), "backbone_pdb": r.get("pdb_output", "")}
           for i, r in enumerate(raw) if str(r.get("sequence", "")).strip()]
display(pd.DataFrame([{k: d[k] for k in ("sample_id", "sequence", "rewards")} for d in designs]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Monomer fold gate (ESMFold pLDDT)

# COMMAND ----------

PLDDT_CUTOFF = 70.0
for d in designs:
    d["mean_plddt"] = mean_plddt(esmfold(d["sequence"]))
folded = sorted([d for d in designs if d["mean_plddt"] >= PLDDT_CUTOFF],
                key=lambda d: d["mean_plddt"], reverse=True)
print(f"{len(folded)}/{len(designs)} passed pLDDT ≥ {PLDDT_CUTOFF}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Complex confidence (Boltz-2) — only when there's a binding partner
# MAGIC Protein target: `protein_A:<binder>;protein_B:<target_seq>`. Ligand target: `protein_A:<binder>;ligand_B:<SMILES>`.
# MAGIC (For a de novo monomer with no partner, skip to Step 5.)

# COMMAND ----------

TOP_N, IPTM_CUTOFF = 4, 0.50
shortlist = folded[:TOP_N]
for d in shortlist:
    spec = f"protein_A:{d['sequence']};protein_B:{target_seq}"   # ligand: f"...;ligand_B:{LIGAND_SMILES}"
    res = boltz_complex(spec)
    d["iptm"] = to_float(res.get("iptm")); d["ptm"] = to_float(res.get("ptm"))
    d["complex_pdb"] = res.get("pdb", "")
ranked = sorted(shortlist, key=lambda d: (d["iptm"] if not np.isnan(d["iptm"]) else -1), reverse=True)
passers = [d for d in ranked if d["iptm"] >= IPTM_CUTOFF] or ranked[:TOP_N]
print(f"{len([d for d in ranked if d['iptm'] >= IPTM_CUTOFF])}/{len(shortlist)} passed ipTM ≥ {IPTM_CUTOFF}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Developability screen

# COMMAND ----------

for d in passers:
    d.update(developability(d["sequence"]))
display(pd.DataFrame([{"sample_id": d["sample_id"], **{k: d.get(k) for k in
        ("iptm", "mean_plddt", "predicted_solubility", "predicted_tm_celsius",
         "predicted_stability", "predicted_immuno_burden")}} for d in passers]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · Composite reward & selection

# COMMAND ----------

df = pd.DataFrame(passers)
axis_cols = {"plddt": ("mean_plddt", True), "iptm": ("iptm", True),
             "solubility": ("predicted_solubility", True), "stability": ("predicted_stability", True),
             "tm": ("predicted_tm_celsius", True), "immuno": ("predicted_immuno_burden", False)}
# Drop axes you didn't compute (e.g. no 'iptm' for a monomer-only task).
axis_cols = {k: v for k, v in axis_cols.items() if v[0] in df.columns}
df["composite_reward"] = composite_reward(df, axis_cols)
df = df.sort_values("composite_reward", ascending=False).reset_index(drop=True)
display(df[["sample_id", "composite_reward"] + [v[0] for v in axis_cols.values()] + ["sequence"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · Reflection & honest scope
# MAGIC - What survived each gate, and why? Which axis did the damage?
# MAGIC - **In silico ≠ wet lab.** ipTM is a *confidence* two chains dock as modeled, not a measured Kd;
# MAGIC   PLTNUM is a relative ranker, not hours; none of these prove function. The funnel's job is to turn an
# MAGIC   intractable search into a short, orderable list — then the experiments decide.
# MAGIC - **Compare to the paper:** fetch any deposited design/complex and eyeball it against your top pick.
