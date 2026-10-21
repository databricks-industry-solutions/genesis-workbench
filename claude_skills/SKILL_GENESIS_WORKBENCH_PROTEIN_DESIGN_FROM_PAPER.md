---
name: genesis-workbench-protein-design-from-paper
description: Scaffold a complete, problem-specific protein-design notebook from a research paper or a one-line goal, using Genesis Workbench serving endpoints only (no library import). Covers the full design family — binder-vs-protein, binder-vs-ligand, motif scaffolding, de novo monomer, and optimization — plus folding, complex-confidence, developability, and selection. Use when a user says "use this paper as a guideline and design/optimize XXX" and wants a ready-to-run notebook for their own target.
---

# Protein Design From a Paper — notebook scaffolder

When a user says something like:

> "Use *this paper* as the guideline and design a binder against *my target* / optimize *this protein* / scaffold *this motif* — build me the whole notebook."

…produce a **ready-to-run Databricks notebook** that reproduces the paper's *in-silico* design funnel for
**their** target, driving Genesis Workbench **only through its serving endpoints**. This skill is the
generalization of the worked snake-venom example at `workshops/snake_venom_binders/`.

## Hard rules (do not violate)

1. **Endpoints only — never import `genesis_workbench`.** No `initialize()`, `run_chain()`,
   `get_endpoint_name_for_uc_model()`, or executor internals. Call endpoints with
   `WorkspaceClient.serving_endpoints.query(...)`. Reuse the shared toolkit
   (`workshops/protein_design_from_paper/00_toolkit.py`) via `%run ./00_toolkit` — it already wraps every
   endpoint and discovers names at runtime. (Rationale: attendees learn to consume GWB as a portable
   service; see the `gwb-workshops-endpoints-only` memory.)
2. **Start from the template**, don't hand-roll structure: copy
   `workshops/protein_design_from_paper/TEMPLATE_protein_design_notebook.py`, keep the ONE design-variant
   block that matches the task, delete the others, and fill every `TODO`/`<...>`.
3. **Be honest about scope.** The wet-lab steps (affinity, measured Tm, neutralization, PK) are not
   reproducible in silico; the developability endpoints are *proxies* that prioritize what to order.
4. **Fail loud on missing models.** Endpoint discovery (`get_ep`) raises if a model is undeployed/asleep —
   keep that behavior; tell the user to warm endpoints (they get reaped for inactivity).

## Step 1 — Classify the task

Read the paper/goal and extract: **target type**, **target source**, **design intent**, **constraints**.
Map to exactly one task:

| If the user wants to… | Task | GWB generation endpoint (toolkit fn) |
|---|---|---|
| Bind a **protein** target (block an interaction, neutralize a toxin, antibody-alternative) | `binder_vs_protein` | **Proteina-Complexa Binder** → `gen_protein_binders(target_pdb, target_chain, hotspots, …)` |
| Bind a **small molecule / ligand** | `binder_vs_ligand` | **Proteina-Complexa Ligand** → `gen_ligand_binders(ligand_pdb, …)` |
| Graft/preserve a **functional motif** (active site, epitope) onto a new scaffold | `motif_scaffold` | **Proteina-Complexa AME** → `scaffold_motif(motif_pdb, motif_chain, …)` |
| Generate/diversify a **de novo monomer** fold | `denovo_monomer` | **RFdiffusion-inpaint + ProteinMPNN** → `rfdiffusion_inpaint(pdb, i, j)` then `proteinmpnn(bb)` |
| **Improve** an existing design toward developability | `optimize` | re-sample (endpoints) or the `run_enzyme_optimization_gwb` job (see Step 4) |

Target source → how to get the model input:
- **PDB id** → `fetch_pdb(id)`; sequence via `chain_sequence(pdb, chain)`.
- **Sequence only** → fold it first with `esmfold(seq)` to get a target PDB.
- **Ligand** → `extract_ligand_pdb(fetch_pdb(id), HET_CODE)` for a ligand-only PDB; keep the SMILES for Boltz/DiffDock.
- **Hotspots** → comma residue numbers on the target chain (from the paper's interface/epitope); optional.

## Step 2 — The common funnel (every task except pure monomer uses all of it)

1. **Generate** N designs with the task endpoint above → `{sample_id, sequence, rewards, pdb_output}`.
2. **Monomer gate:** `mean_plddt(esmfold(seq)) ≥ 70` — drop designs that don't fold.
3. **Complex confidence (the key filter)** with Boltz-2 — the modern analog of AF2-multimer pae/ipTM:
   - protein target: `boltz_complex(f"protein_A:{binder};protein_B:{target_seq}")`
   - ligand target: `boltz_complex(f"protein_A:{binder};ligand_B:{SMILES}")`
   - rank on `iptm` / `protein_iptm` (BindCraft-style; start cutoff ≈ 0.5). Co-fold only the top monomer-passers (slow).
4. **Developability:** `developability(seq)` → solubility↑, Tm °C↑, stability↑ (relative, *not hours*), immuno burden↓.
   For a **ligand** campaign, also screen the small molecule with `chemprop_admet([...])`.
5. **Composite reward & select:** `composite_reward(df, axis_cols)` with `DEFAULT_WEIGHTS`; rank; pick finalists.
   Drop axes you didn't compute (e.g. no `iptm` for a monomer-only task).
6. **Compare to the paper:** fetch any deposited design/complex PDB and eyeball it against the top pick.
7. (Optional) log the funnel to **MLflow** (standard Databricks, not a GWB library).

## Step 3 — Per-task deltas

- **binder_vs_protein** — the full funnel. Hotspots steer the interface. *(Worked example: `workshops/snake_venom_binders/`.)*
- **binder_vs_ligand** — generate with `gen_ligand_binders`; validate the pocket with `diffdock(binder_pdb, SMILES)` (confidence↑) and/or Boltz `ligand_B:`; add `chemprop_admet` on the ligand.
- **motif_scaffold** — generate with `scaffold_motif`; optionally redesign surface with `proteinmpnn`; verify the motif is preserved by folding (ESMFold) and, if there's a partner, Boltz. There is **no Rosetta/InterfaceAnalyzer** in GWB — rank on pLDDT/ipTM.
- **denovo_monomer** — no binding partner: `rfdiffusion_inpaint` a span of a seed PDB, `proteinmpnn` the backbone, ESMFold-gate; **skip Boltz**; score on pLDDT + developability.
- **optimize** — two options: (a) **endpoints-only re-sampling** — re-run the generator around the winner's length/hotspot with more samples, re-funnel; (b) **GWB job** — dispatch `run_enzyme_optimization_gwb` via `w.jobs.run_now(...)` (reward-weighted resample loop, 7 axes). Flag that the job is an **enzyme/motif-scaffolding capstone** (needs a motif PDB + substrate SMILES), long-running (hours), not a drop-in binder campaign.

## Step 4 — Generate the notebook

1. Copy the template to `workshops/<slug>/NN_<task>_<target>.py` (or the user's chosen path);
   ensure `00_toolkit.py` sits beside it for `%run ./00_toolkit`.
2. Fill the paper/objective/task header, the problem-definition cell (target source + constraints), and keep
   only the matching TASK block in the generation cell.
3. Wire Step 4's Boltz `spec` to the target type (protein vs ligand), and trim `axis_cols` in Step 6 to the
   axes actually computed.
4. Set sensible run sizes for a session: `NUM_DESIGNS` 6–8, `num_cofold` (TOP_N) 3–4.
5. Keep the honest-scope reflection cell.

## Endpoint reference (slug → payload → output)

| Toolkit fn | Endpoint slug | Payload style | Key outputs |
|---|---|---|---|
| `gen_protein_binders` | `proteina_complexa` (excl. ligand/ame) | `dataframe_split` cols `target_pdb,binder_length_min,binder_length_max,num_samples,hotspot_residues,target_chain` | `sample_id, pdb_output, sequence, rewards` |
| `gen_ligand_binders` | `proteina_complexa_ligand` | same cols, ligand PDB in `target_pdb` slot | `sample_id, pdb_output, sequence, rewards` |
| `scaffold_motif` | `proteina_complexa_ame` | same cols, motif PDB + `target_chain` (default B) | `sample_id, pdb_output, sequence, rewards` |
| `rfdiffusion_inpaint` | `rfdiffusion` | `dataframe_records [{pdb,start_idx,end_idx}]` | backbone PDB string(s) |
| `proteinmpnn` | `proteinmpnn` | `dataframe_records [{pdb,fixed_positions}]` | sequence strings |
| `esmfold` | `esmfold` | `inputs=[seq]` | PDB (pLDDT in B-factors) |
| `boltz_complex` | `boltz` | `inputs=[{input,msa,use_msa_server}]` | `pdb, iptm, protein_iptm, ptm, confidence_score, complex_plddt` |
| `diffdock` | `diffdock_esm_embeddings` + `diffdock` | `dataframe_split` | best `ligand_sdf`, `confidence` |
| `developability` | `netsolp` / `deepstabp` / `pltnum` / `mhcflurry` | `dataframe_records [{sequence,…}]` | `predicted_solubility` / `predicted_tm_celsius` / `predicted_stability` / `predicted_immuno_burden` |
| `chemprop_admet` | `chemprop_admet` / `chemprop_bbbp` / `chemprop_clintox` | `inputs=[smiles,…]` | per-model predictions |

## GWB-vs-literature deviations to state in the generated notebook

- **No target-conditioned RFdiffusion.** Proteina-Complexa Binder *is* the RFdiffusion+ProteinMPNN analog
  (joint backbone+sequence, hotspot-aware). GWB's `rfdiffusion` is motif-inpainting only.
- **AlphaFold2 is monomer-only** in GWB (no interface pAE) → use **Boltz-2** for the complex and ipTM as the
  interface filter.
- **No Rosetta ddG / InterfaceAnalyzer** → rank on ipTM + pLDDT (+ optional backbone RMSD via BioPython).
- **Developability are proxies**, not assays; PLTNUM is a 0–1 ranker, not hours.

## References

- Toolkit: `workshops/protein_design_from_paper/00_toolkit.py`
- Template: `workshops/protein_design_from_paper/TEMPLATE_protein_design_notebook.py`
- Worked example (binder_vs_protein): `workshops/snake_venom_binders/01_denovo_toxin_binder_workshop.py`
- Related skills: `SKILL_GENESIS_WORKBENCH_WORKFLOWS.md` (what each model does), `SKILL_GENESIS_WORKBENCH_BATCH_WORKFLOW_PATTERN.md` (the enzyme-optimization job).
