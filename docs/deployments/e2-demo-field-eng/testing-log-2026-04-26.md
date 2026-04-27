# GWB MVP clickthrough testing log

**Tester:** may.merkletan@databricks.com
**Date started:** 2026-04-26
**App URL:** https://gwb-mmt-demo-1444828305810485.aws.databricksapps.com
**Workspace:** e2-demo-field-eng
**Branch / state:** `mmt/e2fe_gwb_deploy` post-reconstruction (warehouse `9b5370ee2ef1e248`, all 19 endpoints READY + canonical)

Goal: verify each module's primary workflow path on the live e2fe deploy. Total time budget ~10 min for full sweep, ~5 min if BioNeMo + Parabricks runs are skipped.

---

## 0. Sanity (~30 sec)

- [ ] App loads (no 500/error banner)
- [ ] Sidebar shows all module tabs (Protein Studies, Single Cell, Small Molecule, Disease Biology, BioNeMo, Parabricks)
- [ ] Settings tab → SQL Warehouse ID = `9b5370ee2ef1e248`, all module flags = ✓ deployed

## 1. Protein Studies (~2 min)

- [ ] Tab opens; subtabs visible (Structure Prediction / Protein Design / Inverse Folding / Sequence Search)
- [ ] **Structure Prediction** → paste short sequence (e.g. `MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK`) → select **ESMFold** → Run → ~5s → 3D structure renders in Mol* viewer

*Optional: Boltz / RFDiffusion / ProteinMPNN — skip unless time allows.*

## 2. Single Cell (~2 min)

- [ ] Tab opens; subtabs visible (Cell Type Annotation / Cell Similarity / Perturbation / Processing)
- [ ] **Cell Similarity** → SCimilarity → pick a small h5ad sample → Run gene_order endpoint → expect sorted gene list output

*Optional: scGPT Perturbation needs prior Processing run + scGPT endpoint — skip for MVP.*

## 3. Small Molecule (~2 min) — NEW MODULE today

- [ ] Tab opens; subtabs visible (ADMET / Ligand Binder / Binder Design / Motif Scaffolding)
- [ ] **ADMET & Safety** → paste SMILES (e.g. aspirin: `CC(=O)Oc1ccccc1C(=O)O`) → run Chemprop endpoints → expect BBBP / ClinTox / ADMET risk cards

## 4. Disease Biology (~2 min) — NEW MODULE today

- [ ] Tab opens; subtabs visible (VCF Ingestion / Variant Annotation / GWAS)
- [ ] **Variant Annotation** → use sample VCF or pre-ingested table → Run → expect ClinVar-annotated variants table; filter by gene = BRCA

## 5. BioNeMo (~30 sec — form check only)

- [ ] Tab opens; ESM2 Finetune + ESM2 Inference subtabs render
- [ ] Path defaults pre-populated (BLAT_ECOLX from init data — recent fix)
- [ ] Don't actually launch the job — just verify form renders and defaults work

## 6. Parabricks (~30 sec — form check only)

- [ ] Tab opens; Germline workflow renders
- [ ] Don't launch — just verify form and confirm `parabricks_cluster` reference exists

---

## Issues found

### Issue template (copy-paste per finding)

```
- TAB / WORKFLOW: <name>
- WHAT I CLICKED: <action>
- EXPECTED: <what should happen>
- ACTUAL: <what happened>
- SEVERITY: blocker / annoying / cosmetic
- SCREENSHOT: <filename in gwb_app_imgs/>
- NOTES: <freeform>
```

### Findings

<!-- paste new findings below this line as you go -->

---

## Run notes

<!-- freeform: app load times, ambient observations, things to follow up on -->
