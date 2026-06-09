# KERMT — Fine-tune + Serve an ADMET Model

## Introduction

**KERMT** (Kinetic GROVER Multi-Task) is an NVIDIA-BioNeMo graph neural network for small-molecule
property prediction (an enhanced GROVER reimplementation, Apache-2.0). Genesis Workbench lets you
**fine-tune** KERMT from the pretrained GROVERbase checkpoint on a SMILES + target dataset, write the
fine-tuned weights to Unity Catalog, and **deploy** the result as a real-time serving endpoint that the
**ADMET & Safety** tab calls **side-by-side with Chemprop** — a second, independent opinion on toxicity /
ADMET (the same pattern as TEDDY running alongside SCimilarity for cell-type annotation).

## What It Achieves

- Fine-tunes KERMT/GROVER on any SMILES + target table (classification or regression)
- Writes each fine-tuned model to the `kermt_weights` table + a UC volume; tracked in MLflow with
  Search Past Runs (`submitted → training → complete`)
- Deploys a chosen fine-tuned model as the `kermt_admet` serving endpoint
- Surfaces KERMT predictions next to Chemprop's in the ADMET & Safety tab for direct comparison
- Ships a bundled **TDC ClinTox** sample so the default fine-tune works out of the box

## How to Use

### Fine-tune (Small Molecule → **KERMT (ADMET fine-tune)**)

1. The train/val/test CSV paths are pre-filled with the bundled TDC **ClinTox** sample
   (`/Volumes/<catalog>/<schema>/kermt/ft_data/clintox_{train,val,test}.csv`). Each CSV has a `smiles`
   column plus one column per task. Point them at your own UC-volume CSVs to fine-tune on a different
   assay.
2. Set the **target column(s)**, **task type** (classification or regression), a **fine-tune label**,
   epochs / batch / FFN size, and the MLflow experiment + run name.
3. Click **Fine-tune KERMT** — a Databricks job runs on a classic A10 GPU cluster; the run appears
   immediately in **Search Past Runs** and advances to `complete` (test AUC/metric logged to MLflow).

### Deploy (same tab → **Deploy a fine-tuned model**)

4. Pick a completed fine-tuned model from the dropdown and click **Deploy**. This registers it as the
   `kermt_admet` UC model and (re)deploys the `gwb_*_kermt_admet_endpoint` serving endpoint. *(The GPU
   endpoint build takes ~30–60 min.)*

### Use in ADMET & Safety

5. In **ADMET & Safety**, the **KERMT** predictor toggle is on by default. Run profiling and each
   molecule card shows **Toxicity (KERMT)** next to **Toxicity (Chemprop)** for a side-by-side read.

### Inputs

| Field | Description | Default |
|-------|-------------|---------|
| Train / Validation / Test CSV | UC-volume CSVs with a `smiles` column + task column(s) | bundled TDC ClinTox |
| Target column(s) | Comma-separated task names (the non-`smiles` columns) | `toxicity` |
| Task type | `classification` or `regression` | classification |
| Epochs / Batch / FFN hidden | Fine-tune hyper-parameters | 20 / 16 / 700 |

### Outputs

- A row in `kermt_weights` (`ft_id`, label, dataset_type, task_names, run_id, checkpoint location)
- The fine-tuned checkpoint under `/Volumes/<catalog>/<schema>/kermt/finetuned/<label>/`
- An MLflow run (tag `feature=kermt_finetune`) with the test metric
- After deploy: the `kermt_admet` serving endpoint, queryable via the ADMET tab

## How It's Implemented

```
GROVERbase (UC volume)
      │  Fine-tune job (classic A10, pip-only — KERMT on the RDKit featurization path)
      ▼
fine-tuned checkpoint → /Volumes/.../kermt/finetuned/<label>/  +  kermt_weights row  +  MLflow run
      │  Deploy job (T4): wrap checkpoint in an MLflow PyFunc (in-process predict, plain RDKit)
      ▼
UC model kermt_admet → deploy_model → gwb_*_kermt_admet_endpoint (GPU)
      │  inputs=[smiles…] → predictions=[{task: value}…]  (Chemprop ADMET contract)
      ▼
ADMET & Safety tab queries it side-by-side with Chemprop
```

**Why pip-only / no container, and the cuik-molmaker note.** KERMT is installed from a vendored, pinned
copy of the repo on a classic GPU cluster (the ChemProp pattern — no custom Docker). KERMT's
`cuik_molmaker` accelerator is conda-only and hard-imported at module top; a small **lazy-import patch**
guards those imports so KERMT runs on the **plain-RDKit featurization path** (`rdkit_2d_normalized_onthefly`)
with a pip-only env. This keeps both the fine-tune job and — critically — the Model Serving endpoint env
buildable. Datasets with salts/mixtures (e.g. ClinTox) can yield NaN RDKit descriptors; the vendored
collator sanitizes them (`np.nan_to_num`) so they don't propagate to NaN model outputs.

### Key Files

- `modules/small_molecule/kermt/kermt_v1/kermt_src/` — vendored KERMT (pinned commit) + lazy-import/NaN patches
- `…/notebooks/01_register_kermt.py` — stage GROVERbase + `kermt_weights` table + TDC sample
- `…/notebooks/02_kermt_finetune.py` — fine-tune orchestrator (MLflow status, writes weights)
- `…/notebooks/03_kermt_register_serving.py` — PyFunc + register + deploy endpoint
- `modules/core/app/backend/app/services/kermt.py` — dispatch + search + list-weights
- `modules/core/app/backend/app/services/admet_safety.py` — `predict_kermt`
- `modules/core/app/frontend/src/components/KermtFinetuneTab.tsx` + `AdmetSafetyTab.tsx`

### Dependencies

- KERMT (NVIDIA-BioNeMo/KERMT, Apache-2.0) — pretrained GROVERbase checkpoint (pre-staged to the
  `kermt` UC volume)
- Classic A10 GPU compute for fine-tune; T4 for deploy; a GPU serving endpoint (`GPU_SMALL`) for inference
