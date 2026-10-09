# RFD4-Proteina Design

## Introduction

RFD4-Proteina is the NVIDIA×Baker flow-matching protein-design foundation model, served in Genesis
Workbench as a real-time endpoint. It generates novel protein backbones (and their sequences) from a
length-only de-novo prompt or, in first-draft form, from a target structure with a contig + hotspot
conditioning spec. It is deployed on a serverless **GPU_XLARGE (1× H100)** endpoint via MLflow
**Express deployments**, which is what makes the torch-2.14 / CUDA-13 model servable without the
standard serving container rebuild.

## What It Achieves

- De-novo monomer design: generate `num_samples` novel backbones of a requested length, returning each
  as a PDB plus its designed sequence (**confirmed end-to-end**)
- First-draft target-conditioned design (binder / motif): condition generation on an uploaded target
  structure, a contig, and hotspot residues — refined by the forthcoming design workflows
- Serves a Proteina-Complexa-compatible contract (plus a `task` column), so it is a drop-in for the
  existing design surfaces
- Optional LoRA fine-tuning: adapt the base flow model on your own structures and deploy the fine-tuned
  version onto the same endpoint

## How to Use

RFD4-Proteina ships as a **served model/endpoint** (not yet a dedicated UI tab — the guided design
workflows that drive it from the UI are in progress). Until then it is queried directly:

1. From the **AI Playground** or the **MCP server** (`mcp-genesis-workbench`), select the deployed
   endpoint `gwb_<prefix>_rfd4_proteina_endpoint` (e.g. `gwb_demo_rfd4_proteina_endpoint`) and send a
   record matching the input schema below.
2. Or query it over the serving REST API / SDK, e.g.:

   ```bash
   databricks serving-endpoints query gwb_demo_rfd4_proteina_endpoint \
     --json '{"dataframe_records":[{"task":"de_novo","target_pdb":"","binder_length_min":60,
              "binder_length_max":60,"num_samples":1,"hotspot_residues":"","target_chain":"A"}]}'
   ```

De-novo generation on an H100 returns within a request; larger `num_samples` and target-conditioned
runs take proportionally longer.

### Inputs

| Field | Description | Example |
|-------|-------------|---------|
| task | `de_novo` (length-only monomer) or `binder` / `motif` (target-conditioned, first-draft) | `de_novo` |
| target_pdb | Target structure as PDB text (binder/motif only; empty ⇒ de novo) | PDB string |
| binder_length_min | Minimum designed-chain length | `60` |
| binder_length_max | Maximum designed-chain length | `60` |
| num_samples | Number of designs to generate | `1` |
| hotspot_residues | Optional comma-separated target residues to bind (binder/motif) | `R45,E78,K112` |
| target_chain | Chain ID on the target to condition against | `A` |

### Outputs

A row per generated design:

- **sample_id** — index of the design within the request
- **pdb_output** — the designed structure as PDB text (falls back to mmCIF if PDB column widths overflow)
- **sequence** — the designed amino-acid sequence
- **rewards** — reserved for scoring (populated by the design workflows, not the raw endpoint)

## How It's Implemented

### Pipeline

```
Request row (task, target_pdb, lengths, num_samples, hotspots, chain)
  ↓
de_novo → bundled length-only monomer CIF    │ binder/motif → condition_spec (contig + C_CRD/C_SEQ/C_HOT)
  ↓
rfproteina get_contig_or_design_problem_dataset → collate_batch(ProteinaDataSample) → predict_step (warm, H100)
  ↓
generated AtomArray → PDB/mmCIF + sequence
```

The model is driven through rfproteina's own supported inference path (Hydra-composed inference config →
`load_ckpt_n_configure_inference` → `build_design_validation_pipeline` → `predict_step`), loaded once and
kept warm. Serving uses Express **env_pack**: the serverless-GPU/CUDA-13 register environment
(torch 2.14.1/cu132 + cuEquivariance-cu13) is packaged at registration and restored verbatim at serving,
so there is no deploy-time container rebuild and no torch force-install.

### Key Files

- `modules/large_molecule/rfd4_proteina/rfd4_proteina_v1/notebooks/01_register_rfd4_proteina.py` — register
  base model + Express-deploy the endpoint
- `modules/large_molecule/rfd4_proteina/rfd4_proteina_v1/notebooks/02_rfd4_finetune.py` — LoRA fine-tune
  (writes the `rfd4_weights` table)
- `modules/large_molecule/rfd4_proteina/rfd4_proteina_v1/notebooks/03_rfd4_register_serving.py` — register +
  Express-deploy a fine-tuned (base + LoRA adapter) version onto the same endpoint

### Underlying models / endpoints

- **Endpoint:** `gwb_<prefix>_rfd4_proteina_endpoint` — GPU_XLARGE (1× H100), Express env_pack, no scale-to-zero
- **UC model:** `<catalog>.<schema>.rfd4_proteina` (base + fine-tuned versions)
- **Source:** `rfproteina` pip-installed at job runtime from the NVIDIA-BioNeMo RFD4-Proteina repo (nothing
  vendored); checkpoints from the public CDN. The register / fine-tune / deploy jobs run on serverless `GPU_1xH100`.

## Limitations and known issues

- **De-novo is the confirmed path.** Binder / motif target-conditioning is first-draft (the contig and
  hotspot `select` construction is refined by the forthcoming guided design workflows).
- **No dedicated UI tab yet** — the model is reachable via the serving API, the AI Playground, and the MCP
  server; the guided antibody / design workflows that drive it from the UI are in progress.
- **GPU_XLARGE (1× H100)** is the proven serving tier (account-team enrolled, `us-west-2`); cheaper CUDA-13
  tiers may also work via env parity but are untested. The endpoint has scale-to-zero disabled.
