# Demo example inputs — shared registry (MVP)

**Date:** 2026-06-25 · **Branch:** `feat/gwb-demo-ux` (off `mmt/usw2_main_redeploy`)

## Problem

Two surfaces let a user run GWB workflows: the **Vortex canvas** (composed node graphs) and the **per-module walkthrough tabs** (GWAS, KERMT, single-cell, …). Today:

- **Module tabs** prefill file inputs from per-module `GET /defaults` endpoints (`genomics`, `kermt`, `bionemo`, `large_molecule/enzyme_optimization`), each **hardcoding** its own `/Volumes/{catalog}/{schema}/…` paths.
- **The canvas does not prefill at all** — every prebuilt-workflow node loads with empty required inputs → "N issues to fix before running."

So example paths are **duplicated and drifting** across module routers, and the canvas is unusable for a quick demo. We want **one source of truth** for example inputs that both surfaces read, and that a future per-disease scenario overlay can override in one place.

## Scope (MVP)

Build a **shared example-inputs registry** (the SSOT), make the canvas + the existing module `/defaults` both read it, and prefill the canvas. **Out of MVP** (fast-follow, same branch unless noted): curated `examples` dropdown + live "Browse Volume" endpoint; `scgpt_perturbation` payload; collapsible nav; and the scenario-overlay/disease-thread system (separate guided-demo project — this spec only leaves the seam for it).

All example files are **verified present** on `fevm-mmt-aws-usw2` under `/Volumes/mmt_aws_usw2/genesis_workbench/` (2026-06-25); the MVP baseline is intentionally scoped to these already-present inputs (no staging).

## Architecture

`demo_inputs` registry (shared lib) → consumed by **(a)** `publish_node_catalog` (sets `Port.example`) and **(b)** the module `/defaults` endpoints → app reads → frontend prefills both surfaces. Paths templated `{catalog}/{schema}`, substituted with the app's catalog/schema.

### 1. Shared registry — `genesis_workbench/demo_inputs.py` (new, in the wheel)

A single mapping from a stable input key to an example spec. Key = `"{node_type}.{port}"` (canvas vocabulary; module tabs reference the same keys). Importable by both FastAPI routers and the publish notebook (same wheel the app + executor already use).

```python
# key -> example value (templated path | literal sequence/SMILES | /Volumes/*.json payload path)
DEMO_INPUTS: dict[str, str] = {
    "variant_calling.fastq_r1": "/Volumes/{catalog}/{schema}/gwas_data/sample_fastq/sample_1.fq.gz",
    "variant_calling.fastq_r2": "/Volumes/{catalog}/{schema}/gwas_data/sample_fastq/sample_2.fq.gz",
    "gwas.vcf":        "/Volumes/{catalog}/{schema}/gwas_data/sample_vcf/ALL.chr6.…GRCh38.phased.vcf.gz",
    "gwas.phenotype":  "/Volumes/{catalog}/{schema}/gwas_data/sample_phenotype/breast_cancer_phenotype.tsv",
    "vcf_ingestion.vcf": "…/gwas_data/sample_vcf/…vcf.gz",
    "kermt_finetune.train_data": "…/kermt/ft_data/clintox_train.csv",   # + val/test
    "esm2_finetune.train_data":  "…/bionemo/esm2/ft_data/BLAT_ECOLX_…_train.csv",  # + eval
    "alphafold2.sequence": "<short demo protein sequence>",   # also pltnum/netsolp/esmfold/… sequence
    "admet_screen.smiles": "<demo SMILES>",                    # also chemprop_*/kermt_admet
    "scimilarity_get_embedding.cells": "/Volumes/{catalog}/{schema}/ai_canvas_demo/scimilarity_cells_demo.json",
    "scgpt_embeddings.cells": "/Volumes/{catalog}/{schema}/ai_canvas_demo/scgpt_cells_demo.json",  # + teddy
}

def example_for(key: str, catalog: str, schema: str) -> str | None:
    v = DEMO_INPUTS.get(key)
    return v.format(catalog=catalog, schema=schema) if v else None
```

Resolution helper keeps `{catalog}/{schema}` substitution in one place. (Sequence/SMILES literals defined once as module constants.)

### 2. Canvas consumes the registry

- `node_catalog.Port` gains `example: str | None` (round-tripped via `_serialize_port` / `port_from_dict`).
- `publish_node_catalog` sets each port's `example` from `DEMO_INPUTS["{node_type}.{port}"]` (left templated; substituted client-side). Re-publish so the table carries it.
- Frontend (Vortex inputs panel): seed a required input's field with its `example`, substituting `{catalog}/{schema}` from `bootstrap.env`. Editable; validation passes once a value is present → "issues" clears.

### 3. Module `/defaults` endpoints read the registry

Refactor `genomics`/`kermt`/`bionemo`/`large_molecule` `/defaults` to build their responses from `demo_inputs.example_for(...)` (backend substitutes catalog/schema) instead of hardcoded literals. Response shapes unchanged → no frontend-tab changes. Removes the duplication; both surfaces now resolve identical paths.

### 4. AnnData payload helper (`cells` JSON inputs)

`cells` payloads are *processed* artifacts and **differ by endpoint** (verified `capabilities.py`):

- **SCimilarity** (`scimilarity_get_embedding`, line 215): `celltype_sample:json` (+ param `celltype_sample_obs`); built by `scimilarity.py:69-71` via gene-order-align + lognorm + subsample.
- **`_SCGPT_EMB`** (`scgpt_embeddings`, `teddy`, line 174/210-211): "anndata-style sparse matrix + obs/var (real genes)" — different shape.

MVP prebakes **two** small demo payloads (subsampled, via the app's existing builder per shape) from an existing `…/raw_h5ad/*.h5ad`, written to `…/ai_canvas_demo/{scimilarity,scgpt}_cells_demo.json` (one-time script; locate exact builders in planning — `scimilarity.py`, and the scgpt/teddy builder in `single_cell.py`/`endpoints.py`). `scgpt_perturbation` (third shape) is out of MVP.

**Executor reads path-as-JSON** — mirror `_resolve_pdb_content` (`executor.py:354`: `/Volumes/` → `w.files.download(v).contents.read()`, else literal) with `_resolve_json_content` (download + `json.loads`), applied to JSON inputs where inputs are assembled before `_query_endpoint` (line 92). Non-path JSON passes through.

## Relationship to the guided demo (layering — forward-compat)

- **Baseline (this MVP):** `DEMO_INPUTS` registry = a runnable default per input. Disconnected across modules, but every workflow (canvas + tabs) loads runnable from one source.
- **Scenario overlay (guided-demo project):** a *named* scenario (e.g. `breast_cancer`) **overrides** registry entries to tell one coherent R&D story across BOTH surfaces (variant → gene/protein → structure → single-cell context → candidate molecule → ADMET).
- **Scenario data provisioning:** a scenario references themed inputs that **may not exist in UC** (BRCA seq, breast-tumor h5ad, PARP-inhibitor SMILES, chr-correct GWAS). So a scenario carries a **data contract** — declared assets + *stage-if-missing* (like module `initialize` jobs) + activation gated on availability. Built in the guided-demo, not here.
- **Additional use-cases** = more scenario overlays via the same registry — adding a use-case is adding a scenario.

**Resolution precedence:** active scenario overlay → `DEMO_INPUTS` baseline → empty. MVP requirement: backend `/defaults` and the frontend canvas-prefill each resolve through a **single function** so the overlay slots in ahead of the baseline later without rework. MVP does **not** build the overlay/scenario/provisioning system.

## Portability

Registry stores `{catalog}/{schema}` tokens; resolved with the app's catalog/schema (backend for `/defaults`, client for canvas). Sequence/SMILES are workspace-agnostic constants. Works on usw2 now and upstream unchanged.

## Error handling

- A prefilled path that's missing/edited-wrong fails at run with the existing node error — prefill is best-effort, never blocks editing.
- Missing catalog/schema in `bootstrap.env` → leave the `{…}` token visible rather than render a broken path.
- Prebake script fails loudly with the path if the chosen h5ad is absent.

## Testing

- **Unit:** `Port` round-trips `example`; `example_for` substitution; `node_catalog` republish includes example; module `/defaults` resolve to the same paths the canvas does (parity test against the registry).
- **Executor:** JSON input given `/Volumes/*.json` loads + parses; non-path JSON passes through.
- **Manual (usw2):** each prebuilt workflow loads prefilled, "issues" cleared; module tabs still prefill (unchanged shapes); run the genomics chain, GWAS, a Fine-Tune, AlphaFold, SCimilarity, scGPT/TEDDY end-to-end.

## Out of scope (fast-follow, same branch unless noted)

- Curated `examples` list + dropdown + live "Browse Volume" endpoint.
- `scgpt_perturbation` `cells` payload (third shape).
- Collapsible nav (`Layout.tsx`).
- Guided demo — **separate project**: scenario overlays + provisioning + walkthrough + per-customer-use-case tracks, built on this registry.
