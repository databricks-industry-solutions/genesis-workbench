# Vortex prefilled demo inputs — design (MVP)

**Date:** 2026-06-25 · **Branch:** `feat/gwb-demo-ux` (off `mmt/usw2_main_redeploy`)

## Problem

Every prebuilt-workflow node dropped on the Vortex canvas loads with **empty required inputs**, so demos hit "N issues to fix before running" (FASTQ R1/R2 empty, sequence empty, AnnData payload empty). A presenter must hand-type UC paths before anything runs. Goal: prebuilt workflows load **runnable** with usable example inputs (presenter can still edit), so end-to-end canvas demos work out of the box.

## Scope (MVP)

Prefill demo-usable values for the required **source** inputs of prebuilt-workflow + endpoint nodes. Explicitly **out of MVP** (same branch, fast-follow): the curated-`examples` dropdown + live "Browse Volume" endpoint (Approach 3); the collapsible nav; and the scGPT-perturbation `cells` payload.

All example files are **verified present** on `fevm-mmt-aws-usw2` under `/Volumes/mmt_aws_usw2/genesis_workbench/` (2026-06-25). Paths are stored templated as `{catalog}/{schema}` and substituted client-side from `bootstrap.env`, so this is portable (works upstream / any workspace following the same volume layout).

### Input → example mapping (verified against builtin_nodes.py)

| Node `type` (label) | Input port `name`:`dtype` | Example (templated) |
| --- | --- | --- |
| `variant_calling` (Variant Calling) | `fastq_r1`:PATH, `fastq_r2`:PATH | `…/gwas_data/sample_fastq/sample_1.fq.gz`, `…/sample_2.fq.gz` |
| `vcf_ingestion` (VCF Ingestion) | `vcf`:PATH | `…/gwas_data/sample_vcf/ALL.chr6.…GRCh38.phased.vcf.gz` |
| `gwas` (GWAS) | `vcf`:PATH, `phenotype`:PATH | sample VCF (above), `…/gwas_data/sample_phenotype/breast_cancer_phenotype.tsv` |
| `esm2_finetune` (Fine-Tune ESM2) | `train_data`:PATH, `evaluation_data`:PATH | `…/bionemo/esm2/ft_data/BLAT_ECOLX_…_train.csv`, `…_eval.csv` |
| `kermt_finetune` (Fine-Tune KERMT) | `train_data`,`validation_data`,`test_data`:PATH | `…/kermt/ft_data/clintox_train.csv`, `…_val.csv`, `…_test.csv` |
| `alphafold2`, `pltnum`, `netsolp`, `deepstabp`, `esm2_embeddings`, `esmfold`, `boltz` | `sequence`:SEQUENCE | a short literal demo protein sequence (constant) |
| `admet_screen`, `chemprop_*`, `kermt_admet` | `smiles`:SMILES | a literal demo SMILES (constant) |
| `scimilarity_get_embedding` (SCimilarity) | `cells`:JSON | prebaked **SCimilarity** payload path (see §3) |
| `scgpt_embeddings`, `teddy` | `cells`:JSON | prebaked **`_SCGPT_EMB`** payload path (see §3) |

**Wired-from-upstream, NOT prefilled** (dtype is a pipeline product, not a source): `variant_annotation.table` (TABLE — output of VCF Ingestion), ProteinMPNN `pdb`, etc. The genomics demo runs as a wired chain (Variant Calling → VCF Ingestion → Variant Annotation) whose only source input (FASTQ) is prefilled.

## Design

Follows the existing node-catalog flow: `builtin_nodes.py` → `publish_node_catalog` → `node_catalog` Delta table → app reads → frontend renders.

### 1. Model — `genesis_workbench/node_catalog.py`

Add one optional field to `Port`:

- `example: str | None = None` — prefill default (templated path, literal sequence/SMILES, or a `/Volumes/*.json` path to a prebaked payload).

Update `_serialize_port` (`"example": p.example`) and `port_from_dict` (`d.get("example")`) so it round-trips into the `node_catalog` table. No change to `required`/dtype semantics.

### 2. Data — `builtin_nodes.py`

Set `example=` on each source input port per the mapping table. Use `{catalog}`/`{schema}` tokens for volume paths. Sequence/SMILES examples are short constants defined once.

### 3. AnnData payload helper (`cells` JSON inputs)

The `cells` payload is a *processed* artifact, and **the shapes differ by endpoint** (verified `capabilities.py`):

- **SCimilarity** (`scimilarity_get_embedding`, line 215): `celltype_sample:json` (+ param `celltype_sample_obs`). Built by `scimilarity.py:69-71` (`[{"celltype_sample": <expr_df.to_json(orient="split")>, "celltype_sample_obs": <obs json>}]`) via gene-order-align + lognorm + subsample.
- **`_SCGPT_EMB`** (`scgpt_embeddings`, `teddy`, line 174/210-211): "anndata-style sparse matrix + obs/var (real genes needed)" — a different shape.

So MVP prebakes **two** demo payloads, each via the app's existing builder for that shape, from a chosen `…/raw_h5ad/*.h5ad`, subsampled small (~tens of cells), written to:

- `…/ai_canvas_demo/scimilarity_cells_demo.json`
- `…/ai_canvas_demo/scgpt_cells_demo.json`

A one-time prebake script (locate the exact builders in planning: `scimilarity.py` for the first; the scgpt/teddy payload builder in `single_cell.py`/`endpoints.py` for the second) generates both. `scgpt_perturbation` (yet another shape: `expression + gene_names + genes_to_perturb`) is **out of MVP**.

**Executor reads path-as-JSON** — mirror the existing `_resolve_pdb_content` (`executor.py:354`: `if value.startswith("/Volumes/"): w.files.download(value).contents.read()` else literal). Add `_resolve_json_content` that downloads + `json.loads` a `/Volumes/*.json` value, applied to JSON-dtype inputs where inputs are assembled before `_query_endpoint` (line 92). Non-path JSON still passes through unchanged.

**Prefill:** `scimilarity_get_embedding.cells.example` and `scgpt_embeddings/teddy.cells.example` = their respective demo-payload paths.

### 4. Frontend — Vortex inputs panel

For a required input with an `example`: seed the input field with `example`, substituting `{catalog}`/`{schema}` from `bootstrap.env`. The presenter can edit/clear it. Existing validation passes once a value is present → "issues to fix" clears. (No dropdown/browse in MVP.)

### 5. Publish

Re-run `publish_node_catalog` so the table carries `example`.

## Portability

Paths are `{catalog}/{schema}`-templated and substituted client-side. Sequence/SMILES are workspace-agnostic constants. The prebaked-payload paths are also templated. Works on usw2 now and upstream unchanged.

## Error handling

- A prefilled path that's missing/edited-wrong fails at run with the existing node error — prefill is best-effort, never blocks editing.
- Frontend substitution: if `bootstrap.env` lacks catalog/schema, leave the `{…}` token visible rather than render a broken path.
- Prebake script: if the chosen h5ad is absent, fail loudly with the path so it can be staged.

## Testing

- **Unit:** `Port` serialize/deserialize round-trips `example`; `node_catalog` republish includes it; `{catalog}/{schema}` substitution helper.
- **Executor:** a JSON input given a `/Volumes/*.json` path loads + parses; non-path JSON passes through.
- **Manual (usw2):** load each prebuilt workflow → source inputs prefilled, "issues" cleared; run the genomics chain, GWAS, a Fine-Tune, AlphaFold, SCimilarity, and scGPT/TEDDY end-to-end.

## Relationship to the guided demo (layering — forward-compat)

The MVP is the **baseline** layer; it must not preclude the guided demo's per-use-case tracks:

- **Baseline (this MVP):** `Port.example` = a runnable default per input (today's module-local real files). Disconnected across modules, but every workflow loads runnable.
- **Scenario overlay (guided-demo project):** a *named* scenario (e.g. `breast_cancer`) maps node/port → themed input and **overrides** the baseline when active — turning the canvas into one coherent R&D story (variant → gene/protein → structure → single-cell context → candidate molecule → ADMET). This is where narrative + data-correctness curation lives (e.g. chr-correct GWAS, BRCA protein seq, breast-tumor h5ad, PARP-inhibitor SMILES).
- **Additional use-cases:** each new disease/customer = **another scenario overlay** via the same mechanism — adding a use-case is adding a scenario, not re-plumbing.

**Resolution precedence:** active scenario overlay → `Port.example` baseline → empty. MVP requirement: keep the frontend value-resolution a single function so the overlay slots in ahead of the baseline later without rework. MVP does **not** build the overlay/scenario system — only the baseline + an overlay-ready resolution seam.

## Out of scope (fast-follow, same branch unless noted)

- Curated `examples` list + dropdown + live "Browse Volume" backend endpoint (Approach 3).
- `scgpt_perturbation` `cells` payload (third shape).
- Collapsible nav (`Layout.tsx`).
- Guided demo — **separate project**: end-to-end R&D, canvas-centric, per-customer-use-case tracks.
