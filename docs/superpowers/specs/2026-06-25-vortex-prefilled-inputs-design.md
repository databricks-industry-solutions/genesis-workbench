# Vortex prefilled demo inputs — design (MVP)

**Date:** 2026-06-25 · **Branch:** `feat/gwb-demo-ux` (off `mmt/usw2_main_redeploy`)

## Problem

Every prebuilt-workflow node dropped on the Vortex canvas loads with **empty required inputs**, so demos hit "N issues to fix before running" (e.g. FASTQ R1/R2 empty, sequence empty, AnnData payload empty). A presenter must hand-type UC paths before anything runs. Goal: prebuilt workflows load **runnable** with usable example inputs (presenter can still edit), so end-to-end canvas demos work out of the box.

## Scope (MVP)

Prefill demo-usable values for the required **source** inputs of prebuilt-workflow + endpoint nodes. Two follow-ups are explicitly **out of MVP** (same branch, after): the curated-`examples` dropdown + live "Browse Volume" endpoint (Approach 3), and the collapsible nav.

All example files are **verified present** on `fevm-mmt-aws-usw2` under `/Volumes/mmt_aws_usw2/genesis_workbench/` (2026-06-25). Paths are stored templated as `{catalog}/{schema}` and substituted client-side from `bootstrap.env`, so this is portable (works upstream / any workspace whose volumes follow the same layout).

### Input → example mapping (verified)

Node `type` (label) · input port `name`:`dtype` → example:

| Node | Input(s) | Example (templated) |
|---|---|---|
| `variant_calling` (Variant Calling) | `fastq_r1`:PATH, `fastq_r2`:PATH | `…/gwas_data/sample_fastq/sample_1.fq.gz`, `…/sample_2.fq.gz` |
| `vcf_ingestion` (VCF Ingestion) | `vcf`:PATH | `…/gwas_data/sample_vcf/ALL.chr6.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz` |
| `gwas` (GWAS) | `vcf`:PATH, `phenotype`:PATH | sample VCF (above), `…/gwas_data/sample_phenotype/breast_cancer_phenotype.tsv` |
| `esm2_finetune` (Fine-Tune ESM2) | `train_data`:PATH, `evaluation_data`:PATH | `…/bionemo/esm2/ft_data/BLAT_ECOLX_Tenaillon2013_metadata_train.csv`, `…_eval.csv` |
| `kermt_finetune` (Fine-Tune KERMT) | `train_data`,`validation_data`,`test_data`:PATH | `…/kermt/ft_data/clintox_train.csv`, `…_val.csv`, `…_test.csv` |
| `alphafold2` (AlphaFold), `pltnum` (PLTNUM), `netsolp`, `deepstabp`, `esm2_embeddings`, `esmfold`, `boltz` | `sequence`:SEQUENCE | a short literal demo protein sequence (constant) |
| `admet_screen`, `chemprop_*`, `kermt_admet` | `smiles`:SMILES | a literal demo SMILES (constant, e.g. aspirin) |
| `scimilarity_get_embedding` (SCimilarity), `scgpt_*`, `teddy` | `cells`:JSON (AnnData payload) | **prebaked demo payload** — see "AnnData payload helper" |

**Wired-from-upstream, NOT prefilled** (dtype is a pipeline product, not a source): `variant_annotation.table` (TABLE — output of VCF Ingestion), `vcf_ingestion`-downstream tables, ProteinMPNN `pdb`, etc. These are satisfied by wiring the chain (Variant Calling → VCF Ingestion → Variant Annotation). The spec does not prefill them; the genomics demo runs as a wired chain whose only source input (FASTQ) is prefilled.

## Design

Follows the existing node-catalog flow: `builtin_nodes.py` → `publish_node_catalog` → `node_catalog` Delta table → app reads → frontend renders.

### 1. Model — `genesis_workbench/node_catalog.py`
Add one optional field to `Port`:
- `example: str | None = None` — the prefill default (templated path, literal sequence/SMILES, or — for `cells` — a `/Volumes/...json` path to the prebaked payload).

Update `_serialize_port` (add `"example": p.example`) and `port_from_dict` (read `d.get("example")`) so it round-trips into the `node_catalog` table. No change to `required`/dtype semantics.

### 2. Data — `builtin_nodes.py`
Set `example=` on each source input port per the mapping table. Use `{catalog}`/`{schema}` tokens for volume paths. Sequence/SMILES examples are short module-appropriate constants defined once.

### 3. AnnData payload helper (for `cells` JSON inputs)
The `cells` payload is a *processed* artifact (`scimilarity.py:69-71`: `[{"celltype_sample": <expr_df.to_json(orient="split")>, "celltype_sample_obs": <obs json>}]`), built via gene-order-align + lognorm + subsample — not a raw h5ad dump. So:
- **Prebake once:** a one-time script (reusing `scimilarity.py` align/lognorm/subsample) materializes a small (~tens of cells) canonical payload from an existing `…/raw_h5ad/*.h5ad`, written to `…/ai_canvas_demo/scimilarity_cells_demo.json`.
- **Executor reads path-as-JSON:** extend the canvas executor so a JSON-dtype input whose value is a `/Volumes/….json` path loads + parses the file at run time (small, deterministic, no per-run processing). (Confirm exact executor input-coercion site during planning — `executor.py`.)
- **Prefill:** `scimilarity_get_embedding.cells.example` = the demo-payload path. Same payload shape works for `scgpt_*`/`teddy` (verify each endpoint's expected payload during planning; if they differ, prebake one per shape).

### 4. Frontend — Vortex inputs panel
For a required input with an `example`: seed the input field with `example`, substituting `{catalog}`/`{schema}` from `bootstrap.env`. The presenter can edit/clear it. Existing validation passes once a value is present → "issues to fix" clears. (No dropdown/browse in MVP.)

### 5. Publish
Re-run `publish_node_catalog` so the table carries `example`. App reads it unchanged otherwise.

## Portability
Paths are `{catalog}/{schema}`-templated and substituted client-side. Sequence/SMILES are workspace-agnostic constants. The prebaked payload path is also templated. Works on usw2 now and upstream unchanged.

## Error handling
- A prefilled path that's missing/edited-wrong just fails at run with the existing node error — prefill is best-effort, never blocks editing.
- Frontend substitution: if `bootstrap.env` lacks catalog/schema, leave the `{…}` token visible (presenter sees it needs filling) rather than render a broken path.
- Prebake script: if the chosen h5ad is absent, fail loudly with the path so it's staged.

## Testing
- **Unit:** `Port` serialize/deserialize round-trips `example`; `node_catalog` republish includes it; `{catalog}/{schema}` substitution helper.
- **Executor:** JSON input given a `/Volumes/*.json` path loads+parses; non-path JSON still passes through.
- **Manual (usw2):** load each prebuilt workflow → source inputs prefilled, "issues" cleared; run Variant Calling→VCF Ingestion→Variant Annotation chain, GWAS, a Fine-Tune, AlphaFold, and SCimilarity end-to-end.

## Out of scope (fast-follow, same branch)
- Curated `examples` list + dropdown + live "Browse Volume" backend endpoint (Approach 3).
- Collapsible nav (`Layout.tsx`).
- Guided demo (separate project: end-to-end R&D, canvas-centric, per-customer-use-case tracks).
