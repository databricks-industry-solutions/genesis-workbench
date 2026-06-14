# fevm-mmt-aws-usw2 — Deployment Changes

Workspace-local deploy log for `fevm-mmt-aws-usw2`. Mirrors the reusable items in root `CHANGELOG.md`, plus workspace-specific IDs/context.

## 2026-06-14 — Finish genomics, fix app SP, enable workspace gates, clean residual

**Goal:** complete the genomics module (was ⬜ partial on 06-13), fix the app "(service principal id not available)" bug, and clear residual from a killed deploy session.

- **App SP fix deployed.** Removed the hardcoded `DATABRICKS_APP_NAME: "genesis-workbench"` override in `modules/core/app/app.yml` (it shadowed the runtime's auto-injected real name `gwb-app`). Redeployed via `./deploy.sh core aws` (inside `conda activate gwb`); `gwb-app` redeployed ACTIVE. **Verified live:** Profile → MLflow Setup now shows SP `d1f5a1dd-5247-4461-b99b-ca6b71101049`. Committed `9037e48` (local, unpushed). See root `CHANGELOG.md` → Unreleased.
- **DCS (custom containers) enabled.** `databricks api patch /api/2.0/workspace-conf --json '{"enableDcs":"true"}'` (was `null`/off). Unblocked the parabricks all-purpose Docker cluster.
- **genomics ✅ fully deployed** (`./deploy.sh genomics aws`, then `--only-submodule parabricks/parabricks_v1` after DCS): gwas, vcf_ingestion, variant_annotation, parabricks. **All 4 `*_initial_setup` jobs TERMINATED SUCCESS** + genomics `initialize_module_job` SUCCESS. `variant_annotation_data` populated (ACMG `ACMG_SFv3.2_GRCh38.bed` + sample `brca_pathogenic_corrected.vcf` + ClinVar). Parabricks GPU cluster `0614-052453-6w7vyoku` + job `dbx_parabricks_initial_setup` (`34115058686293`).
- **parabricks target migration prod → prod_aws.** Source targets `prod_aws`; the stale `prod` remote-state target was empty (0 managed resources, never succeeded pre-DCS) → removed (`databricks workspace delete .../genesis_workbench_parabricks/prod --recursive`). No orphaned workspace resources.
- **AI/BI dashboard embedding enabled for the app.** Workspace default was `aibi-dashboard-embedding-access-policy = ALLOW_APPROVED_DOMAINS` with an **empty** approved list (so embeds blocked despite not being "Deny"). Added the app domain via `databricks settings aibi-dashboard-embedding-approved-domains update` → approved: `gwb-app-7474658466980277.aws.databricksapps.com`. Unblocks the admin-usage dashboard embed on the Monitoring page.
- **Residual cleanup (from the killed 06-13 session):** deleted 3 orphaned April bionemo jobs (superseded by the 06-13 redeploy; remote state tracks the June IDs `302357411181646`/`59899400130378`/`983769939153440`); removed ~2 GB of leftover pre-rename dirs (`disease_biology/`, `protein_studies/`, `parabricks/` — untracked cruft, 0 tracked files); cleared a stale `gwas` `deploy.lock` (`.../genesis_workbench_gwas/prod_aws/state/deploy.lock`, acquired 06-13 21:53 by a killed deploy); corrected the stale root `env.env` (catalog/schema/warehouse/app-name + plaintext docker tokens → secret-scope refs).
- **bionemo:** no redeploy needed — jobs are current (06-13). It only needed DCS to **run** (now enabled); re-run any job that previously failed on DCS-off.
- **`batch_models` registration cleanup.** The app's Registered Workflows showed phantom `Disease Biology` / `Protein Studies` rows (stale old-module-name registrations left by the rename) and a dangling bionemo job ref. Repaired via 3 `UPDATE`s on `mmt_aws_usw2.genesis_workbench.batch_models` (warehouse `dafc5dff8eb8094a`): `is_active=false` for `protein_studies/alphafold2` + `disease_biology/glow` (dups of the `large_molecule`/`genomics` rows, same job_ids); repoint `bionemo/esm2` `job_id 311891731048924 → 59899400130378` (the deleted April orphan → live finetune job). Active rows now: bionemo, genomics (Glow + Parabricks), large_molecule (AlphaFold2), single_cell (Rapids + Scanpy). See root `CHANGELOG.md` → Unreleased for the upstream gap.

### Status (modules) — as of 2026-06-14
- **core** ✅ ; **large_molecule** ✅ ; **small_molecule** ✅ ; **single_cell** ✅ ; **genomics** ✅ (gwas/vcf/variant_annotation/parabricks — all init jobs SUCCESS) ; **bionemo** ✅ deployed (runnable now DCS is on).
- **Open follow-ups:** push `9037e48`; guided-tour sibling app (scope via `app_names` multi-app pattern); still-open data-prep from 06-13 (`gene_sequences` via `ingest_uniprot_genes.py`; HGSOC `download_cellxgene` local-then-copy patch).

## 2026-06-13 — Redeploy from v2.1.0 main (reuse existing catalog)

**Goal:** bring usw2 up to v2.1.0 (`origin/main`, Vortex/AI Canvas, renamed modules) reusing the existing `mmt_aws_usw2.genesis_workbench` catalog so downloaded model weights aren't re-pulled.

- **Branch:** `mmt/usw2_main_redeploy` (off `origin/main`, f798ca1).
- **App:** UI renamed `gwb-mmt-sandbox` → **`gwb-app`** (`https://gwb-app-7474658466980277.aws.databricksapps.com`); MCP server app `mcp-genesis-workbench` (STOPPED). App SP = `d1f5a1dd-5247-4461-b99b-ca6b71101049`.
- **Config:** `core_catalog_name=mmt_aws_usw2`, `core_schema_name=genesis_workbench`, `sql_warehouse_id=dafc5dff8eb8094a` (Serverless Starter), `secret_scope_name=mmt`, `dev_user_prefix=mmt`, `llm_endpoint_name=databricks-claude-sonnet-4-6`.
- **Catalog reuse confirmed:** AlphaFold genetic-DB download job completed in **4.9 min** (skip, not a multi-hour re-pull) — the `if [ ! -d "$MODEL_VOLUME/datasets/<db>" ]` guards fired against the existing `alphafold` Volume. esmfold/boltz/esm2/scimilarity skip via existence checks / HF `cache_dir`.

### Fixes

- **`genmol` deploy failure (SP / app_name defaults).** See root `CHANGELOG.md` → Unreleased for the full root cause. usw2-specific fix applied to `modules/small_molecule/genmol/genmol_v1/variables.yml`:
  - `app_service_principal_id` default `bea46a71-…` → **`d1f5a1dd-5247-4461-b99b-ca6b71101049`** (gwb-app SP)
  - `app_name` default `genesis-workbench` → **`gwb-app`**

- **`scimilarity` redeploy tried to DELETE the weights Volume.** Cancelled the prompt with `n` (data safe). Restored `modules/single_cell/scimilarity/scimilarity_v0.4.0_weights_v1.1/resources/volumes.yml` (main had dropped it) → re-deploy via `--only-submodule scimilarity/...` adopted the existing volume in place, **no delete**. Volume `mmt_aws_usw2.genesis_workbench.scimilarity` (model_v1.1 + Adams 2020 IPF) intact. See CHANGELOG → Unreleased.

- **Sequence Search `embed_gene_sequences` FAILED — `gene_sequences` table missing.** `mmt_aws_usw2.genesis_workbench.gene_sequences` was never built (core's `ingest_uniprot_genes.py` is unwired). Protein search OK; gene companion index + gene-name resolution blocked. **TODO:** run `ingest_uniprot_genes.py` (catalog `mmt_aws_usw2`, schema `genesis_workbench`, organism_id `9606`), then re-run `embed_gene_sequences` + `create_gene_vector_index` (job `1110022555776750`). See CHANGELOG → Unreleased.

- **`download_cellxgene` HGSOC stage FAILED — Errno 95 writing to Volume.** `download_cellxgene.py:286` writes the `.h5ad` straight to `/Volumes/.../raw_h5ad/hgsoc_demo_15k.h5ad` (h5py can't write to the Volume FUSE). Dataset = MSK SPECTRUM HGSOC ~15k cells w/ PARP1/BRCA1/BRCA2 (great for a BRCA/ovarian journey). **Fix:** write to `/local_disk0` then `dbutils.fs.cp` to the Volume. See CHANGELOG → Unreleased. *(Existing Adams IPF data already supports basic SCimilarity demos; HGSOC is a journey nice-to-have.)*

### Deploy gotchas hit (reusable — see CHANGELOG for generic form)

- **Terraform state lineage mismatch.** This repo was used to deploy to 3 AWS workspaces (e2-demo-field-eng, usw2, fe-vm-hls-amer); DABs keeps one local state slot per target (`prod_aws`), so they collide. Fix: swap the saved usw2 state into the active slot before deploying — `mv prod_aws prod_aws.e2fe.bak && cp -R prod_aws.fevm-mmt-aws-usw2 prod_aws` (done for `core` + the unchanged-path submodules under `single_cell`/`small_molecule`). Renamed modules (`large_molecule`/`genomics`) are fresh paths → pull remote state cleanly, no swap. **Durable fix: a separate clone/worktree per workspace.**
- **Empty `dev_user_prefix` breaks secret-scope step.** `modules/core/deploy.sh` does `put-secret … dev_user_prefix --string-value "${dev_user_prefix:-}"`; an empty value is rejected by the CLI ("Secret value must be specified"). usw2's existing endpoints are named `gwb_mmt_<model>_endpoint`, so `dev_user_prefix=mmt` is both required and correct for in-place reuse.
- **`DATABRICKS_TF_VERSION` mismatch.** `deploy.sh` hardcoded `1.3.9`; local terraform is `1.14.9` — aligned to avoid the expired-PGP-key download path.
- **Profile selection.** Bundle target `prod_aws` has no `host:` → uses active CLI auth. Deploy with `export DATABRICKS_CONFIG_PROFILE=fevm-mmt-aws-usw2`.

### Status (modules)

- **core** ✅ ; **large_molecule** ✅ (8 submodules; endpoints READY) ; **small_molecule** ✅ (9 submodules incl. genmol fix) ; **single_cell** ✅ (5 submodules; scimilarity/scgpt endpoints READY, volume preserved) ; **genomics** ⬜ partial — parabricks/gwas need DCS+Docker ; **bionemo** ⬜ gated on DCS grant + Docker image.
- **Open data-prep follow-ups (non-blocking for model endpoints):** (1) build `gene_sequences` via `ingest_uniprot_genes.py` → re-run gene-embed/index; (2) patch + re-run `download_cellxgene` HGSOC stage (local-then-copy). Both feed the BRCA journey.
