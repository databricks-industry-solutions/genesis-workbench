# Sandbox Deploy Notes — fevm-mmt-aws-usw2

Workspace: `fevm-mmt-aws-usw2.cloud.databricks.com` (o=7474658466980277)
Catalog: `mmt_aws_usw2` · Region: AWS us-west-2
Branch: `version_pinning`

## Deadline

Mon **2026-04-27** — SA HUNTER internal enablement session.

## Decisions

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-20 | Deploy to sandbox workspace rather than fe-vm-hls-amer | Preserves paused Merck demo; full authority over naming/paths |
| 2026-04-20 | **Path B for Docker creds: secret-scope refs from day 1** | Do it right on the sandbox; fix the plaintext gap; become the reference for a future improvement PR back to `version_pinning` |
| 2026-04-20 | Clone Srijit's BioNeMo + Parabricks images to May's `mmtdb` Docker Hub namespace | Independence from Srijit's repo access; May already owns `mmtdb`; backup if Srijit rotates or revokes |
| 2026-04-21 | **Defer SP-based deploy until post-Monday** | 3-8 hr cost for marginal narrative benefit; threatens playbook time. Documented migration path in playbook instead. |
| 2026-04-21 | **Target catalog: `mmt_aws_usw2`** (not `mmt_aws_usw2_catalog`) | Dedicated external S3 storage (`mmt-aws-usw2-ext-s3-...`) vs. the default shared bucket. May has ALL_PRIVILEGES. |
| 2026-04-21 | **Target warehouse: `Serverless Starter Warehouse`** (id `dafc5dff8eb8094a`) | Only healthy warehouse; auto-starts. |
| 2026-04-21 | **LLM endpoint: `databricks-claude-sonnet-4-6`** (default) | Already READY on sandbox; no provisioning needed. |

## Docker secrets migration plan

This is a **self-contained checklist** — follow top-to-bottom if you need to redo this weeks from now. All `{{secrets/...}}` syntax is documented at https://docs.databricks.com/aws/en/compute/custom-containers#use-secrets-for-authentication.

### Step 0 — Verify Docker Hub access (open question, 2026-04-20)

```bash
# Local Docker daemon working?
docker info

# mmtdb PAT still valid?
docker login
# If SSO-blocked or PAT expired: https://app.docker.com/settings/personal-access-tokens
# Databricks SSO/Okta setup refs (internal Freshservice): search "Docker-Desktop-Okta-SSO-Setup"
# Need a Databricks-sponsored Docker license? Request via R2-DB or Opal.
```

### Step 1 — Clone Srijit's images to `mmtdb` namespace

Pull using Srijit's creds (from `modules/bionemo/module.env` and `modules/parabricks/module.env`), retag, push to `mmtdb`.

```bash
# BioNeMo
docker login -u srijitnair254       # token from modules/bionemo/module.env
docker pull srijitnair254/bionemo_dbx_amd64:0.1
docker tag  srijitnair254/bionemo_dbx_amd64:0.1    mmtdb/bionemo_dbx_amd64:0.1

# Parabricks
docker login -u srijitnair254       # token from modules/parabricks/module.env
docker pull srijitnair254/parabricks_dbx_amd64:0.1
docker tag  srijitnair254/parabricks_dbx_amd64:0.1 mmtdb/parabricks_dbx_amd64:0.1

# Push both to mmtdb (log in as mmtdb first)
docker login                        # use mmtdb PAT
docker push mmtdb/bionemo_dbx_amd64:0.1
docker push mmtdb/parabricks_dbx_amd64:0.1
```

NOTE: if Docker Hub throttling bites, `docker login` as a paid account before pulling. Also consider `docker buildx imagetools create --tag mmtdb/... srijitnair254/...` to copy without local pull.

### Step 2 — Create `mmt` secret scope on the sandbox

```bash
# Target the sandbox workspace
databricks auth login --host https://fevm-mmt-aws-usw2.cloud.databricks.com
databricks current-user me          # confirm right workspace

# Create the scope (idempotent-ish; fails if exists, that's OK)
databricks secrets create-scope mmt

# Put the two secrets
databricks secrets put-secret mmt docker_PAT_user --string-value mmtdb
databricks secrets put-secret mmt docker_PAT_pw   --string-value '<mmtdb PAT>'

# Verify
databricks secrets list-secrets mmt
```

### Step 3 — Point `module.env` at the secrets

Edit `modules/bionemo/module.env`:
```
bionemo_docker_userid={{secrets/mmt/docker_PAT_user}}
bionemo_docker_token={{secrets/mmt/docker_PAT_pw}}
bionemo_docker_image=mmtdb/bionemo_dbx_amd64:0.1
```

Edit `modules/parabricks/module.env`:
```
parabricks_docker_userid={{secrets/mmt/docker_PAT_user}}
parabricks_docker_token={{secrets/mmt/docker_PAT_pw}}
parabricks_docker_image=mmtdb/parabricks_dbx_amd64:0.1
```

The values flow: module.env → DAB `var.*` → cluster YAML `docker_image.basic_auth` → Databricks resolves `{{secrets/...}}` at cluster launch.

### Step 4 — Verify before full deploy

Risk: GWB deploy may treat module.env values as literals and never reach the `docker_image.basic_auth` path where Databricks resolves them. Two things to check:

1. `grep -r 'var.bionemo_docker' modules/bionemo/` — confirm the vars are only used inside `docker_image.basic_auth` blocks, not substituted into shell commands or logs.
2. Smoke test with the bionemo inference cluster: deploy once, then check `databricks clusters get <cluster-id>` — the `docker_image.basic_auth.username/password` should come back as the literal `{{secrets/...}}` reference (NOT resolved to raw text). If resolved to raw text, the env plumbing breaks the secret abstraction.

If the smoke test fails, fall back to plaintext in module.env for Monday and add a GWB deploy-flow fix to the follow-on improvement PR.

### Step 5 — UC Volumes from DCS (applies to BioNeMo, probably Parabricks)

DCS clusters need this Spark config to see `/Volumes/`:
```
spark.databricks.unityCatalog.volumes.enabled: true
```

Check existing cluster YAMLs (`modules/bionemo/resources/*.yml`, `modules/parabricks/resources/*.yml`) — if missing, that's a gap-doc entry.

## Image cloning note

`mmtdb/bionemo_dbx_v0_amd64:latest` already exists from May's 2025 testing. Keep it — may be useful as a fallback if Srijit's image is out of date or breaks. Don't delete without confirming.

## References

- Best practice pattern (internal memory): `~/.claude/projects/.../reference_docker_secrets_best_practice.md`
- Image inventory (internal memory): `~/.claude/projects/.../reference_gwb_docker_images.md`
- May's Docker build notes (source of truth for this approach): https://docs.google.com/document/d/1IIr7UaxmCvT9jC76A6xBuIsx6qToPDUpKIgBr6zE9ko/edit
- Databricks DCS secrets docs: https://docs.databricks.com/aws/en/compute/custom-containers#use-secrets-for-authentication

## PLAN PIVOT (2026-04-21)

Srijit independently deployed full version_pinning stack to fe-vm-hls-amer:
- App: `https://genesis-workbench-1602460480284688.aws.databricksapps.com` (created 2026-04-20)
- **39 jobs across every GWB module** (alphafold, bionemo, boltz, chemprop, core, diffdock, disease_biology, esm2_embeddings, esm_fold, parabricks, protein_mpnn, proteina_complexa, rapidssinglecell, rfdiffusion, scanpy, scgpt, scimilarity, sequence_search, single_cell)
- Likely missing: open_babel (or tagged differently — verify when May has catalog access)

Your original `gwb-mmt-app-1602460480284688` basic demo still ACTIVE, untouched.

**Consequence:** sandbox deploy to fevm-mmt-aws-usw2 is NO LONGER required for Monday's SA HUNTER session. Use Srijit's fe-vm-hls-amer deployment as the demo target. Sandbox deploy stays on the board as a post-Monday "I stood it up myself" narrative followup.

**Only remaining blocker for fe-vm-hls-amer playbook:** May has no catalog access on that workspace yet (asked Srijit). Can still do UI walkthrough / knob inventory without it.

## Updates from upstream

- **2026-04-21 — pulled Srijit commit `1eeb717` "version pinning fixes"** (8 files, 94 ins / 19 del). Key implications:
  - **disease_biology ALSO needs Docker creds** (uses parabricks/GWAS). Copy same values to `modules/parabricks/module.env` + `modules/disease_biology/module.env`.
  - Deploy wizard skill updated: warehouses being STOPPED is fine; module ordering is a contract; terraform workaround should verify path exists + fall back to `$(which terraform)` if not.
  - Bug fixes in: `disease_biology.py`, `chemprop multitask_admet`, `proteina_complexa registration`, `esmfold registration`, alphafold `register_and_download.yml` (6 new lines, likely Figshare fix).
  - New `CLAUDE.md` at repo root — agent guidance for the repo.
- **Merge status:** version_pinning was merged to main/development at PR #126 (commit `46bbd86`). Srijit has since made 6 more commits on version_pinning (unmerged). Main has 10 additional commits (PRs #127 development back-merge, #128 claude_wizard). Staying on version_pinning for deploy; pull periodically.

## Open questions

- [x] Does May still have working Docker Hub (mmtdb) SSO/PAT? — **YES, confirmed 2026-04-21.**
- [ ] Does the GWB deploy flow correctly propagate `{{secrets/...}}` through module.env → DAB vars → cluster YAML? — answered by Step 4 smoke test.
- [ ] Does current Parabricks/BioNeMo cluster YAML set `spark.databricks.unityCatalog.volumes.enabled=true`? — check before deploy.
- [ ] Install local Terraform (`brew install terraform`) — required for HashiCorp PGP workaround in deploy.sh. Only outstanding preflight blocker as of 2026-04-21.

## Deploy.sh fixes applied on `mmt/ver_pin_sandbox_setup` branch (2026-04-21)

All fixes local to feature branch `mmt/ver_pin_sandbox_setup` (off `version_pinning`). Documented in `UX-GAPS.md`; candidates for upstream PR back to `version_pinning` post-Monday.

| # | Fix | Affected files |
|---|---|---|
| 1 | **Dynamic Terraform path + version** (was hardcoded `/opt/homebrew/bin/terraform` + `1.3.9`) | `deploy.sh:7-8` → `command -v terraform` / `terraform version -json` |
| 2 | **Poetry install via curl** instead of `pip install` (bypasses PEP 668 on Homebrew Python + pip-vs-pip3 on macOS) | `deploy.sh:33-50` |
| 3 | **`DATABRICKS_CONFIG_PROFILE`** auto-exported from `application.env` (was defaulting to expired DEFAULT profile) | `deploy.sh:15-22` + `application.env` adds `databricks_profile=` |
| 4 | **`databricks_profile` filtered out** of EXTRA_PARAMS before piping to DAB (was causing `variable not defined` validation error) | Outer `deploy.sh` + all 7 `modules/<module>/deploy.sh` |

All fixes logged in `UX-GAPS.md` as entries #1, #7, #8, #9.

## Preflight status (2026-04-21)

| Check | Status |
|---|---|
| Auth profile `fevm-mmt-aws-usw2` | ✅ |
| Target catalog `mmt_aws_usw2` + user has ALL_PRIVILEGES | ✅ |
| SQL warehouse `Serverless Starter Warehouse` (dafc5dff8eb8094a) | ✅ healthy, auto-starts |
| LLM endpoint `databricks-claude-sonnet-4-6` | ✅ READY |
| Python ≥ 3.11 | ✅ 3.14.3 |
| Poetry / jq | ✅ |
| Terraform | ✅ 1.14.9 — installed via `brew tap hashicorp/tap && brew install hashicorp/tap/terraform` (NOT plain `brew install terraform`) |
| Databricks CLI ≥ 0.295 | ✅ 0.297.2 (upgraded from 0.285) |
| Docker Hub (mmtdb) access | ✅ |

**Residual tool concern:** deploy.sh:8 pins `DATABRICKS_TF_VERSION=1.3.9` but local Terraform is 1.14.9 — watch for DAB version mismatch at first deploy; Task #20 PR would fix portability long-term.
