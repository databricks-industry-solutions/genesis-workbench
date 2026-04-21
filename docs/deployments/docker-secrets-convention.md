# GWB Docker Secrets — Naming Convention & Setup

**Audience:** anyone deploying Genesis Workbench who needs Docker Hub credentials for BioNeMo, Parabricks, or Disease Biology modules.
**Scope:** workspace-agnostic — applies to any GWB deploy (sandbox, hls-amer, future workspaces).

## Why secret scope, not plaintext

GWB's default `modules/<module>/module.env` files ship with **plaintext Docker credentials** (e.g., `bionemo_docker_token=dckr_pat_...`). This is a known security gap — anyone with repo-read access sees the token.

The Databricks-native fix: reference secret scope entries via `{{secrets/<scope>/<key>}}` in module.env values. Databricks resolves these at cluster launch; no plaintext lands in cluster logs or config dumps.

Docs: https://docs.databricks.com/aws/en/compute/custom-containers#use-secrets-for-authentication

## Naming convention

**Secret scope:** `mmt` (or per-deployment: `gwb_<env>_secret_scope` — keep it short and memorable).

**Key format:** `gwb_<module>_docker_<user|token>`

| Key | Purpose | Default value |
|---|---|---|
| `gwb_bionemo_docker_user` | Docker Hub user for BioNeMo image pull | `srijitnair254` |
| `gwb_bionemo_docker_token` | Docker Hub PAT for same | Srijit's PAT (from `modules/bionemo/module.env`) |
| `gwb_parabricks_docker_user` | Docker Hub user for Parabricks image pull | `srijitnair254` |
| `gwb_parabricks_docker_token` | Docker Hub PAT for same | Srijit's PAT (from `modules/parabricks/module.env`) |

**Disease Biology** reuses Parabricks creds (its GWAS pipeline invokes Parabricks for alignment) — same keys, no additional secrets needed.

## Rationale for the `gwb_` prefix

The prefix signals **"these are shared GWB-project Docker creds, currently pointing at Srijit's Docker Hub repos."**

When/if you migrate to your own image namespace (e.g., `mmtdb/bionemo_dbx_amd64:0.1`, owned by you personally), drop the `gwb_` prefix:

| Phase | Key name | Points at |
|---|---|---|
| **Current** (using Srijit's images) | `gwb_bionemo_docker_*` | `srijitnair254/*` |
| **Post-migration** (using your own images) | `bionemo_docker_*` | `mmtdb/*` |

Keeping the prefix naming distinct between "shared GWB" and "personal" makes the migration explicit in code review and deploy logs.

## Setup (per new workspace)

```bash
# 1. Create the scope (idempotent-ish; errors if exists but that's fine)
databricks secrets create-scope mmt --profile <your-profile>

# 2. Put each secret (values from the repo's plaintext module.env files for now)
databricks secrets put-secret mmt gwb_bionemo_docker_user \
  --string-value srijitnair254 --profile <your-profile>
databricks secrets put-secret mmt gwb_bionemo_docker_token \
  --string-value '<paste from modules/bionemo/module.env>' --profile <your-profile>
databricks secrets put-secret mmt gwb_parabricks_docker_user \
  --string-value srijitnair254 --profile <your-profile>
databricks secrets put-secret mmt gwb_parabricks_docker_token \
  --string-value '<paste from modules/parabricks/module.env>' --profile <your-profile>

# 3. Verify
databricks secrets list-secrets mmt --profile <your-profile>
```

## Module.env reference pattern

**`modules/bionemo/module.env`:**
```
bionemo_docker_userid={{secrets/mmt/gwb_bionemo_docker_user}}
bionemo_docker_token={{secrets/mmt/gwb_bionemo_docker_token}}
bionemo_docker_image=srijitnair254/bionemo_dbx_amd64:0.1
```

**`modules/parabricks/module.env`:**
```
parabricks_docker_userid={{secrets/mmt/gwb_parabricks_docker_user}}
parabricks_docker_token={{secrets/mmt/gwb_parabricks_docker_token}}
parabricks_docker_image=srijitnair254/parabricks_dbx_amd64:0.1
```

**`modules/disease_biology/module.env`** (reuses parabricks creds per Srijit's `version_pinning` commit `1eeb717`):
```
parabricks_docker_userid={{secrets/mmt/gwb_parabricks_docker_user}}
parabricks_docker_token={{secrets/mmt/gwb_parabricks_docker_token}}
parabricks_docker_image=srijitnair254/parabricks_dbx_amd64:0.1
```

## Smoke test (after first module deploy)

After deploying any module using the secret refs, verify Databricks isn't passing plaintext values through:

```bash
databricks clusters get <cluster-id> --profile <your-profile> | grep -A3 docker_image
```

You should see `"username": "{{secrets/mmt/gwb_bionemo_docker_user}}"` — the literal reference string, NOT the resolved plaintext. If you see the raw username/token, the DAB → cluster pipeline is flattening refs → fix the variable substitution before trusting production secrets.

## Related

- Image inventory: [`reference_gwb_docker_images.md`](../../.claude/projects/-Users-may-merkletan-Documents-Projects-GWB/memory/reference_gwb_docker_images.md) *(internal memory — path for reference only)*
- Secret scope best-practice source: May's internal Google Doc `Gwb_docker_build_notes` (2025 BioNeMo/DCS testing)
- Per-workspace deploy runbooks: `docs/deployments/<workspace>/SESSION-NOTES.md`
