# Redeploying Failed Sub-Module Jobs

When individual sub-module jobs fail (e.g., spot preemption, transient errors), you can
redeploy and re-trigger specific jobs without redeploying the entire module.

---

## Prerequisites

```bash
# Authenticate (opens browser)
databricks auth login --host https://fe-vm-hls-amer.cloud.databricks.com --profile fe-vm-hls-amer
```

---

## Step 1: Redeploy the Bundle

Each sub-module has its own DAB bundle. Redeploying updates the job definition in-place
(e.g., cluster config, notebook paths) without affecting other modules.

Navigate to the sub-module directory and run `bundle deploy`:

```bash
# Pattern
cd modules/<domain>/<tool>/<version>
databricks bundle deploy \
  --profile fe-vm-hls-amer \
  --var="$(paste -sd, ../../../../application.env),$(paste -sd, ../../../../aws.env)"
```

### Examples

```bash
# esmfold
cd modules/protein_studies/esmfold/esmfold_v1
databricks bundle deploy \
  --profile fe-vm-hls-amer \
  --var="$(paste -sd, ../../../../application.env),$(paste -sd, ../../../../aws.env)"

# boltz
cd modules/protein_studies/boltz/boltz_1
databricks bundle deploy \
  --profile fe-vm-hls-amer \
  --var="$(paste -sd, ../../../../application.env),$(paste -sd, ../../../../aws.env)"

# alphafold
cd modules/protein_studies/alphafold/alphafold_v2.3.2
databricks bundle deploy \
  --profile fe-vm-hls-amer \
  --var="$(paste -sd, ../../../../application.env),$(paste -sd, ../../../../aws.env)"

# scgpt
cd modules/single_cell/scgpt/scgpt_v0.2.4
databricks bundle deploy \
  --profile fe-vm-hls-amer \
  --var="$(paste -sd, ../../../../application.env),$(paste -sd, ../../../../aws.env)"

# scimilarity
cd modules/single_cell/scimilarity/scimilarity_v0.4.0_weights_v1.1
databricks bundle deploy \
  --profile fe-vm-hls-amer \
  --var="$(paste -sd, ../../../../application.env),$(paste -sd, ../../../../aws.env)"

# scanpy
cd modules/single_cell/scanpy/scanpy_v0.0.1
databricks bundle deploy \
  --profile fe-vm-hls-amer \
  --var="$(paste -sd, ../../../../application.env),$(paste -sd, ../../../../aws.env)"

# rapidssinglecell
cd modules/single_cell/rapidssinglecell/rapidssinglecell_v0.0.1
databricks bundle deploy \
  --profile fe-vm-hls-amer \
  --var="$(paste -sd, ../../../../application.env),$(paste -sd, ../../../../aws.env)"

# rfdiffusion
cd modules/protein_studies/rfdiffusion/rfdiffusion_v1.1.0
databricks bundle deploy \
  --profile fe-vm-hls-amer \
  --var="$(paste -sd, ../../../../application.env),$(paste -sd, ../../../../aws.env)"

# protein_mpnn
cd modules/protein_studies/protein_mpnn/protein_mpnn_v0.1.0
databricks bundle deploy \
  --profile fe-vm-hls-amer \
  --var="$(paste -sd, ../../../../application.env),$(paste -sd, ../../../../aws.env)"
```

---

## Step 2: Find the Job ID

Bundle deploy updates the job definition but does not trigger a run. You need the
job ID to re-trigger it.

### List all GWB jobs

```bash
databricks jobs list --profile fe-vm-hls-amer --output json | python3 -c "
import json, sys
jobs = json.load(sys.stdin)
for j in sorted(jobs, key=lambda x: x.get('settings',{}).get('name','')):
    name = j.get('settings',{}).get('name','')
    tags = j.get('settings',{}).get('tags',{})
    if tags.get('application') == 'genesis_workbench':
        print(f\"{j['job_id']:>20}  {name}\")
"
```

### Search for a specific job by name

```bash
# Exact match
databricks jobs list --profile fe-vm-hls-amer --output json | python3 -c "
import json, sys
for j in json.load(sys.stdin):
    if j.get('settings',{}).get('name','') == 'register_esmfold':
        print(j['job_id'])
"

# Partial match (case-insensitive)
databricks jobs list --profile fe-vm-hls-amer --output json | python3 -c "
import json, sys
search = 'alphafold'
for j in json.load(sys.stdin):
    name = j.get('settings',{}).get('name','')
    if search.lower() in name.lower():
        print(f\"{j['job_id']:>20}  {name}\")
"
```

### Known job IDs (as of 2026-03-12)

**Protein Studies**

| Job ID | Name |
|--------|------|
| `747548824052399` | register_esmfold |
| `970641084894060` | register_boltz |
| `151110797461064` | alphafold_register_and_downloads |
| `117338573165412` | run_alphafold |
| `1101307541345893` | register_rfdiffusion |
| `344546572169600` | register_proteinmpnn |

**Single Cell**

| Job ID | Name |
|--------|------|
| `66857169007008` | register_scanpy_job |
| `353231618209721` | run_scanpy_gwb |
| `156576662480449` | register_rapidssinglecell_job |
| `466711515644609` | run_rapidssinglecell_gwb |
| `929386102458411` | register_scgpt |
| `632000935628382` | register_scimilarity |
| `599617698721092` | download_gene_references_gwb |

**Core**

| Job ID | Name |
|--------|------|
| `714759487700712` | dbx_gwb_initialize_core |
| `207562594195501` | dbx_gwb_deploy_model_job |
| `448719758304462` | dbx_gwb_initialize_module_job |
| `1083186048758593` | dbx_gwb_destroy_module_job |

> **Note:** Job IDs are stable across `bundle deploy` — redeploys update the job
> definition but do not change the ID.

---

## Step 3: Trigger the Job

```bash
# Default: CLI blocks until job completes (may timeout on long-running jobs)
databricks jobs run-now <JOB_ID> --profile fe-vm-hls-amer

# Recommended for long-running jobs: --no-wait returns immediately with run ID
databricks jobs run-now <JOB_ID> --profile fe-vm-hls-amer --no-wait
```

> **Note:** GWB model registration and download jobs routinely take 1-6+ hours. The
> CLI will timeout waiting, showing `Error: timed out:` — this does NOT mean the job
> failed. The job continues running on the cluster. Always use `--no-wait` or check
> the actual run status rather than relying on CLI exit code.

### Monitor active runs

```bash
# List active runs
databricks runs list --active-only --profile fe-vm-hls-amer --output table

# Check a specific run
databricks runs get --run-id <RUN_ID> --profile fe-vm-hls-amer

# Check latest run for a specific job (via API)
databricks api get /api/2.1/jobs/runs/list --profile fe-vm-hls-amer \
  --json '{"job_id": <JOB_ID>, "limit": 1}'

# Get error details for a failed task
databricks api get /api/2.1/jobs/runs/get-output --profile fe-vm-hls-amer \
  --json '{"run_id": <RUN_ID>}'
```

---

## Note: AlphaFold Job Spins Up All Clusters on Retry

The `alphafold_register_and_downloads` job defines 6 separate `job_clusters`. Databricks
provisions **all** clusters at job start, even if their tasks will skip immediately
(because data already exists in the Volume). Tasks check
`if [ ! -d "$MODEL_VOLUME/datasets/..." ]` and exit in seconds, but the cluster still
takes ~5-10 min to provision and terminate.

**Impact:** ~$0.50 per unnecessary `i3.4xlarge` cluster on retries. Acceptable for
one-time setup, but wasteful if re-running repeatedly.

**Alternative for targeted retries:** Instead of re-running the whole job, run individual
failed notebooks on an existing cluster or create temporary single-task jobs. For example,
to retry just `pdb_mmcif`:
1. Start an `i3.4xlarge` all-purpose cluster
2. Attach and run the `download_setup.py` notebook (sets up env vars, clones repo, patches scripts)
3. Run the `download_pdb_mmcif.py` notebook

---

## Common Failure: Spot Instance Preemption

**Symptom:**
```
Cluster terminated because driver node is a spot instance that was
terminated by the cloud provider.
```

**Fix:** All job cluster definitions should include:
```yaml
aws_attributes:
  availability: ON_DEMAND
```

This was added to all job clusters across the repo (18 clusters in 12 YAML files,
including bionemo). If a new sub-module is added without `aws_attributes`, it will
default to spot instances on AWS and may be preempted during long-running jobs.

**Why ON_DEMAND for all jobs:** GWB jobs are long-running (hours for model
downloads/registration), non-checkpointed, and partially non-idempotent. Spot
preemption wastes all compute up to the preemption point. The ~30-40% cost premium
is negligible compared to wasted spot compute and manual re-triggering.

---

## Common Failure: FTP/Rsync Blocked (AlphaFold Downloads)

**Symptom:** aria2c shows `0B/0B CN:1 DL:0B` for minutes, then `errorCode=2 Timeout`
on FTP URLs. Or rsync hangs on port 33444.

**Root cause:** Five layered issues, each only visible after fixing the previous one:
1. **Spot preemption** (infra) — cluster terminated by AWS reclaiming spot instance
2. **FTP/rsync blocked** (network) — AWS VPC blocks outbound FTP and rsync port 33444
3. **Heredoc quoting** (notebook format) — `# MAGIC %sh` heredocs produce malformed scripts
4. **Path parsing + silent wget failure** (script bugs) — `grep`/`sed` left trailing `"`
   in subdir names; `wget -r` silently failed on HTTPS dir listings; `|| true` +
   `2>/dev/null` hid all evidence
5. **HTML parsing kept "href" prefix** (regex bug) — `sed 's/[^a-z0-9]//g'` meant to
   strip non-alphanumeric chars but "href" is alphanumeric — subdirs parsed as `href0a`
   instead of `0a`, all 1119 dirs showed `(0 files)`

**Fix:** The `download_setup.py` notebook patches FTP→HTTPS via `sed` after cloning the
AlphaFold repo. A Python cell creates `download_pdb_mmcif_https.sh` which uses `aria2c`
(16 parallel connections) to download `.cif.gz` files per subdir from the EBI HTTPS mirror.

Key implementation notes:
- Script must be created via Python cell, not `%sh` heredoc (quoting issues in `# MAGIC`)
- Uses `aria2c` not `wget -r` — wget recursive is unreliable on HTTPS directory listings
- Parses actual `.cif.gz` file links from each subdir, not recursive crawling
- Includes file count verification — exits with error if zero files downloaded
- Never combine `|| true` with `2>/dev/null` on downloads — hides all failure evidence
- Use positional extraction (`cut -d'"' -f2`) not character-class deletion (`sed 's/[^a-z0-9]//g'`) when parsing HTML attributes — field names like "href" are themselves alphanumeric
- The HTTPS script provides per-file aria2c progress (`OK | 130MiB/s | /path/to/file.cif.gz`) and a subdir counter (`[314/1119] Downloading dm/ (291 files)...`). Slower than bulk rsync due to per-subdir `curl` to list files, but gives clear visibility into what's being downloaded

Redeploy the alphafold bundle to pick up the fix:

```bash
cd modules/protein_studies/alphafold/alphafold_v2.3.2
databricks bundle deploy --profile fe-vm-hls-amer \
  --var="$(paste -sd, ../../../../application.env),$(paste -sd, ../../../../aws.env)"
databricks jobs run-now 151110797461064 --profile fe-vm-hls-amer
```

---

## Common Failure: Auth Token Expired

**Symptom:**
```
error getting token: a new access token could not be retrieved because
the refresh token is invalid
```

**Fix:** Re-authenticate in a separate terminal (it opens a browser):
```bash
databricks auth login --host https://fe-vm-hls-amer.cloud.databricks.com --profile fe-vm-hls-amer
```

Then retry the deploy/run command.

---

## Common Failure: Missing Wheel

**Symptom:**
```
ModuleNotFoundError: No module named 'genesis_workbench'
```

**Fix:** Manually copy the wheel to UC Volumes:
```bash
databricks fs cp \
  modules/core/library/genesis_workbench/dist/genesis_workbench-0.1.0-py3-none-any.whl \
  dbfs:/Volumes/hls_amer_catalog/mmt_genesis_workbench/libraries/genesis_workbench-0.1.0-py3-none-any.whl \
  --overwrite --profile fe-vm-hls-amer
```
