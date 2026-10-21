# Playbook — Migrate a serving model to Express deployments (env_pack)

Repeatable recipe for moving a Genesis Workbench serving endpoint off the standard Model-Serving
container build onto **Express deployments** (`env_pack`), which snapshots the serverless notebook env at
**registration** and restores it verbatim at serving (deploys in ~6 min, no 30–38 min rebuild, no torch
force-install). Derived from the `rfd4_proteina` reference and the diffdock migration.

## The core principle (keep the blast radius tiny)

**Only the MLflow registration step changes.** `deploy_model` / the `02_import_model_gwb` notebook /
`model_deployments` / `get_endpoint_name_for_uc_model` resolution are **left exactly as-is** — do NOT bypass
`deploy_model`. `deploy_model_endpoint` is just `serving_endpoints.create_and_wait(...)` referencing the UC
model **version**; whether that version serves as Express or standard is decided entirely by *how it was
registered*. So: register the version with `env_pack`, and the unchanged `deploy_model` serves it as Express.

## Eligibility (confirm before starting)

- Custom `mlflow.pyfunc` (or transformers) model, registered in **Unity Catalog**. ✅ all GWB models.
- Register notebook runs on a **serverless notebook, environment/client version ≥ 3** (`environments.spec.client`
  in the register `*.yml`). GPU register jobs are usually client `'6'`/`'4'` (OK). **CPU register jobs on
  client `'2'` must be bumped to `'3'`+** (and re-validated) — this is the one case where a non-registration
  file (the resource YAML) also changes.
- **Compute parity**: serving compute must match register compute. GPU endpoint ⇒ register FROM serverless
  GPU; CPU endpoint ⇒ register FROM serverless CPU. (Logging a GPU model from CPU makes the GPU endpoint fail
  to start.)
- Env ≤ **200 GB**. Not Python-version-locked below the serverless runtime (**scimilarity = py3.10-locked ⇒
  NOT eligible**).

## The change (per register notebook)

1. **Install cell** — bump the MLflow + SDK pins to the env_pack API:
   `mlflow==2.22.0` → `mlflow>=3.12`, `databricks-sdk==0.50.0` → `databricks-sdk>=0.150.0`.
   (This is the only change beyond `log_model` itself, and it's mandatory for `env_pack`.)
2. **`log_model`** — three edits, all inside the existing `log_model(...)` call:
   - Drop `registered_model_name=...` (Express registers in a separate step).
   - `pip_requirements=["mlflow", "cloudpickle", "pandas", "numpy"]` (minimal — `env_pack` defines the serving
     env; the old ABI/torch pins become irrelevant at serving).
   - If the call uses **`code_path=`**, rename to **`code_paths=`** (mlflow 3.x renamed it — a silent
     `TypeError` otherwise). Keep `artifacts=` / `signature=` / `input_example=` unchanged.
3. **Register separately with env_pack** — right after `log_model`:
   ```python
   from mlflow.utils.env_pack import EnvPackConfig
   version = mlflow.register_model(
       model_uri=model_info.model_uri, name=f"{catalog}.{schema}.{model_name}",
       env_pack=EnvPackConfig(name="databricks_model_serving", install_dependencies=False),
   ).version
   ```
   `install_dependencies=False` snapshots site-packages **verbatim** — required when the env has
   git/binary/local-tagged wheels (torch `+cuXXX`, PyG, cuda-13); the default `True` re-resolves them and
   fails `No matching distribution`.
4. **Nothing else.** No `deploy_model` change, no endpoint-creation code in the register notebook, no asserts,
   no artifact-path env vars, no `log_params` churn. Leave `02_import_model_gwb` / warmup untouched.

## Deploy + validate (one model at a time)

1. **Delete the existing endpoint(s) first** so `deploy_model_endpoint` takes the clean *create* path instead
   of `update_config` (and you prove Express serving from scratch):
   `databricks serving-endpoints delete gwb_<prefix>_<uc_name>_endpoint`
2. `cd modules/<module> && DATABRICKS_CONFIG_PROFILE=<profile> ./deploy.sh <cloud> --only-submodule <sub>`
   — bundle deploy + run the register job (`01` env_pack-registers on GPU → `02` import + `deploy_model` +
   warmup). Monitor the job to a terminal state.
3. Confirm the endpoint goes **READY** and smoke-test it; a healthy Express deploy comes up in minutes (no
   long container build). Then commit (register notebook diff should be registration-only + the version bump).

## Gotchas (learned)

- **Register-install pins must match the model's `conda_env`/requirements.** env_pack snapshots the REGISTER
  KERNEL, not the model's declared `conda_env`. If the register install leaves deps UNPINNED while the model
  declares a pinned `conda_env` (the env standard serving used), env_pack captures the *latest* versions →
  version-drift `ImportError` at serving. Real case: proteina_complexa installed `atomworks` unpinned →
  latest dropped `AtomSelectionStack` → `from proteinfoundation…` failed at model load; the conda_env pinned
  `atomworks==1.0.3`. Fix: pin the register install to the conda_env versions so env_pack reproduces the
  proven env. (Models whose register install == their `pip_requirements`/requirements.txt file, e.g. teddy,
  don't drift.) Serving model-load errors are UI-only (CLI logs key to the active config, absent for a failed
  pending deploy); the service log's `DATABRICKS_MODEL_LOAD_DIAGNOSTIC_V1` line gives the exact failing import.
- **A later install can downgrade the mlflow bump.** If the notebook installs a package AFTER the
  `mlflow>=3.12` bump that pulls an older mlflow (e.g. boltz's `dbboltz[gpu]`), the register kernel ends up on
  mlflow < 3.12 → `ModuleNotFoundError: No module named 'mlflow.utils.env_pack'` at `register_model`. Fix:
  reinstall `mlflow>=3.12` (+ `databricks-sdk>=0.150`) as the LAST `%pip` before `restartPython()`. (GWB-lib
  installs are safe — the lib pins no mlflow.)
- `code_path` → `code_paths` under mlflow 3.x (TypeError).
- `env_pack` registration MUST run on serverless GPU for a GPU endpoint (parity) — the install-cell torch is
  what gets packaged.
- `install_dependencies=False` is essential for binary/git/cuda-tagged wheels.
- CPU register jobs on serverless `client: '2'` need a bump to `'3'`+ to be Express-eligible.
- Delete the live endpoint before redeploying for a clean create.
- **The "model never runs at register + a py3.11 `conda_env`" trap (NOT Express-eligible).** Some
  register notebooks deliberately **don't exercise the model** — they only build an `input_example` and
  log the pyfunc, declaring the real serving env as an explicit `conda_env` pinned to an OLDER python
  (e.g. `python=3.11`) with wheels that have **no cp312 build** (`pandas==1.5.3`, `numpy==1.26.4`). On
  the py3.12 serverless register kernel those exact wheels can't be installed, so the install cell leaves
  numpy/pandas unpinned → the kernel resolves pandas 2.x / numpy 2.x. Standard serving rebuilt the py3.11
  `conda_env` so the model ran against the pinned old versions; **env_pack would instead snapshot the
  py3.12 kernel (pandas 2.x / numpy 2.x) and silently change the serving runtime** — more than "only the
  registration step changes," and the py3.12 env was never validated for inference. Treat these like the
  python-version locks: **leave on standard serving** (real case: `pltnum_v1` — its notebook says "this
  notebook never runs the model … the serving env is the explicit `conda_env` below (python=3.11)").
  Contrast `chemprop_*`, which install the full inference stack (chemprop/rdkit/torch/lightning) on the
  py3.12 kernel with no separate py3.11 `conda_env` → the register env IS the serving env → clean migrate.

## Status / scope

**Migrated + deployed + verified READY** (Express via this playbook): `rfd4_proteina` (reference),
`diffdock` (+`diffdock_esm_embeddings`), `esm2_embeddings`, `esmfold`, `proteinmpnn`,
`rfdiffusion_inpainting`, `scgpt` (+`scgpt_perturbation`), `teddy`, `deepstabp_v1`, `proteina_complexa`
(×3: `proteina_complexa`, `proteina_complexa_ame`, `proteina_complexa_ligand`), `boltz`, `kermt_admet`,
`chemprop_*` (×3: `chemprop_bbbp` / `01`, `chemprop_clintox` / `02`, `chemprop_admet` / `03`; one
`register_chemprop` job, 3 GPU_1xA10 tasks — clean migrate: full inference stack on py3.12, no py3.11
`conda_env`). **The GPU Express-migration pass is complete** — every GPU-eligible endpoint is migrated.

**NOT eligible — leave on standard serving**:
- `scimilarity` (`scimilarity_get_embedding`, `scimilarity_gene_order`) — py3.10-locked.
- `genmol` — py3.11-locked (pinned deps with no py3.12 wheels), and registers on CPU / serves on GPU.
- `pltnum_v1` — the "model never runs at register + py3.11 `conda_env` (pandas 1.5.3 / numpy 1.26.4, no
  cp312 wheel)" trap (see Gotchas). env_pack would change the serving runtime to py3.12 / pandas 2.x,
  unvalidated → leave on standard serving.

**CPU-only, Express-eligible but out of the current GPU-migration scope** (register `client: '4'`,
no bump needed): `hlaiipred_v1`, `mhcflurry_v2`, `netsolp_v1`.
