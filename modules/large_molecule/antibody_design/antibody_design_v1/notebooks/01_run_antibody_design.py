# Databricks notebook source
# MAGIC %md
# MAGIC ### Antibody Design (VHH) — Reward-Weighted Generation Loop
# MAGIC
# MAGIC Orchestrator job for the Antibody Design tab. Generates single-domain (VHH / nanobody) antibodies
# MAGIC against an antigen epitope with **RFD4-Proteina loaded in-process on an H100**, then optimizes them
# MAGIC over a reward-weighted loop. Each iteration:
# MAGIC
# MAGIC 1. Generate K VHH candidates with RFD4 conditioned on the antigen epitope (C_HOT hotspots).
# MAGIC 2. Number each with `anarcii`; if it numbers as a V-domain, (optionally) ProteinMPNN-redesign the
# MAGIC    **framework** while FIXING the CDR loops (preserve the binding RFD4 just designed).
# MAGIC 3. ESMFold each candidate → structure + mean pLDDT (fold confidence).
# MAGIC 4. (Optional) Boltz co-fold antigen + VHH → ipTM (binding/interface confidence).
# MAGIC 5. Developability: NetSolP (solubility), PLTNUM-anchored half-life, DeepSTABp Tm, MHCflurry
# MAGIC    immunogenic burden (lower is better — critical for an antibody).
# MAGIC 6. Compose a per-candidate composite reward (z-score→min-max within the batch, weighted sum),
# MAGIC    log to MLflow, resample parents for the next iteration.
# MAGIC
# MAGIC Dispatched by `start_antibody_design_job` in `modules/core/app/backend/app/services/antibody_design.py`.
# MAGIC
# MAGIC > ⚠️ FIRST-DRAFT VHH conditioning. The RFD4 condition_spec (in `utils.generate_vhh`) generates a
# MAGIC > VHH-length binder against the epitope; it does NOT yet scaffold a true Ig framework (keep framework,
# MAGIC > design only CDR loops). anarcii annotates how antibody-like each design is. Expect a deploy-time
# MAGIC > iteration or two on the condition_spec — same caveat as the RFD4 register notebook's binder/motif path.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("cache_dir", "antibody_design", "Cache dir (UC volume) for this workflow")
dbutils.widgets.text("rfd4_cache_dir", "rfd4_proteina", "RFD4 cache dir holding the flow + AE checkpoints")
dbutils.widgets.text("rfd4_flow_ckpt", "chk_epoch_00000169_step_000000340000-ema.ckpt", "Flow checkpoint filename")
dbutils.widgets.text("rfd4_ae_ckpt", "2026_09_04_release_ae_z16_400k-ema.ckpt", "Autoencoder checkpoint filename")
dbutils.widgets.text("sql_warehouse_id", "", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")

dbutils.widgets.text("rfd4_git_url", "github.com/NVIDIA-BioNeMo/RFD4-Proteina-Open-Model-Private.git", "RFD4 git host/path")
dbutils.widgets.text("rfd4_git_ref", "84d83b98442e5946a6a626d0d99a19773207e894", "RFD4 git ref")
dbutils.widgets.text("rfd4_git_token_scope", "dbx_genesis_workbench", "Secret scope for the GitHub PAT")
dbutils.widgets.text("rfd4_git_token_key", "rfd4_github_token", "Secret key for the GitHub PAT")

dbutils.widgets.text("mlflow_experiment", "", "MLflow experiment tag")
dbutils.widgets.text("mlflow_run_name", "", "MLflow run name")
dbutils.widgets.text("mlflow_run_id", "", "Pre-created MLflow run id (set by dispatcher; empty = create new)")
dbutils.widgets.text("antigen_pdb_path", "", "UC volume path to the antigen PDB")
dbutils.widgets.text("epitope_residues_csv", "", "Epitope residue numbers on the antigen (CSV ints)")
dbutils.widgets.text("antigen_chain", "A", "Antigen chain id in the input PDB")
dbutils.widgets.text("vhh_length_min", "110", "VHH length min")
dbutils.widgets.text("vhh_length_max", "130", "VHH length max")
dbutils.widgets.text("num_samples", "8", "K — candidates per iteration")
dbutils.widgets.text("num_iterations", "6", "N — iteration ceiling (convergence usually exits earlier)")
dbutils.widgets.text("run_proteinmpnn", "true", "ProteinMPNN-redesign the framework (fix CDRs) per candidate")
dbutils.widgets.text("resampling_temperature", "0.1", "Resampling softmax temperature")
dbutils.widgets.text("strategy", "resample", "Strategy: resample | noop")
dbutils.widgets.text("dev_user_prefix", "", "Dev user prefix (matches DEV_USER_PREFIX)")
dbutils.widgets.text("cofold_antigen", "true", "Co-fold antigen+VHH with Boltz to score binding (ipTM)")
dbutils.widgets.text("references_json", "[]", "Reference antibody JSON list (anchors the half-life axis)")
dbutils.widgets.text("half_life_margin", "0.05", "Half-life anchor margin")
dbutils.widgets.text(
    "weights_json",
    '{"plddt":1.3,"boltz":2.0,"solubility":1.0,"half_life":1.0,"thermostab":1.0,"immuno":1.5}',
    "Per-axis weights JSON",
)

dbutils.widgets.text("convergence_threshold", "0.01", "Convergence: min iter_max_reward improvement; negative disables")
dbutils.widgets.text("convergence_window", "2", "Convergence: iterations to compare")
dbutils.widgets.text("target_reward", "", "Threshold stop: composite reward target (empty = disabled)")
dbutils.widgets.text("best_k_target", "", "Best-K stop: number of candidates above threshold (empty = disabled)")
dbutils.widgets.text("best_k_threshold", "", "Best-K stop: composite reward threshold (empty = disabled)")

# COMMAND ----------

# DBTITLE 1,Install the rfproteina CUDA-13 stack (RFD4 loads in-process) — then restart
import os, subprocess, sys
g = dbutils.widgets.get
git_url, git_ref = g("rfd4_git_url"), g("rfd4_git_ref")
git_token_scope, git_token_key = g("rfd4_git_token_scope"), g("rfd4_git_token_key")

try:
    gh_token = dbutils.secrets.get(git_token_scope, git_token_key)
except Exception:
    gh_token = None  # OK once the repo is public — fall back to an unauthenticated clone
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
open("/tmp/bc.txt", "w").write("torch==2.14.1\n")
# Install rfproteina into the kernel env (diffdock pattern). --torch-backend cu132 pins CUDA-13 torch.
cred = f"x-access-token:{gh_token}@" if gh_token else ""
spec = f"rfd4-proteina[metrics-cuda13] @ git+https://{cred}{git_url}@{git_ref}"
_r = subprocess.run(["uv", "pip", "install", "--python", sys.executable, "--break-system-packages",
                     "--build-constraint", "/tmp/bc.txt", "--torch-backend", "cu132", spec],
                    capture_output=True, text=True)
if _r.returncode != 0:
    _err = (_r.stderr or "")
    if gh_token:
        _err = _err.replace(gh_token, "<redacted>")
    hint = "" if gh_token else f" (no token at {git_token_scope}/{git_token_key}; set a PAT if the repo is still private)"
    raise RuntimeError(f"uv install failed rc={_r.returncode}{hint}; stderr tail:\n{_err[-3000:]}")
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Install the GWB library + mlflow/sdk pins (for initialize + MLflow logging) — then restart
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
gwb_library_path = None
for lib in dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries"):
    if lib.name.startswith("genesis_workbench"):
        gwb_library_path = lib.path.replace("dbfs:", "")
print(f"GWB library wheel: {gwb_library_path}")

# COMMAND ----------

# MAGIC %pip install {gwb_library_path} --force-reinstall
# MAGIC %pip install databricks-sdk==0.50.0 databricks-sql-connector==4.0.3 mlflow==2.22.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Re-read widgets + parse (kernel was wiped by the restarts above)
import os, sys, json, math, tempfile
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import mlflow

g = dbutils.widgets.get
catalog, schema, cache_dir = g("catalog"), g("schema"), g("cache_dir")
rfd4_cache_dir = g("rfd4_cache_dir")
rfd4_flow_ckpt, rfd4_ae_ckpt = g("rfd4_flow_ckpt"), g("rfd4_ae_ckpt")
sql_warehouse_id, user_email = g("sql_warehouse_id"), g("user_email")

mlflow_experiment = g("mlflow_experiment")
mlflow_run_name = g("mlflow_run_name")
mlflow_run_id = g("mlflow_run_id") or None
antigen_pdb_path = g("antigen_pdb_path")
epitope_residues_csv = g("epitope_residues_csv")
antigen_chain = g("antigen_chain")
vhh_length_min = int(g("vhh_length_min"))
vhh_length_max = int(g("vhh_length_max"))
num_samples = int(g("num_samples"))
num_iterations = int(g("num_iterations"))
run_proteinmpnn_flag = g("run_proteinmpnn").lower() in ("true", "1", "yes")
resampling_temperature = float(g("resampling_temperature"))
strategy_name = g("strategy")
dev_user_prefix = g("dev_user_prefix")
cofold_antigen = g("cofold_antigen").lower() in ("true", "1", "yes")
references_json = g("references_json")
half_life_margin = float(g("half_life_margin"))
weights_json = g("weights_json")

import re
def _residue_num(tok: str) -> int:
    m = re.search(r"(\d+)\s*$", tok.strip())
    if not m:
        raise ValueError(f"epitope residue {tok!r} has no residue number — use plain integers like '31,52,99'.")
    return int(m.group(1))

epitope_residues = [_residue_num(r) for r in epitope_residues_csv.split(",") if r.strip()]
weights = json.loads(weights_json) if weights_json else {}
references = json.loads(references_json) if references_json else []

def _parse_optional_float(s: str) -> "Optional[float]":
    s = (s or "").strip()
    return float(s) if s else None

def _parse_optional_int(s: str) -> "Optional[int]":
    s = (s or "").strip()
    return int(s) if s else None

convergence_threshold = float(g("convergence_threshold") or "0.01")
convergence_window = int(g("convergence_window") or "2")
target_reward = _parse_optional_float(g("target_reward"))
best_k_target = _parse_optional_int(g("best_k_target"))
best_k_threshold = _parse_optional_float(g("best_k_threshold"))

flow_ckpt_path = f"/Volumes/{catalog}/{schema}/{rfd4_cache_dir}/flow_checkpoints/{rfd4_flow_ckpt}"
ae_ckpt_path = f"/Volumes/{catalog}/{schema}/{rfd4_cache_dir}/ae_checkpoints/{rfd4_ae_ckpt}"

print(f"antigen_pdb_path:  {antigen_pdb_path}")
print(f"epitope_residues:  {epitope_residues}")
print(f"antigen_chain:     {antigen_chain}")
print(f"vhh_length:        [{vhh_length_min}, {vhh_length_max}]")
print(f"K (candidates):    {num_samples}   N (iterations): {num_iterations}")
print(f"weights:           {weights}")
print(f"cofold_antigen:    {cofold_antigen}   references: {len(references)}")
print(f"flow_ckpt:         {flow_ckpt_path}")
print(f"ae_ckpt:           {ae_ckpt_path}")

# COMMAND ----------

# DBTITLE 1,Load utils.py from the bundle's notebooks/ directory
_notebook_dir = os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
sys.path.insert(0, "/Workspace" + _notebook_dir)
import importlib
import utils as ab_utils
importlib.reload(ab_utils)
from utils import (
    PredictorAxis, compose_rewards, half_life_anchor_threshold, half_life_anchor_rewards,
    make_strategy, number_vhh, load_rfd4, generate_vhh,
    call_esmfold, call_proteinmpnn, call_boltz,
    call_netsolp, call_pltnum, call_deepstabp, call_mhcflurry,
    warmup_developability_endpoints, _extract_mean_plddt_from_pdb,
)

# COMMAND ----------

# DBTITLE 1,Initialize GWB + MLflow experiment
from genesis_workbench.workbench import initialize
from genesis_workbench.models import set_mlflow_experiment

databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema,
           sql_warehouse_id=sql_warehouse_id, token=databricks_token)
experiment = set_mlflow_experiment(experiment_tag=mlflow_experiment or "antibody_design", user_email=user_email)
mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# COMMAND ----------

# DBTITLE 1,Read the antigen PDB the app wrote to the cache volume + extract its sequence
with open(antigen_pdb_path, "r") as f:
    antigen_pdb_str = f.read()
print(f"Loaded antigen PDB ({len(antigen_pdb_str)} chars) from {antigen_pdb_path}")

def _antigen_sequence(pdb_str: str, chain: str) -> str:
    """One-letter sequence of the antigen chain (for the Boltz complex input)."""
    import io
    from biotite.structure.io.pdb import PDBFile
    from biotite.structure import to_sequence
    pf = PDBFile.read(io.StringIO(pdb_str))
    arr = pf.get_structure(model=1)
    arr = arr[arr.chain_id == chain]
    try:
        seqs, _ = to_sequence(arr)
        return "".join(str(s) for s in seqs)
    except Exception:
        return ""

antigen_seq = _antigen_sequence(antigen_pdb_str, antigen_chain) if cofold_antigen else ""
print(f"antigen sequence length: {len(antigen_seq)}")

# COMMAND ----------

# DBTITLE 1,Reward axes + half-life anchor + strategy
AXES = [
    PredictorAxis("plddt",      weights.get("plddt", 0.0)),
    PredictorAxis("boltz",      weights.get("boltz", 0.0)),
    PredictorAxis("solubility", weights.get("solubility", 0.0)),
    PredictorAxis("half_life",  weights.get("half_life", 0.0), pre_normalized=True),
    PredictorAxis("thermostab", weights.get("thermostab", 0.0)),
    PredictorAxis("immuno",     weights.get("immuno", 0.0), lower_is_better=True),
]
print("Enabled axes:", [a.name for a in AXES if a.enabled])

anchor_threshold = -math.inf
if any(a.name == "half_life" and a.enabled for a in AXES) and references:
    ref_seqs = [r["sequence"] for r in references]
    ref_scores = call_pltnum(ref_seqs, dev_user_prefix=dev_user_prefix)
    anchor_threshold = half_life_anchor_threshold(ref_scores, margin=half_life_margin)
    print(f"PLTNUM anchor threshold: {anchor_threshold:.4f} (from {len(ref_scores)} reference antibodies)")
else:
    print("Half-life axis disabled or no references — anchor threshold = -inf")

strategy = make_strategy(strategy_name, temperature=resampling_temperature)

# COMMAND ----------

# DBTITLE 1,Load RFD4-Proteina in-process (H100) + warm the scoring endpoints
import torch
if not torch.cuda.is_available():
    raise RuntimeError("Antibody Design requires a GPU (H100) — torch.cuda.is_available() is False. "
                       "The orchestrator job is pinned to GPU_1xH100; check the job's compute.")
rfd4_ctx = load_rfd4(flow_ckpt_path, ae_ckpt_path)
print("Warming developability endpoints...")
print(warmup_developability_endpoints(dev_user_prefix=dev_user_prefix))

# COMMAND ----------

# DBTITLE 1,Per-batch scoring across all enabled axes
def score_iteration(seqs: List[str], pdbs: List[str], plddts: List[float],
                    anchor: float) -> Dict[str, List[float]]:
    K = len(seqs)
    scores: Dict[str, List[float]] = {}

    if any(a.name == "plddt" and a.enabled for a in AXES):
        scores["plddt"] = [float(p) for p in plddts]

    # Boltz binding: co-fold antigen + VHH, take ipTM (interface confidence).
    if any(a.name == "boltz" and a.enabled for a in AXES) and cofold_antigen and antigen_seq:
        boltz_scores = []
        for seq in seqs:
            if not seq:
                boltz_scores.append(0.0); continue
            boltz_in = f"protein_A:{antigen_seq};protein_B:{seq}"
            try:
                out = call_boltz(boltz_in, dev_user_prefix=dev_user_prefix, timeout_seconds=1200)
                boltz_scores.append(float(out.get("ipTM", out.get("iLDDT", 0.0))))
            except Exception as e:
                print(f"Boltz failed for one candidate: {e}")
                boltz_scores.append(0.0)
        scores["boltz"] = boltz_scores

    if any(a.name == "solubility" and a.enabled for a in AXES):
        scores["solubility"] = call_netsolp(seqs, dev_user_prefix=dev_user_prefix)
    if any(a.name == "thermostab" and a.enabled for a in AXES):
        scores["thermostab"] = call_deepstabp(seqs, dev_user_prefix=dev_user_prefix)
    if any(a.name == "immuno" and a.enabled for a in AXES):
        scores["immuno"] = call_mhcflurry(seqs, dev_user_prefix=dev_user_prefix)
    if any(a.name == "half_life" and a.enabled for a in AXES):
        raw = call_pltnum(seqs, dev_user_prefix=dev_user_prefix)
        scores["half_life"] = ([0.5] * K if math.isinf(anchor)
                               else half_life_anchor_rewards(raw, anchor, beta=0.05))
    return scores

# COMMAND ----------

# DBTITLE 1,Generate + validate one batch of VHH candidates
def generate_and_validate(num: int) -> Dict[str, List]:
    """RFD4 generate → anarcii number → (optional) ProteinMPNN framework redesign
    fixing the CDRs → ESMFold. Returns dict of parallel lists."""
    gen_df = generate_vhh(rfd4_ctx, antigen_pdb_str, epitope_residues, antigen_chain,
                          vhh_length_min, vhh_length_max, num)
    seqs = list(gen_df["designed_sequence"])
    pdbs = list(gen_df["designed_pdb"])
    is_vhh_flags = []

    if run_proteinmpnn_flag:
        new_seqs = []
        for seq, pdb in zip(seqs, pdbs):
            info = number_vhh(seq) if seq else {"is_vhh": False, "cdr_positions": []}
            is_vhh_flags.append(bool(info["is_vhh"]))
            # Preserve the binding CDRs RFD4 designed; let ProteinMPNN re-pick the
            # FRAMEWORK (fix CDR positions). Only when anarcii actually found CDRs
            # — otherwise keep the RFD4 sequence as-is (don't risk destroying binding).
            if info["is_vhh"] and info["cdr_positions"] and pdb:
                try:
                    redesigned = call_proteinmpnn(
                        pdb, fixed_positions={"A": info["cdr_positions"]},
                        dev_user_prefix=dev_user_prefix)
                    new_seqs.append(redesigned[0] if redesigned else seq)
                except Exception as e:
                    print(f"  ⚠️ ProteinMPNN framework redesign failed: {e}; keeping RFD4 sequence")
                    new_seqs.append(seq)
            else:
                new_seqs.append(seq)
        seqs = new_seqs
    else:
        is_vhh_flags = [bool(number_vhh(s)["is_vhh"]) if s else False for s in seqs]

    # ESMFold each (possibly redesigned) sequence for a clean structure + pLDDT.
    folded_pdbs, plddts = [], []
    for s in seqs:
        if not s:
            folded_pdbs.append(""); plddts.append(0.0); continue
        try:
            out = call_esmfold(s, dev_user_prefix=dev_user_prefix)
            folded_pdbs.append(out["pdb"]); plddts.append(out["mean_plddt"])
        except Exception as e:
            print(f"  ESMFold failed: {e}")
            folded_pdbs.append(pdbs[len(folded_pdbs)] if len(folded_pdbs) < len(pdbs) else "")
            plddts.append(0.0)
    return {"seqs": seqs, "pdbs": folded_pdbs, "plddts": plddts, "is_vhh": is_vhh_flags}

# COMMAND ----------

# DBTITLE 1,Main loop
trajectory_rows: List[Dict[str, Any]] = []
all_candidates: List[Dict[str, Any]] = []

from mlflow.tracking import MlflowClient as _MlflowClient
_active_run_id = mlflow_run_id
if mlflow_run_id:
    _run_ctx = mlflow.start_run(run_id=mlflow_run_id)
else:
    _run_ctx = mlflow.start_run(run_name=mlflow_run_name or "antibody_design",
                                experiment_id=experiment.experiment_id)

try:
    with _run_ctx as run:
        _active_run_id = run.info.run_id
        mlflow.log_params({
            "format": "VHH",
            "vhh_length_min": vhh_length_min,
            "vhh_length_max": vhh_length_max,
            "num_samples": num_samples,
            "num_iterations": num_iterations,
            "antigen_chain": antigen_chain,
            "n_epitope_residues": len(epitope_residues),
            "run_proteinmpnn": run_proteinmpnn_flag,
            "cofold_antigen": cofold_antigen,
            "resampling_temperature": resampling_temperature,
            "strategy": strategy_name,
            "weights": weights_json,
            "n_references": len(references),
            "anchor_threshold": (None if math.isinf(anchor_threshold) else anchor_threshold),
        })
        mlflow.set_tag("origin", "genesis_workbench")
        mlflow.set_tag("feature", "antibody_design")
        mlflow.set_tag("created_by", user_email)
        mlflow.set_tag("job_status", "started")

        parents: List[Dict[str, Any]] = []
        gen_num = num_samples
        iter_max_history: List[float] = []
        cumulative_above_threshold = 0
        stop_reason: Optional[str] = None

        for it in range(num_iterations):
            print(f"\n=== Iteration {it+1}/{num_iterations} ===")
            if it > 0:
                proposed = strategy.propose(parents, [p["composite_reward"] for p in parents],
                                            vhh_length_min, vhh_length_max, num_samples)
                if proposed is None:
                    print("Strategy.propose() returned None — stopping (NoOpStrategy).")
                    break
                gen_num = proposed["num_samples"]

            mlflow.set_tag("job_status", f"iter_{it+1}_generating")
            batch = generate_and_validate(gen_num)
            seqs, pdbs, plddts, is_vhh = batch["seqs"], batch["pdbs"], batch["plddts"], batch["is_vhh"]

            mlflow.set_tag("job_status", f"iter_{it+1}_scoring")
            per_axis = score_iteration(seqs, pdbs, plddts, anchor_threshold)
            rewards = compose_rewards(per_axis, AXES)

            for k in range(len(seqs)):
                cand_id = f"iter{it+1}_cand{k+1}"
                row = {
                    "iteration": it + 1,
                    "candidate_id": cand_id,
                    "designed_sequence": seqs[k],
                    "is_vhh": bool(is_vhh[k]) if k < len(is_vhh) else False,
                    "composite_reward": float(rewards[k]),
                }
                for axis_name, vals in per_axis.items():
                    v = vals[k]
                    row[axis_name] = float(v) if not (isinstance(v, float) and math.isnan(v)) else None
                trajectory_rows.append(row)
                all_candidates.append({**row, "designed_pdb": pdbs[k]})

                for ak, av in row.items():
                    if ak in ("iteration", "candidate_id", "designed_sequence", "is_vhh"):
                        continue
                    if isinstance(av, (int, float)) and av is not None and not math.isnan(av):
                        mlflow.log_metric(f"{cand_id}/{ak}", float(av), step=it + 1)

                if pdbs[k]:
                    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as f:
                        f.write(pdbs[k]); pdb_local = f.name
                    mlflow.log_artifact(pdb_local, artifact_path=f"pdbs/iter_{it+1}")

            parents = [{**c, "designed_pdb": pdbs[i]}
                       for i, c in enumerate(trajectory_rows[-len(seqs):])]
            iter_max = float(max(rewards)); iter_mean = float(np.mean(rewards))
            mlflow.log_metric("iter_max_reward", iter_max, step=it + 1)
            mlflow.log_metric("iter_mean_reward", iter_mean, step=it + 1)
            mlflow.set_tag("job_status", f"iter_{it+1}_complete")
            iter_max_history.append(iter_max)

            # Stopping criteria (first to fire wins).
            if target_reward is not None and iter_max >= target_reward:
                stop_reason = f"target_reward (iter_max={iter_max:.4f} >= {target_reward:.4f})"
            elif best_k_target is not None and best_k_threshold is not None:
                cumulative_above_threshold += sum(1 for r in rewards if r >= best_k_threshold)
                mlflow.log_metric("cumulative_above_threshold", cumulative_above_threshold, step=it + 1)
                if cumulative_above_threshold >= best_k_target:
                    stop_reason = (f"best_k (cumulative={cumulative_above_threshold} >= "
                                   f"{best_k_target} above {best_k_threshold:.4f})")
            if (stop_reason is None and convergence_threshold >= 0
                    and len(iter_max_history) > convergence_window):
                window = iter_max_history[-(convergence_window + 1):]
                improvement = window[-1] - window[0]
                if improvement < convergence_threshold:
                    stop_reason = (f"convergence (improvement {improvement:.4f} over last "
                                   f"{convergence_window} iters < {convergence_threshold:.4f})")
            if stop_reason is not None:
                print(f"\nEarly exit at iteration {it+1}: {stop_reason}")
                mlflow.set_tag("stop_reason", stop_reason)
                mlflow.log_metric("iterations_completed", it + 1)
                break
        else:
            mlflow.set_tag("stop_reason", "n_ceiling")
            mlflow.log_metric("iterations_completed", num_iterations)

        # Final artifacts: ranked CSV + top-K PDBs.
        traj_df = pd.DataFrame(trajectory_rows).sort_values("composite_reward", ascending=False)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "reward_trajectory.csv")
            traj_df.to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path, artifact_path="results")
            topk = sorted(all_candidates, key=lambda r: r["composite_reward"], reverse=True)[:max(num_samples, 8)]
            topk_dir = os.path.join(tmp, "topK_pdbs"); os.makedirs(topk_dir, exist_ok=True)
            for r in topk:
                if r.get("designed_pdb"):
                    with open(os.path.join(topk_dir, r["candidate_id"] + ".pdb"), "w") as f:
                        f.write(r["designed_pdb"])
            mlflow.log_artifacts(topk_dir, artifact_path="results/topK_pdbs")

        mlflow.set_tag("job_status", "complete")
        print(f"\nDone. MLflow run id: {run.info.run_id}")
        if len(traj_df):
            print(f"Top candidate: {traj_df.iloc[0]['candidate_id']} "
                  f"(composite_reward={traj_df.iloc[0]['composite_reward']:.4f})")
except Exception as _exc:
    if _active_run_id:
        try:
            _MlflowClient().set_tag(_active_run_id, "job_status", "failed")
            _MlflowClient().set_tag(_active_run_id, "failure_reason", str(_exc)[:500])
        except Exception:
            pass
    raise
