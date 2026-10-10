# Databricks notebook source
# MAGIC %md
# MAGIC ### Vaccine Immunogen Design — Reward-Weighted Motif-Scaffolding Loop
# MAGIC
# MAGIC Orchestrator job for the Vaccine Immunogen Design tab. A vaccine teaches the immune system to
# MAGIC recognize a small patch on a pathogen — the **epitope**. That patch alone is usually floppy and
# MAGIC hard to manufacture. This workflow uses **RFD4-Proteina loaded in-process on an H100** to design a
# MAGIC brand-new, stable scaffold protein that holds the epitope locked in its native 3D conformation
# MAGIC ("keep the gem fixed, design the ring around it"), then optimizes the designs over a reward-weighted
# MAGIC loop. Each iteration:
# MAGIC
# MAGIC 1. Generate K scaffold candidates with RFD4 **motif-scaffolding** — the epitope motif is held FIXED
# MAGIC    in coordinates (`C_CRD`) and sequence (`C_SEQ`); the flanks are generated de novo.
# MAGIC 2. (Optional) ProteinMPNN-redesign the **scaffold** while FIXING the epitope residues (so the
# MAGIC    presented epitope sequence is never touched, only the carrier).
# MAGIC 3. ESMFold each candidate → structure + mean pLDDT (scaffold fold confidence).
# MAGIC 4. Score: **motif RMSD** (fold the design, measure how faithfully it presents the epitope — the
# MAGIC    headline axis, lower is better), scaffold pLDDT, solubility (NetSolP), Tm (DeepSTABp), a
# MAGIC    sequence-liability scan (manufacturability), and OPTIONAL scaffold self-reactivity (MHCflurry
# MAGIC    MHC-I / HLAIIPred MHC-II, off by default — a vaccine is meant to be immunogenic).
# MAGIC 5. Compose a per-candidate composite reward (z-score→min-max within the batch, weighted sum),
# MAGIC    log to MLflow, resample parents for the next iteration.
# MAGIC
# MAGIC Dispatched by `start_vaccine_immunogen_job` in `modules/core/app/backend/app/services/vaccine_immunogen.py`.
# MAGIC
# MAGIC > ⚠️ FIRST-DRAFT conditioning. The motif is centered with symmetric flexible flanks (see
# MAGIC > `utils._flank_segments`); a terminal placement or a discontinuous (multi-segment) epitope is a
# MAGIC > future refinement. Expect a deploy-time iteration or two on the condition_spec — same caveat as
# MAGIC > the antibody_design + RFD4 register notebooks.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("cache_dir", "vaccine_immunogen", "Cache dir (UC volume) for this workflow")
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
dbutils.widgets.text("motif_pdb_path", "", "UC volume path to the epitope motif PDB")
dbutils.widgets.text("motif_residues_csv", "", "Epitope motif residue numbers (CSV ints; empty = whole chain)")
dbutils.widgets.text("motif_chain", "A", "Motif chain id in the input PDB")
dbutils.widgets.text("scaffold_length_min", "80", "Total designed-scaffold length min (incl. the motif)")
dbutils.widgets.text("scaffold_length_max", "120", "Total designed-scaffold length max (incl. the motif)")
dbutils.widgets.text("num_samples", "8", "K — candidates per iteration")
dbutils.widgets.text("num_iterations", "6", "N — iteration ceiling (convergence usually exits earlier)")
dbutils.widgets.text("run_proteinmpnn", "true", "ProteinMPNN-redesign the scaffold (fix the epitope) per candidate")
dbutils.widgets.text("resampling_temperature", "0.1", "Resampling softmax temperature")
dbutils.widgets.text("strategy", "resample", "Strategy: resample | noop")
dbutils.widgets.text("dev_user_prefix", "", "Dev user prefix (matches DEV_USER_PREFIX)")
dbutils.widgets.text(
    "weights_json",
    '{"motif_rmsd":2.0,"plddt":1.5,"solubility":1.0,"thermostab":1.0,"liability":1.0,"immuno":0.0,"immuno_mhc2":0.0}',
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
motif_pdb_path = g("motif_pdb_path")
motif_residues_csv = g("motif_residues_csv")
motif_chain = g("motif_chain")
scaffold_length_min = int(g("scaffold_length_min"))
scaffold_length_max = int(g("scaffold_length_max"))
num_samples = int(g("num_samples"))
num_iterations = int(g("num_iterations"))
run_proteinmpnn_flag = g("run_proteinmpnn").lower() in ("true", "1", "yes")
resampling_temperature = float(g("resampling_temperature"))
strategy_name = g("strategy")
dev_user_prefix = g("dev_user_prefix")
weights_json = g("weights_json")

import re
def _residue_num(tok: str) -> int:
    m = re.search(r"(\d+)\s*$", tok.strip())
    if not m:
        raise ValueError(f"motif residue {tok!r} has no residue number — use plain integers like '254,255,256'.")
    return int(m.group(1))

motif_residues = [_residue_num(r) for r in motif_residues_csv.split(",") if r.strip()]
weights = json.loads(weights_json) if weights_json else {}

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

print(f"motif_pdb_path:    {motif_pdb_path}")
print(f"motif_residues:    {motif_residues or '(whole chain)'}")
print(f"motif_chain:       {motif_chain}")
print(f"scaffold_length:   [{scaffold_length_min}, {scaffold_length_max}]")
print(f"K (candidates):    {num_samples}   N (iterations): {num_iterations}")
print(f"weights:           {weights}")
print(f"flow_ckpt:         {flow_ckpt_path}")
print(f"ae_ckpt:           {ae_ckpt_path}")

# COMMAND ----------

# DBTITLE 1,Load utils.py from the bundle's notebooks/ directory
_notebook_dir = os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
sys.path.insert(0, "/Workspace" + _notebook_dir)
import importlib
import utils as vi_utils
importlib.reload(vi_utils)
from utils import (
    PredictorAxis, compose_rewards, make_strategy,
    load_rfd4, generate_scaffold,
    call_esmfold, call_proteinmpnn,
    call_netsolp, call_deepstabp, call_mhcflurry, call_hlaiipred,
    warmup_developability_endpoints, _extract_mean_plddt_from_pdb,
    _chain_res_range, motif_sequence, locate_motif_in_sequence, motif_backbone_rmsd_located,
    liability_weighted_count, liability_detail,
)

# COMMAND ----------

# DBTITLE 1,Initialize GWB + MLflow experiment
from genesis_workbench.workbench import initialize
from genesis_workbench.models import set_mlflow_experiment

databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema,
           sql_warehouse_id=sql_warehouse_id, token=databricks_token)
experiment = set_mlflow_experiment(experiment_tag=mlflow_experiment or "vaccine_immunogen", user_email=user_email)
mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# COMMAND ----------

# DBTITLE 1,Read the epitope motif PDB the app wrote to the cache volume + resolve the motif range
with open(motif_pdb_path, "r") as f:
    motif_pdb_str = f.read()
print(f"Loaded epitope motif PDB ({len(motif_pdb_str)} chars) from {motif_pdb_path}")

# Motif range for the contig: min/max of the given residues, else the whole chain.
if motif_residues:
    motif_lo, motif_hi = min(motif_residues), max(motif_residues)
else:
    _rng = _chain_res_range(motif_pdb_str, motif_chain)
    if _rng is None:
        raise RuntimeError(f"Motif chain '{motif_chain}' not found in the input PDB.")
    motif_lo, motif_hi = _rng
motif_seq = motif_sequence(motif_pdb_str, motif_chain, motif_lo, motif_hi)
print(f"motif range: {motif_chain}{motif_lo}-{motif_hi}  ({motif_hi - motif_lo + 1} residues)")
print(f"motif sequence: {motif_seq}")
if scaffold_length_min <= (motif_hi - motif_lo + 1):
    raise RuntimeError(
        f"scaffold_length_min ({scaffold_length_min}) must exceed the motif length "
        f"({motif_hi - motif_lo + 1}) — there must be room for a scaffold around the epitope."
    )

# COMMAND ----------

# DBTITLE 1,Reward axes + strategy
AXES = [
    PredictorAxis("motif_rmsd",  weights.get("motif_rmsd", 0.0),  lower_is_better=True),
    PredictorAxis("plddt",       weights.get("plddt", 0.0)),
    PredictorAxis("solubility",  weights.get("solubility", 0.0)),
    PredictorAxis("thermostab",  weights.get("thermostab", 0.0)),
    PredictorAxis("immuno",      weights.get("immuno", 0.0), lower_is_better=True),
    PredictorAxis("immuno_mhc2", weights.get("immuno_mhc2", 0.0), lower_is_better=True),
    PredictorAxis("liability",   weights.get("liability", 0.0), lower_is_better=True),
]
print("Enabled axes:", [a.name for a in AXES if a.enabled])

strategy = make_strategy(strategy_name, temperature=resampling_temperature)

# COMMAND ----------

# DBTITLE 1,Load RFD4-Proteina in-process (H100) + warm the scoring endpoints
import torch
if not torch.cuda.is_available():
    raise RuntimeError("Vaccine Immunogen Design requires a GPU (H100) — torch.cuda.is_available() is False. "
                       "The orchestrator job is pinned to GPU_1xH100; check the job's compute.")
rfd4_ctx = load_rfd4(flow_ckpt_path, ae_ckpt_path)
print("Warming developability endpoints...")
print(warmup_developability_endpoints(dev_user_prefix=dev_user_prefix))

# COMMAND ----------

# DBTITLE 1,Per-batch scoring across all enabled axes
def score_iteration(seqs: List[str], pdbs: List[str], plddts: List[float],
                    motif_positions: List[List[int]]) -> Dict[str, List[float]]:
    K = len(seqs)
    scores: Dict[str, List[float]] = {}

    # motif_rmsd: fold-consistency of the epitope presentation. Lower = the scaffold
    # folds to present the epitope in its native conformation = a better immunogen.
    if any(a.name == "motif_rmsd" and a.enabled for a in AXES):
        scores["motif_rmsd"] = [
            motif_backbone_rmsd_located(motif_pdb_str, motif_chain, motif_lo, motif_hi,
                                        pdbs[k], motif_positions[k])
            for k in range(K)
        ]

    if any(a.name == "plddt" and a.enabled for a in AXES):
        scores["plddt"] = [float(p) for p in plddts]
    if any(a.name == "solubility" and a.enabled for a in AXES):
        scores["solubility"] = call_netsolp(seqs, dev_user_prefix=dev_user_prefix)
    if any(a.name == "thermostab" and a.enabled for a in AXES):
        scores["thermostab"] = call_deepstabp(seqs, dev_user_prefix=dev_user_prefix)
    # OPTIONAL scaffold self-reactivity — off by default (a vaccine is meant to be immunogenic).
    if any(a.name == "immuno" and a.enabled for a in AXES):
        scores["immuno"] = call_mhcflurry(seqs, dev_user_prefix=dev_user_prefix)
    if any(a.name == "immuno_mhc2" and a.enabled for a in AXES):
        scores["immuno_mhc2"] = call_hlaiipred(seqs, dev_user_prefix=dev_user_prefix)
    # Sequence-liability scan — rule-based, no endpoint. Weighted motif count (lower = more manufacturable).
    if any(a.name == "liability" and a.enabled for a in AXES):
        scores["liability"] = [liability_weighted_count(s) for s in seqs]
    return scores

# COMMAND ----------

# DBTITLE 1,Generate + validate one batch of scaffold candidates
def generate_and_validate(num: int) -> Dict[str, List]:
    """RFD4 motif-scaffold → (optional) ProteinMPNN scaffold redesign fixing the
    epitope → ESMFold. Returns dict of parallel lists (incl. the located motif
    positions in each FINAL sequence, for the motif-RMSD axis)."""
    gen_df = generate_scaffold(rfd4_ctx, motif_pdb_str, motif_chain,
                               (motif_lo, motif_hi), scaffold_length_min, scaffold_length_max, num)
    seqs = list(gen_df["designed_sequence"])
    pdbs = list(gen_df["designed_pdb"])

    if run_proteinmpnn_flag:
        new_seqs = []
        for seq, pdb in zip(seqs, pdbs):
            # Fix the grafted epitope (locate it in the RFD4 sequence) and let
            # ProteinMPNN re-pick the surrounding scaffold. Only redesign when the
            # epitope is found verbatim — otherwise keep the RFD4 sequence as-is.
            positions = locate_motif_in_sequence(seq, motif_seq)
            if positions and pdb:
                try:
                    redesigned = call_proteinmpnn(
                        pdb, fixed_positions={"A": positions},
                        dev_user_prefix=dev_user_prefix)
                    new_seqs.append(redesigned[0] if redesigned else seq)
                except Exception as e:
                    print(f"  ⚠️ ProteinMPNN scaffold redesign failed: {e}; keeping RFD4 sequence")
                    new_seqs.append(seq)
            else:
                new_seqs.append(seq)
        seqs = new_seqs

    # ESMFold each (possibly redesigned) sequence for a clean structure + pLDDT, then
    # locate the epitope in the FINAL sequence for the motif-RMSD axis.
    folded_pdbs, plddts, motif_positions = [], [], []
    for s in seqs:
        if not s:
            folded_pdbs.append(""); plddts.append(0.0); motif_positions.append([]); continue
        try:
            out = call_esmfold(s, dev_user_prefix=dev_user_prefix)
            folded_pdbs.append(out["pdb"]); plddts.append(out["mean_plddt"])
        except Exception as e:
            print(f"  ESMFold failed: {e}")
            folded_pdbs.append(""); plddts.append(0.0)
        motif_positions.append(locate_motif_in_sequence(s, motif_seq))
    return {"seqs": seqs, "pdbs": folded_pdbs, "plddts": plddts, "motif_positions": motif_positions}

# COMMAND ----------

# DBTITLE 1,Main loop
trajectory_rows: List[Dict[str, Any]] = []
all_candidates: List[Dict[str, Any]] = []

from mlflow.tracking import MlflowClient as _MlflowClient
_active_run_id = mlflow_run_id
if mlflow_run_id:
    _run_ctx = mlflow.start_run(run_id=mlflow_run_id)
else:
    _run_ctx = mlflow.start_run(run_name=mlflow_run_name or "vaccine_immunogen",
                                experiment_id=experiment.experiment_id)

try:
    with _run_ctx as run:
        _active_run_id = run.info.run_id
        mlflow.log_params({
            "format": "immunogen",
            "scaffold_length_min": scaffold_length_min,
            "scaffold_length_max": scaffold_length_max,
            "num_samples": num_samples,
            "num_iterations": num_iterations,
            "motif_chain": motif_chain,
            "motif_range": f"{motif_lo}-{motif_hi}",
            "motif_length": motif_hi - motif_lo + 1,
            "run_proteinmpnn": run_proteinmpnn_flag,
            "resampling_temperature": resampling_temperature,
            "strategy": strategy_name,
            "weights": weights_json,
        })
        mlflow.set_tag("origin", "genesis_workbench")
        mlflow.set_tag("feature", "vaccine_immunogen")
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
                                            scaffold_length_min, scaffold_length_max, num_samples)
                if proposed is None:
                    print("Strategy.propose() returned None — stopping (NoOpStrategy).")
                    break
                gen_num = proposed["num_samples"]

            mlflow.set_tag("job_status", f"iter_{it+1}_generating")
            batch = generate_and_validate(gen_num)
            seqs, pdbs, plddts = batch["seqs"], batch["pdbs"], batch["plddts"]
            motif_positions = batch["motif_positions"]

            mlflow.set_tag("job_status", f"iter_{it+1}_scoring")
            per_axis = score_iteration(seqs, pdbs, plddts, motif_positions)
            rewards = compose_rewards(per_axis, AXES)

            for k in range(len(seqs)):
                cand_id = f"iter{it+1}_cand{k+1}"
                row = {
                    "iteration": it + 1,
                    "candidate_id": cand_id,
                    "designed_sequence": seqs[k],
                    "epitope_presented": bool(motif_positions[k]),
                    "composite_reward": float(rewards[k]),
                }
                for axis_name, vals in per_axis.items():
                    v = vals[k]
                    row[axis_name] = float(v) if not (isinstance(v, float) and math.isnan(v)) else None
                # Human-readable liability breakdown for the result dialog (string → not a metric).
                row["liability_detail"] = liability_detail(seqs[k])
                trajectory_rows.append(row)
                all_candidates.append({**row, "designed_pdb": pdbs[k]})

                for ak, av in row.items():
                    if ak in ("iteration", "candidate_id", "designed_sequence", "epitope_presented"):
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
