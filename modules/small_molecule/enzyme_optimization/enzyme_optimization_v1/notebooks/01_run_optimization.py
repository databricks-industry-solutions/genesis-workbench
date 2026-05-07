# Databricks notebook source
# MAGIC %md
# MAGIC ### Guided Enzyme Optimization — Reward-Weighted Resampling Loop
# MAGIC
# MAGIC The orchestrator job for the Guided Enzyme Optimization tab. Each iteration:
# MAGIC
# MAGIC 1. Generate K candidates via Proteina-Complexa-AME.
# MAGIC 2. (Optional) ProteinMPNN redesign each scaffold's sequence.
# MAGIC 3. ESMFold each candidate → structure + mean pLDDT.
# MAGIC 4. Score each candidate on every enabled axis (motif RMSD, pLDDT, optional
# MAGIC    Boltz complex confidence, NetSolP, PLTNUM-anchored half-life, DeepSTABp Tm,
# MAGIC    MHCflurry immuno burden).
# MAGIC 5. Compose a per-candidate composite reward (z-score+min-max within the batch,
# MAGIC    weighted sum) and log to MLflow.
# MAGIC 6. Resample parents for the next iteration via the configured Strategy.
# MAGIC
# MAGIC The loop is dispatched by `start_enzyme_optimization_job` in
# MAGIC `modules/core/app/utils/enzyme_optimization_tools.py`.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("cache_dir", "enzyme_optimization", "Cache dir (UC volume)")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")

dbutils.widgets.text("mlflow_experiment", "", "MLflow experiment path")
dbutils.widgets.text("mlflow_run_name", "", "MLflow run name")
dbutils.widgets.text("motif_pdb_path", "", "UC volume path to the motif PDB")
dbutils.widgets.text("motif_residues_csv", "", "Motif residue numbers (CSV ints)")
dbutils.widgets.text("target_chain", "A", "Motif chain id in the input PDB")
dbutils.widgets.text("scaffold_length_min", "80", "Scaffold length min")
dbutils.widgets.text("scaffold_length_max", "120", "Scaffold length max")
dbutils.widgets.text("num_samples", "8", "K — candidates per iteration")
dbutils.widgets.text("num_iterations", "10", "N — iteration count (ceiling; convergence usually exits earlier)")
dbutils.widgets.text("substrate_smiles", "", "Substrate SMILES (optional, gates Boltz axis)")
dbutils.widgets.text("references_json", "[]", "Reference enzymes JSON list")
dbutils.widgets.text("half_life_margin", "0.05", "Half-life anchor margin")
dbutils.widgets.text(
    "weights_json",
    '{"motif_rmsd":1.0,"plddt":1.0,"boltz":0.5,"solubility":1.0,"half_life":1.0,"thermostab":1.0,"immuno":1.0}',
    "Per-axis weights JSON",
)
dbutils.widgets.text("resampling_temperature", "0.1", "Resampling softmax temperature")
dbutils.widgets.text("strategy", "resample", "Strategy: resample | noop")
dbutils.widgets.text("run_proteinmpnn", "true", "Run ProteinMPNN redesign per scaffold")
dbutils.widgets.text("dev_user_prefix", "", "Dev user prefix (matches DEV_USER_PREFIX)")

# Stopping criteria — convergence ON by default; the other two are opt-in
# (an empty string is parsed as "disabled" by _parse_optional_float / int below).
dbutils.widgets.text("convergence_threshold", "0.01", "Convergence: min iter_max_reward improvement; negative disables")
dbutils.widgets.text("convergence_window", "2", "Convergence: number of iterations to compare")
dbutils.widgets.text("target_reward", "", "Threshold stop: composite reward target (empty = disabled)")
dbutils.widgets.text("best_k_target", "", "Best-K stop: number of candidates above threshold (empty = disabled)")
dbutils.widgets.text("best_k_threshold", "", "Best-K stop: composite reward threshold (empty = disabled)")

# Phase 1.5 — generation-mode toggle and FK-steering knobs. The toggle is set
# on the dispatcher side (Streamlit form → start_enzyme_optimization_job) and
# the cluster spec for each job pre-defaults this widget; we still read it at
# runtime so a misdispatch fails loud rather than silently mis-running.
dbutils.widgets.text("use_inprocess_ame", "false", "Generation mode: false=Fast (endpoint), true=Accurate (in-process AME + FK steering)")
dbutils.widgets.text("fk_n_branch", "4", "FK steering: branches per checkpoint")
dbutils.widgets.text("fk_beam_width", "4", "FK steering: beam width")
dbutils.widgets.text("fk_temperature", "1.0", "FK steering: importance-sampling temperature")
dbutils.widgets.text("fk_step_checkpoints", "0,25,50,75,100", "FK steering: denoising-step checkpoints (CSV ints; last value must equal nsteps)")

# COMMAND ----------

# Phase 1.5 Accurate-path install must run *before* torch is loaded into this
# kernel by anything else (mlflow, biopython, etc. transitively pull torch on
# the GPU runtime). The Databricks 16.4.x-gpu-ml image ships its own torch;
# our requirements.txt pins torch==2.7.0 to match the PyG cu126 wheel ABI,
# but installing 2.7.0 with the runtime's torch already imported leaves
# torch_scatter referencing the wrong C++ symbols and crashing on first use.
#
# Solution: read the toggle, install proteinfoundation deps if needed, then
# restartPython() so the new torch is the one loaded by every subsequent
# import. After restart, the notebook resumes from the next cell with a
# fresh kernel — widgets persist, Python state does not.
import os, sys, subprocess

_use_inprocess_ame_early = (
    dbutils.widgets.get("use_inprocess_ame").lower() in ("true", "1", "yes")
)

if _use_inprocess_ame_early:
    # Resolve the requirements.txt path the same way utils.py does later, but
    # without depending on utils.py (we can't import it before the install).
    _notebook_dir = "/Workspace" + os.path.dirname(
        dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    )
    _req_path = os.path.join(_notebook_dir, "proteinfoundation_requirements.txt")
    print(f"[install] Accurate path — installing proteinfoundation deps from {_req_path}")
    _r = subprocess.run(
        ["pip", "install", "--find-links",
         "https://data.pyg.org/whl/torch-2.7.0+cu126.html",
         "-r", _req_path],
        capture_output=True, text=True,
    )
    if _r.stdout: print(_r.stdout)
    if _r.stderr: print("[install][stderr]", _r.stderr)
    if _r.returncode != 0:
        raise RuntimeError(
            f"proteinfoundation deps install failed (exit {_r.returncode}). "
            "See stderr above."
        )
    # Graphein has unique build-isolation needs.
    _r2 = subprocess.run(
        ["pip", "install", "--no-build-isolation", "--no-deps", "graphein==1.7.7"],
        capture_output=True, text=True,
    )
    if _r2.stdout: print(_r2.stdout)
    if _r2.stderr: print("[install][stderr]", _r2.stderr)
    if _r2.returncode != 0:
        raise RuntimeError(f"graphein install failed (exit {_r2.returncode})")
    # Clone proteinfoundation upstream + install --no-deps.
    _repo_dir = "/tmp/proteina_complexa_repo"
    if not os.path.exists(_repo_dir):
        subprocess.check_call([
            "git", "clone", "--branch", "dev",
            "https://github.com/NVIDIA-Digital-Bio/Proteina-Complexa.git",
            _repo_dir,
        ])
    subprocess.check_call(["pip", "install", "--no-deps", _repo_dir])
    print("[install] proteinfoundation installed; restarting Python kernel so the new torch becomes active...")
    dbutils.library.restartPython()
    # ↑ everything below this line runs in a fresh Python kernel. Cells above
    # have been executed; this cell terminates here on the Accurate path.

# COMMAND ----------

# Fast-path-only basic deps install. The Accurate path already installed all
# of these (and the heavier proteinfoundation stack) in the early-install
# cell above and restarted the kernel; running this %pip install again would
# pull conflicting transitive versions on top of biotite==1.4.0 / torch==2.7.0
# / etc. Use a Python subprocess gated on the toggle so we can skip cleanly.
import subprocess as _subprocess
if not (dbutils.widgets.get("use_inprocess_ame").lower() in ("true", "1", "yes")):
    print("[install] Fast path — installing basic deps")
    _subprocess.check_call([
        "pip", "install",
        "databricks-sdk==0.50.0",
        "databricks-sql-connector==4.0.2",
        "mlflow==2.22.0",
        "numpy==1.26.4",
        "pandas==1.5.3",
        "biopython==1.84",
    ])
else:
    print("[install] Accurate path — basic deps already covered by proteinfoundation_requirements.txt; skipping")

# COMMAND ----------

import os, sys, json, math, tempfile, shutil
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import mlflow

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# Resolve the GWB library wheel and install for genesis_workbench.workbench/.models imports.
gwb_library_path = None
for lib in dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries"):
    if lib.name.startswith("genesis_workbench"):
        gwb_library_path = lib.path.replace("dbfs:", "")
print(f"GWB library wheel: {gwb_library_path}")

# COMMAND ----------

# MAGIC %pip install {gwb_library_path} --force-reinstall
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os, sys, json, math, tempfile, shutil
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import mlflow

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
cache_dir = dbutils.widgets.get("cache_dir")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")
user_email = dbutils.widgets.get("user_email")

mlflow_experiment = dbutils.widgets.get("mlflow_experiment")
mlflow_run_name = dbutils.widgets.get("mlflow_run_name")
motif_pdb_path = dbutils.widgets.get("motif_pdb_path")
motif_residues_csv = dbutils.widgets.get("motif_residues_csv")
target_chain = dbutils.widgets.get("target_chain")
scaffold_length_min = int(dbutils.widgets.get("scaffold_length_min"))
scaffold_length_max = int(dbutils.widgets.get("scaffold_length_max"))
num_samples = int(dbutils.widgets.get("num_samples"))
num_iterations = int(dbutils.widgets.get("num_iterations"))
substrate_smiles = dbutils.widgets.get("substrate_smiles").strip()
references_json = dbutils.widgets.get("references_json")
half_life_margin = float(dbutils.widgets.get("half_life_margin"))
weights_json = dbutils.widgets.get("weights_json")
resampling_temperature = float(dbutils.widgets.get("resampling_temperature"))
strategy_name = dbutils.widgets.get("strategy")
run_proteinmpnn_flag = dbutils.widgets.get("run_proteinmpnn").lower() in ("true", "1", "yes")
dev_user_prefix = dbutils.widgets.get("dev_user_prefix")

motif_residues = [int(r.strip()) for r in motif_residues_csv.split(",") if r.strip()]
weights = json.loads(weights_json) if weights_json else {}
references = json.loads(references_json) if references_json else []


def _parse_optional_float(s: str) -> "Optional[float]":
    s = (s or "").strip()
    return float(s) if s else None


def _parse_optional_int(s: str) -> "Optional[int]":
    s = (s or "").strip()
    return int(s) if s else None


convergence_threshold = float(dbutils.widgets.get("convergence_threshold") or "0.01")
convergence_window = int(dbutils.widgets.get("convergence_window") or "2")
target_reward = _parse_optional_float(dbutils.widgets.get("target_reward"))
best_k_target = _parse_optional_int(dbutils.widgets.get("best_k_target"))
best_k_threshold = _parse_optional_float(dbutils.widgets.get("best_k_threshold"))

use_inprocess_ame = dbutils.widgets.get("use_inprocess_ame").lower() in ("true", "1", "yes")
fk_n_branch = int(dbutils.widgets.get("fk_n_branch") or "4")
fk_beam_width = int(dbutils.widgets.get("fk_beam_width") or "4")
fk_temperature = float(dbutils.widgets.get("fk_temperature") or "1.0")
fk_step_checkpoints = [
    int(x.strip()) for x in dbutils.widgets.get("fk_step_checkpoints").split(",") if x.strip()
] or [0, 25, 50, 75, 100]   # last must equal inf_cfg.args.nsteps (=100)

print(f"motif_pdb_path:      {motif_pdb_path}")
print(f"motif_residues:      {motif_residues}")
print(f"target_chain:        {target_chain}")
print(f"scaffold_length:     [{scaffold_length_min}, {scaffold_length_max}]")
print(f"K (candidates/iter): {num_samples}")
print(f"N (iterations):      {num_iterations}")
print(f"substrate_smiles:    {substrate_smiles or '(none — Boltz axis disabled)'}")
print(f"weights:             {weights}")
print(f"strategy:            {strategy_name}")
print(f"references:          {len(references)} reference enzyme(s) supplied")

# COMMAND ----------

# Load utils.py from the bundle's notebooks/ directory.
_notebook_dir = os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
_utils_path = "/Workspace" + _notebook_dir
sys.path.insert(0, _utils_path)
import utils as orchestrator_utils

# Re-resolve so we always pick up edits.
import importlib
importlib.reload(orchestrator_utils)
from utils import (
    PredictorAxis,
    compose_rewards,
    half_life_anchor_threshold,
    half_life_anchor_rewards,
    make_strategy,
    call_ame,
    call_esmfold,
    call_proteinmpnn,
    call_boltz,
    call_netsolp,
    call_pltnum,
    call_deepstabp,
    call_mhcflurry,
)

# COMMAND ----------

from genesis_workbench.workbench import initialize
from genesis_workbench.models import set_mlflow_experiment

databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(
    core_catalog_name=catalog,
    core_schema_name=schema,
    sql_warehouse_id=sql_warehouse_id,
    token=databricks_token,
)

experiment = set_mlflow_experiment(
    experiment_tag=mlflow_experiment or "enzyme_optimization",
    user_email=user_email,
)
mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# COMMAND ----------

# Read the motif PDB the app wrote into the cache volume.
with open(motif_pdb_path, "r") as f:
    motif_pdb_str = f.read()
print(f"Loaded motif PDB ({len(motif_pdb_str)} chars) from {motif_pdb_path}")

# COMMAND ----------

# Build the reward axes from the user-supplied weights.
AXES = [
    PredictorAxis("motif_rmsd",  weights.get("motif_rmsd", 0.0),  lower_is_better=True),
    PredictorAxis("plddt",       weights.get("plddt", 0.0)),
    PredictorAxis("boltz",       weights.get("boltz", 0.0)),
    PredictorAxis("solubility",  weights.get("solubility", 0.0)),
    PredictorAxis("half_life",   weights.get("half_life", 0.0), pre_normalized=True),
    PredictorAxis("thermostab",  weights.get("thermostab", 0.0)),
    PredictorAxis("immuno",      weights.get("immuno", 0.0),    lower_is_better=True),
]
print("Enabled axes:", [a.name for a in AXES if a.enabled])

# Pre-compute the half-life anchor threshold from references' PLTNUM scores.
anchor_threshold = -math.inf
if any(a.name == "half_life" and a.enabled for a in AXES) and references:
    ref_seqs = [r["sequence"] for r in references]
    ref_scores = call_pltnum(ref_seqs, dev_user_prefix=dev_user_prefix)
    anchor_threshold = half_life_anchor_threshold(ref_scores, margin=half_life_margin)
    print(f"PLTNUM anchor threshold: {anchor_threshold:.4f}  (from {len(ref_scores)} reference scores)")
else:
    print("Half-life axis disabled or no references provided — anchor threshold = -inf")

strategy = make_strategy(strategy_name, temperature=resampling_temperature)

# ─── Endpoint warmup (both paths) ───────────────────────────────────────────
# Every endpoint we use has scale_to_zero=true. Without a warmup at job start,
# the first call to each (mid-loop) hits a 5-20 min cold start — we got bitten
# in three consecutive Phase 1.5 verification runs (AME iter-2 cold start,
# DeepSTABp iter-1 scoring cold start, AME iter-1 >20 min cold start). Pay
# the warmup cost once up front so the loop sees only warm endpoints.
#
# Generation-side warmup (AME, ESMFold, ProteinMPNN) is conditional on the
# Fast path because the Accurate path runs AME in-process — only ESMFold and
# ProteinMPNN are still endpoint-served on that side, and warming them
# remains useful.
from utils import warmup_developability_endpoints, warmup_generation_endpoints
print("Warming up developability endpoints (one dummy call each)...")
_dev_warmup = warmup_developability_endpoints(dev_user_prefix=dev_user_prefix)
print(f"Developability warmup: {_dev_warmup}")
print("Warming up generation endpoints (AME 1-sample, ESMFold tiny, ProteinMPNN)...")
_gen_warmup = warmup_generation_endpoints(
    motif_pdb_str=motif_pdb_str, target_chain=target_chain,
    dev_user_prefix=dev_user_prefix,
)
print(f"Generation warmup: {_gen_warmup}")

# ─── Phase 1.5 Accurate-path setup ──────────────────────────────────────────
# When the dispatcher routed us to the GPU job (use_inprocess_ame=true), also
# install proteinfoundation, load AME on-cluster, and build the FK-steering
# reward instance before the loop kicks off. The Fast path skips these.
ame_model = None
inprocess_reward = None
if use_inprocess_ame:
    import torch
    if not torch.cuda.is_available():
        # Defensive: the CPU job's spec defaults use_inprocess_ame=false. Reaching
        # here means something mis-routed (the dispatcher pointed the toggle at the
        # CPU job by mistake). Bail loud now rather than crashing mid-pip-install.
        raise RuntimeError(
            "use_inprocess_ame=true requires CUDA, but torch.cuda.is_available() "
            "returned False. The Streamlit dispatcher should route Accurate-mode "
            "jobs to run_enzyme_optimization_gwb_inprocess_ame (A10 GPU cluster)."
        )
    from utils import (
        load_ame_model,
        DevelopabilityCompositeReward,
    )
    # proteinfoundation was installed at the top of this notebook (in the
    # early-install cell) BEFORE the kernel restart, so torch and torch_scatter
    # are now in their correct ABI-matched state. Just verify it's importable.
    import proteinfoundation  # noqa: F401 — fast-fail if the early install missed
    ame_model = load_ame_model(catalog, schema, cache_dir)
    inprocess_reward = DevelopabilityCompositeReward(
        weights=weights,
        anchor_threshold=anchor_threshold,
        dev_user_prefix=dev_user_prefix,
    )
    print("[Accurate path] AME model + reward ready.")
else:
    print("[Fast path] Using endpoint-based AME (gwb_*_proteina_complexa_ame_endpoint).")

# COMMAND ----------

from utils import _extract_mean_plddt_from_pdb

def _motif_backbone_rmsd(input_pdb, designed_pdb, motif_residues, input_chain, designed_chain="A"):
    """Backbone (N, CA, C) RMSD over the specified motif residues. Vendored
    minimal version of `modules/core/app/utils/structure_utils.py:motif_backbone_rmsd`
    so this orchestrator notebook doesn't need the app utils at runtime.

    Returns NaN for empty/missing input — common when an upstream call
    (ProteinMPNN, ESMFold) failed for a candidate; the reward composer treats
    NaN as "skip this axis" instead of crashing the whole iteration."""
    if not input_pdb or not designed_pdb:
        return float("nan")
    import Bio.PDB as bp
    pdb_parser = bp.PDBParser(QUIET=True)
    with tempfile.TemporaryDirectory() as tmp:
        ip, dp = os.path.join(tmp, "in.pdb"), os.path.join(tmp, "de.pdb")
        with open(ip, "w") as f: f.write(input_pdb)
        with open(dp, "w") as f: f.write(designed_pdb)
        try:
            in_struct = pdb_parser.get_structure("in", ip)
            de_struct = pdb_parser.get_structure("de", dp)
        except Exception:
            return float("nan")

    motif_set = set(motif_residues)
    def _bb(structure, chain_id):
        atoms = []
        chain = structure[0][chain_id]
        for residue in chain:
            if residue.id[0] != " ": continue
            if residue.id[1] not in motif_set: continue
            for name in ("N", "CA", "C"):
                if name in residue:
                    atoms.append(residue[name])
        return atoms

    in_atoms = _bb(in_struct, input_chain)
    de_atoms = _bb(de_struct, designed_chain)
    if len(in_atoms) != len(de_atoms) or not in_atoms:
        return float("nan")
    imp = bp.Superimposer()
    imp.set_atoms(in_atoms, de_atoms)
    return float(imp.rms)

# COMMAND ----------

def score_iteration(seqs: List[str], pdbs: List[str],
                    references_pltnum_threshold: float) -> Dict[str, List[float]]:
    """Run all enabled scoring axes for a batch of K candidates.
    Returns a dict {axis_name: List[float]} of length K."""
    K = len(seqs)
    scores: Dict[str, List[float]] = {}

    # motif_rmsd: per-candidate RMSD against the input motif
    if any(a.name == "motif_rmsd" and a.enabled for a in AXES):
        scores["motif_rmsd"] = [
            _motif_backbone_rmsd(motif_pdb_str, pdb, motif_residues, target_chain, "A")
            for pdb in pdbs
        ]

    # pLDDT: parsed from each ESMFold PDB's CA B-factors
    if any(a.name == "plddt" and a.enabled for a in AXES):
        scores["plddt"] = [_extract_mean_plddt_from_pdb(pdb) for pdb in pdbs]

    # Boltz: only if substrate SMILES given
    if any(a.name == "boltz" and a.enabled for a in AXES) and substrate_smiles:
        boltz_scores = []
        for seq in seqs:
            boltz_in = f"protein_A:{seq};smiles_B:{substrate_smiles}"
            try:
                out = call_boltz(boltz_in, dev_user_prefix=dev_user_prefix, timeout_seconds=900)
                boltz_scores.append(float(out.get("ipTM", out.get("iLDDT", 0.0))))
            except Exception as e:
                print(f"Boltz failed for one candidate: {e}")
                boltz_scores.append(0.0)
        scores["boltz"] = boltz_scores

    # Batched calls: one round-trip per axis, all K sequences.
    if any(a.name == "solubility" and a.enabled for a in AXES):
        scores["solubility"] = call_netsolp(seqs, dev_user_prefix=dev_user_prefix)

    if any(a.name == "thermostab" and a.enabled for a in AXES):
        scores["thermostab"] = call_deepstabp(seqs, dev_user_prefix=dev_user_prefix)

    if any(a.name == "immuno" and a.enabled for a in AXES):
        scores["immuno"] = call_mhcflurry(seqs, dev_user_prefix=dev_user_prefix)

    # Half-life: PLTNUM raw → anchor-based reward in [0,1] (pre-normalized).
    if any(a.name == "half_life" and a.enabled for a in AXES):
        raw = call_pltnum(seqs, dev_user_prefix=dev_user_prefix)
        if math.isinf(references_pltnum_threshold):
            scores["half_life"] = [0.5] * K  # neutral if no anchor
        else:
            scores["half_life"] = half_life_anchor_rewards(
                raw, references_pltnum_threshold, beta=0.05
            )
    return scores

# COMMAND ----------

def generate_candidates(num: int, length_min: int, length_max: int,
                        hotspots: str = "") -> pd.DataFrame:
    """Returns a DataFrame with at minimum 'designed_sequence' and
    'designed_pdb' columns. Falls back to graceful exit if AME response
    shape isn't recognized.

    Branches on `use_inprocess_ame`:
    - Fast path: SDK call to gwb_*_proteina_complexa_ame_endpoint via call_ame().
    - Accurate path: in-process FK steering via run_ame_with_rewards(), with
      developability rewards biasing the diffusion process.
    """
    if use_inprocess_ame:
        from utils import run_ame_with_rewards
        df = run_ame_with_rewards(
            model=ame_model,
            target_pdb=motif_pdb_str,
            target_chain=target_chain,
            length_min=length_min,
            length_max=length_max,
            num_samples=num,
            reward=inprocess_reward,
            fk_n_branch=fk_n_branch,
            fk_beam_width=fk_beam_width,
            fk_temperature=fk_temperature,
            fk_step_checkpoints=fk_step_checkpoints,
            hotspots=hotspots,
        )
    else:
        df = call_ame(motif_pdb_str, target_chain, length_min, length_max, num,
                      hotspots=hotspots, dev_user_prefix=dev_user_prefix)
    print(f"AME returned columns: {list(df.columns)}")
    if "designed_sequence" not in df.columns or "designed_pdb" not in df.columns:
        raise RuntimeError(
            f"AME response missing expected columns; got {list(df.columns)}. "
            "Update the orchestrator to match the new response shape."
        )
    return df

# COMMAND ----------

trajectory_rows: List[Dict[str, Any]] = []
all_candidates: List[Dict[str, Any]] = []

with mlflow.start_run(run_name=mlflow_run_name or "enzyme_optimization",
                      experiment_id=experiment.experiment_id) as run:
    mlflow.log_params({
        "scaffold_length_min": scaffold_length_min,
        "scaffold_length_max": scaffold_length_max,
        "num_samples": num_samples,
        "num_iterations": num_iterations,
        "substrate_smiles": substrate_smiles,
        "half_life_margin": half_life_margin,
        "resampling_temperature": resampling_temperature,
        "strategy": strategy_name,
        "run_proteinmpnn": run_proteinmpnn_flag,
        "weights": weights_json,
        "n_references": len(references),
        "anchor_threshold": (None if math.isinf(anchor_threshold) else anchor_threshold),
        "use_inprocess_ame": use_inprocess_ame,
        "generation_mode": "Accurate" if use_inprocess_ame else "Fast",
        **(
            {"fk_n_branch": fk_n_branch, "fk_beam_width": fk_beam_width,
             "fk_temperature": fk_temperature,
             "fk_step_checkpoints": ",".join(str(s) for s in fk_step_checkpoints)}
            if use_inprocess_ame else {}
        ),
    })

    parents: List[Dict[str, Any]] = []
    gen_args = {
        "length_min": scaffold_length_min,
        "length_max": scaffold_length_max,
        "num_samples": num_samples,
        "hotspot_residues": "",
    }

    # Stop-criteria state — accumulated across iterations.
    iter_max_history: List[float] = []
    cumulative_above_threshold: int = 0
    stop_reason: Optional[str] = None

    for it in range(num_iterations):
        print(f"\n=== Iteration {it+1}/{num_iterations} ===")

        if it > 0:
            proposed = strategy.propose(
                parents, [p["composite_reward"] for p in parents],
                length_min=scaffold_length_min,
                length_max=scaffold_length_max,
                num_samples_next=num_samples,
            )
            if proposed is None:
                print("Strategy.propose() returned None — skipping further generation (NoOpStrategy).")
                break
            gen_args.update(proposed)

        ame_df = generate_candidates(
            gen_args["num_samples"], gen_args["length_min"], gen_args["length_max"],
            hotspots=gen_args.get("hotspot_residues", ""),
        )
        seqs = list(ame_df["designed_sequence"])
        pdbs = list(ame_df["designed_pdb"])

        # Optional ProteinMPNN redesign: replace each AME sequence with the
        # MPNN-redesigned one, then re-fold. The motif residues stay fixed via
        # ProteinMPNN's --fixed_positions_jsonl path so the catalytic identities
        # (e.g. His-Asp-Ser) survive the redesign.
        #
        # IMPORTANT: `target_chain` refers to the chain in the user's *input*
        # motif PDB (typically "B"). AME always emits its generated scaffolds
        # on chain "A". ProteinMPNN is being asked to redesign the AME output,
        # so fixed_positions must use the AME output's chain ("A"), not
        # target_chain. Using target_chain here was a Phase 1.4 bug: the
        # upstream tied_featurize KeyError'd on `fixed_position_dict["my_pdb"]["A"]`
        # because only "B" was in the dict, the orchestrator's broad except
        # clause swallowed the error, and motif preservation silently no-op'd.
        AME_OUTPUT_CHAIN = "A"
        mpnn_fixed_positions = (
            {AME_OUTPUT_CHAIN: motif_residues} if motif_residues else None
        )
        if run_proteinmpnn_flag:
            print(f"Running ProteinMPNN redesign on {len(seqs)} scaffolds (fixed_positions={mpnn_fixed_positions})...")
            new_seqs = []
            for pdb in pdbs:
                try:
                    redesigned = call_proteinmpnn(
                        pdb,
                        fixed_positions=mpnn_fixed_positions,
                        dev_user_prefix=dev_user_prefix,
                    )
                    new_seqs.append(redesigned[0] if redesigned else "")
                except Exception as e:
                    # The fallback to the AME sequence is a soft-degrade —
                    # but motif preservation is precisely what fixed_positions
                    # is for. Make the warning loud so a silent regression
                    # doesn't hide behind "the run completed".
                    print(f"  ⚠️  ProteinMPNN failed: {e}")
                    print(f"  ⚠️  Falling back to AME sequence — MOTIF PRESERVATION DID NOT RUN for this candidate.")
                    print(f"  ⚠️  fixed_positions={mpnn_fixed_positions}; AME PDB chain may differ from the key. Check earlier logs.")
                    new_seqs.append("")
            seqs = [ns or s for ns, s in zip(new_seqs, seqs)]

            # Re-fold with ESMFold for accurate post-redesign structures + pLDDT.
            print(f"ESMFolding {len(seqs)} redesigned sequences...")
            new_pdbs, new_plddts = [], []
            for s in seqs:
                try:
                    out = call_esmfold(s, dev_user_prefix=dev_user_prefix)
                    new_pdbs.append(out["pdb"])
                    new_plddts.append(out["mean_plddt"])
                except Exception as e:
                    print(f"  ESMFold failed: {e}")
                    new_pdbs.append("")
                    new_plddts.append(0.0)
            pdbs = new_pdbs

        # Score the batch on every enabled axis.
        per_axis = score_iteration(seqs, pdbs, anchor_threshold)
        rewards = compose_rewards(per_axis, AXES)

        for k in range(len(seqs)):
            cand_id = f"iter{it+1}_cand{k+1}"
            row = {
                "iteration": it + 1,
                "candidate_id": cand_id,
                "designed_sequence": seqs[k],
                "composite_reward": float(rewards[k]),
            }
            for axis_name, vals in per_axis.items():
                row[axis_name] = float(vals[k]) if not (isinstance(vals[k], float) and math.isnan(vals[k])) else None
            trajectory_rows.append(row)
            all_candidates.append({**row, "designed_pdb": pdbs[k]})

            for k_axis, val in row.items():
                if k_axis in ("iteration", "candidate_id", "designed_sequence"):
                    continue
                if isinstance(val, (int, float)) and val is not None and not math.isnan(val):
                    mlflow.log_metric(f"{cand_id}/{k_axis}", float(val), step=it + 1)

            # Persist the designed PDB as an MLflow artifact.
            with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as f:
                f.write(pdbs[k])
                pdb_local = f.name
            mlflow.log_artifact(pdb_local, artifact_path=f"pdbs/iter_{it+1}")

        parents = [
            {**c, "designed_pdb": pdbs[i]}
            for i, c in enumerate(trajectory_rows[-len(seqs):])
        ]
        # Iteration-level summary metrics.
        iter_max = float(max(rewards))
        iter_mean = float(np.mean(rewards))
        mlflow.log_metric("iter_max_reward", iter_max, step=it + 1)
        mlflow.log_metric("iter_mean_reward", iter_mean, step=it + 1)
        iter_max_history.append(iter_max)

        # ── Stopping criteria (any can short-circuit; first to fire wins) ──

        # 1. Reward threshold: any candidate >= target_reward
        if target_reward is not None and iter_max >= target_reward:
            stop_reason = (
                f"target_reward (iter_max={iter_max:.4f} >= target={target_reward:.4f})"
            )

        # 2. Best-K cap: cumulative count of candidates above threshold reaches target
        elif best_k_target is not None and best_k_threshold is not None:
            cumulative_above_threshold += sum(1 for r in rewards if r >= best_k_threshold)
            mlflow.log_metric("cumulative_above_threshold", cumulative_above_threshold, step=it + 1)
            if cumulative_above_threshold >= best_k_target:
                stop_reason = (
                    f"best_k (cumulative={cumulative_above_threshold} >= "
                    f"target={best_k_target} above threshold={best_k_threshold:.4f})"
                )

        # 3. Convergence: iter_max_reward improvement < threshold over the last
        #    convergence_window iterations. Negative threshold disables.
        if (stop_reason is None and convergence_threshold >= 0
                and len(iter_max_history) > convergence_window):
            window = iter_max_history[-(convergence_window + 1):]
            improvement = window[-1] - window[0]
            if improvement < convergence_threshold:
                stop_reason = (
                    f"convergence (improvement {improvement:.4f} over last "
                    f"{convergence_window} iters < threshold {convergence_threshold:.4f})"
                )

        if stop_reason is not None:
            print(f"\nEarly exit at iteration {it+1}: {stop_reason}")
            mlflow.set_tag("stop_reason", stop_reason)
            mlflow.log_metric("iterations_completed", it + 1)
            break
    else:
        # Loop ran to num_iterations without any stop trigger.
        mlflow.set_tag("stop_reason", "n_ceiling")
        mlflow.log_metric("iterations_completed", num_iterations)

    # Final artifacts: ranked CSV + top-K consolidated PDBs.
    traj_df = pd.DataFrame(trajectory_rows).sort_values("composite_reward", ascending=False)
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "reward_trajectory.csv")
        traj_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path="results")

        topk = sorted(all_candidates, key=lambda r: r["composite_reward"], reverse=True)[:max(num_samples, 8)]
        topk_dir = os.path.join(tmp, "topK_pdbs")
        os.makedirs(topk_dir, exist_ok=True)
        for r in topk:
            with open(os.path.join(topk_dir, r["candidate_id"] + ".pdb"), "w") as f:
                f.write(r["designed_pdb"])
        mlflow.log_artifacts(topk_dir, artifact_path="results/topK_pdbs")

    print(f"\nDone. MLflow run id: {run.info.run_id}")
    print(f"Top candidate: {traj_df.iloc[0]['candidate_id']}  "
          f"(composite_reward={traj_df.iloc[0]['composite_reward']:.4f})")
