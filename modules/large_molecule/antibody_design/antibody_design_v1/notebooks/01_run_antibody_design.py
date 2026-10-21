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
dbutils.widgets.text("sql_warehouse_id", "", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")

dbutils.widgets.text("rfd4_hf_repo", "nvidia/RFD4-Proteina", "HF repo for RFD4-Proteina weights")
dbutils.widgets.text("rfd4_hf_revision", "a06e2509c7033b010fc43614e9fd840303588502", "HF revision for weights")
dbutils.widgets.text("rfd4_flow_ckpt", "rfd4-proteina-flow-release-2026-10-19.ckpt", "Flow checkpoint filename")
dbutils.widgets.text("rfd4_ae_ckpt", "rfd4-proteina-ae-release-2026-10-19.ckpt", "Autoencoder checkpoint filename")
dbutils.widgets.text("rfd4_hf_token_scope", "dbx_genesis_workbench", "Secret scope for HF token")
dbutils.widgets.text("rfd4_hf_token_key", "hf_token", "Secret key for HF token")
dbutils.widgets.text("rfd4_git_url", "github.com/RosettaCommons/RFD4-Proteina.git", "RFD4 git host/path (public)")
dbutils.widgets.text("rfd4_git_ref", "v0.0.1", "RFD4 git ref (release tag)")

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
dbutils.widgets.text("framework_conditioning", "true", "Generate on a real VHH framework (design only CDRs) — native RFD4 antibody task")
dbutils.widgets.text("graft_framework", "false", "Downstream QC: CDR-graft onto a VHH framework if a design isn't antibody-like")
dbutils.widgets.text("framework_vhh_sequence", "", "Acceptor VHH framework sequence (blank = built-in humanized default)")
dbutils.widgets.text("resampling_temperature", "0.1", "Resampling softmax temperature")
dbutils.widgets.text("strategy", "resample", "Strategy: resample | noop")
dbutils.widgets.text("dev_user_prefix", "", "Dev user prefix (matches DEV_USER_PREFIX)")
dbutils.widgets.text("cofold_antigen", "true", "Co-fold antigen+VHH with Boltz to score binding (ipTM)")
dbutils.widgets.text("references_json", "[]", "Reference antibody JSON list (anchors the half-life axis)")
dbutils.widgets.text("half_life_margin", "0.05", "Half-life anchor margin")
dbutils.widgets.text(
    "weights_json",
    '{"plddt":1.3,"boltz":2.0,"self_consistency":1.5,"solubility":1.0,"half_life":1.0,"thermostab":1.0,"immuno":1.5}',
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
hf_token_scope, hf_token_key = g("rfd4_hf_token_scope"), g("rfd4_hf_token_key")

# Set HF_TOKEN environment variable for weight downloads (the HF repo may be gated).
try:
    hf_token = dbutils.secrets.get(hf_token_scope, hf_token_key)
    os.environ["HF_TOKEN"] = hf_token
except Exception as e:
    print(f"[HF_TOKEN] not set (will try unauthenticated download); {type(e).__name__}: {str(e)[:100]}")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
open("/tmp/bc.txt", "w").write("torch==2.14.1\n")
# Install rfproteina from the PUBLIC GitHub repo. --torch-backend cu132 pins CUDA-13 torch.
spec = f"rfd4-proteina[metrics-cuda13] @ git+https://{git_url}@{git_ref}"
_r = subprocess.run(["uv", "pip", "install", "--python", sys.executable, "--break-system-packages",
                     "--build-constraint", "/tmp/bc.txt", "--torch-backend", "cu132", spec],
                    capture_output=True, text=True)
if _r.returncode != 0:
    _err = (_r.stderr or "")
    raise RuntimeError(f"uv install failed rc={_r.returncode}; stderr tail:\n{_err[-3000:]}")
# anarcii (antibody numbering) — number_vhh uses it to recognise a VHH framework/CDRs and
# gate the ProteinMPNN framework redesign. It is NOT a rfproteina dependency, so install it
# explicitly. Separate call (no --torch-backend) so it reuses the torch we just installed.
# Best-effort: number_vhh degrades gracefully if it's absent, so a failure here only means
# designs won't be numbered (is_vhh=False) rather than killing the whole run.
_a = subprocess.run(["uv", "pip", "install", "--python", sys.executable, "--break-system-packages", "anarcii"],
                    capture_output=True, text=True)
if _a.returncode != 0:
    print(f"[anarcii] install failed rc={_a.returncode} (VHH numbering + ProteinMPNN framework "
          f"redesign will be skipped); stderr tail:\n{(_a.stderr or '')[-1500:]}")
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
rfd4_hf_repo, rfd4_hf_revision = g("rfd4_hf_repo"), g("rfd4_hf_revision")
rfd4_flow_ckpt, rfd4_ae_ckpt = g("rfd4_flow_ckpt"), g("rfd4_ae_ckpt")
rfd4_hf_token_scope, rfd4_hf_token_key = g("rfd4_hf_token_scope"), g("rfd4_hf_token_key")
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
framework_conditioning_flag = g("framework_conditioning").lower() in ("true", "1", "yes")
graft_framework_flag = g("graft_framework").lower() in ("true", "1", "yes")
_fw = g("framework_vhh_sequence").strip()
acceptor_framework = _fw if _fw else None   # None → utils' built-in humanized default
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

# Set up HF token for weight downloads (Hugging Face repo may be gated).
try:
    hf_token = dbutils.secrets.get(rfd4_hf_token_scope, rfd4_hf_token_key)
    os.environ["HF_TOKEN"] = hf_token
except Exception as e:
    print(f"[HF_TOKEN] not set (will try unauthenticated download); {type(e).__name__}: {str(e)[:100]}")

print(f"antigen_pdb_path:  {antigen_pdb_path}")
print(f"epitope_residues:  {epitope_residues}")
print(f"antigen_chain:     {antigen_chain}")
print(f"vhh_length:        [{vhh_length_min}, {vhh_length_max}]")
print(f"K (candidates):    {num_samples}   N (iterations): {num_iterations}")
print(f"weights:           {weights}")
print(f"cofold_antigen:    {cofold_antigen}   references: {len(references)}")
print(f"rfd4_hf_repo:      {rfd4_hf_repo}@{rfd4_hf_revision}")
print(f"rfd4_flow_ckpt:    {rfd4_flow_ckpt}")
print(f"rfd4_ae_ckpt:      {rfd4_ae_ckpt}")

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
    self_consistency_rmsd,
    make_strategy, number_vhh, graft_cdrs_onto_framework,
    build_framework_ctx, default_framework_ctx, CANONICAL_VHH_FRAMEWORK, load_rfd4, generate_vhh,
    call_esmfold, call_proteinmpnn, call_boltz, boltz_iptm,
    call_netsolp, call_pltnum, call_deepstabp, call_mhcflurry, call_hlaiipred,
    mhc_burden_per_residue,
    warmup_developability_endpoints, _extract_mean_plddt_from_pdb,
    liability_weighted_count, liability_detail,
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
    PredictorAxis("immuno_mhc2", weights.get("immuno_mhc2", 0.0), lower_is_better=True),
    PredictorAxis("liability",  weights.get("liability", 0.0), lower_is_better=True),
    # Self-consistency / designability (RFD4-Proteina core finding): Cα scRMSD between the RFD4
    # design and the ESMFold refold of its sequence — lower = sequence folds back to the design.
    PredictorAxis("self_consistency", weights.get("self_consistency", 0.0), lower_is_better=True),
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
rfd4_ctx = load_rfd4(rfd4_hf_repo, rfd4_hf_revision)
print("Warming developability endpoints...")
print(warmup_developability_endpoints(dev_user_prefix=dev_user_prefix))

# COMMAND ----------

# DBTITLE 1,Framework-conditioning context (real VHH framework, prepared once)
# Native RFD4 "Antibody/TCR framework conditioning": design CDR loops onto a REAL VHH framework, which
# the model docks to the antigen via internal-coordinate conditioning. Default = the h-NbBCII10
# universal humanized scaffold extracted straight from RFD4's own nanobody benchmark (real coords + FR
# partition known by construction; no ESMFold, no anarcii — this is what makes conditioning actually
# engage, vs. the earlier hand-built acceptor that failed to fold/number and fell back every run).
# A user-supplied `framework_vhh_sequence` instead folds+numbers that custom acceptor (best-effort).
framework_ctx = None
if framework_conditioning_flag:
    try:
        if acceptor_framework:
            _fw_fold = call_esmfold(acceptor_framework, dev_user_prefix=dev_user_prefix)
            framework_ctx = build_framework_ctx(acceptor_framework, _fw_fold["pdb"])
        else:
            framework_ctx = default_framework_ctx()
    except Exception as _e:  # noqa: BLE001
        print(f"[framework] could not prepare the framework ({type(_e).__name__}: {str(_e)[:160]}); "
              "falling back to de-novo binder generation")
        framework_ctx = None
    print(f"[framework] framework_conditioning={'ON' if framework_ctx else 'OFF (fallback de-novo)'}")

# COMMAND ----------

# DBTITLE 1,Per-batch scoring across all enabled axes
def score_iteration(seqs: List[str], pdbs: List[str], plddts: List[float],
                    anchor: float, rfd4_pdbs=None, sc_refold_pdbs=None) -> Dict[str, List[float]]:
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
                boltz_scores.append(boltz_iptm(out))  # iptm (lowercase, string) — see boltz_iptm
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
    # MHC-II (CD4 / anti-drug-antibody) presentation — the right immunogenicity signal for VHH.
    if any(a.name == "immuno_mhc2" and a.enabled for a in AXES):
        scores["immuno_mhc2"] = call_hlaiipred(seqs, dev_user_prefix=dev_user_prefix)
    # Sequence-liability scan — rule-based, no endpoint. Weighted motif count (lower = more inert).
    if any(a.name == "liability" and a.enabled for a in AXES):
        scores["liability"] = [liability_weighted_count(s) for s in seqs]
    if any(a.name == "half_life" and a.enabled for a in AXES):
        raw = call_pltnum(seqs, dev_user_prefix=dev_user_prefix)
        scores["half_life"] = ([0.5] * K if math.isinf(anchor)
                               else half_life_anchor_rewards(raw, anchor, beta=0.05))
    # Self-consistency / designability: Cα scRMSD(RFD4 design, ESMFold refold of the DE-NOVO
    # sequence); lower is better. When grafting, the de-novo refold (sc_refold_pdbs) is the
    # residue-aligned comparison to the RFD4 structure — the grafted structure has a different
    # framework, so it isn't aligned; fall back to pdbs for the de-novo (non-graft) path.
    if any(a.name == "self_consistency" and a.enabled for a in AXES) and rfd4_pdbs:
        refolds = sc_refold_pdbs if sc_refold_pdbs is not None else pdbs
        scores["self_consistency"] = [
            self_consistency_rmsd(rfd4_pdbs[k], refolds[k])
            if (k < len(rfd4_pdbs) and k < len(refolds) and rfd4_pdbs[k] and refolds[k]) else float("nan")
            for k in range(K)
        ]
    return scores

# COMMAND ----------

# DBTITLE 1,Generate + validate one batch of VHH candidates
def _esmfold_or_empty(seq: str):
    """ESMFold a sequence → (pdb, mean_plddt); ("", 0.0) on empty/failure."""
    if not seq:
        return "", 0.0
    try:
        out = call_esmfold(seq, dev_user_prefix=dev_user_prefix)
        return out["pdb"], out["mean_plddt"]
    except Exception as e:  # noqa: BLE001
        print(f"  ESMFold failed: {e}")
        return "", 0.0


def generate_and_validate(num: int) -> Dict[str, List]:
    """RFD4 generate (framework-conditioned when `framework_ctx` is set → real antibody folds;
    else de-novo binder) → hard VHH gate (anarcii) → optional CDR-graft / ProteinMPNN framework
    redesign QC → ESMFold. Returns dict of parallel lists over the SURVIVING candidates (the gate
    can shrink the batch)."""
    gen_df = generate_vhh(rfd4_ctx, antigen_pdb_str, epitope_residues, antigen_chain,
                          vhh_length_min, vhh_length_max, num, framework_ctx=framework_ctx)
    seqs = list(gen_df["designed_sequence"])
    rfd4_vhh_pdbs = list(gen_df["designed_pdb"])             # raw RFD4 VHH structure (persisted)
    complex_pdbs = (list(gen_df["complex_pdb"]) if "complex_pdb" in gen_df
                    else [""] * len(seqs))                   # RFD4 antigen+VHH complex pose (persisted)

    # ---- Hard VHH gate: keep only designs anarcii numbers as a V-domain --------------
    # A design with no cleanly-numberable CDRs can't be grafted and isn't antibody-like, so
    # drop it. If EVERY design is gated out, fall back to keeping them all with a loud warning
    # rather than failing the iteration.
    vhh_info = [number_vhh(s) if s else {"is_vhh": False, "cdr_positions": []} for s in seqs]
    keep = [bool(info["is_vhh"]) for info in vhh_info]
    if not any(keep):
        print(f"  ⚠️ VHH gate: 0/{len(seqs)} designs numbered as a VHH — keeping all (gate off this batch)")
        keep = [True] * len(seqs)
    elif sum(1 for k in keep if not k):
        print(f"  VHH gate: dropped {sum(1 for k in keep if not k)}/{len(seqs)} non-antibody designs")

    def _f(lst):
        return [x for x, k in zip(lst, keep) if k]
    seqs, rfd4_vhh_pdbs, complex_pdbs, vhh_info = _f(seqs), _f(rfd4_vhh_pdbs), _f(complex_pdbs), _f(vhh_info)
    denovo_seqs = list(seqs)                                 # pre-graft sequences (for self-consistency)
    is_vhh_flags = [True] * len(seqs)

    # ---- Make each design a REAL antibody --------------------------------------------
    if graft_framework_flag:
        grafted = [graft_cdrs_onto_framework(s, acceptor_framework) if acceptor_framework
                   else graft_cdrs_onto_framework(s) for s in seqs]
        gkeep = [gs is not None for gs in grafted]
        if not any(gkeep):
            print("  ⚠️ graft: no design could be grafted — keeping ungrafted RFD4 sequences")
        else:
            if sum(1 for k in gkeep if not k):
                print(f"  graft: dropped {sum(1 for k in gkeep if not k)} design(s) with no graftable CDRs")
            def _g(lst):
                return [x for x, k in zip(lst, gkeep) if k]
            seqs = [gs for gs in grafted if gs is not None]
            denovo_seqs, rfd4_vhh_pdbs, complex_pdbs = _g(denovo_seqs), _g(rfd4_vhh_pdbs), _g(complex_pdbs)
            is_vhh_flags = [True] * len(seqs)
            print(f"  grafted {len(seqs)} design(s) onto the humanized VHH framework")
    elif run_proteinmpnn_flag:
        # De-novo path: ProteinMPNN re-picks the FRAMEWORK while fixing the RFD4 CDRs.
        new_seqs = []
        for seq, info, pdb in zip(seqs, vhh_info, rfd4_vhh_pdbs):
            if info["is_vhh"] and info["cdr_positions"] and pdb:
                try:
                    redesigned = call_proteinmpnn(pdb, fixed_positions={"A": info["cdr_positions"]},
                                                  dev_user_prefix=dev_user_prefix)
                    new_seqs.append(redesigned[0] if redesigned else seq)
                except Exception as e:  # noqa: BLE001
                    print(f"  ⚠️ ProteinMPNN framework redesign failed: {e}; keeping RFD4 sequence")
                    new_seqs.append(seq)
            else:
                new_seqs.append(seq)
        seqs = new_seqs
        denovo_seqs = list(seqs)

    # ---- ESMFold the DELIVERABLE sequence (grafted / redesigned / de-novo) -----------
    folded_pdbs, plddts = [], []
    for s in seqs:
        pdb, pl = _esmfold_or_empty(s)
        folded_pdbs.append(pdb); plddts.append(pl)

    # ---- Self-consistency refold target ----------------------------------------------
    # scRMSD compares the RFD4 structure to the refold of the DESIGN sequence. When grafting,
    # that design is the DE-NOVO sequence (residue-aligned to the RFD4 structure), so fold it
    # separately (only if the self_consistency axis is on — else skip the extra ESMFold calls);
    # without grafting the de-novo sequence IS what we already folded.
    _sc_on = any(a.name == "self_consistency" and a.enabled for a in AXES)
    if graft_framework_flag:
        sc_refold_pdbs = [_esmfold_or_empty(s)[0] for s in denovo_seqs] if _sc_on else [""] * len(seqs)
    else:
        sc_refold_pdbs = folded_pdbs

    return {"seqs": seqs, "pdbs": folded_pdbs, "plddts": plddts, "is_vhh": is_vhh_flags,
            "rfd4_vhh_pdbs": rfd4_vhh_pdbs, "complex_pdbs": complex_pdbs,
            "sc_refold_pdbs": sc_refold_pdbs, "denovo_seqs": denovo_seqs}

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
            # Persist framework-conditioning status (stdout is unreachable on the job): whether it
            # engaged and which real VHH scaffold was used, so a glance at the run tells us if the
            # outputs came from framework conditioning vs. the de-novo fallback.
            "framework_conditioning_requested": framework_conditioning_flag,
            "framework_conditioning_active": framework_ctx is not None,
            "framework_scaffold": (framework_ctx.get("source") if framework_ctx else "none (de-novo)"),
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
            complex_pdbs = batch.get("complex_pdbs", [""] * len(seqs))      # RFD4 antigen+VHH complex
            rfd4_vhh_pdbs = batch.get("rfd4_vhh_pdbs", [""] * len(seqs))    # raw RFD4 VHH structure
            sc_refold_pdbs = batch.get("sc_refold_pdbs", pdbs)             # de-novo refold (self-consistency)
            denovo_seqs = batch.get("denovo_seqs", seqs)                   # pre-graft RFD4 sequence
            if not seqs:
                print(f"  iter {it+1}: no candidates survived the VHH gate/graft — skipping")
                continue

            mlflow.set_tag("job_status", f"iter_{it+1}_scoring")
            per_axis = score_iteration(seqs, pdbs, plddts, anchor_threshold, rfd4_vhh_pdbs, sc_refold_pdbs)
            rewards = compose_rewards(per_axis, AXES)

            for k in range(len(seqs)):
                cand_id = f"iter{it+1}_cand{k+1}"
                row = {
                    "iteration": it + 1,
                    "candidate_id": cand_id,
                    "designed_sequence": seqs[k],                                  # deliverable (grafted when enabled)
                    "denovo_sequence": denovo_seqs[k] if k < len(denovo_seqs) else seqs[k],  # pre-graft RFD4 design
                    "framework_conditioned": bool(framework_ctx is not None),      # generated on a real VHH framework
                    "grafted": bool(graft_framework_flag),
                    "is_vhh": bool(is_vhh[k]) if k < len(is_vhh) else False,
                    "composite_reward": float(rewards[k]),
                }
                for axis_name, vals in per_axis.items():
                    v = vals[k]
                    row[axis_name] = float(v) if not (isinstance(v, float) and math.isnan(v)) else None
                # Human-readable liability breakdown for the result dialog (string → not a metric).
                row["liability_detail"] = liability_detail(seqs[k])
                trajectory_rows.append(row)
                all_candidates.append({**row, "designed_pdb": pdbs[k],
                                       "complex_pdb": complex_pdbs[k] if k < len(complex_pdbs) else "",
                                       "rfd4_vhh_pdb": rfd4_vhh_pdbs[k] if k < len(rfd4_vhh_pdbs) else ""})

                for ak, av in row.items():
                    if ak in ("iteration", "candidate_id", "designed_sequence",
                              "denovo_sequence", "grafted", "framework_conditioned", "is_vhh"):
                        continue
                    if isinstance(av, (int, float)) and av is not None and not math.isnan(av):
                        mlflow.log_metric(f"{cand_id}/{ak}", float(av), step=it + 1)

                # Persist ALL structures for this candidate (clean names): the ESMFold monomer,
                # the raw RFD4 VHH, and the RFD4 antigen+VHH complex (binding pose/epitope).
                _cdir = tempfile.mkdtemp()
                for _tag, _content in (("esmfold", pdbs[k]),
                                       ("rfd4", rfd4_vhh_pdbs[k] if k < len(rfd4_vhh_pdbs) else ""),
                                       ("complex", complex_pdbs[k] if k < len(complex_pdbs) else "")):
                    if _content:
                        _p = os.path.join(_cdir, f"{cand_id}_{_tag}.pdb")
                        with open(_p, "w") as f:
                            f.write(_content)
                        mlflow.log_artifact(_p, artifact_path=f"pdbs/iter_{it+1}")

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

        # Final artifacts: ranked CSV + designed-sequence FASTA + inputs + top-K structures.
        traj_df = pd.DataFrame(trajectory_rows).sort_values("composite_reward", ascending=False)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "reward_trajectory.csv")
            traj_df.to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path, artifact_path="results")

            # Designed sequences (all candidates), reward-ranked, as FASTA.
            fasta_path = os.path.join(tmp, "designed_sequences.fasta")
            with open(fasta_path, "w") as f:
                for r in sorted(all_candidates, key=lambda r: r["composite_reward"], reverse=True):
                    if r.get("designed_sequence"):
                        f.write(f">{r['candidate_id']} reward={r['composite_reward']:.4f} "
                                f"is_vhh={r.get('is_vhh')}\n{r['designed_sequence']}\n")
            mlflow.log_artifact(fasta_path, artifact_path="results")

            # Per-residue MHC-II immunogenicity (ADA) burden for EVERY generated candidate, computed
            # here while HLAIIPred is warm and persisted as a run artifact. The viewer reads this to
            # project immunogenicity onto the structure with NO serving dependency (keyed by the base
            # candidate_id; the array is one value per VHH residue, in sequence order).
            mhc_burden_map: Dict[str, List[float]] = {}
            for r in all_candidates:
                seq = r.get("designed_sequence")
                if seq:
                    burden = mhc_burden_per_residue(seq, dev_user_prefix=dev_user_prefix)
                    if burden:
                        mhc_burden_map[r["candidate_id"]] = [round(x, 4) for x in burden]
            if mhc_burden_map:
                mlflow.log_dict(mhc_burden_map, "results/mhc_burden.json")

            # Inputs (reproducibility): antigen structure + epitope residues.
            in_dir = os.path.join(tmp, "inputs"); os.makedirs(in_dir, exist_ok=True)
            with open(os.path.join(in_dir, "antigen.pdb"), "w") as f:
                f.write(antigen_pdb_str)
            with open(os.path.join(in_dir, "epitope_residues.txt"), "w") as f:
                f.write(",".join(str(e) for e in epitope_residues))
            mlflow.log_artifacts(in_dir, artifact_path="inputs")

            # Top-K structures: ESMFold monomer, raw RFD4 VHH, and the RFD4 antigen+VHH complex.
            topk = sorted(all_candidates, key=lambda r: r["composite_reward"], reverse=True)[:max(num_samples, 8)]
            topk_dir = os.path.join(tmp, "topK_pdbs"); os.makedirs(topk_dir, exist_ok=True)
            for r in topk:
                cid = r["candidate_id"]
                for _tag, _key in (("esmfold", "designed_pdb"), ("rfd4", "rfd4_vhh_pdb"), ("complex", "complex_pdb")):
                    if r.get(_key):
                        with open(os.path.join(topk_dir, f"{cid}_{_tag}.pdb"), "w") as f:
                            f.write(r[_key])
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
