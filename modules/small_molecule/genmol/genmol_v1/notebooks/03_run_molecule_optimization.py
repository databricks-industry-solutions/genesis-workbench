# Databricks notebook source
# MAGIC %md
# MAGIC ### Guided Molecule Optimization — generate → score → reseed loop
# MAGIC
# MAGIC The orchestrator job for the small-molecule "Guided Molecule Optimization"
# MAGIC tab (the twin of Guided Enzyme Optimization). Each iteration:
# MAGIC  1. GenMol grows the current seed scaffold(s) into K candidate molecules.
# MAGIC  2. Score each: QED (RDKit) + ADMET clinical-tox (Chemprop ClinTox endpoint);
# MAGIC     composite reward = w_qed·QED + w_admet·(1 − tox).
# MAGIC  3. Keep the top parents → their Murcko scaffolds seed the next iteration.
# MAGIC  4. Log per-iteration trajectory metrics to MLflow.
# MAGIC After N iterations, the global top-K are (best-effort) docked into the target
# MAGIC structure (DiffDock) and written as the `top_k.json` artifact.
# MAGIC
# MAGIC Dispatched by `start_molecule_optimization_job` in services/molecule_optimization.py.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")
dbutils.widgets.text("mlflow_experiment", "", "MLflow experiment path")
dbutils.widgets.text("mlflow_run_name", "", "MLflow run name")
dbutils.widgets.text("mlflow_run_id", "", "Pre-created MLflow run id (set by dispatcher)")

dbutils.widgets.text("seed_smiles_csv", "", "Seed fragment SMILES (CSV) — the binding motif(s)")
dbutils.widgets.text("num_samples", "24", "K — candidates generated per iteration")
dbutils.widgets.text("num_iterations", "5", "N — iteration count")
dbutils.widgets.text("select_top", "3", "Parents kept per iteration (reseed scaffolds)")
dbutils.widgets.text("dock_top_k", "5", "Top-K to dock at the end")
dbutils.widgets.text("weights_json", '{"qed":1.0,"admet":1.0}', "Per-axis weights JSON")
dbutils.widgets.text("temperature", "1.2", "GenMol softmax temperature")
dbutils.widgets.text("randomness", "2.0", "GenMol randomness")
dbutils.widgets.text("target_sequence", "", "Target protein sequence (folded via ESMFold; enables in-reward docking)")
dbutils.widgets.text("dock_per_iter", "8", "Candidates docked per iteration (top by cheap score)")
dbutils.widgets.text("dock_samples", "3", "DiffDock samples_per_complex per ligand")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# MAGIC %pip install rdkit==2025.3.6
# MAGIC %pip install mlflow[databricks]==2.22.0 databricks-sdk==0.50.0 databricks-sql-connector==4.0.3

# COMMAND ----------

gwb_library_path = None
for lib in dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries"):
    if lib.name.startswith("genesis_workbench"):
        gwb_library_path = lib.path.replace("dbfs:", "")
print("GWB library:", gwb_library_path)

# COMMAND ----------

# MAGIC %pip install {gwb_library_path} --force-reinstall
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import json
import mlflow
from databricks.sdk import WorkspaceClient

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")
user_email = dbutils.widgets.get("user_email")
mlflow_experiment = dbutils.widgets.get("mlflow_experiment")
mlflow_run_name = dbutils.widgets.get("mlflow_run_name")
mlflow_run_id = dbutils.widgets.get("mlflow_run_id")

seed_smiles = [s.strip() for s in dbutils.widgets.get("seed_smiles_csv").split(",") if s.strip()] or [""]
K = int(dbutils.widgets.get("num_samples"))
N = int(dbutils.widgets.get("num_iterations"))
SELECT_TOP = int(dbutils.widgets.get("select_top"))
DOCK_TOP_K = int(dbutils.widgets.get("dock_top_k"))
# Hard-constraint targets (transported via the weights_json param slot):
# keep only molecules with QED >= QED_MIN and ClinTox <= TOX_MAX. Guarded away
# from the divide-by-zero / degenerate ends.
_targets = json.loads(dbutils.widgets.get("weights_json") or "{}")
QED_MIN = min(max(float(_targets.get("qed_min", 0.5)), 0.0), 0.99)
TOX_MAX = min(max(float(_targets.get("tox_max", 0.3)), 0.01), 1.0)
print(f"Hard constraints: QED >= {QED_MIN}, ClinTox <= {TOX_MAX}")
temperature = float(dbutils.widgets.get("temperature"))
randomness = float(dbutils.widgets.get("randomness"))
target_sequence = dbutils.widgets.get("target_sequence").strip()
DOCK_PER_ITER = int(dbutils.widgets.get("dock_per_iter"))
DOCK_SAMPLES = int(dbutils.widgets.get("dock_samples"))

from genesis_workbench.workbench import initialize
from genesis_workbench.models import get_endpoint_name_for_uc_model, set_mlflow_experiment
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema,
           sql_warehouse_id=sql_warehouse_id, token=databricks_token)

GENMOL_EP = get_endpoint_name_for_uc_model("genmol")
CLINTOX_EP = get_endpoint_name_for_uc_model("chemprop_clintox")
w = WorkspaceClient()

# COMMAND ----------

import time

from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem.Scaffolds import MurckoScaffold


def wait_for_endpoint_ready(ep_name, timeout_s=1800, poll_s=20):
    """Block until a serving endpoint reports READY.

    GenMol/Chemprop can scale to zero; the first request then waits on a GPU
    container cold start that can exceed the SDK's 5-minute request timeout and
    crash the whole run (TimeoutError). Polling the endpoint state up front means
    the per-iteration queries hit a warm endpoint."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            ep = w.serving_endpoints.get(ep_name)
            ready = getattr(getattr(ep, "state", None), "ready", None)
            if str(ready).endswith("READY"):
                print(f"endpoint {ep_name} READY")
                return
            print(f"waiting for endpoint {ep_name} (ready={ready})…")
        except Exception as e:
            print(f"endpoint {ep_name} not queryable yet: {e}")
        time.sleep(poll_s)
    raise TimeoutError(f"endpoint {ep_name} not READY after {timeout_s}s")


def genmol_generate(seeds, n):
    body = {
        "dataframe_split": {"columns": ["fragment"], "data": [[s] for s in seeds]},
        "params": {"num_molecules": int(n), "temperature": temperature,
                   "randomness": randomness, "scoring": "qed", "unique": True},
    }
    # Retry once through a re-warm: a mid-loop scale-down can cold-start the next
    # call past the 5-minute SDK timeout.
    for attempt in range(2):
        try:
            resp = w.api_client.do("POST", f"/serving-endpoints/{GENMOL_EP}/invocations", body=body)
            preds = resp.get("predictions", resp) if isinstance(resp, dict) else resp
            return [p.get("smiles") for p in (preds or []) if p.get("smiles")]
        except Exception as e:
            print(f"genmol_generate attempt {attempt} failed: {e}")
            if attempt == 0:
                wait_for_endpoint_ready(GENMOL_EP)
            else:
                raise


def clintox(smiles_list):
    """Clinical-toxicity probability per molecule (0..1; higher = more toxic)."""
    try:
        resp = w.serving_endpoints.query(name=CLINTOX_EP, inputs=smiles_list)
        raw = resp.predictions
        return [None if v is None else float(v) for v in (raw or [])]
    except Exception as e:
        print("clintox failed:", e)
        return [None] * len(smiles_list)


import math


def cheap_score(smiles_list):
    """QED (RDKit) + ADMET clinical-tox (Chemprop). Flags hard-constraint
    feasibility (QED >= QED_MIN and ClinTox <= TOX_MAX) and a composite used to
    rank candidates (high QED + low tox). No GPU dock here."""
    tox = clintox(smiles_list)
    rows = []
    for smi, tx in zip(smiles_list, tox):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        q = float(QED.qed(mol))
        # Feasible only if BOTH targets are met AND tox is actually known.
        feasible = (q >= QED_MIN) and (tx is not None and tx <= TOX_MAX)
        # Composite (also the reseed/ranking signal): higher QED + lower tox.
        comp = q + (1.0 - (tx if tx is not None else 1.0))
        rows.append({"smiles": Chem.MolToSmiles(mol), "qed": q, "tox": tx,
                     "feasible": feasible, "cheap_reward": comp, "dock_confidence": None})
    return rows


# ── In-reward docking: fold the target (ESMFold) + ESM-embed it ONCE, then
# score each ligand against it. The target comes in as a SEQUENCE (resolved from
# a gene or pasted) — same "Target from gene / Paste sequence" pattern as the
# Structure Prediction tab — and we fold it here to the structure DiffDock needs.
DOCK_ENABLED = bool(target_sequence)
_target_pdb, _esm_emb = "", "{}"
if DOCK_ENABLED:
    from databricks.sdk.service.serving import DataframeSplitInput
    try:
        ESMFOLD_EP = get_endpoint_name_for_uc_model("esmfold")
        ESM_EP = get_endpoint_name_for_uc_model("diffdock_esm_embeddings")
        DIFFDOCK_EP = get_endpoint_name_for_uc_model("diffdock")

        # 1) Fold the target sequence → PDB (ESMFold).
        fr = w.serving_endpoints.query(name=ESMFOLD_EP, inputs=[target_sequence])
        fout = fr.predictions[0]
        _target_pdb = fout.get("pdb", "") if isinstance(fout, dict) else str(fout)
        if not _target_pdb:
            raise RuntimeError("ESMFold returned no PDB")
        print(f"Folded target ({len(target_sequence)} aa) → PDB ({len(_target_pdb)} chars)")

        # 2) ESM-embed the folded structure once (reused for every ligand dock).
        r = w.serving_endpoints.query(
            name=ESM_EP,
            dataframe_split=DataframeSplitInput(columns=["protein_pdb"], data=[[_target_pdb]]),
        )
        preds = r.predictions
        rec = preds[0] if isinstance(preds, list) and preds else preds
        _esm_emb = rec.get("embeddings_b64", "{}") if isinstance(rec, dict) else "{}"
        print("Target embedded — docking is in the reward.")
    except Exception as e:
        print("Fold/embed failed; disabling in-reward docking:", e)
        DOCK_ENABLED = False


def dock_confidence(smi):
    """Best DiffDock pose confidence for a ligand vs the (pre-embedded) target."""
    try:
        r = w.serving_endpoints.query(
            name=DIFFDOCK_EP,
            dataframe_split=DataframeSplitInput(
                columns=["protein_pdb", "ligand_smiles", "samples_per_complex", "esm_embeddings_b64"],
                data=[[_target_pdb, smi, DOCK_SAMPLES, _esm_emb]],
            ),
        )
        preds = r.predictions
        confs = [float(p["confidence"]) for p in (preds or [])
                 if isinstance(p, dict) and p.get("confidence") is not None]
        return max(confs) if confs else None
    except Exception as e:
        print("dock failed:", e)
        return None


def full_reward(row):
    """Composite (QED + low-tox) plus a docking bonus (sigmoid of DiffDock
    confidence ∈ 0..1) when the candidate was docked. Used to RANK feasible
    molecules — feasibility itself is the hard QED/tox gate, applied separately."""
    r = row["cheap_reward"]
    if row.get("dock_confidence") is not None:
        r += 1.0 / (1.0 + math.exp(-row["dock_confidence"]))
    return r


def murcko(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None
    except Exception:
        return None

# COMMAND ----------

# Resume the dispatcher's pre-created run, or create one (so the orchestrator can
# also be run standalone, e.g. for testing). Either way tag it so Search finds it.
if mlflow_run_id:
    _run_ctx = mlflow.start_run(run_id=mlflow_run_id)
else:
    _exp = set_mlflow_experiment(
        experiment_tag=(mlflow_experiment or "gwb_molecule_optimization"),
        user_email=user_email, host=None, token=None,
    )
    _run_ctx = mlflow.start_run(
        run_name=(mlflow_run_name or "mol_opt"), experiment_id=_exp.experiment_id
    )

with _run_ctx:
    mlflow.set_tag("origin", "genesis_workbench")
    mlflow.set_tag("feature", "molecule_optimization")
    mlflow.set_tag("created_by", user_email)
    mlflow.set_tag("job_status", "running")
    # Only self-log inputs when run standalone — the app dispatcher already logs
    # the full input config on the pre-created run, and re-logging a param with a
    # different value/format would raise.
    if not mlflow_run_id:
        mlflow.log_params({"num_samples": K, "num_iterations": N, "select_top": SELECT_TOP,
                           "seed_smiles": ",".join(seed_smiles)[:480]})

    seeds = list(seed_smiles)
    best_by_smiles: dict[str, dict] = {}   # feasible only → the valid candidates / top-K
    all_by_smiles: dict[str, dict] = {}    # every molecule explored → the "other results" table
    trajectory_rows: list[dict] = []       # every candidate, every iteration → full MLflow capture

    mlflow.set_tag("docking_in_reward", str(DOCK_ENABLED).lower())

    try:
        # Warm the endpoints used every iteration so a GPU cold start doesn't blow
        # past the SDK's 5-minute request timeout and crash the run.
        wait_for_endpoint_ready(GENMOL_EP)
        wait_for_endpoint_ready(CLINTOX_EP)

        for it in range(N):
            cands = genmol_generate(seeds, K)
            scored = cheap_score(cands)
            if not scored:
                print(f"iter {it}: no valid candidates"); continue

            # Dock the most promising — feasible candidates first (they pass the
            # hard QED/tox gate) — and fold binding into the ranking reward.
            if DOCK_ENABLED:
                scored.sort(key=lambda x: (x["feasible"], x["cheap_reward"]), reverse=True)
                for row in scored[:DOCK_PER_ITER]:
                    row["dock_confidence"] = dock_confidence(row["smiles"])

            for row in scored:
                row["reward"] = full_reward(row)
                # Full trajectory — EVERY candidate in EVERY iteration (captured to
                # trajectory.json so MLflow records the complete search, not just
                # the deduped survivors/explored).
                trajectory_rows.append({
                    "iteration": it,
                    "smiles": row["smiles"],
                    "qed": row["qed"],
                    "tox": row["tox"],
                    "feasible": row["feasible"],
                    "reward": row["reward"],
                    "dock_confidence": row.get("dock_confidence"),
                })
                # Track every molecule explored (for the "other results" table)...
                prevall = all_by_smiles.get(row["smiles"])
                if prevall is None or row["reward"] > prevall["reward"]:
                    all_by_smiles[row["smiles"]] = row
                # ...and keep ONLY feasible ones for the valid-candidate top-K.
                if row["feasible"]:
                    prev = best_by_smiles.get(row["smiles"])
                    if prev is None or row["reward"] > prev["reward"]:
                        best_by_smiles[row["smiles"]] = row

            survivors = [s for s in scored if s["feasible"]]
            # Reseed from feasible survivors; if an iteration has none, reseed from
            # the least-violating candidates so the search keeps moving toward the
            # feasible region instead of dead-ending.
            pool = sorted(survivors or scored, key=lambda x: x["reward"], reverse=True)
            parents = pool[:SELECT_TOP]
            seeds = [murcko(p["smiles"]) or p["smiles"] for p in parents]

            docked = [s for s in scored if s.get("dock_confidence") is not None]
            metrics = {
                "iter_n_candidates": len(scored),
                "iter_n_survivors": len(survivors),
                "iter_best_reward": parents[0]["reward"],
                "iter_mean_reward": sum(s["reward"] for s in scored) / len(scored),
                "iter_best_qed": max(s["qed"] for s in scored),
            }
            if docked:
                metrics["iter_best_dock"] = max(s["dock_confidence"] for s in docked)
            mlflow.log_metrics(metrics, step=it)
            print(f"iter {it}: survivors={len(survivors)}/{len(scored)} "
                  f"best_reward={parents[0]['reward']:.3f} docked={len(docked)}")

        # Global top-K — only feasible molecules ever entered best_by_smiles, so
        # the shortlist is guaranteed to meet the hard QED/tox targets.
        mlflow.log_metric("n_feasible_total", len(best_by_smiles))
        if not best_by_smiles:
            print(f"No molecules met the targets (QED>={QED_MIN}, ClinTox<={TOX_MAX}). "
                  "Loosen the constraints or run more iterations.")
        top = sorted(best_by_smiles.values(), key=lambda x: x["reward"], reverse=True)[:DOCK_TOP_K]

        # Dock the final shortlist so every top-K row carries a binding confidence
        # (per-iter docking only covers a subset, leaving the View's Dock column
        # blank for shortlisted molecules that were never docked). When a target was
        # provided, this guarantees the column is populated.
        if DOCK_ENABLED:
            for row in top:
                if row.get("dock_confidence") is None:
                    row["dock_confidence"] = dock_confidence(row["smiles"])
                    row["reward"] = full_reward(row)
            top = sorted(top, key=lambda x: x["reward"], reverse=True)

        # "Other molecules explored" — best attempts NOT in the valid top-K
        # (includes the closest-to-feasible when nothing met the targets). Always
        # populated as long as the generator produced anything, so the View has a
        # data table to show even when there are zero valid candidates.
        valid_smiles = {r["smiles"] for r in top}
        explored = [r for r in sorted(all_by_smiles.values(), key=lambda x: x["reward"], reverse=True)
                    if r["smiles"] not in valid_smiles][:25]

        mlflow.log_dict({"top_k": top, "explored": explored}, "top_k.json")
        # Full per-candidate trajectory across all iterations (complete capture).
        mlflow.log_dict({"trajectory": trajectory_rows}, "trajectory.json")
        mlflow.log_metric("iterations_completed", N)
        # Always succeed — "no candidates found" is a valid, non-failing outcome.
        mlflow.set_tag("job_status", "complete")
        print(f"DONE. valid candidates: {len(top)} | explored: {len(all_by_smiles)} | "
              f"top reward: {top[0]['reward'] if top else None}")
    except Exception as e:
        # Mark the run failed so Search Past Runs shows 🟥 instead of a stuck
        # "running" when the orchestrator dies mid-loop.
        mlflow.set_tag("job_status", "failed")
        mlflow.set_tag("error", str(e)[:500])
        print("FAILED:", e)
        raise
