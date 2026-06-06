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
dbutils.widgets.text("target_pdb_path", "", "UC volume path to target PDB for shortlist docking (optional)")

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
weights = {"qed": 1.0, "admet": 1.0, **json.loads(dbutils.widgets.get("weights_json") or "{}")}
temperature = float(dbutils.widgets.get("temperature"))
randomness = float(dbutils.widgets.get("randomness"))
target_pdb_path = dbutils.widgets.get("target_pdb_path").strip()

from genesis_workbench.workbench import initialize
from genesis_workbench.models import get_endpoint_name_for_uc_model
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema,
           sql_warehouse_id=sql_warehouse_id, token=databricks_token)

GENMOL_EP = get_endpoint_name_for_uc_model("genmol")
CLINTOX_EP = get_endpoint_name_for_uc_model("chemprop_clintox")
w = WorkspaceClient()

# COMMAND ----------

from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem.Scaffolds import MurckoScaffold


def genmol_generate(seeds, n):
    body = {
        "dataframe_split": {"columns": ["fragment"], "data": [[s] for s in seeds]},
        "params": {"num_molecules": int(n), "temperature": temperature,
                   "randomness": randomness, "scoring": "qed", "unique": True},
    }
    resp = w.api_client.do("POST", f"/serving-endpoints/{GENMOL_EP}/invocations", body=body)
    preds = resp.get("predictions", resp) if isinstance(resp, dict) else resp
    return [p.get("smiles") for p in (preds or []) if p.get("smiles")]


def clintox(smiles_list):
    """Clinical-toxicity probability per molecule (0..1; higher = more toxic)."""
    try:
        resp = w.serving_endpoints.query(name=CLINTOX_EP, inputs=smiles_list)
        raw = resp.predictions
        return [None if v is None else float(v) for v in (raw or [])]
    except Exception as e:
        print("clintox failed:", e)
        return [None] * len(smiles_list)


def score_candidates(smiles_list):
    tox = clintox(smiles_list)
    rows = []
    for smi, tx in zip(smiles_list, tox):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        q = float(QED.qed(mol))
        tox_term = (1.0 - tx) if tx is not None else 0.5
        reward = weights["qed"] * q + weights["admet"] * tox_term
        rows.append({"smiles": Chem.MolToSmiles(mol), "qed": q, "tox": tx, "reward": reward})
    return rows


def murcko(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None
    except Exception:
        return None

# COMMAND ----------

# Resume the pre-created run; run the loop, logging a trajectory + top-K.
with mlflow.start_run(run_id=mlflow_run_id):
    mlflow.set_tag("job_status", "running")
    mlflow.log_params({"num_samples": K, "num_iterations": N, "select_top": SELECT_TOP,
                       "seed_smiles": ",".join(seed_smiles)[:480]})

    seeds = list(seed_smiles)
    best_by_smiles: dict[str, dict] = {}

    for it in range(N):
        cands = genmol_generate(seeds, K)
        scored = score_candidates(cands)
        for r in scored:
            prev = best_by_smiles.get(r["smiles"])
            if prev is None or r["reward"] > prev["reward"]:
                best_by_smiles[r["smiles"]] = r
        if not scored:
            print(f"iter {it}: no valid candidates"); continue
        scored.sort(key=lambda x: x["reward"], reverse=True)
        parents = scored[:SELECT_TOP]
        # Reseed next iteration with the parents' Murcko scaffolds (fall back to the
        # parent molecule itself when it has no ring scaffold).
        seeds = [murcko(p["smiles"]) or p["smiles"] for p in parents]

        mlflow.log_metrics({
            "iter_best_reward": parents[0]["reward"],
            "iter_mean_reward": sum(s["reward"] for s in scored) / len(scored),
            "iter_best_qed": max(s["qed"] for s in scored),
            "iter_n_candidates": len(scored),
        }, step=it)
        print(f"iter {it}: best_reward={parents[0]['reward']:.3f} n={len(scored)}")

    # Global top-K
    top = sorted(best_by_smiles.values(), key=lambda x: x["reward"], reverse=True)[:DOCK_TOP_K]

    # Best-effort shortlist docking against the target structure (optional).
    if target_pdb_path:
        try:
            from databricks.sdk.service.serving import DataframeSplitInput
            diffdock_ep = get_endpoint_name_for_uc_model("diffdock")
            pdb = "".join(open(target_pdb_path).read()) if target_pdb_path.startswith("/Volumes") else ""
            for t in top:
                try:
                    resp = w.serving_endpoints.query(
                        name=diffdock_ep,
                        dataframe_split=DataframeSplitInput(
                            columns=["protein_pdb", "ligand_smiles"], data=[[pdb, t["smiles"]]]),
                    )
                    preds = resp.predictions
                    rec = preds[0] if isinstance(preds, list) and preds else preds
                    t["dock_confidence"] = float(rec.get("confidence")) if isinstance(rec, dict) and rec.get("confidence") is not None else None
                except Exception as e:
                    print("dock failed for one:", e); t["dock_confidence"] = None
        except Exception as e:
            print("docking step skipped:", e)

    mlflow.log_dict({"top_k": top}, "top_k.json")
    mlflow.log_metric("iterations_completed", N)
    mlflow.set_tag("job_status", "complete")
    print("DONE. top reward:", top[0]["reward"] if top else None)
