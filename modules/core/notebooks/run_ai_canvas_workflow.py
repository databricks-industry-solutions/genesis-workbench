# Databricks notebook source
# MAGIC %md
# MAGIC ### Vortex (ai_canvas) — generic workflow orchestrator
# MAGIC Interprets a user-composed canvas graph and runs it node-by-node.
# MAGIC
# MAGIC The graph is **pre-enriched by the app dispatcher**: every node carries an
# MAGIC `exec` block (`kind`, resolved `endpoint_name` / `job_id` / `io_kind`,
# MAGIC `invoke_style`, input/output port names). This notebook therefore needs no
# MAGIC copy of the node registry — it is a pure interpreter. Adding a new node
# MAGIC kind only means teaching the dispatcher to emit its `exec` block and adding
# MAGIC a branch in `run_node()` below.
# MAGIC
# MAGIC Everything is logged to a single pre-created MLflow run (one run per
# MAGIC execution); per-node progress is surfaced as `node:<id>:status` tags.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "", "User email")
dbutils.widgets.text("mlflow_experiment", "gwb_ai_canvas", "MLflow experiment tag")
dbutils.widgets.text("mlflow_run_name", "ai_canvas_run", "MLflow run name")
dbutils.widgets.text("mlflow_run_id", "", "Pre-created MLflow run id")
dbutils.widgets.text("graph_path", "", "UC Volume path to graph.json")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")
user_email = dbutils.widgets.get("user_email")
mlflow_experiment = dbutils.widgets.get("mlflow_experiment")
mlflow_run_name = dbutils.widgets.get("mlflow_run_name")
mlflow_run_id = dbutils.widgets.get("mlflow_run_id") or None
graph_path = dbutils.widgets.get("graph_path")

# COMMAND ----------

gwb_library_path = None
libraries = dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries")
for lib in libraries:
    if lib.name.startswith("genesis_workbench"):
        gwb_library_path = lib.path.replace("dbfs:", "")
print(gwb_library_path)
# Fail fast with a clear message if the wheel is missing (e.g. a redeploy was
# swapping it on the Volume when this run launched) — otherwise the next cell
# does `%pip install None` and dies later with a cryptic ModuleNotFoundError.
if not gwb_library_path:
    raise RuntimeError(
        f"genesis_workbench wheel not found in /Volumes/{catalog}/{schema}/libraries "
        "— a deploy may have been mid-flight. Re-run this workflow."
    )

# COMMAND ----------

# MAGIC %pip install {gwb_library_path} --force-reinstall
# MAGIC %pip install databricks-sdk==0.50.0 databricks-sql-connector==4.0.3 mlflow==2.22.0 biopython==1.84 rdkit==2025.3.6
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")
user_email = dbutils.widgets.get("user_email")
mlflow_experiment = dbutils.widgets.get("mlflow_experiment")
mlflow_run_name = dbutils.widgets.get("mlflow_run_name")
mlflow_run_id = dbutils.widgets.get("mlflow_run_id") or None
graph_path = dbutils.widgets.get("graph_path")

# Early failed-status guard: if setup fails before the run body starts, flip the
# pre-created MLflow run's job_status to "failed" so Past Runs doesn't sit at
# "submitted" forever. (mlflow is available after the %pip restartPython above.)
import mlflow
from mlflow.tracking import MlflowClient


def _mark_run_failed(msg):
    if not mlflow_run_id:
        return
    try:
        c = MlflowClient()
        c.set_tag(mlflow_run_id, "job_status", "failed")
        c.set_tag(mlflow_run_id, "failure_reason", str(msg)[:500])
    except Exception:
        pass


try:
    from genesis_workbench.workbench import initialize, execute_workflow, wait_for_job_run_completion
    databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
    initialize(core_catalog_name=catalog, core_schema_name=schema, sql_warehouse_id=sql_warehouse_id, token=databricks_token)
except Exception as _setup_exc:
    _mark_run_failed(_setup_exc)
    raise

# COMMAND ----------

import json
import tempfile
import os

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from mlflow.tracking import MlflowClient
from genesis_workbench.models import set_mlflow_experiment

# Long HTTP timeout: this client makes the synchronous serving-endpoint calls for
# chain/endpoint nodes, and heavy realtime models can exceed the SDK's 60s default.
try:
    w = WorkspaceClient(config=Config(http_timeout_seconds=600))
    # Load the enriched graph the dispatcher uploaded.
    with open(graph_path) as f:
        graph = json.load(f)
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    edges = graph.get("edges", [])
    print(f"Loaded graph: {len(nodes)} nodes, {len(edges)} edges")
except Exception as _setup_exc:
    _mark_run_failed(_setup_exc)
    raise

# COMMAND ----------

# ── topological order (Kahn) ─────────────────────────────────────────────────
try:
    succ = {nid: [] for nid in nodes}
    indeg = {nid: 0 for nid in nodes}
    incoming = {nid: [] for nid in nodes}  # (src_id, src_port, dst_port)
    for e in edges:
        s, t = e.get("source"), e.get("target")
        if s in nodes and t in nodes:
            succ[s].append(t)
            indeg[t] += 1
            incoming[t].append((s, e.get("sourceHandle"), e.get("targetHandle")))

    order, queue = [], [nid for nid, d in indeg.items() if d == 0]
    while queue:
        nid = queue.pop(0)
        order.append(nid)
        for s in succ[nid]:
            indeg[s] -= 1
            if indeg[s] == 0:
                queue.append(s)
    if len(order) != len(nodes):
        raise RuntimeError("Workflow graph has a cycle — cannot execute.")
    print("Execution order:", order)
except Exception as _setup_exc:
    _mark_run_failed(_setup_exc)
    raise

# COMMAND ----------

# ── node executors ───────────────────────────────────────────────────────────
results = {}        # node_id -> {output_port: value}
final_outputs = {}  # output_sink label -> value


def gather_inputs(node_id):
    """Resolve this node's input-port values: inline value typed on the node,
    overridden by an upstream edge if one is wired to that port (convertible
    fields). Precedence: edge > inline."""
    node = nodes[node_id]
    in_ports = node.get("exec", {}).get("inputs", [])
    # Start from inline values typed on the node.
    inline = node.get("inputs", {}) or {}
    collected = {p: inline[p] for p in in_ports if inline.get(p) not in (None, "")}
    # Wired edges override the inline value for their target port.
    for src_id, src_port, dst_port in incoming.get(node_id, []):
        src_out = results.get(src_id, {})
        value = src_out.get(src_port) if src_port else (next(iter(src_out.values()), None))
        key = dst_port or (in_ports[0] if in_ports else "input")
        # A wired input that resolves to None means an upstream node produced no
        # value for the port it feeds. Fail fast — running a step on null yields a
        # meaningless ("fake") result while the job misleadingly reports success.
        if value is None:
            src_label = nodes.get(src_id, {}).get("label", src_id)
            raise RuntimeError(
                f"input '{key}' of '{node.get('label', node_id)}' is empty: upstream "
                f"'{src_label}' ({src_id}.{src_port or '?'}) produced no value. "
                f"Refusing to run on null."
            )
        collected[key] = value
    return collected


def run_node(node_id):
    node = nodes[node_id]
    ex = node.get("exec", {})
    kind = ex.get("kind")
    params = node.get("params", {}) or {}
    inputs = gather_inputs(node_id)
    out_ports = ex.get("outputs", []) or ["output"]
    first_out = out_ports[0]

    if kind == "io":
        io_kind = ex.get("io_kind")
        if io_kind == "text_input":
            return {first_out: params.get("value", "")}
        if io_kind == "volume_input":
            return {first_out: params.get("path", "")}
        if io_kind == "delta_input":
            return {first_out: params.get("table", "")}
        if io_kind == "output_sink":
            final_outputs[node.get("label", node_id)] = inputs
            return {}
        raise RuntimeError(f"Unknown io_kind '{io_kind}'")

    if kind == "endpoint":
        endpoint_name = ex.get("endpoint_name")
        if not endpoint_name:
            raise RuntimeError(f"Node {node_id}: endpoint not resolved (not deployed?).")
        if ex.get("invoke_style") == "inputs":
            value = next(iter(inputs.values()), "")
            resp = w.serving_endpoints.query(name=endpoint_name, inputs=[value])
        else:
            record = {**inputs, **params}
            resp = w.serving_endpoints.query(name=endpoint_name, dataframe_records=[record])
        preds = resp.predictions
        value = preds[0] if isinstance(preds, list) and len(preds) == 1 else preds
        return {first_out: value}

    if kind == "batch":
        job_name = ex.get("job_name")
        # Jobs with an output-collecting adapter (e.g. enzyme optimization) run via
        # the shared executor: it pre-creates a per-node child MLflow run, dispatches
        # the job into it, waits, and reads the job's output back — so a downstream
        # node consumes the real result, not just a run id.
        from genesis_workbench.executor import run_workflow_job, RUNNABLE_JOBS
        if job_name in RUNNABLE_JOBS:
            ctx = {
                "catalog": catalog, "schema": schema, "sql_warehouse": sql_warehouse_id,
                "user_email": user_email, "experiment_id": experiment.experiment_id,
                "experiment_name": mlflow_experiment, "parent_run_id": active_run_id,
                "node_id": node_id, "run_name": nodes[node_id].get("label", node_id),
                "dev_user_prefix": os.environ.get("DEV_USER_PREFIX", "") or "",
            }
            return {first_out: run_workflow_job(job_name, inputs, params, w, ctx=ctx)}
        # Fallback: plain trigger-and-wait; downstream nodes only get the run id.
        job_id = ex.get("job_id")
        if not job_id:
            raise RuntimeError(f"Node {node_id}: batch job id not resolved.")
        job_params = {k: str(v) for k, v in {**params, **inputs}.items()}
        run_id = execute_workflow(int(job_id), job_params)
        wait_for_job_run_completion(int(run_id), timeout=21600, poll_interval=30)
        return {first_out: {"job_run_id": run_id}}

    if kind == "chain":
        # Endpoint-chain (Protein Design / ADMET Screen) via the shared executor.
        from genesis_workbench.executor import run_chain
        res = run_chain(ex.get("chain"), inputs, params, w)
        # Map the chain's result dict onto the node's declared output ports so a
        # downstream node consuming e.g. "designs" gets the designs list — not the
        # whole {initial, sequences, designs} result wrapped under one port.
        if isinstance(res, dict) and any(p in res for p in out_ports):
            return {p: res.get(p) for p in out_ports}
        return {first_out: res}

    if kind == "transform":
        # Deterministic reshape via the shared executor; returns {out_port: value}.
        from genesis_workbench.executor import run_transform
        return run_transform(ex.get("op"), inputs, params, w)

    raise RuntimeError(f"Node {node_id}: unknown exec kind '{kind}'")

# COMMAND ----------

# ── run inside the pre-created MLflow run ────────────────────────────────────
experiment = set_mlflow_experiment(
    experiment_tag=mlflow_experiment, user_email=user_email, shared=False
)

if mlflow_run_id:
    run_ctx = mlflow.start_run(run_id=mlflow_run_id)
else:
    run_ctx = mlflow.start_run(run_name=mlflow_run_name, experiment_id=experiment.experiment_id)

active_run_id = None
try:
    with run_ctx as run:
        active_run_id = run.info.run_id
        mlflow.set_tag("origin", "genesis_workbench")
        mlflow.set_tag("feature", "ai_canvas")
        mlflow.set_tag("created_by", user_email)
        mlflow.set_tag("job_status", "started")
        mlflow.log_param("node_count", len(nodes))
        mlflow.log_param("edge_count", len(edges))
        for nid in order:
            mlflow.set_tag(f"node:{nid}:status", "pending")

        mlflow.set_tag("job_status", "running")
        for nid in order:
            label = nodes[nid].get("label", nid)
            print(f"▶ {nid} ({label})")
            mlflow.set_tag(f"node:{nid}:status", "running")
            try:
                results[nid] = run_node(nid)
                mlflow.set_tag(f"node:{nid}:status", "complete")
            except Exception as node_exc:  # noqa: BLE001
                mlflow.set_tag(f"node:{nid}:status", "failed")
                mlflow.set_tag(f"node:{nid}:error", str(node_exc)[:500])
                raise

        # Persist the full result set as an artifact.
        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, "workflow_results.json")
            with open(local, "w") as f:
                json.dump(
                    {"node_outputs": results, "final_outputs": final_outputs},
                    f, default=str, indent=2,
                )
            mlflow.log_artifact(local, artifact_path="results")

        mlflow.set_tag("job_status", "complete")
        print("✅ Workflow complete")
except Exception as exc:  # noqa: BLE001
    cid = active_run_id or mlflow_run_id
    if cid:
        client = MlflowClient()
        client.set_tag(cid, "job_status", "failed")
        client.set_tag(cid, "failure_reason", str(exc)[:500])
    raise
