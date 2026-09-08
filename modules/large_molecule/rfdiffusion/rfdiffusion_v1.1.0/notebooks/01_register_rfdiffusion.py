# Databricks notebook source
# MAGIC %md
# MAGIC ## Register RFdiffusion (RFdiffusion3 / RFD3) on serverless GPU
# MAGIC Ports the motif-inpainting model off the torch-1.11 / SE3-Transformer stack (no
# MAGIC py3.12 wheels, classic-GPU only) to **RFD3 via `rc-foundry`**, which runs on the
# MAGIC serverless GPU AI runtime. The **serving contract is preserved exactly**:
# MAGIC `predict([{pdb, start_idx, end_idx}]) -> [backbone_pdb_str]`, registered as
# MAGIC `…​.rfdiffusion_inpainting`, so no app/executor/node/UI changes are needed.
# MAGIC RFD3 emits gzipped mmCIF → the pyfunc converts it back to a backbone PDB.
# MAGIC (The old unused `rfdiffusion_unconditional` model is dropped.)

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("model_name", "rfdiffusion", "Model Name")
dbutils.widgets.text("experiment_name", "dbx_genesis_workbench_modules", "Experiment Name")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")
dbutils.widgets.text("cache_dir", "rfdiffussion_cache_dir", "Cache dir")
dbutils.widgets.text("workload_type", "GPU_MEDIUM", "Workload Type for endpoints")

# COMMAND ----------

# rc-foundry (RFD3) + biopython. torch/CUDA + mlflow are preinstalled on the
# serverless GPU AI runtime (Python 3.12); do NOT reinstall them.
# MAGIC %pip install -r requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
MODEL_NAME = dbutils.widgets.get("model_name")
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")
USER_EMAIL = dbutils.widgets.get("user_email")
SQL_WAREHOUSE_ID = dbutils.widgets.get("sql_warehouse_id")
WORKLOAD_TYPE = dbutils.widgets.get("workload_type")

# COMMAND ----------

import os, tempfile, subprocess

# Route the (2.7GB) UC model upload through the presigned-URL/S3 path (boto3
# multipart, no 5-min cap). Same fix as esmfold/boltz.
os.environ["MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC"] = "false"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fetch the RFD3 checkpoint → fast LOCAL temp (packaged into the model)
# MAGIC `foundry install rfd3` downloads `rfd3_latest.ckpt` (2.7GB, ~1 min). We install
# MAGIC it to a fast local temp dir (serverless has no `/local_disk0`, and `/Volumes`
# MAGIC FUSE writes are pathologically slow) and package it into the model as the
# MAGIC `checkpoints` artifact. At serving, `FOUNDRY_CHECKPOINT_DIRS` points RFD3 at
# MAGIC that artifact dir (Model Serving does not read `/Volumes` at runtime).

# COMMAND ----------

RFD3_CKPT_DIR = tempfile.mkdtemp(prefix="rfd3_ckpt_")
subprocess.run(["foundry", "install", "rfd3", "-d", RFD3_CKPT_DIR], check=True)
print("RFD3 checkpoint dir:", RFD3_CKPT_DIR, "->", os.listdir(RFD3_CKPT_DIR))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the RFD3 inpainting pyfunc (preserves the rfdiffusion_inpainting contract)

# COMMAND ----------

import mlflow
from mlflow.types.schema import ColSpec, Schema
from typing import Any, Dict, List, Optional


class RFD3Inpainting(mlflow.pyfunc.PythonModel):
    """RFdiffusion3 motif inpainting behind the legacy rfdiffusion_inpainting contract.

    predict(model_input=[{"pdb": <str>, "start_idx": <int>, "end_idx": <int>}])
      -> [<backbone_pdb_str>]

    start_idx/end_idx are 1-indexed, inclusive; the [start_idx, end_idx] span is
    regenerated (masked) and the flanking residues are held fixed — matching the
    original RFdiffusion inpainting semantics. RFD3 output (gzipped mmCIF) is
    converted to a backbone-only PDB so downstream (ProteinMPNN) is unchanged.
    """

    # RFD3 diffusion steps. The original RFdiffusion used T=20 for serving; keep the
    # inference fast enough for a real-time endpoint.
    NUM_TIMESTEPS = 20

    def load_context(self, context):
        # Point RFD3 at the checkpoint packaged into the model (no /Volumes at serving).
        self.ckpt_dir = context.artifacts["checkpoints"]
        os.environ["FOUNDRY_CHECKPOINT_DIRS"] = self.ckpt_dir

    @staticmethod
    def _max_resid(pdb_str: str) -> int:
        idxs = [int(l[22:26]) for l in pdb_str.splitlines() if l.startswith("ATOM")]
        if not idxs:
            raise ValueError("input pdb has no ATOM records")
        return max(idxs)

    @staticmethod
    def _contigs(start_idx: int, end_idx: int, n_res: int):
        """Build RFD3 contig + select_fixed_atoms, omitting empty flanks.

        Mirrors the old RFdiffusion contig `A1-{s-1}/{len}-{len}/A{e+1}-{N}` in RFD3
        comma syntax: fixed left flank, {len} regenerated residues, fixed right flank.
        """
        x_len = end_idx - start_idx + 1
        left = f"A1-{start_idx - 1}" if start_idx > 1 else None
        right = f"A{end_idx + 1}-{n_res}" if end_idx < n_res else None
        contig = ",".join([p for p in (left, str(x_len), right) if p])
        fixed = ",".join([p for p in (left, right) if p])
        return contig, fixed

    @staticmethod
    def _cif_to_backbone_pdb(cif_path: str) -> str:
        """Convert an (all-atom) mmCIF to a backbone-only (N,CA,C,O) PDB string."""
        import gzip, shutil
        from Bio.PDB import MMCIFParser, PDBIO, Select

        if cif_path.endswith(".gz"):
            plain = cif_path[:-3]
            with gzip.open(cif_path, "rb") as fi, open(plain, "wb") as fo:
                shutil.copyfileobj(fi, fo)
            cif_path = plain

        structure = MMCIFParser(QUIET=True).get_structure("rfd3", cif_path)

        class _Backbone(Select):
            def accept_atom(self, atom):
                return atom.get_name() in ("N", "CA", "C", "O")

        io = PDBIO()
        io.set_structure(structure)
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            out_pdb = f.name
        io.save(out_pdb, select=_Backbone())
        with open(out_pdb, "r") as f:
            return f.read()

    def _run_inference(self, input_pdb: str, start_idx: int, end_idx: int) -> List[str]:
        import glob, json

        n_res = self._max_resid(input_pdb)
        contig, fixed = self._contigs(start_idx, end_idx, n_res)
        print("RFD3 contig:", contig, "| fixed:", fixed)

        with tempfile.TemporaryDirectory() as work:
            pdb_path = os.path.join(work, "input.pdb")
            with open(pdb_path, "w") as f:
                f.write(input_pdb)

            in_json = os.path.join(work, "inputs.json")
            with open(in_json, "w") as f:
                json.dump(
                    {"rfd3_inpaint": {"input": pdb_path, "contig": contig,
                                      "select_fixed_atoms": fixed}},
                    f,
                )

            out_dir = os.path.join(work, "out")
            proc = subprocess.run(
                ["rfd3", "design", f"out_dir={out_dir}", f"inputs={in_json}",
                 "n_batches=1", "diffusion_batch_size=1",
                 f"inference_sampler.num_timesteps={self.NUM_TIMESTEPS}"],
                capture_output=True, text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"rfd3 design failed (rc={proc.returncode})\n"
                    f"STDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
                )

            outputs = sorted(
                glob.glob(os.path.join(out_dir, "**", "*.cif.gz"), recursive=True)
                + glob.glob(os.path.join(out_dir, "**", "*.cif"), recursive=True)
            )
            if not outputs:
                raise RuntimeError(
                    f"no RFD3 mmCIF output found under {out_dir}. "
                    f"STDOUT:\n{proc.stdout[-2000:]}"
                )
            return [self._cif_to_backbone_pdb(p) for p in outputs]

    def predict(self, context, model_input: List[Dict[str, Any]], params=None) -> List[str]:
        # Dict values are mixed types (pdb: str, start_idx/end_idx: int as sent by the
        # executor), so the hint must be Dict[str, Any] — mlflow 2.22 enforces predict
        # type hints and rejects Dict[str, str] when int values are present.
        if len(model_input) > 1:
            raise ValueError("input must be a list with a single entry")
        d = model_input[0]
        return self._run_inference(d["pdb"], int(d["start_idx"]), int(d["end_idx"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Smoke-test the model
# MAGIC Grab an example PDB, reindex chain A to be 1-indexed / HETATM-free (the input
# MAGIC shape the endpoint receives from the executor), then inpaint residues 12-22.

# COMMAND ----------

import requests
from Bio import PDB
from Bio.PDB import PDBParser


def extract_chain_reindex(structure, chain_id="A"):
    chain = structure[0][chain_id]
    new_structure = PDB.Structure.Structure("new_structure")
    new_model = PDB.Model.Model(0)
    new_chain = PDB.Chain.Chain(chain_id)
    for i, residue in enumerate((r for r in chain if r.id[0] == " "), start=1):
        residue.id = (" ", i, " ")
        new_chain.add(residue)
    new_model.add(new_chain)
    new_structure.add(new_model)
    io = PDB.PDBIO()
    io.set_structure(new_structure)
    with tempfile.NamedTemporaryFile(suffix=".pdb") as f:
        io.save(f.name)
        with open(f.name, "r") as fh:
            return fh.read()


with tempfile.TemporaryDirectory() as td:
    resp = requests.get("https://files.rcsb.org/download/8dgr.pdb")
    pdb_path = os.path.join(td, "8dgr.pdb")
    with open(pdb_path, "wb") as f:
        f.write(resp.content)
    structure = PDBParser(QUIET=True).get_structure("8DGR", pdb_path)

example_pdb = extract_chain_reindex(structure)

# COMMAND ----------

model = RFD3Inpainting()
model.load_context(mlflow.pyfunc.PythonModelContext(
    artifacts={"checkpoints": RFD3_CKPT_DIR}, model_config={}))

input_example = [{"pdb": example_pdb, "start_idx": 12, "end_idx": 22}]
result = model.predict(None, input_example)
print("designs:", len(result), "| first design head:")
print("\n".join(result[0].splitlines()[:5]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log and register RFD3 as `rfdiffusion_inpainting` on Unity Catalog

# COMMAND ----------

from databricks.sdk import WorkspaceClient


def set_mlflow_experiment(experiment_tag, user_email):
    w = WorkspaceClient()
    base = "Shared/dbx_genesis_workbench_models"
    w.workspace.mkdirs(f"/Workspace/{base}")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    return mlflow.set_experiment(f"/{base}/{experiment_tag}")


signature = mlflow.models.infer_signature(input_example, result)
print(signature)

experiment = set_mlflow_experiment(EXPERIMENT_NAME, USER_EMAIL)

with mlflow.start_run(run_name="rfdiffusion_inpainting", experiment_id=experiment.experiment_id):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="rfdiffusion",
        python_model=RFD3Inpainting(),
        artifacts={"checkpoints": RFD3_CKPT_DIR},
        input_example=input_example,
        signature=signature,
        conda_env="rfd_env.yml",
        registered_model_name=f"{CATALOG}.{SCHEMA}.rfdiffusion_inpainting",
    )
    print("logged:", model_info.model_uri)
