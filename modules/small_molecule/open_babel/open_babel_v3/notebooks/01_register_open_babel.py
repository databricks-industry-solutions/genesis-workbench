# Databricks notebook source
# MAGIC %md
# MAGIC ### Open Babel Molecular Format Converter
# MAGIC
# MAGIC This notebook registers an Open Babel-based molecular format conversion model
# MAGIC as a PyFunc in Unity Catalog and deploys it to Model Serving.
# MAGIC
# MAGIC Open Babel supports 100+ molecular file formats (SMILES, SDF, PDB, PDBQT, MOL2,
# MAGIC InChI, etc.) and can add partial charges, generate 3D coordinates, and perform
# MAGIC format conversions critical to drug discovery pipelines (e.g., preparing ligands
# MAGIC for molecular docking with AutoDock Vina).
# MAGIC
# MAGIC **Key capabilities exposed via this endpoint:**
# MAGIC - Molecular format conversion (e.g., SMILES → SDF, SDF → PDBQT, PDB → PDBQT)
# MAGIC - 3D coordinate generation from 2D SMILES
# MAGIC - Partial charge addition for docking preparation
# MAGIC
# MAGIC Reference: [Open Babel](https://github.com/openbabel/openbabel)

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("model_name", "open_babel_converter", "Model Name")
dbutils.widgets.text("experiment_name", "dbx_genesis_workbench_modules", "Experiment Name")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")
dbutils.widgets.text("cache_dir", "open_babel", "Cache dir")
dbutils.widgets.text("workload_type", "CPU", "Workload Type for endpoints")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Installing dependencies

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.50.0 databricks-sql-connector==4.0.3
# MAGIC %pip install openbabel-wheel==3.1.1.23 rdkit==2025.3.6 mlflow==2.22.0

# COMMAND ----------

gwb_library_path = None
libraries = dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries")
for lib in libraries:
    if(lib.name.startswith("genesis_workbench")):
        gwb_library_path = lib.path.replace("dbfs:","")

print(gwb_library_path)

# COMMAND ----------

# MAGIC %pip install {gwb_library_path} --force-reinstall
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
experiment_name = dbutils.widgets.get("experiment_name")
user_email = dbutils.widgets.get("user_email")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")
cache_dir = dbutils.widgets.get("cache_dir")
workload_type = dbutils.widgets.get("workload_type")

print(f"Cache dir: {cache_dir}")
cache_full_path = f"/Volumes/{catalog}/{schema}/{cache_dir}"
print(f"Cache full path: {cache_full_path}")

# COMMAND ----------

spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{cache_dir}")

# COMMAND ----------

# Initialize Genesis Workbench
from genesis_workbench.workbench import initialize
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema, sql_warehouse_id=sql_warehouse_id, token=databricks_token)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify Open Babel installation and demonstrate format conversion
# MAGIC
# MAGIC Open Babel's `OBConversion` class handles conversion between molecular formats.
# MAGIC This is the same pattern used in the QSAR Drug Discovery pipeline for preparing
# MAGIC ligands (SDF → PDBQT) and receptors (PDB → PDBQT) for molecular docking.

# COMMAND ----------

from openbabel import openbabel
import pandas as pd
import json

# Verify Open Babel is working
ob_conv = openbabel.OBConversion()
print(f"Open Babel version: {openbabel.OBReleaseVersion()}")
print(f"Supported input formats: {len(ob_conv.GetSupportedInputFormat())}")
print(f"Supported output formats: {len(ob_conv.GetSupportedOutputFormat())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Demonstrate core conversion capabilities
# MAGIC
# MAGIC **Pattern from QSAR notebook:** The QSAR Drug Discovery pipeline uses Open Babel for:
# MAGIC 1. `babel_convert()` — SDF → PDBQT conversion with charge assignment for ligand docking
# MAGIC 2. `babel_convert_pdb()` — PDB → rigid PDBQT conversion for receptor preparation
# MAGIC
# MAGIC This model generalizes those patterns into a single flexible endpoint.

# COMMAND ----------

def demo_smiles_to_sdf(smiles: str) -> str:
    """Convert SMILES to SDF with 3D coordinates (gen3d)."""
    mol = openbabel.OBMol()
    conv = openbabel.OBConversion()
    conv.SetInAndOutFormats("smi", "sdf")

    conv.ReadString(mol, smiles)

    # Generate 3D coordinates
    builder = openbabel.OBBuilder()
    builder.Build(mol)
    ff = openbabel.OBForceField.FindForceField("mmff94")
    if ff.Setup(mol):
        ff.ConjugateGradients(500)
        ff.GetCoordinates(mol)

    return conv.WriteString(mol)


def demo_sdf_to_pdbqt(sdf_string: str) -> str:
    """Convert SDF to PDBQT — the pattern from QSAR notebook's babel_convert()."""
    mol = openbabel.OBMol()
    conv = openbabel.OBConversion()
    conv.SetInAndOutFormats("sdf", "pdbqt")
    conv.ReadString(mol, sdf_string)
    return conv.WriteString(mol)


# Test: Aspirin SMILES → SDF → PDBQT
aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
aspirin_sdf = demo_smiles_to_sdf(aspirin_smiles)
aspirin_pdbqt = demo_sdf_to_pdbqt(aspirin_sdf)

print("=== Aspirin SDF (first 20 lines) ===")
print("\n".join(aspirin_sdf.split("\n")[:20]))
print("\n=== Aspirin PDBQT (first 20 lines) ===")
print("\n".join(aspirin_pdbqt.split("\n")[:20]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wrap Open Babel in MLflow PyFunc for serving
# MAGIC
# MAGIC The model accepts a DataFrame with columns:
# MAGIC - `input_data` (str): molecular data string (SMILES, SDF, PDB, etc.)
# MAGIC - `input_format` (str): input format identifier (e.g., "smi", "sdf", "pdb")
# MAGIC - `output_format` (str): desired output format (e.g., "sdf", "pdbqt", "mol2", "inchi")
# MAGIC - `gen3d` (str, optional): "true" to generate 3D coordinates (useful for SMILES input)
# MAGIC
# MAGIC Returns a DataFrame with an `output_data` column containing the converted molecule.

# COMMAND ----------

import mlflow

class OpenBabelConverterModel(mlflow.pyfunc.PythonModel):
    """
    MLflow PyFunc wrapper for Open Babel molecular format conversion.

    Supports conversion between 100+ molecular formats including:
    SMILES, SDF/MOL, PDB, PDBQT, MOL2, InChI, CML, XYZ, etc.

    Optionally generates 3D coordinates from 2D input (e.g., SMILES)
    using MMFF94 force field optimization.
    """

    def load_context(self, context):
        from openbabel import openbabel
        self.openbabel = openbabel

    def _convert_molecule(self, input_data, input_format, output_format, gen3d=False):
        """Convert a single molecule between formats."""
        mol = self.openbabel.OBMol()
        conv = self.openbabel.OBConversion()
        conv.SetInAndOutFormats(input_format, output_format)

        success = conv.ReadString(mol, input_data)
        if not success:
            return f"ERROR: Failed to parse input as '{input_format}'"

        if gen3d and mol.NumAtoms() > 0:
            builder = self.openbabel.OBBuilder()
            builder.Build(mol)
            ff = self.openbabel.OBForceField.FindForceField("mmff94")
            if ff.Setup(mol):
                ff.ConjugateGradients(500)
                ff.GetCoordinates(mol)

        return conv.WriteString(mol)

    def predict(self, context, model_input, params=None):
        import pandas as pd

        if isinstance(model_input, pd.DataFrame):
            results = []
            for _, row in model_input.iterrows():
                input_data = str(row.get("input_data", ""))
                input_format = str(row.get("input_format", "smi"))
                output_format = str(row.get("output_format", "sdf"))
                gen3d = str(row.get("gen3d", "false")).lower() == "true"

                result = self._convert_molecule(input_data, input_format, output_format, gen3d)
                results.append(result)
            return results
        else:
            return ["ERROR: Expected DataFrame input"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the model locally

# COMMAND ----------

converter = OpenBabelConverterModel()
converter.load_context(None)

# COMMAND ----------

test_input = pd.DataFrame([
    {
        "input_data": "CC(=O)Oc1ccccc1C(=O)O",
        "input_format": "smi",
        "output_format": "sdf",
        "gen3d": "true"
    },
    {
        "input_data": "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
        "input_format": "smi",
        "output_format": "inchi",
        "gen3d": "false"
    },
    {
        "input_data": "c1ccccc1",
        "input_format": "smi",
        "output_format": "mol2",
        "gen3d": "true"
    },
])

predictions = converter.predict(None, test_input)
for i, pred in enumerate(predictions):
    print(f"=== Conversion {i+1} ===")
    print(pred[:200])
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register model in Unity Catalog

# COMMAND ----------

from genesis_workbench.models import (ModelCategory,
                                      import_model_from_uc,
                                      get_latest_model_version,
                                      deploy_model,
                                      set_mlflow_experiment)

from genesis_workbench.workbench import wait_for_job_run_completion

# COMMAND ----------

from mlflow.types.schema import ColSpec, Schema

input_schema = Schema([
    ColSpec(type="string", name="input_data"),
    ColSpec(type="string", name="input_format"),
    ColSpec(type="string", name="output_format"),
    ColSpec(type="string", name="gen3d"),
])

output_schema = Schema([
    ColSpec(type="string", name="output_data"),
])

signature = mlflow.models.signature.ModelSignature(
    inputs=input_schema,
    outputs=output_schema,
    params=None,
)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

experiment = set_mlflow_experiment(experiment_tag=experiment_name,
                                   user_email=user_email,
                                   host=None,
                                   token=None,
                                   shared=True)

test_example = pd.DataFrame([{
    "input_data": "CC(=O)Oc1ccccc1C(=O)O",
    "input_format": "smi",
    "output_format": "sdf",
    "gen3d": "true",
}])

with mlflow.start_run(run_name=f"{model_name}", experiment_id=experiment.experiment_id):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="open_babel_converter",
        python_model=converter,
        pip_requirements=[
            "openbabel-wheel==3.1.1.23",
            "numpy",
            "pandas",
        ],
        input_example=test_example,
        signature=signature,
        registered_model_name=f"{catalog}.{schema}.{model_name}",
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import model into Genesis Workbench and deploy serving endpoint

# COMMAND ----------

model_uc_name = f"{catalog}.{schema}.{model_name}"
model_version = get_latest_model_version(model_uc_name)

gwb_model_id = import_model_from_uc(user_email=user_email,
                    model_category=ModelCategory.SMALL_MOLECULE,
                    model_uc_name=model_uc_name,
                    model_uc_version=model_version,
                    model_name="Open Babel Converter",
                    model_display_name="Open Babel Molecular Format Converter",
                    model_source_version="v3.1.1",
                    model_description_url="https://github.com/openbabel/openbabel")

# COMMAND ----------

run_id = deploy_model(user_email=user_email,
                gwb_model_id=gwb_model_id,
                deployment_name=f"Open Babel Converter",
                deployment_description="Open Babel molecular format converter — supports SMILES, SDF, PDB, PDBQT, MOL2, InChI and 100+ formats with optional 3D coordinate generation",
                input_adapter_str="none",
                output_adapter_str="none",
                sample_input_data_dict_as_json="none",
                sample_params_as_json="none",
                workload_type=workload_type,
                workload_size="Small")

# COMMAND ----------

result = wait_for_job_run_completion(run_id, timeout=3600)
