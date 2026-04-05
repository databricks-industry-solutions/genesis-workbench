# Databricks notebook source
# MAGIC %md
# MAGIC ### Proteina-Complexa Protein Binder Design
# MAGIC
# MAGIC This notebook registers NVIDIA's [Proteina-Complexa](https://github.com/NVIDIA-Digital-Bio/Proteina-Complexa)
# MAGIC protein binder design models into MLflow on Databricks and deploys them via Genesis Workbench.
# MAGIC
# MAGIC **Proteina-Complexa** is a generative flow-matching model that designs novel protein binders
# MAGIC for protein targets, small-molecule ligands, and scaffolds functional motifs — all in a fully
# MAGIC atomistic manner (backbone + side chains + sequence jointly).
# MAGIC
# MAGIC **Models registered:**
# MAGIC | Model | Use Case |
# MAGIC |-------|----------|
# MAGIC | `proteina-complexa` | Protein-protein binder design |
# MAGIC | `proteina-complexa-ligand` | Small-molecule binder design |
# MAGIC | `proteina-complexa-ame` | Motif scaffolding with ligand context |
# MAGIC
# MAGIC **Cluster requirements:** GPU instance (A10G/A100 recommended, >= 24 GB VRAM)

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("model_name", "proteina_complexa_v1", "Model Name")
dbutils.widgets.text("experiment_name", "dbx_genesis_workbench_modules", "Experiment Name")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")
dbutils.widgets.text("cache_dir", "proteina_complexa", "Cache dir")
dbutils.widgets.text("workload_type", "GPU_SMALL", "Workload Type for endpoints")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install dependencies

# COMMAND ----------

# MAGIC %pip install -q databricks-sdk>=0.50.0 databricks-sql-connector>=4.0.2 mlflow>=2.15
# MAGIC %pip install -q \
# MAGIC     --find-links https://data.pyg.org/whl/torch-2.7.0+cu126.html \
# MAGIC     torch==2.7.1 \
# MAGIC     lightning==2.6.1 \
# MAGIC     hydra-core==1.3.1 \
# MAGIC     omegaconf==2.3.0 \
# MAGIC     torch_geometric==2.7.0 \
# MAGIC     torch_scatter==2.1.2 \
# MAGIC     torch_sparse==0.6.18 \
# MAGIC     torch_cluster==1.6.3 \
# MAGIC     biotite==1.6.0 \
# MAGIC     loralib==0.1.2 \
# MAGIC     einops==0.8.2 \
# MAGIC     transformers==5.5.0 \
# MAGIC     jaxtyping
# MAGIC %pip install -q --no-deps "proteinfoundation @ git+https://github.com/NVIDIA-Digital-Bio/Proteina-Complexa.git"

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

import os, sys, subprocess, shutil, tempfile, importlib, json, re
import torch
import numpy as np
import pandas as pd
import mlflow

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

CKPT_DIR = "/tmp/proteina_checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

# COMMAND ----------

spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{cache_dir}")

# COMMAND ----------

from genesis_workbench.workbench import initialize
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema, sql_warehouse_id=sql_warehouse_id, token=databricks_token)

# COMMAND ----------

# Install transitive deps required by proteinfoundation (installed with --no-deps).
subprocess.check_call([
    "pip", "install", "-q",
    "--find-links", "https://data.pyg.org/whl/torch-2.7.0+cu126.html",
    "torch==2.7.1", "torch_geometric==2.7.0",
    "torch_scatter==2.1.2", "torch_sparse==0.6.18", "torch_cluster==1.6.3",
    "atomworks", "jax", "colabdesign", "jaxtyping", "loguru",
    "biopandas", "biopython", "cpdb-protein", "deepdiff", "dm-tree",
    "h5py", "mdtraj", "ml-collections", "modin", "multipledispatch",
    "plotly", "prody", "pydantic", "python-dotenv", "rich", "rich-click",
    "seaborn", "wandb", "wget", "xarray", "toolz", "openbabel-wheel",
])
subprocess.check_call(["pip", "install", "-q", "--no-build-isolation", "--no-deps", "graphein"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download Model Checkpoints from NGC

# COMMAND ----------

MODELS = {
    "complexa": {
        "ngc_id": "nvidia/clara/proteina_complexa",
        "main_ckpt": "complexa.ckpt",
        "ae_ckpt": "complexa_ae.ckpt",
        "mlflow_name": "proteina-complexa",
        "use_v2_arch": False,
        "description": "Protein-protein binder design",
    },
    "complexa_ligand": {
        "ngc_id": "nvidia/clara/proteina_complexa_ligand",
        "main_ckpt": "complexa_ligand.ckpt",
        "ae_ckpt": "complexa_ligand_ae.ckpt",
        "mlflow_name": "proteina-complexa-ligand",
        "use_v2_arch": True,
        "description": "Small-molecule ligand binder design",
    },
    "complexa_ame": {
        "ngc_id": "nvidia/clara/proteina_complexa_ame",
        "main_ckpt": "complexa_ame.ckpt",
        "ae_ckpt": "complexa_ame_ae.ckpt",
        "mlflow_name": "proteina-complexa-ame",
        "use_v2_arch": True,
        "description": "Motif scaffolding with ligand context (AME)",
    },
}

# COMMAND ----------

def download_checkpoints():
    """Download all Proteina-Complexa checkpoints directly from NGC API."""
    for name, info in MODELS.items():
        model_dir = os.path.join(CKPT_DIR, name)
        os.makedirs(model_dir, exist_ok=True)

        ngc_model = info["ngc_id"].split("/")[-1]
        ngc_base = f"https://api.ngc.nvidia.com/v2/models/org/nvidia/team/clara/{ngc_model}/1.0/files?redirect=true&path="

        for ckpt_file in [info["main_ckpt"], info["ae_ckpt"]]:
            dest = os.path.join(model_dir, ckpt_file)
            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                print(f"  {name}/{ckpt_file} already exists, skipping")
                continue

            url = f"{ngc_base}{ckpt_file}"
            print(f"Downloading {name}/{ckpt_file} ...")
            result = subprocess.run(
                ["wget", "--content-disposition", "-q", "--show-progress",
                 "-O", dest, url],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0 or not os.path.exists(dest) or os.path.getsize(dest) == 0:
                print(f"  WARNING: Failed to download {ckpt_file}: {result.stderr[:200]}")
            else:
                size_mb = os.path.getsize(dest) / (1024 * 1024)
                print(f"  Downloaded {ckpt_file} ({size_mb:.0f} MB)")

download_checkpoints()

# COMMAND ----------

for name, info in MODELS.items():
    main_path = os.path.join(CKPT_DIR, name, info["main_ckpt"])
    ae_path = os.path.join(CKPT_DIR, name, info["ae_ckpt"])
    main_ok = "OK" if os.path.exists(main_path) else "MISSING"
    ae_ok = "OK" if os.path.exists(ae_path) else "MISSING"
    print(f"{name}: main={main_ok}, autoencoder={ae_ok}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define MLflow PyFunc Wrappers

# COMMAND ----------

class _ProteinaComplexaBase(mlflow.pyfunc.PythonModel):
    """Shared infrastructure for all Proteina-Complexa model variants.

    Subclasses must set MAIN_CKPT, AE_CKPT, USE_V2_ARCH class attributes
    and implement their own predict() method.
    """

    _RESTYPES_3 = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
        "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
        "THR", "TRP", "TYR", "VAL",
    ]
    _SOLVENT_NAMES = {"HOH", "WAT", "H2O", "DOD", "SOL"}

    MAIN_CKPT = None
    AE_CKPT = None
    USE_V2_ARCH = False

    def load_context(self, context):
        import torch
        import lightning as L
        from omegaconf import OmegaConf

        torch.set_float32_matmul_precision("high")

        artifacts_dir = context.artifacts["checkpoints_dir"]
        main_ckpt = os.path.join(artifacts_dir, self.MAIN_CKPT)
        ae_ckpt = os.path.join(artifacts_dir, self.AE_CKPT)

        if self.USE_V2_ARCH:
            os.environ["USE_V2_COMPLEXA_ARCH"] = "True"

        _original_torch_load = torch.serialization.load
        torch.load = lambda *a, **kw: _original_torch_load(
            *a, **{**kw, "weights_only": False}
        )

        from proteinfoundation.proteina import Proteina

        self.model = Proteina.load_from_checkpoint(
            main_ckpt, strict=False, autoencoder_ckpt_path=ae_ckpt,
        )

        torch.load = _original_torch_load

        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Fix: set ligand attribute for v2 variants so that
        # single_pass_generation.search() skips prepend_target_to_samples()
        if self.USE_V2_ARCH:
            self.model.ligand = True
        print(f"Loaded {self.__class__.__name__} on {self.device}")

    @staticmethod
    def _write_temp_pdb(pdb_content: str) -> str:
        """Write PDB content to a temp file and return path."""
        tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w")
        tmp.write(pdb_content)
        tmp.close()
        return tmp.name

    @staticmethod
    def _pdb_chain_contig(pdb_path: str, chain_id: str) -> str:
        """Parse PDB and return a contig string for the given chain."""
        import biotite.structure.io.pdb as pdb_io
        import numpy as np

        pdb_file = pdb_io.PDBFile.read(pdb_path)
        struct = pdb_file.get_structure(model=1)
        chain_mask = (struct.chain_id == chain_id) & (struct.atom_name == "CA")
        res_ids = np.unique(struct.res_id[chain_mask])

        if len(res_ids) == 0:
            raise ValueError(f"No CA atoms found for chain {chain_id} in {pdb_path}")

        ranges = []
        start = int(res_ids[0])
        prev = start
        for rid in res_ids[1:]:
            rid = int(rid)
            if rid != prev + 1:
                ranges.append((start, prev))
                start = rid
            prev = rid
        ranges.append((start, prev))

        return ",".join(f"{chain_id}{s}-{e}" for s, e in ranges)

    @staticmethod
    def _extract_ligand_res_names(pdb_path: str) -> list:
        """Extract unique non-solvent HETATM residue names from a PDB file."""
        _SOLVENT = {"HOH", "WAT", "H2O", "DOD", "SOL"}
        names, seen = [], set()
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("HETATM"):
                    name = line[17:20].strip()
                    if name and name not in _SOLVENT and name not in seen:
                        names.append(name)
                        seen.add(name)
        return names

    @staticmethod
    def _parse_hotspots(hotspot_str, target_chain: str) -> list:
        """Parse 'idx1,idx2,...' into ['<chain><idx>', ...] format."""
        hotspots = []
        if hotspot_str and str(hotspot_str).strip():
            hotspots = [
                f"{target_chain}{x.strip()}"
                for x in str(hotspot_str).split(",")
            ]
        return hotspots

    @staticmethod
    def _build_motif_atom_spec(pdb_path: str, chain_id: str) -> str:
        """Auto-generate motif_atom_spec from all heavy atoms per residue."""
        from collections import OrderedDict
        residue_atoms = OrderedDict()
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM"):
                    chain = line[21]
                    if chain != chain_id:
                        continue
                    atom_name = line[12:16].strip()
                    element = line[76:78].strip()
                    if element == "H" or atom_name.startswith("H"):
                        continue
                    res_id = int(line[22:26].strip())
                    residue_atoms.setdefault(res_id, [])
                    if atom_name not in residue_atoms[res_id]:
                        residue_atoms[res_id].append(atom_name)
        if not residue_atoms:
            raise ValueError(
                f"No heavy ATOM records for chain {chain_id} in {pdb_path}"
            )
        parts = []
        for res_id, atoms in residue_atoms.items():
            parts.append(f"{chain_id}{res_id}: [{', '.join(atoms)}]")
        return "; ".join(parts)

    def _coords_to_pdb(self, coords, residue_types, mask, chain_index=None) -> str:
        """Convert atom37 coordinates to PDB format string (CA-only)."""
        lines = []
        atom_num = 1
        for i in range(len(residue_types)):
            if not mask[i]:
                continue
            idx = int(residue_types[i])
            res_name = self._RESTYPES_3[idx] if idx < len(self._RESTYPES_3) else "UNK"
            chain = chr(65 + int(chain_index[i])) if chain_index is not None else "A"
            x, y, z = coords[i, 1] * 10.0
            lines.append(
                f"ATOM  {atom_num:5d}  CA  {res_name:3s} {chain}{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  "
            )
            atom_num += 1
        lines.append("END")
        return "\n".join(lines)

    def _residue_types_to_sequence(self, residue_types, mask) -> str:
        """Convert residue type indices to amino acid sequence."""
        AA_MAP = "ACDEFGHIKLMNPQRSTVWY"
        seq = []
        for i, rt in enumerate(residue_types):
            if mask[i]:
                idx = int(rt)
                seq.append(AA_MAP[idx] if idx < len(AA_MAP) else "X")
        return "".join(seq)

    def _run_generation(self, conditional_features, binder_len_min,
                        binder_len_max, num_samples, start_sample_id=0,
                        transforms=None):
        """Build dataset, run inference, and return list of result dicts."""
        import torch
        import random
        import lightning as L
        from omegaconf import OmegaConf
        from proteinfoundation.datasets.gen_dataset import GenDataset, collate_fn

        nres_pool = list(range(binder_len_min, binder_len_max + 1))
        nres = sorted(random.choices(nres_pool, k=num_samples))

        ds_kwargs = dict(
            nres=nres, nrepeat_per_sample=1,
            conditional_features=conditional_features,
        )
        if transforms:
            ds_kwargs["transforms"] = transforms

        dataset = GenDataset(**ds_kwargs)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=num_samples, shuffle=False, collate_fn=collate_fn,
        )

        gen_cfg = self.model.cfg_exp.generation
        inf_cfg = OmegaConf.create({
            "args": OmegaConf.to_container(gen_cfg.args, resolve=True),
            "model": OmegaConf.to_container(gen_cfg.model.ode, resolve=True),
            "search": {"algorithm": "single-pass"},
        })
        inf_cfg.args.nsteps = 100
        self.model.configure_inference(inf_cfg, nn_ag=None)

        trainer = L.Trainer(
            accelerator="gpu", devices=1,
            inference_mode=False, enable_progress_bar=False, logger=False,
        )
        predictions = trainer.predict(self.model, dataloader)

        results = []
        for pred_batch in predictions:
            coords = pred_batch["coors"].cpu().numpy()
            res_types = pred_batch["residue_type"].cpu().numpy()
            masks = pred_batch["mask"].cpu().numpy()
            chain_idx = pred_batch.get("chain_index", None)
            if chain_idx is not None:
                chain_idx = chain_idx.cpu().numpy()
            rewards = pred_batch.get("rewards", None)

            for i in range(coords.shape[0]):
                ci = chain_idx[i] if chain_idx is not None else None
                pdb_str = self._coords_to_pdb(
                    coords[i], res_types[i], masks[i], ci,
                )
                seq = self._residue_types_to_sequence(res_types[i], masks[i])
                reward_val = float(rewards[i]) if rewards is not None else None
                results.append({
                    "sample_id": start_sample_id + len(results),
                    "pdb_output": pdb_str,
                    "sequence": seq,
                    "rewards": reward_val,
                })

        return results


class ProteinaComplexaBinderModel(_ProteinaComplexaBase):
    """Protein-protein binder design."""

    MAIN_CKPT = "complexa.ckpt"
    AE_CKPT = "complexa_ae.ckpt"
    USE_V2_ARCH = False

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        from proteinfoundation.datasets.gen_dataset import TargetFeatures

        results = []
        for _, row in model_input.iterrows():
            target_pdb_path = self._write_temp_pdb(row["target_pdb"])
            try:
                target_chain = str(row.get("target_chain", "A"))
                contig = self._pdb_chain_contig(target_pdb_path, target_chain)
                hotspots = self._parse_hotspots(
                    row.get("hotspot_residues", None), target_chain,
                )

                target_feat = TargetFeatures(
                    task_name="binder_design",
                    binder_gen_only=True,
                    pdb_path=target_pdb_path,
                    input_spec=contig,
                    target_hotspots=hotspots,
                )

                results.extend(self._run_generation(
                    conditional_features=[target_feat],
                    binder_len_min=int(row.get("binder_length_min", 50)),
                    binder_len_max=int(row.get("binder_length_max", 100)),
                    num_samples=int(row.get("num_samples", 1)),
                    start_sample_id=len(results),
                ))
            finally:
                os.unlink(target_pdb_path)

        return pd.DataFrame(results)


class ProteinaComplexaLigandModel(_ProteinaComplexaBase):
    """Small-molecule ligand binder design."""

    MAIN_CKPT = "complexa_ligand.ckpt"
    AE_CKPT = "complexa_ligand_ae.ckpt"
    USE_V2_ARCH = True

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        from proteinfoundation.datasets.gen_dataset import LigandFeatures
        from proteinfoundation.datasets.transforms import (
            CoordsTensorCenteringTransform,
        )

        results = []
        for _, row in model_input.iterrows():
            target_pdb_path = self._write_temp_pdb(row["target_pdb"])
            try:
                ligand_res_names = self._extract_ligand_res_names(target_pdb_path)
                if not ligand_res_names:
                    raise ValueError(
                        "No ligand HETATM records found in the PDB. "
                        "The ligand variant expects a ligand-only PDB."
                    )

                ligand_feat = LigandFeatures(
                    task_name="ligand",
                    pdb_path=target_pdb_path,
                    ligand=(
                        ligand_res_names
                        if len(ligand_res_names) > 1
                        else ligand_res_names[0]
                    ),
                    ligand_only=True,
                    use_bonds_from_file=True,
                )

                ligand_transform = CoordsTensorCenteringTransform(
                    tensor_name="x_target",
                    mask_name="target_mask",
                    data_mode="ligand_only",
                )

                results.extend(self._run_generation(
                    conditional_features=[ligand_feat],
                    binder_len_min=int(row.get("binder_length_min", 50)),
                    binder_len_max=int(row.get("binder_length_max", 100)),
                    num_samples=int(row.get("num_samples", 1)),
                    start_sample_id=len(results),
                    transforms=[ligand_transform],
                ))
            finally:
                os.unlink(target_pdb_path)

        return pd.DataFrame(results)


class ProteinaComplexaAMEModel(_ProteinaComplexaBase):
    """Active-site Motif scaffolding with ligand context (AME)."""

    MAIN_CKPT = "complexa_ame.ckpt"
    AE_CKPT = "complexa_ame_ae.ckpt"
    USE_V2_ARCH = True

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        from proteinfoundation.datasets.gen_dataset import (
            MotifFeatures, LigandFeatures,
        )
        from proteinfoundation.datasets.transforms import (
            CoordsTensorCenteringTransform,
        )

        results = []
        for _, row in model_input.iterrows():
            target_pdb_path = self._write_temp_pdb(row["target_pdb"])
            try:
                target_chain = str(row.get("target_chain", "B"))

                motif_spec = row.get("motif_atom_spec", None)
                if not motif_spec or not str(motif_spec).strip():
                    motif_spec = self._build_motif_atom_spec(
                        target_pdb_path, target_chain,
                    )

                motif_feat = MotifFeatures(
                    task_name="ame",
                    pdb_path=target_pdb_path,
                    motif_atom_spec=str(motif_spec),
                )

                conditional_features = [motif_feat]

                ligand_res_names = self._extract_ligand_res_names(target_pdb_path)
                if ligand_res_names:
                    ligand_feat = LigandFeatures(
                        task_name="ame",
                        pdb_path=target_pdb_path,
                        ligand=(
                            ligand_res_names
                            if len(ligand_res_names) > 1
                            else ligand_res_names[0]
                        ),
                        use_bonds_from_file=True,
                    )
                    conditional_features.append(ligand_feat)

                ame_transform = CoordsTensorCenteringTransform(
                    tensor_name="x_motif",
                    mask_name=None,
                    data_mode="all-atom",
                    additional_tensors=[
                        {"tensor_name": "x_target", "mask_name": "target_mask"},
                    ],
                )

                results.extend(self._run_generation(
                    conditional_features=conditional_features,
                    binder_len_min=int(row.get("binder_length_min", 50)),
                    binder_len_max=int(row.get("binder_length_max", 100)),
                    num_samples=int(row.get("num_samples", 1)),
                    start_sample_id=len(results),
                    transforms=[ame_transform],
                ))
            finally:
                os.unlink(target_pdb_path)

        return pd.DataFrame(results)


VARIANT_CLASSES = {
    "complexa": ProteinaComplexaBinderModel,
    "complexa_ligand": ProteinaComplexaLigandModel,
    "complexa_ame": ProteinaComplexaAMEModel,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define conda environment and signatures for model serving

# COMMAND ----------

conda_env = {
    "channels": ["defaults", "conda-forge", "pytorch", "nvidia", "pyg"],
    "dependencies": [
        "python=3.12",
        "pip",
        {
            "pip": [
                "torch==2.7.1",
                "torchvision==0.22.1",
                "lightning==2.6.1",
                "hydra-core==1.3.1",
                "omegaconf==2.3.0",
                "torch_geometric==2.7.0",
                "--find-links https://data.pyg.org/whl/torch-2.7.0+cu126.html",
                "torch_scatter==2.1.2",
                "torch_sparse==0.6.18",
                "torch_cluster==1.6.3",
                "biotite==1.6.0",
                "loralib==0.1.2",
                "einops==0.8.2",
                "transformers==5.5.0",
                "mlflow==3.10.1",
                "jaxtyping",
                "numpy>=2.0,<3",
                "scipy>=1.13,<2",
                "scikit-learn",
                "pandas>=2.3.0",
                "torchmetrics>=1.6,<2.0",
                "atomworks",
                "jax",
                "colabdesign",
                "loguru",
                "biopandas",
                "biopython",
                "cpdb-protein",
                "deepdiff",
                "dm-tree",
                "h5py",
                "mdtraj",
                "ml-collections",
                "modin",
                "multipledispatch",
                "plotly",
                "prody",
                "pydantic",
                "python-dotenv",
                "rich",
                "rich-click",
                "seaborn",
                "wandb",
                "wget",
                "xarray",
                "toolz",
                "joblib==1.4.2",
                "tqdm==4.66.4",
                "openbabel-wheel",
                "cloudpickle",
                "pyyaml",
            ]
        },
    ],
    "name": "proteina-complexa-env",
}

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

output_schema = Schema([
    ColSpec("integer", "sample_id"),
    ColSpec("string", "pdb_output"),
    ColSpec("string", "sequence"),
    ColSpec("double", "rewards"),
])

binder_input_schema = Schema([
    ColSpec("string", "target_pdb"),
    ColSpec("integer", "binder_length_min"),
    ColSpec("integer", "binder_length_max"),
    ColSpec("integer", "num_samples"),
    ColSpec("string", "hotspot_residues"),
    ColSpec("string", "target_chain"),
])

ligand_input_schema = Schema([
    ColSpec("string", "target_pdb"),
    ColSpec("integer", "binder_length_min"),
    ColSpec("integer", "binder_length_max"),
    ColSpec("integer", "num_samples"),
    ColSpec("string", "hotspot_residues"),
    ColSpec("string", "target_chain"),
])

ame_input_schema = Schema([
    ColSpec("string", "target_pdb"),
    ColSpec("integer", "binder_length_min"),
    ColSpec("integer", "binder_length_max"),
    ColSpec("integer", "num_samples"),
    ColSpec("string", "hotspot_residues"),
    ColSpec("string", "target_chain"),
])

VARIANT_SIGNATURES = {
    "complexa": ModelSignature(inputs=binder_input_schema, outputs=output_schema),
    "complexa_ligand": ModelSignature(inputs=ligand_input_schema, outputs=output_schema),
    "complexa_ame": ModelSignature(inputs=ame_input_schema, outputs=output_schema),
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Stage proteinfoundation code for code_paths

# COMMAND ----------

_pf_spec = importlib.util.find_spec("proteinfoundation")
_pf_src_dir = os.path.dirname(_pf_spec.origin)
PROTEINA_CODE_DIR = tempfile.mkdtemp(prefix="proteina_code_")
shutil.copytree(
    _pf_src_dir,
    os.path.join(PROTEINA_CODE_DIR, "proteinfoundation"),
    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
)

_of_spec = importlib.util.find_spec("openfold")
if _of_spec and _of_spec.origin:
    _of_src_dir = os.path.dirname(_of_spec.origin)
    shutil.copytree(
        _of_src_dir,
        os.path.join(PROTEINA_CODE_DIR, "openfold"),
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.so"),
    )
    print(f"  openfold: copied full package from {_of_src_dir}")
else:
    _of_tmp = os.path.join(PROTEINA_CODE_DIR, "_openfold_clone")
    subprocess.run(
        ["git", "clone", "--depth=1", "--filter=blob:none",
         "https://github.com/aqlaboratory/openfold.git", _of_tmp],
        check=True, capture_output=True,
    )
    _of_pkg = os.path.join(_of_tmp, "openfold")
    if os.path.exists(_of_pkg):
        shutil.copytree(
            _of_pkg,
            os.path.join(PROTEINA_CODE_DIR, "openfold"),
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.so"),
        )
        print("  openfold: cloned from GitHub")
    shutil.rmtree(_of_tmp, ignore_errors=True)

_gr_spec = importlib.util.find_spec("graphein")
if _gr_spec and _gr_spec.origin:
    _gr_src_dir = os.path.dirname(_gr_spec.origin)
    shutil.copytree(
        _gr_src_dir,
        os.path.join(PROTEINA_CODE_DIR, "graphein"),
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.so"),
    )
    print(f"  graphein: copied full package from {_gr_src_dir}")
else:
    print("  WARNING: graphein not installed - cannot bundle into code_paths")

# Patch concat_pair_feature_factory.py to match NGC checkpoint dims (272->271)
_cpff_path = os.path.join(PROTEINA_CODE_DIR, "proteinfoundation", "nn",
    "feature_factory", "concat_pair_feature_factory.py")
if os.path.exists(_cpff_path):
    with open(_cpff_path) as _f:
        _cpff = _f.read()
    _patched = re.sub(
        r'([ \t]*)(CrossSequenceChainIndexPairFeat\(\))',
        r'\1# \2  # patched: removed to match NGC checkpoint (271 vs 272)',
        _cpff,
    )
    if _patched != _cpff:
        with open(_cpff_path, "w") as _f:
            _f.write(_patched)
        print("Patched concat_pair_feature_factory.py: 272->271 (matches NGC checkpoint)")
    else:
        print("concat_pair_feature_factory.py: already patched or pattern not found")
else:
    print(f"WARNING: {_cpff_path} not found - patch skipped")

proteina_code_paths = [
    os.path.join(PROTEINA_CODE_DIR, "proteinfoundation"),
    os.path.join(PROTEINA_CODE_DIR, "openfold"),
    os.path.join(PROTEINA_CODE_DIR, "graphein"),
]
print(f"Staged proteinfoundation code at {PROTEINA_CODE_DIR}")
print(f"code_paths: {proteina_code_paths}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare example inputs and register all 3 models to Unity Catalog

# COMMAND ----------

import requests as _req

_examples_dir = os.path.join(CKPT_DIR, "example_inputs")
os.makedirs(_examples_dir, exist_ok=True)

_RAW_BASE = "https://raw.githubusercontent.com/NVIDIA-Digital-Bio/proteina-complexa/dev/assets/target_data"

def _fetch_example(subpath, local_name):
    local = os.path.join(_examples_dir, local_name)
    if not os.path.exists(local):
        url = f"{_RAW_BASE}/{subpath}"
        resp = _req.get(url)
        resp.raise_for_status()
        with open(local, "w") as f:
            f.write(resp.text)
        print(f"  Downloaded {local_name} ({len(resp.text)} bytes)")
    with open(local) as f:
        return f.read()

_pdb_binder = _fetch_example("bindcraft_targets/PD1.pdb", "PD1.pdb")
_pdb_ligand = _fetch_example("ligand_targets/7v11_ligand_centered.pdb", "7v11_ligand.pdb")
_pdb_ame = _fetch_example("ame_input_structures/M0024_1nzy_v3.pdb", "M0024_1nzy_v3.pdb")

_variant_input_examples = {
    "complexa": pd.DataFrame([{
        "target_pdb": _pdb_binder,
        "binder_length_min": np.int32(50),
        "binder_length_max": np.int32(80),
        "num_samples": np.int32(1),
        "hotspot_residues": "",
        "target_chain": "A",
    }]),
    "complexa_ligand": pd.DataFrame([{
        "target_pdb": _pdb_ligand,
        "binder_length_min": np.int32(50),
        "binder_length_max": np.int32(80),
        "num_samples": np.int32(1),
        "hotspot_residues": "",
        "target_chain": "A",
    }]),
    "complexa_ame": pd.DataFrame([{
        "target_pdb": _pdb_ame,
        "binder_length_min": np.int32(50),
        "binder_length_max": np.int32(80),
        "num_samples": np.int32(1),
        "hotspot_residues": "",
        "target_chain": "B",
    }]),
}
print("Example inputs ready for all 3 variants")

# COMMAND ----------

from genesis_workbench.workbench import set_mlflow_experiment
set_mlflow_experiment(experiment_name)

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# COMMAND ----------

registered_models = {}

for variant_key, model_info in MODELS.items():
    ckpt_dir = os.path.join(CKPT_DIR, variant_key)
    main_ckpt = os.path.join(ckpt_dir, model_info["main_ckpt"])

    if not os.path.exists(main_ckpt):
        print(f"Skipping {variant_key}: checkpoint not found at {main_ckpt}")
        continue

    mlflow_name = model_info["mlflow_name"]
    uc_model_name = f"{catalog}.{schema}.{mlflow_name.replace('-', '_')}"

    model_cls = VARIANT_CLASSES[variant_key]
    model_signature = VARIANT_SIGNATURES[variant_key]

    print(f"\nRegistering {mlflow_name} ({model_info['description']})...")
    print(f"  Class: {model_cls.__name__}")

    with mlflow.start_run(run_name=f"register-{mlflow_name}") as run:
        mlflow.log_params({
            "model_variant": variant_key,
            "model_class": model_cls.__name__,
            "ngc_id": model_info["ngc_id"],
            "use_v2_arch": model_info["use_v2_arch"],
            "description": model_info["description"],
        })

        model_info_logged = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=model_cls(),
            artifacts={"checkpoints_dir": ckpt_dir},
            code_paths=proteina_code_paths,
            conda_env=conda_env,
            signature=model_signature,
            input_example=_variant_input_examples[variant_key],
            registered_model_name=uc_model_name,
        )

        registered_models[mlflow_name] = {
            "run_id": run.info.run_id,
            "model_uri": model_info_logged.model_uri,
            "uc_name": uc_model_name,
        }
        print(f"  Registered as {uc_model_name}")
        print(f"  Run ID: {run.info.run_id}")

# COMMAND ----------

print("\nAll registered models:")
for name, info in registered_models.items():
    print(f"  {name}: {info['uc_name']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import models into Genesis Workbench and deploy serving endpoints

# COMMAND ----------

from genesis_workbench.models import (ModelCategory,
                                      import_model_from_uc,
                                      get_latest_model_version,
                                      deploy_model,
                                      )
from genesis_workbench.workbench import wait_for_job_run_completion

# COMMAND ----------

VARIANT_GWB_META = {
    "complexa": {
        "gwb_model_name": "Proteina-Complexa Binder",
        "gwb_display_name": "Proteina-Complexa Protein Binder Design",
        "gwb_deploy_name": "Proteina-Complexa Protein Binder Design",
        "gwb_deploy_desc": "Proteina-Complexa protein-protein binder design — generates novel protein binders for target protein chains using generative flow-matching",
    },
    "complexa_ligand": {
        "gwb_model_name": "Proteina-Complexa Ligand",
        "gwb_display_name": "Proteina-Complexa Small-Molecule Binder Design",
        "gwb_deploy_name": "Proteina-Complexa Small-Molecule Binder Design",
        "gwb_deploy_desc": "Proteina-Complexa ligand binder design — generates protein binders around small-molecule ligands using generative flow-matching",
    },
    "complexa_ame": {
        "gwb_model_name": "Proteina-Complexa AME",
        "gwb_display_name": "Proteina-Complexa Motif Scaffolding (AME)",
        "gwb_deploy_name": "Proteina-Complexa Motif Scaffolding (AME)",
        "gwb_deploy_desc": "Proteina-Complexa AME motif scaffolding — designs protein scaffolds that incorporate functional motif residues with bound ligand context",
    },
}

deploy_run_ids = []

for variant_key, model_info in MODELS.items():
    if variant_key not in registered_models:
        print(f"Skipping {variant_key}: not registered")
        continue

    reg = registered_models[model_info["mlflow_name"]]
    meta = VARIANT_GWB_META[variant_key]
    uc_model_name = reg["uc_name"]
    model_version = get_latest_model_version(uc_model_name)

    print(f"\nImporting {model_info['mlflow_name']} into Genesis Workbench...")

    gwb_model_id = import_model_from_uc(
        user_email=user_email,
        model_category=ModelCategory.SMALL_MOLECULES,
        model_uc_name=uc_model_name,
        model_uc_version=model_version,
        model_name=meta["gwb_model_name"],
        model_display_name=meta["gwb_display_name"],
        model_source_version="v1.0 (NGC)",
        model_description_url="https://github.com/NVIDIA-Digital-Bio/Proteina-Complexa",
    )

    run_id = deploy_model(
        user_email=user_email,
        gwb_model_id=gwb_model_id,
        deployment_name=meta["gwb_deploy_name"],
        deployment_description=meta["gwb_deploy_desc"],
        input_adapter_str="none",
        output_adapter_str="none",
        sample_input_data_dict_as_json="none",
        sample_params_as_json="none",
        workload_type=workload_type,
        workload_size="Small",
    )
    deploy_run_ids.append((model_info["mlflow_name"], run_id))
    print(f"  Deploy run ID: {run_id}")

# COMMAND ----------

for mlflow_name, run_id in deploy_run_ids:
    print(f"\nWaiting for deployment: {mlflow_name} (run_id={run_id})")
    result = wait_for_job_run_completion(run_id, timeout=3600)
    print(f"  Result: {result}")
