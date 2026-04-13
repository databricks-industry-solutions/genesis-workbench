# Databricks notebook source
# MAGIC %md
# MAGIC ### DiffDock Molecular Docking
# MAGIC
# MAGIC This notebook registers [DiffDock](https://github.com/gcorso/DiffDock) (commit `0f9c419`, ICLR 2023)
# MAGIC as a PyFunc model in Unity Catalog. The GWB import and serving deployment
# MAGIC is handled separately in `02_import_model_gwb.py` on serverless compute.
# MAGIC
# MAGIC DiffDock uses diffusion generative modeling to predict 3D binding poses for
# MAGIC protein–ligand complexes. It includes a score model (reverse diffusion) and a
# MAGIC confidence model to rank predicted poses.
# MAGIC
# MAGIC **Cluster requirements:**
# MAGIC - Runtime: **DBR 14.3 LTS ML GPU**
# MAGIC - Node type: GPU instance (e.g., `g5.2xlarge` / A10G)
# MAGIC - Single node is sufficient

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("model_name", "diffdock_v1", "Model Name")
dbutils.widgets.text("experiment_name", "dbx_genesis_workbench_modules", "Experiment Name")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")
dbutils.widgets.text("cache_dir", "diffdock", "Cache dir")
dbutils.widgets.text("workload_type", "GPU_SMALL", "Workload Type for endpoints")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Installing dependencies

# COMMAND ----------

# MAGIC %pip install -q databricks-sdk>=0.50.0 databricks-sql-connector>=4.0.2 mlflow>=2.15
# MAGIC %pip install -q pyyaml==6.0.1 scipy==1.7.3 networkx==2.6.3 biopython==1.79 rdkit-pypi==2022.03.5 e3nn==0.5.0 spyrmsd==0.5.2 pandas==1.5.3 biopandas==0.4.1 prody==2.6.1 fair-esm==2.0.0

# COMMAND ----------

import torch, subprocess, sys

torch_ver = torch.__version__.split("+")[0]
cuda_tag = torch.__version__.split("+")[1] if "+" in torch.__version__ else "cu118"
pyg_url = f"https://data.pyg.org/whl/torch-{torch_ver}+{cuda_tag}.html"
print(f"PyTorch: {torch.__version__}, PyG wheel index: {pyg_url}")

for pkg in ["torch-scatter", "torch-sparse", "torch-cluster"]:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", pkg,
        "-f", pyg_url, "--quiet",
    ])
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "torch-geometric==2.3.1", "--quiet",
])

# COMMAND ----------

import os, sys, subprocess, shutil, copy, json, tempfile
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

WORK_DIR = "/local_disk0/diffdock"
DIFFDOCK_DIR = os.path.join(WORK_DIR, "DiffDock")
os.makedirs(WORK_DIR, exist_ok=True)

# COMMAND ----------

spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{cache_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify installations

# COMMAND ----------

print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

from torch_geometric.data import Data
from torch_cluster import radius, radius_graph
from torch_scatter import scatter
import torch_geometric
print(f"PyG: {torch_geometric.__version__}")
print("All imports OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clone DiffDock and ESM

# COMMAND ----------

if not os.path.exists(DIFFDOCK_DIR):
    subprocess.run(
        ["git", "clone", "https://github.com/gcorso/DiffDock.git", DIFFDOCK_DIR],
        check=True,
    )
    subprocess.run(["git", "checkout", "0f9c419"], cwd=DIFFDOCK_DIR, check=True)
    print("Cloned DiffDock at commit 0f9c419")
else:
    print(f"DiffDock already at {DIFFDOCK_DIR}")

esm_shadow = os.path.join(DIFFDOCK_DIR, "esm")
if os.path.exists(esm_shadow):
    shutil.rmtree(esm_shadow)
    print("Removed DiffDock's esm/ directory (shadows fair-esm)")

dd_datasets_dir = os.path.join(DIFFDOCK_DIR, "dd_datasets")
datasets_dir = os.path.join(DIFFDOCK_DIR, "datasets")
if os.path.exists(datasets_dir):
    os.rename(datasets_dir, dd_datasets_dir)
    for root, dirs, files in os.walk(DIFFDOCK_DIR):
        dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__')]
        for fname in files:
            if not fname.endswith('.py'):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath) as f:
                content = f.read()
            new_content = content.replace('from datasets.', 'from dd_datasets.').replace('import datasets.', 'import dd_datasets.')
            if new_content != content:
                with open(fpath, 'w') as f:
                    f.write(new_content)
    print("Renamed datasets/ -> dd_datasets/ and updated imports")
else:
    print("dd_datasets/ already set up")

if DIFFDOCK_DIR not in sys.path:
    sys.path.insert(0, DIFFDOCK_DIR)

print("Setup complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test inference — download example PDB and run DiffDock
# MAGIC
# MAGIC This section validates the installation by running a full inference pass.
# MAGIC The downloaded model weights (score, confidence, ESM2) are then bundled
# MAGIC as MLflow artifacts during registration.

# COMMAND ----------

import requests

PDB_ID = "6agt"
SMILES = "COc(cc1)ccc1C#N"

PDB_DIR = os.path.join(WORK_DIR, "pdb_files")
os.makedirs(PDB_DIR, exist_ok=True)
pdb_path = os.path.join(PDB_DIR, f"{PDB_ID}.pdb")

if not os.path.exists(pdb_path):
    resp = requests.get(f"http://files.rcsb.org/view/{PDB_ID}.pdb")
    resp.raise_for_status()
    with open(pdb_path, "w") as f:
        f.write(resp.text)
    print(f"Downloaded {PDB_ID}.pdb")
else:
    print(f"Already have {PDB_ID}.pdb")

# COMMAND ----------

import esm

from Bio.PDB import PDBParser

THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

parser = PDBParser(QUIET=True)
structure = parser.get_structure("protein", pdb_path)

ESM_OUTPUT_DIR = os.path.join(DIFFDOCK_DIR, "data", "esm2_output")
os.makedirs(ESM_OUTPUT_DIR, exist_ok=True)

sequences = {}
for model in structure:
    for chain in model:
        seq = ""
        for residue in chain:
            rn = residue.get_resname().strip()
            if rn in THREE_TO_ONE:
                seq += THREE_TO_ONE[rn]
        if seq:
            sequences[chain.id] = seq

print(f"Chains: {list(sequences.keys())}, lengths: {[len(v) for v in sequences.values()]}")

esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
esm_model = esm_model.eval().cuda()
batch_converter = alphabet.get_batch_converter()

pdb_basename = os.path.basename(pdb_path)
for chain_id, seq in sequences.items():
    label = f"{pdb_basename}_{chain_id}"
    _, _, toks = batch_converter([(label, seq)])
    toks = toks.cuda()
    with torch.no_grad():
        results = esm_model(toks, repr_layers=[33], return_contacts=False)
    truncated = results["representations"][33][0, 1:len(seq)+1].clone().cpu()
    torch.save({'representations': {33: truncated}}, os.path.join(ESM_OUTPUT_DIR, f"{label}.pt"))
    print(f"  {label}: {truncated.shape}")

del esm_model
torch.cuda.empty_cache()
print("ESM embeddings computed.")

# COMMAND ----------

import yaml
from argparse import Namespace
from functools import partial
from torch_geometric.loader import DataLoader as PyGDataLoader
from rdkit.Chem import RemoveAllHs

from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.sampling import randomize_position, sampling
from dd_datasets.pdbbind import PDBBind
from utils.utils import get_model
from dd_datasets.process_mols import write_mol_with_coords

SCORE_MODEL_DIR = CONFIDENCE_MODEL_DIR = None
for d in [os.path.join(DIFFDOCK_DIR, "workdir", "paper_score_model"),
          os.path.join(DIFFDOCK_DIR, "workdir", "v1.1", "score_model")]:
    if os.path.exists(os.path.join(d, "model_parameters.yml")):
        SCORE_MODEL_DIR = d; break
for d in [os.path.join(DIFFDOCK_DIR, "workdir", "paper_confidence_model"),
          os.path.join(DIFFDOCK_DIR, "workdir", "v1.1", "confidence_model")]:
    if os.path.exists(os.path.join(d, "model_parameters.yml")):
        CONFIDENCE_MODEL_DIR = d; break

if SCORE_MODEL_DIR is None:
    from utils.download import download_and_extract
    download_and_extract(
        "https://github.com/gcorso/DiffDock/releases/download/v1.1/diffdock_models.zip",
        os.path.join(DIFFDOCK_DIR, "workdir"))
    SCORE_MODEL_DIR = os.path.join(DIFFDOCK_DIR, "workdir", "v1.1", "score_model")
    CONFIDENCE_MODEL_DIR = os.path.join(DIFFDOCK_DIR, "workdir", "v1.1", "confidence_model")

print(f"Score model: {SCORE_MODEL_DIR}")
print(f"Confidence model: {CONFIDENCE_MODEL_DIR}")

with open(os.path.join(SCORE_MODEL_DIR, "model_parameters.yml")) as f:
    score_model_args = Namespace(**yaml.full_load(f))
with open(os.path.join(CONFIDENCE_MODEL_DIR, "model_parameters.yml")) as f:
    confidence_args = Namespace(**yaml.full_load(f))

device = torch.device("cuda")
t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)

score_model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True)
ckpt = [f for f in os.listdir(SCORE_MODEL_DIR) if f.endswith(".pt")][0]
score_model.load_state_dict(torch.load(os.path.join(SCORE_MODEL_DIR, ckpt), map_location="cpu"), strict=True)
score_model = score_model.to(device).eval()
print(f"Score model loaded: {ckpt}")

confidence_model = get_model(confidence_args, device, t_to_sigma=t_to_sigma, no_parallel=True, confidence_mode=True)
conf_ckpt = [f for f in os.listdir(CONFIDENCE_MODEL_DIR) if f.endswith(".pt")][0]
confidence_model.load_state_dict(torch.load(os.path.join(CONFIDENCE_MODEL_DIR, conf_ckpt), map_location="cpu"), strict=True)
confidence_model = confidence_model.to(device).eval()
print(f"Confidence model loaded: {conf_ckpt}")

INPUT_CSV = os.path.join(WORK_DIR, "input_protein_ligand.csv")
with open(INPUT_CSV, "w") as f:
    f.write("protein_path,ligand\n")
    f.write(f"{pdb_path},{SMILES}\n")

CACHE_DIR = os.path.join(WORK_DIR, "cache")
if os.path.exists(CACHE_DIR):
    shutil.rmtree(CACHE_DIR)

N_SAMPLES = 10

print("Building score dataset...")
test_dataset = PDBBind(
    root="", transform=None, protein_path_list=[pdb_path], ligand_descriptions=[SMILES],
    receptor_radius=score_model_args.receptor_radius, cache_path=CACHE_DIR,
    remove_hs=score_model_args.remove_hs, max_lig_size=None,
    c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors,
    matching=False, keep_original=False,
    popsize=score_model_args.matching_popsize, maxiter=score_model_args.matching_maxiter,
    all_atoms=score_model_args.all_atoms, atom_radius=score_model_args.atom_radius,
    atom_max_neighbors=score_model_args.atom_max_neighbors,
    esm_embeddings_path=ESM_OUTPUT_DIR, require_ligand=True, num_workers=1)

use_orig_cache = getattr(confidence_args, 'use_original_model_cache', False)
transfer = getattr(confidence_args, 'transfer_weights', False)

if not use_orig_cache and not transfer:
    print("Building confidence dataset (all_atoms=True)...")
    confidence_test_dataset = PDBBind(
        root="", transform=None, protein_path_list=[pdb_path], ligand_descriptions=[SMILES],
        receptor_radius=confidence_args.receptor_radius, cache_path=CACHE_DIR,
        remove_hs=confidence_args.remove_hs, max_lig_size=None,
        c_alpha_max_neighbors=confidence_args.c_alpha_max_neighbors,
        matching=False, keep_original=False,
        popsize=confidence_args.matching_popsize, maxiter=confidence_args.matching_maxiter,
        all_atoms=confidence_args.all_atoms, atom_radius=confidence_args.atom_radius,
        atom_max_neighbors=confidence_args.atom_max_neighbors,
        esm_embeddings_path=ESM_OUTPUT_DIR if confidence_args.esm_embeddings_path is not None else None,
        require_ligand=True, num_workers=1)
    confidence_complex_dict = {d.name: d for d in confidence_test_dataset}
else:
    confidence_complex_dict = None

print(f"Score dataset: {len(test_dataset)} complexes")

test_loader = PyGDataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
tr_schedule = get_t_schedule(inference_steps=20)
OUT_DIR = os.path.join(WORK_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

for idx, orig in enumerate(test_loader):
    try:
        data_list = [copy.deepcopy(orig) for _ in range(N_SAMPLES)]
        randomize_position(data_list, score_model_args.no_torsion, False, score_model_args.tr_sigma_max)

        if confidence_complex_dict is not None:
            name = orig.name[0] if hasattr(orig, 'name') else f"complex_{idx}"
            conf_data = [copy.deepcopy(confidence_complex_dict[name]) for _ in range(N_SAMPLES)] if name in confidence_complex_dict else None
        else:
            conf_data = None

        data_list, confidence = sampling(
            data_list=data_list, model=score_model, inference_steps=19,
            tr_schedule=tr_schedule, rot_schedule=tr_schedule, tor_schedule=tr_schedule,
            device=device, t_to_sigma=t_to_sigma, model_args=score_model_args,
            confidence_model=confidence_model, confidence_data_list=conf_data,
            confidence_model_args=confidence_args, batch_size=10, no_final_step_noise=True)

        lig = orig.mol[0]
        positions = np.asarray([cg["ligand"].pos.cpu().numpy() + orig.original_center.cpu().numpy() for cg in data_list])

        if confidence is not None and isinstance(getattr(confidence_args, 'rmsd_classification_cutoff', 2), list):
            scores = confidence[:,0].cpu().numpy()
        elif confidence is not None:
            scores = confidence.cpu().numpy()
        else:
            scores = np.zeros(N_SAMPLES)

        order = np.argsort(scores)[::-1]
        scores, positions = scores[order], positions[order]

        cdir = os.path.join(OUT_DIR, f"complex_{idx}")
        os.makedirs(cdir, exist_ok=True)
        for r, (pos, sc) in enumerate(zip(positions, scores), 1):
            m = copy.deepcopy(lig)
            if score_model_args.remove_hs: m = RemoveAllHs(m)
            write_mol_with_coords(m, pos, os.path.join(cdir, f"rank{r}_confidence{sc:.2f}.sdf"))
        print(f"Complex {idx}: {N_SAMPLES} poses, top confidence={scores[0]:.3f}")
    except Exception as e:
        import traceback
        print(f"Complex {idx}: FAILED - {e}")
        traceback.print_exc()

print("Inference complete")

# COMMAND ----------

del score_model, confidence_model
torch.cuda.empty_cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define MLflow PyFunc wrapper
# MAGIC
# MAGIC The `DiffDockModel` class wraps the full DiffDock pipeline (ESM embeddings,
# MAGIC score model, confidence model, reverse diffusion) as an MLflow PyFunc.
# MAGIC
# MAGIC Uses lazy loading: `load_context()` stores paths only (completes in seconds),
# MAGIC while models are loaded on the first `predict()` call. This avoids the
# MAGIC 300-second model-loading timeout on Databricks Model Serving endpoints.

# COMMAND ----------

class DiffDockModel(mlflow.pyfunc.PythonModel):
    """MLflow PyFunc wrapper for DiffDock. Pure Python inference, no subprocess."""

    def _add_code_paths(self, context):
        """Ensure DiffDock code directories are on sys.path."""
        import sys, os

        repo_dir = context.artifacts.get("repo_dir", "")
        if repo_dir:
            if repo_dir in sys.path:
                sys.path.remove(repo_dir)
            sys.path.insert(0, repo_dir)

        for art_path in context.artifacts.values():
            model_root = os.path.dirname(os.path.dirname(art_path))
            code_dir = os.path.join(model_root, "code")
            if os.path.isdir(code_dir):
                if code_dir in sys.path:
                    sys.path.remove(code_dir)
                sys.path.insert(0, code_dir)
                break

    def load_context(self, context):
        """Lightweight context setup — stores paths, defers model loading."""
        import sys, os

        self._context = context
        self.repo_dir = context.artifacts["repo_dir"]
        self._add_code_paths(context)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._models_loaded = False
        print(f"DiffDock context stored (lazy loading). Device: {self.device}")

    def _ensure_models_loaded(self):
        """Load all neural network models on first call."""
        if self._models_loaded:
            return

        import os, yaml, torch
        import esm as esm_module
        from argparse import Namespace
        from functools import partial

        self._add_code_paths(self._context)

        # DiffDock's torus.py loads .npy files via relative paths (np.load('.score.npy')).
        # On serving endpoints the CWD is not the code directory, so we must chdir.
        os.chdir(self.repo_dir)

        from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
        from utils.utils import get_model

        context = self._context

        esm_dir = context.artifacts.get("esm_model_dir")
        if esm_dir:
            esm_pt = os.path.join(esm_dir, "esm2_t33_650M_UR50D.pt")
            print(f"Loading ESM2 from artifact: {esm_pt}")
            self.esm_model, self.esm_alphabet = esm_module.pretrained.load_model_and_alphabet_local(esm_pt)
        else:
            print("Downloading ESM2 (no artifact bundled)...")
            self.esm_model, self.esm_alphabet = esm_module.pretrained.esm2_t33_650M_UR50D()
        self.esm_model = self.esm_model.eval().to(self.device)
        self.esm_batch_converter = self.esm_alphabet.get_batch_converter()

        score_dir = context.artifacts.get("score_model_dir")
        conf_dir = context.artifacts.get("confidence_model_dir")

        with open(os.path.join(score_dir, "model_parameters.yml")) as f:
            self.score_model_args = Namespace(**yaml.full_load(f))
        with open(os.path.join(conf_dir, "model_parameters.yml")) as f:
            self.confidence_args = Namespace(**yaml.full_load(f))

        self.t_to_sigma = partial(t_to_sigma_compl, args=self.score_model_args)

        self.score_model = get_model(
            self.score_model_args, self.device,
            t_to_sigma=self.t_to_sigma, no_parallel=True,
        )
        ckpt = [f for f in os.listdir(score_dir) if f.endswith(".pt")][0]
        self.score_model.load_state_dict(
            torch.load(os.path.join(score_dir, ckpt), map_location="cpu"), strict=True,
        )
        self.score_model = self.score_model.to(self.device).eval()

        self.confidence_model = get_model(
            self.confidence_args, self.device,
            t_to_sigma=self.t_to_sigma, no_parallel=True, confidence_mode=True,
        )
        conf_ckpt = [f for f in os.listdir(conf_dir) if f.endswith(".pt")][0]
        self.confidence_model.load_state_dict(
            torch.load(os.path.join(conf_dir, conf_ckpt), map_location="cpu"), strict=True,
        )
        self.confidence_model = self.confidence_model.to(self.device).eval()

        self._models_loaded = True
        print(f"DiffDock models loaded on {self.device}")

    def _compute_esm_embeddings(self, protein_path, esm_output_dir):
        import os, torch
        from Bio.PDB import PDBParser

        three_to_one = {
            "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
            "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
            "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
            "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
        }
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", protein_path)
        os.makedirs(esm_output_dir, exist_ok=True)

        for model in structure:
            for chain in model:
                seq = "".join(three_to_one.get(r.get_resname().strip(), "") for r in chain)
                if not seq:
                    continue
                label = f"{os.path.basename(protein_path)}_{chain.id}"
                _, _, toks = self.esm_batch_converter([(label, seq)])
                toks = toks.to(self.device)
                with torch.no_grad():
                    results = self.esm_model(toks, repr_layers=[33], return_contacts=False)
                truncated = results["representations"][33][0, 1:len(seq)+1].clone().cpu()
                torch.save({'representations': {33: truncated}}, os.path.join(esm_output_dir, f"{label}.pt"))

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        import sys, copy, os, tempfile
        import numpy as np
        import torch

        self._add_code_paths(context)
        self._ensure_models_loaded()

        from torch_geometric.loader import DataLoader as PyGDataLoader
        from rdkit.Chem import RemoveAllHs
        from utils.diffusion_utils import get_t_schedule
        from utils.sampling import randomize_position, sampling
        from dd_datasets.pdbbind import PDBBind
        from dd_datasets.process_mols import write_mol_with_coords

        all_results = []
        use_orig = getattr(self.confidence_args, 'use_original_model_cache', False)
        transfer = getattr(self.confidence_args, 'transfer_weights', False)

        for _, row in model_input.iterrows():
            pdb_content = row["protein_pdb"]
            smiles = row["ligand_smiles"]
            n_samples = int(row.get("samples_per_complex", 10))

            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    pdb_path = os.path.join(tmpdir, "protein.pdb")
                    with open(pdb_path, "w") as f:
                        f.write(pdb_content)

                    esm_out = os.path.join(tmpdir, "esm2_output")
                    self._compute_esm_embeddings(pdb_path, esm_out)

                    dataset = PDBBind(
                        root="", transform=None,
                        protein_path_list=[pdb_path], ligand_descriptions=[smiles],
                        receptor_radius=self.score_model_args.receptor_radius,
                        cache_path=os.path.join(tmpdir, "cache"),
                        remove_hs=self.score_model_args.remove_hs, max_lig_size=None,
                        c_alpha_max_neighbors=self.score_model_args.c_alpha_max_neighbors,
                        matching=False, keep_original=False,
                        popsize=self.score_model_args.matching_popsize,
                        maxiter=self.score_model_args.matching_maxiter,
                        all_atoms=self.score_model_args.all_atoms,
                        atom_radius=self.score_model_args.atom_radius,
                        atom_max_neighbors=self.score_model_args.atom_max_neighbors,
                        esm_embeddings_path=esm_out, require_ligand=True, num_workers=1,
                    )

                    confidence_complex_dict = None
                    if not use_orig and not transfer:
                        conf_dataset = PDBBind(
                            root="", transform=None,
                            protein_path_list=[pdb_path], ligand_descriptions=[smiles],
                            receptor_radius=self.confidence_args.receptor_radius,
                            cache_path=os.path.join(tmpdir, "cache_conf"),
                            remove_hs=self.confidence_args.remove_hs, max_lig_size=None,
                            c_alpha_max_neighbors=self.confidence_args.c_alpha_max_neighbors,
                            matching=False, keep_original=False,
                            popsize=self.confidence_args.matching_popsize,
                            maxiter=self.confidence_args.matching_maxiter,
                            all_atoms=self.confidence_args.all_atoms,
                            atom_radius=self.confidence_args.atom_radius,
                            atom_max_neighbors=self.confidence_args.atom_max_neighbors,
                            esm_embeddings_path=esm_out if self.confidence_args.esm_embeddings_path is not None else None,
                            require_ligand=True, num_workers=1,
                        )
                        confidence_complex_dict = {d.name: d for d in conf_dataset}

                    loader = PyGDataLoader(dataset=dataset, batch_size=1, shuffle=False)
                    tr_schedule = get_t_schedule(inference_steps=10)

                    for orig_complex_graph in loader:
                        try:
                            data_list = [copy.deepcopy(orig_complex_graph) for _ in range(n_samples)]
                            randomize_position(
                                data_list, self.score_model_args.no_torsion, False,
                                self.score_model_args.tr_sigma_max,
                            )

                            if confidence_complex_dict is not None:
                                name = orig_complex_graph.name[0]
                                conf_data = [copy.deepcopy(confidence_complex_dict[name]) for _ in range(n_samples)] if name in confidence_complex_dict else None
                            else:
                                conf_data = None

                            data_list, confidence = sampling(
                                data_list=data_list, model=self.score_model, inference_steps=9,
                                tr_schedule=tr_schedule, rot_schedule=tr_schedule, tor_schedule=tr_schedule,
                                device=self.device, t_to_sigma=self.t_to_sigma, model_args=self.score_model_args,
                                confidence_model=self.confidence_model, confidence_data_list=conf_data,
                                confidence_model_args=self.confidence_args, batch_size=10, no_final_step_noise=True,
                            )

                            lig = orig_complex_graph.mol[0]
                            ligand_pos = np.asarray([
                                cg["ligand"].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy()
                                for cg in data_list
                            ])

                            if confidence is not None and isinstance(getattr(self.confidence_args, 'rmsd_classification_cutoff', 2), list):
                                conf_scores = confidence[:, 0].cpu().numpy()
                            elif confidence is not None:
                                conf_scores = confidence.cpu().numpy()
                            else:
                                conf_scores = np.zeros(n_samples)

                            order = np.argsort(conf_scores)[::-1]
                            conf_scores, ligand_pos = conf_scores[order], ligand_pos[order]

                            for rank, (pos, score) in enumerate(zip(ligand_pos, conf_scores), start=1):
                                mol_pred = copy.deepcopy(lig)
                                if self.score_model_args.remove_hs:
                                    mol_pred = RemoveAllHs(mol_pred)
                                sdf_path = os.path.join(tmpdir, f"rank{rank}.sdf")
                                write_mol_with_coords(mol_pred, pos, sdf_path)
                                with open(sdf_path) as fh:
                                    all_results.append({"rank": rank, "confidence": float(score), "ligand_sdf": fh.read()})
                        except Exception as e:
                            all_results.append({"rank": 1, "confidence": float("nan"), "ligand_sdf": f"ERROR (sampling): {e}"})

            except Exception as e:
                all_results.append({"rank": 1, "confidence": float("nan"), "ligand_sdf": f"ERROR (preprocessing): {e}"})

        return pd.DataFrame(all_results) if all_results else pd.DataFrame(columns=["rank", "confidence", "ligand_sdf"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register model in Unity Catalog

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

# COMMAND ----------

signature = ModelSignature(
    inputs=Schema([
        ColSpec("string", "protein_pdb"),
        ColSpec("string", "ligand_smiles"),
        ColSpec("long", "samples_per_complex"),
    ]),
    outputs=Schema([
        ColSpec("long", "rank"),
        ColSpec("double", "confidence"),
        ColSpec("string", "ligand_sdf"),
    ]),
)

def get_version(pkg):
    import importlib.metadata
    return importlib.metadata.version(pkg)

torch_version = torch.__version__
torch_base = torch_version.split("+")[0]
cuda_tag = torch_version.split("+")[1] if "+" in torch_version else "cu118"
pyg_whl_url = f"https://data.pyg.org/whl/torch-{torch_base}+{cuda_tag}.html"

pip_requirements = [
    f"torch=={torch_base}",
    f"torch-geometric=={get_version('torch-geometric')}",
    f"torch-scatter=={get_version('torch-scatter')}",
    f"torch-sparse=={get_version('torch-sparse')}",
    f"torch-cluster=={get_version('torch-cluster')}",
    "pyyaml==6.0.1",
    "scipy==1.7.3",
    "networkx==2.6.3",
    "biopython==1.79",
    "rdkit-pypi==2022.03.5",
    "e3nn==0.5.0",
    "spyrmsd==0.5.2",
    "pandas==1.5.3",
    "biopandas==0.4.1",
    "prody==2.6.1",
    "fair-esm==2.0.0",
]

print(f"PyG find-links: {pyg_whl_url}")
for r in pip_requirements:
    print(f"  {r}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Stage code, bundle artifacts, and log model

# COMMAND ----------

from databricks.sdk import WorkspaceClient

def set_mlflow_experiment(experiment_tag, user_email):
    _token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
    _host = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}"
    w = WorkspaceClient(host=_host, token=_token)
    mlflow_experiment_base_path = "Shared/dbx_genesis_workbench_models"
    w.workspace.mkdirs(f"/Workspace/{mlflow_experiment_base_path}")
    experiment_path = f"/{mlflow_experiment_base_path}/{experiment_tag}"
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    return mlflow.set_experiment(experiment_path)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

experiment = set_mlflow_experiment(experiment_tag=experiment_name, user_email=user_email)

STAGE_DIR = os.path.join(WORK_DIR, "diffdock_staged")
if os.path.exists(STAGE_DIR):
    shutil.rmtree(STAGE_DIR)
os.makedirs(STAGE_DIR, exist_ok=True)

EXCLUDE_DIRS = {"workdir", ".git", "__pycache__", "results", "sample_data", "data"}
EXCLUDE_EXTS = {".zip", ".pt", ".pth", ".ckpt", ".bin", ".safetensors"}

for item in os.listdir(DIFFDOCK_DIR):
    src_path = os.path.join(DIFFDOCK_DIR, item)
    dst_path = os.path.join(STAGE_DIR, item)
    if item in EXCLUDE_DIRS:
        continue
    if os.path.isdir(src_path):
        shutil.copytree(
            src_path, dst_path,
            ignore=shutil.ignore_patterns("*.pt", "*.pth", "*.ckpt", "*.bin", "__pycache__", ".git"),
        )
    elif not any(item.endswith(ext) for ext in EXCLUDE_EXTS):
        shutil.copy2(src_path, dst_path)

for root, dirs, files in os.walk(STAGE_DIR):
    if root == STAGE_DIR:
        continue
    if any(f.endswith(".py") for f in files) and "__init__.py" not in files:
        init_path = os.path.join(root, "__init__.py")
        open(init_path, "w").close()
        print(f"  Added __init__.py to {os.path.relpath(root, STAGE_DIR)}/")

code_items = [os.path.join(STAGE_DIR, item) for item in os.listdir(STAGE_DIR)]
print(f"Staged {len(code_items)} items")

ESM_SAVE_DIR = os.path.join(WORK_DIR, "esm2_model")
if os.path.exists(ESM_SAVE_DIR):
    shutil.rmtree(ESM_SAVE_DIR)
os.makedirs(ESM_SAVE_DIR, exist_ok=True)

esm_hub_cache = os.path.join(torch.hub.get_dir(), "checkpoints")
esm_bundled = 0
for fname in ["esm2_t33_650M_UR50D.pt", "esm2_t33_650M_UR50D-contact-regression.pt"]:
    src = os.path.join(esm_hub_cache, fname)
    if os.path.exists(src):
        size_mb = os.path.getsize(src) / (1024 * 1024)
        shutil.copy2(src, os.path.join(ESM_SAVE_DIR, fname))
        print(f"  Bundled {fname} ({size_mb:.0f} MB)")
        esm_bundled += 1

if esm_bundled == 0:
    raise FileNotFoundError(
        f"ESM2 model not found in {esm_hub_cache}. "
        "Run the ESM embedding cell first to download and cache the model."
    )

# Build a trimmed PDB (first 30 residues of chain A) for the input_example.
with open(pdb_path) as f:
    full_pdb_lines = f.readlines()

MAX_RESIDUES = 30
TARGET_CHAIN = "A"
seen_resseq = set()
trimmed_lines = []
for line in full_pdb_lines:
    if line.startswith(("ATOM", "HETATM")):
        chain_id = line[21]
        if chain_id != TARGET_CHAIN:
            continue
        resseq = line[22:27].strip()
        seen_resseq.add(resseq)
        if len(seen_resseq) > MAX_RESIDUES:
            break
        trimmed_lines.append(line)
    elif line.startswith("END"):
        continue
trimmed_lines.append("END\n")
example_pdb = "".join(trimmed_lines)
print(f"Trimmed input_example PDB: {len(seen_resseq)} residues, "
      f"{len(trimmed_lines)} lines, {len(example_pdb)} chars")

with mlflow.start_run(run_name=f"{model_name}", experiment_id=experiment.experiment_id):
    mlflow.log_params({
        "model": "DiffDock",
        "version": "v1.0 (commit 0f9c419)",
        "pytorch_version": torch.__version__,
        "pyg_version": get_version("torch-geometric"),
        "runtime": "DBR 14.3 LTS ML GPU",
        "esm_bundled": True,
    })

    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=DiffDockModel(),
        artifacts={
            "repo_dir": DIFFDOCK_DIR,
            "score_model_dir": SCORE_MODEL_DIR,
            "confidence_model_dir": CONFIDENCE_MODEL_DIR,
            "esm_model_dir": ESM_SAVE_DIR,
        },
        code_path=code_items,
        pip_requirements=[f"--find-links {pyg_whl_url}"] + pip_requirements,
        signature=signature,
        input_example=pd.DataFrame([{
            "protein_pdb": example_pdb,
            "ligand_smiles": SMILES,
            "samples_per_complex": 5,
        }]),
        registered_model_name=f"{catalog}.{schema}.{model_name}",
    )

print(f"Registered: {catalog}.{schema}.{model_name}")
print(f"Model URI: {model_info.model_uri}")
