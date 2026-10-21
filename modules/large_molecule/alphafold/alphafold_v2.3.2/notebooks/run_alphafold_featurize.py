# Databricks notebook source
# MAGIC %md
# MAGIC # AlphaFold featurize — serverless CPU (pyhmmer MSA, no conda)
# MAGIC Builds the AlphaFold input features on **serverless CPU** with `pyhmmer`
# MAGIC (pip-only; bundles HMMER's jackhmmer) instead of the classic
# MAGIC Miniconda/`conda env`/binary-jackhmmer bootstrap. Runs the reduced-DBs MSA
# MAGIC (jackhmmer over uniref90 + mgnify + small_bfd), builds a template-free feature
# MAGIC dict (paired with the template-free fold models), and pickles it to the same
# MAGIC Volume path the fold step + GWB app expect:
# MAGIC `/Volumes/{catalog}/{schema}/{model_volume}/results/{run_id}/{run_id}/features.pkl`.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_workbench", "Schema")
dbutils.widgets.text("model_volume", "alphafold", "Volume")
dbutils.widgets.text("run_id", "b3c99d3b49ba4893aa402a4342a70cd1", "Run Id")
dbutils.widgets.text("protein_sequence", "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE", "Protein Sequence")
dbutils.widgets.text("user_email", "a@b.com", "User Email")
# Per-DB cap on target sequences scanned (bounds RAM/time on serverless CPU).
# Read in blocks; a larger cap = deeper MSA. "0" = scan the whole DB.
dbutils.widgets.text("max_msa_seqs", "2000000", "Max MSA target seqs per DB")

# COMMAND ----------

# pyhmmer bundles HMMER (jackhmmer) as a pip wheel — no conda/apt. biopython is
# needed for the AF data-pipeline import (SCOPData shimmed below). No jax/tf here.
# MAGIC %pip install -q pyhmmer==0.12.3 biopython absl-py ml-collections dm-tree
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import io, os, sys, subprocess, tempfile, time, pickle

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("model_volume")
RUN_ID = dbutils.widgets.get("run_id")
PROTEIN_SEQUENCE = dbutils.widgets.get("protein_sequence").strip()
USER_EMAIL = dbutils.widgets.get("user_email")
MAX_MSA_SEQS = int(dbutils.widgets.get("max_msa_seqs"))

BASEDIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/datasets"
# results/{run_id}/{run_id}/ — matches the GWB app's ranked_0.pdb pull path.
OUTDIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/results/{RUN_ID}/{RUN_ID}"
os.makedirs(OUTDIR, exist_ok=True)

# reduced_dbs MSA sources (same as the classic pipeline)
DBS = [
    ("uniref90", f"{BASEDIR}/uniref90/uniref90.fasta"),
    ("mgnify", f"{BASEDIR}/mgnify/mgy_clusters_2022_05.fa"),
    ("small_bfd", f"{BASEDIR}/small_bfd/bfd-first_non_consensus_sequences.fasta"),
]
print("run_id:", RUN_ID, "| seq len:", len(PROTEIN_SEQUENCE), "| out:", OUTDIR)

# COMMAND ----------

# MAGIC %md
# MAGIC ### AlphaFold source + Biopython SCOPData shim
# MAGIC AF-2.3.2 isn't on PyPI → clone it. Biopython ≥1.80 removed `Bio.Data.SCOPData`
# MAGIC (which AF imports); the 3→1 residue map now lives in `Bio.Data.PDBData`.

# COMMAND ----------

import types, Bio.Data
try:
    from Bio.Data import SCOPData  # noqa: F401  (old biopython)
except ImportError:
    from Bio.Data import PDBData
    _shim = types.ModuleType("Bio.Data.SCOPData")
    _shim.protein_letters_3to1 = dict(PDBData.protein_letters_3to1_extended)
    sys.modules["Bio.Data.SCOPData"] = _shim
    Bio.Data.SCOPData = _shim

_af = tempfile.mkdtemp(prefix="af2_")
_repo = os.path.join(_af, "alphafold")
subprocess.run(["git", "clone", "--depth", "1", "--branch", "v2.3.2",
                "https://github.com/google-deepmind/alphafold.git", _repo],
               check=True, capture_output=True, text=True)
sys.path.insert(0, _repo)

import pyhmmer
from alphafold.data import pipeline, parsers

# COMMAND ----------

# MAGIC %md
# MAGIC ### MSA search (pyhmmer jackhmmer, 1 iteration) over each DB

# COMMAND ----------

_ALPHABET = pyhmmer.easel.Alphabet.amino()


def jackhmmer_msa(query_seq: str, db_path: str, cap: int):
    """Single-iteration jackhmmer of query vs a DB (capped) -> AF parsers.Msa."""
    q = pyhmmer.easel.TextSequence(name=b"query", sequence=query_seq).digitize(_ALPHABET)
    block = pyhmmer.easel.DigitalSequenceBlock(_ALPHABET)
    n = 0
    with pyhmmer.easel.SequenceFile(db_path, digital=True, alphabet=_ALPHABET) as sf:
        for s in sf:
            block.append(s)
            n += 1
            if cap and n >= cap:
                break
    for res in pyhmmer.hmmer.jackhmmer([q], block, max_iterations=1, cpus=os.cpu_count()):
        bio = io.BytesIO()
        res.msa.write(bio, "stockholm")
        return parsers.parse_stockholm(bio.getvalue().decode()), n
    return parsers.Msa(sequences=[query_seq], deletion_matrix=[[0] * len(query_seq)],
                       descriptions=["query"]), n


msas = []
for name, path in DBS:
    t0 = time.time()
    msa, scanned = jackhmmer_msa(PROTEIN_SEQUENCE, path, MAX_MSA_SEQS)
    msas.append(msa)
    print(f"{name}: scanned {scanned:,} | MSA rows {len(msa.sequences):,} | {time.time()-t0:.0f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build + pickle the (template-free) feature dict

# COMMAND ----------

L = len(PROTEIN_SEQUENCE)
feature_dict = {
    **pipeline.make_sequence_features(PROTEIN_SEQUENCE, RUN_ID, L),
    **pipeline.make_msa_features(msas),
}
print("total MSA rows:", int(feature_dict["num_alignments"][0]))

features_path = os.path.join(OUTDIR, "features.pkl")
with open(features_path, "wb") as f:
    pickle.dump(feature_dict, f, protocol=4)
print("wrote", features_path)

# also drop the fasta next to it (parity with the classic pipeline)
with open(os.path.join(OUTDIR, f"{RUN_ID}.fasta"), "w") as f:
    f.write(f">{RUN_ID}\n{PROTEIN_SEQUENCE}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Mark featurize complete (GWB progress tag)

# COMMAND ----------

import mlflow

# Best-effort progress tag (RUN_ID is a real MLflow run when dispatched by the GWB
# app; a direct/manual run may not have one — don't fail featurize over tagging).
try:
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    with mlflow.start_run(run_id=RUN_ID):
        mlflow.log_param("mode", "monomer")
        mlflow.log_param("results_path", OUTDIR)
        mlflow.set_tag("job_status", "featurize_complete")
    print("featurize_complete")
except Exception as e:
    print(f"WARN: could not set featurize_complete tag on run {RUN_ID}: {e}")
