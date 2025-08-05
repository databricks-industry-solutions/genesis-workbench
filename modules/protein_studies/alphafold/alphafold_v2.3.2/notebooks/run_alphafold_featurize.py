# Databricks notebook source
# MAGIC %md
# MAGIC # Run only the featurization process of Alphafold v2.3.2
# MAGIC
# MAGIC - This notebook sets up the environment for running Alphafold v2.3.2.
# MAGIC - It installs Miniconda and creates a conda environment from a YAML file.
# MAGIC - This is not normally recommended for distributed workloads, but since we will distribute via Workflows and each job runs on 8-16 cores max, single worker is reasonable here 
# MAGIC - The notebook clones the Alphafold repository and checks out version 2.3.2.
# MAGIC   - We use a (Databricks) modified version of the alphafold run script that can run either featurization or folding inpdendently
# MAGIC   - this is essential for splitting CPU and GPU tasks for efficiency
# MAGIC - It includes a Python script to handle protein sequences and prepare input files.
# MAGIC - The script determines if the input is a monomer or multimer and writes the appropriate FASTA file.
# MAGIC - Environment variables are set for the FASTA file and mode.
# MAGIC - The output directory is created if it does not exist.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "dev_srijit_nair_dbx_genesis_workbench_core", "Schema")
dbutils.widgets.text("model_volume", "alphafold", "Volume")
dbutils.widgets.text("run_id", "b3c99d3b49ba4893aa402a4342a70cd1", "Run Id")
dbutils.widgets.text("protein_sequence", "QVQLVESGGGLVQAGGSLRLACIASGRTFHSYVMAWFRQAPGKEREFVAAISWSSTPTYYGESVKGRFTISRDNAKNTVYLQMNRLKPEDTAVYFCAADRGESYYYTRPTEYEFWGQGTQVTVSS", "Protein Sequence")
dbutils.widgets.text("user_email", "srijit.nair@databricks.com", "User Email")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("model_volume")
RUN_ID = dbutils.widgets.get("run_id")
PROTEIN_SEQUENCE = dbutils.widgets.get("protein_sequence")
USER_EMAIL = dbutils.widgets.get("user_email")


# COMMAND ----------

# DBTITLE 1,prepare conda and dependencies
# MAGIC %sh
# MAGIC
# MAGIC mkdir -p /miniconda3
# MAGIC wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda3/miniconda.sh
# MAGIC bash /miniconda3/miniconda.sh -b -u -p /miniconda3
# MAGIC
# MAGIC cat > /miniconda3/.condarc <<EOF
# MAGIC channels:
# MAGIC   - conda-forge
# MAGIC   - bioconda
# MAGIC   - nodefaults
# MAGIC channel_priority: strict
# MAGIC EOF
# MAGIC
# MAGIC rm -rf /miniconda3/miniconda.sh
# MAGIC
# MAGIC source /miniconda3/bin/activate
# MAGIC
# MAGIC conda env create -f ../envs/alphafold_env.yml 
# MAGIC
# MAGIC mkdir -p /alphafold
# MAGIC cd /alphafold
# MAGIC git clone https://github.com/google-deepmind/alphafold.git
# MAGIC cd alphafold
# MAGIC git checkout v2.3.2
# MAGIC cd /
# MAGIC
# MAGIC conda activate alphafold_env
# MAGIC pip install --no-deps /alphafold/alphafold

# COMMAND ----------

from datetime import datetime

# Get the current datetime object
now = datetime.now()

# Format the datetime object into the desired string format
# %Y for 4-digit year
# %m for 2-digit month (with leading zero)
# %d for 2-digit day (with leading zero)
# %H for 2-digit hour (24-hour format, with leading zero)
# %M for 2-digit minute (with leading zero)
# %S for 2-digit second (with leading zero)
formatted_datetime = now.strftime("%Y%m%d_%H%M%S")

# Print the result
print(formatted_datetime)

# COMMAND ----------

# DBTITLE 1,prepare input files
import os
from datetime import datetime

def write_monomer(f,protein):
    f.writelines(['>protein\n',protein])

def write_multimer(f,protein):
    for i,p in enumerate(protein.split(':')):
        f.write('>chain_{}\n'.format(i))
        f.write(p+'\n')

def write(f,protein,mode):
    if mode=='monomer':
        write_monomer(f,protein)
    elif mode=='multimer':
        write_multimer(f,protein)
    else:
        raise ValueError('no mode {} is avaliable, only monomer or multimer'.format(mode))

mode = 'multimer' if ':' in PROTEIN_SEQUENCE else 'monomer'

tmpdir = '/local_disk0/'
tmp_file = os.path.join(tmpdir,RUN_ID+'.fasta') 
with open(tmp_file,'w') as f:
    write(f,PROTEIN_SEQUENCE,mode)

# Where databases etc are stored
BASEDIR=f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/datasets"

# Get the current datetime object
now = datetime.now()
formatted_datetime = now.strftime("%Y%m%d_%H%M%S")
#Where results are stored
OUTDIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/results/{RUN_ID}"

if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)

os.environ['BASEDIR'] = BASEDIR
os.environ['OUTDIR'] = OUTDIR

os.environ['AF_FASTA_FILE'] = tmp_file
os.environ['AF_MODE'] = mode

print(os.environ['BASEDIR'])
print(os.environ['OUTDIR'])
print(os.environ['AF_MODE'])
print(os.environ['AF_FASTA_FILE'])

# COMMAND ----------

# DBTITLE 1,run alphafold - featurize
# MAGIC %sh
# MAGIC
# MAGIC FLAGS="--data_dir=${BASEDIR}\
# MAGIC   --fasta_paths=${AF_FASTA_FILE}\
# MAGIC   --output_dir=${OUTDIR}\
# MAGIC   --db_preset=reduced_dbs\
# MAGIC   --model_preset="${AF_MODE}"\
# MAGIC   --uniref90_database_path="${BASEDIR}/uniref90/uniref90.fasta"\
# MAGIC   --mgnify_database_path="${BASEDIR}/mgnify/mgy_clusters_2022_05.fa"\
# MAGIC   --small_bfd_database_path="${BASEDIR}/small_bfd/bfd-first_non_consensus_sequences.fasta"\
# MAGIC   --template_mmcif_dir="${BASEDIR}/pdb_mmcif/mmcif_files/"\
# MAGIC   --max_template_date=2020-05-14\
# MAGIC   --obsolete_pdbs_path="${BASEDIR}/pdb_mmcif/obsolete.dat"\
# MAGIC   --nouse_gpu_relax\
# MAGIC   --only_featurize\
# MAGIC   --nofold_from_precalculated_features"
# MAGIC
# MAGIC if [ "${AF_MODE}" == "multimer" ]; then
# MAGIC   FLAGS="${FLAGS} --uniprot_database_path=${BASEDIR}/uniprot/uniprot.fasta"
# MAGIC   FLAGS="${FLAGS} --pdb_seqres_database_path=${BASEDIR}/pdb_seqres/pdb_seqres.txt"
# MAGIC fi
# MAGIC if [ "${AF_MODE}" == "monomer" ]; then
# MAGIC   FLAGS="${FLAGS} --pdb70_database_path=${BASEDIR}/pdb70/pdb70"
# MAGIC fi
# MAGIC echo $FLAGS
# MAGIC
# MAGIC source /miniconda3/bin/activate
# MAGIC conda activate alphafold_env
# MAGIC
# MAGIC echo ""
# MAGIC echo "===Running alphafold"
# MAGIC python ../scripts/run_alphafold_split.py ${FLAGS}
# MAGIC echo ""
# MAGIC echo "===Run complete, Copying results"
# MAGIC
# MAGIC cp ${AF_FASTA_FILE} "${OUTDIR}/$(basename "$AF_FASTA_FILE" .fasta)/"

# COMMAND ----------

import os
import mlflow

# COMMAND ----------

with mlflow.start_run(run_id=RUN_ID) as run:
  mlflow.log_param("protein_sequence", PROTEIN_SEQUENCE)
  mlflow.log_param("mode", mode)  
  mlflow.log_param("results_path", OUTDIR)
  mlflow.log_param("fasta_file", os.path.basename(os.environ['AF_FASTA_FILE']))
  mlflow.set_tag("job_status","featurize_complete")

# COMMAND ----------


