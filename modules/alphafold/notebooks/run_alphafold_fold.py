# Databricks notebook source
# MAGIC %md
# MAGIC # Run only the folding process of Alphafold v2.3.2
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

# MAGIC %md
# MAGIC For folding consider GPU size
# MAGIC  - T4 may be ok for small proteins
# MAGIC  - A100 probably preferred for most proteins
# MAGIC  - very long proteins may need to be run on larger GPU

# COMMAND ----------

# DBTITLE 1,prepare conda and dependencies
# MAGIC %sh
# MAGIC
# MAGIC mkdir -p /miniconda3
# MAGIC wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda3/miniconda.sh
# MAGIC bash /miniconda3/miniconda.sh -b -u -p /miniconda3
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

# DBTITLE 1,prepare input files
import os

dbutils.widgets.text("run_name", "")
dbutils.widgets.text("protein", "")

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

protein = dbutils.widgets.get("protein")
run_name = dbutils.widgets.get("run_name")
mode = 'multimer' if ':' in protein else 'monomer'

tmpdir = '/local_disk0/'
tmp_file = os.path.join(tmpdir,run_name+'.fasta') 
with open(tmp_file,'w') as f:
    write(f,protein,mode)
os.environ['AF_FASTA_FILE'] = tmp_file

os.environ['AF_MODE'] = mode

outdir = os.path.join(
    "/Volumes/protein_folding/alphafold/results/",
    run_name
)
if not os.path.exists(outdir):
    os.mkdir(outdir)

print(os.environ['AF_MODE'])
print(os.environ['AF_FASTA_FILE'])

# COMMAND ----------

# MAGIC %md
# MAGIC Additional setup for folding script 
# MAGIC - require chmeical properties file to be placed in the repo
# MAGIC - patch openmm with Deepmind's patch

# COMMAND ----------

# DBTITLE 1,chemical properties file to alphafold lib
# MAGIC %sh
# MAGIC cp /Volumes/protein_folding/alphafold/datasets/common/stereo_chemical_props.txt /miniconda3/envs/alphafold_env/lib/python3.8/site-packages/alphafold/common

# COMMAND ----------

# DBTITLE 1,patch openmmlib
# MAGIC %sh
# MAGIC cd /miniconda3/envs/alphafold_env/lib/python3.8/site-packages
# MAGIC patch -p0 < /alphafold/alphafold/docker/openmm.patch

# COMMAND ----------

# DBTITLE 1,run alphafold folding-only
# MAGIC %sh
# MAGIC # Where databases etc are stored
# MAGIC BASEDIR="/Volumes/protein_folding/alphafold/datasets"
# MAGIC
# MAGIC # Choose output_path
# MAGIC OUTDIR="/Volumes/protein_folding/alphafold/results/"
# MAGIC echo $OUTDIR
# MAGIC
# MAGIC FLAGS="--data_dir=${BASEDIR}\
# MAGIC  --fasta_paths=${AF_FASTA_FILE}\
# MAGIC  --output_dir=${OUTDIR}\
# MAGIC  --db_preset=reduced_dbs\
# MAGIC  --model_preset="${AF_MODE}"\
# MAGIC  --uniref90_database_path="${BASEDIR}/uniref90/uniref90.fasta"\
# MAGIC  --mgnify_database_path="${BASEDIR}/mgnify/mgy_clusters_2022_05.fa"\
# MAGIC  --small_bfd_database_path="${BASEDIR}/small_bfd/bfd-first_non_consensus_sequences.fasta"\
# MAGIC  --template_mmcif_dir="${BASEDIR}/pdb_mmcif/mmcif_files/"\
# MAGIC  --max_template_date=2020-05-14\
# MAGIC  --obsolete_pdbs_path="${BASEDIR}/pdb_mmcif/obsolete.dat"\
# MAGIC  --use_gpu_relax\
# MAGIC  --noonly_featurize\
# MAGIC  --fold_from_precalculated_features"
# MAGIC
# MAGIC if [ "${AF_MODE}" == "multimer" ]; then
# MAGIC   FLAGS="${FLAGS} --uniprot_database_path=${BASEDIR}/uniprot/uniprot.fasta"
# MAGIC   FLAGS="${FLAGS} --pdb_seqres_database_path=${BASEDIR}/pdb_seqres/pdb_seqres.txt"
# MAGIC fi
# MAGIC if [ "${AF_MODE}" == "monomer" ]; then
# MAGIC   FLAGS="${FLAGS} --pdb70_database_path=${BASEDIR}/pdb70/pdb70"
# MAGIC fi
# MAGIC
# MAGIC source /miniconda3/bin/activate
# MAGIC conda activate alphafold_env
# MAGIC python ../scripts/run_alphafold_split.py ${FLAGS}

# COMMAND ----------

# MAGIC %sh
# MAGIC cat ${AF_FASTA_FILE}
