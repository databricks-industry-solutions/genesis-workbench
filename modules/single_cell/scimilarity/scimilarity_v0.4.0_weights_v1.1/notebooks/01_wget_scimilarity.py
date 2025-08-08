# Databricks notebook source
# dbutils.widgets.removeAll()

# COMMAND ----------

# DBTITLE 1,gwb_variablesNparams
dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "dev_srijit_nair_dbx_genesis_workbench_core", "Schema")
dbutils.widgets.text("model_name", "SCimilarity", "Model Name") 
dbutils.widgets.text("experiment_name", "gwb_modules_scimilarity", "Experiment Name")
dbutils.widgets.text("sql_warehouse_id", "8f210e00850a2c16", "SQL Warehouse Id") 
dbutils.widgets.text("user_email", "srijit.nair@databricks.com", "User Id/Email")
dbutils.widgets.text("cache_dir", "scimilarity", "Cache dir")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

MODEL_NAME = dbutils.widgets.get("model_name")
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")
USER_EMAIL = dbutils.widgets.get("user_email")
SQL_WAREHOUSE_ID = dbutils.widgets.get("sql_warehouse_id")

CACHE_DIR = dbutils.widgets.get("cache_dir")

print(f"Cache dir: {CACHE_DIR}")
cache_full_path = f"/Volumes/{CATALOG}/{SCHEMA}/{CACHE_DIR}"
print(f"Cache full path: {cache_full_path}")

# COMMAND ----------

# DBTITLE 1,scimilarity UC paths
CATALOG = CATALOG 
DB_SCHEMA = SCHEMA 
MODEL_FAMILY = CACHE_DIR 

print("CATALOG :", CATALOG)
print("DB_SCHEMA :", DB_SCHEMA)
print("MODEL_FAMILY :", MODEL_FAMILY)

# COMMAND ----------

# DBTITLE 1,Model File Paths
model_path = f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/model/model_v1.1"
geneOrder_path = f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/model/model_v1.1/gene_order.tsv"
sampledata_path = f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/data/adams_etal_2020/GSE136831_subsample.h5ad"

print("model_path :", model_path)
print("geneOrder_path :", geneOrder_path)
print("sampledata_path :", sampledata_path)

# COMMAND ----------

# DBTITLE 1,model and data reference urls
## add to markdown

# Downloading the pretrained models
# You can download the following pretrained models for use with SCimilarity from Zenodo: https://zenodo.org/records/10685499
# Version 1.1
# 10.5281/zenodo.10685499
# Feb 20, 2024

# For Sample Query data. We will use Adams et al., 2020 healthy and IPF lung scRNA-seq data. Download tutorial data.
# https://zenodo.org/records/13685881

# COMMAND ----------

# DBTITLE 1,test scimilarity_setup -- ~60++mins?
import os
import subprocess
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("scimilarity_setup")

class ScimilaritySetup:
    def __init__(self, base_dir=f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}"):
        """Initialize with configurable base directory"""
        self.base_dir = base_dir
        self.model_dir = f"{base_dir}/model"
        self.downloads_dir = f"{base_dir}/downloads"
        self.data_dir = f"{base_dir}/data/adams_etal_2020"
        
        # URLs for downloads
        self.model_url = "http://zenodo.org/records/10685499/files/model_v1.1.tar.gz?download=1"
        self.sample_data_url = "https://zenodo.org/records/13685881/files/GSE136831_subsample.h5ad?download=1"
        
        # File paths
        self.model_tarball = f"{self.downloads_dir}/model_v1.1.tar.gz"
        self.sample_data_path = f"{self.data_dir}/GSE136831_subsample.h5ad"

    def create_directory(self, directory_path):
        """Create directory if it doesn't exist"""
        os.makedirs(directory_path, exist_ok=True)
        logger.info(f"Directory {directory_path} is ready")
    
    def download_file(self, url, destination):
        """Download a file from URL to destination"""
        if os.path.exists(destination):
            logger.info(f"File already exists at {destination}")
            return
        
        logger.info(f"Downloading {url} to {destination}")
        try:
            subprocess.run(["wget", "-O", destination, url, "--verbose"], check=True)
            logger.info(f"Download complete: {destination}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Download failed: {e}")
            raise
    
    def extract_tarball(self, tarball_path, extract_to):
        """Extract a tarball to specified directory"""
        logger.info(f"Extracting {tarball_path} to {extract_to}")
        try:
            subprocess.run(["tar", "--no-same-owner", "-xzvf", tarball_path, "-C", extract_to], check=True)
            logger.info(f"Extraction complete to {extract_to}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Extraction failed: {e}")
            raise
    
    def list_directory(self, directory_path):
        """List files in directory with details"""
        logger.info(f"Contents of {directory_path}:")
        try:
            result = subprocess.run(["ls", "-lah", directory_path], check=True, capture_output=True, text=True)
            logger.info(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to list directory: {e}")
    
    def setup_model(self):
        """Download and extract the model"""
        self.create_directory(self.downloads_dir)
        self.create_directory(self.model_dir)
        
        self.download_file(self.model_url, self.model_tarball)
        
        # Set permissions and extract
        subprocess.run(["chmod", "u+rx", self.model_tarball], check=True)
        self.extract_tarball(self.model_tarball, self.model_dir)
        
        # Verify extraction
        self.list_directory(self.model_dir)
        
        return self.model_dir
    
    def setup_sample_data(self):
        """Download sample data"""
        self.create_directory(self.data_dir)
        self.download_file(self.sample_data_url, self.sample_data_path)
        
        # Verify download
        self.list_directory(self.data_dir)
        
        return self.sample_data_path
    
    def run_full_setup(self):
        """Run complete setup process"""
        start_time = time.time()
        logger.info("Starting SCimilarity setup...")
        
        if os.path.exists(self.model_tarball) and os.path.exists(self.sample_data_path):
            logger.info("Files already exist. Skipping setup.")
            return {
                "model_dir": self.model_dir,
                "sample_data_path": self.sample_data_path
            }
        
        model_dir = self.setup_model()
        sample_data_path = self.setup_sample_data()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Setup complete. SCimilarity model and sample data are ready. Elapsed time: {elapsed_time:.2f} seconds.")
        
        return {
            "model_dir": model_dir,
            "sample_data_path": sample_data_path
        }

# For use in DABs or as a module
def setup_scimilarity(base_dir=f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}", run_model_setup=True, run_data_setup=True):
    """Entry point function to call"""
    setup = ScimilaritySetup(base_dir)
    
    if run_model_setup and run_data_setup:
        return setup.run_full_setup()
    elif run_model_setup:
        return {"model_dir": setup.setup_model()}
    elif run_data_setup:
        return {"sample_data_path": setup.setup_sample_data()}
    else:
        logger.info("No setup functions were run.")
        return {}

# If running directly (not imported)
if __name__ == "__main__":
    setup_scimilarity()
