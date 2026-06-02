# Databricks notebook source
# MAGIC %md
# MAGIC # GWAS Analysis with Glow
# MAGIC
# MAGIC Performs genome-wide association analysis using Glow:
# MAGIC 1. Reads VCF data and filters to phenotyped samples
# MAGIC 2. Normalizes variants against the reference genome
# MAGIC 3. Computes Hardy-Weinberg equilibrium p-values and filters
# MAGIC 4. Runs logistic regression GWAS
# MAGIC
# MAGIC Adapted from mini-glow-demo notebook 3.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("vcf_path", "", "VCF file path")
dbutils.widgets.text("phenotype_path", "", "Phenotype file path")
dbutils.widgets.text("phenotype_column", "phenotype", "Phenotype column")
dbutils.widgets.text("contigs", "6", "Contigs to analyze (comma-separated)")
dbutils.widgets.text("hwe_cutoff", "0.01", "HWE p-value cutoff")
dbutils.widgets.text("pvalue_threshold", "0.01", "GWAS p-value threshold for Firth correction")
dbutils.widgets.text("mlflow_run_id", "", "MLflow Run ID")
dbutils.widgets.text("user_email", "a@b.com", "User Email")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
vcf_path = dbutils.widgets.get("vcf_path")
mlflow_run_id = dbutils.widgets.get("mlflow_run_id")
contigs = dbutils.widgets.get("contigs")
hwe_cutoff = float(dbutils.widgets.get("hwe_cutoff"))
pvalue_threshold = float(dbutils.widgets.get("pvalue_threshold"))

# COMMAND ----------

glow_whl_path = None
for lib in dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries"):
    if lib.name.startswith("glow") and lib.name.endswith(".whl"):
        glow_whl_path = lib.path.replace("dbfs:", "")

print(f"Glow wheel: {glow_whl_path}")

# COMMAND ----------

# MAGIC %pip install {glow_whl_path} --force-reinstall
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
vcf_path = dbutils.widgets.get("vcf_path")
mlflow_run_id = dbutils.widgets.get("mlflow_run_id")
contigs = dbutils.widgets.get("contigs")
hwe_cutoff = float(dbutils.widgets.get("hwe_cutoff"))
pvalue_threshold = float(dbutils.widgets.get("pvalue_threshold"))

# COMMAND ----------

import glow
spark = glow.register(spark)
spark.conf.set("spark.sql.codegen.wholeStage", False)

import pyspark.sql.functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read VCF and save as Delta

# COMMAND ----------

raw_table = f"gwas_raw_vcf_{mlflow_run_id.replace('-', '_')}"

variants_df = spark.read.format('vcf').load(vcf_path)
variants_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.{raw_table}")
variants_df = spark.table(f"{catalog}.{schema}.{raw_table}")

print(f"Loaded {variants_df.count()} variants from {vcf_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Filter to phenotyped samples

# COMMAND ----------

phenotype_table = f"gwas_phenotype_{mlflow_run_id.replace('-', '_')}"
phenotype_df = spark.table(f"{catalog}.{schema}.{phenotype_table}")

filtered_variants_df = variants_df.withColumn(
    "included_sample_ids",
    F.expr("filter(transform(genotypes, g -> struct(g.sampleId, aggregate(g.calls, 0, (acc, x) -> acc + x))), x -> x.col2 >= 1).sampleId")
)
filtered_variants_df = (
    filtered_variants_df
    .join(
        phenotype_df.select("sampleId", "phenotype"),
        on=F.array_contains(filtered_variants_df.included_sample_ids, phenotype_df.sampleId),
        how="leftsemi"
    )
)

sample_ids = phenotype_df.select('sampleId').toPandas()['sampleId'].values
sample_ids_str = ','.join([f"'{sid}'" for sid in sample_ids])
filtered_variants_df = filtered_variants_df.withColumn(
    "genotypes",
    F.expr(f"filter(genotypes, x -> array_contains(array({sample_ids_str}), x.sampleId))")
)

filtered_table = f"gwas_filtered_vcf_{mlflow_run_id.replace('-', '_')}"
filtered_variants_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.{filtered_table}")
filtered_variants_df = spark.table(f"{catalog}.{schema}.{filtered_table}")

print(f"Filtered to {filtered_variants_df.count()} variants with phenotyped samples")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Normalize variants

# COMMAND ----------

REFERENCE_GENOME_PATH = "/dbfs/genesis_workbench/gwas/reference/grch38/GRCh38_full_analysis_set_plus_decoy_hla.fa"

biallelic_df = filtered_variants_df.where(F.size(F.col("alternateAlleles")) == 1)
indels_df = biallelic_df.where((F.length("referenceAllele") > 1) | (F.length(F.col("alternateAlleles")[0]) > 1))
snps_df = biallelic_df.where((F.length("referenceAllele") == 1) & (F.length(F.col("alternateAlleles")[0]) == 1))

normalized_variants_df = glow.transform(
    "normalize_variants",
    indels_df,
    reference_genome_path=REFERENCE_GENOME_PATH
)

normalized_table = f"gwas_normalized_vcf_{mlflow_run_id.replace('-', '_')}"
snps_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.{normalized_table}")
normalized_variants_df.drop("normalizationStatus").write.mode("append").saveAsTable(f"{catalog}.{schema}.{normalized_table}")

delta_vcf = spark.table(f"{catalog}.{schema}.{normalized_table}")
print(f"Normalized variants: {delta_vcf.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compute HWE and filter

# COMMAND ----------

delta_gwas_vcf = delta_vcf \
    .withColumn('values', glow.mean_substitute(glow.genotype_states('genotypes'))) \
    .filter(F.size(F.array_distinct('values')) > 1)

gwas_stats_df = delta_gwas_vcf.select(
    F.expr("*"),
    glow.expand_struct(glow.call_summary_stats(F.col("genotypes"))),
    glow.expand_struct(glow.hardy_weinberg(F.col("genotypes")))
).withColumn(
    "log10pValueHwe",
    F.when(F.col("pValueHwe") == 0, 26).otherwise(-F.log10(F.col("pValueHwe")))
)

summary_table = f"gwas_summary_stats_{mlflow_run_id.replace('-', '_')}"
gwas_stats_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.{summary_table}")
gwas_stats_df = spark.table(f"{catalog}.{schema}.{summary_table}")

passing_count = gwas_stats_df.filter(F.col('pValueHwe') < hwe_cutoff).count()
total_count = gwas_stats_df.count()
print(f"Variants passing HWE filter: {passing_count}/{total_count} ({passing_count/max(total_count,1)*100:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run GWAS logistic regression

# COMMAND ----------

hwe_filtered_df = gwas_stats_df.filter(F.col('pValueHwe') < hwe_cutoff)

phenotype_pdf = phenotype_df.select('sampleId', 'phenotype').toPandas()

unique_sample_ids = hwe_filtered_df.selectExpr("explode(genotypes.sampleId) as sampleId").distinct()
samples_pdf = unique_sample_ids.toPandas()
samples_pdf['sampleId'] = samples_pdf['sampleId'].astype(str)
phenotype_pdf['sampleId'] = phenotype_pdf['sampleId'].astype(str)

phenotype_pdf = phenotype_pdf.merge(samples_pdf, on='sampleId')
phenotype_pdf['phenotype'] = phenotype_pdf['phenotype'].astype(int)
phenotype_pdf.set_index('sampleId', inplace=True)

spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 100)

contig_list = [c.strip() for c in contigs.split(",")]

results = glow.gwas.logistic_regression(
    hwe_filtered_df,
    phenotype_pdf,
    values_column='values',
    correction='approx-firth',
    pvalue_threshold=pvalue_threshold,
    contigs=contig_list
)

results_table = f"gwas_results_{mlflow_run_id.replace('-', '_')}"
results.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.{results_table}")
print(f"GWAS results written to {catalog}.{schema}.{results_table}")

# COMMAND ----------

result_count = spark.table(f"{catalog}.{schema}.{results_table}").count()
print(f"Total GWAS result rows: {result_count}")
