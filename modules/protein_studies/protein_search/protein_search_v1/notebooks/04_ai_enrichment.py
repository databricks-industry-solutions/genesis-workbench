# Databricks notebook source
# MAGIC %md
# MAGIC ## AI-Driven Protein Enrichment
# MAGIC
# MAGIC Uses [`ai_query()`](https://docs.databricks.com/en/sql/language-manual/functions/ai_query.html)
# MAGIC with Databricks Foundation Model APIs to:
# MAGIC 1. Convert scientific organism names to layman-friendly terms
# MAGIC 2. Extract protein research information (recent findings, under-researched areas) for drug discovery
# MAGIC
# MAGIC All LLM calls use `ai_query()` directly with `responseFormat` -- no UDF registration needed.

# COMMAND ----------

# DBTITLE 1,Run utils (declares widgets, creates UC resources)
# MAGIC %run ./utils

# COMMAND ----------

# DBTITLE 1,Read widget values
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume_name = dbutils.widgets.get("volume_name")
foundation_model_endpoint = dbutils.widgets.get("foundation_model_endpoint")

print(f"catalog:                      {catalog}")
print(f"schema:                       {schema}")
print(f"volume_name:                  {volume_name}")
print(f"foundation_model_endpoint:    {foundation_model_endpoint}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Part 1: Simplify Scientific Organism Names
# MAGIC
# MAGIC Extracts unique organism names from classified proteins and uses
# MAGIC `databricks-claude-sonnet-4-5` (Foundation Model API) to produce
# MAGIC simple layman terms and definitions.

# COMMAND ----------

# DBTITLE 1,Extract unique organism names
sDF = spark.table(f"{catalog}.{schema}.proteinclassification_tiny")

sDF.select("OrganismName").distinct().write.mode("overwrite").option(
    "mergeSchema", "true"
).saveAsTable(f"{catalog}.{schema}.tinysample_organism_info")

organism_count = spark.table(f"{catalog}.{schema}.tinysample_organism_info").count()
print(f"Unique organisms: {organism_count}")

# COMMAND ----------

# DBTITLE 1,Simplify organism names via ai_query
orginfo_sDF = spark.sql(f"""
    SELECT
        OrganismName,
        parsed.simple_term AS Organism_SimpleTerm,
        parsed.meaning AS Organism_Definition
    FROM (
        SELECT
            OrganismName,
            from_json(
                ai_query(
                    'databricks-claude-sonnet-4-5',
                    CONCAT(
                        'As a knowledgeable encyclopedia, provide a simple layman term and brief meaning ',
                        'for this scientific organism name: ', OrganismName,
                        '. Respond with JSON only: {{"simple_term": "...", "meaning": "..."}}'
                    ),
                    responseFormat => 'STRUCT<response: STRING>'
                ).response,
                'simple_term STRING, meaning STRING'
            ) AS parsed
        FROM {catalog}.{schema}.tinysample_organism_info
    )
""")

orginfo_sDF.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(
    f"{catalog}.{schema}.tinysample_organism_info_scientificNsimple"
)

print(f"Written to {catalog}.{schema}.tinysample_organism_info_scientificNsimple")
display(spark.table(f"{catalog}.{schema}.tinysample_organism_info_scientificNsimple"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Part 2: Validate Foundation Model Endpoint

# COMMAND ----------

# DBTITLE 1,Validate foundation model endpoint exists
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

try:
    ep = w.serving_endpoints.get(foundation_model_endpoint)
    print(f"Endpoint '{foundation_model_endpoint}' found (state: {ep.state.ready if ep.state else 'unknown'})")
except Exception as e:
    raise RuntimeError(
        f"Foundation model endpoint '{foundation_model_endpoint}' not found. "
        f"Ensure the endpoint exists before running this workflow. Error: {e}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Part 3: Extract Protein Research Information
# MAGIC
# MAGIC Joins classified proteins with simplified organism names, then calls
# MAGIC `ai_query()` on the foundation model endpoint to extract research insights
# MAGIC for each protein.

# COMMAND ----------

# DBTITLE 1,Build base dataset: proteins joined with simplified organisms
from pyspark.sql import functions as F

SYSTEM_PROMPT = (
    "You are a membrane proteins and drug discovery expert. "
    "Analyze the given protein and provide: a brief overview (information), "
    "key recent findings (recent_research), and promising under-researched "
    "drug discovery opportunities (under_researched_areas). Be concise."
)

base_query = f"""
SELECT
    p.ProteinName,
    p.label AS ProteinType,
    p.score AS ProteinClassificationScore,
    p.GeneName,
    p.Sequence,
    p.Molecular_Weight,
    p.OrganismName,
    o.Organism_SimpleTerm
FROM
    {catalog}.{schema}.proteinclassification_tiny p
JOIN
    {catalog}.{schema}.tinysample_organism_info_scientificNsimple o
    ON p.OrganismName = o.OrganismName
"""

df_base = spark.sql(base_query)
print(f"Base dataset row count: {df_base.count()}")

# COMMAND ----------

# DBTITLE 1,Enrich with protein research info via ai_query
RESEARCH_JSON_INSTRUCTION = (
    ' Respond with JSON only: {"information": "...", "recent_research": "...", "under_researched_areas": "..."}'
)

df_enriched = (
    df_base.withColumn(
        "researchDict",
        F.from_json(
            F.expr(f"""
                ai_query(
                    '{foundation_model_endpoint}',
                    concat('{SYSTEM_PROMPT} Protein: \"', replace(ProteinName, '"', ''), '\"{RESEARCH_JSON_INSTRUCTION}'),
                    responseFormat => 'STRUCT<response: STRING>'
                ).response
            """),
            "information STRING, recent_research STRING, under_researched_areas STRING",
        ),
    )
    .select(
        "OrganismName",
        "Organism_SimpleTerm",
        "ProteinName",
        F.col("researchDict.information").alias("information"),
        F.col("researchDict.recent_research").alias("recent_research"),
        F.col("researchDict.under_researched_areas").alias("under_researched_areas"),
        "ProteinType",
        "ProteinClassificationScore",
    )
)

output_table = f"{catalog}.{schema}.tinysample_organism_protein_research_info"
spark.sql(f"DROP TABLE IF EXISTS {output_table}")
df_enriched.write.mode("overwrite").saveAsTable(output_table)

print(f"Results saved to {output_table}")

# COMMAND ----------

# DBTITLE 1,Preview protein research results
display(spark.table(f"{catalog}.{schema}.tinysample_organism_protein_research_info"))
