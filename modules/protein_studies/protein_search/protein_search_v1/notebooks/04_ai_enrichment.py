# Databricks notebook source
# MAGIC %md
# MAGIC ## AI-Driven Protein Enrichment
# MAGIC
# MAGIC Uses [`ai_query()`](https://docs.databricks.com/en/sql/language-manual/functions/ai_query.html)
# MAGIC with structured outputs to:
# MAGIC 1. Convert scientific organism names to layman-friendly terms via Foundation Model APIs
# MAGIC 2. Optionally create an external model serving endpoint (Azure OpenAI GPT-4o)
# MAGIC 3. Extract protein research information (recent findings, under-researched areas) for drug discovery
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
external_endpoint_name = dbutils.widgets.get("external_endpoint_name")

print(f"catalog:                 {catalog}")
print(f"schema:                  {schema}")
print(f"volume_name:             {volume_name}")
print(f"external_endpoint_name:  {external_endpoint_name}")

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
            ai_query(
                'databricks-claude-sonnet-4-5',
                CONCAT(
                    'As a knowledgeable encyclopedia, provide a simple layman term and brief meaning ',
                    'for this scientific organism name: ', OrganismName
                ),
                responseFormat => 'STRUCT<simple_term: STRING, meaning: STRING>'
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
# MAGIC ### Part 2: Create External Model Serving Endpoint (Optional)
# MAGIC
# MAGIC Uses the Databricks SDK to create an Azure OpenAI GPT-4o external endpoint
# MAGIC for protein research queries. Skip if the endpoint already exists.

# COMMAND ----------

# DBTITLE 1,Check / create external endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    ExternalModel,
    OpenAiConfig,
    AiGatewayConfig,
    AiGatewayUsageTrackingConfig,
    AiGatewayInferenceTableConfig,
)

w = WorkspaceClient()

endpoint_exists = False
try:
    w.serving_endpoints.get(external_endpoint_name)
    endpoint_exists = True
    print(f"Endpoint '{external_endpoint_name}' already exists — skipping creation.")
except Exception:
    print(f"Endpoint '{external_endpoint_name}' not found — will attempt creation if secrets are configured.")

if not endpoint_exists:
    scope_name = "<your_scope_name>"
    scope_key = "openai_api_key"
    openai_api_base = "https://<your-resource>.openai.azure.com/"
    openai_deployment_name = "gpt-4o-2024-11-20"
    openai_api_version = "2025-01-01-preview"

    try:
        w.serving_endpoints.create(
            name=external_endpoint_name,
            config=EndpointCoreConfigInput(
                served_entities=[
                    ServedEntityInput(
                        name="az-openai-completions",
                        external_model=ExternalModel(
                            name="gpt-4o",
                            provider="openai",
                            task="llm/v1/chat",
                            openai_config=OpenAiConfig(
                                openai_api_type="azure",
                                openai_api_key=f"{{{{secrets/{scope_name}/{scope_key}}}}}",
                                openai_api_base=openai_api_base,
                                openai_deployment_name=openai_deployment_name,
                                openai_api_version=openai_api_version,
                            ),
                        ),
                    )
                ]
            ),
            ai_gateway=AiGatewayConfig(
                usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True),
                inference_table_config=AiGatewayInferenceTableConfig(
                    catalog_name=catalog,
                    schema_name=schema,
                    enabled=True,
                ),
            ),
            tags=[
                {"key": "application", "value": "genesis_workbench"},
                {"key": "module", "value": "protein_search"},
            ],
        )
        print(f"Endpoint '{external_endpoint_name}' created successfully.")
    except Exception as e:
        print(f"Endpoint creation skipped or failed: {e}")
        print("Continuing — the endpoint may need manual configuration for your Azure OpenAI credentials.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Part 3: Extract Protein Research Information
# MAGIC
# MAGIC Joins classified proteins with simplified organism names, then calls
# MAGIC `ai_query()` on the external endpoint to extract research insights
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
df_enriched = (
    df_base.withColumn(
        "researchDict",
        F.expr(f"""
            ai_query(
                '{external_endpoint_name}',
                concat('{SYSTEM_PROMPT} Protein: \"', replace(ProteinName, '"', ''), '\"'),
                responseFormat => 'STRUCT<information: STRING, recent_research: STRING, under_researched_areas: STRING>'
            )
        """),
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
