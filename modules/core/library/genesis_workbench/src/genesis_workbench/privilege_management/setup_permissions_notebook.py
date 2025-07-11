# Databricks notebook source
# MAGIC %md
# MAGIC # Genesis Workbench Permissions Setup
# MAGIC
# MAGIC This notebook demonstrates how to set up and configure the permissions control system for Genesis Workbench.
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC The permissions system manages access control for different workflows:
# MAGIC - **protein**: Protein folding and analysis workflows
# MAGIC - **bionemo**: NVIDIA BioNeMo model workflows
# MAGIC - **single_cell**: Single cell analysis workflows
# MAGIC - **small_molecules**: Small molecule analysis workflows
# MAGIC
# MAGIC ## User Types
# MAGIC - **admin**: Full administrative access
# MAGIC - **user**: Regular user access
# MAGIC - **service_principal**: Automated process access

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC Set up the catalog and schema for the permissions system.

# COMMAND ----------

# Note: dbutils, spark, and display are globally available in Databricks notebooks
# These imports are for type hints and linting support only
try:
    from pyspark.sql import SparkSession
    from pyspark.dbutils import DBUtils

    # dbutils and spark are automatically available in Databricks
    # These variables are defined in the Databricks runtime
    dbutils  # type: ignore
    spark  # type: ignore
    display  # type: ignore
except (ImportError, NameError):
    # For local development/testing
    pass

# Create widgets for configuration
dbutils.widgets.text("catalog_name", "genesis_workbench", "Catalog Name")
dbutils.widgets.text("schema_name", "permissions", "Schema Name")
dbutils.widgets.dropdown(
    "environment", "dev", ["dev", "staging", "prod"], "Environment"
)

# Get values from widgets
catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
environment = dbutils.widgets.get("environment")

print(f"Setting up permissions for:")
print(f"  Catalog: {catalog_name}")
print(f"  Schema: {schema_name}")
print(f"  Environment: {environment}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Required Modules
# MAGIC
# MAGIC Import the permissions manager and configuration modules.

# COMMAND ----------

import sys
import os
from typing import List, Dict

# Import our custom modules
from permissions_manager import (
    create_permissions_table,
    setup_default_permissions,
    insert_permission,
    get_permissions_by_workflow,
    get_permissions_by_group,
    check_user_permission,
    generate_grant_statements,
    audit_permission_changes,
)

from permissions_config import (
    WORKFLOW_TYPES,
    USER_TYPES,
    AVAILABLE_PRIVILEGES,
    DEFAULT_GROUPS,
)

print("Successfully imported permissions management modules!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Create Catalog and Schema
# MAGIC
# MAGIC Ensure the catalog and schema exist before creating the permissions table.

# COMMAND ----------

# Create catalog if it doesn't exist
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
print(f"✓ Catalog '{catalog_name}' ready")

# Create schema if it doesn't exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")
print(f"✓ Schema '{catalog_name}.{schema_name}' ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Create Permissions Table
# MAGIC
# MAGIC Create the main permissions control table with optimal Delta table properties.

# COMMAND ----------

# Create the permissions table
create_permissions_table(catalog_name=catalog_name, schema_name=schema_name)

print(f"✓ Permissions table created successfully!")

# Verify table structure
display(
    spark.sql(
        f"DESCRIBE TABLE EXTENDED {catalog_name}.{schema_name}.permissions_control"
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Set Up Default Permissions
# MAGIC
# MAGIC Configure default permissions for all workflow types and user types.

# COMMAND ----------

# Set up default permissions for all workflows
setup_default_permissions(catalog_name=catalog_name, schema_name=schema_name)

print("✓ Default permissions configured!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Verify Permissions Setup
# MAGIC
# MAGIC Check that permissions have been set up correctly.

# COMMAND ----------

# Check permissions for each workflow type
print("=== Permissions Summary ===")
for workflow_name in WORKFLOW_TYPES.keys():
    permissions = get_permissions_by_workflow(
        workflow_type=workflow_name, catalog_name=catalog_name, schema_name=schema_name
    )
    print(f"\n{workflow_name.upper()} Workflow:")
    for perm in permissions:
        print(
            f"  - {perm['resource']} | {perm['user_type']} | {perm['privilege']} | {perm['groups']}"
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Display All Permissions
# MAGIC
# MAGIC View all permissions in the system.

# COMMAND ----------

# Display all permissions
permissions_df = spark.sql(
    f"""
    SELECT 
        workflow_type,
        resource,
        user_type,
        privilege,
        groups,
        is_active,
        created_at,
        updated_at
    FROM {catalog_name}.{schema_name}.permissions_control
    WHERE is_active = true
    ORDER BY workflow_type, resource, user_type
"""
)

display(permissions_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Add Custom Permissions (Optional)
# MAGIC
# MAGIC Add any custom permissions specific to your environment.

# COMMAND ----------

# Example: Add a custom group for contractors
if environment == "dev":
    print("Adding development-specific permissions...")

    # Add contractors group to single_cell workflow
    insert_permission(
        workflow_type="single_cell",
        resource="output_tables",
        user_type="user",
        privilege="SELECT",
        groups=["contractors", "external-users"],
        catalog_name=catalog_name,
        schema_name=schema_name,
    )

    print("✓ Development permissions added!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Test Permission Checking
# MAGIC
# MAGIC Test the permission checking functionality.

# COMMAND ----------

# Test permission checking for different users
test_cases = [
    {
        "description": "Admin user accessing protein model",
        "workflow_type": "protein",
        "resource": "model",
        "user_groups": ["genesis-admin-group"],
        "required_privilege": "OWNER",
    },
    {
        "description": "Regular user accessing single_cell output",
        "workflow_type": "single_cell",
        "resource": "output_tables",
        "user_groups": ["genesis-users"],
        "required_privilege": "SELECT",
    },
    {
        "description": "Contractor accessing protein model (should fail)",
        "workflow_type": "protein",
        "resource": "model",
        "user_groups": ["contractors"],
        "required_privilege": "OWNER",
    },
]

print("=== Permission Check Tests ===")
for test in test_cases:
    has_permission = check_user_permission(
        workflow_type=test["workflow_type"],
        resource=test["resource"],
        user_groups=test["user_groups"],
        required_privilege=test["required_privilege"],
        catalog_name=catalog_name,
        schema_name=schema_name,
    )

    status = "✓ PASS" if has_permission else "✗ FAIL"
    print(f"{status} - {test['description']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Generate Grant Statements
# MAGIC
# MAGIC Generate Unity Catalog GRANT statements from the permissions table.

# COMMAND ----------

# Generate GRANT statements
grant_statements = generate_grant_statements(
    catalog_name=catalog_name, schema_name=schema_name
)

print("=== Generated GRANT Statements ===")
for statement in grant_statements[:10]:  # Show first 10
    print(statement)

if len(grant_statements) > 10:
    print(f"... and {len(grant_statements) - 10} more statements")

print(f"\nTotal statements generated: {len(grant_statements)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Audit and Monitoring
# MAGIC
# MAGIC Check recent permission changes and set up monitoring.

# COMMAND ----------

# Audit recent permission changes
recent_changes = audit_permission_changes(
    days_back=7, catalog_name=catalog_name, schema_name=schema_name
)

print(f"=== Recent Permission Changes (Last 7 Days) ===")
print(f"Found {len(recent_changes)} changes")

for change in recent_changes[:5]:  # Show first 5
    print(
        f"- {change['workflow_type']}.{change['resource']} | {change['user_type']} | {change['updated_at']}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10: Application Integration Examples
# MAGIC
# MAGIC Examples of how to integrate with your Databricks application.

# COMMAND ----------


# Example function for application integration
def user_has_workflow_access(
    user_groups: List[str], workflow_type: str
) -> Dict[str, bool]:
    """
    Check if a user has access to different aspects of a workflow.
    This is what your application would call to determine UI visibility.
    """
    access_check = {}

    # Check different privilege levels
    privileges_to_check = ["SELECT", "EXECUTE", "MODIFY", "OWNER"]

    for privilege in privileges_to_check:
        access_check[privilege.lower()] = check_user_permission(
            workflow_type=workflow_type,
            resource="model",  # Check against model resource
            user_groups=user_groups,
            required_privilege=privilege,
            catalog_name=catalog_name,
            schema_name=schema_name,
        )

    return access_check


# Test with different user types
print("=== Application Integration Examples ===")

# Admin user
admin_access = user_has_workflow_access(["genesis-admin-group"], "protein")
print(f"Admin access to protein workflow: {admin_access}")

# Regular user
user_access = user_has_workflow_access(["genesis-users"], "single_cell")
print(f"User access to single_cell workflow: {user_access}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 11: Setup Completion Summary
# MAGIC
# MAGIC Summary of what was configured.

# COMMAND ----------

print("🎉 PERMISSIONS SETUP COMPLETE! 🎉")
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"✓ Catalog: {catalog_name}")
print(f"✓ Schema: {schema_name}")
print(f"✓ Environment: {environment}")
print(f"✓ Permissions table created with Delta optimizations")
print(f"✓ Default permissions configured for {len(WORKFLOW_TYPES)} workflows")
print(f"✓ {len(grant_statements)} GRANT statements generated")
print(f"✓ Permission checking functions ready for application integration")

print("\n" + "=" * 50)
print("NEXT STEPS")
print("=" * 50)
print("1. Integrate permission checking into your Databricks application")
print("2. Set up monitoring alerts for permission changes")
print("3. Configure group membership in your identity provider")
print("4. Test with real users before production deployment")
print("5. Set up automated backups of the permissions table")

print("\n" + "=" * 50)
print("USAGE IN YOUR APPLICATION")
print("=" * 50)
print("from permissions_manager import check_user_permission")
print("")
print("# Check if user can access a workflow")
print("has_access = check_user_permission(")
print("    workflow_type='protein',")
print("    resource='model',")
print("    user_groups=['genesis-users'],")
print("    required_privilege='SELECT'")
print(")")
print("")
print("# Use has_access to show/hide UI elements")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleanup (Optional)
# MAGIC
# MAGIC Uncomment the cell below to clean up the permissions table if needed.

# COMMAND ----------

# # WARNING: This will delete all permissions data!
# # Uncomment only if you want to start over
#
# # spark.sql(f"DROP TABLE IF EXISTS {catalog_name}.{schema_name}.permissions_control")
# # print("⚠️  Permissions table dropped!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resources
# MAGIC
# MAGIC - [Unity Catalog Permissions Documentation](https://docs.databricks.com/security/unity-catalog/manage-privileges/index.html)
# MAGIC - [Delta Table Optimizations](https://docs.databricks.com/delta/optimizations/index.html)
# MAGIC - [Databricks Groups and Service Principals](https://docs.databricks.com/administration-guide/users-groups/index.html)
