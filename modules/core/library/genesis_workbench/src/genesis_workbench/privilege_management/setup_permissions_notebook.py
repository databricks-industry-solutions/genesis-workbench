# Databricks notebook source
# MAGIC %md
# MAGIC # Genesis Workbench App Permissions Setup
# MAGIC
# MAGIC This notebook sets up the permissions system for the Genesis Workbench application.
# MAGIC It creates the permissions table and configures initial admin access.
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC The permissions system manages access to modules and submodules:
# MAGIC - **protein_studies**: Protein folding and analysis workflows
# MAGIC - **nvidia_bionemo**: NVIDIA BioNeMo model workflows
# MAGIC - **single_cell**: Single cell analysis workflows
# MAGIC - **monitoring_alerts**: Monitoring and alerting workflows
# MAGIC - **master_settings**: Administrative settings (admin only)
# MAGIC
# MAGIC ## User Types & Access Levels
# MAGIC - **admin**: Full access to all modules and submodules
# MAGIC - **user**: Module-specific access based on group membership
# MAGIC   - **view**: Read-only access - can view but not modify
# MAGIC   - **full**: Full access - can view and modify

# COMMAND ----------

%pip install -U databricks-sql-connector==4.0.2

# COMMAND ----------

# Create widgets for configuration - these will be parameterized via DAB workflow resources
# DAB task parameters will override these default values at runtime
dbutils.widgets.text("catalog_name", "genesis_workbench", "Catalog Name")
dbutils.widgets.text("schema_name", "permissions", "Schema Name")
dbutils.widgets.text(
    "initial_admin_user", "", "Initial Admin User (leave empty to use current user)"
)
dbutils.widgets.text("default_admin_group", "genesis-admin-group", "Default Admin Group")
dbutils.widgets.dropdown(
    "environment", "dev", ["dev", "staging", "prod"], "Environment"
)

# Get values from widgets
catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
initial_admin_user = dbutils.widgets.get("initial_admin_user")
environment = dbutils.widgets.get("environment")

# Determine admin user - can be configured via DAB variables and passed as environment variable
if initial_admin_user.strip():
    admin_user = initial_admin_user.strip()
else:
    # Use current user as admin
    admin_user = spark.sql("SELECT current_user() as user").collect()[0]["user"]

print(f"Setting up app permissions for:")
print(f"  Catalog: {catalog_name}")
print(f"  Schema: {schema_name}")
print(f"  Environment: {environment}")
print(f"  Initial Admin User: {admin_user}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Required Modules

# COMMAND ----------

import sys
import os
from typing import List, Dict

# Import our custom modules
from permissions_manager_app import AppPermissionsManager
from permissions_config import (
    MODULES,
    USER_TYPES,
    PERMISSION_TYPES,
    ACCESS_LEVELS,
    DEFAULT_GROUPS,
    DEFAULT_CATALOG,
    DEFAULT_SCHEMA,
    PERMISSIONS_TABLE_NAME,
)

print("Successfully imported app permissions modules!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Create Catalog and Schema
# MAGIC
# MAGIC Ensure the catalog and schema exist before creating the permissions table.

# COMMAND ----------

# Create catalog if it doesn't exist
try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
    print(f"✓ Catalog '{catalog_name}' ready")
except Exception as e:
    print(f"Note: Catalog creation failed (may already exist or lack permissions): {e}")

# Create schema if it doesn't exist
try:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")
    print(f"✓ Schema '{catalog_name}.{schema_name}' ready")
except Exception as e:
    print(f"Error creating schema: {e}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Initialize Permissions Manager
# MAGIC
# MAGIC Set up the permissions manager for app usage.

# COMMAND ----------

# Initialize the permissions manager
# Note: In a Databricks app, these would come from environment variables
# For the setup notebook, we'll use the current session
try:
    # Get connection details from current Databricks environment
    server_hostname = spark.conf.get("spark.databricks.workspaceUrl", "")

    # For setup purposes, we'll create a basic manager instance
    # The actual app will use proper credentials
    permissions_manager = AppPermissionsManager(
        catalog_name=catalog_name,
        schema_name=schema_name,
        table_name=PERMISSIONS_TABLE_NAME,
    )
    print("✓ Permissions manager initialized")
except Exception as e:
    print(f"Note: Full manager initialization failed (expected in setup): {e}")
    print("Will create table using Spark SQL directly")
    permissions_manager = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Create Permissions Table
# MAGIC
# MAGIC Create the app permissions table with access level support.

# COMMAND ----------

if permissions_manager:
    try:
        permissions_manager.create_permissions_table()
        print("✓ Permissions table created using AppPermissionsManager")
    except Exception as e:
        print(f"Manager table creation failed: {e}")
        permissions_manager = None

# Fallback: Create table using Spark SQL if manager initialization failed
if not permissions_manager:
    table_sql = f"""
    CREATE OR REPLACE TABLE {catalog_name}.{schema_name}.{PERMISSIONS_TABLE_NAME} (
        module_name STRING NOT NULL COMMENT 'Module name (e.g., protein_studies, nvidia_bionemo)',
        submodule_name STRING COMMENT 'Submodule name (null for module-level access)',
        permission_type STRING NOT NULL COMMENT 'Type of permission (module_access, submodule_access)',
        user_type STRING NOT NULL COMMENT 'User type (admin, user)',
        access_level STRING NOT NULL COMMENT 'Access level (view, full) - admins always have full',
        groups ARRAY<STRING> NOT NULL COMMENT 'Databricks groups with this permission',
        is_active BOOLEAN DEFAULT true COMMENT 'Whether this permission is active',
        created_at TIMESTAMP DEFAULT current_timestamp() COMMENT 'Creation timestamp',
        updated_at TIMESTAMP DEFAULT current_timestamp() COMMENT 'Last update timestamp',
        created_by STRING DEFAULT current_user() COMMENT 'User who created this permission',
        updated_by STRING DEFAULT current_user() COMMENT 'User who last updated this permission'
    )
    USING DELTA
    COMMENT 'Application permissions management for Genesis Workbench modules and submodules'
    TBLPROPERTIES (
        'delta.autoOptimize.optimizeWrite' = 'true',
        'delta.autoOptimize.autoCompact' = 'true',
        'delta.feature.allowColumnDefaults' = 'supported'
    )
    """

    try:
        spark.sql(table_sql)
        print(
            f"Permissions table created: {catalog_name}.{schema_name}.{PERMISSIONS_TABLE_NAME}"
        )
    except Exception as e:
        print(f"Error creating permissions table: {e}")
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Set Up Initial Admin Permissions
# MAGIC
# MAGIC Grant the initial admin user full access to all modules and submodules.
# MAGIC Note: This uses a Spark SQL version of the AppPermissionsManager.setup_admin_permissions method.

# COMMAND ----------

print(f"Setting up admin permissions for user: {admin_user}")

# Use the AppPermissionsManager.setup_admin_permissions method if available
if permissions_manager:
    try:
        permissions_manager.setup_admin_permissions(admin_user)
        print("✓ Initial admin permissions setup completed using AppPermissionsManager!")
    except Exception as e:
        print(f"Manager setup failed: {e}")
        permissions_manager = None

# Fallback: Manual setup if manager is not available
if not permissions_manager:
    print("Falling back to manual setup...")
    
    # Create admin group name based on the admin user
    admin_group = f"genesis-admin-{admin_user.replace('@', '-').replace('.', '-')}"
    print(f"Using admin group: {admin_group}")

    # Insert admin permissions for all modules
    for module_name, module_config in MODULES.items():
        try:
            # Grant module access (admins always get full access)
            module_insert_sql = f"""
            INSERT INTO {catalog_name}.{schema_name}.{PERMISSIONS_TABLE_NAME}
            (module_name, submodule_name, permission_type, user_type, access_level, groups)
            VALUES ('{module_name}', NULL, 'module_access', 'admin', 'full', array('{admin_group}'))
            """
            spark.sql(module_insert_sql)
            print(f"  ✓ Granted full module access: {module_name}")

            # Grant access to all submodules
            for submodule in module_config.submodules:
                submodule_insert_sql = f"""
                INSERT INTO {catalog_name}.{schema_name}.{PERMISSIONS_TABLE_NAME}
                (module_name, submodule_name, permission_type, user_type, access_level, groups)
                VALUES ('{module_name}', '{submodule}', 'submodule_access', 'admin', 'full', array('{admin_group}'))
                """
                spark.sql(submodule_insert_sql)
                print(f"    ✓ Granted full submodule access: {module_name}.{submodule}")

        except Exception as e:
            print(f"  Warning: Permission may already exist for {module_name}: {e}")

    print("✓ Initial admin permissions setup completed!")



# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Verify Setup
# MAGIC
# MAGIC Check that the permissions table was created correctly and contains the expected data.

# COMMAND ----------

# Verify table exists and show structure
try:
    table_info = spark.sql(
        f"DESCRIBE TABLE {catalog_name}.{schema_name}.{PERMISSIONS_TABLE_NAME}"
    )
    print("✓ Table structure:")
    display(table_info)
except Exception as e:
    print(f"Error describing table: {e}")

# Show current permissions
try:
    permissions_data = spark.sql(
        f"""
        SELECT 
            module_name,
            submodule_name,
            permission_type,
            user_type,
            access_level,
            groups,
            created_at
        FROM {catalog_name}.{schema_name}.{PERMISSIONS_TABLE_NAME}
        ORDER BY module_name, submodule_name
    """
    )

    count = permissions_data.count()
    print(f"✓ Permissions table contains {count} records")

    if count > 0:
        print("Current permissions:")
        display(permissions_data)
    else:
        print("Warning: No permissions found in table")

except Exception as e:
    print(f"Error querying permissions: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Access Level Examples
# MAGIC
# MAGIC Show how to grant different access levels to user groups.

# COMMAND ----------

print("=" * 50)
print("ACCESS LEVEL EXAMPLES")
print("=" * 50)
print()
print("To grant permissions via the app UI or programmatically:")
print()
print("# Grant view access to a module")
print("INSERT INTO permissions_table VALUES (..., 'user', 'view', ...)")
print()
print("# Grant full access to a submodule")
print("INSERT INTO permissions_table VALUES (..., 'user', 'full', ...)")
print()
print("Available access levels:")
for level, description in ACCESS_LEVELS.items():
    print(f"  • {level}: {description}")
print()
print("Note: Admins always receive 'full' access regardless of specified level")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Setup Summary
# MAGIC
# MAGIC Display a summary of the setup process.

# COMMAND ----------

print("=" * 60)
print("GENESIS WORKBENCH APP PERMISSIONS SETUP COMPLETE")
print("=" * 60)
print(f"Catalog: {catalog_name}")
print(f"Schema: {schema_name}")
print(f"Table: {PERMISSIONS_TABLE_NAME}")
print(f"Environment: {environment}")
print(f"Initial Admin: {admin_user}")
print(f"Admin Group: {admin_group}")
print()
print("Available Modules:")
for module_name, module_config in MODULES.items():
    print(f"  • {module_config.display_name} ({module_name})")
    for submodule in module_config.submodules:
        print(f"    - {submodule}")
print()
print("Access Levels:")
for level, description in ACCESS_LEVELS.items():
    print(f"  • {level}: {description}")
print()
print("Next Steps:")
print("1. Deploy the Genesis Workbench app")
print("2. Configure app environment variables:")
print("   - DATABRICKS_SERVER_HOSTNAME")
print("   - DATABRICKS_HTTP_PATH")
print("   - DATABRICKS_TOKEN")
print("3. Use the app UI to manage user permissions with access levels")
print(
    "4. Users will automatically get their group memberships from Databricks Identity API"
)
print("5. Grant 'view' or 'full' access levels based on user needs")
print("=" * 60)
