"""
Permissions Manager Module for Genesis Workbench
Provides functions for managing permissions control in Unity Catalog.
"""

import logging
from typing import List, Dict, Optional, Union
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    array,
    array_contains,
    array_union,
    array_remove,
    explode,
    concat,
    lit,
)
from permissions_config import (
    WORKFLOW_TYPES,
    USER_TYPES,
    AVAILABLE_PRIVILEGES,
    DEFAULT_GROUPS,
    DEFAULT_CATALOG,
    DEFAULT_SCHEMA,
    PERMISSIONS_TABLE_NAME,
    PERMISSIONS_TABLE_COMMENT,
    DELTA_TABLE_PROPERTIES,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_spark() -> SparkSession:
    """Get or create SparkSession. Not needed in Databricks notebook,
    but useful for local testing without Databircks connect and for
    open source deployments. Will simply get the active notebook session"""
    return SparkSession.builder.getOrCreate()


def create_permissions_table(
    catalog_name: str = DEFAULT_CATALOG,
    schema_name: str = DEFAULT_SCHEMA,
    table_name: str = PERMISSIONS_TABLE_NAME,
) -> None:
    """
    Create the permissions control table with optimal Delta table properties.

    Args:
        catalog_name: Name of the Unity Catalog
        schema_name: Name of the schema
        table_name: Name of the permissions table
    """
    spark = get_spark()

    # Build table properties string
    table_props = ",\n  ".join(
        [f"'{k}' = '{v}'" for k, v in DELTA_TABLE_PROPERTIES.items()]
    )

    create_table_sql = f"""
    CREATE OR REPLACE TABLE {catalog_name}.{schema_name}.{table_name} (
        workflow_type STRING NOT NULL COMMENT 'Workflow type (protein, bionemo, single_cell, small_molecules)',
        resource STRING NOT NULL COMMENT 'Specific resource name within the workflow',
        user_type STRING NOT NULL COMMENT 'User type (admin, user, service_principal)',
        privilege STRING NOT NULL COMMENT 'Granted privilege level',
        groups ARRAY<STRING> NOT NULL COMMENT 'Associated Databricks groups',
        is_active BOOLEAN DEFAULT true COMMENT 'Whether this permission is active',
        created_at TIMESTAMP DEFAULT current_timestamp() COMMENT 'Creation timestamp',
        updated_at TIMESTAMP DEFAULT current_timestamp() COMMENT 'Last update timestamp',
        created_by STRING DEFAULT current_user() COMMENT 'User who created this permission',
        updated_by STRING DEFAULT current_user() COMMENT 'User who last updated this permission'
    )
    USING DELTA
    COMMENT '{PERMISSIONS_TABLE_COMMENT}'
    TBLPROPERTIES (
        {table_props}
    )
    """

    try:
        spark.sql(create_table_sql)
        logger.info(
            f"Successfully created permissions table: {catalog_name}.{schema_name}.{table_name}"
        )
    except Exception as e:
        logger.error(f"Error creating permissions table: {e}")
        raise


def insert_permission(
    workflow_type: str,
    resource: str,
    user_type: str,
    privilege: str,
    groups: List[str],
    catalog_name: str = DEFAULT_CATALOG,
    schema_name: str = DEFAULT_SCHEMA,
    table_name: str = PERMISSIONS_TABLE_NAME,
    is_active: bool = True,
) -> None:
    """
    Insert a new permission record.

    Args:
        workflow_type: Type of workflow (protein, bionemo, etc.)
        resource: Specific resource name
        user_type: Type of user (admin, user, service_principal)
        privilege: Privilege level to grant
        groups: List of group names
        catalog_name: Catalog name
        schema_name: Schema name
        table_name: Table name
        is_active: Whether the permission is active
    """
    # Validate inputs
    if workflow_type not in WORKFLOW_TYPES:
        raise ValueError(f"Invalid workflow_type: {workflow_type}")
    if user_type not in USER_TYPES:
        raise ValueError(f"Invalid user_type: {user_type}")
    if privilege not in AVAILABLE_PRIVILEGES:
        raise ValueError(f"Invalid privilege: {privilege}")

    spark = get_spark()

    insert_sql = f"""
    INSERT INTO {catalog_name}.{schema_name}.{table_name}
    (workflow_type, resource, user_type, privilege, groups, is_active)
    VALUES ('{workflow_type}', '{resource}', '{user_type}', '{privilege}', 
            array({','.join([f"'{g}'" for g in groups])}), {is_active})
    """

    try:
        spark.sql(insert_sql)
        logger.info(
            f"Successfully inserted permission: {workflow_type}.{resource} for {user_type}"
        )
    except Exception as e:
        logger.error(f"Error inserting permission: {e}")
        raise


def update_permission_groups(
    workflow_type: str,
    resource: str,
    user_type: str,
    new_groups: List[str],
    catalog_name: str = DEFAULT_CATALOG,
    schema_name: str = DEFAULT_SCHEMA,
    table_name: str = PERMISSIONS_TABLE_NAME,
) -> None:
    """
    Update groups for an existing permission.

    Args:
        workflow_type: Type of workflow
        resource: Specific resource name
        user_type: Type of user
        new_groups: New list of groups to assign
        catalog_name: Catalog name
        schema_name: Schema name
        table_name: Table name
    """
    spark = get_spark()

    update_sql = f"""
    UPDATE {catalog_name}.{schema_name}.{table_name}
    SET groups = array({','.join([f"'{g}'" for g in new_groups])}),
        updated_at = current_timestamp(),
        updated_by = current_user()
    WHERE workflow_type = '{workflow_type}' 
      AND resource = '{resource}' 
      AND user_type = '{user_type}'
      AND is_active = true
    """

    try:
        spark.sql(update_sql)
        logger.info(
            f"Successfully updated groups for: {workflow_type}.{resource}.{user_type}"
        )
    except Exception as e:
        logger.error(f"Error updating permission groups: {e}")
        raise


def add_group_to_permission(
    workflow_type: str,
    resource: str,
    user_type: str,
    group_to_add: str,
    catalog_name: str = DEFAULT_CATALOG,
    schema_name: str = DEFAULT_SCHEMA,
    table_name: str = PERMISSIONS_TABLE_NAME,
) -> None:
    """
    Add a group to an existing permission.

    Args:
        workflow_type: Type of workflow
        resource: Specific resource name
        user_type: Type of user
        group_to_add: Group name to add
        catalog_name: Catalog name
        schema_name: Schema name
        table_name: Table name
    """
    spark = get_spark()

    update_sql = f"""
    UPDATE {catalog_name}.{schema_name}.{table_name}
    SET groups = array_union(groups, array('{group_to_add}')),
        updated_at = current_timestamp(),
        updated_by = current_user()
    WHERE workflow_type = '{workflow_type}' 
      AND resource = '{resource}' 
      AND user_type = '{user_type}'
      AND is_active = true
    """

    try:
        spark.sql(update_sql)
        logger.info(
            f"Successfully added group {group_to_add} to: {workflow_type}.{resource}.{user_type}"
        )
    except Exception as e:
        logger.error(f"Error adding group to permission: {e}")
        raise


def remove_group_from_permission(
    workflow_type: str,
    resource: str,
    user_type: str,
    group_to_remove: str,
    catalog_name: str = DEFAULT_CATALOG,
    schema_name: str = DEFAULT_SCHEMA,
    table_name: str = PERMISSIONS_TABLE_NAME,
) -> None:
    """
    Remove a group from an existing permission.

    Args:
        workflow_type: Type of workflow
        resource: Specific resource name
        user_type: Type of user
        group_to_remove: Group name to remove
        catalog_name: Catalog name
        schema_name: Schema name
        table_name: Table name
    """
    spark = get_spark()

    update_sql = f"""
    UPDATE {catalog_name}.{schema_name}.{table_name}
    SET groups = array_remove(groups, '{group_to_remove}'),
        updated_at = current_timestamp(),
        updated_by = current_user()
    WHERE workflow_type = '{workflow_type}' 
      AND resource = '{resource}' 
      AND user_type = '{user_type}'
      AND is_active = true
    """

    try:
        spark.sql(update_sql)
        logger.info(
            f"Successfully removed group {group_to_remove} from: {workflow_type}.{resource}.{user_type}"
        )
    except Exception as e:
        logger.error(f"Error removing group from permission: {e}")
        raise


def get_permissions_by_group(
    group_name: str,
    catalog_name: str = DEFAULT_CATALOG,
    schema_name: str = DEFAULT_SCHEMA,
    table_name: str = PERMISSIONS_TABLE_NAME,
) -> List[Dict]:
    """
    Get all permissions for a specific group.

    Args:
        group_name: Name of the group
        catalog_name: Catalog name
        schema_name: Schema name
        table_name: Table name

    Returns:
        List of permission dictionaries
    """
    spark = get_spark()

    query_sql = f"""
    SELECT workflow_type, resource, user_type, privilege, groups, created_at, updated_at
    FROM {catalog_name}.{schema_name}.{table_name}
    WHERE array_contains(groups, '{group_name}') AND is_active = true
    ORDER BY workflow_type, resource, user_type
    """

    try:
        result = spark.sql(query_sql)
        permissions = [row.asDict() for row in result.collect()]
        logger.info(f"Found {len(permissions)} permissions for group: {group_name}")
        return permissions
    except Exception as e:
        logger.error(f"Error getting permissions by group: {e}")
        raise


def get_permissions_by_workflow(
    workflow_type: str,
    catalog_name: str = DEFAULT_CATALOG,
    schema_name: str = DEFAULT_SCHEMA,
    table_name: str = PERMISSIONS_TABLE_NAME,
) -> List[Dict]:
    """
    Get all permissions for a specific workflow type.

    Args:
        workflow_type: Type of workflow
        catalog_name: Catalog name
        schema_name: Schema name
        table_name: Table name

    Returns:
        List of permission dictionaries
    """
    spark = get_spark()

    query_sql = f"""
    SELECT workflow_type, resource, user_type, privilege, groups, created_at, updated_at
    FROM {catalog_name}.{schema_name}.{table_name}
    WHERE workflow_type = '{workflow_type}' AND is_active = true
    ORDER BY resource, user_type
    """

    try:
        result = spark.sql(query_sql)
        permissions = [row.asDict() for row in result.collect()]
        logger.info(
            f"Found {len(permissions)} permissions for workflow: {workflow_type}"
        )
        return permissions
    except Exception as e:
        logger.error(f"Error getting permissions by workflow: {e}")
        raise


def check_user_permission(
    workflow_type: str,
    resource: str,
    user_groups: List[str],
    required_privilege: str = "SELECT",
    catalog_name: str = DEFAULT_CATALOG,
    schema_name: str = DEFAULT_SCHEMA,
    table_name: str = PERMISSIONS_TABLE_NAME,
) -> bool:
    """
    Check if a user has the required permission for a specific resource.

    Args:
        workflow_type: Type of workflow
        resource: Specific resource name
        user_groups: List of groups the user belongs to
        required_privilege: Required privilege level
        catalog_name: Catalog name
        schema_name: Schema name
        table_name: Table name

    Returns:
        True if user has permission, False otherwise
    """
    spark = get_spark()

    # Create a temporary view for the user's groups
    groups_list = ",".join([f"'{g}'" for g in user_groups])

    query_sql = f"""
    SELECT COUNT(*) as permission_count
    FROM {catalog_name}.{schema_name}.{table_name}
    WHERE workflow_type = '{workflow_type}'
      AND resource = '{resource}'
      AND privilege = '{required_privilege}'
      AND is_active = true
      AND EXISTS (
        SELECT 1 FROM VALUES ({groups_list}) as t(group_name)
        WHERE array_contains(groups, group_name)
      )
    """

    try:
        result = spark.sql(query_sql)
        count = result.collect()[0]["permission_count"]
        has_permission = count > 0
        logger.info(
            f"User permission check: {workflow_type}.{resource} - {has_permission}"
        )
        return has_permission
    except Exception as e:
        logger.error(f"Error checking user permission: {e}")
        raise


def generate_grant_statements(
    catalog_name: str = DEFAULT_CATALOG,
    schema_name: str = DEFAULT_SCHEMA,
    table_name: str = PERMISSIONS_TABLE_NAME,
) -> List[str]:
    """
    Generate GRANT statements from the permissions control table.

    Args:
        catalog_name: Catalog name
        schema_name: Schema name
        table_name: Table name

    Returns:
        List of GRANT statements
    """
    spark = get_spark()

    query_sql = f"""
    SELECT DISTINCT
        CONCAT('GRANT ', privilege, ' ON ', workflow_type, '.', resource, ' TO `', group_name, '`;') AS grant_statement
    FROM {catalog_name}.{schema_name}.{table_name}
    LATERAL VIEW EXPLODE(groups) AS group_name
    WHERE is_active = true
    ORDER BY grant_statement
    """

    try:
        result = spark.sql(query_sql)
        statements = [row["grant_statement"] for row in result.collect()]
        logger.info(f"Generated {len(statements)} GRANT statements")
        return statements
    except Exception as e:
        logger.error(f"Error generating GRANT statements: {e}")
        raise


def audit_permission_changes(
    days_back: int = 30,
    catalog_name: str = DEFAULT_CATALOG,
    schema_name: str = DEFAULT_SCHEMA,
    table_name: str = PERMISSIONS_TABLE_NAME,
) -> List[Dict]:
    """
    Audit recent permission changes.

    Args:
        days_back: Number of days to look back
        catalog_name: Catalog name
        schema_name: Schema name
        table_name: Table name

    Returns:
        List of audit records
    """
    spark = get_spark()

    query_sql = f"""
    SELECT 
        workflow_type,
        resource,
        user_type,
        privilege,
        groups,
        created_by,
        updated_by,
        created_at,
        updated_at
    FROM {catalog_name}.{schema_name}.{table_name}
    WHERE updated_at >= current_date() - INTERVAL {days_back} DAYS
    ORDER BY updated_at DESC
    """

    try:
        result = spark.sql(query_sql)
        changes = [row.asDict() for row in result.collect()]
        logger.info(
            f"Found {len(changes)} permission changes in the last {days_back} days"
        )
        return changes
    except Exception as e:
        logger.error(f"Error auditing permission changes: {e}")
        raise


def setup_default_permissions(
    catalog_name: str = DEFAULT_CATALOG,
    schema_name: str = DEFAULT_SCHEMA,
    table_name: str = PERMISSIONS_TABLE_NAME,
) -> None:
    """
    Set up default permissions for all workflow types.

    Args:
        catalog_name: Catalog name
        schema_name: Schema name
        table_name: Table name
    """
    logger.info("Setting up default permissions for all workflow types...")

    for workflow_name, workflow_config in WORKFLOW_TYPES.items():
        for resource in workflow_config.resources:
            # Set up admin permissions
            for privilege in workflow_config.default_admin_privileges:
                try:
                    insert_permission(
                        workflow_type=workflow_name,
                        resource=resource,
                        user_type="admin",
                        privilege=privilege,
                        groups=DEFAULT_GROUPS["admin"],
                        catalog_name=catalog_name,
                        schema_name=schema_name,
                        table_name=table_name,
                    )
                except Exception as e:
                    logger.warning(f"Permission may already exist: {e}")

            # Set up user permissions
            for privilege in workflow_config.default_user_privileges:
                try:
                    insert_permission(
                        workflow_type=workflow_name,
                        resource=resource,
                        user_type="user",
                        privilege=privilege,
                        groups=DEFAULT_GROUPS["user"],
                        catalog_name=catalog_name,
                        schema_name=schema_name,
                        table_name=table_name,
                    )
                except Exception as e:
                    logger.warning(f"Permission may already exist: {e}")

    logger.info("Default permissions setup completed!")


def deactivate_permission(
    workflow_type: str,
    resource: str,
    user_type: str,
    catalog_name: str = DEFAULT_CATALOG,
    schema_name: str = DEFAULT_SCHEMA,
    table_name: str = PERMISSIONS_TABLE_NAME,
) -> None:
    """
    Deactivate a permission (soft delete).

    Args:
        workflow_type: Type of workflow
        resource: Specific resource name
        user_type: Type of user
        catalog_name: Catalog name
        schema_name: Schema name
        table_name: Table name
    """
    spark = get_spark()

    update_sql = f"""
    UPDATE {catalog_name}.{schema_name}.{table_name}
    SET is_active = false,
        updated_at = current_timestamp(),
        updated_by = current_user()
    WHERE workflow_type = '{workflow_type}'
      AND resource = '{resource}'
      AND user_type = '{user_type}'
      AND is_active = true
    """

    try:
        spark.sql(update_sql)
        logger.info(
            f"Successfully deactivated permission: {workflow_type}.{resource}.{user_type}"
        )
    except Exception as e:
        logger.error(f"Error deactivating permission: {e}")
        raise
