"""
App-Compatible Permissions Manager Module for Genesis Workbench
Provides functions for managing permissions control in Unity Catalog from Databricks Apps.
Uses serverless SQL warehouse instead of Spark for app compatibility.
"""

import logging
import re
from typing import List, Dict, Optional, Union, Any
from databricks import sql
import os
from datetime import datetime
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


class PermissionsManagerApp:
    """App-compatible permissions manager using SQL warehouse connection."""

    def __init__(
        self,
        server_hostname: Optional[str] = None,
        http_path: Optional[str] = None,
        access_token: Optional[str] = None,
        catalog_name: str = DEFAULT_CATALOG,
        schema_name: str = DEFAULT_SCHEMA,
        table_name: str = PERMISSIONS_TABLE_NAME,
    ):
        """
        Initialize the permissions manager with SQL warehouse connection.

        Args:
            server_hostname: Databricks workspace hostname
            http_path: SQL warehouse HTTP path
            access_token: Databricks access token
            catalog_name: Default catalog name
            schema_name: Default schema name
            table_name: Default table name
        """
        self.server_hostname = server_hostname or os.getenv(
            "DATABRICKS_SERVER_HOSTNAME"
        )
        self.http_path = http_path or os.getenv("DATABRICKS_HTTP_PATH")
        self.access_token = access_token or os.getenv("DATABRICKS_TOKEN")
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.table_name = table_name

        if not all([self.server_hostname, self.http_path, self.access_token]):
            raise ValueError(
                "Missing required connection parameters. Provide server_hostname, "
                "http_path, and access_token or set environment variables: "
                "DATABRICKS_SERVER_HOSTNAME, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN"
            )

    def _sanitize_sql_identifier(self, identifier: str) -> str:
        """
        Sanitize SQL identifiers to prevent injection attacks.

        Args:
            identifier: The identifier to sanitize

        Returns:
            Sanitized identifier
        """
        # Allow only alphanumeric characters, underscores, and dots
        if not re.match(r"^[a-zA-Z0-9_\.]+$", identifier):
            raise ValueError(f"Invalid SQL identifier: {identifier}")
        return identifier

    def _sanitize_sql_string(self, value: str) -> str:
        """
        Sanitize SQL string values to prevent injection attacks.

        Args:
            value: The string value to sanitize

        Returns:
            Sanitized string value
        """
        # Escape single quotes and validate basic patterns
        if "'" in value:
            value = value.replace("'", "''")
        # Additional validation for dangerous patterns
        dangerous_patterns = [
            r";\s*(drop|delete|update|insert)",
            r"union\s+select",
            r"--",
            r"/\*",
            r"\*/",
        ]
        value_lower = value.lower()
        for pattern in dangerous_patterns:
            if re.search(pattern, value_lower):
                raise ValueError(f"Potentially dangerous SQL pattern detected: {value}")
        return value

    def _execute_sql(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        fetch_results: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Execute SQL query with proper error handling and connection management.

        Args:
            query: SQL query to execute
            parameters: Optional query parameters
            fetch_results: Whether to fetch and return results

        Returns:
            Query results as list of dictionaries
        """
        connection = None
        cursor = None

        try:
            connection = sql.connect(
                server_hostname=self.server_hostname,
                http_path=self.http_path,
                access_token=self.access_token,
            )

            cursor = connection.cursor()

            if parameters:
                cursor.execute(query, parameters)
            else:
                cursor.execute(query)

            if fetch_results:
                columns = (
                    [desc[0] for desc in cursor.description]
                    if cursor.description
                    else []
                )
                rows = cursor.fetchall()
                results = [dict(zip(columns, row)) for row in rows]
                logger.info(
                    f"Query executed successfully, returned {len(results)} rows"
                )
                return results
            else:
                logger.info("Query executed successfully")
                return []

        except Exception as e:
            logger.error(f"Error executing SQL query: {e}")
            logger.error(f"Query: {query}")
            raise
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

    def create_permissions_table(
        self,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
    ) -> None:
        """
        Create the permissions control table with optimal Delta table properties.

        Args:
            catalog_name: Name of the Unity Catalog
            schema_name: Name of the schema
            table_name: Name of the permissions table
        """
        catalog = self._sanitize_sql_identifier(catalog_name or self.catalog_name)
        schema = self._sanitize_sql_identifier(schema_name or self.schema_name)
        table = self._sanitize_sql_identifier(table_name or self.table_name)

        # Build table properties string
        table_props = ",\n  ".join(
            [f"'{k}' = '{v}'" for k, v in DELTA_TABLE_PROPERTIES.items()]
        )

        create_table_sql = f"""
        CREATE OR REPLACE TABLE {catalog}.{schema}.{table} (
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

        self._execute_sql(create_table_sql, fetch_results=False)
        logger.info(
            f"Successfully created permissions table: {catalog}.{schema}.{table}"
        )

    def insert_permission(
        self,
        workflow_type: str,
        resource: str,
        user_type: str,
        privilege: str,
        groups: List[str],
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
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

        catalog = self._sanitize_sql_identifier(catalog_name or self.catalog_name)
        schema = self._sanitize_sql_identifier(schema_name or self.schema_name)
        table = self._sanitize_sql_identifier(table_name or self.table_name)

        # Sanitize string inputs
        workflow_type = self._sanitize_sql_string(workflow_type)
        resource = self._sanitize_sql_string(resource)
        user_type = self._sanitize_sql_string(user_type)
        privilege = self._sanitize_sql_string(privilege)
        groups_sanitized = [self._sanitize_sql_string(g) for g in groups]

        groups_array = "array(" + ",".join([f"'{g}'" for g in groups_sanitized]) + ")"

        insert_sql = f"""
        INSERT INTO {catalog}.{schema}.{table}
        (workflow_type, resource, user_type, privilege, groups, is_active)
        VALUES ('{workflow_type}', '{resource}', '{user_type}', '{privilege}', 
                {groups_array}, {is_active})
        """

        self._execute_sql(insert_sql, fetch_results=False)
        logger.info(
            f"Successfully inserted permission: {workflow_type}.{resource} for {user_type}"
        )

    def update_permission_groups(
        self,
        workflow_type: str,
        resource: str,
        user_type: str,
        new_groups: List[str],
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
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
        catalog = self._sanitize_sql_identifier(catalog_name or self.catalog_name)
        schema = self._sanitize_sql_identifier(schema_name or self.schema_name)
        table = self._sanitize_sql_identifier(table_name or self.table_name)

        # Sanitize inputs
        workflow_type = self._sanitize_sql_string(workflow_type)
        resource = self._sanitize_sql_string(resource)
        user_type = self._sanitize_sql_string(user_type)
        new_groups_sanitized = [self._sanitize_sql_string(g) for g in new_groups]

        groups_array = (
            "array(" + ",".join([f"'{g}'" for g in new_groups_sanitized]) + ")"
        )

        update_sql = f"""
        UPDATE {catalog}.{schema}.{table}
        SET groups = {groups_array},
            updated_at = current_timestamp(),
            updated_by = current_user()
        WHERE workflow_type = '{workflow_type}' 
          AND resource = '{resource}' 
          AND user_type = '{user_type}'
          AND is_active = true
        """

        self._execute_sql(update_sql, fetch_results=False)
        logger.info(
            f"Successfully updated groups for: {workflow_type}.{resource}.{user_type}"
        )

    def add_group_to_permission(
        self,
        workflow_type: str,
        resource: str,
        user_type: str,
        group_to_add: str,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
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
        catalog = self._sanitize_sql_identifier(catalog_name or self.catalog_name)
        schema = self._sanitize_sql_identifier(schema_name or self.schema_name)
        table = self._sanitize_sql_identifier(table_name or self.table_name)

        # Sanitize inputs
        workflow_type = self._sanitize_sql_string(workflow_type)
        resource = self._sanitize_sql_string(resource)
        user_type = self._sanitize_sql_string(user_type)
        group_to_add = self._sanitize_sql_string(group_to_add)

        update_sql = f"""
        UPDATE {catalog}.{schema}.{table}
        SET groups = array_union(groups, array('{group_to_add}')),
            updated_at = current_timestamp(),
            updated_by = current_user()
        WHERE workflow_type = '{workflow_type}' 
          AND resource = '{resource}' 
          AND user_type = '{user_type}'
          AND is_active = true
        """

        self._execute_sql(update_sql, fetch_results=False)
        logger.info(
            f"Successfully added group {group_to_add} to: {workflow_type}.{resource}.{user_type}"
        )

    def remove_group_from_permission(
        self,
        workflow_type: str,
        resource: str,
        user_type: str,
        group_to_remove: str,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
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
        catalog = self._sanitize_sql_identifier(catalog_name or self.catalog_name)
        schema = self._sanitize_sql_identifier(schema_name or self.schema_name)
        table = self._sanitize_sql_identifier(table_name or self.table_name)

        # Sanitize inputs
        workflow_type = self._sanitize_sql_string(workflow_type)
        resource = self._sanitize_sql_string(resource)
        user_type = self._sanitize_sql_string(user_type)
        group_to_remove = self._sanitize_sql_string(group_to_remove)

        update_sql = f"""
        UPDATE {catalog}.{schema}.{table}
        SET groups = array_remove(groups, '{group_to_remove}'),
            updated_at = current_timestamp(),
            updated_by = current_user()
        WHERE workflow_type = '{workflow_type}' 
          AND resource = '{resource}' 
          AND user_type = '{user_type}'
          AND is_active = true
        """

        self._execute_sql(update_sql, fetch_results=False)
        logger.info(
            f"Successfully removed group {group_to_remove} from: {workflow_type}.{resource}.{user_type}"
        )

    def get_permissions_by_group(
        self,
        group_name: str,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
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
        catalog = self._sanitize_sql_identifier(catalog_name or self.catalog_name)
        schema = self._sanitize_sql_identifier(schema_name or self.schema_name)
        table = self._sanitize_sql_identifier(table_name or self.table_name)

        group_name = self._sanitize_sql_string(group_name)

        query_sql = f"""
        SELECT workflow_type, resource, user_type, privilege, groups, created_at, updated_at
        FROM {catalog}.{schema}.{table}
        WHERE array_contains(groups, '{group_name}') AND is_active = true
        ORDER BY workflow_type, resource, user_type
        """

        permissions = self._execute_sql(query_sql)
        logger.info(f"Found {len(permissions)} permissions for group: {group_name}")
        return permissions

    def get_permissions_by_workflow(
        self,
        workflow_type: str,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
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
        catalog = self._sanitize_sql_identifier(catalog_name or self.catalog_name)
        schema = self._sanitize_sql_identifier(schema_name or self.schema_name)
        table = self._sanitize_sql_identifier(table_name or self.table_name)

        workflow_type = self._sanitize_sql_string(workflow_type)

        query_sql = f"""
        SELECT workflow_type, resource, user_type, privilege, groups, created_at, updated_at
        FROM {catalog}.{schema}.{table}
        WHERE workflow_type = '{workflow_type}' AND is_active = true
        ORDER BY resource, user_type
        """

        permissions = self._execute_sql(query_sql)
        logger.info(
            f"Found {len(permissions)} permissions for workflow: {workflow_type}"
        )
        return permissions

    def check_user_permission(
        self,
        workflow_type: str,
        resource: str,
        user_groups: List[str],
        required_privilege: str = "SELECT",
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
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
        catalog = self._sanitize_sql_identifier(catalog_name or self.catalog_name)
        schema = self._sanitize_sql_identifier(schema_name or self.schema_name)
        table = self._sanitize_sql_identifier(table_name or self.table_name)

        # Sanitize inputs
        workflow_type = self._sanitize_sql_string(workflow_type)
        resource = self._sanitize_sql_string(resource)
        required_privilege = self._sanitize_sql_string(required_privilege)
        user_groups_sanitized = [self._sanitize_sql_string(g) for g in user_groups]

        groups_list = ",".join([f"'{g}'" for g in user_groups_sanitized])

        query_sql = f"""
        SELECT COUNT(*) as permission_count
        FROM {catalog}.{schema}.{table}
        WHERE workflow_type = '{workflow_type}'
          AND resource = '{resource}'
          AND privilege = '{required_privilege}'
          AND is_active = true
          AND EXISTS (
            SELECT 1 FROM VALUES ({groups_list}) as t(group_name)
            WHERE array_contains(groups, group_name)
          )
        """

        result = self._execute_sql(query_sql)
        count = result[0]["permission_count"] if result else 0
        has_permission = count > 0
        logger.info(
            f"User permission check: {workflow_type}.{resource} - {has_permission}"
        )
        return has_permission

    def generate_grant_statements(
        self,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
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
        catalog = self._sanitize_sql_identifier(catalog_name or self.catalog_name)
        schema = self._sanitize_sql_identifier(schema_name or self.schema_name)
        table = self._sanitize_sql_identifier(table_name or self.table_name)

        query_sql = f"""
        SELECT DISTINCT
            CONCAT('GRANT ', privilege, ' ON ', workflow_type, '.', resource, ' TO `', group_name, '`;') AS grant_statement
        FROM {catalog}.{schema}.{table}
        LATERAL VIEW EXPLODE(groups) AS group_name
        WHERE is_active = true
        ORDER BY grant_statement
        """

        result = self._execute_sql(query_sql)
        statements = [row["grant_statement"] for row in result]
        logger.info(f"Generated {len(statements)} GRANT statements")
        return statements

    def audit_permission_changes(
        self,
        days_back: int = 30,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
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
        catalog = self._sanitize_sql_identifier(catalog_name or self.catalog_name)
        schema = self._sanitize_sql_identifier(schema_name or self.schema_name)
        table = self._sanitize_sql_identifier(table_name or self.table_name)

        # Validate days_back is a positive integer
        if not isinstance(days_back, int) or days_back <= 0:
            raise ValueError("days_back must be a positive integer")

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
        FROM {catalog}.{schema}.{table}
        WHERE updated_at >= current_date() - INTERVAL {days_back} DAYS
        ORDER BY updated_at DESC
        """

        changes = self._execute_sql(query_sql)
        logger.info(
            f"Found {len(changes)} permission changes in the last {days_back} days"
        )
        return changes

    def deactivate_permission(
        self,
        workflow_type: str,
        resource: str,
        user_type: str,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
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
        catalog = self._sanitize_sql_identifier(catalog_name or self.catalog_name)
        schema = self._sanitize_sql_identifier(schema_name or self.schema_name)
        table = self._sanitize_sql_identifier(table_name or self.table_name)

        # Sanitize inputs
        workflow_type = self._sanitize_sql_string(workflow_type)
        resource = self._sanitize_sql_string(resource)
        user_type = self._sanitize_sql_string(user_type)

        update_sql = f"""
        UPDATE {catalog}.{schema}.{table}
        SET is_active = false,
            updated_at = current_timestamp(),
            updated_by = current_user()
        WHERE workflow_type = '{workflow_type}'
          AND resource = '{resource}'
          AND user_type = '{user_type}'
          AND is_active = true
        """

        self._execute_sql(update_sql, fetch_results=False)
        logger.info(
            f"Successfully deactivated permission: {workflow_type}.{resource}.{user_type}"
        )


# Convenience functions for easy app integration
def create_permissions_manager(
    server_hostname: Optional[str] = None,
    http_path: Optional[str] = None,
    access_token: Optional[str] = None,
    catalog_name: str = DEFAULT_CATALOG,
    schema_name: str = DEFAULT_SCHEMA,
    table_name: str = PERMISSIONS_TABLE_NAME,
) -> PermissionsManagerApp:
    """
    Factory function to create a PermissionsManagerApp instance.

    Args:
        server_hostname: Databricks workspace hostname
        http_path: SQL warehouse HTTP path
        access_token: Databricks access token
        catalog_name: Default catalog name
        schema_name: Default schema name
        table_name: Default table name

    Returns:
        PermissionsManagerApp instance
    """
    return PermissionsManagerApp(
        server_hostname=server_hostname,
        http_path=http_path,
        access_token=access_token,
        catalog_name=catalog_name,
        schema_name=schema_name,
        table_name=table_name,
    )


# Standalone functions for direct app usage
def get_permissions_by_group_standalone(
    group_name: str,
    server_hostname: Optional[str] = None,
    http_path: Optional[str] = None,
    access_token: Optional[str] = None,
    catalog_name: str = DEFAULT_CATALOG,
    schema_name: str = DEFAULT_SCHEMA,
    table_name: str = PERMISSIONS_TABLE_NAME,
) -> List[Dict[str, Any]]:
    """Standalone function to get permissions by group."""
    manager = create_permissions_manager(
        server_hostname, http_path, access_token, catalog_name, schema_name, table_name
    )
    return manager.get_permissions_by_group(group_name)


def check_user_permission_standalone(
    workflow_type: str,
    resource: str,
    user_groups: List[str],
    required_privilege: str = "SELECT",
    server_hostname: Optional[str] = None,
    http_path: Optional[str] = None,
    access_token: Optional[str] = None,
    catalog_name: str = DEFAULT_CATALOG,
    schema_name: str = DEFAULT_SCHEMA,
    table_name: str = PERMISSIONS_TABLE_NAME,
) -> bool:
    """Standalone function to check user permissions."""
    manager = create_permissions_manager(
        server_hostname, http_path, access_token, catalog_name, schema_name, table_name
    )
    return manager.check_user_permission(
        workflow_type, resource, user_groups, required_privilege
    )


def add_group_to_permission_standalone(
    workflow_type: str,
    resource: str,
    user_type: str,
    group_to_add: str,
    server_hostname: Optional[str] = None,
    http_path: Optional[str] = None,
    access_token: Optional[str] = None,
    catalog_name: str = DEFAULT_CATALOG,
    schema_name: str = DEFAULT_SCHEMA,
    table_name: str = PERMISSIONS_TABLE_NAME,
) -> None:
    """Standalone function to add group to permission."""
    manager = create_permissions_manager(
        server_hostname, http_path, access_token, catalog_name, schema_name, table_name
    )
    manager.add_group_to_permission(workflow_type, resource, user_type, group_to_add)
