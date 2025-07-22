"""
App Permissions Manager for Genesis Workbench
Manages module and submodule access permissions for the Databricks app.
Uses SQL warehouse for database operations and Databricks Identity API for user groups.
"""

import logging
import re
import requests
from typing import List, Dict, Optional, Set
from databricks import sql
import os
from datetime import datetime
from permissions_config import (
    MODULES,
    USER_TYPES,
    PERMISSION_TYPES,
    ACCESS_LEVELS,
    DEFAULT_GROUPS,
    DEFAULT_CATALOG,
    DEFAULT_SCHEMA,
    PERMISSIONS_TABLE_NAME,
    PERMISSIONS_TABLE_COMMENT,
    DELTA_TABLE_PROPERTIES,
    DATABRICKS_API_VERSION,
    DATABRICKS_SCIM_API_VERSION,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AppPermissionsManager:
    """App-specific permissions manager for Genesis Workbench modules and submodules."""

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
        Initialize the app permissions manager.

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
        """Sanitize SQL identifiers to prevent injection attacks."""
        if not re.match(r"^[a-zA-Z0-9_\.]+$", identifier):
            raise ValueError(f"Invalid SQL identifier: {identifier}")
        return identifier

    def _sanitize_sql_string(self, value: str) -> str:
        """Sanitize SQL string values to prevent injection attacks."""
        if "'" in value:
            value = value.replace("'", "''")
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
        parameters: Optional[Dict] = None,
        fetch_results: bool = True,
    ) -> List[Dict]:
        """Execute SQL query with proper error handling."""
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

    def get_current_user_groups(self) -> List[str]:
        """
        Fetch the current user's groups using Databricks Identity Management API.

        Returns:
            List of group names the current user belongs to
        """
        try:
            # Get current user info first
            user_url = f"https://{self.server_hostname}/api/2.0/preview/scim/v2/Me"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            }

            response = requests.get(user_url, headers=headers)
            response.raise_for_status()
            user_data = response.json()

            user_id = user_data.get("id")
            if not user_id:
                logger.warning("Could not get current user ID")
                return []

            # Get user's groups
            groups_url = f"https://{self.server_hostname}/api/2.0/preview/scim/v2/Users/{user_id}"
            response = requests.get(groups_url, headers=headers)
            response.raise_for_status()
            user_details = response.json()

            groups = []
            if "groups" in user_details:
                for group in user_details["groups"]:
                    groups.append(group.get("display", ""))

            logger.info(f"Found {len(groups)} groups for current user")
            return groups

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching user groups from API: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching user groups: {e}")
            return []

    def create_permissions_table(self) -> None:
        """Create the app permissions table with access level support."""
        catalog = self._sanitize_sql_identifier(self.catalog_name)
        schema = self._sanitize_sql_identifier(self.schema_name)
        table = self._sanitize_sql_identifier(self.table_name)

        table_props = ",\n  ".join(
            [f"'{k}' = '{v}'" for k, v in DELTA_TABLE_PROPERTIES.items()]
        )

        create_table_sql = f"""
        CREATE OR REPLACE TABLE {catalog}.{schema}.{table} (
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
        COMMENT '{PERMISSIONS_TABLE_COMMENT}'
        TBLPROPERTIES (
            {table_props}
        )
        """

        try:
            self._execute_sql(create_table_sql, fetch_results=False)
            logger.info(
                f"Successfully created permissions table: {catalog}.{schema}.{table}"
            )
        except Exception as e:
            logger.error(f"Error creating permissions table: {e}")
            raise

    def grant_module_access(
        self,
        module_name: str,
        groups: List[str],
        user_type: str = "user",
        access_level: str = "view",
        submodule_name: Optional[str] = None,
    ) -> None:
        """
        Grant access to a module or submodule for specified groups.

        Args:
            module_name: Name of the module
            groups: List of group names to grant access
            user_type: Type of user (admin, user)
            access_level: Level of access (view, full) - ignored for admins who always get full
            submodule_name: Optional submodule name for submodule-specific access
        """
        # Validate inputs
        if module_name not in MODULES:
            raise ValueError(f"Invalid module: {module_name}")
        if user_type not in USER_TYPES:
            raise ValueError(f"Invalid user_type: {user_type}")
        if access_level not in ACCESS_LEVELS:
            raise ValueError(f"Invalid access_level: {access_level}")
        if submodule_name and submodule_name not in MODULES[module_name].submodules:
            raise ValueError(
                f"Invalid submodule: {submodule_name} for module: {module_name}"
            )

        # Admins always get full access
        if user_type == "admin":
            access_level = "full"

        # Sanitize inputs
        module = self._sanitize_sql_string(module_name)
        user_t = self._sanitize_sql_string(user_type)
        access_l = self._sanitize_sql_string(access_level)
        submodule = (
            self._sanitize_sql_string(submodule_name) if submodule_name else None
        )

        permission_type = "submodule_access" if submodule_name else "module_access"
        groups_sql = ",".join([f"'{self._sanitize_sql_string(g)}'" for g in groups])

        insert_sql = f"""
        INSERT INTO {self.catalog_name}.{self.schema_name}.{self.table_name}
        (module_name, submodule_name, permission_type, user_type, access_level, groups)
        VALUES ('{module}', {f"'{submodule}'" if submodule else "NULL"}, '{permission_type}', '{user_t}', '{access_l}', array({groups_sql}))
        """

        try:
            self._execute_sql(insert_sql, fetch_results=False)
            target = (
                f"{module_name}.{submodule_name}" if submodule_name else module_name
            )
            logger.info(
                f"Successfully granted {access_level} {permission_type} to {target} for groups: {groups}"
            )
        except Exception as e:
            logger.error(f"Error granting module access: {e}")
            raise

    def check_user_module_access(
        self,
        user_groups: List[str],
        module_name: str,
        submodule_name: Optional[str] = None,
        required_access_level: str = "view",
    ) -> bool:
        """
        Check if user has access to a module or submodule at the required level.

        Args:
            user_groups: List of groups the user belongs to
            module_name: Name of the module to check
            submodule_name: Optional submodule name
            required_access_level: Required access level (view, full)

        Returns:
            True if user has access, False otherwise
        """
        if not user_groups:
            return False

        module = self._sanitize_sql_string(module_name)
        submodule = (
            self._sanitize_sql_string(submodule_name) if submodule_name else None
        )

        # For submodule access, user needs both module access and submodule access
        if submodule_name:
            # Check module access first
            module_access = self._check_access_query(
                user_groups, module, None, required_access_level
            )
            if not module_access:
                return False
            # Then check submodule access
            return self._check_access_query(
                user_groups, module, submodule, required_access_level
            )
        else:
            # Just check module access
            return self._check_access_query(
                user_groups, module, None, required_access_level
            )

    def _check_access_query(
        self,
        user_groups: List[str],
        module_name: str,
        submodule_name: Optional[str],
        required_access_level: str,
    ) -> bool:
        """Helper method to execute access check query with access level."""
        groups_list = ",".join(
            [f"'{self._sanitize_sql_string(g)}'" for g in user_groups]
        )

        submodule_condition = (
            f"AND submodule_name = '{submodule_name}'"
            if submodule_name
            else "AND submodule_name IS NULL"
        )

        # Check for required access level or higher
        # If user has 'full' access, they can do anything
        # If user has 'view' access, they can only view
        access_condition = (
            f"AND (access_level = 'full' OR access_level = '{required_access_level}')"
            if required_access_level == "view"
            else "AND access_level = 'full'"
        )

        query_sql = f"""
        SELECT COUNT(*) as access_count
        FROM {self.catalog_name}.{self.schema_name}.{self.table_name}
        WHERE module_name = '{module_name}'
          {submodule_condition}
          {access_condition}
          AND is_active = true
          AND EXISTS (
            SELECT 1 FROM VALUES ({groups_list}) as t(group_name)
            WHERE array_contains(groups, group_name)
          )
        """

        try:
            result = self._execute_sql(query_sql)
            count = result[0]["access_count"] if result else 0
            return count > 0
        except Exception as e:
            logger.error(f"Error checking access: {e}")
            return False

    def get_accessible_modules(
        self, user_groups: List[str]
    ) -> Dict[str, Dict[str, str]]:
        """
        Get all modules and submodules accessible to the user with their access levels.

        Args:
            user_groups: List of groups the user belongs to

        Returns:
            Dictionary with module names as keys and dicts of submodules with access levels as values
            Example: {"protein_studies": {"settings": "full", "prediction": "view"}}
        """
        accessible = {}

        for module_name in MODULES:
            # Check module access (view level minimum)
            if self.check_user_module_access(user_groups, module_name, None, "view"):
                accessible[module_name] = {}

                # Determine module access level
                module_access_level = (
                    "full"
                    if self.check_user_module_access(
                        user_groups, module_name, None, "full"
                    )
                    else "view"
                )
                accessible[module_name]["_module_access"] = module_access_level

                # Check each submodule
                for submodule in MODULES[module_name].submodules:
                    if self.check_user_module_access(
                        user_groups, module_name, submodule, "view"
                    ):
                        # Determine submodule access level
                        submodule_access_level = (
                            "full"
                            if self.check_user_module_access(
                                user_groups, module_name, submodule, "full"
                            )
                            else "view"
                        )
                        accessible[module_name][submodule] = submodule_access_level

        return accessible

    def get_permissions_for_group(self, group_name: str) -> List[Dict]:
        """Get all permissions for a specific group."""
        group = self._sanitize_sql_string(group_name)

        query_sql = f"""
        SELECT module_name, submodule_name, permission_type, user_type, access_level, created_at, updated_at
        FROM {self.catalog_name}.{self.schema_name}.{self.table_name}
        WHERE array_contains(groups, '{group}') AND is_active = true
        ORDER BY module_name, submodule_name
        """

        try:
            result = self._execute_sql(query_sql)
            logger.info(f"Found {len(result)} permissions for group: {group_name}")
            return result
        except Exception as e:
            logger.error(f"Error getting permissions for group: {e}")
            raise

    def update_access_level(
        self,
        module_name: str,
        group_name: str,
        new_access_level: str,
        submodule_name: Optional[str] = None,
    ) -> None:
        """Update the access level for a group's permission."""
        if new_access_level not in ACCESS_LEVELS:
            raise ValueError(f"Invalid access_level: {new_access_level}")

        module = self._sanitize_sql_string(module_name)
        group = self._sanitize_sql_string(group_name)
        access_level = self._sanitize_sql_string(new_access_level)
        submodule = (
            self._sanitize_sql_string(submodule_name) if submodule_name else None
        )

        submodule_condition = (
            f"AND submodule_name = '{submodule}'"
            if submodule_name
            else "AND submodule_name IS NULL"
        )

        update_sql = f"""
        UPDATE {self.catalog_name}.{self.schema_name}.{self.table_name}
        SET access_level = '{access_level}',
            updated_at = current_timestamp(),
            updated_by = current_user()
        WHERE module_name = '{module}'
          {submodule_condition}
          AND array_contains(groups, '{group}')
          AND is_active = true
        """

        try:
            self._execute_sql(update_sql, fetch_results=False)
            target = (
                f"{module_name}.{submodule_name}" if submodule_name else module_name
            )
            logger.info(
                f"Successfully updated access level to {new_access_level} for {target} and group: {group_name}"
            )
        except Exception as e:
            logger.error(f"Error updating access level: {e}")
            raise

    def revoke_access(
        self,
        module_name: str,
        group_name: str,
        submodule_name: Optional[str] = None,
    ) -> None:
        """Revoke access for a group to a module or submodule."""
        module = self._sanitize_sql_string(module_name)
        group = self._sanitize_sql_string(group_name)
        submodule = (
            self._sanitize_sql_string(submodule_name) if submodule_name else None
        )

        submodule_condition = (
            f"AND submodule_name = '{submodule}'"
            if submodule_name
            else "AND submodule_name IS NULL"
        )

        update_sql = f"""
        UPDATE {self.catalog_name}.{self.schema_name}.{self.table_name}
        SET groups = array_remove(groups, '{group}'),
            updated_at = current_timestamp(),
            updated_by = current_user()
        WHERE module_name = '{module}'
          {submodule_condition}
          AND is_active = true
        """

        try:
            self._execute_sql(update_sql, fetch_results=False)
            target = (
                f"{module_name}.{submodule_name}" if submodule_name else module_name
            )
            logger.info(
                f"Successfully revoked access to {target} for group: {group_name}"
            )
        except Exception as e:
            logger.error(f"Error revoking access: {e}")
            raise

    def setup_admin_permissions(self, admin_user: str) -> None:
        """
        Set up initial admin permissions for all modules and submodules.

        Args:
            admin_user: Username or service principal to grant admin access
        """
        logger.info(f"Setting up admin permissions for user: {admin_user}")

        # Create a group for this admin user if needed
        admin_group = f"genesis-admin-{admin_user.replace('@', '-').replace('.', '-')}"

        for module_name, module_config in MODULES.items():
            try:
                # Grant module access (admins always get full access)
                self.grant_module_access(
                    module_name=module_name,
                    groups=[admin_group],
                    user_type="admin",
                    access_level="full",  # Will be enforced in grant_module_access
                )

                # Grant access to all submodules
                for submodule in module_config.submodules:
                    self.grant_module_access(
                        module_name=module_name,
                        groups=[admin_group],
                        user_type="admin",
                        access_level="full",
                        submodule_name=submodule,
                    )

            except Exception as e:
                logger.warning(f"Permission may already exist for {module_name}: {e}")

        logger.info("Admin permissions setup completed!")


# Standalone functions for backwards compatibility and ease of use
def create_permissions_manager(**kwargs) -> AppPermissionsManager:
    """Create an AppPermissionsManager instance."""
    return AppPermissionsManager(**kwargs)


def get_user_accessible_modules(
    user_groups: List[str], **kwargs
) -> Dict[str, Dict[str, str]]:
    """Get accessible modules for user groups with access levels."""
    manager = AppPermissionsManager(**kwargs)
    return manager.get_accessible_modules(user_groups)


def check_user_access(
    user_groups: List[str],
    module_name: str,
    submodule_name: Optional[str] = None,
    required_access_level: str = "view",
    **kwargs,
) -> bool:
    """Check if user has access to module/submodule at required level."""
    manager = AppPermissionsManager(**kwargs)
    return manager.check_user_module_access(
        user_groups, module_name, submodule_name, required_access_level
    )
