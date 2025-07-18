"""
Usage Example: App-Compatible Permissions Manager for Databricks Apps

This example shows how to use the permissions manager from a Databricks app
that connects to a serverless SQL warehouse instead of using Spark.
"""

import os
from permissions_manager_app import (
    PermissionsManagerApp,
    create_permissions_manager,
    get_permissions_by_group_standalone,
    check_user_permission_standalone,
    add_group_to_permission_standalone,
)
import streamlit as st


def streamlit_integration_example():
    """
    Example showing how to integrate with Streamlit or similar app frameworks.
    """
    print("\n=== Example 6: Streamlit integration pattern ===")

    def get_cached_permissions(group_name):
        """Get permissions with caching for better app performance."""
        return get_permissions_by_group_standalone(group_name)

    def check_user_access(user_groups, workflow, resource, privilege="SELECT"):
        """Check if current user has access to a resource."""
        return check_user_permission_standalone(
            workflow_type=workflow,
            resource=resource,
            user_groups=user_groups,
            required_privilege=privilege,
        )

    print("Integration functions defined for Streamlit app")


# Example 7: Configuration management
def example_configuration():
    """
    Example showing how to manage configuration for different environments.
    """
    print("\n=== Example 7: Configuration management ===")

    # Environment-specific configuration
    config = {
        "dev": {
            "catalog_name": "genesis_workbench_dev",
            "schema_name": "permissions",
        },
        "staging": {
            "catalog_name": "genesis_workbench_staging",
            "schema_name": "permissions",
        },
        "prod": {
            "catalog_name": "genesis_workbench",
            "schema_name": "permissions",
        },
    }

    # Get environment from env var or default to dev
    env = os.getenv("ENVIRONMENT", "dev")
    env_config = config.get(env, config["dev"])

    manager = PermissionsManagerApp(
        catalog_name=env_config["catalog_name"], schema_name=env_config["schema_name"]
    )

    print(f"Initialized manager for {env} environment")
    print(f"Using catalog: {env_config['catalog_name']}")


if __name__ == "__main__":
    """
    Run all examples. Make sure you have:
    1. Set environment variables: DATABRICKS_SERVER_HOSTNAME, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN
    2. Installed dependencies: pip install -r requirements_app.txt
    3. Created the permissions table using the original permissions_manager.py
    """

    print("Running Permissions Manager App Examples")
    print("=" * 50)

    try:
        example_with_env_vars()
        example_with_explicit_params()
        example_with_standalone_functions()
        example_with_error_handling()
        example_batch_operations()
        streamlit_integration_example()
        example_configuration()

        print("\n" + "=" * 50)
        print("All examples completed successfully!")

    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have set up your Databricks connection parameters.")
