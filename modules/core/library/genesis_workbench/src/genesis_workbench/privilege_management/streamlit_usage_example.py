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
