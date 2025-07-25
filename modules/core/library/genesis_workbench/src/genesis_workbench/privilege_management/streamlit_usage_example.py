"""
Streamlit Usage Example: App Permissions Manager for Genesis Workbench

This example shows how to integrate the app permissions system with Streamlit
to control access to different modules and submodules based on user groups and access levels.
"""

try:
    import streamlit as st
except ImportError:
    # Streamlit not available in Databricks environment
    # This file is for reference only when deploying to external Streamlit environments
    raise ImportError(
        "Streamlit is not available in Databricks environment. "
        "This file is for external Streamlit deployments only."
    )

import os
from typing import List, Dict, Optional
from permissions_manager_app import (
    AppPermissionsManager,
    create_permissions_manager,
    get_user_accessible_modules,
    check_user_access,
)
from permissions_config import MODULES, ACCESS_LEVELS


def initialize_permissions_manager() -> AppPermissionsManager:
    """Initialize the permissions manager with environment variables."""
    return AppPermissionsManager(
        server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_TOKEN"),
    )


def get_current_user_groups(permissions_manager: AppPermissionsManager) -> List[str]:
    """Get the current user's groups from Databricks Identity API."""
    try:
        return permissions_manager.get_current_user_groups()
    except Exception as e:
        st.error(f"Error fetching user groups: {e}")
        return []


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_accessible_modules(user_groups: List[str]) -> Dict[str, Dict[str, str]]:
    """Get accessible modules with access levels, cached for better performance."""
    if not user_groups:
        return {}

    try:
        permissions_manager = initialize_permissions_manager()
        return permissions_manager.get_accessible_modules(user_groups)
    except Exception as e:
        st.error(f"Error getting accessible modules: {e}")
        return {}


def check_module_access(
    user_groups: List[str],
    module_name: str,
    submodule_name: Optional[str] = None,
    required_access_level: str = "view",
) -> bool:
    """Check if user has access to a specific module or submodule at the required level."""
    try:
        permissions_manager = initialize_permissions_manager()
        return permissions_manager.check_user_module_access(
            user_groups, module_name, submodule_name, required_access_level
        )
    except Exception as e:
        st.error(f"Error checking module access: {e}")
        return False


if __name__ == "__main__":
    ### main app runs

    # Add admin interface in sidebar if user has access
    try:
        permissions_manager = initialize_permissions_manager()
        user_groups = get_current_user_groups(permissions_manager)

        if check_module_access(user_groups, "master_settings"):
            ### Ipmlement permissions interface here
                
    except:
        pass  # Silently fail if permissions check fails
