"""
Configuration file for the Genesis Workbench app permissions system.
Defines modules and submodules that users can access in the web application.
"""

from typing import List, Dict
from dataclasses import dataclass


@dataclass
class ModuleConfig:
    """Configuration for a Genesis Workbench module."""

    name: str
    display_name: str
    description: str
    submodules: List[str]


# Define all available modules and their submodules
MODULES = {
    "protein_studies": ModuleConfig(
        name="protein_studies",
        display_name="Protein Studies",
        description="Protein folding and analysis workflows",
        submodules=["settings", "protein_structure_prediction", "protein_design"],
    ),
    "nvidia_bionemo": ModuleConfig(
        name="nvidia_bionemo",
        display_name="NVIDIA BioNeMo",
        description="NVIDIA BioNeMo model workflows",
        submodules=["settings", "esm2_finetune", "esm2_inference"],
    ),
    "single_cell": ModuleConfig(
        name="single_cell",
        display_name="Single Cell Analysis",
        description="Single cell analysis workflows",
        submodules=["settings", "embeddings"],
    ),
    "monitoring_alerts": ModuleConfig(
        name="monitoring_alerts",
        display_name="Monitoring & Alerts",
        description="Monitoring and alerting workflows",
        submodules=["workflows", "dashboard", "alerts"],
    ),
    "master_settings": ModuleConfig(
        name="master_settings",
        display_name="Master Settings",
        description="Administrative settings and configuration",
        submodules=["settings"],
    ),
}

# Permission types - simplified for app usage
PERMISSION_TYPES = {
    "module_access": "Access to view and use a module",
    "submodule_access": "Access to specific submodule functionality",
}

# Access levels for granular permissions
ACCESS_LEVELS = {
    "view": "Read-only access - can view but not modify",
    "full": "Full access - can view and modify",
}

# User types
USER_TYPES = {
    "admin": "Administrative users with full access",
    "user": "Regular users with module-specific access",
}

# Default groups
# TODO: Pull these from variables.yml
DEFAULT_GROUPS = {
    "admin": ["genesis-admin-group"],
    "user": ["genesis-users"],
}

DEFAULT_CATALOG = "genesis_workbench"  # TODO: Can this be parameterized from the DAB deploy? I don't think so, probably need to make sure we're instead building a workflow resource for the setup notebook and configuring the job to parameterize widgets in the notebook.
DEFAULT_SCHEMA = "permissions"  # TODO: Can this be parameterized from the DAB deploy? I don't think so, probably need to make sure we're instead building a workflow resource for the setup notebook and configuring the job to parameterize widgets in the notebook.
PERMISSIONS_TABLE_NAME = "app_permissions"
PERMISSIONS_TABLE_COMMENT = (
    "Application permissions management for Genesis Workbench modules and submodules"
)

DELTA_TABLE_PROPERTIES = {
    "delta.autoOptimize.optimizeWrite": "true",
    "delta.autoOptimize.autoCompact": "true",
    "delta.feature.allowColumnDefaults": "supported",
}

DATABRICKS_API_VERSION = "2.0"
DATABRICKS_SCIM_API_VERSION = "2.0"
