"""
Configuration file for the Genesis Workbench permissions control system.
"""

from typing import List
from dataclasses import dataclass


@dataclass
class WorkflowConfig:
    """Configuration for a specific workflow type."""

    name: str
    description: str
    submodules: List[str]
    default_admin_privileges: List[str]
    default_user_privileges: List[str]


mapvar = {
    "single_cell": ["settings", "embeddings"],
    "protein_studies": ["settings", "protein_structure_prediction", "protein_design"],
    "nvidia_bionemo": ["settings", "esm2_finetune", "esm2_inference"],
    "monitoring_alerts": ["workflows", "dashboard", "alerts"],
    "master_settings": ["settings"],
}

WORKFLOW_TYPES = {
    "protein": WorkflowConfig(
        name="protein",
        description="Protein folding and analysis workflows",
        submodules=["settings", "protein_structure_prediction", "protein_design"],
        default_admin_privileges=["OWNER", "SELECT", "MODIFY", "EXECUTE"],
        default_user_privileges=["SELECT", "EXECUTE"],
    ),
    "bionemo": WorkflowConfig(
        name="bionemo",
        description="NVIDIA BioNeMo model workflows",
        submodules=["model", "inference_endpoint", "training_job", "output_tables"],
        default_admin_privileges=["OWNER", "SELECT", "MODIFY", "EXECUTE"],
        default_user_privileges=["SELECT", "EXECUTE"],
    ),
    "single_cell": WorkflowConfig(
        name="single_cell",
        description="Single cell analysis workflows",
        submodules=[
            "embeddings",
            "settings",
            "model",
            "inference_endpoint",
            "output_tables",
        ],
        default_admin_privileges=["OWNER", "SELECT", "MODIFY", "EXECUTE"],
        default_user_privileges=["SELECT", "EXECUTE"],
    ),
    "small_molecules": WorkflowConfig(
        name="small_molecules",
        description="Small molecule analysis workflows",
        submodules=["model", "inference_endpoint", "training_job", "output_tables"],
        default_admin_privileges=["OWNER", "SELECT", "MODIFY", "EXECUTE"],
        default_user_privileges=["SELECT", "EXECUTE"],
    ),
}

USER_TYPES = {
    "admin": "Administrative users with full access",
    "user": "Regular users with limited access",
    "service_principal": "Service principals for automated processes",
}

AVAILABLE_PRIVILEGES = {
    "OWNER": "Full ownership of the resource",
    "SELECT": "Read access to the resource",
    "MODIFY": "Write/modify access to the resource",
    "EXECUTE": "Execute access for functions/procedures",
    "USE_CATALOG": "Use catalog permission",
    "USE_SCHEMA": "Use schema permission",
    "CREATE_SCHEMA": "Create schema permission",
    "CREATE_TABLE": "Create table permission",
    "CREATE_FUNCTION": "Create function permission",
}

DEFAULT_GROUPS = {
    "admin": ["genesis-admin-group"],
    "user": ["genesis-users"],
    "service_principal": ["genesis-service-principals"],
}

DEFAULT_CATALOG = "genesis_workbench"
DEFAULT_SCHEMA = "permissions"

PERMISSIONS_TABLE_NAME = "permissions_control"
PERMISSIONS_TABLE_COMMENT = (
    "Centralized permissions management table for Genesis Workbench."
)

DELTA_TABLE_PROPERTIES = {
    "delta.autoOptimize.optimizeWrite": "true",
    "delta.autoOptimize.autoCompact": "true",
    "delta.feature.allowColumnDefaults": "supported",
    "delta.enableChangeDataFeed": "true",
}
