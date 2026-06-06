"""Resolve a model display name to its current serving endpoint name.

Display name → UC short name mapping plus
`get_endpoint_name_for_uc_model` from the genesis_workbench lib. The display
name -> UC short name map is the canonical user-facing index of models the
React app can hit. The UC short name is looked up against the live
`model_deployments` table at call time, so deployment renames flow through
without restarting the app."""
from __future__ import annotations

from genesis_workbench.models import get_endpoint_name_for_uc_model


# Display name -> UC short name. Keep in sync with the canonical
# `_MODEL_ENDPOINT_MAP`. When new models register, add their entry here.
DISPLAY_TO_UC: dict[str, str] = {
    # Small Molecule
    "DiffDock ESM Embeddings": "diffdock_esm_embeddings",
    "DiffDock": "diffdock",
    "Proteina-Complexa Binder": "proteina_complexa",
    "Proteina-Complexa Ligand": "proteina_complexa_ligand",
    "Proteina-Complexa AME": "proteina_complexa_ame",
    "NetSolP Solubility": "netsolp_v1",
    "PLTNUM Half-Life Stability": "pltnum_v1",
    "DeepSTABp Tm": "deepstabp_v1",
    "MHCflurry Immunogenicity": "mhcflurry_v2",
    "Chemprop BBBP": "chemprop_bbbp",
    "Chemprop ClinTox": "chemprop_clintox",
    "Chemprop ADMET": "chemprop_admet",
    "GenMol Molecule Generator": "genmol",
    # Single Cell
    "SCimilarity Gene Order": "scimilarity_gene_order",
    "SCimilarity Get Embedding": "scimilarity_get_embedding",
    # Large Molecule
    "ESM2 Embeddings": "esm2_embeddings",
    "ESMFold": "esmfold",
    "ProteinMPNN": "proteinmpnn",
    "RFDiffusion": "rfdiffusion_inpainting",
    "Boltz": "boltz",
    "AlphaFold2": "alphafold2",
    # Single Cell
    "scGPT Perturbation": "scgpt_perturbation",
    "TEDDY Annotation": "teddy",
}


def get_endpoint_name(display_name: str) -> str:
    uc_name = DISPLAY_TO_UC.get(display_name)
    if uc_name is None:
        raise RuntimeError(
            f"Unknown model '{display_name}'. Add it to DISPLAY_TO_UC in services/endpoints.py"
        )
    return get_endpoint_name_for_uc_model(uc_name)
