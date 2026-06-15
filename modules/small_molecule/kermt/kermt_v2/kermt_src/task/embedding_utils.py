#!/usr/bin/env python
"""
Shared utility functions for working with molecular embeddings.

This module provides common functionality used across multiple embedding analysis scripts:
- extract_embeddings.py
- probing_classifiers.py
- latent_visualization.py
"""

import csv
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Available embedding types from KERMT encoder (dual-view architecture)
EMBEDDING_TYPES = ["atom_from_atom", "atom_from_bond", "bond_from_atom", "bond_from_bond"]

# Optional cMIM/Hybrid projection-layer output (saved as projected.npy by
# extract_embeddings.py --projection). Treated separately from EMBEDDING_TYPES
# because (a) it has a different shape (latent_dim ~= 512, not hidden_size 800)
# and (b) it is absent for grover_base / KermtTask checkpoints.
PROJECTED_EMBEDDING_TYPE = "projected"


def load_embeddings(
    embeddings_path: str,
    embedding_type: str = "atom_from_atom"
) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    Load embeddings from file or directory.

    Supports multiple formats:
    - Directory with 4 .npy files (atom_from_atom.npy, etc.) from extract_embeddings.py
    - Single .npy file (legacy format)
    - Single .pkl file

    Args:
        embeddings_path: Path to embeddings directory or file
        embedding_type: Which embedding to use. Options:
            - "atom_from_atom", "atom_from_bond", "bond_from_atom", "bond_from_bond"
            - "concat": Concatenate all four types [N, 4*hidden_size]
            - "mean": Average all four types [N, hidden_size]

    Returns:
        Tuple of (embeddings array, metadata dict or None)
    """
    path = Path(embeddings_path)
    
    # Case 1: Directory with multiple .npy files (new format from extract_embeddings.py)
    if path.is_dir():
        # Check for the 4 encoder readout files plus optional projected.npy
        available_types = []
        for etype in EMBEDDING_TYPES:
            if (path / f"{etype}.npy").exists():
                available_types.append(etype)
        has_projected = (path / f"{PROJECTED_EMBEDDING_TYPE}.npy").exists()

        if not available_types and not has_projected:
            raise ValueError(
                f"No embedding files found in {path}. Expected at least one of "
                f"{EMBEDDING_TYPES + [PROJECTED_EMBEDDING_TYPE]}."
            )
        
        # Load metadata if available
        metadata = None
        meta_path = path / "metadata.pkl"
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)
        
        # Load embeddings based on embedding_type
        if embedding_type in EMBEDDING_TYPES:
            if embedding_type not in available_types:
                raise ValueError(f"Embedding type '{embedding_type}' not found. Available: {available_types}")
            embeddings = np.load(path / f"{embedding_type}.npy")
            print(f"  Loaded {embedding_type} embeddings: {embeddings.shape}")

        elif embedding_type == PROJECTED_EMBEDDING_TYPE:
            proj_path = path / f"{PROJECTED_EMBEDDING_TYPE}.npy"
            if not proj_path.exists():
                raise ValueError(
                    f"'{PROJECTED_EMBEDDING_TYPE}.npy' not found in {path}. "
                    "Only cMIM and Hybrid checkpoints produce a projected latent — "
                    "re-run extract_embeddings.py with --projection on those models, "
                    "or skip this readout for grover_base / KermtTask checkpoints."
                )
            embeddings = np.load(proj_path)
            print(f"  Loaded projected embeddings: {embeddings.shape}")

        elif embedding_type == "concat":
            # Concatenate all available types
            emb_list = []
            for etype in EMBEDDING_TYPES:
                if etype in available_types:
                    emb_list.append(np.load(path / f"{etype}.npy"))
            embeddings = np.concatenate(emb_list, axis=1)
            print(f"  Concatenated {len(emb_list)} embedding types: {embeddings.shape}")
        
        elif embedding_type == "mean":
            # Average all available types
            emb_list = []
            for etype in EMBEDDING_TYPES:
                if etype in available_types:
                    emb_list.append(np.load(path / f"{etype}.npy"))
            embeddings = np.mean(emb_list, axis=0)
            print(f"  Averaged {len(emb_list)} embedding types: {embeddings.shape}")
        
        else:
            raise ValueError(
                f"Unknown embedding_type: {embedding_type}. "
                f"Options: {EMBEDDING_TYPES + ['concat', 'mean', PROJECTED_EMBEDDING_TYPE]}"
            )
        
        return embeddings, metadata
    
    # Case 2: Single .npy file (legacy format)
    elif str(path).endswith(".npy"):
        embeddings = np.load(path)
        
        # Try to load metadata
        meta_path = str(path).replace(".npy", "_meta.pkl")
        metadata = None
        if Path(meta_path).exists():
            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)
        
        return embeddings, metadata
    
    # Case 3: .pkl file
    elif str(path).endswith(".pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            # New format: dict with "embeddings" key containing dict of 4 types
            if "embeddings" in data and isinstance(data["embeddings"], dict):
                emb_dict = data["embeddings"]
                available_types = list(emb_dict.keys())
                
                if embedding_type in available_types:
                    embeddings = emb_dict[embedding_type]
                elif embedding_type == "concat":
                    emb_list = [emb_dict[et] for et in EMBEDDING_TYPES if et in emb_dict]
                    embeddings = np.concatenate(emb_list, axis=1)
                elif embedding_type == "mean":
                    emb_list = [emb_dict[et] for et in EMBEDDING_TYPES if et in emb_dict]
                    embeddings = np.mean(emb_list, axis=0)
                else:
                    raise ValueError(f"Unknown embedding_type: {embedding_type}")
                
                return embeddings, data
            
            # Legacy format: dict with "embeddings" key containing single array
            elif "embeddings" in data:
                return data["embeddings"], data
            else:
                raise ValueError(f"Could not find 'embeddings' key in pickle file")
        else:
            return data, None
    
    else:
        raise ValueError(f"Unknown file format: {embeddings_path}")


def load_properties(
    properties_path: str,
    columns: Optional[List[str]] = None,
    include_validity: bool = False
) -> Union[Tuple[List[str], Dict[str, np.ndarray]], 
           Tuple[List[str], Dict[str, np.ndarray], List[bool]]]:
    """
    Load molecular properties from CSV file.

    Args:
        properties_path: Path to properties CSV file
        columns: List of columns to load (None = all numeric columns)
        include_validity: If True, return validity list as third element

    Returns:
        If include_validity=False:
            Tuple of (SMILES list, properties dict)
        If include_validity=True:
            Tuple of (SMILES list, properties dict, validity list)
    """
    smiles_list = []
    validity_list = []
    properties = {}
    
    with open(properties_path, "r") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        
        # Determine which columns to load
        if columns is None:
            # Load all columns except smiles/valid
            skip_cols = {"smiles", "canonical_smiles", "valid"}
            columns = [h for h in headers if h not in skip_cols]
        
        # Initialize property lists
        for col in columns:
            if col in headers:
                properties[col] = []
        
        # Read data
        for row in reader:
            smiles_list.append(row.get("smiles", row.get("canonical_smiles", "")))
            validity_list.append(row.get("valid", "True").lower() == "true")
            
            for col in properties.keys():
                val = row.get(col, "")
                if val == "" or val == "None":
                    properties[col].append(np.nan)
                else:
                    try:
                        properties[col].append(float(val))
                    except ValueError:
                        properties[col].append(np.nan)
    
    # Convert to numpy arrays
    for col in properties:
        properties[col] = np.array(properties[col])
    
    if include_validity:
        return smiles_list, properties, validity_list
    return smiles_list, properties


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Canonicalize SMILES using RDKit.
    
    Args:
        smiles: Input SMILES string
    
    Returns:
        Canonical SMILES or None if invalid
    """
    from rdkit import Chem
    
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)
