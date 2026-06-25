#!/usr/bin/env python
"""
Extract molecular embeddings from pretrained GROVER/KERMT model checkpoints.

This script loads any pretrained checkpoint (vocab, CMIM, or hybrid mode) and extracts
molecular embeddings from the shared KERMT encoder. It works with all model variants
since they all share the same encoder architecture.

================================================================================
Embedding Types:
================================================================================
The KERMT encoder produces four types of embeddings (dual-view architecture):
- atom_from_atom: Atom embeddings from the atom-centric message passing
- atom_from_bond: Atom embeddings from the bond-centric message passing
- bond_from_atom: Bond embeddings from the atom-centric message passing
- bond_from_bond: Bond embeddings from the bond-centric message passing

These are aggregated to molecule-level using mean pooling (no learnable parameters).

================================================================================
Output Formats:
================================================================================
- NPY: NumPy array files (one per embedding type, efficient for large datasets)
- PKL: Single pickle file with all embeddings and metadata

================================================================================
Usage:
================================================================================
    # Extract embeddings from any checkpoint (auto-detects model type)
    python task/extract_embeddings.py \
        -c model.pt \
        -i molecules.csv \
        -o embeddings_dir/

    # Output as pickle (single file with all embeddings)
    python task/extract_embeddings.py \
        -c model.pt \
        -i molecules.csv \
        -o embeddings.pkl \
        --format pkl

    # With batch processing for large datasets
    python task/extract_embeddings.py \
        -c model.pt \
        -i molecules.csv \
        -o embeddings_dir/ \
        --batch_size 128
"""

import argparse
import csv
import pickle
import time
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.CRITICAL)

from kermt.data.molgraph import mol2graph
from kermt.model.models import KERMTEmbedding
from kermt.model.layers import Readout
from collections import OrderedDict
from torch import nn


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize SMILES using RDKit. Returns None if invalid."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def load_encoder_from_checkpoint(
    checkpoint_path: str,
    device: str = "cuda"
) -> Tuple[Any, Readout, argparse.Namespace]:
    """
    Load KERMT encoder from any checkpoint (vocab, CMIM, hybrid, or grover_base).
    
    This is a minimal loader that only builds KERMTEmbedding (encoder),
    not the full finetune model. This avoids needing all finetune-specific args.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on

    Returns:
        Tuple of (encoder, readout, args)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = state["args"]
    loaded_state_dict = state["state_dict"]

    print(f"Checkpoint from epoch {state.get('epoch', 'unknown')}, "
          f"step {state.get('scheduler_step', 'unknown')}")
    
    # === Transform parameter names for compatibility (same as utils.py) ===
    # 1. Replace old "grover" naming with "kermt"
    loaded_state_dict = OrderedDict([
        (k.replace("grover", "kermt"), v) for k, v in loaded_state_dict.items()
    ])
    
    # 2. Handle CMIM checkpoint format: strip "latent_dist." prefix
    has_cmim_params = any("latent_dist." in k for k in loaded_state_dict.keys())
    if has_cmim_params:
        print("Detected CMIM checkpoint format. Transforming 'latent_dist.kermt.*' -> 'kermt.*'")
    loaded_state_dict = OrderedDict([
        (k.replace("latent_dist.", ""), v) for k, v in loaded_state_dict.items()
    ])
    
    # === Set default args for KERMTEmbedding (if missing from old checkpoints) ===
    # These are the args required by KERMTEmbedding.__init__ and mol2graph
    encoder_defaults = {
        # Core model architecture
        'hidden_size': 800,
        'depth': 6,
        'dropout': 0.0,
        'activation': 'PReLU',
        'bias': False,
        'undirected': False,
        'dense': False,
        'atom_message': False,
        
        # Transformer / attention args
        'backbone': 'gtrans',
        'num_mt_block': 1,
        'num_attn_head': 4,
        'embedding_output_type': 'both',  # Need all 4 embeddings
        
        # Other
        'cuda': device == "cuda",
        'nencoders': 3,
        'coord': 15,
        'input_layer': 'fc',
        'no_attach_fea': False,
        'self_attention': False,
        'attn_hidden': 4,
        'attn_out': 8,
        
        # Featurization (used by mol2graph)
        'use_cuikmolmaker_featurization': False,  # Not used in launch scripts
        'bond_drop_rate': 0.0,
        'no_cache': True,
    }
    
    for key, value in encoder_defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
            print(f"  Set default: {key} = {value}")
    
    # Force embedding_output_type to 'both' for full embedding extraction
    args.embedding_output_type = 'both'
    args.cuda = device == "cuda"
    
    # === Build encoder ===
    encoder = KERMTEmbedding(args)
    
    # === Extract encoder weights from checkpoint ===
    # Encoder is stored under "kermt." prefix after transformations
    encoder_state_dict = {}
    for key, value in loaded_state_dict.items():
        if key.startswith("kermt."):
            encoder_key = key[len("kermt."):]
            encoder_state_dict[encoder_key] = value
    
    if not encoder_state_dict:
        raise ValueError(
            f"Could not find encoder weights (kermt.*) in checkpoint. "
            f"Available keys: {list(loaded_state_dict.keys())[:10]}..."
        )
    
    print(f"Found {len(encoder_state_dict)} encoder parameters")
    
    # === Load weights with flexible matching (skip mismatched) ===
    model_state_dict = encoder.state_dict()
    pretrained_state_dict = {}
    skipped = []
    
    for param_name, param_value in encoder_state_dict.items():
        if param_name not in model_state_dict:
            skipped.append(f"'{param_name}' (not in model)")
        elif model_state_dict[param_name].shape != param_value.shape:
            skipped.append(f"'{param_name}' (shape mismatch)")
        else:
            pretrained_state_dict[param_name] = param_value
    
    if skipped:
        print(f"Skipped {len(skipped)} incompatible parameters: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
    
    model_state_dict.update(pretrained_state_dict)
    encoder.load_state_dict(model_state_dict)
    print(f"Loaded {len(pretrained_state_dict)}/{len(encoder_state_dict)} encoder parameters")
    
    if device == "cuda":
        encoder = encoder.cuda()
    encoder.eval()
    
    # === Create readout layer (mean pooling, stateless) ===
    hidden_size = args.hidden_size
    readout = Readout(rtype="mean", hidden_size=hidden_size)
    if device == "cuda":
        readout = readout.cuda()

    print(f"Encoder loaded successfully. Hidden size: {hidden_size}")

    return encoder, readout, args


class ProjectionExtractor(nn.Module):
    """Replicates KERMTLatentDistribution's readout + fc_mean_logscale projection.

    Used to extract the cMIM "real latent" — the 512-d mean vector that the
    contrastive loss operates on — from a trained cMIM or Hybrid checkpoint.

    Forward: takes the encoder's 4 atom/bond-level outputs and the (a_scope,
    b_scope) tuples, applies the trained readout to each, averages them, and
    runs the trained Linear(hidden, 2*latent_dim) projection. Returns the mean
    half (first latent_dim columns).

    grover_base / KermtTask checkpoints have no projection layer; building this
    extractor against such a checkpoint raises a ValueError.
    """

    def __init__(self, hidden_size: int, latent_dim: int, readout: Readout) -> None:
        super().__init__()
        self.readout = readout
        self.fc_mean_logscale = nn.Linear(hidden_size, 2 * latent_dim)
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size

    def forward(self, encoder_output: Dict, a_scope, b_scope) -> torch.Tensor:
        readouts = []
        scope_for = {
            "atom_from_atom": a_scope, "atom_from_bond": a_scope,
            "bond_from_atom": b_scope, "bond_from_bond": b_scope,
        }
        for etype in ("atom_from_atom", "atom_from_bond", "bond_from_atom", "bond_from_bond"):
            emb = encoder_output.get(etype)
            if emb is not None:
                readouts.append(self.readout(emb, scope_for[etype]))
        if not readouts:
            raise RuntimeError("No encoder outputs to project")
        mol_emb = torch.stack(readouts, dim=0).mean(dim=0)  # [batch, hidden_size]
        params = self.fc_mean_logscale(mol_emb)             # [batch, 2*latent_dim]
        mean, _log_scale = params.chunk(2, dim=-1)
        return mean  # [batch, latent_dim]


def load_projection_from_checkpoint(
    checkpoint_path: str,
    model_args: argparse.Namespace,
    device: str = "cuda",
) -> ProjectionExtractor:
    """Build a ProjectionExtractor loaded with weights from a cMIM/Hybrid checkpoint.

    Raises:
        ValueError: if the checkpoint has no `latent_dist.fc_mean_logscale.*`
            keys (i.e. it's a grover_base / KermtTask checkpoint without a
            projection layer).
    """
    print(f"Loading projection layer from: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_state_dict = state["state_dict"]

    # Mirror the encoder loader's "grover" -> "kermt" rename so older
    # checkpoints' projection keys are reachable too. We do NOT strip the
    # "latent_dist." prefix here — we need it to identify projection keys.
    raw_state_dict = OrderedDict([
        (k.replace("grover", "kermt"), v) for k, v in raw_state_dict.items()
    ])

    proj_prefix = "latent_dist.fc_mean_logscale."
    proj_keys = [k for k in raw_state_dict if k.startswith(proj_prefix)]
    if not proj_keys:
        raise ValueError(
            "Checkpoint contains no projection layer "
            "(no 'latent_dist.fc_mean_logscale.*' keys). "
            "This is expected for grover_base / KermtTask checkpoints, which "
            "have no fc_mean_logscale projection — only cMIM and Hybrid "
            "checkpoints expose a projected latent. "
            "Drop the --projection flag for KERMT-baseline extractions."
        )

    weight = raw_state_dict[f"{proj_prefix}weight"]
    bias = raw_state_dict[f"{proj_prefix}bias"]
    out_dim, hidden_size = weight.shape
    if out_dim % 2 != 0:
        raise ValueError(
            f"Unexpected projection output dim {out_dim} (must be 2*latent_dim)"
        )
    latent_dim = out_dim // 2
    print(f"  Found projection: Linear({hidden_size} -> {out_dim}); latent_dim={latent_dim}")

    # Detect readout type: self-attention if the checkpoint contains
    # latent_dist.readout.attn.* weights, otherwise mean (stateless).
    readout_attn_keys = [k for k in raw_state_dict if k.startswith("latent_dist.readout.attn.")]
    if readout_attn_keys:
        attn_hidden = getattr(model_args, "attn_hidden", 4)
        attn_out = getattr(model_args, "attn_out", 8)
        print(
            "  Detected self_attention readout in checkpoint "
            f"(attn_hidden={attn_hidden}, attn_out={attn_out})"
        )
        readout = Readout(
            rtype="self_attention", hidden_size=hidden_size,
            attn_hidden=attn_hidden, attn_out=attn_out,
        )
    else:
        print("  Using mean readout (no self-attention weights in checkpoint)")
        readout = Readout(rtype="mean", hidden_size=hidden_size)

    extractor = ProjectionExtractor(
        hidden_size=hidden_size, latent_dim=latent_dim, readout=readout,
    )

    # Load weights into the extractor. The keys we care about all live under
    # `latent_dist.` in the checkpoint; strip that prefix to match the
    # extractor's own attribute names ("fc_mean_logscale.*", "readout.*").
    load_dict = {}
    for k, v in raw_state_dict.items():
        if k.startswith("latent_dist."):
            inner = k[len("latent_dist."):]
            # Skip `latent_dist.kermt.*` — that's the encoder, loaded separately.
            if inner.startswith("kermt."):
                continue
            load_dict[inner] = v

    result = extractor.load_state_dict(load_dict, strict=False)
    if result.missing_keys:
        # Mean readout has only a non-trainable cached_zero_vector buffer;
        # absent in the checkpoint dict is fine.
        non_trivial_missing = [
            k for k in result.missing_keys if not k.endswith("cached_zero_vector")
        ]
        if non_trivial_missing:
            print(f"  Warning: missing projection-layer keys: {non_trivial_missing}")
    if result.unexpected_keys:
        print(f"  Warning: unexpected keys ignored: {result.unexpected_keys}")

    extractor.eval()
    if device == "cuda":
        extractor = extractor.cuda()
    return extractor


def smiles_to_graph_batch(smiles_list: List[str], args: argparse.Namespace) -> Tuple:
    """
    Convert SMILES strings to graph batch format for the encoder.

    Args:
        smiles_list: List of SMILES strings
        args: Model arguments

    Returns:
        Graph batch tuple
    """
    shared_dict = {}
    batch_graph = mol2graph(smiles_list, shared_dict, args)
    return batch_graph.get_components()


def extract_embeddings_batch(
    encoder: torch.nn.Module,
    readout: Readout,
    smiles_batch: List[str],
    args: argparse.Namespace,
    device: str = "cuda",
    projection_extractor: Optional[ProjectionExtractor] = None,
) -> Tuple[Dict[str, np.ndarray], List[bool]]:
    """
    Extract all four molecular embeddings for a batch of SMILES.

    Args:
        encoder: The KERMT encoder
        readout: Readout layer (mean pooling, stateless)
        smiles_batch: List of SMILES strings
        args: Model arguments
        device: Device for computation
        projection_extractor: Optional ProjectionExtractor; when supplied, the
            returned dict additionally contains a "projected" key with the
            cMIM/Hybrid 512-d projected latent per molecule.

    Returns:
        Tuple of (embeddings dict with 4 arrays — plus "projected" when a
        projection extractor is provided —, validity mask)
    """
    # Filter valid SMILES
    valid_smiles = []
    valid_indices = []
    
    for i, smi in enumerate(smiles_batch):
        can_smi = canonicalize_smiles(smi)
        if can_smi is not None:
            valid_smiles.append(can_smi)
            valid_indices.append(i)
    
    batch_size = len(smiles_batch)
    hidden_size = args.hidden_size

    # Initialize output arrays for all four embedding types
    embedding_types = ["atom_from_atom", "atom_from_bond", "bond_from_atom", "bond_from_bond"]
    embeddings = {
        etype: np.zeros((batch_size, hidden_size), dtype=np.float32)
        for etype in embedding_types
    }
    if projection_extractor is not None:
        embeddings["projected"] = np.zeros(
            (batch_size, projection_extractor.latent_dim), dtype=np.float32
        )
    validity = [False] * batch_size
    
    if not valid_smiles:
        return embeddings, validity
    
    # Convert to graph batch
    graph_batch = smiles_to_graph_batch(valid_smiles, args)
    
    # Extract scope information for readout
    # graph_batch = (f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a)
    f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = graph_batch
    
    # Move tensors to device
    graph_batch_device = tuple(
        t.to(device) if isinstance(t, torch.Tensor) else t for t in graph_batch
    )
    
    with torch.no_grad():
        # Get embeddings from encoder
        encoder_output = encoder(graph_batch_device)
        
        # Extract molecule-level embeddings using mean readout
        # atom_from_atom and atom_from_bond use a_scope
        # bond_from_atom and bond_from_bond use b_scope
        # Note: same readout instance works for all since mean pooling is stateless
        
        mol_embeddings = {}
        
        if encoder_output["atom_from_atom"] is not None:
            mol_embeddings["atom_from_atom"] = readout(
                encoder_output["atom_from_atom"], a_scope
            )
        
        if encoder_output["atom_from_bond"] is not None:
            mol_embeddings["atom_from_bond"] = readout(
                encoder_output["atom_from_bond"], a_scope
            )
        
        if encoder_output["bond_from_atom"] is not None:
            mol_embeddings["bond_from_atom"] = readout(
                encoder_output["bond_from_atom"], b_scope
            )
        
        if encoder_output["bond_from_bond"] is not None:
            mol_embeddings["bond_from_bond"] = readout(
                encoder_output["bond_from_bond"], b_scope
            )

        # Optional: run the trained projection layer (cMIM/Hybrid only) on the
        # SAME encoder forward pass and accumulate into mol_embeddings["projected"].
        if projection_extractor is not None:
            projected = projection_extractor(encoder_output, a_scope, b_scope)
            mol_embeddings["projected"] = projected

        # Place embeddings at correct indices
        output_keys = list(embedding_types)
        if projection_extractor is not None:
            output_keys.append("projected")
        for batch_idx, orig_idx in enumerate(valid_indices):
            validity[orig_idx] = True
            for etype in output_keys:
                if etype in mol_embeddings and mol_embeddings[etype] is not None:
                    embeddings[etype][orig_idx] = mol_embeddings[etype][batch_idx].cpu().numpy()

    return embeddings, validity


def extract_all_embeddings(
    encoder: torch.nn.Module,
    readout: Readout,
    smiles_list: List[str],
    args: argparse.Namespace,
    batch_size: int = 64,
    device: str = "cuda",
    show_progress: bool = True,
    projection_extractor: Optional[ProjectionExtractor] = None,
) -> Tuple[Dict[str, np.ndarray], List[str], List[bool]]:
    """
    Extract all four embeddings for all SMILES in a list.

    Args:
        encoder: The KERMT encoder
        readout: Readout layer (mean pooling, stateless)
        smiles_list: List of SMILES strings
        args: Model arguments
        batch_size: Batch size for processing
        device: Device for computation
        show_progress: Show progress bar
        projection_extractor: Optional. When provided, the returned embeddings
            dict additionally contains a "projected" array of shape
            [N, latent_dim] with the trained cMIM/Hybrid projection.

    Returns:
        Tuple of (embeddings dict [N, hidden_size] x 4 (plus "projected"
                  [N, latent_dim] when applicable),
                  canonical SMILES list, validity list)
    """
    n_molecules = len(smiles_list)
    hidden_size = args.hidden_size

    embedding_types = ["atom_from_atom", "atom_from_bond", "bond_from_atom", "bond_from_bond"]
    all_embeddings = {
        etype: np.zeros((n_molecules, hidden_size), dtype=np.float32)
        for etype in embedding_types
    }
    if projection_extractor is not None:
        all_embeddings["projected"] = np.zeros(
            (n_molecules, projection_extractor.latent_dim), dtype=np.float32
        )
    all_validity = [False] * n_molecules
    all_canonical = [None] * n_molecules

    # Pre-canonicalize to store canonical SMILES
    for i, smi in enumerate(smiles_list):
        all_canonical[i] = canonicalize_smiles(smi)

    # Process in batches
    n_batches = (n_molecules + batch_size - 1) // batch_size
    iterator = range(0, n_molecules, batch_size)

    if show_progress:
        iterator = tqdm(iterator, total=n_batches, desc="Extracting embeddings")

    output_keys = list(embedding_types)
    if projection_extractor is not None:
        output_keys.append("projected")

    for start_idx in iterator:
        end_idx = min(start_idx + batch_size, n_molecules)
        batch_smiles = smiles_list[start_idx:end_idx]

        batch_embeddings, batch_validity = extract_embeddings_batch(
            encoder=encoder,
            readout=readout,
            smiles_batch=batch_smiles,
            args=args,
            device=device,
            projection_extractor=projection_extractor,
        )

        for etype in output_keys:
            all_embeddings[etype][start_idx:end_idx] = batch_embeddings[etype]
        all_validity[start_idx:end_idx] = batch_validity

    return all_embeddings, all_canonical, all_validity


def save_embeddings(
    output_path: str,
    embeddings: Dict[str, np.ndarray],
    smiles_list: List[str],
    canonical_list: List[str],
    validity_list: List[bool],
    format: str = "npy",
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Save embeddings to file(s).

    Args:
        output_path: Output path (directory for npy, file for pkl)
        embeddings: Dict of embedding arrays {type: [N, hidden_size]}
        smiles_list: Original SMILES list
        canonical_list: Canonical SMILES list
        validity_list: Validity list
        format: Output format ('npy' or 'pkl')
        metadata: Optional metadata dictionary
    """
    n_molecules = len(smiles_list)
    hidden_size = list(embeddings.values())[0].shape[1]
    embedding_types = list(embeddings.keys())
    
    if format == "npy":
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each embedding type as separate .npy file
        for etype, emb_array in embeddings.items():
            np.save(output_dir / f"{etype}.npy", emb_array)
            print(f"Saved {etype} embeddings to {output_dir / f'{etype}.npy'}")
        
        # Save metadata
        meta = {
            "smiles": smiles_list,
            "canonical_smiles": canonical_list,
            "valid": validity_list,
            "n_molecules": n_molecules,
            "hidden_size": hidden_size,
            "embedding_types": embedding_types,
            **(metadata or {}),
        }
        meta_path = output_dir / "metadata.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)
        print(f"Saved metadata to {meta_path}")
    
    elif format == "pkl":
        # Save everything in a single pickle file
        data = {
            "embeddings": embeddings,
            "smiles": smiles_list,
            "canonical_smiles": canonical_list,
            "valid": validity_list,
            "n_molecules": n_molecules,
            "hidden_size": hidden_size,
            "embedding_types": embedding_types,
            **(metadata or {}),
        }
        with open(output_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved all embeddings to {output_path}")
    
    else:
        raise ValueError(f"Unknown format: {format}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract molecular embeddings from pretrained GROVER/KERMT checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to model checkpoint (works with vocab, CMIM, or hybrid)",
    )

    # Input
    parser.add_argument(
        "--input_file", "-i",
        type=str,
        required=True,
        help="Input CSV file with SMILES (first column)",
    )
    parser.add_argument(
        "--smiles_column",
        type=int,
        default=0,
        help="Column index containing SMILES (0-indexed)",
    )

    # Output
    parser.add_argument(
        "--output_path", "-o",
        type=str,
        required=True,
        help="Output path (directory for npy format, file for pkl format)",
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["npy", "pkl"],
        default="npy",
        help="Output format: 'npy' saves 4 separate files, 'pkl' saves single file",
    )

    # Processing options
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=64,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for computation",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable progress bar",
    )
    parser.add_argument(
        "--projection",
        action="store_true",
        help=(
            "Also extract the cMIM/Hybrid projected latent (the 512-d 'real' "
            "cMIM latent from fc_mean_logscale, downstream of the encoder). "
            "Saves an additional 'projected.npy' alongside the 4 readout files. "
            "Errors with a clear message if the checkpoint has no projection "
            "layer (e.g. grover_base / KermtTask)."
        ),
    )
    parser.add_argument(
        "--projection_only",
        action="store_true",
        help=(
            "Implies --projection. When set, skip writing the 4 encoder readout "
            "files and save only 'projected.npy'. Useful when re-extracting just "
            "the projection layer for a checkpoint whose readouts already exist."
        ),
    )

    args = parser.parse_args()
    if args.projection_only:
        args.projection = True

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # Load encoder from checkpoint
    encoder, readout, model_args = load_encoder_from_checkpoint(
        args.checkpoint, device=args.device
    )

    # Optional: load projection layer (cMIM / Hybrid only). Errors clearly when
    # the checkpoint has no fc_mean_logscale (e.g., a grover_base checkpoint).
    projection_extractor = None
    if args.projection:
        projection_extractor = load_projection_from_checkpoint(
            args.checkpoint, model_args, device=args.device
        )

    # Read input SMILES
    smiles_list = []
    with open(args.input_file, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header
        for row in reader:
            if row and len(row) > args.smiles_column:
                smiles_list.append(row[args.smiles_column].strip())

    print(f"Loaded {len(smiles_list)} SMILES from {args.input_file}")

    # Extract embeddings
    start_time = time.time()

    embeddings, canonical_list, validity_list = extract_all_embeddings(
        encoder=encoder,
        readout=readout,
        smiles_list=smiles_list,
        args=model_args,
        batch_size=args.batch_size,
        device=args.device,
        show_progress=not args.no_progress,
        projection_extractor=projection_extractor,
    )

    # When --projection_only, drop the 4 encoder readouts before save so we
    # only emit projected.npy (and metadata).
    if args.projection_only:
        if "projected" not in embeddings:
            raise RuntimeError("--projection_only set but no projected output produced")
        embeddings = {"projected": embeddings["projected"]}

    elapsed = time.time() - start_time

    # Print summary
    n_valid = sum(validity_list)
    n_invalid = len(smiles_list) - n_valid
    print(f"\nExtraction complete in {elapsed:.1f}s")
    print(f"  Valid molecules:   {n_valid} ({100 * n_valid / len(smiles_list):.1f}%)")
    print(f"  Invalid molecules: {n_invalid} ({100 * n_invalid / len(smiles_list):.1f}%)")
    print("  Embedding shapes:")
    for etype, emb in embeddings.items():
        print(f"    {etype}: {emb.shape}")

    # Save
    metadata = {
        "checkpoint": args.checkpoint,
        "embedding_output_type": model_args.embedding_output_type,
    }
    
    save_embeddings(
        output_path=args.output_path,
        embeddings=embeddings,
        smiles_list=smiles_list,
        canonical_list=canonical_list,
        validity_list=validity_list,
        format=args.format,
        metadata=metadata,
    )


if __name__ == "__main__":
    main()
