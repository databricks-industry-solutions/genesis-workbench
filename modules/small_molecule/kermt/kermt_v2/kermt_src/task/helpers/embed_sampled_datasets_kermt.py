#!/usr/bin/env python
"""Encode sampled dataset SMILES to embeddings using KERMT encoder.

Reads a CSV with (smiles, source) columns from sample_datasets.py,
encodes with KERMT encoder, saves pickle compatible with plot_embeddings.py.

KERMT produces 4 embedding types (atom_from_atom, atom_from_bond,
bond_from_atom, bond_from_bond), each [N, hidden_size]. For visualization,
we concatenate atom_from_atom and atom_from_bond to get a single
[N, 2*hidden_size] embedding per molecule (the same representation used
by the finetune FFN input).

Example (run inside container with KERMT environment):
    python task/helpers/embed_sampled_datasets_kermt.py \
        --input /path/to/sampled_datasets.csv \
        --output /path/to/embeddings_kermt.pkl \
        --checkpoint /models/last_checkpoint.pt \
        --batch_size 64
"""

import argparse
import csv
import os
import pickle
import sys
import time
from pathlib import Path

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch

from task.extract_embeddings import (  # noqa: E402
    load_encoder_from_checkpoint,
    extract_all_embeddings,
)


def main():
    parser = argparse.ArgumentParser(
        description="Encode sampled SMILES with KERMT encoder"
    )
    parser.add_argument("--input", required=True, help="Input CSV (smiles, source)")
    parser.add_argument("--output", required=True, help="Output pickle file")
    parser.add_argument(
        "--checkpoint", required=True, help="KERMT checkpoint path (.pt)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Encoding batch size (default: 64)"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--embedding_mode",
        default="atom_concat",
        choices=["atom_concat", "atom_from_atom", "atom_from_bond", "all_concat"],
        help="How to combine KERMT's 4 embedding types into a single vector. "
             "atom_concat: concat atom_from_atom + atom_from_bond (default, matches finetune FFN input). "
             "atom_from_atom/atom_from_bond: use a single view. "
             "all_concat: concat all 4 types.",
    )
    args = parser.parse_args()

    # Read sampled CSV
    smiles_list = []
    sources_list = []
    with open(args.input, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles_list.append(row["smiles"])
            sources_list.append(row["source"])
    print(f"Loaded {len(smiles_list)} SMILES from {args.input}")

    # Load encoder
    encoder, readout, model_args = load_encoder_from_checkpoint(
        args.checkpoint, device=args.device
    )
    hidden_size = model_args.hidden_size

    # Extract embeddings
    t0 = time.time()
    embeddings_dict, canonical_list, validity_list = extract_all_embeddings(
        encoder=encoder,
        readout=readout,
        smiles_list=smiles_list,
        args=model_args,
        batch_size=args.batch_size,
        device=args.device,
        show_progress=True,
    )
    elapsed = time.time() - t0
    n_valid = sum(validity_list)
    print(f"Encoded {n_valid}/{len(smiles_list)} valid SMILES in {elapsed:.1f}s")

    # Combine embedding types into a single vector
    if args.embedding_mode == "atom_concat":
        combined = np.concatenate(
            [embeddings_dict["atom_from_atom"], embeddings_dict["atom_from_bond"]],
            axis=1,
        )
        dim = hidden_size * 2
    elif args.embedding_mode == "all_concat":
        combined = np.concatenate(
            [
                embeddings_dict["atom_from_atom"],
                embeddings_dict["atom_from_bond"],
                embeddings_dict["bond_from_atom"],
                embeddings_dict["bond_from_bond"],
            ],
            axis=1,
        )
        dim = hidden_size * 4
    else:
        combined = embeddings_dict[args.embedding_mode]
        dim = hidden_size

    print(f"Combined embeddings shape: {combined.shape} (mode={args.embedding_mode})")

    # Filter to valid only, keeping source alignment
    valid_smiles = []
    valid_sources = []
    valid_embeddings = []
    for i, is_valid in enumerate(validity_list):
        if is_valid:
            valid_smiles.append(canonical_list[i])
            valid_sources.append(sources_list[i])
            valid_embeddings.append(combined[i])

    embeddings_array = np.stack(valid_embeddings, axis=0)
    print(f"Final: {len(valid_smiles)} valid embeddings, dim={dim}")

    # Save in same format as molmim embed script
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(
            {
                "smiles": valid_smiles,
                "sources": valid_sources,
                "embeddings": embeddings_array,
                "dim": dim,
                "model": f"kermt_{args.embedding_mode}",
            },
            f,
        )
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
