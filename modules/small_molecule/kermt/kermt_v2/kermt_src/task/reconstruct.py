#!/usr/bin/env python
"""
SMILES reconstruction and generation from pretrained GROVER+CMIM model.

This script loads a pretrained GROVER+CMIM checkpoint and uses it to:
1. Encode an input SMILES string to latent space
2. Decode the latent representation to reconstruct/generate SMILES
3. Compute edit distance between canonicalized input and raw output

Usage:
    # Reconstruct using mean (deterministic)
    python task/reconstruct.py -c model.pt -v smiles_vocab.pkl -s "CCO" --use_mean

    # Generate multiple samples using z_latent (stochastic)
    python task/reconstruct.py -c model.pt -v smiles_vocab.pkl -s "CCO" -n 5

    # With temperature for sampling during decoding
    python task/reconstruct.py -c model.pt -v smiles_vocab.pkl -s "CCO" -t 0.8

    # Use top-k sampling
    python task/reconstruct.py -c model.pt -v smiles_vocab.pkl -s "CCO" --top_k 50

    # Batch processing from file
    python task/reconstruct.py -c model.pt -v smiles_vocab.pkl -i input.csv -o results.csv
"""

import argparse
import csv
import time
from typing import List, Tuple, Optional
import statistics

import torch
import torch.nn.functional as F

from rdkit import Chem
from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.CRITICAL)

from fast_edit_distance import edit_distance

from kermt.data.torchvocab import SMILESVocab, SMILES_SPECIAL_TOKENS
from kermt.data.molgraph import mol2graph
from kermt.model.models import KermtCMIMTask, KERMTEmbedding


def is_valid_smiles(smiles: str) -> bool:
    """Check if a SMILES string is valid using RDKit."""
    if not smiles:
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize SMILES using RDKit. Returns None if invalid."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def load_cmim_checkpoint(
    checkpoint_path: str, smiles_vocab: SMILESVocab, device: str = "cuda"
) -> Tuple[KermtCMIMTask, argparse.Namespace]:
    """
    Load a GROVER+CMIM model from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        smiles_vocab: SMILES vocabulary instance
        device: Device to load the model on ('cuda' or 'cpu')

    Returns:
        Tuple of (model, args)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = state["args"]
    loaded_state_dict = state["state_dict"]

    print(
        f"Checkpoint from epoch {state.get('epoch', 'unknown')}, step {state.get('scheduler_step', 'unknown')}"
    )

    # Build the model with saved args
    smiles_vocab_size = len(smiles_vocab)

    # Ensure required args are set
    args.cuda = device == "cuda"

    # Build encoder
    kermt_embedding = KERMTEmbedding(args)

    # Build complete CMIM model
    model = KermtCMIMTask(
        args,
        kermt=kermt_embedding,
        latent_dim=args.latent_dim,
        contrastive_temperature=args.contrastive_temperature,
        smiles_vocab_size=smiles_vocab_size,
    )

    # Load state dict
    model.load_state_dict(loaded_state_dict)
    model.to(device)
    model.eval()

    print(f"Model loaded successfully. Latent dim: {args.latent_dim}")

    return model, args


def smiles_to_graph_batch(smiles_list: List[str], args: argparse.Namespace) -> Tuple:
    """
    Convert SMILES strings to graph batch format for the encoder.

    Args:
        smiles_list: List of SMILES strings
        args: Model arguments

    Returns:
        Graph batch tuple (f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a)
    """
    shared_dict = {}

    # Create graph batch
    batch_graph = mol2graph(smiles_list, shared_dict, args)

    return batch_graph.get_components()


def create_causal_mask(
    seq_len: int, positional_encoding: str, device: str = "cpu"
) -> torch.Tensor:
    """
    Create causal mask for autoregressive generation.

    Args:
        seq_len: Sequence length
        positional_encoding: Type of positional encoding ('rope' or 'sinusoidal')
        device: Device to create mask on

    Returns:
        Causal mask tensor
    """
    if positional_encoding == "sinusoidal":
        # PyTorch's nn.TransformerDecoder expects BoolTensor
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
        )
    elif positional_encoding == "rope":
        # Custom RoPE implementation expects FloatTensor with additive masking
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
    else:
        raise ValueError(f"Unknown positional_encoding: {positional_encoding}")

    return mask


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
) -> int:
    """
    Sample a token from logits with optional temperature, top-k, and top-p.

    Args:
        logits: Logits for next token [vocab_size]
        temperature: Temperature for sampling (1.0 = no change, <1 = sharper, >1 = smoother)
        top_k: If set, only sample from top k tokens
        top_p: If set, use nucleus sampling with this probability threshold

    Returns:
        Sampled token ID
    """
    if temperature <= 0:
        # Greedy decoding
        return logits.argmax().item()

    # Apply temperature
    logits = logits / temperature

    # Apply top-k filtering
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float("-inf")

    # Apply top-p (nucleus) filtering
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = float("-inf")

    # Sample from distribution
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).item()

    return next_token


def autoregressive_decode(
    model: KermtCMIMTask,
    memory: torch.Tensor,
    smiles_vocab: SMILESVocab,
    max_len: int = 512,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    device: str = "cuda",
) -> str:
    """
    Autoregressively decode SMILES from latent representation.

    Args:
        model: The CMIM model
        memory: Latent representation [1, 1, latent_dim] (for single sample)
        smiles_vocab: SMILES vocabulary
        max_len: Maximum sequence length
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        device: Device for computation

    Returns:
        Decoded SMILES string
    """
    decoder = model.decoder
    positional_encoding = decoder.positional_encoding

    # Start with <start> token
    start_token_id = smiles_vocab.start_index
    end_token_id = smiles_vocab.end_index

    # Initialize sequence with <start>
    generated_ids = [start_token_id]

    with torch.no_grad():
        for _ in range(max_len - 1):
            # Current sequence as tensor
            decoder_input = torch.tensor(
                [generated_ids], dtype=torch.long, device=device
            )  # [1, seq_len]

            # Create causal mask
            seq_len = decoder_input.size(1)
            causal_mask = create_causal_mask(seq_len, positional_encoding, device)

            # Forward through decoder
            logits = decoder(
                decoder_input=decoder_input,
                memory=memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
            )  # [1, seq_len, vocab_size]

            # Get logits for the last position
            next_logits = logits[0, -1, :]  # [vocab_size]

            # Sample next token
            next_token_id = sample_token(next_logits, temperature, top_k, top_p)

            # Append to sequence
            generated_ids.append(next_token_id)

            # Stop if <end> token is generated
            if next_token_id == end_token_id:
                break

    # Convert token IDs to SMILES string
    # Skip special tokens (<start>, <end>, <pad>, etc.)
    smiles = ids_to_smiles(generated_ids, smiles_vocab, skip_special=True)

    return smiles


def ids_to_smiles(ids: List[int], vocab: SMILESVocab, skip_special: bool = True) -> str:
    """
    Convert token IDs to SMILES string.

    Args:
        ids: List of token IDs
        vocab: SMILES vocabulary
        skip_special: Whether to skip special tokens

    Returns:
        SMILES string
    """
    special_tokens = set(SMILES_SPECIAL_TOKENS)
    tokens = []

    for idx in ids:
        if idx < len(vocab.itos):
            token = vocab.itos[idx]
        else:
            token = "<unk>"

        if skip_special and token in special_tokens:
            continue

        tokens.append(token)

    return "".join(tokens)


def encode_smiles(
    model: KermtCMIMTask,
    smiles: str,
    args: argparse.Namespace,
    use_mean: bool = False,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Encode a SMILES string to latent representation.

    Args:
        model: The CMIM model
        smiles: Input SMILES string
        args: Model arguments
        use_mean: If True, use mean of latent distribution (deterministic)
                  If False, sample from distribution (stochastic)
        device: Device for computation

    Returns:
        Latent representation [1, latent_dim]
    """
    # Convert SMILES to graph batch
    graph_batch = smiles_to_graph_batch([smiles], args)

    # Move to device (graph_batch is a tuple of tensors)
    graph_batch = tuple(
        t.to(device) if isinstance(t, torch.Tensor) else t for t in graph_batch
    )

    # Get latent representation
    with torch.no_grad():
        if use_mean:
            # Use mean of distribution (deterministic)
            mean, log_scale = model.latent_dist.forward(graph_batch)
            z_latent = mean
        else:
            # Sample from distribution (stochastic)
            z_latent = model.latent_dist.sample(graph_batch, return_params=False)

    return z_latent


def reconstruct_smiles(
    model: KermtCMIMTask,
    smiles: str,
    smiles_vocab: SMILESVocab,
    args: argparse.Namespace,
    use_mean: bool = False,
    num_samples: int = 1,
    max_len: int = 512,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    device: str = "cuda",
) -> List[str]:
    """
    Reconstruct/generate SMILES from input SMILES.

    Args:
        model: The CMIM model
        smiles: Input SMILES string
        smiles_vocab: SMILES vocabulary
        args: Model arguments
        use_mean: If True, use mean of latent distribution
        num_samples: Number of SMILES to generate
        max_len: Maximum sequence length
        temperature: Sampling temperature for decoding
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        device: Device for computation

    Returns:
        List of reconstructed/generated SMILES strings
    """
    results = []

    for i in range(num_samples):
        # Encode to latent space
        z_latent = encode_smiles(model, smiles, args, use_mean=use_mean, device=device)

        # Prepare memory for decoder [batch_size, 1, latent_dim]
        memory = z_latent.unsqueeze(1)  # [1, 1, latent_dim]

        # Decode
        generated = autoregressive_decode(
            model=model,
            memory=memory,
            smiles_vocab=smiles_vocab,
            max_len=max_len,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
        )

        results.append(generated)

    return results


def process_batch_from_file(
    model: KermtCMIMTask,
    smiles_vocab: SMILESVocab,
    model_args: argparse.Namespace,
    input_file: str,
    output_file: str,
    use_mean: bool = False,
    num_samples: int = 1,
    max_len: int = 512,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    device: str = "cuda",
):
    """
    Process a batch of SMILES from file and write results.

    Input SMILES are canonicalized before passing to the model.
    Edit distance is computed between canonicalized input and raw output.

    Args:
        model: The CMIM model
        smiles_vocab: SMILES vocabulary
        model_args: Model arguments
        input_file: Path to input CSV file (first column should be SMILES)
        output_file: Path to output CSV file
        use_mean: If True, use mean of latent distribution
        num_samples: Number of SMILES to generate per input
        max_len: Maximum sequence length
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        device: Device for computation
    """
    # Read input SMILES
    raw_input_smiles = []
    with open(input_file, "r") as f:
        reader = csv.reader(f)
        _header = next(reader, None)  # Skip header if present
        for row in reader:
            if row:
                raw_input_smiles.append(row[0].strip())

    print(f"Loaded {len(raw_input_smiles)} SMILES from {input_file}")

    # Canonicalize input SMILES (skip invalid ones)
    valid_inputs = []
    skipped_count = 0
    for raw_smi in raw_input_smiles:
        can_smi = canonicalize_smiles(raw_smi)
        if can_smi is not None:
            valid_inputs.append({"raw": raw_smi, "canonical": can_smi})
        else:
            skipped_count += 1

    if skipped_count > 0:
        print(f"Skipped {skipped_count} invalid input SMILES")
    print(f"Processing {len(valid_inputs)} valid SMILES")

    # Process each SMILES
    # Each row in results will be a single (input, generated) pair
    results = []
    all_edit_distances = []  # Collect all edit distances for statistics
    all_input_lengths = []  # Collect input lengths for statistics
    all_generated_lengths = []  # Collect generated lengths for statistics (valid only)
    total_samples = 0
    valid_samples = 0
    invalid_samples = 0

    for i, input_data in enumerate(valid_inputs):
        if (i + 1) % 100 == 0:
            print(f"Processing {i + 1}/{len(valid_inputs)}...")

        raw_input = input_data["raw"]
        canonical_input = input_data["canonical"]
        input_len = len(canonical_input)
        all_input_lengths.append(input_len)

        try:
            # Use canonicalized input for the model
            generated = reconstruct_smiles(
                model=model,
                smiles=canonical_input,
                smiles_vocab=smiles_vocab,
                args=model_args,
                use_mean=use_mean,
                num_samples=num_samples,
                max_len=max_len,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=device,
            )

            # Create a row for each (input, generated) pair
            for sample_idx, gen_smi in enumerate(generated):
                is_valid = is_valid_smiles(gen_smi)
                total_samples += 1
                gen_len = len(gen_smi)

                if is_valid:
                    valid_samples += 1
                    # Edit distance between canonicalized input and raw output
                    ed = edit_distance(canonical_input, gen_smi)
                    all_edit_distances.append(ed)
                    all_generated_lengths.append(gen_len)
                else:
                    invalid_samples += 1
                    ed = None

                results.append(
                    {
                        "raw_input": raw_input,
                        "canonical_input": canonical_input,
                        "input_len": input_len,
                        "sample_idx": sample_idx + 1,  # 1-indexed
                        "generated": gen_smi,
                        "generated_len": gen_len,
                        "valid": is_valid,
                        "edit_distance": ed,
                    }
                )

        except Exception as e:
            print(f"Error processing SMILES '{raw_input}': {e}")
            for sample_idx in range(num_samples):
                results.append(
                    {
                        "raw_input": raw_input,
                        "canonical_input": canonical_input,
                        "input_len": input_len,
                        "sample_idx": sample_idx + 1,
                        "generated": "ERROR",
                        "generated_len": 0,
                        "valid": False,
                        "edit_distance": None,
                    }
                )
                invalid_samples += 1
                total_samples += 1

    # Write results (overwrites if file exists)
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)

        # Write header - one row per (input, generated) pair
        writer.writerow(
            [
                "raw_input",
                "canonical_input",
                "input_len",
                "sample_idx",
                "generated",
                "generated_len",
                "valid",
                "edit_distance",
            ]
        )

        # Write data - each row is a single (input, generated) pair
        for r in results:
            ed_str = r["edit_distance"] if r["edit_distance"] is not None else "N/A"
            row = [
                r["raw_input"],
                r["canonical_input"],
                r["input_len"],
                r["sample_idx"],
                r["generated"],
                r["generated_len"],
                r["valid"],
                ed_str,
            ]
            writer.writerow(row)

    print(f"\nResults written to {output_file} (file overwritten if existed)")

    # Print statistics
    print(f"\n{'=' * 70}")
    print("RECONSTRUCTION STATISTICS")
    print(f"{'=' * 70}")

    print("\n--- Sample Validity ---")
    print(f"  Total input molecules:  {len(valid_inputs)}")
    print(f"  Samples per molecule:   {num_samples}")
    print(f"  Total samples:          {total_samples}")
    print(
        f"  Valid samples:          {valid_samples} ({100 * valid_samples / total_samples:.1f}%)"
    )
    print(
        f"  Invalid samples:        {invalid_samples} ({100 * invalid_samples / total_samples:.1f}%)"
    )

    # Input length statistics
    if all_input_lengths:
        print("\n--- Input Length Distribution (canonical SMILES) ---")
        print(f"  Count:    {len(all_input_lengths)}")
        print(f"  Min:      {min(all_input_lengths)}")
        print(f"  Max:      {max(all_input_lengths)}")
        print(f"  Mean:     {statistics.mean(all_input_lengths):.2f}")
        print(f"  Median:   {statistics.median(all_input_lengths):.1f}")
        if len(all_input_lengths) > 1:
            print(f"  Std Dev:  {statistics.stdev(all_input_lengths):.2f}")

    # Generated length statistics (valid only)
    if all_generated_lengths:
        print("\n--- Generated Length Distribution (valid samples only) ---")
        print(f"  Count:    {len(all_generated_lengths)}")
        print(f"  Min:      {min(all_generated_lengths)}")
        print(f"  Max:      {max(all_generated_lengths)}")
        print(f"  Mean:     {statistics.mean(all_generated_lengths):.2f}")
        print(f"  Median:   {statistics.median(all_generated_lengths):.1f}")
        if len(all_generated_lengths) > 1:
            print(f"  Std Dev:  {statistics.stdev(all_generated_lengths):.2f}")

    if all_edit_distances:
        print("\n--- Edit Distance Distribution (valid samples only) ---")
        print(f"  Count:    {len(all_edit_distances)}")
        print(f"  Min:      {min(all_edit_distances)}")
        print(f"  Max:      {max(all_edit_distances)}")
        print(f"  Mean:     {statistics.mean(all_edit_distances):.2f}")
        print(f"  Median:   {statistics.median(all_edit_distances):.1f}")
        if len(all_edit_distances) > 1:
            print(f"  Std Dev:  {statistics.stdev(all_edit_distances):.2f}")

        # Percentiles
        sorted_eds = sorted(all_edit_distances)
        n = len(sorted_eds)
        print("\n--- Edit Distance Percentiles ---")
        for p in [25, 50, 75, 90, 95, 99]:
            idx = int(n * p / 100)
            idx = min(idx, n - 1)
            print(f"  {p}th percentile: {sorted_eds[idx]}")

        # Distribution buckets
        print("\n--- Edit Distance Buckets ---")
        exact_match = sum(1 for ed in all_edit_distances if ed == 0)
        ed_1_5 = sum(1 for ed in all_edit_distances if 1 <= ed <= 5)
        ed_6_10 = sum(1 for ed in all_edit_distances if 6 <= ed <= 10)
        ed_11_20 = sum(1 for ed in all_edit_distances if 11 <= ed <= 20)
        ed_21_plus = sum(1 for ed in all_edit_distances if ed > 20)

        print(
            f"  ED = 0 (exact):    {exact_match} ({100 * exact_match / len(all_edit_distances):.1f}%)"
        )
        print(
            f"  ED 1-5:            {ed_1_5} ({100 * ed_1_5 / len(all_edit_distances):.1f}%)"
        )
        print(
            f"  ED 6-10:           {ed_6_10} ({100 * ed_6_10 / len(all_edit_distances):.1f}%)"
        )
        print(
            f"  ED 11-20:          {ed_11_20} ({100 * ed_11_20 / len(all_edit_distances):.1f}%)"
        )
        print(
            f"  ED > 20:           {ed_21_plus} ({100 * ed_21_plus / len(all_edit_distances):.1f}%)"
        )
    else:
        print("\n  No valid samples to compute edit distance statistics.")

    print(f"\n{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct/generate SMILES from pretrained GROVER+CMIM model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        required=True,
        help="Path to CMIM model checkpoint",
    )
    parser.add_argument(
        "--smiles_vocab_path",
        "-v",
        type=str,
        required=True,
        help="Path to SMILES vocabulary file (.pkl)",
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--smiles", "-s", type=str, help="Input SMILES string to encode and reconstruct"
    )
    input_group.add_argument(
        "--input_file", "-i", type=str, help="Input CSV file with SMILES (first column)"
    )

    # Output for batch mode
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        default=None,
        help="Output CSV file for batch mode (required with --input_file)",
    )

    # Latent sampling options
    parser.add_argument(
        "--use_mean",
        action="store_true",
        help="Use mean of latent distribution (deterministic encoding)",
    )
    parser.add_argument(
        "--num_samples",
        "-n",
        type=int,
        default=1,
        help="Number of SMILES to generate (uses different latent samples)",
    )

    # Decoding options
    parser.add_argument(
        "--max_len",
        type=int,
        default=512,
        help="Maximum sequence length for generation",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=1.0,
        help="Sampling temperature (0 = greedy, <1 = sharper, >1 = smoother)",
    )
    parser.add_argument(
        "--top_k", type=int, default=None, help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--top_p", type=float, default=None, help="Nucleus (top-p) sampling parameter"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for computation",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.input_file and not args.output_file:
        parser.error("--output_file is required when using --input_file")

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # Load SMILES vocabulary
    print(f"Loading SMILES vocabulary from: {args.smiles_vocab_path}")
    smiles_vocab = SMILESVocab.load_vocab(args.smiles_vocab_path)
    print(f"Vocabulary size: {len(smiles_vocab)}")

    # Load model
    model, model_args = load_cmim_checkpoint(
        args.checkpoint, smiles_vocab, device=args.device
    )

    # Start timing for processing
    start_time = time.time()

    if args.input_file:
        # Batch mode
        process_batch_from_file(
            model=model,
            smiles_vocab=smiles_vocab,
            model_args=model_args,
            input_file=args.input_file,
            output_file=args.output_file,
            use_mean=args.use_mean,
            num_samples=args.num_samples,
            max_len=args.max_len,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device,
        )
    else:
        # Single SMILES mode
        # Canonicalize input first
        canonical_input = canonicalize_smiles(args.smiles)
        if canonical_input is None:
            print(f"Error: Invalid input SMILES: {args.smiles}")
            return

        input_len = len(canonical_input)
        print(f"\nRaw input SMILES:       {args.smiles}")
        print(f"Canonical input SMILES: {canonical_input}")
        print(f"Canonical input length: {input_len}")
        print(f"Use mean: {args.use_mean}")
        print(f"Num samples: {args.num_samples}")
        print(f"Temperature: {args.temperature}")

        # Use canonicalized input for the model
        results = reconstruct_smiles(
            model=model,
            smiles=canonical_input,
            smiles_vocab=smiles_vocab,
            args=model_args,
            use_mean=args.use_mean,
            num_samples=args.num_samples,
            max_len=args.max_len,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device,
        )

        # Print results
        print(f"\n{'=' * 70}")
        print("Generated SMILES:")
        print(f"{'=' * 70}")

        edit_distances = []
        generated_lengths = []
        valid_count = 0

        for i, smi in enumerate(results, 1):
            valid = is_valid_smiles(smi)
            gen_len = len(smi)

            if valid:
                valid_count += 1
                generated_lengths.append(gen_len)
                # Edit distance between canonicalized input and raw output
                ed = edit_distance(canonical_input, smi)
                edit_distances.append(ed)

                status = f"len={gen_len} ED={ed}"
                if ed == 0:
                    status += " ✓ EXACT"
            else:
                status = f"len={gen_len} ✗ INVALID"

            print(f"  [{i}] {smi} {status}")

        # Print statistics
        print(f"\n{'=' * 70}")
        print("Statistics:")
        print(f"{'=' * 70}")
        print(f"  Input length:     {input_len}")
        print(f"  Total samples:    {args.num_samples}")
        print(
            f"  Valid samples:    {valid_count} ({100 * valid_count / args.num_samples:.1f}%)"
        )
        print(
            f"  Invalid samples:  {args.num_samples - valid_count} ({100 * (args.num_samples - valid_count) / args.num_samples:.1f}%)"
        )

        if generated_lengths:
            print("\n  --- Generated Length (valid samples) ---")
            print(f"  Min:        {min(generated_lengths)}")
            print(f"  Max:        {max(generated_lengths)}")
            print(f"  Mean:       {statistics.mean(generated_lengths):.2f}")
            print(f"  Median:     {statistics.median(generated_lengths):.1f}")
            if len(generated_lengths) > 1:
                print(f"  Std Dev:    {statistics.stdev(generated_lengths):.2f}")

        if edit_distances:
            exact_matches = sum(1 for ed in edit_distances if ed == 0)
            print("\n  --- Edit Distance (valid samples) ---")
            print(
                f"  Exact match (ED=0): {exact_matches} ({100 * exact_matches / len(edit_distances):.1f}%)"
            )
            print(f"  Min ED:     {min(edit_distances)}")
            print(f"  Max ED:     {max(edit_distances)}")
            print(f"  Mean ED:    {statistics.mean(edit_distances):.2f}")
            print(f"  Median ED:  {statistics.median(edit_distances):.1f}")
            if len(edit_distances) > 1:
                print(f"  Std Dev:    {statistics.stdev(edit_distances):.2f}")

    # Print elapsed time
    elapsed_time = time.time() - start_time
    if elapsed_time >= 3600:
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60
        print(f"\nTotal processing time: {hours}h {minutes}m {seconds:.1f}s")
    elif elapsed_time >= 60:
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        print(f"\nTotal processing time: {minutes}m {seconds:.1f}s")
    else:
        print(f"\nTotal processing time: {elapsed_time:.2f}s")


if __name__ == "__main__":
    main()
