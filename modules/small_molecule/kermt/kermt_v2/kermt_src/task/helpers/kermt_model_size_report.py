#!/usr/bin/env python
"""
Shared helpers to build KERMT pretrain models and print parameter breakdowns.

Used by task/helpers/check_checkpoint.py --show_model_size (checkpoint args + vocab pickles).
"""
import os
import pickle
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from kermt.model.models import KermtCMIMTask, KermtHybridTask, KermtTask, KERMTEmbedding  # noqa: E402


def load_vocab(vocab_path):
    """Load vocabulary from pickle file and return its size."""
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    return len(vocab)


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def count_parameters_by_component_cmim(model):
    component_params = {}
    if hasattr(model, "latent_dist"):
        component_params["Latent Distribution (Encoder + Readout + Heads)"] = sum(
            p.numel() for p in model.latent_dist.parameters()
        )
        if hasattr(model.latent_dist, "kermt"):
            component_params["  ├─ Encoder (GROVER)"] = sum(
                p.numel() for p in model.latent_dist.kermt.parameters()
            )
        if hasattr(model.latent_dist, "readout"):
            component_params["  ├─ Readout"] = sum(
                p.numel() for p in model.latent_dist.readout.parameters()
            )
        if hasattr(model.latent_dist, "fc_mean_logscale"):
            component_params["  └─ Mean/LogScale Projection"] = sum(
                p.numel() for p in model.latent_dist.fc_mean_logscale.parameters()
            )
    if hasattr(model, "decoder"):
        component_params["Decoder (SMILES Transformer)"] = sum(
            p.numel() for p in model.decoder.parameters()
        )
    return component_params


def count_parameters_by_component_vocab(model):
    component_params = {}
    if hasattr(model, "kermt"):
        component_params["Encoder (GROVER)"] = sum(p.numel() for p in model.kermt.parameters())
    if hasattr(model, "vocab_module"):
        vm = model.vocab_module
        component_params["Vocab Prediction Module (Total)"] = sum(
            p.numel() for p in vm.parameters()
        )
        if hasattr(vm, "av_task_atom"):
            component_params["  ├─ Atom Vocab (from atom)"] = sum(
                p.numel() for p in vm.av_task_atom.parameters()
            )
        if hasattr(vm, "av_task_bond"):
            component_params["  ├─ Atom Vocab (from bond)"] = sum(
                p.numel() for p in vm.av_task_bond.parameters()
            )
        if hasattr(vm, "bv_task_atom"):
            component_params["  ├─ Bond Vocab (from atom)"] = sum(
                p.numel() for p in vm.bv_task_atom.parameters()
            )
        if hasattr(vm, "bv_task_bond"):
            component_params["  ├─ Bond Vocab (from bond)"] = sum(
                p.numel() for p in vm.bv_task_bond.parameters()
            )
        if hasattr(vm, "fg_task_all"):
            component_params["  └─ Functional Group"] = sum(
                p.numel() for p in vm.fg_task_all.parameters()
            )
    return component_params


def count_parameters_by_component_hybrid(model):
    component_params = {}
    if hasattr(model, "latent_dist"):
        component_params["Latent Distribution (Encoder + Readout + Heads)"] = sum(
            p.numel() for p in model.latent_dist.parameters()
        )
        if hasattr(model.latent_dist, "kermt"):
            component_params["  ├─ Encoder (GROVER)"] = sum(
                p.numel() for p in model.latent_dist.kermt.parameters()
            )
        if hasattr(model.latent_dist, "readout"):
            component_params["  ├─ Readout"] = sum(
                p.numel() for p in model.latent_dist.readout.parameters()
            )
        if hasattr(model.latent_dist, "fc_mean_logscale"):
            component_params["  └─ Mean/LogScale Projection"] = sum(
                p.numel() for p in model.latent_dist.fc_mean_logscale.parameters()
            )
    if hasattr(model, "decoder"):
        component_params["Decoder (SMILES Transformer)"] = sum(
            p.numel() for p in model.decoder.parameters()
        )
    if hasattr(model, "vocab_module"):
        component_params["Vocab Prediction Module (Total)"] = sum(
            p.numel() for p in model.vocab_module.parameters()
        )
        if hasattr(model.vocab_module, "av_task_atom"):
            component_params["  ├─ Atom Vocab (from atom)"] = sum(
                p.numel() for p in model.vocab_module.av_task_atom.parameters()
            )
        if hasattr(model.vocab_module, "av_task_bond"):
            component_params["  ├─ Atom Vocab (from bond)"] = sum(
                p.numel() for p in model.vocab_module.av_task_bond.parameters()
            )
        if hasattr(model.vocab_module, "bv_task_atom"):
            component_params["  ├─ Bond Vocab (from atom)"] = sum(
                p.numel() for p in model.vocab_module.bv_task_atom.parameters()
            )
        if hasattr(model.vocab_module, "bv_task_bond"):
            component_params["  ├─ Bond Vocab (from bond)"] = sum(
                p.numel() for p in model.vocab_module.bv_task_bond.parameters()
            )
        if hasattr(model.vocab_module, "fg_task_all"):
            component_params["  └─ Functional Group"] = sum(
                p.numel() for p in model.vocab_module.fg_task_all.parameters()
            )
    return component_params


def format_size(num_params):
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    if num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    if num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    return str(num_params)


def create_cmim_model(args, smiles_vocab_size):
    args.atom_vocab_size = 100
    args.bond_vocab_size = 100
    args.backbone = "gtrans"
    args.embedding_output_type = "both"
    args.dense = False
    args.bias = False
    args.undirected = False
    args.cuda = False
    args.features_dim = 0
    args.no_cache = True
    args.smiles_vocab_path = None
    args.tensorboard = False
    if not hasattr(args, "decoder_gate_self_attn"):
        args.decoder_gate_self_attn = False
    if not hasattr(args, "decoder_gate_cross_attn"):
        args.decoder_gate_cross_attn = False
    kermt = KERMTEmbedding(args)
    model = KermtCMIMTask(
        args,
        kermt=kermt,
        latent_dim=args.latent_dim,
        contrastive_temperature=args.contrastive_temperature,
        smiles_vocab_size=smiles_vocab_size,
    )
    return model, args


def create_vocab_model(args, atom_vocab_size, bond_vocab_size, fg_size):
    args.backbone = "gtrans"
    args.embedding_output_type = "both"
    args.dense = False
    args.bias = False
    args.undirected = False
    args.cuda = False
    args.features_dim = 0
    args.no_cache = True
    kermt = KERMTEmbedding(args)
    model = KermtTask(args, kermt, atom_vocab_size, bond_vocab_size, fg_size)
    return model, args


def create_hybrid_model(args, smiles_vocab_size, atom_vocab_size, bond_vocab_size, fg_size):
    args.atom_vocab_size = atom_vocab_size
    args.bond_vocab_size = bond_vocab_size
    args.backbone = "gtrans"
    args.embedding_output_type = "both"
    args.dense = False
    args.bias = False
    args.undirected = False
    args.cuda = False
    args.features_dim = 0
    args.no_cache = True
    args.smiles_vocab_path = None
    args.tensorboard = False
    if not hasattr(args, "decoder_gate_self_attn"):
        args.decoder_gate_self_attn = False
    if not hasattr(args, "decoder_gate_cross_attn"):
        args.decoder_gate_cross_attn = False
    kermt = KERMTEmbedding(args)
    model = KermtHybridTask(
        args,
        kermt=kermt,
        latent_dim=args.latent_dim,
        contrastive_temperature=args.contrastive_temperature,
        smiles_vocab_size=smiles_vocab_size,
        atom_vocab_size=atom_vocab_size,
        bond_vocab_size=bond_vocab_size,
        fg_size=fg_size,
    )
    return model, args


def print_model_info_cmim(model, args, smiles_vocab_size):
    print("\n" + "=" * 70)
    print("GROVER+CMIM MODEL SIZE")
    print("=" * 70)
    print("\nModel Configuration:")
    print("  Encoder (GROVER):")
    print(f"    Hidden Size:        {args.hidden_size}")
    print(f"    Depth (Layers):     {args.depth}")
    print(f"    Attention Heads:    {args.num_attn_head}")
    print(f"    MT Blocks:          {args.num_mt_block}")
    print(f"    Activation:         {args.activation}")
    print(f"    Dropout:            {args.dropout}")
    print(f"    Readout:            {'Self-Attention' if args.self_attention else 'Mean'}")
    if args.self_attention:
        print(f"      Attn Hidden:      {args.attn_hidden}")
        print(f"      Attn Out:         {args.attn_out}")
    print("\n  Decoder (CMIM):")
    print(f"    SMILES Vocab Size:  {smiles_vocab_size}")
    print(f"    Latent Dim:         {args.latent_dim}")
    print(f"    Decoder Layers:     {args.decoder_num_layers}")
    print(f"    Decoder Heads:      {args.decoder_num_attention_heads}")
    print(f"    Decoder FFN Dim:    {args.decoder_ffn_hidden_size}")
    print(f"    Decoder Dropout:    {args.decoder_dropout}")
    print(f"    Positional Encoding: {args.decoder_positional_encoding}")
    print(
        f"    Decoder G1 gates:   self={args.decoder_gate_self_attn}, "
        f"cross={args.decoder_gate_cross_attn}"
    )
    total_params, trainable_params = count_parameters(model)
    print("\nParameter Count:")
    print(f"  Total Parameters:      {total_params:,} ({format_size(total_params)})")
    print(f"  Trainable Parameters:  {trainable_params:,} ({format_size(trainable_params)})")
    print("\nParameter Breakdown by Component:")
    component_params = count_parameters_by_component_cmim(model)
    for component, count in component_params.items():
        percentage = (count / total_params) * 100
        print(f"  {component:50s}: {count:12,} ({format_size(count):>8s}) [{percentage:5.1f}%]")
    param_size_mb = (total_params * 4) / (1024**2)
    print("\nApproximate Model Size:")
    print(f"  FP32: {param_size_mb:.2f} MB")
    print(f"  FP16: {param_size_mb / 2:.2f} MB")
    print("=" * 70 + "\n")


def print_model_info_vocab(model, args, atom_vocab_size, bond_vocab_size, fg_size):
    print("\n" + "=" * 70)
    print("KERMT (VOCAB-BASED) MODEL SIZE")
    print("=" * 70)
    print("\nModel Configuration:")
    print("  Encoder (GROVER):")
    print(f"    Hidden Size:        {args.hidden_size}")
    print(f"    Depth (Layers):     {args.depth}")
    print(f"    Attention Heads:    {args.num_attn_head}")
    print(f"    MT Blocks:          {args.num_mt_block}")
    print(f"    Activation:         {args.activation}")
    print(f"    Dropout:            {args.dropout}")
    print("\n  Prediction Tasks:")
    print(f"    Atom Vocab Size:    {atom_vocab_size}")
    print(f"    Bond Vocab Size:    {bond_vocab_size}")
    print(f"    FG Size:            {fg_size}")
    total_params, trainable_params = count_parameters(model)
    print("\nParameter Count:")
    print(f"  Total Parameters:      {total_params:,} ({format_size(total_params)})")
    print(f"  Trainable Parameters:  {trainable_params:,} ({format_size(trainable_params)})")
    print("\nParameter Breakdown by Component:")
    component_params = count_parameters_by_component_vocab(model)
    for component, count in component_params.items():
        percentage = (count / total_params) * 100
        print(f"  {component:35s}: {count:12,} ({format_size(count):>8s}) [{percentage:5.1f}%]")
    param_size_mb = (total_params * 4) / (1024**2)
    print("\nApproximate Model Size:")
    print(f"  FP32: {param_size_mb:.2f} MB")
    print(f"  FP16: {param_size_mb / 2:.2f} MB")
    print("=" * 70 + "\n")


def print_model_info_hybrid(
    model, args, smiles_vocab_size, atom_vocab_size, bond_vocab_size, fg_size
):
    print("\n" + "=" * 70)
    print("KERMT HYBRID (CMIM + VOCAB) MODEL SIZE")
    print("=" * 70)
    print("\nModel Configuration:")
    print("  Encoder (GROVER):")
    print(f"    Hidden Size:        {args.hidden_size}")
    print(f"    Depth (Layers):     {args.depth}")
    print(f"    Attention Heads:    {args.num_attn_head}")
    print(f"    MT Blocks:          {args.num_mt_block}")
    print(f"    Activation:         {args.activation}")
    print(f"    Dropout:            {args.dropout}")
    print(f"    Readout:            {'Self-Attention' if args.self_attention else 'Mean'}")
    if args.self_attention:
        print(f"      Attn Hidden:      {args.attn_hidden}")
        print(f"      Attn Out:         {args.attn_out}")
    print("\n  CMIM Components:")
    print(f"    SMILES Vocab Size:  {smiles_vocab_size}")
    print(f"    Latent Dim:         {args.latent_dim}")
    print(f"    Decoder Layers:     {args.decoder_num_layers}")
    print(f"    Decoder Heads:      {args.decoder_num_attention_heads}")
    print(f"    Decoder FFN Dim:    {args.decoder_ffn_hidden_size}")
    print(f"    Decoder Dropout:    {args.decoder_dropout}")
    print(f"    Positional Encoding: {args.decoder_positional_encoding}")
    print(
        f"    Decoder G1 gates:   self={args.decoder_gate_self_attn}, "
        f"cross={args.decoder_gate_cross_attn}"
    )
    print("\n  Vocab Components:")
    print(f"    Atom Vocab Size:    {atom_vocab_size}")
    print(f"    Bond Vocab Size:    {bond_vocab_size}")
    print(f"    FG Size:            {fg_size}")
    print(f"    Vocab Loss Weight:  {getattr(args, 'vocab_loss_weight', 1.0)}")
    total_params, trainable_params = count_parameters(model)
    print("\nParameter Count:")
    print(f"  Total Parameters:      {total_params:,} ({format_size(total_params)})")
    print(f"  Trainable Parameters:  {trainable_params:,} ({format_size(trainable_params)})")
    print("\nParameter Breakdown by Component:")
    component_params = count_parameters_by_component_hybrid(model)
    for component, count in component_params.items():
        percentage = (count / total_params) * 100
        print(f"  {component:50s}: {count:12,} ({format_size(count):>8s}) [{percentage:5.1f}%]")
    param_size_mb = (total_params * 4) / (1024**2)
    print("\nApproximate Model Size:")
    print(f"  FP32: {param_size_mb:.2f} MB")
    print(f"  FP16: {param_size_mb / 2:.2f} MB")
    print("=" * 70 + "\n")


def print_model_info_embedding_only(model, args):
    """Report size for KERMTEmbedding-only weights (legacy grover.* checkpoints)."""
    print("\n" + "=" * 70)
    print("KERMTEmbedding ONLY (encoder / legacy GROVER-style checkpoint)")
    print("=" * 70)
    print("\n  No vocab heads or CMIM decoder in this file; counts are for the graph encoder only.")
    print("\nModel Configuration:")
    print("  Encoder (GROVER / GTrans):")
    print(f"    Hidden Size:        {args.hidden_size}")
    print(f"    Depth (Layers):     {args.depth}")
    print(f"    Attention Heads:    {args.num_attn_head}")
    print(f"    MT Blocks:          {args.num_mt_block}")
    print(f"    Activation:         {args.activation}")
    print(f"    Dropout:            {args.dropout}")
    print(f"    Readout:            {'Self-Attention' if args.self_attention else 'Mean'}")
    if args.self_attention:
        print(f"      Attn Hidden:      {args.attn_hidden}")
        print(f"      Attn Out:         {args.attn_out}")
    total_params, trainable_params = count_parameters(model)
    print("\nParameter Count:")
    print(f"  Total Parameters:      {total_params:,} ({format_size(total_params)})")
    print(f"  Trainable Parameters:  {trainable_params:,} ({format_size(trainable_params)})")
    print("\nParameter Breakdown by Submodule:")
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        pct = (100.0 * n / total_params) if total_params else 0.0
        print(f"  {name:20s}: {n:12,} ({format_size(n):>8s}) [{pct:5.1f}%]")
    param_size_mb = (total_params * 4) / (1024**2)
    print("\nApproximate Model Size:")
    print(f"  FP32: {param_size_mb:.2f} MB")
    print(f"  FP16: {param_size_mb / 2:.2f} MB")
    print("=" * 70 + "\n")
