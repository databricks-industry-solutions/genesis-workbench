#!/usr/bin/env python
"""
Inspect a pretrain or finetune checkpoint and print training status.

Usage:
    python task/helpers/check_checkpoint.py path/to/last_checkpoint.pt
    python task/helpers/check_checkpoint.py path/to/last_checkpoint.pt --show_model_size

With --show_model_size, vocab files are resolved in order:
  1) explicit --smiles_vocab_path / --atom_vocab_path / --bond_vocab_path
  2) same directory as the checkpoint:
       pretrain_smiles_vocab.pkl (always pickle),
       pretrain_atom_vocab.json (then .pkl fallback),
       pretrain_bond_vocab.json (then .pkl fallback)
  3) path stored in the checkpoint (if that file exists)
  4) interactive prompt (unless --non-interactive)
"""
import argparse
import copy
import os
import sys
from collections import OrderedDict
from datetime import datetime

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch


# Default filenames when vocabs sit next to last_checkpoint.pt.
# Atom/bond vocabs default to JSON (the post-2026-05-08 convention); pickle is
# still tried as a fallback so older checkpoints can be inspected. SMILES vocab
# is always pickle — its compiled regex state is not JSON-serializable.
_VOCAB_BASENAMES = {
    "smiles_vocab_path": ("pretrain_smiles_vocab.pkl",),
    "atom_vocab_path": ("pretrain_atom_vocab.json", "pretrain_atom_vocab.pkl"),
    "bond_vocab_path": ("pretrain_bond_vocab.json", "pretrain_bond_vocab.pkl"),
}


def format_size(nbytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def _maybe_strip_ddp_prefix(state_dict):
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if keys and all(k.startswith("module.") for k in keys):
        return OrderedDict((k[len("module.") :], v) for k, v in state_dict.items())
    return state_dict


def _report_state_dict_load(result, label):
    if result is None:
        print(f"  [INFO] state_dict loaded into rebuilt {label} model.")
        return
    missing, unexpected = result.missing_keys, result.unexpected_keys
    if not missing and not unexpected:
        print(f"  [OK] Checkpoint state_dict matches rebuilt {label} model.")
        return
    print(f"  [WARN] state_dict vs rebuilt {label} model (strict=False):")
    if missing:
        tail = " ..." if len(missing) > 12 else ""
        print(f"    missing_keys ({len(missing)}): {missing[:12]}{tail}")
    if unexpected:
        tail = " ..." if len(unexpected) > 12 else ""
        print(f"    unexpected_keys ({len(unexpected)}): {unexpected[:12]}{tail}")


def _infer_pretrain_mode(state_dict):
    """Infer pretrain layout from weight keys when args.pretrain_mode is absent."""
    if not state_dict:
        return None
    keys = list(state_dict.keys())
    has_latent = any(k.startswith("latent_dist.") for k in keys)
    has_vocab_mod = any(k.startswith("vocab_module.") for k in keys)
    has_kermt = any(k.startswith("kermt.") for k in keys)
    has_grover = any(k.startswith("grover.") for k in keys)
    if has_latent and has_vocab_mod:
        return "hybrid"
    if has_latent:
        return "cmim"
    if has_vocab_mod:
        return "vocab"
    # Legacy GROVER release: encoder-only weights under grover.* (no kermt rename yet).
    # Or a bare kermt.* export without vocab / CMIM heads.
    if has_grover or has_kermt:
        return "embedding_only"
    return None


def _args_for_model_build(stored_args):
    if isinstance(stored_args, dict):
        a = argparse.Namespace(**stored_args)
    else:
        a = copy.copy(stored_args)
    defaults = {
        "backbone": "gtrans",
        "embedding_output_type": "both",
        "dense": False,
        "bias": False,
        "undirected": False,
        "cuda": False,
        "features_dim": 0,
        "no_cache": True,
        "tensorboard": False,
        "smiles_vocab_path": None,
    }
    for key, val in defaults.items():
        if not hasattr(a, key):
            setattr(a, key, val)
    if not hasattr(a, "decoder_gate_self_attn"):
        a.decoder_gate_self_attn = False
    if not hasattr(a, "decoder_gate_cross_attn"):
        a.decoder_gate_cross_attn = False
    if not hasattr(a, "dist_coff"):
        a.dist_coff = 0.1
    # Minimal encoder / vocab pretrain fields (legacy grover_base.pt args are often sparse)
    for key, val in (
        ("dropout", 0.1),
        ("num_attn_head", 4),
        ("atom_message", False),
        ("bond_drop_rate", 0.0),
        ("nencoders", 3),
        ("coord", 15),
        ("input_layer", "fc"),
        ("no_attach_fea", False),
        ("use_cuikmolmaker_featurization", False),
    ):
        if not hasattr(a, key):
            setattr(a, key, val)
    return a


def _prompt_path(label):
    while True:
        p = input(f"Enter path to {label} (vocab .pkl): ").strip()
        if p and os.path.isfile(p):
            return p
        print("  Path missing or not a file; try again.")


def resolve_vocab_path(label, checkpoint_dir, stored_path, override, interactive):
    """
    label: smiles_vocab_path | atom_vocab_path | bond_vocab_path
    """
    if override:
        if os.path.isfile(override):
            print(f"  [INFO] {label}: using CLI path {override}")
            return override
        raise FileNotFoundError(f"{label} override not found: {override}")

    for basename in _VOCAB_BASENAMES[label]:
        next_to = os.path.join(checkpoint_dir, basename)
        if os.path.isfile(next_to):
            print(f"  [INFO] {label}: using {next_to}")
            return next_to

    if stored_path and os.path.isfile(stored_path):
        print(f"  [INFO] {label}: using checkpoint-recorded path {stored_path}")
        return stored_path

    if stored_path:
        alt = os.path.join(checkpoint_dir, os.path.basename(stored_path))
        if os.path.isfile(alt):
            print(f"  [INFO] {label}: using {alt} (basename from checkpoint path)")
            return alt

    if interactive:
        print(f"  [WARN] Could not find {basename} next to checkpoint or at recorded path.")
        return _prompt_path(label)

    raise FileNotFoundError(
        f"{label}: place {basename} in {checkpoint_dir}, pass --{label}, "
        f"or run without --non_interactive for a prompt."
    )


def _print_pretrain_model_size(
    stored_args, state_dict, checkpoint_dir, vocab_overrides, interactive, mode
):
    from task.helpers.kermt_model_size_report import (
        create_cmim_model,
        create_hybrid_model,
        create_vocab_model,
        load_vocab,
        print_model_info_cmim,
        print_model_info_hybrid,
        print_model_info_vocab,
    )

    a = _args_for_model_build(stored_args)
    vo = vocab_overrides or {}
    sd = _maybe_strip_ddp_prefix(state_dict) if state_dict is not None else None
    fg_size = 85

    if mode == "hybrid":
        sp = resolve_vocab_path(
            "smiles_vocab_path",
            checkpoint_dir,
            getattr(a, "smiles_vocab_path", None),
            vo.get("smiles_vocab_path"),
            interactive,
        )
        ap = resolve_vocab_path(
            "atom_vocab_path",
            checkpoint_dir,
            getattr(a, "atom_vocab_path", None),
            vo.get("atom_vocab_path"),
            interactive,
        )
        bp = resolve_vocab_path(
            "bond_vocab_path",
            checkpoint_dir,
            getattr(a, "bond_vocab_path", None),
            vo.get("bond_vocab_path"),
            interactive,
        )
        smiles_vocab_size = load_vocab(sp)
        atom_vocab_size = load_vocab(ap)
        bond_vocab_size = load_vocab(bp)
        model, ma = create_hybrid_model(
            copy.copy(a), smiles_vocab_size, atom_vocab_size, bond_vocab_size, fg_size
        )
        if sd is not None:
            _report_state_dict_load(model.load_state_dict(sd, strict=False), "hybrid")
        print_model_info_hybrid(
            model, ma, smiles_vocab_size, atom_vocab_size, bond_vocab_size, fg_size
        )

    elif mode == "cmim":
        sp = resolve_vocab_path(
            "smiles_vocab_path",
            checkpoint_dir,
            getattr(a, "smiles_vocab_path", None),
            vo.get("smiles_vocab_path"),
            interactive,
        )
        smiles_vocab_size = load_vocab(sp)
        model, ma = create_cmim_model(copy.copy(a), smiles_vocab_size)
        if sd is not None:
            _report_state_dict_load(model.load_state_dict(sd, strict=False), "cmim")
        print_model_info_cmim(model, ma, smiles_vocab_size)

    else:
        ap = resolve_vocab_path(
            "atom_vocab_path",
            checkpoint_dir,
            getattr(a, "atom_vocab_path", None),
            vo.get("atom_vocab_path"),
            interactive,
        )
        bp = resolve_vocab_path(
            "bond_vocab_path",
            checkpoint_dir,
            getattr(a, "bond_vocab_path", None),
            vo.get("bond_vocab_path"),
            interactive,
        )
        atom_vocab_size = load_vocab(ap)
        bond_vocab_size = load_vocab(bp)
        model, ma = create_vocab_model(copy.copy(a), atom_vocab_size, bond_vocab_size, fg_size)
        if sd is not None:
            sd_load = OrderedDict(sd)
            if any(k.startswith("grover.") for k in sd_load.keys()):
                sd_load = OrderedDict(
                    (k.replace("grover", "kermt"), v) for k, v in sd_load.items()
                )
            _report_state_dict_load(model.load_state_dict(sd_load, strict=False), "vocab")
        print_model_info_vocab(model, ma, atom_vocab_size, bond_vocab_size, fg_size)


def _ensure_encoder_args_for_embedding_only(a):
    """Fill fields KERMTEmbedding may need when args come from a minimal grover_base checkpoint."""
    enc_defaults = {
        "num_attn_head": 4,
        "dropout": 0.1,
        "atom_message": False,
        "bond_drop_rate": 0.0,
        "nencoders": 3,
        "coord": 15,
        "input_layer": "fc",
        "no_attach_fea": False,
        "use_cuikmolmaker_featurization": False,
    }
    for key, val in enc_defaults.items():
        if not hasattr(a, key):
            setattr(a, key, val)


def _print_embedding_only_model_size(stored_args, state_dict):
    """Encoder-only checkpoint (e.g. grover_base.pt with grover.* keys)."""
    from kermt.model.models import KERMTEmbedding
    from task.helpers.kermt_model_size_report import print_model_info_embedding_only

    a = _args_for_model_build(stored_args)
    _ensure_encoder_args_for_embedding_only(a)
    a.embedding_output_type = "both"
    a.cuda = False

    sd = _maybe_strip_ddp_prefix(state_dict)
    sd = OrderedDict((k.replace("grover", "kermt"), v) for k, v in sd.items())
    if any("latent_dist." in k for k in sd.keys()):
        sd = OrderedDict((k.replace("latent_dist.", ""), v) for k, v in sd.items())

    encoder = KERMTEmbedding(a)
    encoder_sd = {}
    for key, value in sd.items():
        if key.startswith("kermt."):
            encoder_sd[key[len("kermt.") :]] = value

    if not encoder_sd:
        raise ValueError(
            "embedding_only: no kermt.* (or grover.*) weights after key transform; "
            f"sample keys: {list(sd.keys())[:8]}"
        )

    _report_state_dict_load(encoder.load_state_dict(encoder_sd, strict=False), "KERMTEmbedding")
    print_model_info_embedding_only(encoder, a)


def _print_finetune_model_size(stored_args, state_dict=None):
    from kermt.util.nn_utils import param_count_total, param_count_trainable
    from kermt.util.utils import build_model

    if isinstance(stored_args, dict):
        a = argparse.Namespace(**stored_args)
    else:
        a = copy.copy(stored_args)
    model = build_model(a)
    if state_dict is not None:
        sd = _maybe_strip_ddp_prefix(state_dict)
        _report_state_dict_load(model.load_state_dict(sd, strict=False), "finetune")
    total = param_count_total(model)
    trainable = param_count_trainable(model)
    param_size_mb = (total * 4) / (1024**2)
    print("\n" + "-" * 80)
    print("MODEL SIZE (finetune / fingerprint — rebuilt via build_model)")
    print("-" * 80)
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Approx. FP32 size:    {param_size_mb:.2f} MB")
    print("\n  Top-level module parameters:")
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        pct = 100.0 * n / total if total else 0.0
        print(f"    {name}: {n:,} ({pct:.1f}%)")


def _print_model_size_from_checkpoint(ckpt, checkpoint_path, vocab_overrides, interactive):
    stored = ckpt.get("args")
    if stored is None:
        raise ValueError("Checkpoint has no 'args'; cannot rebuild model for size breakdown.")

    state_dict = ckpt.get("state_dict")
    mode = getattr(stored, "pretrain_mode", None)
    if isinstance(stored, dict):
        mode = stored.get("pretrain_mode", mode)
    if mode not in ("vocab", "cmim", "hybrid", "embedding_only"):
        mode = _infer_pretrain_mode(state_dict)
    if mode not in ("vocab", "cmim", "hybrid", "embedding_only"):
        raise ValueError(
            "Could not determine pretrain_mode from args or state_dict keys; "
            "expected hybrid, cmim, vocab (vocab_module), or embedding_only (grover.*/kermt.* encoder only)."
        )

    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    vo = vocab_overrides or {}
    upgraded_encoder_to_vocab = False
    # Legacy GROVER encoder release: file has grover.* only. Former print_model_size.py --mode vocab
    # reported full KermtTask (encoder + vocab heads) using atom/bond pickle sizes — match that when
    # we can resolve those vocabs.
    if mode == "embedding_only":
        tmp_a = _args_for_model_build(stored)
        try:
            resolve_vocab_path(
                "atom_vocab_path",
                checkpoint_dir,
                getattr(tmp_a, "atom_vocab_path", None),
                vo.get("atom_vocab_path"),
                interactive,
            )
            resolve_vocab_path(
                "bond_vocab_path",
                checkpoint_dir,
                getattr(tmp_a, "bond_vocab_path", None),
                vo.get("bond_vocab_path"),
                interactive,
            )
            mode = "vocab"
            upgraded_encoder_to_vocab = True
        except FileNotFoundError:
            pass

    if not isinstance(stored, dict):
        stored_mode = getattr(stored, "pretrain_mode", None)
        if upgraded_encoder_to_vocab:
            print(
                "  [INFO] Encoder-only weights (legacy grover./kermt. keys); atom/bond vocabs resolved. "
                "Reporting full KermtTask size (encoder + vocab heads), same as former print_model_size.py --mode vocab."
            )
        elif stored_mode != mode:
            print(
                f"  [INFO] Inferred pretrain_mode={mode!r} from state_dict "
                f"(checkpoint args had {stored_mode!r})."
            )

    print("\n" + "-" * 80)
    print("MODEL SIZE")
    print("-" * 80)

    if mode in ("vocab", "cmim", "hybrid"):
        _print_pretrain_model_size(
            stored, state_dict, checkpoint_dir, vocab_overrides, interactive, mode
        )
        return

    if mode == "embedding_only":
        _print_embedding_only_model_size(stored, state_dict)
        return

    try:
        _print_finetune_model_size(stored, state_dict=state_dict)
    except Exception as finetune_exc:
        print(f"  [WARN] Could not rebuild finetune model for size report: {finetune_exc}")
        if state_dict:
            sd = _maybe_strip_ddp_prefix(state_dict)
            n_tensors = len(sd)
            elems = sum(v.numel() for v in sd.values())
            nbytes = sum(v.numel() * v.element_size() for v in sd.values())
            print("  (Falling back to raw state_dict counts.)")
            print(f"    param tensors:   {n_tensors}")
            print(f"    total elements:  {elems:,}")
            print(f"    total size:      {format_size(nbytes)}")


def inspect_checkpoint(
    path,
    show_args=False,
    show_optimizer=False,
    show_model_keys=False,
    show_model_size=False,
    vocab_overrides=None,
    interactive_vocab=True,
):
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    file_size = os.path.getsize(path)
    file_mtime = datetime.fromtimestamp(os.path.getmtime(path))

    print("=" * 80)
    print("CHECKPOINT INSPECTION")
    print("=" * 80)
    print(f"  Path:          {os.path.abspath(path)}")
    print(f"  File size:     {format_size(file_size)}")
    print(f"  Last modified: {file_mtime.strftime('%Y-%m-%d %H:%M:%S')}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    print(f"\n  Top-level keys: {sorted(ckpt.keys())}")

    print("\n" + "-" * 80)
    print("TRAINING PROGRESS")
    print("-" * 80)

    epoch = ckpt.get("epoch", "NOT FOUND")
    print(f"  epoch:          {epoch}")

    scheduler_step = ckpt.get("scheduler_step", None)
    batch_idx = ckpt.get("batch_idx", None)
    if scheduler_step is not None:
        print(f"  scheduler_step: {scheduler_step}")
    if batch_idx is not None:
        print(f"  batch_idx:      {batch_idx}")

    if "scheduler" in ckpt:
        sched_state = ckpt["scheduler"]
        if isinstance(sched_state, dict):
            print(f"  scheduler keys: {sorted(sched_state.keys())}")
            if "current_step" in sched_state:
                print(f"  scheduler current_step: {sched_state['current_step']}")
        else:
            print(f"  scheduler:      (type={type(sched_state).__name__})")

    has_optimizer = "optimizer" in ckpt
    print(f"  optimizer:      {'present' if has_optimizer else 'NOT FOUND'}")

    has_state_dict = "state_dict" in ckpt
    print(f"  state_dict:     {'present' if has_state_dict else 'NOT FOUND'}")
    if has_state_dict:
        sd = ckpt["state_dict"]
        n_params = len(sd)
        total_elements = sum(v.numel() for v in sd.values())
        total_bytes = sum(v.numel() * v.element_size() for v in sd.values())
        print(f"    param tensors:   {n_params}")
        print(f"    total elements:  {total_elements:,}")
        print(f"    total size:      {format_size(total_bytes)}")

    data_scaler = ckpt.get("data_scaler", None)
    features_scaler = ckpt.get("features_scaler", None)
    print(f"  data_scaler:    {'present' if data_scaler else 'None'}")
    print(f"  features_scaler:{'present' if features_scaler else 'None'}")

    args_dict = {}
    args = ckpt.get("args", None)
    if args is not None:
        print("\n" + "-" * 80)
        print("TRAINING ARGS (key fields)")
        print("-" * 80)

        key_fields = [
            "pretrain_mode",
            "epochs",
            "batch_size",
            "init_lr",
            "max_lr",
            "final_lr",
            "warmup_epochs",
            "hidden_size",
            "depth",
            "num_heads",
            "num_mt_block",
            "ffn_hidden_size",
            "ffn_num_layers",
            "dropout",
            "activation",
            "weight_decay",
            "fine_tune_coff",
            "latent_dim",
            "contrastive_temperature",
            "decoder_hidden_size",
            "decoder_num_layers",
            "decoder_num_heads",
            "decoder_max_seq_len",
            "decoder_dropout",
            "vocab_loss_weight",
            "decoder_gate_self_attn",
            "self_attention",
            "attn_hidden",
            "attn_out",
            "ffn_num_task_specific_layers",
            "ffn_task_specific_hidden_size",
            "use_mtl_loss",
            "task_wise_checkpoint",
            "dataset_type",
            "metric",
            "num_tasks",
            "task_names",
            "seed",
            "save_dir",
            "train_data_path",
            "val_data_path",
            "data_path",
            "smiles_vocab_path",
            "atom_vocab_path",
            "bond_vocab_path",
        ]

        if hasattr(args, "__dict__"):
            args_dict = vars(args)
        elif isinstance(args, dict):
            args_dict = args
        else:
            args_dict = {}
            print(f"  (args type: {type(args).__name__}, cannot inspect)")

        for field in key_fields:
            if field in args_dict:
                val = args_dict[field]
                if isinstance(val, list) and len(val) > 10:
                    print(f"  {field:<35} = [{val[0]}, ..., {val[-1]}] (len={len(val)})")
                else:
                    print(f"  {field:<35} = {val}")

        if show_args:
            print("\n  --- Full args ---")
            for k, v in sorted(args_dict.items()):
                if isinstance(v, list) and len(v) > 10:
                    print(f"  {k:<40} = [{v[0]}, ..., {v[-1]}] (len={len(v)})")
                else:
                    print(f"  {k:<40} = {v}")

        total_epochs = args_dict.get("epochs", None)
        if total_epochs is not None and isinstance(epoch, (int, float)):
            pct = epoch / total_epochs * 100 if total_epochs > 0 else 0
            print(f"\n  Progress: epoch {epoch} / {total_epochs} ({pct:.1f}%)")

    if show_optimizer and has_optimizer:
        print("\n" + "-" * 80)
        print("OPTIMIZER STATE")
        print("-" * 80)
        opt = ckpt["optimizer"]
        if isinstance(opt, dict):
            print(f"  param_groups: {len(opt.get('param_groups', []))}")
            for i, pg in enumerate(opt.get("param_groups", [])):
                pg_info = {k: v for k, v in pg.items() if k != "params"}
                print(f"    group[{i}]: {pg_info}")
                print(f"      num params: {len(pg.get('params', []))}")

    if show_model_keys and has_state_dict:
        print("\n" + "-" * 80)
        print("MODEL STATE_DICT KEYS")
        print("-" * 80)
        for k, v in sorted(ckpt["state_dict"].items()):
            print(f"  {k:<60} {str(list(v.shape)):>25} {v.dtype}")

    if show_model_size:
        try:
            _print_model_size_from_checkpoint(
                ckpt, path, vocab_overrides or {}, interactive_vocab
            )
        except Exception as exc:
            print(f"  [ERROR] Model size section failed: {exc}")

    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    if isinstance(epoch, (int, float)):
        total_epochs = args_dict.get("epochs", None)
        if total_epochs is not None:
            if epoch >= total_epochs:
                print("  [OK] Training appears COMPLETE (epoch >= total epochs)")
            else:
                print(f"  [INFO] Training IN PROGRESS: epoch {epoch} / {total_epochs}")
        else:
            print(f"  [INFO] Current epoch: {epoch} (total epochs unknown)")
    else:
        print("  [WARN] epoch not found in checkpoint")

    if scheduler_step is not None:
        print(f"  [INFO] Scheduler has taken {scheduler_step} steps")
    if batch_idx is not None:
        if batch_idx == 0:
            print(f"  [INFO] batch_idx=0 → saved at epoch boundary (not mid-epoch)")
        else:
            print(f"  [INFO] batch_idx={batch_idx} → saved mid-epoch")

    if not has_optimizer:
        print("  [WARN] No optimizer state → this is a best-model checkpoint (not resumable)")
    else:
        print("  [OK] Optimizer state present → checkpoint is resumable")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a pretrain or finetune checkpoint .pt file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("checkpoint", help="Path to .pt checkpoint file")
    parser.add_argument(
        "--show_args", action="store_true", help="Print all training args (not just key fields)"
    )
    parser.add_argument(
        "--show_optimizer", action="store_true", help="Print optimizer param group details"
    )
    parser.add_argument(
        "--show_model_keys",
        action="store_true",
        help="Print all model state_dict keys and shapes",
    )
    parser.add_argument(
        "--show_model_size",
        action="store_true",
        help="Rebuild model from checkpoint args; print parameter report (task/helpers/kermt_model_size_report.py)",
    )
    parser.add_argument(
        "--non-interactive",
        "--non_interactive",
        dest="non_interactive",
        action="store_true",
        help="Do not prompt for missing vocab paths (fail if vocabs not beside checkpoint / CLI)",
    )
    parser.add_argument(
        "--smiles_vocab_path",
        type=str,
        default=None,
        help="Override SMILES vocab pickle for --show_model_size",
    )
    parser.add_argument(
        "--atom_vocab_path",
        type=str,
        default=None,
        help="Override atom vocab pickle for --show_model_size",
    )
    parser.add_argument(
        "--bond_vocab_path",
        type=str,
        default=None,
        help="Override bond vocab pickle for --show_model_size",
    )

    args = parser.parse_args()
    vocab_overrides = {
        k: v
        for k, v in (
            ("smiles_vocab_path", args.smiles_vocab_path),
            ("atom_vocab_path", args.atom_vocab_path),
            ("bond_vocab_path", args.bond_vocab_path),
        )
        if v
    }
    inspect_checkpoint(
        args.checkpoint,
        args.show_args,
        args.show_optimizer,
        args.show_model_keys,
        show_model_size=args.show_model_size,
        vocab_overrides=vocab_overrides or None,
        interactive_vocab=not args.non_interactive,
    )


if __name__ == "__main__":
    main()
