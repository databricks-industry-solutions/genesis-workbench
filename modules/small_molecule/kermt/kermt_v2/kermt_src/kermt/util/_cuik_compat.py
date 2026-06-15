# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility shim for the cuik_molmaker feature-name API rename.

cuik_molmaker renamed the `*_feature_names_to_array` family to
`*_feature_names_to_tensor` after 0.2. The kermt:latest docker image ships
the newer name; the host conda env (admet_dev_py311) still pins 0.2 with
the older name. The downstream consumers (`mol_featurizer`,
`batch_mol_featurizer`) accept whichever container type the matching
helper returns, so the call sites only care about which symbol exists.

Import the three names from here instead of from cuik_molmaker directly:

    from kermt.util._cuik_compat import (
        atom_onehot_feature_names_to_tensor,
        atom_float_feature_names_to_tensor,
        bond_feature_names_to_tensor,
    )

Each resolves at module-load time and raises AttributeError with a clear
message if neither variant is present.
"""
from __future__ import annotations

# GWB lazy-import patch: cuik_molmaker is conda-only (rmg channel) and not needed for the
# RDKit featurization path; guard so the module imports on a pip-only env without it.
try:
    import cuik_molmaker
except Exception:
    cuik_molmaker = None
import torch


def _resolve(new_name: str, old_name: str):
    if cuik_molmaker is None:
        return None
    fn = getattr(cuik_molmaker, new_name, None)
    if fn is not None:
        return fn
    fn = getattr(cuik_molmaker, old_name, None)
    if fn is not None:
        return fn
    raise AttributeError(
        f"cuik_molmaker has neither {new_name!r} nor {old_name!r}; "
        "upgrade or downgrade the package."
    )


atom_onehot_feature_names_to_tensor = _resolve(
    "atom_onehot_feature_names_to_tensor",
    "atom_onehot_feature_names_to_array",
)
atom_float_feature_names_to_tensor = _resolve(
    "atom_float_feature_names_to_tensor",
    "atom_float_feature_names_to_array",
)
bond_feature_names_to_tensor = _resolve(
    "bond_feature_names_to_tensor",
    "bond_feature_names_to_array",
)


def to_float_tensor(x) -> torch.Tensor:
    """Coerce a cuik_molmaker featurizer output to a float32 torch.Tensor.

    The pre-0.2 `cuik_molmaker.batch_mol_featurizer` returned numpy arrays;
    the post-0.2 version returns torch.Tensors directly. This helper handles
    both without an isinstance dispatch at every call site.
    """
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.from_numpy(x).float()


# Empty float tensor used as a placeholder for unused feature slots in
# `cuik_molmaker.mol_featurizer` calls. The post-0.2 binding rejects
# `np.array([])` for these slots; an empty torch tensor works for both
# variants.
EMPTY_FLOAT_TENSOR = torch.tensor([], dtype=torch.float64)
