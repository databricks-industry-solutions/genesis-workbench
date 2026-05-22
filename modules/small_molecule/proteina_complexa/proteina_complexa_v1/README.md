# Proteina-Complexa (NVIDIA) — Three serving endpoints

**Proteina-Complexa** is NVIDIA's protein structure-generation stack (MIT-licensed). GWB ships three independent MLflow models from the same source tree, one per Proteina checkpoint:

| GWB UC model | Checkpoint | Capability | Used by (app feature) |
|---|---|---|---|
| `proteina_complexa` | `complexa.ckpt` | Protein-protein binder design | Protein Binder Design tab |
| `proteina_complexa_ligand` | `complexa_ligand.ckpt` | Small-molecule ligand binder design | Ligand Binder Design tab |
| `proteina_complexa_ame` | `complexa_ame.ckpt` | Motif scaffolding with ligand context | Guided Enzyme Optimization workflow (Fast mode endpoint + Accurate mode in-process) |

All three are registered by the same notebook (`notebooks/01_register_proteina_complexa.py`) in a single `for variant_key, model_info in MODELS.items()` loop, sharing one source clone, one `code_paths` set, and one `conda_env` spec. They differ only by `python_model` class and `signature`.

Source: <https://github.com/NVIDIA-Digital-Bio/Proteina-Complexa>

## License posture — GPL-clean

The registration notebook pins Proteina source to **NVIDIA's `remove_openbabel` branch HEAD** (`f95f2d4bbcebcad0613b89a0012edec8637a6334`). This commit:

- Changes `env/build_uv_env.sh`: `atomworks[ml,openbabel,dev]` → `atomworks[ml,dev]` (drops the openbabel extra)
- In `src/proteinfoundation/datasets/atomworks_ligand_transforms.py`: removes the top-level `from atomworks.ml.transforms.openbabel_utils import atom_array_from_openbabel, atom_array_to_openbabel`, and replaces the `if use_openbabel:` codepath with `NotImplementedError`. The `use_rdkit_from_smiles` and `use_bonds_from_file` codepaths are untouched.

`openbabel-wheel` is also absent from both pip-install lines in the registration notebook (driver-side install at line ~144 and conda_env at line ~700).

**Why pin a SHA instead of tracking the branch:** if NVIDIA force-pushes / rebases / merges the branch back into `main`, the GWB build would silently start picking up new commits. SHA pinning keeps deploys deterministic. To intentionally bump, first verify the target commit hasn't re-introduced the `from openbabel ...` top-level import (`gh api repos/NVIDIA-Digital-Bio/Proteina-Complexa/contents/src/proteinfoundation/datasets/atomworks_ligand_transforms.py?ref=<sha>`).

**Why GWB's invocations don't hit the `NotImplementedError`:** GWB's wrappers always pass `use_bonds_from_file=True` to `LigandFeatures` (see `notebooks/01_register_proteina_complexa.py:539` and `:607`). Neither variant — nor any other GWB code path — sets `use_openbabel=True`. The protein-protein binder variant doesn't construct `LigandFeatures` at all. If a future code change starts passing `use_openbabel=True`, the upstream commit will surface the `NotImplementedError` at predict time as a forcing function to re-evaluate.

Full debug arc + verification timeline in [`CHANGELOG.md`](../../../../CHANGELOG.md) under "proteina_no_openbabel (2026-05-22)".

## Compute

- **Endpoint serving (Fast mode in the Enzyme Optimization workflow, default for all three variants):** GPU_MEDIUM (A10), single replica. Auto-scales to 0 when idle.
- **In-process Accurate mode** (only the AME variant, only in Enzyme Optimization workflow when the user picks "Accurate"): a fresh A10 cluster spun up per job run. AME is loaded into the orchestrator's own Python process to unlock Proteina's Feynman-Kac steering hook (which requires direct callback access into the diffusion sampler — not expressible through the standard MLflow serving interface).

## Checkpoints

Pulled from NGC at registration time. ~3 GB per checkpoint plus a smaller `*_ae.ckpt` for each variant. Cached in `/Volumes/{catalog}/{schema}/{cache_dir}/proteina_complexa/<variant_key>/` so re-registrations don't re-download. The cache directory and NGC IDs are configured via the `MODELS` dict at line 166 of the registration notebook.

## Destroy semantics

`destroy.sh` for this submodule removes the three UC models and three serving endpoints. Checkpoints in the Volume are intentionally NOT deleted by the bundle destroy — they're seven gigabytes of slow downloads and re-pulling them every redeploy isn't worth it. Clean them manually via `databricks fs rm -r dbfs:/Volumes/.../proteina_complexa/` when the catalog/schema is being torn down for good.
