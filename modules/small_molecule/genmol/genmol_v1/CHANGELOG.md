# GenMol v1 Changelog

## 2026-06-05 — Initial scaffold

Adds **GenMol** (NVIDIA `NV-GenMol-89M-v2`) as a small-molecule **generator** —
the missing "where do candidate ligands come from?" step in the small-molecule
arc. GenMol generates candidates → DiffDock docks them into the target →
KERMT/Chemprop profile ADMET.

### What this module does
- Downloads the open weights `nvidia/NV-GenMol-89M-v2` from Hugging Face into the
  `genmol` cache volume (no BioNeMo runtime / NIM required — same pattern as
  chemprop, netsolp, etc.).
- Wraps generation in an MLflow PyFunc: input is a seed fragment (empty = de novo;
  SMILES/SAFE = scaffold decoration / growing), params control count / temperature
  / randomness / diffusion steps / scoring; output is ranked generated SMILES.
- Registers to Unity Catalog, imports into Genesis Workbench as `SMALL_MOLECULE`,
  and deploys a `GPU_SMALL` serving endpoint (89M params — light).

### Licensing
- **Weights:** NVIDIA Open Model License — commercial use permitted; **not** for
  life-critical use cases. Surface a research-use disclaimer in the UI.
- **Code:** Apache-2.0 (GenMol package + SAFE library).

## 2026-06-05 — Validated API + py3.11 runtime pivot

Validated the real GenMol API on a classic DBR 15.4 LTS (Python 3.11) cluster via
a series of probe jobs, and finalized the register notebook + job YAML accordingly.

**Why py3.11 (not serverless):** GenMol hard-pins `pandas==2.1.0` and
`transformers==4.52.4`. On serverless (Python 3.12) `pandas==2.1.0` has no wheel and
its C source build fails (`PyArray_Descr has no member 'c_metadata'`). On py3.11 both
pins have wheels and install cleanly. `register_genmol.yml` now uses a **classic
single-node cluster on `15.4.x-scala2.12`** instead of a serverless environment —
which also makes the Model Serving endpoint inherit py3.11 so the same pins resolve
at endpoint-build time.

**Real API (commit `add09fc8…`, pinned):**
- `genmol.sampler.Sampler(path)` — positional checkpoint path; loads via Lightning.
- `de_novo_generation(num_samples, softmax_temp=0.8, randomness=0.5, min_add_len=40)`
- `fragment_completion(fragment, num_samples, apply_filter=True, softmax_temp=1.2, randomness=2, gamma=0)`
- Returns a **list of canonical SMILES strings** (not SAFE) — wrapper just
  RDKit-canonicalizes + scores (QED/LogP).

**`data/len.pk`:** GenMol reads it from `dirname×3(sampler.py)`; present in the repo
but not the pip wheel. The notebook downloads it (commit-pinned) and ships it as a
model artifact; the PyFunc copies it back to that path in `load_context`.

**Serving caveat:** `pip_requirements` uses the `git+…@commit` VCS install, which
needs git in the Model Serving build image (available today). Fallback if a future
image lacks git: log the built `genmol` wheel as an artifact instead.

### Not yet wired (follow-on)
- Backend router endpoint + a **Small-Molecule Generation** frontend tab
  (generate → one-click into DiffDock / ADMET). This scaffold deploys the model
  + endpoint; the app-facing tab is the next step.
