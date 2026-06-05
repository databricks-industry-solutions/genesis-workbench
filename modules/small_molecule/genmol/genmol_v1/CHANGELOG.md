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

### Confirm-on-deploy
GenMol ships as command-line scripts, not a documented inline Python API. The
PyFunc calls `genmol.sampler.Sampler` (`de_novo_generation` / `fragment_completion`).
Verify these class/method names against the pinned GenMol commit on first deploy
and adjust `GenMolGenerator._generate` if upstream differs. The rest of the
module (HF download, SAFE<->SMILES, scoring, MLflow signature, GWB registration,
endpoint deploy) is stable.

### Not yet wired (follow-on)
- Backend router endpoint + a **Small-Molecule Generation** frontend tab
  (generate → one-click into DiffDock / ADMET). This scaffold deploys the model
  + endpoint; the app-facing tab is the next step.
