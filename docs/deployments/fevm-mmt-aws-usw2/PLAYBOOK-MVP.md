# Genesis Workbench — SA HUNTER Playbook (MVP)

**Audience:** internal SA HUNTER team
**Branch:** `version_pinning` (pending merge to main/development)
**Live demo targets:**
- **New release (version_pinning):** https://genesis-workbench-1602460480284688.aws.databricksapps.com (fe-vm-hls-amer, deployed by Srijit 2026-04-20)
- **Previous/basic demo:** https://gwb-mmt-app-1602460480284688.aws.databricksapps.com (fe-vm-hls-amer, unchanged)
- **Alternate/public:** https://genesis-workbench-7474660003062283.aws.databricksapps.com

---

## TL;DR

Genesis Workbench (GWB) is an open-source Databricks application stack for life-sciences AI — protein, single-cell, small molecule, and disease biology workflows on GPU-backed clusters + serving endpoints, with a Streamlit UI. The `version_pinning` release adds **entirely new modules** (Small Molecule, Disease Biology) and **new workflows** on existing modules (Sequence Search, Perturbation Prediction), plus stability fixes across the board.

For SAs: this is a **turnkey demo asset for HLS customers** — deploys via `./deploy.sh`, ships with curated models (ESMFold, AlphaFold, scGPT, SCimilarity, Chemprop, DiffDock, etc.), and exposes them through a consistent UI that makes workflows demonstrable without custom dev.

---

## Part 1 — Existing capabilities (already familiar to most SAs)

### Protein Studies
- **Structure Prediction** — ESMFold (real-time), AlphaFold2 (batch), Boltz (real-time, multi-chain)
- **Protein Design** — ESMFold → RFDiffusion inpainting → ProteinMPNN → ESMFold validation
- **Inverse Folding** — ProteinMPNN: paste PDB backbone → get designed sequences → auto-fold validation

### Single Cell
- **scGPT** — Gene embeddings, zero-shot perturbation prediction
- **SCimilarity** — Cell type annotation + similarity search against 23M-cell reference
- **Scanpy / Rapids-SingleCell** — Raw processing (QC → normalize → HVG → PCA → cluster → UMAP → markers); Rapids is GPU-accelerated

### NVIDIA integrations
- **BioNeMo** — Enterprise biological AI container (ESM2 fine-tuning shipped)
- **Parabricks** — GPU-accelerated genomics (alignment, variant calling)

---

## Part 2 — What's new on `version_pinning`

### New modules (entirely new surface area)

**Small Molecule** — drug discovery workflows
- **Chemprop** — Molecular property prediction (BBBP, ClinTox, ADMET)
- **DiffDock** — Molecular docking via diffusion
- **Proteina-Complexa** — Protein binder design (protein-protein, ligand, motif scaffolding)
- **Open Babel** — Chemical format conversion / sanitization

**Disease Biology** — genomics pipelines
- **VCF Ingestion** — VCF-to-Delta via Glow
- **Variant Annotation** — ClinVar annotation with gene filtering (e.g., BRCA)
- **GWAS Analysis** — Genome-wide association studies pipeline (uses Parabricks for alignment)

### New workflows (on existing modules)

- **Sequence Search** (Protein) — ESM2 embedding → Vector Search (top 500) → Smith-Waterman alignment → ranked results
- **ESM2 Embeddings** (Protein) — 1280-D sequence embeddings for similarity search
- **Perturbation Prediction** (Single Cell) — scGPT zero-shot gene knockout/overexpression prediction

### Under the hood

- **Version pinning** across torchvision, tensorflow, statsmodels, transformers — eliminates a class of "works on my machine" failures
- **AWS attribute hard-coding removed** — deploy.sh is more portable across cloud setups
- **AlphaFold params download fix** — HTTPS + figshare instead of FTP/rsync (fixes VPC-blocked downloads)
- **Claude skills** shipped in-repo (`claude_skills/`) — DEPLOY_WIZARD, WORKFLOWS, TROUBLESHOOTING, INSTALLATION, DEVELOPMENT — lets Claude/Cursor guide SAs through setup

---

## Part 3 — Flagship demo flows

Pick 2-3 based on audience familiarity. Each should fit 3-5 minutes.

### Demo A — Protein Studies: end-to-end structure → design → validate

1. **Structure Prediction** tab → paste sequence → select ESMFold → ~5 seconds → 3D structure in Mol\* viewer
2. **Protein Design** tab → same sequence with `[bracket]` marking a region to redesign → run → shows ESMFold + RFDiffusion + ProteinMPNN + ESMFold-validation pipeline with overlaid structures
3. Talking point: "one UI, three deploy-hardened models pipelined for you"

### Demo B — Single Cell: from raw h5ad to annotated clusters to perturbation prediction

1. **Raw Single Cell Processing** → pick an h5ad from Volumes → Scanpy pipeline → UMAP with clusters
2. **Cell Type Annotation** → same run → SCimilarity annotates clusters against 23M-cell reference
3. **Perturbation Prediction** → pick a cluster → select genes → Knockout → bar chart of top-20 affected genes
4. Talking point: "takes hours of bespoke analysis into a UI-driven pipeline; runs on your Databricks GPU compute"

### Demo C — Small Molecule: ADMET + binder design (NEW)

1. **ADMET & Safety** → paste SMILES → parallel Chemprop endpoints (BBBP, ClinTox, ADMET) → red/orange/green risk cards
2. **Binder Design** → target protein (sequence or PDB) → Proteina-Complexa → binder candidates with ESMFold validation
3. Talking point: "this is new — GWB now covers the small-molecule side of drug discovery, not just proteins"

### Demo D — Disease Biology: VCF → variants → GWAS (NEW)

1. **VCF Ingestion** → VCF from Volumes → Delta table via Glow
2. **Variant Annotation** → Delta table → ClinVar annotation, filter by BRCA
3. **GWAS Analysis** → genotype + phenotype → Manhattan + QQ plot
4. Talking point: "genomics customers can now go VCF-to-insight inside GWB; GWAS pipeline uses Parabricks under the hood for alignment"

---

## Part 4 — Known gaps & caveats

**Pre-deploy gotchas (for SAs standing this up for customers):**

| Gotcha | Detail |
|---|---|
| Databricks CLI ≥ 0.295 required | `modules/core/databricks.yml` pins it; `brew upgrade databricks` |
| Terraform needed locally | Workaround for expired HashiCorp PGP key. `brew tap hashicorp/tap && brew install hashicorp/tap/terraform` (plain `brew install terraform` fails since license change) |
| Hardcoded Terraform path in deploy.sh | `/opt/homebrew/bin/terraform` + version pin 1.3.9. Portable fix queued as improvement PR. |
| Docker-registry creds required for 3 modules | BioNeMo, Parabricks, **AND Disease Biology** (reuses Parabricks). Use secret-scope refs `{{secrets/<scope>/<key>}}` rather than plaintext in module.env |
| DiffDock's `local/` Dockerfiles | Dev-only sanity check. Not needed for deploy. Easy to misread. |
| Long-running registration jobs | AlphaFold, Parabricks, BioNeMo can run for hours. Use "Start All Endpoints" in Settings pre-demo to avoid scale-from-zero delays. |

**Runtime caveats to mention in demos:**

- **Endpoint scale-to-zero** — first request after idle can take 60-90s. Warm up via Settings → Start All Endpoints (1-12h keep-alive) before the session.
- **Prereqs between workflows** — Cell Type Annotation requires a completed Raw Processing run; Perturbation Prediction requires both a processing run AND a live scGPT Perturbation endpoint.
- **Input format details** — Boltz multi-chain uses `protein_A:SEQ1;rna_B:SEQ2;smiles_C:CCCC`; Protein Design uses `CASRRSG[FTYPGF]FFEQYF` bracket convention.
- **Many parameter knobs** with sparse in-UI guidance — Scanpy/Rapids pipelines have min_genes, min_cells, pct_mt, n_genes_by_counts, target_sum, n_top_genes, n_pcs, resolution. Defaults are sensible for most datasets; flag when you'd deviate.

See also: `docs/deployments/fevm-mmt-aws-usw2/UX-GAPS.md` for detailed improvement-PR-candidates discovered during this deploy prep.

---

## Part 5 — For SAs who want to deploy themselves

1. Fork/clone https://github.com/databricks-industry-solutions/genesis-workbench
2. Check out `version_pinning` (until merged to main)
3. Follow `claude_skills/SKILL_GENESIS_WORKBENCH_DEPLOY_WIZARD.md` — Claude/Cursor can drive the setup conversationally
4. Or run `./deploy.sh core <cloud>` manually after filling in `application.env` + `modules/core/module.env`
5. Deploy modules **one at a time**, waiting for each first-post-deploy job to reach `RUNNING` before launching the next (GPU quota serialization)

Full deploy runbook lives at `docs/deployments/fevm-mmt-aws-usw2/SESSION-NOTES.md` (sandbox-specific but portable).

---

## Appendix — References

- Main skill files (in-repo):
  - `claude_skills/SKILL_GENESIS_WORKBENCH.md` — module overview
  - `claude_skills/SKILL_GENESIS_WORKBENCH_WORKFLOWS.md` — per-workflow UI guide
  - `claude_skills/SKILL_GENESIS_WORKBENCH_TROUBLESHOOTING.md` — failure recipes
  - `claude_skills/SKILL_GENESIS_WORKBENCH_DEPLOY_WIZARD.md` — interactive deploy guide
  - `claude_skills/SKILL_GENESIS_WORKBENCH_INSTALLATION.md` — reference install
- Local SA notes:
  - `docs/deployments/fevm-mmt-aws-usw2/SESSION-NOTES.md` — sandbox deploy runbook
  - `docs/deployments/fevm-mmt-aws-usw2/UX-GAPS.md` — running UX + deploy gap log
- Existing fe-vm-hls-amer deployment docs (prior release):
  - `docs/deployments/fe-vm-hls-amer/genesis-workbench-redeploy.md`
  - `docs/deployments/fe-vm-hls-amer/alphafold-debug-summary.md`
  - `docs/deployments/fe-vm-hls-amer/scanpy-gene-mapping-notes.md`

---

*Last updated: 2026-04-21. MVP draft — review, edit, or expand as you walk through the live deployment.*
