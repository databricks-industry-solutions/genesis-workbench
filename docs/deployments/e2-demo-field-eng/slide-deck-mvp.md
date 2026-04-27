# GWB Customer Intro — MVP Slide Deck (v0)

**Status:** v0 MVP draft 2026-04-26 — get to "showable" by tmr morning. Visuals lean on existing `gwb_app_imgs/` screenshots from initial setup; **expect to refresh during/after live clickthrough testing** (per `testing-log-2026-04-26.md`).

**Source script:** `podcast-script-customer-intro.md` (~12 min, two-host)
**Length:** 13 slides for ~12 min of speaking (~55 sec/slide)
**Audience:** LS customer SAs / data leaders / R&D engineering managers — tech-fluent, not deep computational biology
**Format suggestion:** import to Google Slides or Keynote — bullets per slide, screenshots as full-bleed images on most module slides

Each slide below shows: **Title** / **Visual** / **Bullets/talking points** / **Speaker notes** (mapped to script).

---

## Slide 1 — Title

**Title:** Genesis Workbench
**Subtitle:** AI-Native Drug Discovery on Databricks — open-source models, governed enterprise data, one unified workbench
**Visual:** `gwb_app_imgs/Home_UpdatedAppDeploy_0.png` (full-bleed; the actual app's home screen)
**Footer:** Presenter / Date / Internal-only or customer name
**Speaker notes:**
- Opening line: "Today I'm going to walk you through the Databricks Genesis Workbench — an open-source blueprint that bundles state-of-the-art AI for drug discovery into one Databricks-native stack."

---

## Slide 2 — The R&D bottleneck

**Title:** Drug discovery is the most expensive engineering problem in biology
**Visual:** No screenshot — clean stat slide. Three large numbers:
- **$2.6B** per approved drug (Tufts CSDD)
- **10–15 yr** median timeline
- **~90%** Phase 2 attrition
**Speaker notes (from script opening):**
- Traditional drug discovery costs ~$2.6B per approved drug, takes over a decade, mostly in physical wet labs.
- Genesis Workbench moves much of this into the digital "in silico" world.
- Key win: proprietary enterprise data and open-source AI models live in the same governed system.

---

## Slide 3 — The drug discovery pipeline (4 stages)

**Title:** Four stages, one workbench
**Visual:** Pipeline diagram (build from scratch — boxes left-to-right):
```
Identify disease → Find target → Understand 3D shape → Design drug → Preclinical
```
**Bullets:**
- Each stage maps directly to a Genesis Workbench module
- All stages share Unity Catalog governance + MLflow lineage + GPU-aware compute
**Speaker notes:**
- "Drug discovery follows a specific path. Each Genesis Workbench module maps to a stage."

---

## Slide 4 — Modules 1 & 2: Find the target

**Title:** From population genomics to single-cell biology
**Visual:** Two-column layout:
- Left: `gwb_app_imgs/DiseaseBiology_VariantAnnotate_table_0.png`
- Right: any single-cell screenshot if available; OR placeholder note "single-cell tab — refresh after testing"
**Bullets:**
- **Disease Biology** *(newer addition)* — VCF ingestion, variant annotation against ClinVar, GWAS at population scale
- **Single Cell** — scGPT (gene-knockout prediction), SCimilarity (search 23M annotated cells), Scanpy + RAPIDS pipelines
**Speaker notes:**
- Disease Biology: "fruit smoothie" metaphor doesn't apply here — this is the population-genomics entry point.
- Single Cell: traditional sequencing = fruit smoothie. Single-cell = analyzing each cell. scGPT predicts what happens if you knock out a gene — huge for target validation.

---

## Slide 5 — Module 3: Understand the target (Protein Studies)

**Title:** From sequence to 3D structure to designed protein
**Visual:** Two screenshots side-by-side if available, else `gwb_app_imgs/Home_UpdatedAppDeploy_0.png` cropped to Protein Studies tab; **TODO during testing: capture Mol* viewer with ESMFold output**
**Bullets:**
- **Structure prediction:** ESMFold (real-time, ~5s per sequence — Meta), AlphaFold2 (batch, higher accuracy), Boltz (multi-chain complexes)
- **Protein design:** RFDiffusion (generate new backbones from scratch) + ProteinMPNN (inverse folding — backbone → sequence)
**Speaker notes:**
- Live deployment serves ESMFold; AlphaFold2 is for batch use only.
- "If you understand the lock, you can design the key."

---

## Slide 6 — Module 4: Design the drug (Small Molecule) — NEW

**Title:** Virtual screening: millions of compounds, computer-only
**Visual:** Chemprop or DiffDock screenshot from testing; **TODO: capture during clickthrough** (Chemprop ADMET cards or DiffDock binding visualization)
**Bullets:**
- **Chemprop** (3 endpoints) — BBBP / ClinTox / ADMET property prediction = virtual safety screen
- **DiffDock** — diffusion-based protein-ligand docking
- **Proteina-Complexa** — 3D folding for protein-ligand complexes
**Speaker notes:**
- Recently added module — opens up small-molecule discovery in addition to protein design.
- Scale impact: 5-year medchem program → weeks of cluster compute.

---

## Slide 7 — Module 5: NVIDIA acceleration (Parabricks + BioNeMo)

**Title:** GPU-accelerated genomics + protein language models on your data
**Visual:** `gwb_app_imgs/BioNeMo_esm2_finetune_0.png` (the recent fix that pre-populates fine-tune defaults from BLAT_ECOLX init data)
**Bullets:**
- **Parabricks** — variant calling: 24 hr CPU → <1 hr on a single A100
- **BioNeMo** — TWO surfaces:
  - ESM2 *inference* — protein embeddings from sequences
  - ESM2 *fine-tuning* — train a 650M-param protein LM on YOUR proprietary sequences
- "Bring the model to your data" rather than the reverse — critical for pharma compliance
**Speaker notes:**
- Parabricks deserves its own beat — clinical pipelines processing hundreds of genomes/day.
- BioNeMo's enterprise unlock is fine-tuning on internal sequences that can't go to a public API.

---

## Slide 8 — Disease Biology deeper dive — NEW

**Title:** End-to-end genomics workflow
**Visual:** `gwb_app_imgs/DiseaseBiology_GWAS_vcfrun_0.png` and `DiseaseBiology_VariantAnnotate_fromIngest_0.png` side-by-side, OR pick the strongest single image
**Bullets:**
- VCF ingestion → Delta tables (Glow)
- Variant annotation against ClinVar; filter by gene (e.g., BRCA)
- GWAS pipeline (uses Parabricks under the hood for alignment)
**Speaker notes:**
- Recently added module, deploys today added 3 sub-workflows
- This is the "from raw VCF to publishable insight" story for population-genomics customers

---

## Slide 9 — The Genesis Workbench advantage

**Title:** Why this accelerator vs. assembling the stack yourself?
**Visual:** Architecture diagram (build from scratch):
```
[ Streamlit App UI ]   ← researchers click
        ↓
[ Genesis Workbench modules ]
        ↓                    ↓
[ Model Serving ]  [ Job Clusters w/ Docker ]
        ↓                    ↓
[ Unity Catalog: data + models + lineage ]
        ↓
[ Databricks platform: GPU compute, MLflow, AI Gateway ]
```
**Bullets:**
- **Auto-provisioned GPU** that scales-to-zero — pay for serving only when used
- **Unity Catalog perimeter** spans proprietary data + OS model weights → one governance story
- **MLflow lineage** notebook → endpoint → inference table (compliance-friendly)
- **NVIDIA integration** — Parabricks + BioNeMo in one bundle deploy
- **Streamlit UI** — researchers click; SAs/IT manage perimeter
**Speaker notes:**
- The "five wins" — these are the differentiators vs. rolling your own.

---

## Slide 10 — Researcher journey (one workflow, no context-switching)

**Title:** Cells → Target → Candidate, without leaving the workbench
**Visual:** `gwb_app_imgs/Home_UpdatedAppDeploy_0.png` showing tab navigation, OR a flow diagram of tab progression with arrows
**Bullets:**
- Single Cell → identify misbehaving cells / pathways
- Disease Biology → confirm at population scale (GWAS)
- Protein Studies → predict target's 3D shape
- Small Molecule → design candidate compound
- Loop back via inference + MLflow tracking
**Speaker notes:**
- This is the "stitched-together platform" story — the modules aren't isolated.

---

## Slide 11 — When to introduce GWB to customers

**Title:** Positioning: GWB vs. AI-Driven Drug Discovery solacc
**Visual:** Spectrum diagram (build from scratch):
```
Lower-ramp                                  Higher-customization
[ AI-Driven Drug Discovery ] ―――――― [ Genesis Workbench ]
data engineering                       custom model serving
+ foundation model APIs                  + R&D workflows
```
**Bullets:**
- Both exist; complementary, not competing
- AI-Driven Drug Discovery: data eng + foundation-model APIs onramp
- GWB: customers ready for custom model serving + R&D workflows
- Customers can adopt both as they mature
**Speaker notes:**
- Helps SA conversations — match GWB to customer-readiness signals.

---

## Slide 12 — Live demo

**Title:** Let's look at it
**Visual:** Switch to live app at `https://gwb-mmt-demo-1444828305810485.aws.databricksapps.com`
**Bullets / talking track:** demo flow per `PLAYBOOK.md` (Demos A-D) — pick 1-2 based on audience time:
- (5 min) Protein Studies → ESMFold structure prediction
- (5 min) Single Cell → SCimilarity cell-similarity search
- (5 min) Small Molecule → Chemprop ADMET (NEW today)
- (5 min) Disease Biology → Variant Annotation w/ BRCA filter (NEW today)
**Speaker notes:**
- Have the app open in another window before this slide.
- If demo fails: fall back to screenshots from `gwb_app_imgs/` per workflow.

---

## Slide 13 — Closing & next steps

**Title:** Faster time to science. Faster time to life-saving treatments.
**Visual:** Same "drug R&D pipeline" diagram from slide 3, with all stages now ✓-marked
**Bullets:**
- Open source — fork the repo
- Deploy in your workspace — `./deploy.sh` against `version_pinning` branch
- Demo target: `gwb-mmt-demo-1444828305810485.aws.databricksapps.com`
- Resources: `claude_skills/SKILL_GENESIS_WORKBENCH_DEPLOY_WIZARD.md` + Databricks blog series
**CTA:**
- Q&A
- "What's the most concrete drug discovery problem you're trying to accelerate?"
**Speaker notes:**
- Closing line from script: "Faster time to science — which ultimately means faster time to life-saving treatments."

---

## Production notes (MVP → polish)

**Tonight:**
- Paste each slide's content into Google Slides / Keynote
- Drop screenshots from `gwb_app_imgs/` into the slides
- Pick a clean Databricks-branded template

**During clickthrough testing:**
- Capture fresh screenshots for slides 5, 6, 7, 8, 10
- Tag them with the same `<Module>_<Workflow>_<Step>.png` naming convention
- Replace the placeholders in this deck

**Post-demo polish:**
- Refine slide 9 architecture diagram (consider GWB-specific architecture-html skill output as a swap-in)
- Slide 11 positioning may need a customer-specific spin
- Slide 13 CTA should be customer-specific (next-step asks)

## Visual gaps to fill (TODO during testing)

| Slide | Current | Need |
|---|---|---|
| 5 | DiseaseBiology table screenshot only | Single-cell tab capture (Cell Type Annotation or scGPT) |
| 6 | None ideal | Mol* viewer with ESMFold output; protein design pipeline view |
| 7 | BioNeMo finetune setup | Parabricks tab + maybe BioNeMo inference output |
| 8 | OK | Maybe a Manhattan plot from GWAS run if a real run is available |
| 10 | Home tab | A multi-tab view or annotated screenshot of tab progression |

## Refresh hooks

- If module sizing / endpoint name changes upstream → update slides 5-7
- If a new module is added → potentially add slide between 8-9
- If `AI-Driven Drug Discovery solacc` positioning evolves → refresh slide 11
- If demo target changes (e.g., e2fe replaced or augmented) → update slide 12
