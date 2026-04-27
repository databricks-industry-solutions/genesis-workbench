# NotebookLM Audio Overview prompt — GWB demo rehearsal

Purpose: a podcast that rehearses what the presenter would say across a
**15–20 minute live presentation** to a life-sciences customer audience.
The actual live talk includes:
- Setup / architecture slides (~3–5 min, intro)
- Live app demo clicking through tabs (~8–10 min, with narration)
- Q&A (~3–5 min)
- → ~10–13 min of actual narrative the presenter speaks aloud

The podcast is sized at ~15–17 min so the presenter can listen, time it,
and derive both slides and demo narration from it.

## Sources to load into NotebookLM (priority order)

1. Databricks blogs (anchor positioning + stats):
   - https://www.databricks.com/blog/genesis-workbench-blueprint-life-sciences-applications-databricks
   - https://www.databricks.com/blog/accelerating-drug-discovery-fasta-files-genai-insights-databricks
2. Playbook Google Doc (`19pTNC5ok6V1DwA7_hz4Pbi8h8P1ZdPlWUtf58z50SvU`) — has demo flow + tab structure
3. Customer-facing presentations (export to PDF or paste content):
   - https://docs.google.com/presentation/d/1Ncj1_oSia-qdMtfID0SuFChwb55p06Ru7JlwIOwGZsM/
   - https://docs.google.com/presentation/d/1yZgNC0xuVIG4-e_D-9z7pRzIWx5PcY6UyPfNRaMWTO4/
4. Per-module READMEs from the repo:
   - `modules/single_cell/scimilarity/scimilarity_v0.4.0_weights_v1.1/README.md` (cleanest workload + capability description)
   - `modules/single_cell/scgpt/scgpt_v0.2.4/README.md`
   - `modules/protein_studies/boltz/boltz_1/README.md` (if present)
   - `modules/protein_studies/esmfold/esmfold_v1/README.md`
   - `modules/bionemo/README.md`
   - `modules/parabricks/README.md`
   - `modules/small_molecule/chemprop/chemprop_v2/README.md`
   - `modules/disease_biology/gwas/gwas_v1/README.md`

**Skip:** the prior Gemini transcript — feeding LLM output into another LLM adds noise and bias toward AlphaFold2 (not in our deployment).

## Customize prompt — paste into NotebookLM Audio Overview

```
This podcast is a REHEARSAL ARTIFACT for a Databricks SA preparing a
15–20 minute live presentation on Genesis Workbench (GWB) to a
life-sciences customer audience. The actual live talk includes:
- Setup/architecture slides (~3–5 min, intro)
- Live app demo clicking through tabs (~8–10 min, with narration)
- Q&A (~3–5 min)

Your job: produce a ~15–17 min audio rehearsal that gives the presenter
a clean talk-track to derive both slides and demo narration from.

Audience: Imagine the listeners are LS customer SAs / data leaders /
R&D engineering managers. Tech-fluent, NOT deep computational biology
experts. Familiar with cloud + ML platforms in concept; assume
Databricks/UC/MLflow/Streamlit as background, not subject of the talk.

Format: two-host conversational — Alex (curious technologist asking
the questions a customer SA would) + Sam (Databricks pharma data/AI
architect, talks like she's done this before).

Tone: warm, grounded, genuinely impressed by the platform's wins
without being salesy. Sam should give "when a customer says X, GWB
does Y" specifics. Alex should ask the questions a real customer raises.

STRUCTURE:

1. OPENING (~60 sec) — frame the drug R&D bottleneck with ONE specific
   stat from the Databricks blogs. Avoid the overused "wet lab vs in
   silico" cliché. Set up: a unified, governed in-silico platform
   matters most when proprietary enterprise data has to live alongside
   open-source models.

2. POSITIONING (~60 sec) — where GWB sits relative to the
   "AI-Driven Drug Discovery" solution accelerator (the lower-ramp,
   data-engineering-first entry point). They're complementary, not
   competing. AI-Driven Drug Discovery = onramp. GWB = customers
   ready for custom model serving + R&D workflows.
   [SLIDE-WORTHY: a 2x2 or spectrum showing where each solacc fits]

3. PIPELINE OVERVIEW (~60 sec) — drug discovery's four stages and
   how GWB modules map:
   - Find the target → Disease Biology + Single Cell
   - Understand target's 3D shape → Protein Studies
   - Design the drug → Small Molecule
   - Cross-cutting accelerators → Parabricks, BioNeMo
   [SLIDE-WORTHY: a pipeline diagram with module overlays]

4. PER-MODULE NARRATION (~90 sec each, 6 modules in this order)
   [INTENDED: this content gets spoken DURING the live demo as the
   presenter clicks through each tab. Optimize for that — what does
   the presenter say while clicking?]

   For each module, cover:
   - The biological problem (one sentence)
   - The SPECIFIC open-source models the version_pinning branch
     deploys — be precise. For Protein Studies that's Boltz-1,
     ProteinMPNN, RFDiffusion, ESMFold, ESM2 Embeddings (NOT
     AlphaFold2 — the live deployment serves ESMFold instead, which
     is faster and from Meta).
   - What the researcher does on the app's tab: input → click → output
   - Why running it on Databricks specifically: UC perimeter spans
     proprietary genomic + open-source model weights, MLflow lineage
     from notebook → registered model → serving endpoint, AI Gateway
     inference tables for audit/cost, scale-to-zero serving.

   Module order:
   Disease Biology → Single Cell → Protein Studies → Small Molecule
   → Parabricks → BioNeMo

5. THE GWB ADVANTAGE (~90 sec) — why use this accelerator vs. stitching
   open-source models manually:
   - Auto-provisioned GPU clusters that scale-to-zero (cost discipline)
   - UC governance perimeter for proprietary + open-source weights:
     one story, not two
   - MLflow lineage = compliance-friendly traceability data → inference
   - BioNeMo + Parabricks tight integration → NVIDIA's accelerated
     stack is one bundle deploy, not months of container engineering
   - Streamlit app = low-code UI on top of governed infra; researchers
     click, IT manages the perimeter
   [SLIDE-WORTHY: an architecture diagram with these labeled]

6. ANTICIPATED Q&A (~90 sec) — 3 likely customer questions and crisp
   answers Sam would give:
   - "How does this work with our existing on-prem genomics cluster?"
     (UC + Lakehouse Federation; or import as external tables)
   - "What's the cost model for these GPU endpoints?"
     (scale-to-zero; per-token AI Gateway; fine-tune amortizes)
   - "Can we use models other than what GWB ships?"
     (yes — register your own MLflow models; GWB tabs are templates)

7. CLOSING (~60 sec) — bring it back to the customer's bottom line:
   faster target validation, higher-confidence virtual screening,
   compliant by construction. Time-to-science as a ROI metric.

THINGS TO COVER THAT PRIOR ATTEMPTS MISSED:
- Parabricks deserves its OWN beat, not lumped into "BioNeMo
  integration." NVIDIA's accelerated genomics — variant calling
  reduced from days on CPU to <1 hr on A100. Runs as a Databricks
  Container Services (DCS) cluster.
- BioNeMo has TWO surfaces: ESM2 fine-tuning on proprietary sequences
  AND ESM2 inference for embeddings. The "fine-tune on YOUR data"
  angle is critical for enterprises who can't send sequences to
  public APIs.
- Small Molecule covers THREE workflows: property prediction
  (Chemprop's BBBP, ClinTox, ADMET endpoints), protein-ligand docking
  (DiffDock), and complex 3D folding (Proteina-Complexa).
- Disease Biology is the entry point for population-genomics customers
  — Glow + VCF ingestion + variant annotation + GWAS pipelines.
- Single Cell features scGPT (foundation model that can predict gene
  perturbation outcomes, key for target validation), SCimilarity
  (cell-type similarity search across millions of cells), and Scanpy.

AVOID:
- Reading model lists like a catalog
- AlphaFold2 promises (live deployment serves ESMFold)
- Long detours into bundle/terraform mechanics — those are
  presenter's homework, not the audience's interest
- Treating BioNeMo or Parabricks as commodity integrations
```

## How to use the output

1. Listen at 1.0× — get the narrative shape
2. Listen at 1.5× and timestamp the **`[SLIDE-WORTHY]`** sections — these become slide concepts
3. The 6 module sections become the **demo click-track narration** (rehearse out loud while clicking the actual app at e2-demo-field-eng/gwb-mmt-demo)
4. The Q&A section gives a 3-question safety net
5. Iterate the prompt + regenerate if anything feels off (most common iterations: trim Disease Biology, expand BioNeMo, anchor more in customer-language stats from blogs)

## Iteration history

- **v1 (2026-04-26)** — rehearsal-format above. Sized for 15-17 min audio that maps to a 15-20 min live presentation (slides + live demo + Q&A). Two-host format with explicit `[SLIDE-WORTHY]` markers and Q&A flavor.
- **v0 (2026-04-26)** — earlier audience-facing draft (~10-12 min). Same content but framed as a podcast FOR LS-customer audiences rather than as presenter rehearsal. Useful if you ever want a customer-direct artifact (e.g. embedded in a microsite or sent ahead of a meeting). Archived below.

---

## Archive — v0 audience-facing version (~10-12 min)

Use this if you want a podcast aimed at the **customer audience directly** (not the presenter rehearsing). Tone is more "explanatory podcast" and less "talk-track for a presenter." No `[SLIDE-WORTHY]` markers, no anticipated-Q&A section.

```
Audience: Databricks Solutions Architects and Field Engineers preparing
to demo or position Genesis Workbench (GWB) to life-sciences customers.
Listeners are tech-fluent but NOT deep computational biology experts.
Assume familiarity with Databricks (Unity Catalog, MLflow, Model Serving,
Streamlit Apps, Asset Bundles), no biology background required.

Length: 10–12 minutes. Conversational two-host format — Alex (curious
technologist) + Sam (pharma data/AI architect). Warm, grounded, slightly
impressed-but-not-fawning. Sam should give concrete "when a customer
says X, GWB does Y" examples, not abstractions.

Structure:

1. OPENING (~30 sec) — frame the drug R&D bottleneck with ONE specific
   stat from the Databricks blogs (e.g., $2.6B per approved drug,
   10-15 year timeline, 90% Phase 2 attrition). Avoid the overused
   "wet lab vs in silico" cliché. Set up: a unified, governed in-silico
   platform changes the calculus, especially when proprietary enterprise
   data has to live alongside open-source models.

2. PIPELINE OVERVIEW (~60 sec) — walk the four stages and map modules:
   - Find the target → Disease Biology + Single Cell
   - Understand target's 3D shape → Protein Studies
   - Design the drug → Small Molecule
   - Cross-cutting accelerators → Parabricks (genomics), BioNeMo
     (protein LLM fine-tuning + inference)

3. PER-MODULE BEATS (~75 sec each, 6 modules in this order):
   Disease Biology → Single Cell → Protein Studies → Small Molecule →
   Parabricks → BioNeMo

   For each module, cover:
   - The biological problem it addresses
   - The SPECIFIC open-source models the version_pinning branch
     deploys (don't generalize — name them per the GitHub repo and
     blogs: e.g., Protein Studies serves Boltz-1, ProteinMPNN,
     RFDiffusion, ESMFold, ESM2 Embeddings — NOT AlphaFold2 even
     though some docs mention it. Live deployment runs ESMFold.)
   - What the researcher actually does in the Streamlit app's tab
     (input → click → output) at a high level
   - Why Databricks: UC governance for proprietary + open-source
     weights, MLflow lineage from notebook → registered model →
     serving endpoint, AI Gateway inference tables for audit/cost,
     scale-to-zero serving, Asset Bundles for reproducible deploys

4. THE GWB ADVANTAGE (~60 sec) — why use this accelerator vs. stitching
   models manually:
   - Auto-provisioned GPU clusters that scale-to-zero (cost discipline)
   - UC perimeter spans BOTH proprietary patient data AND open-source
     model weights → one governance story, not two
   - MLflow lineage = compliance-friendly traceability from data through
     inference
   - BioNeMo + Parabricks tight integration → NVIDIA's accelerated
     stack is one bundle deploy, not months of container engineering
   - Streamlit app = low-code UI on top of governed infrastructure;
     researchers click, IT manages the perimeter

5. CLOSING (~30 sec) — positioning vs. the AI-Driven Drug Discovery
   solution accelerator (the lower-ramp data-engineering-first entry
   point). They coexist: AI-Driven Drug Discovery for customers
   starting with foundation models + data engineering; GWB for
   customers ready for custom model serving + R&D workflows.

THINGS TO COVER THAT PRIOR ATTEMPTS MISSED:
[same as v1 — Parabricks own beat, BioNeMo two surfaces, Small Molecule
three workflows, Disease Biology entry point, Single Cell scGPT
perturbation prediction]

AVOID:
[same as v1 — no model catalogs, no AlphaFold2, no terraform detours]
```

## When to use which version

| If you want… | Use |
|---|---|
| Rehearsal artifact to derive slides + click-track | **v1** (default) |
| Customer-direct podcast (microsite, pre-meeting send-ahead, asynchronous warm-up) | The polished script at [`podcast-script-customer-intro.md`](./podcast-script-customer-intro.md) — use directly (read aloud / TTS / written source-of-truth), no NotebookLM regen needed |
| Customer-direct podcast generated fresh via NotebookLM (~10 min trim) | **v0** archive prompt above |
| Both rehearsal + customer-direct | Use v1 prompt for the rehearsal podcast, plus the polished script for the customer-direct artifact |
| See Gemini's original draft for comparison | Section "gemini's take" below — kept as reference for what the polished script was edited from |


## gemini's take
Drug Discovery with Databricks Workbench  
Link: https://gemini.google.com/share/026aa7b70279  
Created with Pro: April 25, 2026 at 01:48 PM  
Published: April 25, 2026 at 01:58 PM  
GitHub Reference: https://github.com/databricks-industry-solutions/genesis-workbench/tree/version_pinning

Need to understand the drug R&D process based on the modules and models used in this accelerator.

---

**Prompt:**  
Can you create a podcast-style explainer (e.g., via NotebookLM) to help me understand what the individual modules and their respective models do? Why are they useful, and why is implementing this via the solution accelerator beneficial?

---

## Podcast Breakdown

### Title: *In Silico: Decoding the Databricks Genesis Workbench*  
**Hosts:**  
- Alex (Curious Tech & Biology Enthusiast)  
- Sam (Data & AI Architect in Life Sciences)  

*(Theme music fades in and out)*

---

**Alex:**  
Welcome back to the podcast! Today, we’re diving into a massive bottleneck in human history: drug discovery. A listener wants to understand the Databricks Genesis Workbench — a cutting-edge open-source blueprint. Sam, you’ve studied the architecture and the Github repo. What are we looking at?

---

**Sam:**  
Hey Alex. Simply put: Genesis Workbench is an AI engineer’s dream toolkit for life sciences.  
Traditionally, drug discovery takes over a decade and billions of dollars, mostly in physical "wet labs." Genesis Workbench offers pre-packaged modules to move this process into the digital "in silico" world.

---

**Alex:**  
Before we get into code, let’s visualize what we’re trying to solve. For listeners, imagine the drug R&D lifecycle.

---

**Sam:**  
Right. The drug discovery pipeline generally follows this path:  
1. Identifying the disease  
2. Finding a target (e.g., a malfunctioning protein)  
3. Designing a drug to hit that target  
4. Preclinical testing  

The Genesis Workbench modules map directly to these stages. Let’s walk through them.

---

### Module 1 & 2: Finding the Target (Disease Biology & Single Cell)

**Alex:**  
Let’s start with understanding the disease. I see Disease Biology and Single Cell modules in the repo.

**Sam:**  
Exactly. Finding a cure starts with finding the cause.  
- **Disease Biology:** Handles heavy data like VCF (Variant Call Format) ingestion and GWAS (Genome-Wide Association Studies), so researchers can find genetic mutations linked to disease across huge populations.  
- **Single Cell:**  
  - Analyzes individual cells instead of tissue blends.  
  - Models: scGPT, SCimilarity, Scanpy.  
  - scGPT, for example, is a foundational AI model for single-cell biology, helping us understand gene expression at the microscopic level.

*Why it’s useful:*  
Pinpoint exactly which cells are misbehaving, to find your drug "target." These models help uncover the failing cellular pathways.

---

### Module 3: Understanding the Target (Protein Studies)

**Alex:**  
We’ve found our target — usually a protein. Now to Protein Studies. The repo lists AlphaFold2, ESMFold, RFDiffusion, Boltz, and ProteinMPNN.

**Sam:**  
This is where generative AI shines. Proteins are complex 3D biological machines.  
- To block a malfunction, you need to know the exact physical shape.
- **AlphaFold2** and **ESMFold**: Predict 3D protein structures from amino acid sequences in minutes.  
- **RFDiffusion** and **ProteinMPNN**: Allow scientists to generate and design entirely new proteins from scratch.  
- **Boltz**: (details specific to repo, if any)

---

### Module 4: Designing the Drug (Small Molecule)

**Sam:**  
Most traditional drugs are "small molecules" (think daily pills, like aspirin). Once you have your 3D protein target, you need to find a chemical compound that fits perfectly into its "pocket."

- **DiffDock**: Diffusion model that predicts how a small molecule will physically bind (dock) with the target protein.  
- **Chemprop**: Predicts properties of the molecule, like toxicity and absorption by the human body.

*Why it’s useful:*  
Virtual screening. Pharmaceutical companies can test millions of chemical compounds against a disease target *on the computer* before ever mixing a chemical in the lab.

---

## The Genesis Workbench Advantage

**Alex:**  
These open-source AI models are impressive. But why use the Databricks solution accelerator instead of just downloading and running these models yourself?

**Sam:**  
That’s the million-dollar question. Running these models in isolation is difficult — they require:  
- Sizable GPU clusters  
- Complex dependency management  
- Strict data governance

Genesis Workbench acts as an out-of-the-box blueprint that solves these headaches.  
- **Automated Infrastructure:** Databricks provisions massive GPU clusters for models like AlphaFold and spins them down when not in use to save money.  
- **Orchestration & Tracking:** MLflow tracks thousands of drug discovery experiments for reproducibility.  
- **BioNeMo Integration:** Tight NVIDIA BioNeMo integration brings state-of-the-art biological models without container engineering.

*Why it’s beneficial:*  
Deploying this accelerator helps AI engineers and biologists avoid months of configuring servers, managing versions, and fixing software bugs. It delivers a unified, governed workspace for both proprietary data and open-source model weights.

---

**Alex:**  
Faster time to science — ultimately faster time to life-saving treatments. Sam, thanks for breaking that down.

**Sam:**  
Anytime, Alex.

*(Theme music fades out)*