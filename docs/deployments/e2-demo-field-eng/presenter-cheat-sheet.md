# Presenter cheat sheet — GWB 15-20 min live demo

**Source:** distilled from `Pitching_Genesis_Workbench_to_Pharma_Executives` NotebookLM transcript (2026-04-26).
**Use:** 1-page in-the-room reference. Keep on a second screen / printed alongside the slide deck.

---

## First 60 seconds — opener

❌ **DON'T** lead with "wet lab vs in silico" — every vendor uses it; eyes glaze over.

✅ **DO** drop a specific Databricks-blog metric:
> *"The average R&D life cycle for a single new drug spans 10 to 15 years. That timeline is the bleeding neck for these executives. It's what keeps them up at night."*

Then explain WHY it takes so long. Don't let the audience stay on "we don't know enough biology yet."
The real villain = **"hidden toil"**. Your scientists are *amateur cloud infrastructure engineers* — configuring CUDA, managing dependencies, stitching pipelines across a dozen tools. **Fighting software instead of fighting diseases.**

That sets up your value prop: *a unified, governed in-silico platform matters most when proprietary enterprise data must sit safely next to open-source foundation models.*

---

## Setup slide — Unity Catalog as IP-security hero

**The line:** Unity Catalog tracks data lineage from a raw FASTA file all the way to a natural-language query. Airtight governed environment → use public AI without risking private IP.

**Visual:** Watch the executives nod here — IP security is their baseline requirement.

---

## Positioning slide — AI-D3 vs GWB (handle confusion proactively)

Customers know about the "AI-Driven Drug Discovery" solacc and may ask "didn't we already cover this?" Don't dodge — **draw a 2x2 / spectrum**:

| AI-D3 (on-ramp) | Genesis Workbench (destination) |
|---|---|
| Data engineering first | Custom model serving + heavy GPU compute |
| Literature search, biomarker analysis, knowledge graphs | Multi-module biological workflows + R&D pipelines |
| **"Advanced research library"** | **"High-tech fabrication lab"** |

> Both are complementary. AI-D3 organizes the data; GWB builds the lab next door.

---

## Pipeline slide — narrate one disease

Pick a concrete disease — e.g., **targeted lung cancer** — and trace it through 4 stages (~60 sec):

1. **Find the target** → Disease Biology + Single Cell modules (population genetics + microscopic cell expression)
2. **Understand the target's 3D shape** → Protein Studies (folded origami, pockets, crevices)
3. **Design the drug** → Small Molecule (generative AI + diffusion models)
4. **Cross-cutting accelerators** → NVIDIA Parabricks + BioNeMo (heavy GPU compute, makes 1-3 run in seconds vs days)

---

## Live demo — choreography for safety

### Open with the HOME tab — never start with a heavy compute click

> **The terror of live cloud demos:** click button → loading spinner → dead silence → endpoint scaling from zero in the background. *"Career-ending click."*

✅ **Mitigation:** start at HOME tab, type a Claude-powered natural-language prompt like *"how do I predict a protein structure?"*

This **buys 30-60 seconds for heavy endpoints to warm up** while simultaneously proving the integration. Low-risk, high-reward opener.

### Stage 1 — Find the target (Disease Biology + Single Cell)

**GWAS Manhattan plot** — pre-run, NOT live.
- Looks like a city skyline; *"point at the skyscrapers"*
- Variant p-values spiking over the standard genome-wide significance threshold (5×10⁻⁸)
- *"This visual spike is the platform mathematically proving this specific genetic mutation is highly associated with the disease."*

**SCimilarity** — query a single cell against 23 million cells in seconds.
- Returns disease context + original study provenance.

**scGPT zero-shot perturbation — the showstopper.**
- Pick a cancer regulator (e.g. **TP53**); click *Run zero-shot gene knockout*.
- Compare to ChatGPT: *"ChatGPT learned the grammar of human language. scGPT learned the **grammar of cellular biology**. So when you ask it to knock out a gene, it mathematically simulates the cascading biological effects across the entire cell — without touching a single pipette."*
- This is the moment the actual biologists in the room lean forward.

### Stage 2 — Understand the target (Protein Studies)

**ESMFold for the live render — ~5 seconds, interactive 3D viewer.**

⚠️ **AlphaFold2 warning — career-ending click:**
- Customers WILL ask why not AlphaFold2 (industry standard)
- Answer: it's fully integrated. **But don't run it live.** AlphaFold2 uses deep multiple sequence alignments + structural templates, splits compute across CPU+GPU, takes **hours per complex protein**.
- *Show a pre-run AlphaFold2 result via the Past Runs feature to prove it's supported. Use ESMFold for the live wow factor.*

### Stage 3 — Design the drug (Small Molecule)

**DiffDock** — feed it the 3D structure of the target protein + chemical structure of a candidate drug (in SMILES format).

🚫 **Don't use the lock-and-key analogy** — too static and outdated.

✅ **Use this instead:** *"Two complex magnetic puzzles snapping together in 3D space."*
- Diffusion model: starts with random noisy placement, gradually denoises to find the lowest energy state (most stable binding)
- Samples multiple plausible binding poses, ranks by confidence
- Dynamic physical simulation, not a shape-matcher

**Chemprop ADMET — the digital safety check.**
- Predicts blood-brain-barrier penetration, clinical toxicity (probability of liver failure in Phase 2), full ADMET
- *"Digitally screen out toxic compounds today, rather than discovering toxicity 5 years and $50M from now in a wet lab."* Hits the balance sheet directly.

---

## Q&A — anticipated objections + crisp answers

### Objection: "Vendor lock-in. What if AlphaFold3 drops? What if our team built our own $10M proprietary predictor?"

> **GWB is not a rigid hard-coded application — it's a reusable reference architecture. A blueprint.**
>
> The underlying Databricks primitives (Streamlit for UI, Model Serving for real-time inference, Databricks Jobs for batch, MLflow for tracking) stay perfectly intact.
>
> Swap out the open-source model and plug yours into the same pipeline. **Future-proof by design** — biology changes, infrastructure scales.

This is exactly the objection you WANT to hear — it sets up the strongest answer.

### Objection: "We work on ultra-rare diseases. Our exotic cell types aren't in any public training corpus. Does this break for us?"

> **No, the pipeline doesn't break. This is where you bring your proprietary data into the fold.**
>
> With BioNeMo integration, you fine-tune a foundation model like ESM2 on your highly specialized rare-disease data — right on the platform.
>
> BioNeMo is heavily NVIDIA-optimized → fine-tuning scales massively.
>
> Use public knowledge as a starting point, build a **custom proprietary model** that understands your rare disease perfectly. Unity Catalog ensures training data never leaks out. **Their weird/rare data becomes their competitive advantage.**

### Objection: "Speed?" (or to drop in proactively)

> Variant calling on NVIDIA Parabricks — **4-6× faster** than CPU pipelines. Thousands of patient genomes = saving weeks of compute time.

---

## Closing — domain-agnostic provocation

Don't close with "any questions?" — leave them with a thought:

> *"Throughout this presentation we've focused on life sciences — proteins, cells, small molecules. But Genesis Workbench is fundamentally **domain-agnostic**. It's a reusable blueprint."*
>
> *"If this exact architecture can simulate gene knockouts and predict 3D protein structures... what happens when you swap the biological models out? When you apply the same governed blueprint to **material science** (new battery designs), to **climate modeling**?"*
>
> *"The potential extends far beyond the wet lab — it's a fundamental shift in how enterprise-scale research is done across the board."*

> *"You aren't just giving them a better magnifying glass for the murky waters of drug discovery. You're handing them the keys to the entire fabrication lab."*

---

## Rapid recap (memorize this sequence)

1. **Open** — 10-15 yr bottleneck → "hidden toil" → Unity Catalog solves the proprietary-meets-public-data governance issue
2. **Position** — matrix slide: AI-D3 = research library / GWB = fabrication lab
3. **Pipeline** — find target → understand 3D shape → design drug → NVIDIA accelerators
4. **Demo**:
   - Claude AI on Home tab (warm up endpoints)
   - GWAS Manhattan plot (skyscrapers = variants)
   - scGPT cellular grammar → zero-shot TP53 knockout
   - ESMFold 5-sec live; AlphaFold2 only via Past Runs
   - DiffDock magnetic puzzles + Chemprop ADMET digital safety
5. **Handle rare-disease objection** — BioNeMo fine-tuning on proprietary data
6. **Drop the speed pitch** — Parabricks 4-6×
7. **Closing** — domain-agnostic blueprint provocation

---

## Mantra for the green room

> "I'm not selling a biology tool. I'm selling **the fabrication lab** that the entire AI era of enterprise R&D is going to be built in."
