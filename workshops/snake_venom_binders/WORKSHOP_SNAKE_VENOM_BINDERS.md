# Workshop — De Novo Protein Binders Against Snake-Venom Toxins

*A half-day, hands-on Genesis Workbench session that recreates the in-silico design funnel of
Vázquez Torres et al., "De novo designed proteins neutralize lethal snake venom toxins," Nature 639 (2025),
[10.1038/s41586-024-08393-x](https://www.nature.com/articles/s41586-024-08393-x).*

Companion notebook: **`01_denovo_toxin_binder_workshop.py`** (import into Databricks as a notebook).

---

## Facilitator pre-session checklist

Genesis Workbench reaps idle serving endpoints and apps, so **warm everything the day before and again
~30 min before the session**:

- [ ] Confirm the workshop workspace has GWB deployed (app + MCP) and these endpoints **awake**:
      `proteina_complexa`, `esmfold`, `boltz`, `netsolp_v1`, `deepstabp_v1`, `pltnum_v1`, `mhcflurry_v2`.
      (Run notebook cell **0.1** — it fails loud and lists what's available if any are missing/asleep.)
- [ ] If the app or endpoints are missing, re-run the module `update.sh` / re-deploy before the session.
- [ ] Each attendee can reach the workspace and attach the notebook to a cluster/serverless with the
      Databricks SDK available (DBR ML has it; the notebook also `pip install`s `databricks-sdk`, `requests`, `py3Dmol`).
- [ ] Warm the GPU endpoints with one tiny call (fold a short sequence through ESMFold) so the first
      attendee call isn't a cold start.
- [ ] Decide the time budget: `num_designs` (default 8) and `num_cofold` (default 4) are the two knobs
      that drive runtime. Lower them if GPU capacity is tight.
- [ ] (Optional) Pre-fetch `1CTX` and `9BK5` to confirm RCSB egress works from the workspace.

> **Design principle for this material:** we use Genesis Workbench **only as a service** — the notebook
> calls the deployed **serving endpoints directly** (and shows the **MCP** path). It does **not** import
> the `genesis_workbench` library. Attendees learn to orchestrate the funnel themselves from raw endpoint
> calls, which is exactly how an external team would consume GWB.

---

## Title
**From Venom to Antidote: Designing De Novo Protein Binders Against a Snake-Venom Toxin with Genesis Workbench**

## Objective
Reproduce the **computational** half of a landmark de novo protein-design paper. By the end, each attendee
will have, against the real long-chain α-neurotoxin **α-cobratoxin**, run a complete design funnel entirely
through Genesis Workbench serving endpoints: **generate** candidate binders, **fold** them, **co-fold the
binder–toxin complex and score the interface**, **screen developability**, and **rank** the finalists they
would carry into the wet lab. Attendees should leave understanding both the modern binder-design workflow
*and* how to drive it against a deployed model platform — including an honest view of where in-silico design
stops and experiments begin.

## Key Takeaways and Core Concepts
- **De novo binder design is now a funnel, not a lucky hit.** Generate many → filter hard on structure
  confidence → triage on developability → order a few. The paper reached sub-nanomolar, in-vivo-protective
  binders with *limited* experimental screening because this funnel did the enrichment.
- **One GWB call replaces two paper steps.** The paper used **RFdiffusion** (backbone) + **ProteinMPNN**
  (sequence). GWB's **Proteina-Complexa Binder** emits backbone *and* sequence jointly, hotspot-aware, in a
  single endpoint call.
- **Interface confidence is the key filter.** GWB's AlphaFold2 is monomer-only, so the paper's
  "AF2-multimer pae/ipTM" filter is reproduced with **Boltz-2 co-folding** of `protein_A:binder;protein_B:toxin`,
  ranked on **ipTM** (a BindCraft-style interface-confidence metric). Boltz returns the confidence scalars
  directly from the endpoint.
- **Developability predictors are decision tools, not assays.** NetSolP (solubility 0–1), DeepSTABp (Tm °C),
  PLTNUM (relative stability 0–1, *not hours*), MHCflurry (immunogenicity burden, lower = better) tell you
  *which* designs to order — they do not measure binding or neutralization.
- **Consume models as services.** Direct `serving_endpoints.query` calls (and MCP tools) make the workflow
  portable and decoupled from any one platform's internal library.

## Step-by-Step Guide

### Step 0
**Action:** Install the Databricks SDK, create a `WorkspaceClient`, and **auto-discover** the seven GWB
endpoint names from the workspace (cells 0.1–0.3). Define thin one-line callers (`gen_binders`, `esmfold`,
`boltz_complex`, `developability`) and local parse/score helpers.
**Result:** A printed map of resolved endpoints and a loud failure if any model is undeployed/asleep — the
preflight gate for the whole session.

### Step 1
**Action:** Fetch α-cobratoxin from RCSB (`1CTX`, chain A, 71 residues); extract its sequence; nominate a
loop-II hotspot (default `30,33,36` — the paper highlights **Arg33** at the loop-II tip).
**Result:** The target structure (fed to the binder generator) and target sequence (fed to Boltz), plus a
hotspot the attendee can edit to steer designs.

### Step 2
**Action:** Call **Proteina-Complexa Binder** for N designs (default 8) against the target PDB + hotspot.
**Result:** A table of de novo binders — each with a sequence, a backbone, and a generative `rewards` score.

### Step 3
**Action:** Fold each design on its own with **ESMFold**; parse mean pLDDT from the B-factor column; keep
designs with pLDDT ≥ 70.
**Result:** A cheap monomer-foldability gate that drops implausible designs before the expensive co-fold.

### Step 4
**Action:** Co-fold the top survivors *with the toxin* using **Boltz-2** (`protein_A:binder;protein_B:toxin`);
read **ipTM / protein-ipTM / pTM / complex-pLDDT**; rank by ipTM and keep ≥ 0.50. Visualize the top complex.
**Result:** Interface-confidence-ranked binder–toxin complexes — the paper's AF2-multimer filter, reproduced.

### Step 5
**Action:** Run the interface-passing designs through **NetSolP, DeepSTABp, PLTNUM, MHCflurry**.
**Result:** A developability table (solubility, predicted Tm, relative stability, immunogenicity burden) —
the "would we order this?" view.

### Step 6
**Action:** Normalize each axis and combine into one **composite reward**, reusing GWB's Guided-Enzyme-
Optimization default weights (`plddt 1.3, boltz/iptm 0.5, solubility 1.0, half_life/stability 2.6,
thermostab/Tm 1.0, immuno 1.5`); rank; (optionally) log the funnel to MLflow.
**Result:** A ranked finalist list and a single top pick — the designs an attendee would advance.

### Step 7
**Action:** Fetch **9BK5** (the authors' real LNG binder bound to α-cobratoxin) and compare qualitatively.
**Result:** A direct line of sight from the attendee's in-silico funnel to the paper's experimentally
validated binder (design-vs-crystal RMSD 0.42 Å, Kd 1.9 nM, Tm > 95 °C).

### Appendix A — Refinement by re-sampling *(endpoints-only)*
**Action:** Re-run the binder generator focused on the winner's length/hotspot with more samples; feed back
through Steps 3–6. **Result:** A simple analog of the paper's partial-diffusion hit optimization.

### Appendix B — Guided Enzyme Optimization job *(curious attendees)*
**Action:** Dispatch GWB's `run_enzyme_optimization_gwb` job via the Jobs API (reference snippet).
**Result:** Exposure to the automated reward-weighted optimization loop and its 7 axes. **Note:** it is an
enzyme/motif-scaffolding capstone (needs a motif PDB + substrate SMILES) and long-running (hours) — it
illustrates the closed loop, it is not a drop-in binder-vs-toxin campaign.

### Appendix C — The same pipeline via MCP
**Action:** Point an MCP client at the GWB MCP app and drive `endpoint_proteina_complexa`, `endpoint_boltz`,
the developability tools, or `workflow_protein_binder_design` conversationally. **Result:** The agentic way
to run the identical funnel.

## Reflection
This workshop makes the funnel tangible: most generated designs die at the pLDDT or ipTM gate, and the few
survivors still have to clear developability before they're worth ordering. That attrition *is* the method —
it's why a modern lab can reach potent binders with only a handful of wet-lab experiments. It also makes the
boundary honest: Boltz ipTM is a *confidence* that two chains dock as modeled, not a measured affinity;
PLTNUM is a *ranker*, not a half-life in hours; none of these endpoints prove neutralization. The in-silico
funnel's job is **enrichment and prioritization** — turning an intractable search into a short, testable list.
Attendees should also notice how little code it took: the entire coupling to Genesis Workbench is a handful of
`serving_endpoints.query` calls.

## Next Steps & Resources
- **Broaden the campaign.** Repeat against the paper's other two 3FTx subfamilies: short-chain α-neurotoxin
  (consensus ScNtx) and a cytotoxin (consensus / *N. pallida*). Contrast long- vs short-chain binding.
- **Push refinement.** Use Appendix A re-sampling, or the Appendix B optimization loop, to improve a weak hit.
- **Tighten filters.** Explore ipTM/pLDDT cutoffs and hotspot choices; see how the finalist set shifts.
- **Go agentic.** Reproduce the funnel end-to-end through the GWB MCP tools (Appendix C).
- **Reading & data:**
  - Paper: Vázquez Torres et al., Nature 639 (2025) — [10.1038/s41586-024-08393-x](https://www.nature.com/articles/s41586-024-08393-x)
  - Target: α-cobratoxin **1CTX**; deposited design complex **9BK5** (RCSB).
  - Methods context: RFdiffusion, ProteinMPNN, AlphaFold2 initial-guess, Boltz-2, BindCraft (interface ipTM filtering).
