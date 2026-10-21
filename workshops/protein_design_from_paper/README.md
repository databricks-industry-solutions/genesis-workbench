# Protein Design From a Paper — scaffolding kit

A reusable kit for attendees to run **their own** protein-design research in Genesis Workbench, in the
spirit of the worked snake-venom-binder workshop (`../snake_venom_binders/`). Everything here consumes GWB
**only through its serving endpoints** — no `genesis_workbench` library import.

## What's here

| File | Role |
|---|---|
| `00_toolkit.py` | **Hands.** `%run`-able Databricks notebook: direct serving-endpoint callers (binder / ligand-binder / motif / RFdiffusion-inpaint / ProteinMPNN / ESMFold / Boltz / DiffDock / developability / ADMET) + PDB parsing & composite-reward helpers. Endpoints resolve lazily (`get_ep`). |
| `TEMPLATE_protein_design_notebook.py` | **Shell.** A parameterized skeleton: `%run ./00_toolkit`, define your problem, keep the one design-variant block for your task, run the shared funnel (generate → fold → complex-confidence → developability → select). |
| `../../claude_skills/SKILL_GENESIS_WORKBENCH_PROTEIN_DESIGN_FROM_PAPER.md` | **Brain.** The Claude Code skill that classifies a paper/goal into a task and fills the template for the user's target. |

## How an attendee uses it

1. In Claude Code (in this repo), say: *"Use \<paper\> as a guideline and design \<binder against protein X /
   binder for ligand Y / scaffold motif M / de novo monomer / optimize design Z\> — build me the notebook."*
   Claude loads the skill and generates `NN_<task>_<target>.py` from the template.
2. Open the generated notebook in Databricks next to `00_toolkit.py`, confirm endpoints resolve (first
   `get_ep` call fails loud if a model is asleep — warm them first), and run top to bottom.

Prefer to do it by hand? Copy `TEMPLATE_protein_design_notebook.py`, keep the matching TASK block, fill the
`TODO`/`<...>` fields, and run. The toolkit does the endpoint work either way.

## Scope (honest)

The in-silico funnel does **enrichment and prioritization** — a plausible fold, interface confidence, and a
developability triage that decides *which designs to order*. It does **not** measure binding affinity,
stability, or function; those still require the wet lab.
