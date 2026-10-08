---
name: genesis-workbench
description: Entry point for ANY life-sciences, bio-ML, drug-discovery, genomics, or single-cell workflow in a Databricks notebook that uses Genesis Workbench (GWB). Protein folding and design, binder design, molecular docking, molecule generation, ADMET / developability / safety, protein and single-cell embeddings, similarity search, cell-type annotation, and long-running batch workflows. Use this whenever a user asks how to do a life-sciences / bio model workflow in a GWB workspace — it explains how GWB is consumed and routes to the specific skill. Triggers broadly on protein, antibody, binder, enzyme, molecule, ligand, docking, ADMET, toxicity, genomics, variant, single-cell / scRNA-seq, cell type, embedding, similarity search, and "how do I ... in Genesis Workbench".
---

# Genesis Workbench in a notebook — start here

Genesis Workbench packages biological foundation models behind Databricks infrastructure. From a notebook
you consume it as a **service, not a library**: call its **serving endpoints**, query its **Vector Search
indices**, and dispatch its **batch jobs** — **never `import genesis_workbench`**.

## Route to the right skill
| The user wants to… | Use skill |
|---|---|
| Run a model on an input (fold/design a protein, dock, generate molecules, predict ADMET/solubility/Tm, embed sequences or cells) | **gwb-serving-endpoints** |
| Design or optimize a protein for *their own* target (binder / ligand-binder / motif / de novo / optimize) | **protein-design-from-paper** |
| Find similar proteins by embedding | **gwb-sequence-search** |
| Annotate / embed / perturb single cells (scRNA-seq, AnnData) | **gwb-single-cell** |
| Nearest-neighbor / similarity / annotation over any GWB index | **gwb-vector-search** |
| Launch a long/heavy workflow (enzyme optimization, variant calling, GWAS, (re)build a reference) | **gwb-batch-jobs** |
| Embed/score a *large* dataset across many GPUs, or build their own VS index | **gwb-ray-batch-inference** |
| Get a GPU in a notebook/job (serverless A10) | **gwb-serverless-gpu** |
| See what models / indices / jobs are available | **gwb-discover-resources** |

## Golden rules (apply in every generated cell)
1. **Endpoints only — never `import genesis_workbench`.** `WorkspaceClient.serving_endpoints.query(...)`,
   `vector_search_indexes.query_index(...)`, `jobs.run_now(...)`.
2. **Discover + warm.** Resolve endpoints/indices/jobs by model *slug* at runtime; GWB reaps idle serving
   and VS endpoints, so fail loud (and tell the user to warm them) rather than hard-code names.
3. **Be honest about scope.** In-silico outputs *enrich and prioritize*, they don't measure: ipTM is a
   docking confidence not an affinity; developability scores are proxies (PLTNUM is a 0–1 ranker, not
   hours); cell annotation is reference KNN; perturbation is a prediction. State this in the notebook.
4. **GPU work goes serverless.** A10 serverless GPU (no cluster, no EC2 quota); scale with Ray.

## What's installed here
Models span **large-molecule** (ESMFold, Boltz-2, Proteina-Complexa, RFdiffusion, ProteinMPNN, ESM-2,
AlphaFold2-monomer), **small-molecule** (ChemProp/KERMT ADMET, DiffDock, GenMol, NetSolP, DeepSTABp,
PLTNUM, MHCflurry), **single-cell** (TEDDY, scGPT, SCimilarity), and **genomics** (Parabricks, GWAS, VCF).
Run **gwb-discover-resources** to see exactly what's deployed and awake in this workspace.
