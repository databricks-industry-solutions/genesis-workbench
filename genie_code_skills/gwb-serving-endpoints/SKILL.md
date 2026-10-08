---
name: gwb-serving-endpoints
description: Call any Genesis Workbench model from a Databricks notebook through its serving endpoint for a life-sciences / bio-ML workflow — protein folding and design, docking, molecule generation, ADMET/developability/safety prediction, protein and single-cell embeddings. Use whenever a user wants to run a GWB model on their own input (sequence, structure, SMILES, cells) without importing the genesis_workbench library. Covers endpoint discovery, the three payload styles, parsing predictions, and an exact per-model payload reference. Triggers on "fold this protein", "predict ADMET / toxicity / solubility", "generate molecules", "embed sequences/cells", "run <model> on", "query the endpoint", and most "how do I run <bio model> in Genesis Workbench" questions.
---

# Call Genesis Workbench models via serving endpoints

Treat GWB as a **service, not a library**: call `WorkspaceClient.serving_endpoints.query(...)` and never
`import genesis_workbench`. This skill is the contract reference for every GWB model endpoint.

## Toolkit (emit as a code cell)
```python
%pip install -q "databricks-sdk>=0.50.0"
dbutils.library.restartPython()
```
```python
import numpy as np, pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from databricks.sdk.service.serving import DataframeSplitInput

w = WorkspaceClient(config=Config(http_timeout_seconds=900))

_EP, _ALL = {}, None
def get_ep(slug, override="", exclude=()):
    """Resolve a GWB serving-endpoint name by model slug; fail loud if missing/asleep."""
    if slug in _EP: return _EP[slug]
    if override.strip(): _EP[slug] = override.strip(); return _EP[slug]
    global _ALL
    if _ALL is None: _ALL = [e.name for e in w.serving_endpoints.list()]
    cands = [n for n in _ALL if slug in n and not any(x in n for x in exclude)]
    if not cands:
        raise RuntimeError(f"No serving endpoint matches '{slug}'. Available: {sorted(_ALL)}. "
                           f"Deployed and awake? GWB reaps idle endpoints.")
    stem = lambda n: n[:-9] if n.endswith("_endpoint") else n
    _EP[slug] = ([n for n in cands if stem(n).endswith(slug)] or cands)[0]
    return _EP[slug]

def _unwrap(preds):
    if isinstance(preds, dict) and "predictions" in preds: preds = preds["predictions"]
    return [preds] if isinstance(preds, dict) else list(preds or [])

def q_inputs(slug, items, exclude=()):           # style A: inputs=[...]
    return _unwrap(w.serving_endpoints.query(name=get_ep(slug, exclude=exclude), inputs=items).predictions)
def q_records(slug, records, exclude=()):        # style B: dataframe_records=[{...}]
    return _unwrap(w.serving_endpoints.query(name=get_ep(slug, exclude=exclude), dataframe_records=records).predictions)
def q_split(slug, columns, data, exclude=()):    # style C: dataframe_split(columns, data)
    return _unwrap(w.serving_endpoints.query(name=get_ep(slug, exclude=exclude),
        dataframe_split=DataframeSplitInput(columns=columns, data=data)).predictions)
```
Endpoint names are `gwb_<prefix>_<slug>_endpoint` (prefix varies by install, e.g. `demo`). **Always resolve
by slug at runtime** via `get_ep` — don't hard-code the full name.

## Three payload styles
- **A `inputs=[...]`** — ESMFold, Boltz, ESM-2 embeddings, KERMT, ChemProp, TEDDY.
- **B `dataframe_records=[{...}]`** — ProteinMPNN, NetSolP, DeepSTABp, PLTNUM, MHCflurry.
- **C `dataframe_split(columns=[...], data=[[...]])`** — Proteina-Complexa, DiffDock, GenMol, scGPT, SCimilarity.

## Endpoint reference (slug → payload → output)

### Protein structure & design (large_molecule)
| slug | style | payload | output |
|---|---|---|---|
| `esmfold` | A | `[sequence]` | PDB string (per-residue pLDDT in CA B-factor col) |
| `boltz` | A | `[{"input":"protein_A:<seq>;protein_B:<seq>","msa":"no_msa","use_msa_server":"True"}]` | `{pdb, iptm, protein_iptm, ptm, confidence_score, complex_plddt}` (strings→float; `iptm` = interface confidence) |
| `proteina_complexa` (binder; `exclude=("ligand","ame")`) | C | cols `[target_pdb,binder_length_min,binder_length_max,num_samples,hotspot_residues,target_chain]` | `[{sample_id, pdb_output, sequence, rewards}]` |
| `proteina_complexa_ligand` | C | same cols (`target_pdb` = ligand PDB) | binder designs vs a small molecule |
| `proteina_complexa_ame` | C | same cols (`target_pdb` = motif PDB, `target_chain` = motif chain) | motif-scaffold designs |
| `rfdiffusion` | B | `[{pdb, start_idx, end_idx}]` | inpainted backbone PDB(s) (motif inpainting) |
| `proteinmpnn` | B | `[{pdb, fixed_positions}]` | sequence(s) for a backbone |
| `esm2_embeddings` | A | `[seq1, seq2, ...]` | `[[float × 1280], ...]` (mean-pooled, excl. BOS/EOS) |
| AlphaFold2 | — | **no endpoint** — monomer-only *job* | → **gwb-batch-jobs** |

### Developability & safety (protein)
| slug | style | payload | output (units) |
|---|---|---|---|
| `netsolp` | B | `[{"sequence": seq}]` | `predicted_solubility` ∈ [0,1] (higher better) |
| `deepstabp` | B | `[{"sequence": seq, "growth_temp": 37.0, "mt_mode": "Cell"}]` | `predicted_tm_celsius` (°C) |
| `pltnum` | B | `[{"sequence": seq}]` | `predicted_stability` ∈ [0,1] (relative ranker, **NOT hours**) |
| `mhcflurry` | B | `[{"sequence": seq, "alleles": "HLA-A*02:01,HLA-A*01:01,HLA-B*07:02,HLA-B*44:02,HLA-C*07:01,HLA-C*04:01"}]` | `predicted_immuno_burden` (lower better) |

### Small molecule
| slug | style | payload | output |
|---|---|---|---|
| `diffdock` (+ `diffdock_esm_embeddings`) | C (two-step) | 1) embed: cols `[protein_pdb]` → `{embeddings_b64}`; 2) dock: cols `[protein_pdb, ligand_smiles, samples_per_complex, esm_embeddings_b64]` | poses `[{ligand_sdf, confidence}]` (skip `ligand_sdf` starting `ERROR`; rank by `confidence`) |
| `genmol` | C | cols `[fragment]`, data `[["" ]]` (de novo) or `[["<SMILES>"]]` (scaffold); params `{num_molecules:20, temperature:1.0, randomness:1.0, scoring:"qed", unique:True}` | `[{seed, smiles, score}]` |
| `kermt_admet` | A | `[smiles1, smiles2, ...]` | `[{<task>: value, ...}]` per molecule |
| `chemprop_admet` | A | `[smiles1, smiles2, ...]` | `[{Caco2_Wang, Lipophilicity_AstraZeneca, Solubility_AqSolDB, HydrationFreeEnergy_FreeSolv, PPBR_AZ, VDss_Lombardo, Half_Life_Obach, Clearance_Hepatocyte_AZ, LD50_Zhu, hERG}]` |

### Single cell (AnnData-based — see **gwb-single-cell** for a full worked example)
| slug | style | payload | output |
|---|---|---|---|
| `teddy` | A | `[{"adata_sparsematrix": [[expr...]], "adata_obs": <obs df JSON orient=split>, "adata_var": <var df JSON orient=split, index=gene names>}]` | `[{"embedding": [float × 1024]}]` (512/768 for 70M/160M) |
| `scgpt` | C | `[adata_sparsematrix, adata_obs, adata_var]` (+ preprocess params) | cell embeddings (dict output — confirm exact shape via the scGPT wrapper) |
| `scgpt_perturbation` | C | `[expression, gene_names, genes_to_perturb, perturbation_type("knockout"\|"overexpress")]` | `{gene_name, original_expression, predicted_expression, delta, abs_delta}` |
| `scimilarity_get_embedding` | C | `[celltype_sample(JSON split, col "celltype_subsample"), celltype_sample_obs]` | `{embedding: [float × 128], input_index, ...}` |

(`params=` for GenMol/scGPT go in `w.serving_endpoints.query(..., params={...})`.)

## Example
```python
pdb   = q_inputs("esmfold", ["MVLSPADKTNVKAAWGKV..."])[0]              # fold
res   = q_inputs("boltz", [{"input": f"protein_A:{bndr};protein_B:{tgt}",
                            "msa": "no_msa", "use_msa_server": "True"}])[0]
iptm  = float(res["iptm"])                                            # interface confidence
sol   = q_records("netsolp", [{"sequence": bndr}])[0]["predicted_solubility"]
admet = q_inputs("chemprop_admet", ["CC(=O)Oc1ccccc1C(=O)O"])[0]      # aspirin → ADMET dict
```

## Notes
- **Warm first.** GWB reaps idle endpoints; `get_ep` fails loud if a model is missing/asleep — (re)deploy
  or start it before running.
- Predictions may arrive as a bare list or `{"predictions":[...]}`; `_unwrap` handles both. Numeric fields
  often come back as strings — cast with `float(...)`.
- Embeddings → similarity search: **gwb-vector-search**. Long/batch workflows: **gwb-batch-jobs**. Scale a
  forward across GPUs: **gwb-ray-batch-inference**.
