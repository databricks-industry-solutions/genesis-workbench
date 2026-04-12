# ADMET & Safety Profiling

## Introduction

ADMET & Safety Profiling evaluates small molecules for drug-like properties — Absorption, Distribution, Metabolism, Excretion, and Toxicity. It uses ChemProp directed message passing neural networks (D-MPNN) to predict multiple pharmacological properties from molecular structure alone.

## What It Achieves

- Predicts blood-brain barrier penetration (BBB)
- Predicts clinical trial toxicity risk
- Predicts 10 ADMET properties via multi-task regression
- Provides color-coded risk indicators (green/orange/red) for quick assessment
- Supports batch profiling of multiple molecules simultaneously

## How to Use

1. Navigate to **Small Molecules > ADMET & Safety** tab
2. Enter one or more SMILES strings (one per line)
3. Select which models to run:
   - **BBB Penetration** (ChemProp BBBP): Binary prediction of blood-brain barrier crossing
   - **Clinical Toxicity** (ChemProp ClinTox): Binary prediction of clinical trial failure
   - **ADMET Properties** (ChemProp multi-task): 10 continuous ADMET property predictions
4. Configure MLflow tracking
5. Click **Run Profiling**

### Inputs

| Field | Description | Example |
|-------|-------------|---------|
| SMILES | One or more molecules, one per line | `CC(=O)Oc1ccccc1C(=O)O` |
| BBB Penetration | Enable BBB prediction | Checkbox |
| Clinical Toxicity | Enable toxicity prediction | Checkbox |
| ADMET Properties | Enable multi-task ADMET | Checkbox |

### Outputs

- **Risk indicators**: Color-coded tiles (green = safe, orange = caution, red = high risk)
- **Per-molecule scores**: Detailed property values for each molecule
- **Summary table**: All molecules with all predicted properties

## How It's Implemented

### Models

| Model | Task | Output |
|-------|------|--------|
| ChemProp BBBP | Binary classification | BBB penetration probability |
| ChemProp ClinTox | Binary classification | Clinical toxicity probability |
| ChemProp ADMET | Multi-task regression | 10 ADMET property values |

### Architecture

- All three models use ChemProp's directed message passing neural network (D-MPNN) architecture
- Models are deployed as Databricks Model Serving endpoints
- Each SMILES string is converted to a molecular graph and processed by the D-MPNN

### Key Files

- `modules/core/app/views/small_molecules.py` — UI (ADMET & Safety tab)
- `modules/core/app/utils/small_molecule_tools.py` — `hit_chemprop_bbbp()`, `hit_chemprop_clintox()`, `hit_chemprop_admet()`
- `modules/small_molecule/chemprop/chemprop_v2/notebooks/` — Model registration notebooks
