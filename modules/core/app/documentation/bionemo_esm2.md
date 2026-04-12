# BioNeMo ESM2 Fine-tuning & Inference

## Introduction

BioNeMo ESM2 enables fine-tuning and inference with NVIDIA's implementation of the ESM-2 protein language model. ESM-2 learns general protein representations from ~250M sequences; fine-tuning adapts it to predict specific properties (stability, function, fitness) from your own data.

## What It Achieves

- **Fine-tuning**: Adapts pre-trained ESM-2 to custom protein property prediction tasks
- **Inference**: Runs predictions using base ESM-2 or fine-tuned models
- Supports regression (continuous values) and classification (discrete labels)
- LoRA (Low-Rank Adaptation) option for parameter-efficient fine-tuning
- Tracks experiments and model weights in MLflow

## How to Use

### Fine-tuning

1. Navigate to **NVIDIA > BioNeMo ESM** tab
2. Select the **Fine-tuning** section
3. Choose an ESM-2 variant (650M or 3B parameters)
4. Provide training data CSV (UC Volume) with `sequence` and `target` columns
5. Provide evaluation data CSV
6. Select task type: regression or classification
7. Optionally enable LoRA for efficient fine-tuning
8. Configure training parameters:
   - Number of training steps
   - Batch size
   - Precision (fp16/bf16/fp32)
   - Learning rate and dropout
9. Set MLflow experiment and run names
10. Click **Start Fine-tuning**

### Inference

1. Navigate to **NVIDIA > BioNeMo ESM** tab
2. Select the **Inference** section
3. Choose model source:
   - **Base ESM-2**: Use pre-trained model directly
   - **Fine-tuned**: Select from previously fine-tuned weights
4. Provide input data CSV with a sequences column
5. Specify the column name containing sequences
6. Set output location (UC Volume folder)
7. Click **Start Inference**

### Fine-tuning Data Format

```csv
sequence,target
MKTAYIAKQRQISFVKSH,0.85
MGSSHHHHHHSSGLVPR,0.23
```

### Inputs

| Field | Type | Description |
|-------|------|-------------|
| ESM-2 variant | Selection | 650M or 3B parameters |
| Train data | CSV path | `sequence` + `target` columns |
| Eval data | CSV path | Same format as training |
| Task type | Selection | Regression or classification |
| LoRA | Checkbox | Parameter-efficient fine-tuning |
| Training steps | Integer | Number of optimization steps |

### Outputs

- **Fine-tuned weights**: Stored in UC Volume, tracked in `bionemo_weights` table
- **Training metrics**: Loss curves, validation accuracy
- **Inference results**: CSV with predictions per sequence

## How It's Implemented

### Fine-tuning Pipeline

```
Training CSV + Eval CSV
  ↓
Load pre-trained ESM-2 (650M or 3B)
  ↓
[Optional: Configure LoRA adapters]
  ↓
Train task head (regression/classification)
  ↓
Save weights to UC Volume
  ↓
Register in bionemo_weights table + MLflow
```

### Inference Pipeline

```
Input CSV (sequences)
  ↓
Load model (base or fine-tuned weights)
  ↓
Generate predictions
  ↓
Save results CSV to UC Volume
  ↓
Log to MLflow
```

### Key Files

- `modules/core/app/views/nvidia/bionemo_esm.py` — UI
- `modules/bionemo/notebooks/bionemo_esm_finetune.py` — Fine-tuning execution
- `modules/bionemo/notebooks/bionemo_esm_inference.py` — Inference execution
- `modules/bionemo/notebooks/initialize.py` — Module setup (downloads sample BLAT_ECOLX data)
- `modules/bionemo/resources/initial_setup.yml` — Setup job definition
