# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC ### scGPT Perturbation Prediction Model
# MAGIC
# MAGIC Registers a PyFunc model that predicts the effect of gene knockouts or
# MAGIC overexpression on cell state using the same pre-trained scGPT whole-human
# MAGIC weights used for embedding generation.
# MAGIC
# MAGIC **Approach:** Zero-shot perturbation prediction. Takes control cell
# MAGIC expression (log1p values), sets perturbed gene(s) to 0 (knockout) or max
# MAGIC (overexpress), runs the transformer encoder + expression decoder, and
# MAGIC returns the predicted post-perturbation expression profile.

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("perturb_model_name", "scgpt_perturbation", "Perturbation Model Name")
dbutils.widgets.text("experiment_name", "scgpt_genesis_workbench_modules", "Experiment Name")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")
dbutils.widgets.text("cache_dir", "scgpt_cache_dir", "Cache dir")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("perturb_model_name")
experiment_name = dbutils.widgets.get("experiment_name")
user_email = dbutils.widgets.get("user_email")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")
cache_dir = dbutils.widgets.get("cache_dir")

cache_full_path = f"/Volumes/{catalog}/{schema}/{cache_dir}"
print(f"Cache full path: {cache_full_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Locate model weights (already downloaded by 01_register_scgpt)

# COMMAND ----------

import os

model_dir = f"{cache_full_path}/models/"
# Find the actual model directory containing best_model.pt
for root, dirs, files in os.walk(model_dir):
    if "best_model.pt" in files:
        model_dir = root
        break

model_file = f"{model_dir}/best_model.pt"
model_config_file = f"{model_dir}/args.json"
vocab_file = f"{model_dir}/vocab.json"

for f in [model_file, model_config_file, vocab_file]:
    assert os.path.exists(f), f"Missing: {f}"
    print(f"Found: {f} ({os.path.getsize(f)} bytes)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define PerturbationModelWrapper

# COMMAND ----------

import json
import numpy as np
import pandas as pd
import torch
import mlflow.pyfunc
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from typing import Dict, Any


class PerturbationModelWrapper(mlflow.pyfunc.PythonModel):
    """MLflow PyFunc wrapper for scGPT zero-shot perturbation prediction.

    Given control cell expression and gene(s) to perturb, predicts
    the post-perturbation expression profile using the transformer's
    encoder + expression decoder.
    """

    def __init__(self, special_tokens=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.special_tokens = special_tokens or ["<pad>", "<cls>", "<eoc>"]

    def load_context(self, context):
        self.model_file = context.artifacts["model_file"]
        self.model_config_file = context.artifacts["model_config_file"]
        self.vocab_file = context.artifacts["vocab_file"]

        # Load vocabulary
        self.vocab = GeneVocab.from_file(self.vocab_file)
        for s in self.special_tokens:
            if s not in self.vocab:
                self.vocab.append_token(s)
        self.gene2idx = self.vocab.get_stoi()
        self.idx2gene = {v: k for k, v in self.gene2idx.items()}
        self.ntokens = len(self.vocab)

        # Load config
        with open(self.model_config_file, "r") as f:
            self.model_configs = json.load(f)

        self.embsize = self.model_configs["embsize"]
        self.nhead = self.model_configs["nheads"]
        self.d_hid = self.model_configs["d_hid"]
        self.nlayers = self.model_configs["nlayers"]
        self.pad_value = self.model_configs["pad_value"]
        self.n_bins = self.model_configs["n_bins"]

        # Pre-load model
        self.model = TransformerModel(
            ntoken=self.ntokens,
            d_model=self.embsize,
            nhead=self.nhead,
            d_hid=self.d_hid,
            nlayers=self.nlayers,
            vocab=self.vocab,
            pad_value=self.pad_value,
            n_input_bins=self.n_bins,
        )
        try:
            self.model.load_state_dict(torch.load(self.model_file, map_location=self.device))
            print(f"Loaded all model params from {self.model_file}")
        except Exception:
            model_dict = self.model.state_dict()
            pretrained_dict = torch.load(self.model_file, map_location=self.device)
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} matching params")

        # Force all parameters to float32 to avoid dtype mismatches between
        # Embedding outputs (float32) and Linear layer weights (potentially float16
        # from GPU checkpoints).
        self.model.float()
        self.model.to(self.device)
        self.model.eval()

        # Discover model structure for the predict method
        self._model_modules = {name for name, _ in self.model.named_modules()}
        print(f"Perturbation model ready on {self.device}")
        print(f"Model top-level attributes: {[n for n, _ in self.model.named_children()]}")

    def _bin_expression(self, values: np.ndarray) -> np.ndarray:
        """Bin log1p expression values into discrete tokens (0 to n_bins-1)."""
        bins = np.linspace(0, values.max() + 1e-6, self.n_bins)
        binned = np.digitize(values, bins) - 1
        return np.clip(binned, 0, self.n_bins - 1)

    def _get_gene_embeddings(self, gene_ids_tensor, values_tensor):
        """Get combined gene+value embeddings using the model's actual sub-modules.

        The value_encoder is a ContinuousValueEncoder with Linear layers inside,
        so it expects FLOAT tensors (continuous expression values), NOT long/int
        (discrete bin indices).
        """
        # Gene token embeddings (always via .encoder which is nn.Embedding)
        gene_emb = self.model.encoder(gene_ids_tensor)

        # Ensure values are float — ContinuousValueEncoder has Linear layers
        float_values = values_tensor.float()

        # Value/expression embeddings — try known attribute names
        val_encoder = getattr(self.model, "value_encoder", None)
        if val_encoder is None:
            val_encoder = getattr(self.model, "value_enc", None)
        if val_encoder is not None:
            val_emb = val_encoder(float_values)
            return gene_emb + val_emb

        # Fallback: scale gene embeddings by normalized expression values
        scale = float_values / (float_values.max() + 1e-6)
        return gene_emb * scale.unsqueeze(-1)

    def _run_transformer(self, combined_input):
        """Run combined embeddings through the transformer and decode expression.

        """
        if combined_input.dim() == 2:
            combined_input = combined_input.unsqueeze(0)

        # Try transformer_encoder attribute
        transformer = getattr(self.model, "transformer_encoder", None)
        if transformer is not None:
            output = transformer(combined_input)
        else:
            output = combined_input

        # Decode expression
        decoder = getattr(self.model, "expr_decoder", None)
        if decoder is None:
            decoder = getattr(self.model, "decoder", None)
        if decoder is not None:
            result = decoder(output)
            # ExprDecoder may return a dict (e.g. {"pred": tensor}) — extract the tensor
            if isinstance(result, dict):
                # Try common keys: "pred", "mlm_output", or take the first tensor value
                for key in ("pred", "mlm_output", "output"):
                    if key in result and torch.is_tensor(result[key]):
                        return result[key]
                # Fallback: first tensor value in the dict
                for v in result.values():
                    if torch.is_tensor(v):
                        return v
            return result

        # Fallback: mean pool across embedding dim
        return output.mean(dim=-1, keepdim=True)

    def predict(self, context, model_input: pd.DataFrame = None, params: Dict[str, Any] = None) -> Dict:
        """Predict perturbation effects.

        Compares the model's output for control vs perturbed expression to
        compute per-gene deltas. Uses the transformer's gene embeddings +
        value embeddings, runs through the encoder, and decodes expression.

        Args:
            model_input: DataFrame with columns:
                - 'expression': list of log1p expression values
                - 'gene_names': list of gene name strings (or JSON list)
            params: dict with:
                - 'genes_to_perturb': comma-separated gene names or list
                - 'perturbation_type': 'knockout' or 'overexpress'

        Returns:
            dict with gene_name, original_expression, predicted_expression, delta, abs_delta
        """
        params = params or {}

        # Read the first row of model_input
        row = model_input.iloc[0] if hasattr(model_input, 'iloc') else model_input[0]

        # Perturbation params can come from params dict OR from model_input columns
        # (the SDK params kwarg may not always reach custom PyFunc models)
        perturbation_type = params.get("perturbation_type") or row.get("perturbation_type", "knockout")

        genes_to_perturb = params.get("genes_to_perturb") or row.get("genes_to_perturb", "")
        if isinstance(genes_to_perturb, str):
            genes_to_perturb = [g.strip() for g in genes_to_perturb.split(",") if g.strip()]

        expression = np.array(row["expression"], dtype=np.float32)
        gene_names = row["gene_names"]
        if isinstance(gene_names, str):
            gene_names = json.loads(gene_names)

        n_genes = len(gene_names)
        assert len(expression) == n_genes, f"Expression length {len(expression)} != gene count {n_genes}"

        # Map genes to vocab token IDs
        gene_ids = []
        valid_indices = []
        for i, gene in enumerate(gene_names):
            if gene in self.gene2idx:
                gene_ids.append(self.gene2idx[gene])
                valid_indices.append(i)

        if not gene_ids:
            raise ValueError("No input genes found in model vocabulary")

        gene_ids_tensor = torch.tensor(gene_ids, dtype=torch.long).to(self.device)
        valid_expression = expression[valid_indices]

        # Use raw float expression values (not binned) — the value_encoder
        # is a ContinuousValueEncoder with Linear layers that expects floats.
        control_values = valid_expression.copy()

        # Create perturbed version
        perturbed_values = valid_expression.copy()
        for gene in genes_to_perturb:
            if gene in self.gene2idx:
                for j, vi in enumerate(valid_indices):
                    if gene_names[vi] == gene:
                        if perturbation_type == "knockout":
                            perturbed_values[j] = 0.0
                        else:
                            perturbed_values[j] = valid_expression.max()
                        break

        control_tensor = torch.tensor(control_values, dtype=torch.float32).to(self.device)
        perturbed_tensor = torch.tensor(perturbed_values, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            # Get combined embeddings for control and perturbed
            control_emb = self._get_gene_embeddings(gene_ids_tensor, control_tensor)
            perturbed_emb = self._get_gene_embeddings(gene_ids_tensor, perturbed_tensor)

            # Run through transformer + decoder
            control_pred = self._run_transformer(control_emb)
            perturbed_pred = self._run_transformer(perturbed_emb)

            control_expr = control_pred.squeeze().cpu().numpy()
            perturbed_expr = perturbed_pred.squeeze().cpu().numpy()

        # Handle multi-dimensional decoder output (take first column if needed)
        if control_expr.ndim > 1:
            control_expr = control_expr[:, 0]
            perturbed_expr = perturbed_expr[:, 0]

        delta = perturbed_expr - control_expr
        valid_gene_names = [gene_names[i] for i in valid_indices]

        return {
            "gene_name": valid_gene_names,
            "original_expression": control_expr.tolist(),
            "predicted_expression": perturbed_expr.tolist(),
            "delta": delta.tolist(),
            "abs_delta": np.abs(delta).tolist(),
        }


# COMMAND ----------

# MAGIC %md
# MAGIC ### Dry load test

# COMMAND ----------

from mlflow.pyfunc import PythonModelContext

context = PythonModelContext(artifacts={
    "model_file": model_file,
    "model_config_file": model_config_file,
    "vocab_file": vocab_file,
}, model_config={})

perturb_model = PerturbationModelWrapper()
perturb_model.load_context(context)
print(f"Model loaded with {perturb_model.ntokens} gene tokens")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate input example and test prediction

# COMMAND ----------

# Create a small test input: 100 genes with random log1p expression
import random
all_genes = list(perturb_model.gene2idx.keys())
# Pick 100 real gene names from vocab (exclude special tokens)
test_genes = [g for g in all_genes if not g.startswith("<")][:100]
test_expression = [random.uniform(0, 5) for _ in test_genes]

input_data = pd.DataFrame({
    "expression": [test_expression],
    "gene_names": [json.dumps(test_genes)],
    "genes_to_perturb": [test_genes[0]],
    "perturbation_type": ["knockout"],
})

output_example = perturb_model.predict(context, model_input=input_data)
print(f"Prediction returned {len(output_example['gene_name'])} genes")
print(f"Top 5 affected genes by |delta|:")
sorted_idx = np.argsort(output_example["abs_delta"])[::-1][:5]
for i in sorted_idx:
    print(f"  {output_example['gene_name'][i]}: delta={output_example['delta'][i]:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register model in Unity Catalog

# COMMAND ----------

import mlflow
from mlflow.models import infer_signature

signature = infer_signature(
    model_input=input_data,
    model_output=output_example,
)

# COMMAND ----------

from databricks.sdk import WorkspaceClient

def set_mlflow_experiment(experiment_tag, user_email):
    w = WorkspaceClient()
    mlflow_experiment_base_path = "Shared/dbx_genesis_workbench_models"
    w.workspace.mkdirs(f"/Workspace/{mlflow_experiment_base_path}")
    experiment_path = f"/{mlflow_experiment_base_path}/{experiment_tag}"
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    return mlflow.set_experiment(experiment_path)


experiment = set_mlflow_experiment(experiment_tag=experiment_name, user_email=user_email)

with mlflow.start_run(run_name=f"{model_name}", experiment_id=experiment.experiment_id) as run:
    registered_model_name = f"{catalog}.{schema}.{model_name}"

    mlflow.pyfunc.log_model(
        "scgpt_perturbation",
        python_model=PerturbationModelWrapper(
            special_tokens=["<pad>", "<cls>", "<eoc>"]
        ),
        artifacts={
            "model_file": str(model_file),
            "model_config_file": str(model_config_file),
            "vocab_file": str(vocab_file),
        },
        pip_requirements="../requirements.txt",
        signature=signature,
        input_example=input_data,
        registered_model_name=registered_model_name,
    )

    print(f"Model registered as {registered_model_name}")
