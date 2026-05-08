"""PLTNUM-ESM2 PyFunc wrapper, separated from the registration notebook
for MLflow's code-based logging (avoids cloudpickling a class defined in
the notebook's `__main__` scope).

Bound to `mlflow.models.set_model(...)` at the bottom.

The `PLTNUM_PreTrainedModel` class is vendored from
`https://github.com/sagawatatsuya/PLTNUM/blob/main/scripts/models.py`
(MIT, copyright 2024 sagawatatsuya).
"""

from __future__ import annotations

import os

import mlflow
import numpy as np
import pandas as pd


class PLTNUMHalfLifeModel(mlflow.pyfunc.PythonModel):
    """ESM-2-based PLTNUM relative-stability ranker.

    Input  (pd.DataFrame): single column `sequence` of protein sequence strings.
    Output (pd.DataFrame): columns `sequence`, `predicted_stability` (float in [0, 1]).
                           Higher = predicted longer-lived. NOT half-life in hours.
    """

    MAX_SEQ_LEN = 1024

    def load_context(self, context):
        import torch
        import torch.nn as nn
        from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel

        artifacts_dir = context.artifacts["weights_dir"]

        class _PLTNUM_PreTrainedModel(PreTrainedModel):
            config_class = AutoConfig

            def __init__(self, config, task="classification"):
                super().__init__(config)
                self.task = task
                self.model = AutoModel.from_pretrained(self.config._name_or_path)
                self.fc_dropout1 = nn.Dropout(0.8)
                self.fc_dropout2 = nn.Dropout(0.4 if task == "classification" else 0.8)
                self.fc = nn.Linear(self.config.hidden_size, 1)

            def forward(self, inputs):
                outputs = self.model(**inputs)
                last_hidden_state = outputs.last_hidden_state[:, 0]
                output = (
                    self.fc(self.fc_dropout1(last_hidden_state))
                    + self.fc(self.fc_dropout2(last_hidden_state))
                ) / 2
                return output

        self._torch = torch

        config = AutoConfig.from_pretrained(artifacts_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(artifacts_dir)
        self.model = _PLTNUM_PreTrainedModel.from_pretrained(
            artifacts_dir, config=config, task="classification"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"PLTNUM model ready on {self.device}")

    def predict(self, context, model_input, params=None):
        torch = self._torch
        if isinstance(model_input, dict):
            model_input = pd.DataFrame(model_input)
        if "sequence" not in model_input.columns:
            raise ValueError(
                "PLTNUM input must be a DataFrame with a 'sequence' column"
            )

        sequences = model_input["sequence"].astype(str).tolist()
        scores = []
        with torch.no_grad():
            for seq in sequences:
                inputs = self.tokenizer(
                    [seq[: self.MAX_SEQ_LEN]],
                    add_special_tokens=True,
                    max_length=self.MAX_SEQ_LEN,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logit = self.model(inputs)
                prob = float(torch.sigmoid(logit).cpu().squeeze().item())
                scores.append(prob)

        return pd.DataFrame({
            "sequence": sequences,
            "predicted_stability": scores,
        })


mlflow.models.set_model(PLTNUMHalfLifeModel())
