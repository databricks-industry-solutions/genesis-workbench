"""DeepSTABp PyFunc wrapper, separated from the registration notebook for
MLflow's code-based logging.

The `deepSTAPpMLP` class is vendored from
`https://github.com/CSBiology/deepStabP/blob/main/src/Api/app/predictor.py`
(MIT, CSBiology).

Bound to `mlflow.models.set_model(...)` at the bottom.
"""

from __future__ import annotations

import os
import re

import mlflow
import numpy as np
import pandas as pd


class DeepSTABpModel(mlflow.pyfunc.PythonModel):
    """ProtT5-XL + DeepSTABp MLP head Tm regression model.

    Input  (pd.DataFrame): `sequence` (required), optional `growth_temp`
                           (float, °C, default 37.0), optional `mt_mode`
                           ("Cell" | "Lysate", default "Cell").
    Output (pd.DataFrame): `sequence`, `predicted_tm_celsius` (float).
    """

    MAX_SEQ_LEN = 1024
    TM_MIN = 30.441673997070385
    TM_MAX = 97.4166905791789
    GROWTH_TEMP_MIN = 30.44167
    GROWTH_TEMP_MAX = 97.4167

    def load_context(self, context):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import pytorch_lightning as pl
        from transformers import T5EncoderModel, T5Tokenizer

        artifacts_dir = context.artifacts["artifacts_dir"]
        prot_t5_dir = os.path.join(artifacts_dir, "prot_t5_xl_uniref50")
        ckpt_path = os.path.join(artifacts_dir, "deepstabp_mlp.ckpt")

        class _DeepStabpMLP(pl.LightningModule):
            def __init__(self, dropout=0.1, learning_rate=0.01, batch_size=25):
                super().__init__()
                self.learning_rate = learning_rate
                self.batch_size = batch_size
                self.dropout = dropout
                self.zero_layer = nn.Linear(1064, 4098)
                self.zero_dropout = nn.Dropout1d(dropout)
                self.first_layer = nn.Linear(4098, 512)
                self.first_dropout = nn.Dropout1d(dropout)
                self.second_layer = nn.Linear(512, 256)
                self.second_dropout = nn.Dropout1d(dropout)
                self.third_layer = nn.Linear(256, 128)
                self.third_dropout = nn.Dropout1d(dropout)
                self.seventh_layer = nn.Linear(128, 1)
                self.species_layer_one = nn.Linear(1, 20)
                self.species_layer_two = nn.Linear(20, 20)
                self.species_dropout = nn.Dropout1d(dropout)
                self.batch_norm0 = nn.LayerNorm(4098)
                self.batch_norm1 = nn.LayerNorm(512)
                self.batch_norm2 = nn.LayerNorm(256)
                self.batch_norm3 = nn.LayerNorm(128)
                self.lysate = nn.Linear(1, 20)
                self.lysate2 = nn.Linear(20, 10)
                self.lysate_dropout = nn.Dropout1d(dropout)
                self.cell = nn.Linear(1, 20)
                self.cell2 = nn.Linear(20, 10)
                self.cell_dropout = nn.Dropout1d(dropout)

            def forward(self, x, species_feature, lysate, cell):
                x = x.float()
                species_feature = species_feature.float().reshape(-1, 1)
                lysate = lysate.float().reshape(-1, 1)
                cell = cell.float().reshape(-1, 1)
                lysate = self.lysate_dropout(F.selu(self.lysate(lysate)))
                lysate = self.lysate_dropout(F.selu(self.lysate2(lysate)))
                cell = self.cell_dropout(F.selu(self.cell(cell)))
                cell = self.cell_dropout(F.selu(self.cell2(cell)))
                species_feature = self.species_dropout(F.selu(self.species_layer_one(species_feature)))
                species_feature = self.species_dropout(F.selu(self.species_layer_two(species_feature)))
                x = torch.cat([lysate, cell, x, species_feature], dim=1)
                x = self.zero_dropout(self.batch_norm0(F.selu(self.zero_layer(x))))
                x = self.first_dropout(self.batch_norm1(F.selu(self.first_layer(x))))
                x = self.second_dropout(self.batch_norm2(F.selu(self.second_layer(x))))
                x = self.third_dropout(self.batch_norm3(F.selu(self.third_layer(x))))
                return self.seventh_layer(x)

        self._torch = torch
        self._re = re

        self.tokenizer = T5Tokenizer.from_pretrained(prot_t5_dir, do_lower_case=False)
        self.encoder = T5EncoderModel.from_pretrained(prot_t5_dir)

        state = torch.load(ckpt_path, map_location="cpu")
        sd = state.get("state_dict", state)

        self.head = _DeepStabpMLP(dropout=0.1, learning_rate=0.01, batch_size=25)
        self.head.load_state_dict(sd, strict=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = self.encoder.to(self.device).eval()
        self.head = self.head.to(self.device).eval()
        print(f"DeepSTABp model ready on {self.device}")

    def _embed(self, sequence):
        torch = self._torch
        re = self._re
        seq = re.sub(r"[OUJZB]", "X", sequence[: self.MAX_SEQ_LEN])
        spaced = " ".join(seq)
        ids = self.tokenizer.batch_encode_plus(
            [spaced], add_special_tokens=True, padding=True, return_tensors="pt"
        )
        input_ids = ids["input_ids"].to(self.device)
        attention_mask = ids["attention_mask"].to(self.device)
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq_len = (attention_mask[0] == 1).sum().item()
        emb = out.last_hidden_state[0, : seq_len - 1]
        return emb.mean(dim=0).reshape(1, -1)

    def predict(self, context, model_input, params=None):
        torch = self._torch
        if isinstance(model_input, dict):
            model_input = pd.DataFrame(model_input)
        if "sequence" not in model_input.columns:
            raise ValueError(
                "DeepSTABp input must be a DataFrame with a 'sequence' column"
            )

        sequences = model_input["sequence"].astype(str).tolist()
        growth_temps = (
            model_input.get("growth_temp", pd.Series([37.0] * len(sequences)))
            .astype(float).tolist()
        )
        mt_modes = (
            model_input.get("mt_mode", pd.Series(["Cell"] * len(sequences)))
            .astype(str).tolist()
        )

        tms = []
        with torch.no_grad():
            for seq, gt, mt in zip(sequences, growth_temps, mt_modes):
                emb = self._embed(seq)
                gt_norm = (gt - self.GROWTH_TEMP_MIN) / (self.GROWTH_TEMP_MAX - self.GROWTH_TEMP_MIN)
                species_t = torch.tensor([[gt_norm]], dtype=torch.float32).to(self.device)
                if mt.lower().startswith("lys"):
                    lysate_t = torch.tensor([[1]], dtype=torch.float32).to(self.device)
                    cell_t = torch.tensor([[0]], dtype=torch.float32).to(self.device)
                else:
                    lysate_t = torch.tensor([[0]], dtype=torch.float32).to(self.device)
                    cell_t = torch.tensor([[1]], dtype=torch.float32).to(self.device)
                tm_norm = self.head(emb, species_t, lysate_t, cell_t)
                tm_celsius = float(tm_norm.flatten()[0].item()) * (self.TM_MAX - self.TM_MIN) + self.TM_MIN
                tms.append(tm_celsius)

        return pd.DataFrame({
            "sequence": sequences,
            "predicted_tm_celsius": tms,
        })


mlflow.models.set_model(DeepSTABpModel())
