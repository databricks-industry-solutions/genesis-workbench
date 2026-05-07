"""MHCflurry 2.x PyFunc wrapper, separated from the registration notebook
for MLflow's code-based logging.

Bound to `mlflow.models.set_model(...)` at the bottom.
"""

from __future__ import annotations

import os

import mlflow
import numpy as np
import pandas as pd


class MHCFlurryImmunoBurdenModel(mlflow.pyfunc.PythonModel):
    """MHCflurry 2.x peptide-MHC presentation predictor wrapped as a single
    'immunogenic burden' score per protein.

    Input  (pd.DataFrame): `sequence` (required), optional `alleles`
                           (comma-separated string of HLA-I alleles for that row;
                           defaults to DEFAULT_PANEL covering ~95% of global population).
    Output (pd.DataFrame): `sequence`, `predicted_immuno_burden`
                           (strong-binders per residue, lower is better),
                           `max_presentation_score` (worst-case single epitope).
    """

    PEPTIDE_LEN = 9
    STRONG_PRESENTATION_THRESHOLD = 0.5
    DEFAULT_PANEL = (
        "HLA-A*01:01,HLA-A*02:01,HLA-A*03:01,"
        "HLA-B*07:02,HLA-B*08:01,HLA-B*44:02"
    )

    def load_context(self, context):
        os.environ["MHCFLURRY_DATA_DIR"] = context.artifacts["mhcflurry_data"]
        from mhcflurry import Class1PresentationPredictor

        self.predictor = Class1PresentationPredictor.load()
        print("MHCflurry Class1PresentationPredictor loaded")

    @staticmethod
    def _kmers(sequence, k):
        return [sequence[i : i + k] for i in range(0, len(sequence) - k + 1)]

    def _score_one(self, sequence, alleles):
        peptides = self._kmers(sequence, self.PEPTIDE_LEN)
        if not peptides:
            return 0.0, 0.0
        # MHCflurry 2.2.x renamed/removed ``predict_to_dataframe`` — the
        # supported entry point on Class1PresentationPredictor is just
        # ``predict(...)`` which already returns a pandas DataFrame.
        df = self.predictor.predict(
            peptides=peptides,
            alleles=alleles,
            verbose=0,
        )
        ps = df["presentation_score"].astype(float)
        strong = (ps >= self.STRONG_PRESENTATION_THRESHOLD).sum()
        burden = float(strong) / max(len(sequence), 1)
        max_score = float(ps.max()) if len(ps) else 0.0
        return burden, max_score

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, dict):
            model_input = pd.DataFrame(model_input)
        if "sequence" not in model_input.columns:
            raise ValueError(
                "MHCflurry input must be a DataFrame with a 'sequence' column"
            )

        sequences = model_input["sequence"].astype(str).tolist()
        if "alleles" in model_input.columns:
            alleles_per_row = model_input["alleles"].fillna(self.DEFAULT_PANEL).astype(str).tolist()
        else:
            alleles_per_row = [self.DEFAULT_PANEL] * len(sequences)

        burdens, max_scores = [], []
        for seq, allele_str in zip(sequences, alleles_per_row):
            allele_list = [a.strip() for a in allele_str.split(",") if a.strip()]
            burden, max_score = self._score_one(seq, allele_list)
            burdens.append(burden)
            max_scores.append(max_score)

        return pd.DataFrame({
            "sequence": sequences,
            "predicted_immuno_burden": burdens,
            "max_presentation_score": max_scores,
        })


mlflow.models.set_model(MHCFlurryImmunoBurdenModel())
