"""HLAIIPred (Pfizer, Apache-2.0) PyFunc wrapper — MHC class II (CD4) presentation
scored as a single 'immunogenic burden' per protein. Separated from the registration
notebook for MLflow code-based logging; the `hlapred` package ships via `code_paths`
and the model weights + mhcII pseudosequences ship via `artifacts`.

Mirrors the MHCflurry wrapper's contract so it's a drop-in developability axis:
  Input  (pd.DataFrame): `sequence` (required), optional `alleles` (comma-separated HLA-II
                         names; defaults to an 8-allele DRB1 de-immunization panel).
  Output (pd.DataFrame): `sequence`, `predicted_immuno_burden` (strong MHC-II presenters per
                         residue, lower is better), `max_presentation_score` (worst window).

NOTE: HLAIIPred's peptide API accepts sequences up to 30 aa, so we slide a 15-mer window over
the input chain, score each window across the allele panel, average the two released folds
(epT_0 / epT_1), then aggregate. The score is MHC-II *presentation* (a strong ADA proxy), not a
calibrated clinical immunogenicity probability.
"""

from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd


class HLAIIPredImmunoBurdenModel(mlflow.pyfunc.PythonModel):

    PEPTIDE_LEN = 15
    STRONG_PRESENTATION_THRESHOLD = 0.5
    # 8-allele DRB1 panel used for de-immunization screening (covers common global DR types).
    DEFAULT_PANEL = (
        "DRB1*01:01,DRB1*03:01,DRB1*04:01,DRB1*07:01,"
        "DRB1*08:01,DRB1*11:01,DRB1*13:01,DRB1*15:01"
    )

    def load_context(self, context):
        import torch
        from hlapred.predict import HLAIIPredict  # shipped via code_paths

        self._np = np
        device = torch.device("cpu")
        models_dir = context.artifacts["models"]
        mhcii_dir = context.artifacts["mhcII"]
        # Two released folds (epT_0, epT_1) — averaged, matching predict_peptide_example.py.
        self.predictors = [HLAIIPredict(models_dir, fold, device, mhcii_dir) for fold in (0, 1)]
        print("HLAIIPred predictors loaded (folds 0,1) on CPU")

    @staticmethod
    def _kmers(sequence, k):
        if len(sequence) <= k:
            return [sequence] if sequence else []
        return [sequence[i:i + k] for i in range(0, len(sequence) - k + 1)]

    def _score_one(self, sequence, alleles):
        peptides = self._kmers(sequence, self.PEPTIDE_LEN)
        if not peptides:
            return 0.0, 0.0
        allele_lists = [list(alleles) for _ in peptides]  # one allele list per peptide (<=14)
        fold_preds = []
        for predictor in self.predictors:
            inputs = predictor.prepare_input(peptides, allele_lists)
            y_pred, _scores = predictor.predict(inputs, batch_size=32, sigmoid=True)
            fold_preds.append(np.asarray(y_pred, dtype=float).reshape(-1))
        ps = np.mean(np.vstack(fold_preds), axis=0)  # per-window presentation score, averaged over folds
        strong = int((ps >= self.STRONG_PRESENTATION_THRESHOLD).sum())
        burden = float(strong) / max(len(sequence), 1)
        max_score = float(ps.max()) if ps.size else 0.0
        return burden, max_score

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, dict):
            model_input = pd.DataFrame(model_input)
        if "sequence" not in model_input.columns:
            raise ValueError("HLAIIPred input must be a DataFrame with a 'sequence' column")

        sequences = model_input["sequence"].astype(str).tolist()
        if "alleles" in model_input.columns:
            alleles_per_row = model_input["alleles"].fillna(self.DEFAULT_PANEL).astype(str).tolist()
        else:
            alleles_per_row = [self.DEFAULT_PANEL] * len(sequences)

        burdens, max_scores = [], []
        for seq, allele_str in zip(sequences, alleles_per_row):
            allele_list = [a.strip() for a in allele_str.split(",") if a.strip()][:14]  # HLAIIPred cap
            burden, max_score = self._score_one(seq, allele_list)
            burdens.append(burden)
            max_scores.append(max_score)

        return pd.DataFrame({
            "sequence": sequences,
            "predicted_immuno_burden": burdens,
            "max_presentation_score": max_scores,
        })


mlflow.models.set_model(HLAIIPredImmunoBurdenModel())
