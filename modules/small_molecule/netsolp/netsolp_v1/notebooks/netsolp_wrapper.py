"""NetSolP-1.0 PyFunc wrapper, separated from the registration notebook so
MLflow's code-based logging can load it without cloudpickling a class
defined in the notebook's `__main__` scope (which surfaced an IndexError
during `cloudpickle.dump` at registration time).

Bound to `mlflow.models.set_model(...)` at the bottom of the file.
"""

from __future__ import annotations

import os

import mlflow
import numpy as np
import pandas as pd


class NetSolPSolubilityModel(mlflow.pyfunc.PythonModel):
    """ONNX-Runtime wrapper around NetSolP-1.0 (single split of the ESM-12 ensemble).

    Input  (pd.DataFrame): single column `sequence` of protein sequence strings.
    Output (pd.DataFrame): columns `sequence`, `predicted_solubility` (float in [0,1]).

    Sequences longer than 1022 residues are truncated (NetSolP's hard limit).
    """

    MAX_SEQ_LEN = 1022
    ONNX_FILE = "Solubility_ESM12_0_quantized.onnx"
    BACKBONE = "ESM-12 (5-fold ensemble split 0)"

    def load_context(self, context):
        import onnxruntime as ort
        from esm.data import Alphabet

        artifacts_dir = context.artifacts["weights_dir"]
        onnx_path = os.path.join(artifacts_dir, self.ONNX_FILE)

        # The upstream's ``ESM12_alphabet.pkl`` was pickled with a fair-esm
        # version predating ``unique_no_split_tokens`` (added in 2.x). Loading
        # it under the endpoint's fair-esm==2.0.0 yields an Alphabet missing
        # that attribute, and ``batch_converter`` crashes on first call.
        # ESM-12 is in the ESM-1 architecture family — the alphabet built via
        # ``Alphabet.from_architecture("ESM-1")`` is identical for token order,
        # indices, and special tokens, and is fully attribute-correct.
        self.alphabet = Alphabet.from_architecture("ESM-1")
        self.batch_converter = self.alphabet.get_batch_converter()

        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )
        # NetSolP's ONNX graph has 3 named inputs: tokens, lengths, non_pad_mask.
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name
        self.padding_idx = int(self.alphabet.padding_idx)
        print(
            f"NetSolP ONNX session ready (inputs={self.input_names}, "
            f"output={self.output_name}, padding_idx={self.padding_idx})"
        )

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def _tokenize(self, sequences):
        truncated = [
            (str(i), s[: self.MAX_SEQ_LEN]) for i, s in enumerate(sequences)
        ]
        _, _, tokens = self.batch_converter(truncated)
        return tokens.numpy().astype(np.int64)

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, dict):
            model_input = pd.DataFrame(model_input)
        if "sequence" not in model_input.columns:
            raise ValueError(
                "NetSolP input must be a DataFrame with a 'sequence' column"
            )

        sequences = model_input["sequence"].astype(str).tolist()
        token_batch = self._tokenize(sequences)

        scores = []
        for row in token_batch:
            row_2d = row[np.newaxis, :]
            non_pad_bool = (row_2d != self.padding_idx)
            length = np.array([int(non_pad_bool.sum())], dtype=np.int64)
            feed = {
                "tokens": row_2d,
                "lengths": length,
                # ONNX graph declares non_pad_mask as a tensor(bool)
                "non_pad_mask": non_pad_bool.astype(np.bool_),
            }
            # Trim to whichever inputs the graph actually declares (back-compat
            # with earlier 1-input ONNX exports of the same model).
            feed = {k: v for k, v in feed.items() if k in self.input_names}
            logit = self.session.run([self.output_name], feed)[0]
            scores.append(float(self._sigmoid(np.asarray(logit).reshape(-1)[0])))

        return pd.DataFrame({
            "sequence": sequences,
            "predicted_solubility": scores,
        })


mlflow.models.set_model(NetSolPSolubilityModel())
