"""MLflow PyFunc wrapper for TEDDY-G — loaded via `python_model=path` (not
cloudpickle). teddy/ ships as an MLflow **artifact** (artifact key
`teddy_pkg_parent`) so it lands at a well-known path accessible via
context.artifacts; load_context() adds it to sys.path before importing teddy.
"""
import inspect
import os
import sys
from typing import Any, Dict

import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch


class TEDDYEmbedder(mlflow.pyfunc.PythonModel):
    """Wraps TEDDY-G encoder to produce per-cell embeddings.

    Inputs (DataFrame, one row per request):
      - adata_sparsematrix: list[list[float]] — dense expression matrix (cells x genes)
      - adata_var:          str (JSON, orient='split') — gene metadata with `index`
                            column carrying the gene names (must match TEDDY vocab)

    Params:
      - max_seq_len: int (default 2048) — top-K most-expressed genes per cell
      - pooling:     str (default 'mean') — 'mean' or 'cls' over the sequence dim
                     (ignored if the model already returns a 2D pooled tensor)

    Output: list[dict], one per input cell: {"embedding": list[float]}

    Note: with `python_model=path`, MLflow instantiates this with no args. Default
    model_size handles that case. Override is left for in-notebook use.
    """

    _CELL_EMB_NAMES = ("cell_emb", "cell_embedding", "pooled_output", "pooler_output")
    _TOKEN_HIDDEN_NAMES = ("last_hidden_state", "hidden_states", "encoder_last_hidden_state")

    def __init__(self, model_size: str = "70M"):
        self.model_size = model_size

    def load_context(self, context):
        # teddy/ is bundled as an artifact at `teddy_pkg_parent` (a directory that
        # CONTAINS the `teddy/` package). Add it to sys.path before importing.
        teddy_pkg_parent = context.artifacts["teddy_pkg_parent"]
        if teddy_pkg_parent not in sys.path:
            sys.path.insert(0, teddy_pkg_parent)

        from teddy.models.model_directory import get_architecture, model_dict
        from teddy.tokenizer.gene_tokenizer import GeneTokenizer

        model_dir = context.artifacts["model_dir"]
        arch = get_architecture(model_dir)
        config_cls = model_dict[arch]["config_cls"]
        model_cls = model_dict[arch]["model_cls"]

        self.config = config_cls.from_pretrained(model_dir)
        self.model = model_cls.from_pretrained(model_dir, config=self.config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()

        self.tokenizer = GeneTokenizer.from_pretrained(model_dir)
        self._forward_params = set(inspect.signature(self.model.forward).parameters.keys())
        self.add_cls = bool(getattr(self.config, "add_cls", False))
        self.cls_token_id = int(getattr(self.config, "cls_token_id", 0))
        self.d_model = int(getattr(self.config, "d_model", 0))

    def _encode_genes(self, gene_names):
        # TEDDY's GeneTokenizer returns None for OOV tokens from both
        # encode() and convert_tokens_to_ids() — substitute unk_token_id manually.
        if not hasattr(self, "_unk_id"):
            self._unk_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token)
        ids = self.tokenizer.convert_tokens_to_ids(list(gene_names))
        ids = [self._unk_id if i is None else i for i in ids]
        return torch.tensor(ids, dtype=torch.long)

    def _build_batch(self, X, gene_names, max_seq_len):
        token_array = self._encode_genes(gene_names)
        X_t = torch.tensor(X, dtype=torch.float32)
        seq_tokens = max_seq_len - 1 if self.add_cls else max_seq_len
        k = min(seq_tokens, X_t.shape[1])
        _top_vals, top_indices = torch.topk(X_t, k=k, largest=True, sorted=True)
        gene_ids = token_array[top_indices]
        rank_vec = torch.linspace(1.0, -1.0, steps=k)
        gene_values = rank_vec.unsqueeze(0).expand(X_t.shape[0], -1).clone()
        if self.add_cls:
            cls_col = torch.full((X_t.shape[0], 1), self.cls_token_id, dtype=gene_ids.dtype)
            gene_ids = torch.cat([cls_col, gene_ids], dim=1)
            ones_col = torch.ones(X_t.shape[0], 1, dtype=gene_values.dtype)
            gene_values = torch.cat([ones_col, gene_values], dim=1)
        attention_mask = torch.ones_like(gene_ids, dtype=torch.long)
        return (
            gene_ids.to(self.device),
            gene_values.to(self.device),
            attention_mask.to(self.device),
        )

    def _forward(self, gene_ids, gene_values, attention_mask):
        kwargs = {}
        for name in ("gene_ids", "input_ids"):
            if name in self._forward_params:
                kwargs[name] = gene_ids; break
        for name in ("gene_values", "values", "expression_values", "gene_value"):
            if name in self._forward_params:
                kwargs[name] = gene_values; break
        for name in ("attention_mask", "mask"):
            if name in self._forward_params:
                kwargs[name] = attention_mask; break
        with torch.no_grad():
            return self.model(**kwargs)

    @classmethod
    def _extract_hidden(cls, fwd):
        def _get(obj, name):
            return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        for name in cls._CELL_EMB_NAMES:
            v = _get(fwd, name)
            if isinstance(v, torch.Tensor) and v.dim() == 2:
                return v, True
        for name in cls._TOKEN_HIDDEN_NAMES:
            v = _get(fwd, name)
            if isinstance(v, torch.Tensor) and v.dim() >= 2:
                return v, v.dim() == 2
        if isinstance(fwd, (tuple, list)):
            for v in fwd:
                if isinstance(v, torch.Tensor):
                    if v.dim() == 2: return v, True
                    if v.dim() == 3: return v, False
        if isinstance(fwd, torch.Tensor):
            if fwd.dim() == 2: return fwd, True
            if fwd.dim() == 3: return fwd, False
        return None, False

    def predict(self, context, model_input, params=None):
        params = params or {}
        max_seq_len = int(params.get("max_seq_len", 2048))
        pooling = str(params.get("pooling", "mean"))

        out = []
        for _, row in model_input.iterrows():
            X = np.array(row["adata_sparsematrix"], dtype=np.float32)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            var_df = pd.read_json(row["adata_var"], orient="split")
            gene_names = var_df["index"].astype(str).tolist() if "index" in var_df.columns else var_df.index.astype(str).tolist()
            assert X.shape[1] == len(gene_names)

            gene_ids, gene_values, attn = self._build_batch(X, gene_names, max_seq_len)
            fwd = self._forward(gene_ids, gene_values, attn)
            hidden, is_pooled = self._extract_hidden(fwd)
            if hidden is None:
                raise RuntimeError(f"Could not extract hidden state. Got: {type(fwd).__name__}")

            if is_pooled:
                cell_emb = hidden
            elif pooling == "cls":
                cell_emb = hidden[:, 0, :]
            else:
                cell_emb = hidden.mean(dim=1)
            cell_emb = cell_emb.detach().cpu().numpy()

            for i in range(cell_emb.shape[0]):
                out.append({"embedding": cell_emb[i].tolist()})
        return out


# Required by MLflow's `python_model=path` (file-based) API — tells MLflow which
# PythonModel subclass in this file to use. Without this, log_model fails with
# "If the model is logged as code, ensure the model is set using mlflow.models.set_model()".
import mlflow.models  # noqa: E402
mlflow.models.set_model(TEDDYEmbedder())
