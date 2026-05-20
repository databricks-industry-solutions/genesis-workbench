"""Smoke test the gwb_teddy_endpoint after deploy.

Run locally:
  python modules/single_cell/teddy/teddy_g_v1/notebooks/smoke_test.py

Sends a 5-cell x 100-gene synthetic payload and prints the response shape
(cell_type, cell_type_conf, top-k, disease, disease_conf, top-k).
"""
import json
import sys
import numpy as np
import pandas as pd
from databricks.sdk import WorkspaceClient

ENDPOINT = "gwb_teddy_endpoint"

def main():
    w = WorkspaceClient(profile="DEFAULT")

    rng = np.random.default_rng(seed=42)
    n_cells, n_genes = 5, 100
    expr = rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32)
    # Use ENSG-style placeholder ids — TEDDY will fall back to median=1.0 for any
    # unknown ids, so the smoke test exercises the full path (token lookup, padding,
    # softmax) even though predictions will be uninformative.
    gene_names = [f"ENSG_TEST_{i:05d}" for i in range(n_genes)]

    payload = [{
        "adata_sparsematrix": expr.tolist(),
        "adata_obs": pd.DataFrame({"cell_id": [f"c_{i}" for i in range(n_cells)]}).to_json(orient="split"),
        "adata_var": pd.DataFrame({"index": gene_names}).to_json(orient="split"),
    }]
    params = {"max_seq_len": 256, "top_k": 3, "return_embedding": False}

    print(f"→ Calling {ENDPOINT} with {n_cells} cells, {n_genes} genes")
    resp = w.serving_endpoints.query(name=ENDPOINT, inputs=payload, params=params)
    preds = resp.predictions
    print(f"← Got {len(preds)} predictions")

    # Validate shape
    required_keys = {"cell_type", "cell_type_conf", "cell_type_topk",
                     "disease", "disease_conf", "disease_topk"}
    ok = True
    for i, p in enumerate(preds):
        missing = required_keys - set(p.keys())
        if missing:
            print(f"  ✗ cell {i}: missing keys {missing}")
            ok = False
        else:
            print(f"  ✓ cell {i}: ct={p['cell_type']} ({p['cell_type_conf']:.2f}) | "
                  f"ds={p['disease']} ({p['disease_conf']:.2f})")

    if ok:
        print("\n✅ Smoke test PASSED — endpoint shape matches the spec.")
        return 0
    print("\n❌ Smoke test FAILED — response missing required keys.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
