---
name: genesis-workbench-troubleshooting
description: Troubleshoot common Genesis Workbench deployment failures, model registration errors, serving endpoint issues, and runtime errors across all modules.
---

# Genesis Workbench Troubleshooting Skill

Diagnose and fix common issues when deploying, registering models, or running workflows in Genesis Workbench.

## Model Registration Failures

### scGPT: RuntimeError mat1 and mat2 must have the same dtype
**Symptom:** `predict()` fails with `RuntimeError: mat1 and mat2 must have the same dtype` in `F.linear`.
**Root cause:** Model checkpoint has float16 weights (from GPU training), but `nn.Embedding` outputs float32. The `ContinuousValueEncoder` has `nn.Linear` layers that reject mixed dtypes.
**Fix:** Call `self.model.float()` after loading weights and before `.to(device)`:
```python
self.model.float()  # Force all parameters to float32
self.model.to(self.device)
self.model.eval()
```

### scGPT: ContinuousValueEncoder expects float, not long
**Symptom:** `F.linear` fails because `value_encoder` receives integer tensor.
**Root cause:** The `value_encoder` in scGPT is `ContinuousValueEncoder` (with Linear layers), NOT `nn.Embedding`. It expects continuous float expression values, not discrete integer bin indices.
**Fix:** Pass `torch.float32` tensors to value_encoder:
```python
control_tensor = torch.tensor(expression_values, dtype=torch.float32).to(self.device)
```

### scGPT: ExprDecoder returns dict, not tensor
**Symptom:** `AttributeError: 'dict' object has no attribute 'squeeze'`
**Root cause:** The `expr_decoder` returns a dict (e.g., `{"pred": tensor}`) not a raw tensor.
**Fix:** Extract the tensor from the dict:
```python
result = decoder(output)
if isinstance(result, dict):
    for key in ("pred", "mlm_output", "output"):
        if key in result and torch.is_tensor(result[key]):
            return result[key]
```

### scGPT: input_example too large or HVG filtering fails
**Symptom:** Model logging is very slow, or `n_top_genes > number of normalized dispersions` warning followed by errors.
**Root cause:** input_example has too few genes for HVG filtering (subset_hvg defaults to 1200).
**Fix:** Use `adata[:10, :1500]` — 10 cells, 1500 genes. Enough genes to survive HVG filtering, small enough for fast logging.

### SyntaxError: invalid character (U+201C or U+201D)
**Symptom:** `SyntaxError: invalid character '"' (U+201D)` when running notebook.
**Root cause:** Smart/curly quotes got into the Python code (from copy-paste or AI-generated code).
**Fix:** Replace all curly quotes with straight ASCII quotes:
```python
content = content.replace('\u201c', '"').replace('\u201d', '"')
content = content.replace('\u2018', "'").replace('\u2019', "'")
```

## Serving Endpoint Issues

### SCimilarity: Request size cannot exceed 16777216 bytes
**Symptom:** `Request size cannot exceed 16777216 bytes` when calling GetEmbedding.
**Root cause:** Sending too many cells in a single request. Each cell has ~18K gene values at ~10 bytes/float in JSON.
**Fix:** Batch cells into groups of 5 per request:
```python
EMBEDDING_BATCH_SIZE = 5
for batch_start in range(0, total, EMBEDDING_BATCH_SIZE):
    batch = normed.iloc[batch_start:batch_start + EMBEDDING_BATCH_SIZE]
    result = get_cell_embeddings(batch)
```

### SCimilarity: 'dict' object has no attribute 'as_dict'
**Symptom:** Error when calling SCimilarity endpoints via the Databricks SDK.
**Root cause:** Passing `dataframe_split=` kwarg to `serving_endpoints.query()`. The SDK only supports `inputs=`.
**Fix:** Use `inputs=` for all payloads:
```python
response = workspace_client.serving_endpoints.query(
    name=endpoint_name,
    inputs=payload,  # NOT dataframe_split=
)
```

### SCimilarity: NoneType has no attribute 'items' in GetEmbedding
**Symptom:** `AttributeError: 'NoneType' object has no attribute 'items'` inside the GetEmbedding endpoint.
**Root cause:** Passing `"null"` as `celltype_sample_obs` JSON. The model calls `pd.read_json("null")` which returns None.
**Fix:** Pass an empty DataFrame JSON instead of `"null"`:
```python
empty_obs = pd.DataFrame(index=normed_df.index)
obs_json = empty_obs.to_json(orient="split")
```

### SCimilarity: GetEmbedding expects celltype_subsample format
**Symptom:** Empty or wrong embeddings returned.
**Root cause:** The model expects `celltype_sample` as a JSON DataFrame where each row has a `celltype_subsample` column containing a list of expression values — NOT a raw gene-column DataFrame.
**Fix:** Wrap expression values:
```python
dense_rows = normed_df.values.tolist()
celltype_subsample_pdf = pd.DataFrame(
    [{"celltype_subsample": row} for row in dense_rows],
    index=normed_df.index,
)
celltype_sample_json = celltype_subsample_pdf.to_json(orient="split")
```

### Boltz: Missing inputs 'input', 'msa', 'use_msa_server'
**Symptom:** `Model is missing inputs ['input', 'msa', 'use_msa_server']`
**Root cause:** Passing a plain sequence string instead of the expected dict format.
**Fix:** Wrap the sequence in Boltz's expected format:
```python
payload = [{
    "input": f"protein_A:{sequence}",
    "msa": "no_msa",
    "use_msa_server": "True",
}]
```

### Endpoint model name case sensitivity
**Symptom:** `Unknown model 'boltz'` or similar.
**Root cause:** `_MODEL_ENDPOINT_MAP` keys are case-sensitive. `"Boltz"` != `"boltz"`.
**Fix:** Match the exact case in the map when calling `get_endpoint_name()`.

### SearchNearest takes very long
**Symptom:** Cell type annotation or similarity search hangs during neighbor search.
**Root cause:** Too many cells * too many neighbors = thousands of REST calls.
**Fix:** Reduce defaults: `cells_per_cluster=10`, `k_neighbors=20` is sufficient for annotation. Each search call hits the 23M-cell FAISS index.

## Deployment Optimization

### SCimilarity: 60+ minute download on every deploy
**Symptom:** wget step takes 60+ minutes every time.
**Root cause:** No check for existing files — always re-downloads from Zenodo.
**Fix:** Check for extracted model file before downloading:
```python
if os.path.exists(gene_order_path):
    logger.info("Model already extracted, skipping download")
    return
```

### scGPT: Model weights re-loaded on every predict() call
**Symptom:** First request takes 60-90s, subsequent requests also slow.
**Root cause:** `torch.load()` called inside `predict()` instead of `load_context()`.
**Fix:** Pre-load model in `load_context()` and reference `self._loaded_model` in `predict()`.

### GPU nodes used for download/CPU tasks
**Symptom:** Expensive GPU hours wasted on I/O-bound or CPU-bound tasks.
**Fix:** Use `cpu_node_type` and `cpu-ml-scala2.12` runtime for wget and GeneOrder job clusters.

## AWS-Specific Issues

### ON_DEMAND not enforced on initial deploy
**Symptom:** Job clusters use SPOT despite YAML specifying ON_DEMAND.
**Root cause:** DAB bug on initial deployment.
**Fix:** Check and fix via CLI:
```bash
databricks jobs get <job_id>
databricks jobs reset --json '<updated_spec>'
```

### AlphaFold download failures
**Symptom:** FTP/rsync downloads fail or produce empty files.
**Root cause:** VPC may block FTP. Figshare/Zenodo may reject non-browser user agents.
**Fix:** Use HTTPS URLs and `api.figshare.com/v2/file/download/` instead of `ndownloader` URLs.

## UI Issues

### Mol* viewer inconsistent styling
**Symptom:** Some viewers have dark background, others have white/light background.
**Root cause:** `molstar_tools.py` was missing the `MOLSTAR_DARK_CSS` that `small_molecule_tools.py` has.
**Fix:** Add `MOLSTAR_DARK_CSS` to `molstar_tools.py` `<head>` section and standardize viewer div to `width: 100%; height: 500px`.

### destroy.sh fails with "rm: .deployed: No such file"
**Symptom:** destroy.sh exits with error if module was never deployed.
**Fix:** Use `rm -f .deployed` (force flag) in all destroy.sh scripts.

## Instructions

1. When a user reports a deployment error, first identify which module and which step failed (download, registration, endpoint deployment, or UI).
2. Check if the error matches one of the patterns above.
3. For model registration failures, the traceback usually points to the `predict()` or `load_context()` method — look for dtype mismatches, missing attributes, or wrong input formats.
4. For endpoint failures, check the endpoint logs in the Databricks workspace (Serving → endpoint name → Logs tab).
5. For UI errors, check the Streamlit app logs in the Databricks App logs.
6. Always check `CHANGELOG.md` for the latest known issues and fixes.

## When to Use This Skill

- User reports an error during Genesis Workbench deployment or model registration
- User sees endpoint failures or timeout errors
- User encounters dtype mismatches, schema errors, or serialization issues
- User needs to optimize deployment speed or reduce costs
- User has Mol* viewer display issues
