# DiffDock v1 Changelog

## 2026-04-08 — Inference Performance Optimization

### Problem

DiffDock inference was consistently exceeding the Databricks Model Serving
300-second (5-minute) hard timeout, causing requests to fail even after
warm-up. A single protein-ligand docking request was taking 6-18 minutes
on an A10G GPU.

### Root Cause Analysis

| Bottleneck | Impact |
|---|---|
| **FP32 precision** | Not leveraging A10G tensor cores; ~2x slower than FP16 |
| **19 diffusion steps** | Each step = one forward pass through the score model GNN |
| **10 poses per complex** | Linear scaling of sampling time with pose count |
| **Original GitHub DiffDock** | No NVIDIA CUDA kernel optimizations |

Reference: DiffDock-L's score model (~20M params) should complete single-complex
docking in under 60 seconds on an A10G when properly configured
(see: DiffDock-FPGA, GLSVLSI '25, Zhang et al.).

### Changes

1. **FP16 mixed precision** — Wrapped the diffusion sampling loop with
   `torch.amp.autocast(device_type="cuda", dtype=torch.float16)` to leverage
   A10G tensor cores. Expected ~2x speedup. The score model's equivariant
   layers (e3nn tensor products) tolerate reduced precision well, as
   demonstrated by the DiffDock-FPGA paper achieving comparable accuracy
   with fixed-point arithmetic.

2. **Reduced inference steps** — Changed from 19 to 10 diffusion steps
   (schedule generation from 20 to 11). Prior work shows 10 steps retains
   most pose quality while roughly halving sampling time.

3. **Reduced default poses** — Changed `samples_per_complex` default from
   10 to 5. Fewer poses = proportionally less sampling and confidence
   scoring time. Users can still request more via the input parameter.

4. **Reduced batch size** — Changed internal `batch_size` from 10 to 5
   to match default pose count and reduce peak GPU memory.

### Expected Performance

| Metric | Before | After | Improvement |
|---|---|---|---|
| Inference steps | 19 | 10 | ~1.9x |
| Default poses | 10 | 5 | ~2x |
| Precision | FP32 | FP16 | ~2x |
| **Estimated total** | **6-18 min** | **~45-90 sec** | **~4-8x** |

### Future Optimizations

- **NVIDIA DiffDock NIM** — NVIDIA's optimized version includes custom CUDA
  kernels for equivariant GNN operations (claimed 2x additional speedup).
  Would replace the GitHub v1.1.3 source.

- **cuEquivariance** — NVIDIA's `cuequivariance` library is a drop-in
  replacement for e3nn tensor product operations, exploiting Clebsch-Gordan
  sparsity patterns. Claims 3-5x speedup on TP-heavy layers. Currently
  using e3nn 0.5.1.

- **Async job fallback** — For very large proteins that still exceed 300s,
  implement the AlphaFold2-style async job pattern (submit job, poll for
  results) as a graceful degradation path.
