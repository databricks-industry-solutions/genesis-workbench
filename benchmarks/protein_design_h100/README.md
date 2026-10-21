# Protein-design speed + memory benchmark (3-way, H100)

Compares **generation speed** and **GPU memory footprint** of three protein-design models by
loading each registered UC model from MLflow and running it **in-process** on a dedicated **1× H100**
(so NVML/`torch.cuda` can measure memory cleanly — serving endpoints can't expose that). Produces full
plots + a 2-panel/3-bar summary (median generation time; peak GPU memory).

## Models & modes

All three run on the **same template set** and the **same hardware** (1× H100, serverless `client '6'` /
CUDA 13). They don't share a de-novo task (rfdiffusion is inpainting-only, proteina_complexa is
target-conditional, only rfd4 does de-novo), so each runs its **native conditional mode at matched
generated-length** on the shared idealized backbone template:

| Model | UC model | Native mode measured |
|---|---|---|
| RFD4-Proteina | `main.genesis_workbench.rfd4_proteina/18` | binder design (length L) |
| RFdiffusion (RFD3) | `main.genesis_workbench.rfdiffusion_inpainting/4` | inpainting (regenerate span L) |
| Proteina-Complexa | `main.genesis_workbench.proteina_complexa_ame/6` | AME motif-scaffolding (scaffold L) |

Each bar is labeled with its mode; the comparison is matched on template + generated-length (the driver
of speed/memory), not on identical conditioning semantics.

## Files

- `make_dataset.py` — deterministic shared input set (idealized α-helix backbone templates; a length
  sweep so plots can show time/memory vs length). `%run`-able + importable.
- `bench_common.py` — model-agnostic harness: CUDA-synced per-sample timing, peak `torch` alloc/reserved,
  and a background **NVML** sampler for peak device memory + GPU utilization; writes per-model CSV + JSON.
- `nb_rfd4.py` / `nb_rfdiffusion.py` / `nb_proteina.py` — per-model: install the model's code deps
  (env_pack models log only minimal requirements, so the real install is still needed), `load_model`
  (which fetches the logged checkpoints), warmup, then the measured loop.
- `nb_aggregate.py` — reads the three result files → full plots + the summary plot.
- `databricks.yml` — the job: 3 parallel H100 tasks (client '6') + a CPU aggregate task.

## Run

```bash
databricks bundle deploy -t dev -p workshop
databricks bundle run    -t dev -p workshop protein_design_benchmark
# full run: set the max_samples job parameter to 1000 (rfdiffusion ~15 s/sample => hours).
```

Results + PNGs land in the `out_dir` Volume (`summary_speed_memory.png`, `detail_vs_length.png`,
`<model>_samples.csv`, `<model>_summary.json`).

## Caveats

- **GPU parity relies on forward-compat:** rfdiffusion + proteina register on `client '4'` (CUDA 12.x);
  running them on `client '6'` (CUDA 13) leans on CUDA driver forward-compatibility — the warmup cell is
  the canary. If a model can't load on client '6', run that one on its native client (losing strict
  same-GPU parity) and note it.
- **H100 capacity** on this workshop is intermittent (hence `max_retries: 2`); 3 at once may not provision
  — fall back to running the tasks sequentially if needed.
- **Default `max_samples=50`** validates the rig fast; raise to 1000 for the full set.
