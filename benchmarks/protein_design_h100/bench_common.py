# Databricks notebook source
"""Shared, model-agnostic measurement harness for the 3-way protein-design H100 benchmark.

Each model notebook (rfd4_proteina / rfdiffusion / proteina_complexa) imports this so all
three are measured identically:

  * generation wall-time per sample (CUDA-synchronized, warmup excluded),
  * peak torch-allocated / reserved GPU memory per sample (`torch.cuda`),
  * device-level peak memory + GPU utilization sampled in a background thread via NVML
    (the nvidia tool — `pynvml`, with an `nvidia-smi` fallback).

Results are written per-model to a shared UC Volume so the aggregation notebook can merge
the three and render the full + summary plots. The notebooks run on separate dedicated
H100s, so each sampler owns its device (device-level numbers are clean, not cross-contaminated).
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from contextlib import contextmanager

GiB = 1024 ** 3


def _nvml_handle():
    import pynvml

    pynvml.nvmlInit()
    return pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)


def gpu_info() -> dict:
    """GPU name + total memory, via NVML with an nvidia-smi fallback."""
    try:
        pynvml, h = _nvml_handle()
        name = pynvml.nvmlDeviceGetName(h)
        name = name.decode() if isinstance(name, bytes) else name
        total = int(pynvml.nvmlDeviceGetMemoryInfo(h).total)
        return {"gpu_name": name, "gpu_total_gib": round(total / GiB, 3)}
    except Exception:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"], text=True
            ).strip().splitlines()[0]
            name, total_mib = [x.strip() for x in out.split(",")]
            return {"gpu_name": name, "gpu_total_gib": round(float(total_mib) / 1024, 3)}
        except Exception as exc:  # noqa: BLE001
            return {"gpu_name": f"unknown ({type(exc).__name__})", "gpu_total_gib": 0.0}


class DeviceSampler:
    """Background NVML sampler: peak device `memory.used` + GPU utilization over its lifetime.

    Use as a context manager spanning the whole measured generation loop. On a dedicated
    GPU this is the real device-level footprint (CUDA context + framework + tensors) — the
    "GiB" number a user actually cares about, distinct from torch's allocator view.
    """

    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._t: threading.Thread | None = None
        self.peak_used_bytes = 0
        self.util_samples: list[int] = []
        self.mem_series: list[tuple[float, int]] = []  # (t_since_start_s, used_bytes)

    def _run(self):
        try:
            pynvml, h = _nvml_handle()
        except Exception:
            return
        t0 = time.perf_counter()
        while not self._stop.is_set():
            try:
                used = int(pynvml.nvmlDeviceGetMemoryInfo(h).used)
                util = int(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
                self.peak_used_bytes = max(self.peak_used_bytes, used)
                self.util_samples.append(util)
                self.mem_series.append((round(time.perf_counter() - t0, 3), used))
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "DeviceSampler":
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t:
            self._t.join(timeout=2)

    @property
    def peak_used_gib(self) -> float:
        return round(self.peak_used_bytes / GiB, 3)

    @property
    def mean_util(self) -> float:
        return round(sum(self.util_samples) / len(self.util_samples), 1) if self.util_samples else 0.0


@contextmanager
def measure_generation(torch):
    """Time ONE generation (CUDA-synced) + capture peak torch memory.

    Yields a dict that is filled on exit with: seconds, peak_torch_alloc_gib,
    peak_torch_reserved_gib. Caller records its own fields (sample_id, length, ok) into it.
    """
    rec: dict = {}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        yield rec
    finally:
        torch.cuda.synchronize()
        rec["seconds"] = round(time.perf_counter() - t0, 4)
        rec["peak_torch_alloc_gib"] = round(torch.cuda.max_memory_allocated() / GiB, 4)
        rec["peak_torch_reserved_gib"] = round(torch.cuda.max_memory_reserved() / GiB, 4)


def write_results(out_dir: str, model: str, records: list[dict], sampler: DeviceSampler | None = None,
                  extra: dict | None = None) -> dict:
    """Write per-sample CSV + a summary JSON for one model into the shared out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    import pandas as pd

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(out_dir, f"{model}_samples.csv"), index=False)

    ok = df[df.get("ok", True) == True] if "ok" in df.columns else df  # noqa: E712
    summary = {"model": model, "n_samples": int(len(df)), "n_ok": int(len(ok)), **gpu_info()}
    if len(ok):
        summary.update({
            "median_seconds": round(float(ok["seconds"].median()), 4),
            "mean_seconds": round(float(ok["seconds"].mean()), 4),
            "peak_torch_alloc_gib": round(float(ok["peak_torch_alloc_gib"].max()), 4),
            "peak_torch_reserved_gib": round(float(ok["peak_torch_reserved_gib"].max()), 4),
        })
    if sampler is not None:
        summary["peak_device_used_gib"] = sampler.peak_used_gib
        summary["mean_gpu_util_pct"] = sampler.mean_util
        pd.DataFrame(sampler.mem_series, columns=["t_s", "used_bytes"]).to_csv(
            os.path.join(out_dir, f"{model}_mem_series.csv"), index=False)
    if extra:
        summary.update(extra)
    with open(os.path.join(out_dir, f"{model}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def pdb_to_cif(pdb_str: str, out_path: str) -> str:
    """Convert a PDB string to mmCIF (bonds/charges the rfd4/rfd3 parsers want). Mirrors the repo's
    proven converter (antibody_design/utils.py `_pdb_to_cif`): prefer atomworks (the model's own
    parser), fall back to a plain biotite atom_site CIF when the atomworks writer API differs across
    versions — fine for standard residues (e.g. poly-GLY), where bonds are inferred from templates."""
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as f:
        f.write(pdb_str)
        pdb_path = f.name
    try:
        try:
            from atomworks.io.parser import parse as _aw_parse
            try:
                from atomworks.io.writer import to_cif_file as _to_cif
            except Exception:
                from atomworks.io import to_cif_file as _to_cif  # writer location varies by version
            _to_cif(_aw_parse(pdb_path), out_path)
            return out_path
        except Exception as e:  # noqa: BLE001
            print(f"[pdb->cif] atomworks writer unavailable ({type(e).__name__}: {str(e)[:120]}); using biotite")
        import biotite.structure.io.pdb as _pdb
        import biotite.structure.io.pdbx as _pdbx
        arr = _pdb.PDBFile.read(pdb_path).get_structure(model=1)
        cif = _pdbx.CIFFile()
        _pdbx.set_structure(cif, arr)
        cif.write(out_path)
        return out_path
    finally:
        try:
            os.unlink(pdb_path)
        except OSError:
            pass


def run_benchmark(model_name, model, specs, adapt_fn, torch, out_dir, extra=None, warmup=True):
    """Shared measured loop for all three models: warmup, then per-sample timing + peak memory with a
    background NVML sampler. Captures the FIRST failure's full traceback to <model>_error.log so a run
    reveals the real cause (not a truncated string). Writes per-model CSV + summary JSON."""
    import os, traceback
    first_err = None
    if warmup and specs:
        try:
            model.predict(adapt_fn(specs[0]))
        except Exception as e:  # noqa: BLE001
            print("warmup error (continuing):", str(e)[:400])
    records = []
    with DeviceSampler() as smp:
        for s in specs:
            rec = {"sample_id": s["sample_id"], "total_len": s["total_len"],
                   "regen_len": s["regen_len"], "ok": True}
            try:
                with measure_generation(torch) as mrec:
                    model.predict(adapt_fn(s))
                rec.update(mrec)
            except Exception as e:  # noqa: BLE001
                rec["ok"] = False
                rec["error"] = str(e)[:300]
                if first_err is None:
                    first_err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            records.append(rec)
    errlog = os.path.join(out_dir, f"{model_name}_error.log")
    if first_err:
        os.makedirs(out_dir, exist_ok=True)
        with open(errlog, "w") as f:
            f.write(first_err)
        print("FIRST ERROR (full tail):\n", first_err[-2500:])
    elif os.path.exists(errlog):
        os.remove(errlog)  # clear a stale error.log from a prior failed run so a clean run leaves none
    return write_results(out_dir, model_name, records, smp, extra=extra)
