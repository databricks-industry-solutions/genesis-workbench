#!/usr/bin/env python3
"""GWAS Parabricks alignment for the serverless-GPU ai_runtime_task (run_germline task).

Runs INSIDE the stock NGC Parabricks container on serverless GPU_1xA10:
    pbrun fq2bam          (GPU alignment: paired FASTQ + ref -> sorted BAM)
    pbrun haplotypecaller (GPU germline variant calling: BAM -> VCF)

This replaces notebooks/02_parabricks_germline.py, which ran on a classic a10 GPU cluster
with a custom Parabricks Docker container (can't provision at EC2 GPU quota=0). Launched on
demand from the GWB UI (jobs run-now on parabricks_alignment) — the mark_success/mark_failure
notebook tasks handle the MLflow job_status, so this script just runs pbrun and writes outputs.

Inputs come from the UI's run-now job_parameters. Because an ai_runtime_task doesn't support
job task values (Databricks docs), inputs arrive as env vars (command.sh maps the job params)
OR via a shared UC-Volume config JSON at $PB_CONFIG. /Volumes is read+write in the container,
so the user's reference + FASTQs are read directly from their Volume paths and outputs are
written under {output_volume}/alignment/{mlflow_run_id}/ (per-run).
"""
import json
import os
import shlex
import subprocess
import sys
import time


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def load_shared_config() -> None:
    """Merge a shared UC-Volume config JSON into the env (docs' input channel for
    ai_runtime_task). Only fills keys not already set, so run-now env params win."""
    cfg = _env("PB_CONFIG")
    if not cfg or not os.path.exists(cfg):
        return
    try:
        for k, v in json.load(open(cfg)).items():
            os.environ.setdefault(k, str(v))
        print(f"[config] merged config from {cfg}", flush=True)
    except Exception as e:
        print(f"[config] WARN could not read {cfg}: {e}", flush=True)


load_shared_config()

FASTQ_R1 = _env("PB_FASTQ_R1")
FASTQ_R2 = _env("PB_FASTQ_R2")
REF = _env("PB_REF")
OUTPUT_VOLUME = _env("PB_OUTPUT_VOLUME")
MLFLOW_RUN_ID = _env("PB_MLFLOW_RUN_ID")
LOW_MEMORY = _env("PB_LOW_MEMORY", "true").lower() in ("1", "true", "yes")
NUM_GPUS = _env("PB_NUM_GPUS", "1")

_required = {
    "PB_FASTQ_R1": FASTQ_R1, "PB_FASTQ_R2": FASTQ_R2, "PB_REF": REF,
    "PB_OUTPUT_VOLUME": OUTPUT_VOLUME, "PB_MLFLOW_RUN_ID": MLFLOW_RUN_ID,
}
_missing = [k for k, v in _required.items() if not v]
if _missing:
    sys.exit(f"ERROR: missing required inputs {_missing} (pass via run-now job_parameters "
             f"or a $PB_CONFIG shared-Volume JSON)")

OUT_DIR = f"{OUTPUT_VOLUME}/alignment/{MLFLOW_RUN_ID}"
OUT_BAM = f"{OUT_DIR}/output.bam"
OUT_VCF = f"{OUT_DIR}/germline.vcf"


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    print(f"[exit {rc} in {time.time()-t0:.0f}s]", flush=True)
    if rc != 0:
        sys.exit(rc)


def preflight() -> None:
    print("=== gwas alignment preflight ===", flush=True)
    print(f"OUT_DIR={OUT_DIR}", flush=True)
    for label, p in (("FASTQ_R1", FASTQ_R1), ("FASTQ_R2", FASTQ_R2), ("REF", REF)):
        marker = "OK " if os.path.exists(p) else "MISSING"
        print(f"  {label} {marker} {p}", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    for probe in (["nvidia-smi", "-L"], ["pbrun", "version"]):
        try:
            run(probe)
        except FileNotFoundError:
            print(f"  WARN: {probe[0]} not found on PATH", flush=True)


def main() -> None:
    preflight()

    fq2bam = [
        "pbrun", "fq2bam",
        "--ref", REF,
        "--in-fq", FASTQ_R1, FASTQ_R2,
        "--out-bam", OUT_BAM,
        "--num-gpus", NUM_GPUS,
    ]
    if LOW_MEMORY:
        fq2bam.append("--low-memory")
    run(fq2bam)

    run([
        "pbrun", "haplotypecaller",
        "--ref", REF,
        "--in-bam", OUT_BAM,
        "--out-variants", OUT_VCF,
        "--num-gpus", NUM_GPUS,
    ])

    print(f"\n=== DONE ===\n  BAM: {OUT_BAM}\n  VCF: {OUT_VCF}", flush=True)


if __name__ == "__main__":
    main()
