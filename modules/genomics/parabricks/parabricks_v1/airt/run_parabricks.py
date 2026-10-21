#!/usr/bin/env python3
"""Parabricks germline workload for the serverless-GPU ai_runtime_task (run_parabricks job).

Runs INSIDE the stock NGC Parabricks container on serverless GPU_1xA10:
    pbrun fq2bam      (GPU alignment: paired FASTQ + ref -> sorted BAM, +BQSR)
    pbrun deepvariant (GPU variant calling: BAM -> VCF)

Launched on demand from the GWB UI (jobs run-now on this job's id), NOT at deploy time.

Inputs (robust to how the ai_runtime_task receives them — per Databricks docs an
ai_runtime_task does not support job task values, so inputs may arrive two ways):
  1. Env vars PB_FQ1/PB_FQ2/PB_REF/PB_KNOWN_SITES — command.sh maps the job's run-now
     job_parameters (pb_fq1, ...) to these if they surface in the container.
  2. A shared UC-Volume config JSON (the docs' recommended input channel): a JSON object
     of PB_* keys at $PB_CONFIG (default {VOL}/run_config.json), written by the dispatcher.
If no user inputs are supplied, falls back to the NVIDIA getting-started sample (downloaded
to node-local scratch) so a parameter-less run still demonstrates the pipeline.

Data staging: /Volumes is read+write inside the container, but the multi-GB reference is
extracted to node-local /mnt/work (fast NVMe; pbrun random-accesses it hard). Outputs are
copied to the Volume so they persist beyond the ephemeral node.
"""
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


CATALOG = _env("PB_CATALOG", "main")
SCHEMA = _env("PB_SCHEMA", "genesis_workbench")
CACHE_DIR = _env("PB_CACHE_DIR", "parabricks")
VOL = f"/Volumes/{CATALOG}/{SCHEMA}/{CACHE_DIR}"


def load_shared_config() -> None:
    """Merge a shared UC-Volume config JSON into the environment (docs' input channel for
    ai_runtime_task). Only fills keys not already set by env, so run-now params win."""
    cfg_path = _env("PB_CONFIG", f"{VOL}/run_config.json")
    if not os.path.exists(cfg_path):
        return
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            os.environ.setdefault(k, str(v))
        print(f"[config] merged {len(cfg)} keys from {cfg_path}", flush=True)
    except Exception as e:  # a malformed config must not abort — fall back to sample
        print(f"[config] WARN could not read {cfg_path}: {e}", flush=True)


load_shared_config()

# Node-local scratch: fast NVMe, ephemeral. pbrun's working set lives here.
WORK = _env("PB_WORK_DIR", "/mnt/work/parabricks")
LOCAL_SAMPLE = f"{WORK}/parabricks_sample"

# User-supplied inputs (from run-now params via command.sh, or the shared config).
USER_REF = _env("PB_REF")
USER_FQ1 = _env("PB_FQ1")
USER_FQ2 = _env("PB_FQ2")
USER_KNOWN = _env("PB_KNOWN_SITES")
USE_SAMPLE = not (USER_REF and USER_FQ1 and USER_FQ2)

if USE_SAMPLE:
    # NVIDIA getting-started sample (ref assembly38 + prebuilt BWA index + paired FASTQs).
    SAMPLE_URL = _env("PB_SAMPLE_URL", "https://s3.amazonaws.com/parabricks.sample/parabricks_sample.tar.gz")
    VOL_TARBALL = _env("PB_TARBALL_CACHE", f"{VOL}/parabricks_sample.tar.gz")
    CACHE_TARBALL = _env("PB_CACHE_TARBALL", "false").lower() in ("1", "true", "yes")
    REF = f"{LOCAL_SAMPLE}/Ref/Homo_sapiens_assembly38.fasta"
    KNOWN_SITES = f"{LOCAL_SAMPLE}/Ref/Homo_sapiens_assembly38.known_indels.vcf.gz"
    FQ1 = f"{LOCAL_SAMPLE}/Data/sample_1.fq.gz"
    FQ2 = f"{LOCAL_SAMPLE}/Data/sample_2.fq.gz"
else:
    REF, FQ1, FQ2, KNOWN_SITES = USER_REF, USER_FQ1, USER_FQ2, USER_KNOWN

# Outputs -> node-local scratch, then copied to the Volume. Per-run subdir when the
# dispatcher passes an mlflow run id, so concurrent runs don't clobber each other.
LOCAL_OUT = f"{WORK}/output"
OUT_BAM = f"{LOCAL_OUT}/fq2bam.bam"
OUT_RECAL = f"{LOCAL_OUT}/recal.txt"
OUT_VCF = f"{LOCAL_OUT}/deepvariant.vcf"
_run_id = _env("PB_MLFLOW_RUN_ID")
VOL_OUT = _env("PB_OUT_DIR", f"{VOL}/output/{_run_id}" if _run_id else f"{VOL}/output")

NUM_GPUS = _env("PB_NUM_GPUS", "1")  # GPU_1xA10 -> 1
LOW_MEMORY = _env("PB_LOW_MEMORY", "true").lower() in ("1", "true", "yes")  # A10 24GB


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    print(f"[exit {rc} in {time.time()-t0:.0f}s]", flush=True)
    if rc != 0:
        sys.exit(rc)


def download(url: str, dest: str, retries: int = 3) -> None:
    """Stream a URL to dest using the Python stdlib (the stock NGC image has no curl/wget)."""
    t0 = time.time()
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as f:
                shutil.copyfileobj(resp, f, length=8 * 1024 * 1024)
            print(f"[stage] downloaded {os.path.getsize(dest):,} bytes in {time.time()-t0:.0f}s", flush=True)
            return
        except Exception as e:
            last_err = e
            print(f"[stage] download attempt {attempt}/{retries} failed: {e}", flush=True)
            if os.path.exists(dest):
                os.remove(dest)
            time.sleep(5)
    raise RuntimeError(f"download of {url} failed after {retries} attempts: {last_err}")


def ensure_sample_data() -> None:
    """Populate node-local scratch with the NVIDIA sample inputs (idempotent)."""
    if all(os.path.exists(p) for p in (REF, KNOWN_SITES, FQ1, FQ2)):
        print(f"[stage] sample data already present under {LOCAL_SAMPLE}", flush=True)
        return
    os.makedirs(WORK, exist_ok=True)
    local_tb = f"{WORK}/parabricks_sample.tar.gz"
    if not os.path.exists(local_tb):
        if os.path.exists(VOL_TARBALL):
            print(f"[stage] copying cached tarball {VOL_TARBALL} -> {local_tb}", flush=True)
            shutil.copy2(VOL_TARBALL, local_tb)
        else:
            print(f"[stage] downloading {SAMPLE_URL} -> {local_tb}", flush=True)
            part = f"{local_tb}.part"
            download(SAMPLE_URL, part)
            os.replace(part, local_tb)
    print(f"[stage] extracting {local_tb} -> {WORK}", flush=True)
    run(["tar", "-xzf", local_tb, "-C", WORK])
    if CACHE_TARBALL and not os.path.exists(VOL_TARBALL):
        try:
            os.makedirs(VOL, exist_ok=True)
            print(f"[stage] caching tarball to Volume {VOL_TARBALL}", flush=True)
            shutil.copy2(local_tb, VOL_TARBALL)
        except Exception as e:
            print(f"[stage] WARN could not cache tarball to Volume: {e}", flush=True)


def preflight() -> None:
    print("=== parabricks preflight ===", flush=True)
    print(f"mode={'sample' if USE_SAMPLE else 'user-supplied'}  VOL={VOL}  WORK={WORK}", flush=True)
    for label, p in (("REF", REF), ("KNOWN_SITES", KNOWN_SITES), ("FQ1", FQ1), ("FQ2", FQ2)):
        marker = "OK " if (p and os.path.exists(p)) else ("MISSING" if p else "unset")
        print(f"  {label} {marker} {p}", flush=True)
    os.makedirs(LOCAL_OUT, exist_ok=True)
    for probe in (["nvidia-smi", "-L"], ["pbrun", "version"]):
        try:
            run(probe)
        except FileNotFoundError:
            print(f"  WARN: {probe[0]} not found on PATH", flush=True)


def persist_outputs() -> None:
    try:
        os.makedirs(VOL_OUT, exist_ok=True)
    except Exception as e:
        print(f"[persist] WARN could not create {VOL_OUT}: {e}", flush=True)
        return
    for src in (OUT_BAM, f"{OUT_BAM}.bai", OUT_RECAL, OUT_VCF):
        if os.path.exists(src):
            dst = f"{VOL_OUT}/{os.path.basename(src)}"
            print(f"[persist] {src} -> {dst}", flush=True)
            shutil.copy2(src, dst)


def main() -> None:
    if USE_SAMPLE:
        ensure_sample_data()
    preflight()

    fq2bam = [
        "pbrun", "fq2bam",
        "--ref", REF,
        "--in-fq", FQ1, FQ2,
        "--out-bam", OUT_BAM,
        "--out-recal-file", OUT_RECAL,
        "--num-gpus", NUM_GPUS,
    ]
    if KNOWN_SITES:
        fq2bam[7:7] = ["--knownSites", KNOWN_SITES]  # keep --knownSites before outputs
    if LOW_MEMORY:
        fq2bam.append("--low-memory")
    run(fq2bam)

    run([
        "pbrun", "deepvariant",
        "--ref", REF,
        "--in-bam", OUT_BAM,
        "--out-variants", OUT_VCF,
        "--num-gpus", NUM_GPUS,
    ])

    persist_outputs()
    print(f"\n=== DONE ===\n  BAM: {VOL_OUT}/{os.path.basename(OUT_BAM)}"
          f"\n  VCF: {VOL_OUT}/{os.path.basename(OUT_VCF)}", flush=True)


if __name__ == "__main__":
    main()
