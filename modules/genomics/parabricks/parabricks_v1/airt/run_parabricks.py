#!/usr/bin/env python3
"""Parabricks germline workload for the serverless-GPU AI Runtime (`air run`).

Runs INSIDE the Parabricks container (docker_image.url on serverless GPU_1xA10):
    pbrun fq2bam      (GPU alignment: paired FASTQ + ref -> sorted BAM, +BQSR)
    pbrun deepvariant (GPU variant calling: BAM -> VCF)

This replaces the classic-cluster interactive notebook (run_parabricks.html), which
can't run on the EC2-GPU-quota=0 workshop account. `air run` packages this dir (see
parabricks_workload.yaml: code_source) and runs `command` from $CODE_SOURCE_PATH; config
comes from env vars, NOT dbutils widgets.

Data staging (confirmed working on dbc-3d5f56ea, 2026-09-22):
  * /Volumes IS readable AND writable inside the AI Runtime container, so we cache the
    NVIDIA sample tarball there and persist outputs there.
  * The multi-GB reference + BWA index are extracted to node-local scratch (/mnt/work)
    rather than read over the /Volumes FUSE mount — pbrun does heavy random access on the
    reference during alignment, which is far faster on local NVMe than FUSE.
Flow: download parabricks_sample.tar.gz (skip if Volume-cached) -> extract to /mnt/work
-> pbrun (local I/O) -> copy BAM/recal/VCF to /Volumes/<cat>/<schema>/parabricks/output.
"""
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

# NVIDIA Parabricks getting-started sample (ref assembly38 + prebuilt BWA index + paired
# FASTQs + known-indels VCF). The tarball's top-level dir is `parabricks_sample/`.
SAMPLE_URL = _env(
    "PB_SAMPLE_URL",
    "https://s3.amazonaws.com/parabricks.sample/parabricks_sample.tar.gz",
)
VOL_TARBALL = _env("PB_TARBALL_CACHE", f"{VOL}/parabricks_sample.tar.gz")
# Caching the 11.8 GB tarball to the Volume speeds re-runs but costs a slow FUSE write on
# the first run; off by default so a validation run goes straight download -> extract -> pbrun.
CACHE_TARBALL = _env("PB_CACHE_TARBALL", "false").lower() in ("1", "true", "yes")

# Node-local scratch: fast NVMe, ephemeral. pbrun's working set (ref + index + BAM) lives
# here; only the tarball cache and final outputs touch the persistent Volume.
WORK = _env("PB_WORK_DIR", "/mnt/work/parabricks")
LOCAL_SAMPLE = f"{WORK}/parabricks_sample"
REF = _env("PB_REF", f"{LOCAL_SAMPLE}/Ref/Homo_sapiens_assembly38.fasta")
KNOWN_SITES = _env("PB_KNOWN_SITES", f"{LOCAL_SAMPLE}/Ref/Homo_sapiens_assembly38.known_indels.vcf.gz")
FQ1 = _env("PB_FQ1", f"{LOCAL_SAMPLE}/Data/sample_1.fq.gz")
FQ2 = _env("PB_FQ2", f"{LOCAL_SAMPLE}/Data/sample_2.fq.gz")

# Outputs: written to node-local scratch, then copied to the Volume so they persist beyond
# the ephemeral node.
LOCAL_OUT = f"{WORK}/output"
OUT_BAM = f"{LOCAL_OUT}/fq2bam.bam"
OUT_RECAL = f"{LOCAL_OUT}/recal.txt"
OUT_VCF = f"{LOCAL_OUT}/deepvariant.vcf"
VOL_OUT = _env("PB_OUT_DIR", f"{VOL}/output")

NUM_GPUS = _env("PB_NUM_GPUS", "1")  # GPU_1xA10 -> 1
LOW_MEMORY = _env("PB_LOW_MEMORY", "true").lower() in ("1", "true", "yes")  # A10 24GB


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    t0 = time.time()
    # Stream child stdout/stderr straight through so it lands in the AI Runtime run's logs.
    rc = subprocess.run(cmd).returncode
    print(f"[exit {rc} in {time.time()-t0:.0f}s]", flush=True)
    if rc != 0:
        sys.exit(rc)


def download(url: str, dest: str, retries: int = 3) -> None:
    """Stream a URL to dest using the Python stdlib (no curl/wget dependency).

    The stock NVIDIA NGC Parabricks image ships neither curl nor wget, so relying on
    python3 (always present) keeps the pipeline runnable on the bare NGC image as well as
    the custom Databricks-scaffolded one.
    """
    t0 = time.time()
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as f:
                shutil.copyfileobj(resp, f, length=8 * 1024 * 1024)
            print(f"[stage] downloaded {os.path.getsize(dest):,} bytes in {time.time()-t0:.0f}s", flush=True)
            return
        except Exception as e:  # network hiccup -> retry a couple times before giving up
            last_err = e
            print(f"[stage] download attempt {attempt}/{retries} failed: {e}", flush=True)
            if os.path.exists(dest):
                os.remove(dest)
            time.sleep(5)
    raise RuntimeError(f"download of {url} failed after {retries} attempts: {last_err}")


def ensure_sample_data() -> None:
    """Populate node-local scratch with the Parabricks sample inputs (idempotent).

    Prefers the Volume-cached tarball; downloads from NVIDIA's public S3 only on a cold
    cache, then writes the tarball back to the Volume so later runs skip the download.
    """
    inputs = (REF, KNOWN_SITES, FQ1, FQ2)
    if all(os.path.exists(p) for p in inputs):
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

    # Best-effort tarball cache to the Volume (opt-in) — done after extraction so a slow
    # FUSE write never delays pbrun; a failure must not abort the run.
    if CACHE_TARBALL and not os.path.exists(VOL_TARBALL):
        try:
            os.makedirs(VOL, exist_ok=True)
            print(f"[stage] caching tarball to Volume {VOL_TARBALL}", flush=True)
            shutil.copy2(local_tb, VOL_TARBALL)
        except Exception as e:
            print(f"[stage] WARN could not cache tarball to Volume: {e}", flush=True)


def preflight() -> None:
    print("=== parabricks air_run preflight ===", flush=True)
    print(f"VOL={VOL}  WORK={WORK}", flush=True)
    for p in (REF, KNOWN_SITES, FQ1, FQ2):
        print(f"  input {'OK ' if os.path.exists(p) else 'MISSING'} {p}", flush=True)
    os.makedirs(LOCAL_OUT, exist_ok=True)
    # Confirm the GPU + pbrun are visible inside the container.
    for probe in (["nvidia-smi", "-L"], ["pbrun", "version"]):
        try:
            run(probe)
        except FileNotFoundError:
            print(f"  WARN: {probe[0]} not found on PATH", flush=True)


def persist_outputs() -> None:
    """Copy the pbrun outputs from node-local scratch to the Volume for persistence."""
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
    ensure_sample_data()
    preflight()

    fq2bam = [
        "pbrun", "fq2bam",
        "--ref", REF,
        "--in-fq", FQ1, FQ2,
        "--knownSites", KNOWN_SITES,
        "--out-bam", OUT_BAM,
        "--out-recal-file", OUT_RECAL,
        "--num-gpus", NUM_GPUS,
    ]
    if LOW_MEMORY:
        fq2bam.append("--low-memory")
    run(fq2bam)

    deepvariant = [
        "pbrun", "deepvariant",
        "--ref", REF,
        "--in-bam", OUT_BAM,
        "--out-variants", OUT_VCF,
        "--num-gpus", NUM_GPUS,
    ]
    run(deepvariant)

    persist_outputs()
    print(f"\n=== DONE ===\n  BAM: {VOL_OUT}/{os.path.basename(OUT_BAM)}"
          f"\n  VCF: {VOL_OUT}/{os.path.basename(OUT_VCF)}", flush=True)


if __name__ == "__main__":
    main()
