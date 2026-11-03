---
name: variant-calling-from-paper
description: Build a complete, runnable genomics notebook for the user's own sample by generating cells into the current notebook — GPU-accelerated germline variant calling (NVIDIA Parabricks fq2bam + DeepVariant) followed by clinical variant annotation against ClinVar, entirely Glow-free (pure Python/pandas). Use when a user references a research paper or a one-line goal and wants to call variants from sequencing reads and interpret them clinically — e.g. "use this paper to call variants and find pathogenic variants", "GPU variant calling with Parabricks/DeepVariant on my FASTQ", "annotate this VCF against ClinVar", "is my sample a carrier of <disease>", "find the pathogenic variant in <gene>" — in a Databricks workspace with Genesis Workbench (genomics module) deployed.
---

# Variant Calling From a Paper (Genesis Workbench, genomics)

Generate a full **reads → clinically-annotated variants** notebook **into the current notebook**:
dispatch the deployed GPU Parabricks job (fq2bam + DeepVariant) to call germline variants from
sequencing reads, then annotate the resulting VCF against ClinVar — **in pure Python/pandas, no
Glow, no Spark required**. This is the Genie Code version of the Claude Code skill
`claude_skills/SKILL_GENESIS_WORKBENCH_VARIANT_CALLING_FROM_PAPER.md`; keep the two in sync.

## Hard rules
1. **Call variants with the deployed Parabricks GPU job — never run `pbrun` locally.** Resolve the
   `run_parabricks` job by name and launch it with `w.jobs.run_now(...)`. The GPU container reads
   its inputs from a config JSON on a UC Volume (job parameters do NOT reach an `ai_runtime_task`
   container), so write that config first, then launch.
2. **Annotate in pure Python/pandas — do NOT use Glow or Spark.** Parse the VCF as text into a
   DataFrame and join to a ClinVar annotation table (a staged Delta table, or a region CSV). The
   whole post-calling analysis is small (one sample, a gene panel) and needs no distributed engine.
3. **Be honest about scope.** This is a *germline small-variant* workflow on (often region-subset)
   reads; it does not do CNV/SV/somatic calling, phasing, or penetrance/risk prediction. A ClinVar
   "Pathogenic" classification is an assertion about the *variant*, not a diagnosis of the person —
   especially for low-penetrance / recessive alleles (zygosity and inheritance matter).
4. **Fail loud on missing resources.** If the `run_parabricks` job, the reference, the reads, or the
   ClinVar table are missing, raise — don't silently fall back. (Genesis Workbench reaps idle
   compute; the genomics module must be deployed.)

## How to respond (generation procedure)
When the request matches, generate these cells in order:
1. **Markdown** — paper citation (DeepVariant: Poplin et al., *Nat. Biotechnol.* 2018), the one-line
   goal, and the chosen sample + target gene/panel + the variant(s) of interest.
2. **Code: `%pip install -q "databricks-sdk>=0.50.0" pandas` then `dbutils.library.restartPython()`.**
3. **Code: the Toolkit cell** (emit the Python block in *Toolkit* below, unchanged).
4. **Code: Problem definition** — the reads (paired FASTQ paths), reference + BWA index path,
   known-sites path, the output dir, and the ClinVar annotation source + gene panel.
5. **Code: Launch variant calling** — write the config, `run_now` the job, poll to completion,
   locate `deepvariant.vcf`. (Include a `USE_PRECOMPUTED_VCF` switch: in a shared/class setting read
   an already-produced VCF instead of launching a new GPU run — see *Concurrency*.)
6. **Code: Parse + annotate + interpret** (see *Annotation*): VCF→DataFrame, inner-join ClinVar,
   filter to Pathogenic/Likely-pathogenic, report each hit with **zygosity** and gene/condition.
7. **Markdown: Reflection & honest scope.**

## Task inputs (fill these from the paper/goal)
- **Reads** — paired FASTQ (`FQ1`, `FQ2`) on a UC Volume the job can read. For a whole genome, calling
  is slow/expensive; for a teaching run use reads **subset to the target gene locus** (small + fast).
- **Reference** — a GRCh38 FASTA with its `.fai` **and BWA index** (`.amb/.ann/.bwt/.pac/.sa`) on a
  Volume. Reuse the one the genomics module already staged (commonly
  `/Volumes/<catalog>/genesis_workbench/gwas_reference/genomes/GRCh38_full_analysis_set_plus_decoy_hla.fa`).
- **Known-sites** — a bgzipped+tabixed VCF for BQSR (the `run_parabricks` job always does BQSR). Its
  `##contig` lines must match the reference, or `fq2bam` errors.
- **ClinVar table** — the annotation reference, ideally a staged Delta table (or region CSV) filtered
  to the target gene, with columns `chrom,pos,ref,alt,rs,gene,clnsig,clndn,clnrevstat`.
- **Gene / variant of interest** — e.g. gene `HFE`, variant `C282Y` at `chr6:26092913 G>A`.

## Toolkit (emit this as a code cell, unchanged)
```python
import json, time, os
import pandas as pd
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# The deployed GPU job + its fixed config/output Volume (the ai_runtime_task reads this file;
# job parameters do NOT reach the container). PB_VOL default matches the genomics module install.
PB_JOB_NAME = "run_parabricks"
PB_VOL = "/Volumes/main/genesis_workbench/parabricks"   # holds run_config.json + output/

def resolve_job(name):
    """Job id by name; fail loud if the genomics module isn't deployed."""
    jobs = list(w.jobs.list(name=name))
    if not jobs:
        raise RuntimeError(f"Job '{name}' not found — is the Genesis Workbench genomics module deployed?")
    return jobs[0].job_id

def launch_variant_calling(fq1, fq2, ref, out_dir, run_tag, known_sites="", pb_vol=PB_VOL):
    """Write the GPU job's input config to the Volume, then launch it. Returns the Databricks run id.
    out_dir is per-run (keep it unique) so concurrent runs don't clobber each other's outputs."""
    cfg = {"PB_REF": ref, "PB_FQ1": fq1, "PB_FQ2": fq2,
           "PB_MLFLOW_RUN_ID": run_tag, "PB_OUT_DIR": out_dir}
    if known_sites:
        cfg["PB_KNOWN_SITES"] = known_sites
    os.makedirs(pb_vol, exist_ok=True)
    with open(f"{pb_vol}/run_config.json", "w") as f:
        json.dump(cfg, f)
    run = w.jobs.run_now(job_id=resolve_job(PB_JOB_NAME))
    print(f"launched {PB_JOB_NAME} run_id={run.run_id}; outputs -> {out_dir}")
    return run.run_id

def wait_for_run(run_id, poll=45, timeout=5400):
    """Poll a job run to a terminal state; raise on non-success."""
    t0 = time.time()
    while True:
        st = w.jobs.get_run(run_id).state
        life = st.life_cycle_state.value if st.life_cycle_state else "?"
        if life in ("TERMINATED", "INTERNAL_ERROR", "SKIPPED"):
            res = st.result_state.value if st.result_state else ""
            print(f"run {run_id}: {life}/{res}")
            if res != "SUCCESS":
                raise RuntimeError(f"variant-calling run {run_id} ended {life}/{res}: {st.state_message}")
            return
        if time.time() - t0 > timeout:
            raise TimeoutError(f"run {run_id} still {life} after {timeout}s")
        print(f"  … {life} ({int(time.time()-t0)}s)"); time.sleep(poll)

def read_vcf(path):
    """Parse a VCF (text) into a DataFrame: chrom,pos,ref,alt,filter,gt. No Glow/Spark."""
    rows, sample = [], None
    with open(path) as f:
        for ln in f:
            if ln.startswith("#CHROM"):
                cols = ln.rstrip("\n").split("\t"); sample = cols[9] if len(cols) > 9 else None; continue
            if ln.startswith("#"):
                continue
            c = ln.rstrip("\n").split("\t")
            gt = dict(zip(c[8].split(":"), c[9].split(":"))).get("GT", "") if len(c) > 9 else ""
            rows.append({"chrom": c[0], "pos": int(c[1]), "ref": c[3], "alt": c[4],
                         "filter": c[6], "gt": gt})
    df = pd.DataFrame(rows)
    print(f"sample={sample}  calls={len(df)}  PASS={(df['filter']=='PASS').sum() if len(df) else 0}")
    return df

def load_clinvar_table(table_or_csv):
    """ClinVar annotation reference as a DataFrame. Accepts a UC Delta table name
    (catalog.schema.table) via spark, or a CSV path. Columns: chrom,pos,ref,alt,clnsig,clndn,..."""
    if table_or_csv.endswith(".csv"):
        return pd.read_csv(table_or_csv)
    return spark.table(table_or_csv).toPandas()

def zygosity(gt):
    return {"1/1": "homozygous", "1|1": "homozygous",
            "0/1": "heterozygous", "0|1": "heterozygous", "1|0": "heterozygous"}.get(gt, gt)

def annotate(calls_df, clinvar_df):
    """Inner-join called variants to ClinVar on (chrom,pos,ref,alt); add a zygosity column."""
    ann = calls_df.merge(clinvar_df, on=["chrom", "pos", "ref", "alt"], how="inner", suffixes=("", "_cv"))
    if len(ann):
        ann["zygosity"] = ann["gt"].map(zygosity)
    return ann

print("✅ GWB variant-calling toolkit loaded (Parabricks GPU calling + pure-Python ClinVar annotation).")
```

## Launch variant calling (emit; set USE_PRECOMPUTED_VCF for class/shared settings)
```python
RUN_TAG = "my_sample"                    # unique per run/user
OUT_DIR = f"{PB_VOL}/output/{RUN_TAG}"
USE_PRECOMPUTED_VCF = False              # True → skip the GPU run, read an already-produced VCF

if USE_PRECOMPUTED_VCF:
    vcf_path = f"{PRECOMPUTED_OUT_DIR}/deepvariant.vcf"
else:
    run_id = launch_variant_calling(FQ1, FQ2, REF, OUT_DIR, RUN_TAG, known_sites=KNOWN_SITES)
    wait_for_run(run_id)                 # fq2bam + DeepVariant; region-subset reads finish in minutes
    vcf_path = f"{OUT_DIR}/deepvariant.vcf"
```

## Annotation + interpretation (emit)
```python
calls = read_vcf(vcf_path)
clinvar = load_clinvar_table(CLINVAR_TABLE)          # Delta table name or CSV path
ann = annotate(calls, clinvar)

pathogenic = ann[ann["clnsig"].str.contains("athogenic", na=False)]
print(f"ClinVar matches: {len(ann)}   Pathogenic/Likely-pathogenic: {len(pathogenic)}")
for _, r in pathogenic.iterrows():
    print(f"  chr{r['chrom'].lstrip('chr')}:{r['pos']} {r['ref']}>{r['alt']}  "
          f"{r['zygosity']} (GT={r['gt']}, {r['filter']})  gene={r.get('gene','')}\n"
          f"      {r['clnsig']}  —  {str(r.get('clndn',''))[:60]}")
display(pathogenic[["chrom","pos","ref","alt","gt","zygosity","clnsig","gene"]])
```

## Concurrency (important for a shared/class workspace)
`run_parabricks` reads a **single fixed** config file (`{PB_VOL}/run_config.json`) — job parameters
can't reach the `ai_runtime_task` container — so two users launching at the *same moment* can race on
that file. For a solo run it's fine. For a class: either launch runs sequentially, or set
`USE_PRECOMPUTED_VCF = True` and point `PRECOMPUTED_OUT_DIR` at an instructor-produced VCF, so each
attendee does the (parallel-safe, pure-Python) annotation on the same VCF.

## Endpoint/job reference
- **`run_parabricks`** (serverless GPU `ai_runtime_task`, stock NGC `clara-parabricks`): `pbrun fq2bam`
  (paired FASTQ + ref → sorted BAM, +BQSR) then `pbrun deepvariant` (BAM → VCF). Inputs via
  `{PB_VOL}/run_config.json`; outputs `{PB_OUT_DIR}/{deepvariant.vcf, fq2bam.bam, recal.txt}`.
- BQSR is always on → a contig-matched **known-sites** VCF is required (build its `##contig` header
  from the reference `.fai`; populate from a public known-indels set filtered to the target region).
- DeepVariant does not require BQSR scientifically; it's part of this job's fq2bam step.

## Deviations / honest scope to state in the generated notebook
- Germline **small variants (SNV/indel)** only — no CNV/SV/somatic/phasing.
- Region-subset reads call variants only where there is coverage (fast teaching runs; not a genome report).
- ClinVar classifications are variant-level assertions; **zygosity + inheritance** decide clinical meaning
  (e.g. recessive, low-penetrance alleles). Not a diagnosis; wet-lab/clinical confirmation still required.
