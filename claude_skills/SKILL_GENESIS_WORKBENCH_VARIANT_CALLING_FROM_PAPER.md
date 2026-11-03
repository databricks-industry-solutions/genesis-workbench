---
name: genesis-workbench-variant-calling-from-paper
description: Scaffold a complete genomics notebook from a research paper or one-line goal — GPU-accelerated germline variant calling (NVIDIA Parabricks fq2bam + DeepVariant) via the deployed GWB genomics job, then clinical variant annotation against ClinVar, entirely Glow-free (pure Python/pandas). Use when a user says "use this paper to call variants / find the pathogenic variant / annotate my VCF against ClinVar / is my sample a carrier of X" and wants a ready-to-run notebook for their own sequencing sample.
---

# Variant Calling From a Paper — notebook scaffolder (Genesis Workbench genomics)

Produce a **ready-to-run Databricks notebook** that calls germline variants from a user's sequencing
reads on GPU (NVIDIA Parabricks `fq2bam` + DeepVariant, via the deployed `run_parabricks` job) and
annotates the VCF against ClinVar — **Glow-free, pure Python/pandas**. This is the Claude Code twin of
the Genie Code skill `genie_code_skills/variant-calling-from-paper/SKILL.md`; the verbatim notebook
**Toolkit** + cell recipe live there — keep the two in sync.

Anchor paper: DeepVariant — Poplin, Chang, Alexander et al., *Nature Biotechnology* 2018
(DOI 10.1038/nbt.4235), the deep-learning caller `pbrun deepvariant` runs on GPU.

## Hard rules (do not violate)
1. **Call variants with the deployed GPU job, never `pbrun` locally.** Resolve `run_parabricks` by name,
   write its input **config JSON to the Volume**, then `w.jobs.run_now(...)`. An `ai_runtime_task` does
   NOT receive job parameters as env vars — the config file is the input channel.
2. **Annotate in pure Python/pandas — no Glow, no Spark.** Parse the VCF as text → DataFrame; inner-join
   a ClinVar table (staged Delta table or region CSV). One sample + a gene panel needs no cluster.
3. **Honest scope.** Germline SNV/indel only (no CNV/SV/somatic/phasing); region-subset reads call only
   where covered; a ClinVar "Pathogenic" is a variant assertion, not a diagnosis — **zygosity +
   inheritance** decide meaning (recessive / low-penetrance alleles).
4. **Fail loud** if the job, reference, reads, or ClinVar table are missing (genomics module must be
   deployed; GWB reaps idle compute).

## Workflow (the notebook this scaffolds)
`reads (FASTQ) → Parabricks fq2bam+DeepVariant (GPU job) → deepvariant.vcf → parse (pandas) →
join ClinVar → Pathogenic hits with zygosity → clinical interpretation.`

Emit cells in this order (full code in the Genie Code skill's *Toolkit* / *Launch* / *Annotation*):
`%pip install databricks-sdk pandas` → **Toolkit** → **Problem definition** (reads, reference+BWA
index, known-sites, output dir, ClinVar table, gene/variant of interest) → **Launch variant calling**
(write config, `run_now`, poll, locate `deepvariant.vcf`; with a `USE_PRECOMPUTED_VCF` switch) →
**Annotate + interpret** (VCF→DataFrame, join ClinVar, filter Pathogenic, report zygosity) →
**Reflection**.

## The deployed job (verified green)
- **`run_parabricks`** — serverless GPU `ai_runtime_task`, stock NGC `clara-parabricks`:
  `pbrun fq2bam` (paired FASTQ + ref → BAM, +BQSR) then `pbrun deepvariant` (BAM → VCF).
- **Inputs** via `/Volumes/<catalog>/genesis_workbench/parabricks/run_config.json` with keys
  `PB_REF, PB_FQ1, PB_FQ2, PB_KNOWN_SITES, PB_MLFLOW_RUN_ID, PB_OUT_DIR`. **Outputs** →
  `{PB_OUT_DIR}/{deepvariant.vcf, fq2bam.bam, recal.txt}`.
- **BQSR is always on** → a **contig-matched known-sites VCF is required** or `fq2bam` errors. Build its
  `##contig` header from the reference `.fai`; populate from a public known-indels set filtered to the
  target region (so no contig-mismatch and small).
- **Reference**: reuse the one the genomics module staged —
  `/Volumes/<catalog>/genesis_workbench/gwas_reference/genomes/GRCh38_full_analysis_set_plus_decoy_hla.fa`
  (+ `.fai` + BWA index `.amb/.ann/.bwt/.pac/.sa`).

## Data staging (when building an exercise/demo)
- **Reads**: subset a real sample's reads to the target gene locus for a fast teaching run
  (`samtools`/pysam region extract from a CRAM → paired FASTQ; mates via `-M -L`). Region-subset reads
  call variants only there → minutes on an A10.
- **ClinVar table**: fetch the ClinVar GRCh38 VCF region for the target gene (NCBI contig naming is
  `6`, not `chr6`) and stage as a Delta table / CSV with `chrom,pos,ref,alt,rs,gene,clnsig,clndn,clnrevstat`.
- Grant workshop users `CAN_MANAGE_RUN` on `run_parabricks` + READ on the input/output Volumes; for
  per-user GPU launches they also need WRITE where `run_config.json` lives.

## Concurrency caveat
`run_parabricks` reads a **single fixed** `run_config.json` (job params can't reach the container), so
simultaneous per-user launches race. Solo runs are fine; for a class, launch sequentially or set
`USE_PRECOMPUTED_VCF=True` and point at an instructor-produced VCF (annotation is parallel-safe).
A fully isolated per-user launch would need the `gwas` module's run-id-keyed config pattern
(`write_pb_config.py` keyed by `{{job.run_id}}` via `SGC_DEBUG_INFO`) ported into `run_parabricks`.

## References
- Genie Code twin (verbatim toolkit + cells): `genie_code_skills/variant-calling-from-paper/SKILL.md`
- Deployed job code: `modules/genomics/parabricks/parabricks_v1/airt/run_parabricks.py`
- Reference staging: `modules/genomics/gwas/gwas_v1/notebooks/01_download_reference_genome.py`
