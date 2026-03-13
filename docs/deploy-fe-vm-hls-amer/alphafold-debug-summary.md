wee updates from May... before signing off:

*AlphaFold v2.3.2 Database Downloads — Debugging Summary*

The `alphafold_register_and_downloads` job on fe-vm-hls-amer failed repeatedly on 3 of 7 download tasks (`download_uniprot`, `download_unirefs`, `pdb_mmcif`). What looked like a single failure turned out to be 5 issues stacked on top of each other, each only visible after fixing the previous one:

| # | Layer | Issue | Fix |
|---|-------|-------|-----|
| 1 | Infra | AWS spot instance preemption killed long-running clusters | `aws_attributes: availability: ON_DEMAND` on all job clusters |
| 2 | Network | FTP and rsync (port 33444) blocked by AWS VPC | `sed` patches to swap FTP URLs to HTTPS equivalents |
| 3 | Notebook format | Heredocs inside `# MAGIC %sh` cells produce malformed scripts | Moved script creation to a Python cell |
| 4 | Script bugs | `wget -r` silently fails on HTTPS directory listings (can't parse HTML index pages); `|| true` + `2>/dev/null` hid all evidence | Replaced recursive wget with explicit URL parsing (`curl` + `grep` to extract file links) fed to `aria2c -j 16` for parallel download, plus file count verification |
| 5 | Regex bug | `sed 's/[^a-z0-9]//g'` meant to clean HTML but kept "href" (itself alphanumeric) — all 1,119 subdirs parsed as `href0a` instead of `0a` | Switched to `cut -d'"' -f2 | tr -d '/'` |

*Files changed:* `download_setup.py` (FTP patches + HTTPS replacement script), `download_pdb_mmcif.py` (calls HTTPS script instead of rsync original)

*Takeaways:*
• Never combine `|| true` with `2>/dev/null` on downloads — it hides all failure evidence
• `wget -r` is unreliable on HTTPS directory listings; `aria2c` with explicit URLs works
• Always verify after bulk downloads (count files, exit on zero)
• When stripping non-alphanumeric chars from HTML, remember field names like "href" are alphanumeric too — use positional extraction instead

*Operational note:* The HTTPS replacement is slower than rsync (per-subdir `curl` to enumerate files before downloading), but provides clear progress visibility — aria2c logs each file with transfer speed, and the outer loop shows `[N/1119] Downloading XX/ (M files)...`. Worth the tradeoff for debuggability.

*Status (2026-03-13):* 6 of 7 tasks SUCCESS, `pdb_mmcif` still running — 250,359 `.cif.gz` structures downloaded and verified, unzipped, flattened, now copying to `/Volumes/hls_amer_catalog/mmt_genesis_workbench/alphafold`. Attempt #5, all fixes holding. Boltz and ESMFold also succeeded on re-run with ON_DEMAND (the earlier ESMFold "driver is lost" was spot preemption, not OOM).

*Latest run:* https://fe-vm-hls-amer.cloud.databricks.com/jobs/151110797461064/runs/516232058461643

*Next todos:* Deploy `bionemo` module (blocked on Docker/NGC creds), need to test the "Raw Single Cell Processing" + verify the app end-to-end + review demo walkthrough...

nite^2 ...zzz ...{dreaming of successful task run}
