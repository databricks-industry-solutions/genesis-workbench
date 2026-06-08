# Databricks notebook source
# MAGIC %md
# MAGIC ### Initialize small-molecule lookup tables — `target_binders` + `repurposing_hub`
# MAGIC One-time / refreshable ingest that makes the small-molecule features
# MAGIC **self-contained** (no external API at runtime), mirroring the
# MAGIC `gene_sequences` / Target Resolver pattern. Builds two Delta tables:
# MAGIC
# MAGIC - **`target_binders`** — from ChEMBL: per-target known active compounds
# MAGIC   (SMILES + pChEMBL) with their Murcko scaffolds. Seeds GenMol's
# MAGIC   `fragment_completion` (target-aware generation) and powers similarity-based
# MAGIC   repurposing. License: ChEMBL CC BY-SA 3.0.
# MAGIC - **`repurposing_hub`** — from the Broad Drug Repurposing Hub: approved /
# MAGIC   clinical drugs with SMILES, MoA, target, clinical phase, disease area,
# MAGIC   indication. The screenable library for repurposing. License: CC BY 4.0.
# MAGIC
# MAGIC Only this notebook (not the app) touches the internet, and only at ingest
# MAGIC time on a cluster with outbound access. Each table build is independent and
# MAGIC fault-tolerant — a temporarily-unavailable source won't fail the job.

# COMMAND ----------

dbutils.widgets.text("catalog", "srijit_nair_ci_demo_catalog", "Catalog")
dbutils.widgets.text("schema", "genesis_workbench", "Schema")
dbutils.widgets.text("target_genes", "PARP1,BRCA1,BRCA2,EGFR,KRAS,TP53,PARP2,ATM", "Comma-sep gene symbols")
dbutils.widgets.text("max_actives_per_gene", "300", "Top-N actives per target (by pChEMBL)")
dbutils.widgets.text("chembl_base", "https://www.ebi.ac.uk/chembl/api/data", "ChEMBL API base")
dbutils.widgets.text("repurposing_drugs_url",
                     "https://s3.amazonaws.com/data.clue.io/repurposing/downloads/repurposing_drugs_20200324.txt",
                     "Broad Hub drugs TSV")
dbutils.widgets.text("repurposing_samples_url",
                     "https://s3.amazonaws.com/data.clue.io/repurposing/downloads/repurposing_samples_20200324.txt",
                     "Broad Hub samples TSV (has SMILES)")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
target_genes = [g.strip() for g in dbutils.widgets.get("target_genes").split(",") if g.strip()]
max_actives = int(dbutils.widgets.get("max_actives_per_gene"))
chembl_base = dbutils.widgets.get("chembl_base").rstrip("/")
drugs_url = dbutils.widgets.get("repurposing_drugs_url")
samples_url = dbutils.widgets.get("repurposing_samples_url")

# COMMAND ----------

# MAGIC %pip install rdkit==2025.3.6 requests

# COMMAND ----------

# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import io, time, urllib.request
import requests
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
target_genes = [g.strip() for g in dbutils.widgets.get("target_genes").split(",") if g.strip()]
max_actives = int(dbutils.widgets.get("max_actives_per_gene"))
chembl_base = dbutils.widgets.get("chembl_base").rstrip("/")
drugs_url = dbutils.widgets.get("repurposing_drugs_url")
samples_url = dbutils.widgets.get("repurposing_samples_url")


def murcko(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception:
        return None

# COMMAND ----------

# MAGIC %md
# MAGIC ### Table 1 — `target_binders` (ChEMBL)

# COMMAND ----------

def chembl_target_id(gene):
    """Resolve a gene symbol to a human SINGLE PROTEIN ChEMBL target id."""
    r = requests.get(f"{chembl_base}/target/search.json", params={"q": gene}, timeout=60)
    r.raise_for_status()
    targets = r.json().get("targets", [])
    # Prefer a human single-protein target.
    for t in targets:
        if t.get("target_type") == "SINGLE PROTEIN" and t.get("organism") == "Homo sapiens":
            return t.get("target_chembl_id")
    return targets[0]["target_chembl_id"] if targets else None


def chembl_actives(target_chembl_id, cap, max_pages=3, timeout=30):
    """Page activities with a pChEMBL value; return (smiles, pchembl, molecule_id).
    Hard-bounded: short per-request timeout + a max page count, so a slow/large
    target (e.g. EGFR has tens of thousands of activities) can't run away — we
    over-fetch a few pages, then dedup + top-N below."""
    rows, url, pages = [], f"{chembl_base}/activity.json", 0
    params = {"target_chembl_id": target_chembl_id, "pchembl_value__isnull": "false", "limit": 1000}
    while url and pages < max_pages and len(rows) < cap * 4:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        for a in data.get("activities", []):
            smi, pv = a.get("canonical_smiles"), a.get("pchembl_value")
            if smi and pv:
                rows.append((smi, float(pv), a.get("molecule_chembl_id")))
        nxt = (data.get("page_meta") or {}).get("next")
        url = ("https://www.ebi.ac.uk" + nxt) if nxt else None
        params = None
        pages += 1
    return rows


binder_rows = []
try:
    for gene in target_genes:
        try:
            tid = chembl_target_id(gene)
            if not tid:
                print(f"  {gene}: no ChEMBL target found, skipping")
                continue
            acts = chembl_actives(tid, max_actives)
            # Dedup by molecule, keep best pChEMBL, take top-N.
            best = {}
            for smi, pv, mol_id in acts:
                key = mol_id or smi
                if key not in best or pv > best[key][1]:
                    best[key] = (smi, pv, mol_id)
            top = sorted(best.values(), key=lambda x: x[1], reverse=True)[:max_actives]
            for smi, pv, mol_id in top:
                binder_rows.append((gene, tid, mol_id, smi, murcko(smi), pv))
            print(f"  {gene} ({tid}): {len(top)} binders")
        except Exception as e:
            print(f"  {gene}: ERROR {type(e).__name__}: {e}")
        time.sleep(0.2)

    cols = ["gene", "target_chembl_id", "molecule_chembl_id", "binder_smiles", "murcko_scaffold", "pchembl"]
    df = spark.createDataFrame(binder_rows, cols)
    (df.write.mode("overwrite").option("overwriteSchema", "true")
       .saveAsTable(f"{catalog}.{schema}.target_binders"))
    print(f"Wrote {df.count()} rows to {catalog}.{schema}.target_binders")
except Exception as e:
    print(f"target_binders build FAILED (non-fatal): {type(e).__name__}: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Table 2 — `repurposing_hub` (Broad Drug Repurposing Hub)

# COMMAND ----------

def fetch_broad_tsv(url):
    """Broad Hub TSVs have a '!'-prefixed comment block, then a header row."""
    with urllib.request.urlopen(url, timeout=45) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("!")]
    header = lines[0].split("\t")
    rows = []
    for ln in lines[1:]:
        parts = ln.split("\t")
        parts += [""] * (len(header) - len(parts))
        rows.append(dict(zip(header, parts)))
    return rows


try:
    drugs = fetch_broad_tsv(drugs_url)        # pert_iname, clinical_phase, moa, target, disease_area, indication
    samples = fetch_broad_tsv(samples_url)    # pert_iname, smiles, InChIKey, ...
    print(f"Broad Hub: {len(drugs)} drugs, {len(samples)} samples")

    # First valid SMILES per drug name.
    smiles_by_drug = {}
    for s in samples:
        name = (s.get("pert_iname") or "").strip()
        smi = (s.get("smiles") or "").strip()
        if name and smi and name not in smiles_by_drug and Chem.MolFromSmiles(smi) is not None:
            smiles_by_drug[name] = smi

    rep_rows = []
    for d in drugs:
        name = (d.get("pert_iname") or "").strip()
        smi = smiles_by_drug.get(name)
        if not name or not smi:
            continue
        rep_rows.append((
            name, smi,
            (d.get("moa") or "").strip(),
            (d.get("target") or "").strip(),
            (d.get("clinical_phase") or "").strip(),
            (d.get("disease_area") or "").strip(),
            (d.get("indication") or "").strip(),
        ))

    cols = ["drug_name", "smiles", "moa", "target", "clinical_phase", "disease_area", "indication"]
    df = spark.createDataFrame(rep_rows, cols)
    (df.write.mode("overwrite").option("overwriteSchema", "true")
       .saveAsTable(f"{catalog}.{schema}.repurposing_hub"))
    print(f"Wrote {df.count()} rows to {catalog}.{schema}.repurposing_hub")
except Exception as e:
    print(f"repurposing_hub build FAILED (non-fatal): {type(e).__name__}: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sanity check

# COMMAND ----------

for tbl in ("target_binders", "repurposing_hub"):
    try:
        n = spark.table(f"{catalog}.{schema}.{tbl}").count()
        print(f"{tbl}: {n} rows")
    except Exception as e:
        print(f"{tbl}: not built ({e})")

try:
    print("\nPARP1 binders sample:")
    display(spark.sql(
        f"SELECT gene, molecule_chembl_id, pchembl, murcko_scaffold "
        f"FROM {catalog}.{schema}.target_binders WHERE upper(gene)='PARP1' "
        f"ORDER BY pchembl DESC LIMIT 5"))
except Exception:
    pass
