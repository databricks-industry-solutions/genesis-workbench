# Databricks notebook source
# MAGIC %md
# MAGIC # Benchmark aggregation — speed + memory plots (3-way)
# MAGIC Reads the three per-model result files from the shared Volume and renders the full plots
# MAGIC (time & memory vs length) plus the 2-panel / 3-bar summary. Robust to a model whose task failed
# MAGIC (it renders whatever is present). Runs on CPU (`client '4'`).

# COMMAND ----------
dbutils.widgets.text("out_dir", "/Volumes/main/genesis_workbench/rfd4_proteina/benchmark_results", "Results dir (UC Volume)")
OUT_DIR = dbutils.widgets.get("out_dir")

# COMMAND ----------
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "matplotlib", "pandas"], check=False)

# COMMAND ----------
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (key, pretty label, bar color). rfdiffusion is the gray "predecessor" baseline, rfd4 the green hero.
MODELS = [
    ("rfdiffusion_inpainting", "RFdiffusion\n(inpaint)", "#9e9e9e"),
    ("rfd4_proteina", "RFD4-Proteina\n(binder)", "#7cb342"),
    ("proteina_complexa", "Proteina-Complexa\n(binder)", "#4b9fd5"),
]

summaries, samples = {}, {}
for key, _, _ in MODELS:
    sp, cp = os.path.join(OUT_DIR, f"{key}_summary.json"), os.path.join(OUT_DIR, f"{key}_samples.csv")
    if os.path.exists(sp):
        summaries[key] = json.load(open(sp))
    if os.path.exists(cp):
        samples[key] = pd.read_csv(cp)
print("loaded summaries for:", list(summaries))
print(json.dumps(summaries, indent=2))


def _mem(summary):
    """Prefer device-level peak (NVML) as the 'footprint'; fall back to torch-allocated."""
    v = summary.get("peak_device_used_gib")
    return v if v not in (None, 0, 0.0) else summary.get("peak_torch_alloc_gib")


# COMMAND ----------
# DBTITLE 1,Summary — 2 panels x 3 bars (median generation time; peak GPU memory)
present = [(k, l, c) for k, l, c in MODELS if k in summaries]
labels = [l for _, l, _ in present]
colors = [c for _, _, c in present]
med = [summaries[k].get("median_seconds", np.nan) for k, _, _ in present]
mem = [_mem(summaries[k]) or np.nan for k, _, _ in present]

fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
for a, vals, title, unit in ((ax[0], med, "Generation time (median)", "seconds/sample"),
                             (ax[1], mem, "Peak GPU memory", "GiB")):
    a.bar(labels, vals, color=colors)
    a.set_title(title, fontsize=13)
    a.set_ylabel(unit)
    for r, v in zip(a.patches, vals):
        if v == v:
            a.annotate(f"{v:.2f}", (r.get_x() + r.get_width() / 2, v), ha="center", va="bottom", fontsize=11)

base = "rfdiffusion_inpainting"
if base in summaries and "rfd4_proteina" in summaries:
    bt, t4 = summaries[base].get("median_seconds"), summaries["rfd4_proteina"].get("median_seconds")
    bm, m4 = _mem(summaries[base]), _mem(summaries["rfd4_proteina"])
    if bt and t4:
        ax[0].text(0.5, 1.08, f"RFD4-Proteina ~{bt / t4:.1f}x faster", transform=ax[0].transAxes,
                   ha="center", color="#558b2f", fontsize=12, weight="bold")
    if bm and m4:
        ax[1].text(0.5, 1.08, f"RFD4-Proteina ~{bm / m4:.1f}x lower memory", transform=ax[1].transAxes,
                   ha="center", color="#558b2f", fontsize=12, weight="bold")

fig.suptitle("Protein design — generation speed + GPU memory (same template set, 1x H100 each)",
             fontsize=14, weight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "summary_speed_memory.png"), dpi=150, bbox_inches="tight")
display(fig)

# COMMAND ----------
# DBTITLE 1,Full plots — generation time & peak memory vs length
fig2, ax2 = plt.subplots(1, 2, figsize=(13, 5.5))
for k, l, c in present:
    df = samples.get(k)
    if df is None or "ok" not in df.columns:
        continue
    ok = df[df["ok"] == True]  # noqa: E712
    if not len(ok):
        continue
    g = ok.groupby("total_len")
    ax2[0].plot(g["seconds"].median().index, g["seconds"].median().values, marker="o", color=c,
                label=l.replace("\n", " "))
    if "peak_torch_alloc_gib" in ok.columns:
        pm = g["peak_torch_alloc_gib"].max()
        ax2[1].plot(pm.index, pm.values, marker="o", color=c, label=l.replace("\n", " "))

ax2[0].set(xlabel="total residues", ylabel="median seconds", title="Generation time vs length")
ax2[1].set(xlabel="total residues", ylabel="peak torch GiB", title="Peak memory vs length")
for a in ax2:
    a.legend()
    a.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, "detail_vs_length.png"), dpi=150, bbox_inches="tight")
display(fig2)

# COMMAND ----------
# DBTITLE 1,Markdown summary table
rows = ["| model | mode | n_ok | median s | peak torch GiB | peak device GiB | mean util % | GPU |",
        "|---|---|---|---|---|---|---|---|"]
for k, _, _ in present:
    s = summaries[k]
    rows.append(f"| {k} | {s.get('mode','')} | {s.get('n_ok','')} | {s.get('median_seconds','')} | "
                f"{s.get('peak_torch_alloc_gib','')} | {s.get('peak_device_used_gib','')} | "
                f"{s.get('mean_gpu_util_pct','')} | {s.get('gpu_name','')} |")
print("\n".join(rows))
print(f"\nplots written to {OUT_DIR}/summary_speed_memory.png and detail_vs_length.png")
