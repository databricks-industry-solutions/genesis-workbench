---
name: protein-design-from-paper
description: Build a complete, runnable protein-design notebook for the user's own target by generating cells into the current notebook. Use when a user references a research paper or a one-line goal and wants to design or optimize a protein — a binder against a protein, a single-domain antibody (VHH / nanobody) against an antigen epitope, a binder against a small molecule/ligand, a scaffold around a functional motif, a de novo monomer, or an optimization of an existing design — using Genesis Workbench serving endpoints. Triggers on "use this paper to design/optimize", "design a binder against", "design a nanobody/VHH/antibody against", "scaffold this motif", "de novo protein", "optimize this protein" in a Databricks workspace with Genesis Workbench deployed.
---

# Protein Design From a Paper (Genesis Workbench)

Generate a full in-silico protein-design funnel **into the current notebook** for the user's target,
calling Genesis Workbench **only through its serving endpoints**. This is the Genie Code version of the
Claude Code skill `claude_skills/SKILL_GENESIS_WORKBENCH_PROTEIN_DESIGN_FROM_PAPER.md`; keep the two in sync.

## Hard rules
1. **Endpoints only — never import `genesis_workbench`.** Use `WorkspaceClient.serving_endpoints.query(...)`.
   Emit the toolkit below verbatim as an early cell; it discovers endpoint names at runtime (`get_ep`).
2. **Generate cells into the current notebook**, in order (markdown + code). Don't create a separate file.
3. **Be honest about scope.** Wet-lab readouts (affinity, measured Tm, neutralization, PK) are NOT
   reproducible in silico; the developability endpoints are proxies that prioritize what to order.
4. **Fail loud on missing models.** Keep `get_ep`'s loud error; tell the user to warm endpoints (Genesis
   Workbench reaps idle serving endpoints) before running.

## How to respond (generation procedure)
When the request matches, generate these cells in order:
1. **Markdown** — paper citation, one-line goal, and the chosen **task** (see classification).
2. **Code: `%pip install -q "databricks-sdk>=0.50.0" requests` then `dbutils.library.restartPython()`.**
3. **Code: the Toolkit cell** (emit the Python block in *Toolkit* below, unchanged).
4. **Code: Problem definition** — target source (PDB id / sequence / SMILES) + constraints (hotspots, lengths, chains).
5. **Code: Generation** — the one task block from *Per-task generation* that matches the classification.
6. **Code: Monomer gate → Complex confidence → Developability → Composite reward** (see *Shared funnel*),
   trimming the Boltz step for monomer-only tasks and trimming reward axes to those computed.
7. **Code: Final visualization** — render the top design in the **PDBe Mol\* viewer by default**
   (`show_molstar`, see *Shared funnel*): binder chain teal, target chain gold, epitope/hotspot residues
   as red ball-and-stick, with a legend + sequence panel. The plugin loads via a multi-CDN fallback
   (jsDelivr → unpkg → ebi.ac.uk) — PDBe Mol\* is the only Mol\* build that renders in a Databricks
   `displayHTML` iframe, and a single CDN can fail on serverless. PDBe Mol\* is the default; only fall
   back to another viewer if asked. (See `references/protein-structure-3d.md`.)
8. **Markdown: Reflection & honest scope**, and (optional) fetch a deposited design PDB to compare
   (view it the same way with `show_molstar`).

## Task classification
Pick exactly one from the user's intent:

| Intent | Task | Generation call |
|---|---|---|
| Bind a **protein** (block/neutralize/antibody-alternative) | `binder_vs_protein` | `gen_protein_binders(target_pdb, target_chain, hotspots, …)` |
| Design a **single-domain antibody (VHH / nanobody)** against an antigen epitope | `antibody_vhh` | `gen_vhh_antibodies(target_pdb, target_chain, hotspots, length_min=110, length_max=130, …)` |
| Bind a **small molecule / ligand** | `binder_vs_ligand` | `gen_ligand_binders(ligand_pdb, …)` |
| Preserve/graft a **functional motif** | `motif_scaffold` | `scaffold_motif(motif_pdb, motif_chain, …)` |
| Generate/diversify a **de novo monomer** | `denovo_monomer` | `rfdiffusion_inpaint(pdb,i,j)` → `proteinmpnn(bb)` |
| **Improve** an existing design | `optimize` | re-sample around a seed (endpoints) or dispatch `run_enzyme_optimization_gwb` via the Jobs API |

Target source → model input: PDB id → `fetch_pdb(id)` and `chain_sequence(pdb, chain)`; sequence only →
`esmfold(seq)` for a target PDB; ligand → `extract_ligand_pdb(fetch_pdb(id), HET_CODE)` + keep the SMILES.

## Toolkit (emit this as a code cell, unchanged)
```python
import numpy as np, pandas as pd, requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from databricks.sdk.service.serving import DataframeSplitInput

w = WorkspaceClient(config=Config(http_timeout_seconds=900))

_EP_CACHE, _ALL_ENDPOINTS = {}, None
def get_ep(slug, override="", exclude=()):
    """Resolve a GWB serving-endpoint name by model slug; fail loud if missing/asleep."""
    if slug in _EP_CACHE:
        return _EP_CACHE[slug]
    if override.strip():
        _EP_CACHE[slug] = override.strip(); return _EP_CACHE[slug]
    global _ALL_ENDPOINTS
    if _ALL_ENDPOINTS is None:
        _ALL_ENDPOINTS = [e.name for e in w.serving_endpoints.list()]
    cands = [n for n in _ALL_ENDPOINTS if slug in n and not any(x in n for x in exclude)]
    if not cands:
        raise RuntimeError(f"No serving endpoint matches '{slug}'. Available: {sorted(_ALL_ENDPOINTS)}. "
                           f"Is the model deployed and awake?")
    def stem(n): return n[:-len("_endpoint")] if n.endswith("_endpoint") else n
    exact = [n for n in cands if stem(n).endswith(slug)]
    _EP_CACHE[slug] = (exact or cands)[0]
    return _EP_CACHE[slug]

def _unwrap(preds):
    if isinstance(preds, dict) and "predictions" in preds:
        preds = preds["predictions"]
    return [preds] if isinstance(preds, dict) else list(preds or [])

def to_float(x, default=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default

_PC_COLS = ["target_pdb", "binder_length_min", "binder_length_max",
            "num_samples", "hotspot_residues", "target_chain"]

def _proteina(slug, slot_pdb, length_min, length_max, num_samples, hotspots, chain, exclude=()):
    resp = w.serving_endpoints.query(name=get_ep(slug, exclude=exclude),
        dataframe_split=DataframeSplitInput(columns=_PC_COLS,
            data=[[slot_pdb, int(length_min), int(length_max), int(num_samples), str(hotspots), str(chain)]]))
    return [r for r in _unwrap(resp.predictions) if isinstance(r, dict)]

def gen_protein_binders(target_pdb, target_chain="A", hotspots="", length_min=50, length_max=80, num_samples=4):
    return _proteina("proteina_complexa", target_pdb, length_min, length_max, num_samples,
                     hotspots, target_chain, exclude=("ligand", "ame"))

def gen_ligand_binders(ligand_pdb, length_min=50, length_max=80, num_samples=4):
    return _proteina("proteina_complexa_ligand", ligand_pdb, length_min, length_max, num_samples, "", "A")

def scaffold_motif(motif_pdb, motif_chain="B", length_min=50, length_max=80, num_samples=4):
    return _proteina("proteina_complexa_ame", motif_pdb, length_min, length_max, num_samples, "", motif_chain)

# RFD4-Proteina endpoint: proteina_complexa-compatible + a `task` column. A non-"de_novo" task with a
# target PDB + hotspots generates an epitope-conditioned binder (contig "<len>, /0, A" + C_CRD/C_SEQ/C_HOT).
_RFD4_COLS = ["task", "target_pdb", "binder_length_min", "binder_length_max", "num_samples",
              "hotspot_residues", "target_chain", "condition_spec_json", "structure_cif", "exclude_chain"]

def gen_vhh_antibodies(target_pdb, target_chain="A", hotspots="", length_min=110, length_max=130, num_samples=4):
    """Epitope-conditioned single-domain antibody (VHH / nanobody) design via RFD4-Proteina.
    VHH default length window 110-130. Returns dicts normalized to the common
    {sample_id, sequence, rewards, pdb_output} shape so the shared funnel is unchanged
    (RFD4 returns designed_sequence/designed_pdb; complex_pdb is kept for inspection).
    exclude_chain = the antigen chain so `designed_sequence` is the VHH ALONE (not VHH+antigen)
    and `complex_pdb` is the full co-complex."""
    resp = w.serving_endpoints.query(name=get_ep("rfd4_proteina"),
        dataframe_split=DataframeSplitInput(columns=_RFD4_COLS,
            data=[["binder", str(target_pdb), int(length_min), int(length_max), int(num_samples),
                   str(hotspots), str(target_chain), "", "", str(target_chain)]]))
    out = []
    for r in _unwrap(resp.predictions):
        if not isinstance(r, dict):
            continue
        out.append({"sample_id": r.get("sample_id"), "sequence": str(r.get("designed_sequence", "")),
                    "rewards": 0.0, "pdb_output": r.get("designed_pdb", ""),
                    "complex_pdb": r.get("complex_pdb", "")})
    return out

def rfdiffusion_inpaint(pdb, start_idx, end_idx):
    resp = w.serving_endpoints.query(name=get_ep("rfdiffusion"),
        dataframe_records=[{"pdb": pdb, "start_idx": int(start_idx), "end_idx": int(end_idx)}])
    return [str(s) for s in _unwrap(resp.predictions)]

def proteinmpnn(pdb, fixed_positions=""):
    resp = w.serving_endpoints.query(name=get_ep("proteinmpnn"),
        dataframe_records=[{"pdb": pdb, "fixed_positions": fixed_positions}])
    return [str(s) for s in _unwrap(resp.predictions)]

def esmfold(sequence):
    return w.serving_endpoints.query(name=get_ep("esmfold"), inputs=[sequence]).predictions[0]

def boltz_complex(spec, msa="no_msa", use_msa_server="True"):
    resp = w.serving_endpoints.query(name=get_ep("boltz"),
        inputs=[{"input": spec, "msa": msa, "use_msa_server": use_msa_server}])
    out = _unwrap(resp.predictions)
    return out[0] if out else {}

def diffdock(protein_pdb, ligand_smiles, samples=5):
    emb = w.serving_endpoints.query(name=get_ep("diffdock_esm_embeddings"),
        dataframe_split=DataframeSplitInput(columns=["protein_pdb"], data=[[protein_pdb]]))
    b64 = (_unwrap(emb.predictions)[0] or {}).get("embeddings_b64", "{}")
    poses = w.serving_endpoints.query(name=get_ep("diffdock", exclude=("esm_embeddings",)),
        dataframe_split=DataframeSplitInput(
            columns=["protein_pdb", "ligand_smiles", "samples_per_complex", "esm_embeddings_b64"],
            data=[[protein_pdb, ligand_smiles, int(samples), b64]]))
    best = None
    for p in _unwrap(poses.predictions):
        sdf = str(p.get("ligand_sdf", ""))
        if sdf.startswith("ERROR"):
            continue
        c = to_float(p.get("confidence"), -1e9)
        if best is None or c > best[0]:
            best = (c, sdf)
    return (best[1], best[0]) if best else (None, None)

# (builder, endpoint_output_col, out_alias). mhcflurry (MHC-I/CD8) and hlaiipred (MHC-II/CD4) BOTH return
# "predicted_immuno_burden", so hlaiipred is aliased to "predicted_immuno_mhc2" to avoid a silent collision.
_DEV_SPEC = {
    "netsolp":   (lambda s: {"sequence": s}, "predicted_solubility", "predicted_solubility"),
    "deepstabp": (lambda s: {"sequence": s, "growth_temp": 37.0, "mt_mode": "Cell"}, "predicted_tm_celsius", "predicted_tm_celsius"),
    "pltnum":    (lambda s: {"sequence": s}, "predicted_stability", "predicted_stability"),
    "mhcflurry": (lambda s: {"sequence": s,
                  "alleles": "HLA-A*02:01,HLA-A*01:01,HLA-B*07:02,HLA-B*44:02,HLA-C*07:01,HLA-C*04:01"},
                  "predicted_immuno_burden", "predicted_immuno_burden"),   # MHC-I (CD8)
    "hlaiipred": (lambda s: {"sequence": s,
                  "alleles": "DRB1*01:01,DRB1*03:01,DRB1*04:01,DRB1*07:01,DRB1*08:01,DRB1*11:01,DRB1*13:01,DRB1*15:01"},
                  "predicted_immuno_burden", "predicted_immuno_mhc2"),     # MHC-II (CD4 / anti-drug-antibody)
}

def developability(sequence, which=("netsolp", "deepstabp", "pltnum", "mhcflurry")):
    # For antibodies/VHH add "hlaiipred" — MHC-II is the anti-drug-antibody (ADA) signal that matters for a biologic.
    out = {}
    for slug in which:
        build, col, alias = _DEV_SPEC[slug]
        resp = w.serving_endpoints.query(name=get_ep(slug), dataframe_records=[build(sequence)])
        rows = _unwrap(resp.predictions)
        out[alias] = to_float(rows[0].get(col)) if rows else float("nan")
    return out

_AA3TO1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G","HIS":"H",
           "ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W",
           "TYR":"Y","VAL":"V","MSE":"M","SEC":"U","PYL":"O"}

def fetch_pdb(pdb_id):
    r = requests.get(f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb", timeout=30)
    r.raise_for_status()
    return r.text

def chain_sequence(pdb_text, chain="A"):
    seq, seen = [], set()
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA" and line[21] == chain:
            key = line[22:27]
            if key in seen:
                continue
            seen.add(key)
            seq.append(_AA3TO1.get(line[17:20].strip(), "X"))
    return "".join(seq)

def extract_ligand_pdb(pdb_text, resname):
    het = [l for l in pdb_text.splitlines() if l.startswith("HETATM") and l[17:20].strip() == resname]
    serials = {l[6:11].strip() for l in het}
    conect = [l for l in pdb_text.splitlines() if l.startswith("CONECT") and l[6:11].strip() in serials]
    return "\n".join(het + conect) + "\nEND\n"

def mean_plddt(pdb_text):
    vals = [float(l[60:66]) for l in pdb_text.splitlines()
            if l.startswith("ATOM") and l[12:16].strip() == "CA"]
    return float(np.mean(vals)) if vals else float("nan")

def _norm(vals, invert=False):
    arr = np.array([v if v is not None and not np.isnan(v) else np.nan for v in vals], dtype=float)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(lo) or hi == lo:
        base = np.full(arr.shape, 0.5)
    else:
        base = (arr - lo) / (hi - lo)
        base = np.where(np.isnan(base), 0.0, base)
    return (1.0 - base) if invert else base

DEFAULT_WEIGHTS = {"plddt": 1.3, "iptm": 0.5, "solubility": 1.0, "stability": 2.6, "tm": 1.0, "immuno": 1.5, "immuno_mhc2": 1.5}

def composite_reward(df, axis_cols, weights=None):
    weights = weights or DEFAULT_WEIGHTS
    keys = [k for k in axis_cols if k in weights]
    normed = {k: _norm(df[axis_cols[k][0]], invert=not axis_cols[k][1]) for k in keys}
    total = sum(weights[k] for k in keys) or 1.0
    return pd.Series(sum(weights[k] * normed[k] for k in keys) / total, index=df.index)

def show_molstar(structure, fmt="pdb", binder_chain="A", target_chain="B", highlight_resids=None,
                 binder_label="Design", target_label="Target", height=620):
    """DEFAULT 3D viewer for the final design — PDBe Mol* (PDBeMolstarPlugin), inline via displayHTML.
    PDBe Mol* is the only Mol* build that reliably renders inside a Databricks displayHTML iframe (the
    core molstar.Viewer loads but its WebGL canvas stays invisible there). The plugin is loaded via a
    MULTI-CDN fallback (jsDelivr -> unpkg -> ebi.ac.uk) because a single CDN — notably ebi.ac.uk — can
    fail on serverless compute ("PDBeMolstarPlugin is not defined"). Colours the binder chain teal and
    the target chain gold, and draws `highlight_resids` on the target chain as red ball-and-stick (the
    epitope / hotspot interface), with a legend + sequence panel. `structure` is a PDB/mmCIF string,
    passed as a base64 data: URL. (Rendering pattern verified in a Databricks notebook.)"""
    import base64, json
    if not structure:
        print("show_molstar: no structure to display"); return
    b64 = base64.b64encode(structure.encode()).decode()
    sel = [{"auth_asym_id": binder_chain, "color": {"r": 51, "g": 181, "b": 229}},
           {"auth_asym_id": target_chain, "color": {"r": 230, "g": 168, "b": 23}}]
    for r in (highlight_resids or []):
        sel.append({"auth_asym_id": target_chain, "auth_residue_number": int(r),
                    "color": {"r": 231, "g": 76, "b": 60}, "sideChain": True,
                    "representation": "ball-and-stick", "representationColor": {"r": 231, "g": 76, "b": 60}})
    hl = ", ".join(str(int(r)) for r in (highlight_resids or []))
    epi_legend = ('<span><span class="sw" style="background:#e74c3c;"></span>Epitope/hotspots (' + hl
                  + ')</span>') if highlight_resids else ""
    html = ("""
<!DOCTYPE html><html><head><style>
  body{margin:0;padding:8px;background:#1a1a2e;font-family:'Segoe UI',Arial,sans-serif;}
  #molviewer{position:relative;width:100%;height:__H__px;}
  .legend{display:flex;flex-wrap:wrap;gap:16px;padding:8px 2px;color:#aaa;font-size:13px;line-height:1.6;}
  .sw{display:inline-block;width:13px;height:13px;border-radius:3px;margin-right:5px;vertical-align:middle;}
  .hint{color:#666;font-style:italic;}
</style></head><body>
  <div id="molviewer"></div>
  <div class="legend">
    <span><span class="sw" style="background:#33b5e5;"></span>__BINDER__ (chain __BC__)</span>
    <span><span class="sw" style="background:#e6a817;"></span>__TARGET__ (chain __TC__)</span>
    __EPI_LEGEND__
    <span class="hint">drag = rotate &middot; scroll = zoom &middot; click = residue info</span>
  </div>
  <script>
  (function(){
    var CDNS=[
      {js:"https://cdn.jsdelivr.net/npm/pdbe-molstar@3.3.0/build/pdbe-molstar-plugin.js",
       css:"https://cdn.jsdelivr.net/npm/pdbe-molstar@3.3.0/build/pdbe-molstar.css"},
      {js:"https://unpkg.com/pdbe-molstar@3.3.0/build/pdbe-molstar-plugin.js",
       css:"https://unpkg.com/pdbe-molstar@3.3.0/build/pdbe-molstar.css"},
      {js:"https://www.ebi.ac.uk/pdbe/pdb-component-library/js/pdbe-molstar-plugin-3.1.3.js",
       css:"https://www.ebi.ac.uk/pdbe/pdb-component-library/css/pdbe-molstar-3.1.3.css"}];
    var el=document.getElementById("molviewer");
    function render(){
      try{
        var v=new PDBeMolstarPlugin();
        v.render(el,{customData:{url:"data:text/plain;base64,__B64__",format:"__FMT__"},
                     bgColor:{r:26,g:26,b:46},hideControls:false,sequencePanel:true,landscape:true});
        v.events.loadComplete.subscribe(function(){ try{ v.visual.select({data:__SEL__}); }catch(e){} });
      }catch(e){ el.innerText="PDBe Mol* render failed: "+(e&&e.message); }
    }
    function tryCdn(i){
      if(typeof PDBeMolstarPlugin!=="undefined"){render();return;}
      if(i>=CDNS.length){el.innerText="PDBe Mol* failed: all CDNs unreachable";return;}
      var link=document.createElement("link");link.rel="stylesheet";link.href=CDNS[i].css;document.head.appendChild(link);
      var s=document.createElement("script");s.src=CDNS[i].js;
      s.onload=function(){ (typeof PDBeMolstarPlugin!=="undefined")?render():tryCdn(i+1); };
      s.onerror=function(){tryCdn(i+1);};
      document.head.appendChild(s);
    }
    tryCdn(0);
  })();
  </script>
</body></html>
""".replace("__H__", str(int(height))).replace("__B64__", b64).replace("__FMT__", fmt)
   .replace("__BC__", binder_chain).replace("__TC__", target_chain)
   .replace("__BINDER__", binder_label).replace("__TARGET__", target_label)
   .replace("__EPI_LEGEND__", epi_legend).replace("__SEL__", json.dumps(sel)))
    displayHTML(html)

print("✅ GWB protein-design toolkit loaded (endpoints resolve lazily via get_ep; Mol* = default viewer).")
```

## Per-task generation (emit the one matching the classification)
```python
# binder_vs_protein — target PDB in target_pdb_text, sequence in target_seq
raw = gen_protein_binders(target_pdb=target_pdb_text, target_chain=TARGET_CHAIN,
                          hotspots=HOTSPOTS, length_min=LENGTH_MIN, length_max=LENGTH_MAX, num_samples=NUM_DESIGNS)

# antibody_vhh — epitope-conditioned single-domain antibody (VHH/nanobody) via RFD4-Proteina.
# Same target+epitope inputs as binder_vs_protein, but a VHH length window (110-130) and the RFD4 endpoint.
# Score the MHC-II / anti-drug-antibody axis in the funnel (add "hlaiipred" to developability's `which`).
# raw = gen_vhh_antibodies(target_pdb=target_pdb_text, target_chain=TARGET_CHAIN,
#                          hotspots=HOTSPOTS, length_min=110, length_max=130, num_samples=NUM_DESIGNS)

# binder_vs_ligand
# ligand_pdb = extract_ligand_pdb(fetch_pdb("<PDB_WITH_LIGAND>"), "<HET_CODE>")
# raw = gen_ligand_binders(ligand_pdb, length_min=LENGTH_MIN, length_max=LENGTH_MAX, num_samples=NUM_DESIGNS)

# motif_scaffold
# raw = scaffold_motif(fetch_pdb("<MOTIF_PDB>"), motif_chain="B", length_min=LENGTH_MIN, length_max=LENGTH_MAX, num_samples=NUM_DESIGNS)

# denovo_monomer (no partner; skip the Boltz step)
# bb = rfdiffusion_inpaint(target_pdb_text, start_idx=<i>, end_idx=<j>)[0]
# raw = [{"sample_id": k, "sequence": s, "rewards": 0.0, "pdb_output": bb} for k, s in enumerate(proteinmpnn(bb))]

designs = [{"sample_id": str(r.get("sample_id", i)), "sequence": str(r.get("sequence", "")),
            "rewards": to_float(r.get("rewards", 0.0), 0.0), "backbone_pdb": r.get("pdb_output", "")}
           for i, r in enumerate(raw) if str(r.get("sequence", "")).strip()]
```

## Shared funnel (emit after generation)
```python
# 3) monomer gate
PLDDT_CUTOFF = 70.0
for d in designs:
    d["mean_plddt"] = mean_plddt(esmfold(d["sequence"]))
folded = sorted([d for d in designs if d["mean_plddt"] >= PLDDT_CUTOFF], key=lambda d: d["mean_plddt"], reverse=True)

# 4) complex confidence (skip for denovo_monomer). protein target shown; ligand: f"...;ligand_B:{LIGAND_SMILES}"
TOP_N, IPTM_CUTOFF = 4, 0.50
shortlist = folded[:TOP_N]
for d in shortlist:
    res = boltz_complex(f"protein_A:{d['sequence']};protein_B:{target_seq}")
    d["iptm"] = to_float(res.get("iptm")); d["ptm"] = to_float(res.get("ptm")); d["complex_pdb"] = res.get("pdb", "")
ranked = sorted(shortlist, key=lambda d: (d["iptm"] if not np.isnan(d["iptm"]) else -1), reverse=True)
passers = [d for d in ranked if d["iptm"] >= IPTM_CUTOFF] or ranked[:TOP_N]

# 5) developability  (antibody_vhh: pass which=(..., "hlaiipred") to add the MHC-II / ADA axis)
DEV_WHICH = ("netsolp", "deepstabp", "pltnum", "mhcflurry")  # + ("hlaiipred",) for antibody_vhh
for d in passers:
    d.update(developability(d["sequence"], which=DEV_WHICH))

# 6) composite reward + selection
df = pd.DataFrame(passers)
axis_cols = {"plddt": ("mean_plddt", True), "iptm": ("iptm", True),
             "solubility": ("predicted_solubility", True), "stability": ("predicted_stability", True),
             "tm": ("predicted_tm_celsius", True), "immuno": ("predicted_immuno_burden", False),
             "immuno_mhc2": ("predicted_immuno_mhc2", False)}  # MHC-II/ADA — present only when hlaiipred ran
axis_cols = {k: v for k, v in axis_cols.items() if v[0] in df.columns}
df["composite_reward"] = composite_reward(df, axis_cols)
df = df.sort_values("composite_reward", ascending=False).reset_index(drop=True)
display(df[["sample_id", "composite_reward"] + [v[0] for v in axis_cols.values()] + ["sequence"]])

# 7) visualize the top design — PDBe Mol* viewer by DEFAULT: binder chain teal, target chain gold, and
#    the epitope/hotspot residues on the target as red ball-and-stick + a legend. (denovo_monomer: no
#    complex → pass the backbone, no highlight.) PDBe Mol* is the default — don't swap it out.
top = df.iloc[0]
_hot = [int(x) for x in str(HOTSPOTS).replace(" ", "").split(",") if x.strip().isdigit()]
show_molstar(top.get("complex_pdb") or top.get("backbone_pdb") or "",
             binder_chain="A", target_chain="B", highlight_resids=_hot,
             binder_label="Design", target_label="Target")
```

## Endpoint reference (slug → payload → output)
- `proteina_complexa` (excl. `ligand`,`ame`) / `proteina_complexa_ligand` / `proteina_complexa_ame`:
  `dataframe_split` cols `target_pdb,binder_length_min,binder_length_max,num_samples,hotspot_residues,target_chain` → `sample_id,pdb_output,sequence,rewards`.
- `rfd4_proteina` (antibody/VHH + general binder): `dataframe_split` cols `task,target_pdb,binder_length_min,binder_length_max,num_samples,hotspot_residues,target_chain,condition_spec_json,structure_cif,exclude_chain` → `sample_id,designed_pdb,designed_sequence,complex_pdb`. A non-`de_novo` `task` + a `target_pdb` + `hotspot_residues` gives an epitope-conditioned binder (contig `<len>, /0, A` + C_CRD/C_SEQ/C_HOT).
- `rfdiffusion`: `dataframe_records [{pdb,start_idx,end_idx}]` → backbone PDB(s). `proteinmpnn`: `[{pdb,fixed_positions}]` → sequences.
- `esmfold`: `inputs=[seq]` → PDB (pLDDT in B-factors). `boltz`: `inputs=[{input,msa,use_msa_server}]` → `pdb,iptm,protein_iptm,ptm,confidence_score,complex_plddt`.
- `netsolp`/`deepstabp`/`pltnum`/`mhcflurry`/`hlaiipred`: `dataframe_records [{sequence,…}]` → `predicted_solubility`/`predicted_tm_celsius`/`predicted_stability`/`predicted_immuno_burden`. mhcflurry = MHC-I (CD8); hlaiipred = MHC-II (CD4/ADA, pass `alleles` DRB1 panel) — both emit `predicted_immuno_burden`, so alias hlaiipred's to `predicted_immuno_mhc2`.

## Deviations to state in the generated notebook
- Proteina-Complexa Binder = the RFdiffusion+ProteinMPNN analog (GWB has no target-conditioned RFdiffusion).
- GWB AlphaFold2 is monomer-only → **Boltz-2 ipTM** is the complex/interface filter.
- No Rosetta ddG → rank on ipTM + pLDDT. Developability are proxies; PLTNUM is a 0–1 ranker, not hours.
- Antibody (`antibody_vhh`): RFD4-Proteina designs a VHH-length binder conditioned on the epitope (C_HOT) — a first-draft VHH, not a full immunoglobulin-framework scaffold. MHC-II (HLAIIPred) is the anti-drug-antibody (ADA) signal and the right immunogenicity axis for a biologic; MHC-I (MHCflurry) is the CD8 axis.
