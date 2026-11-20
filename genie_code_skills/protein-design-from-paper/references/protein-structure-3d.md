# Rendering protein structures in 3D inside Databricks notebooks

Reference for `show_molstar` (the default structure viewer used by this skill).

## PDBe Mol* in Databricks notebooks (verified pattern — use this)

The **PDBe Mol\* wrapper** (`PDBeMolstarPlugin`) is the only Mol\* variant that reliably renders in
Databricks `displayHTML()` iframes. The core `molstar.Viewer` (jsDelivr `molstar@4.5.0`) loads and
executes all JS steps successfully but the WebGL canvas remains invisible — cause unknown, likely a
CSS/layout interaction in the Databricks iframe sandbox. (rcsb-molstar renders fine in a standalone
browser but was not reliable here either; py3Dmol works but is lower-fidelity.)

**CDN fallback (required).** A single CDN can fail to load on serverless compute
(`PDBeMolstarPlugin is not defined`) — the canonical `ebi.ac.uk` CDN notably so. Always use a
multi-CDN loader that injects the `<script>` and retries on error:

- **jsDelivr** (primary): `https://cdn.jsdelivr.net/npm/pdbe-molstar@3.3.0/build/pdbe-molstar-plugin.js` (+ `…/pdbe-molstar.css`)
- **unpkg** (secondary): `https://unpkg.com/pdbe-molstar@3.3.0/build/pdbe-molstar-plugin.js` (+ `…/pdbe-molstar.css`)
- **ebi.ac.uk** (final fallback): `https://www.ebi.ac.uk/pdbe/pdb-component-library/js/pdbe-molstar-plugin-3.1.3.js` (+ `…/css/pdbe-molstar-3.1.3.css`)

> **Versions.** jsDelivr/unpkg serve `pdbe-molstar@3.3.0`; ebi.ac.uk's versioned-filename hosting
> stops at **3.1.3** (`pdbe-molstar-plugin-3.3.0.js` does **not** exist on ebi.ac.uk — it 404s). The
> `PDBeMolstarPlugin.render(...)` / `visual.select(...)` / `events.loadComplete` API used here is
> compatible across 3.1.3–3.3.0, so the fallback can mix versions safely.

## Loading a structure you generated (not a PDB id)

Designs are PDB **strings**, not deposited PDB ids — pass them to `customData` as a base64 **data URL**:

```js
viewer.render(el, {
  customData: { url: "data:text/plain;base64," + B64, format: "pdb" },  // binary:false implied
  bgColor: { r:26, g:26, b:46 }, hideControls: false, sequencePanel: true, landscape: true
});
```

## Chain colours + epitope/hotspot highlight (the "complex" look)

After the structure loads, `visual.select` colours each chain and draws the interface residues as
ball-and-stick. Subscribe to `events.loadComplete` so the selection applies once the model is ready:

```js
viewer.events.loadComplete.subscribe(function(){
  viewer.visual.select({ data: [
    { auth_asym_id: "A", color: {r:51,g:181,b:229} },   // binder  → teal
    { auth_asym_id: "B", color: {r:230,g:168,b:23} },   // target  → gold
    // one entry per epitope/hotspot residue on the target chain:
    { auth_asym_id: "B", auth_residue_number: 30, color: {r:231,g:76,b:60},
      sideChain: true, representation: "ball-and-stick", representationColor: {r:231,g:76,b:60} }
  ]});
});
```

Pair it with a small HTML legend (teal = binder, gold = target, red = epitope/hotspots).

## The helper

`show_molstar(structure, fmt="pdb", binder_chain="A", target_chain="B", highlight_resids=None,
binder_label="Design", target_label="Target", height=620)` — in `00_toolkit.py` (and emitted inline by
the Genie Code skill). It builds a full standalone HTML document (multi-CDN loader + the data-URL
`customData` render + the `visual.select` chain/epitope colouring + legend) and calls `displayHTML`.
Pass the problem's `HOTSPOTS` as `highlight_resids` to light up the epitope/paratope interface.

## Gotchas
- **Must be the PDBe wrapper**, not core Mol\* (invisible canvas) — don't "simplify" to `molstar.Viewer`.
- **Multi-CDN is not optional** on serverless — a single `<script src>` to one CDN intermittently yields
  `PDBeMolstarPlugin is not defined`.
- `visual.select` must run **after** `loadComplete`, or it no-ops.
- A real design complex has exactly two chains (binder + target); if a reference PDB (e.g. a crystal
  with multiple copies) has extra chains, they keep PDBe's default colours — harmless.
