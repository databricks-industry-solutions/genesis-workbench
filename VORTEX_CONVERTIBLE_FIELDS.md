# Vortex — Convertible Fields (input ⇆ variable) design

## Problem
Vortex hard-splits a node's data at registration time:
- `inputs` → typed **ports**, only fed by an edge (cannot be typed inline).
- `params` → scalar **fields**, only edited inline (cannot be wired).

Consequences: (1) multi-input nodes spawn indistinguishable "Text Input" IO nodes;
(2) whether a value (e.g. a SMILES) is external vs inline is decided globally at
registration, not per-instance by the user.

## Model: every data field is inline-editable AND connectable
No inputs-vs-params distinction at the dataflow level. For each field:
**resolved value = upstream value if its handle is wired, else the inline value,
else the default.** A wired connection visually disables the inline editor.

Result: the enzyme node shows "Motif PDB" + "Substrate SMILES" fields directly
(typed inline OR wired) — no separate Text Input nodes.

## Chosen representation (low-churn, backward compatible)
Keep the registry's `inputs` (ports) and `params` (fields) lists as-is — do NOT
collapse them (would churn every node def + the MCP seed + capabilities). Instead:
- An **input port** gains an inline editor (derived from its dtype) + optional
  `default`. It keeps its handle.
- The graph node stores inline input values in a new `inputs` value map on node
  data (separate from `params`), keyed by port name.
- **Resolution precedence (orchestrator):** for each input port,
  `edge value  >  node.data.inputs[port]  >  port default`.
- `params` keep working exactly as today.

Backward compat: old saved graphs (wired Text Inputs → input ports) keep working —
edges still feed inputs. New graphs use inline values. Both coexist. MCP is
unaffected (it already takes inputs+params as tool kwargs).

### dtype → inline editor
- sequence / smiles / string / path / table → single-line text
- pdb / json / sequences → textarea
- score / embedding → text (rarely typed; usually wired)
- (params already carry their own editor type: int/float/bool/select/text/string)

## Status
**Increments 1–3 DONE** (app-only deploy, backward compatible): data model +
orchestrator resolution (edge > inline), inline editors in NodeParamPanel, auto-IO
retired, validation = wired-or-inline. Increment 4 (show inline values in the node
card) is optional polish, pending.

## Increments
1. **Backend data model + orchestrator** (no UX change yet):
   - FE `toCanvasGraph`/`fromCanvasGraph`: serialize `node.data.inputs` (inline values).
   - BE `CanvasNode` model + `_enrich_graph` + `run_node`: resolve each input as
     `edge ?? inline ?? default`. (`run_node` today merges `{**params, **edge_inputs}`;
     add the inline fallback.)
   - Wheel bump.
2. **UI inline editors**:
   - `NodeParamPanel`: render an **Inputs** section (one dtype-editor per input
     port), disabled + "← from <upstream node>" when the handle is wired; then the
     existing **Parameters** section.
   - `VortexTab`: store inline input values (`onChangeInput`), pass the selected
     node's wired-input set to the panel.
   - **Remove auto-IO Text Input spawning** (inputs are inline now). Dropping a
     prebuilt node just drops the node. (Output sink auto-add also retired; results
     are captured to MLflow regardless.)
3. **Validation**: an input is satisfied if **wired OR has a non-empty inline/default
   value** (updates `graphValidationErrors` → the checklist + red/green borders + Run gate).
4. **CanvasNode (optional polish)**: show set inline input values in the card body
   (like params) so the node reads at a glance.

## Out of scope / unaffected
- MCP tools (already kwargs-based), the prebuilt_workflows seed, capabilities registry.
- Endpoint/chain/job semantics — only how the canvas supplies their inputs changes.
