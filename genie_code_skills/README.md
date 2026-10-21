# Genie Code workspace skills

Version-controlled source for **Genie Code** (the in-notebook Databricks Assistant) **workspace skills**.
Each subfolder is one skill containing a `SKILL.md` ([Agent Skills](https://agentskills.io) spec: `name` +
`description` frontmatter, then Markdown instructions). Deployed to the workspace, Genie Code auto-loads a
skill when a request matches its `description`; users can also force it with `@<skill-name>`.

## Skills here

| Skill | Purpose |
|---|---|
| `genesis-workbench/` | **Entry point.** Broad trigger for any life-sciences / bio-ML workflow; explains how GWB is consumed (endpoints / VS / jobs, never the library) and routes to the specific skill below. |
| `gwb-serving-endpoints/` | Call any GWB model via its serving endpoint — discovery, the three payload styles, and an exact per-model payload reference (protein, developability, small-molecule, single-cell, embeddings). |
| `gwb-vector-search/` | Query the GWB Vector Search indices (UniRef + human protein, SCimilarity cells, TEDDY cells); the embed-then-query pattern; building your own index. |
| `gwb-batch-jobs/` | Dispatch long/heavy GWB workflows via the Jobs API (enzyme optimization, Parabricks, GWAS, (re)builds), poll, and read MLflow / Delta / Volume results. |
| `gwb-ray-batch-inference/` | Scale a forward pass across many A10s with Ray on serverless GPU (the teddy / sequence_search VS-reference pattern): stage→embed, UC Volume bridge, CTAS. |
| `gwb-serverless-gpu/` | Get a GPU in a notebook or job (serverless A10 AI Runtime); the HF / torch / disk gotchas. Foundation for `gwb-ray-batch-inference`. |
| `gwb-discover-resources/` | Discover which endpoints / indices / jobs are available (and awake) + the metadata tables and MCP server. |
| `gwb-single-cell/` | Applied scRNA-seq / AnnData: cell-type annotation by reference KNN (TEDDY / SCimilarity), embeddings, scGPT perturbation. |
| `gwb-sequence-search/` | Applied protein similarity: ESM-2 embed → UniRef + human-gene indices. |
| `protein-design-from-paper/` | Generate a full endpoints-only protein-design funnel **into the current notebook** for the user's target (binder / ligand-binder / motif / de novo / optimize). Self-contained toolkit. Mirrors `claude_skills/SKILL_GENESIS_WORKBENCH_PROTEIN_DESIGN_FROM_PAPER.md` — keep in sync. |

## Deploy (workspace-level — requires workspace admin)

Genie Code reads workspace skills from `/.assistant/skills/<name>/SKILL.md` at the workspace root.

**Automatic (preferred):** the core deploy (`modules/core/deploy.sh`) copies every skill folder here to
`/.assistant/skills/<name>/SKILL.md`, and `.assistant_workspace_instructions.md` to the workspace root, on
every `./deploy.sh core <cloud>` run. It's tolerant — a non-admin workspace just logs a warning and skips —
so a fresh install ships these skills with no extra step.

**Manual / out-of-band** (refresh without a redeploy — `workshop` profile shown; loops over every skill):

```bash
for d in genie_code_skills/*/; do
  name=$(basename "$d"); [ -f "${d}SKILL.md" ] || continue
  databricks workspace mkdirs "/.assistant/skills/$name" --profile workshop
  databricks workspace import "/.assistant/skills/$name/SKILL.md" --file "${d}SKILL.md" \
    --format AUTO --overwrite --profile workshop
done

Verify:
```bash
databricks workspace list /.assistant/skills/protein-design-from-paper --profile workshop
databricks workspace export /.assistant/skills/protein-design-from-paper/SKILL.md --format AUTO --profile workshop | head
```

**Current status:** deployed to the workshop workspace (`dbc-3d5f56ea-cc38.cloud.databricks.com`) on
2026-10-01, byte-identical to the source here.

### Workspace-wide instructions (also deployed)

`.assistant_workspace_instructions.md` (source here) is the admin-only global "system prompt" for Genie
Code — it sets the endpoints-only rule for GWB work across all notebooks and points at the skill above.
Deployed to the workspace root:

```bash
databricks workspace import /.assistant_workspace_instructions.md \
  --file genie_code_skills/.assistant_workspace_instructions.md \
  --format AUTO --overwrite --profile workshop
```
Workspace instructions take precedence over user `/Users/<you>/.assistant_instructions.md`.

## Notes
- **Apply changes:** start a **new** Genie Code chat (edits don't affect active chats); hard-refresh the tab
  if skill metadata looks stale.
- **User-level alternative:** drop the folder under `/Users/<you>/.assistant/skills/` to prototype before
  promoting to the workspace root.
- **Scope reminder:** the generated notebook does in-silico enrichment only — it calls GWB serving endpoints
  and never imports the `genesis_workbench` library (see the `gwb-workshops-endpoints-only` memory).
- Related: workspace-wide instructions live in `/.assistant_workspace_instructions.md` (admin-only), which
  take precedence over user `/Users/<you>/.assistant_instructions.md`.
