# Genie Code workspace skills

Version-controlled source for **Genie Code** (the in-notebook Databricks Assistant) **workspace skills**.
Each subfolder is one skill containing a `SKILL.md` ([Agent Skills](https://agentskills.io) spec: `name` +
`description` frontmatter, then Markdown instructions). Deployed to the workspace, Genie Code auto-loads a
skill when a request matches its `description`; users can also force it with `@<skill-name>`.

## Skills here

| Skill | Purpose | Mirrors |
|---|---|---|
| `protein-design-from-paper/` | Generate a full endpoints-only protein-design funnel **into the current notebook** for the user's target (binder / ligand-binder / motif / de novo / optimize). Self-contained — embeds the GWB endpoint toolkit so it works in a blank notebook anywhere. | `claude_skills/SKILL_GENESIS_WORKBENCH_PROTEIN_DESIGN_FROM_PAPER.md` (keep in sync) |

## Deploy (workspace-level — requires workspace admin)

Genie Code reads workspace skills from `/.assistant/skills/<name>/SKILL.md` at the workspace root. Deploy
with the Databricks CLI (`workshop` profile = the workshop workspace):

```bash
databricks workspace mkdirs /.assistant/skills/protein-design-from-paper --profile workshop
databricks workspace import /.assistant/skills/protein-design-from-paper/SKILL.md \
  --file genie_code_skills/protein-design-from-paper/SKILL.md \
  --format AUTO --overwrite --profile workshop
```

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
