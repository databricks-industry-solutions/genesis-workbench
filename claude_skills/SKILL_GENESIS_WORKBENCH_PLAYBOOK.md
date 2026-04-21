---
name: genesis-workbench-playbook
description: Maintain and extend the Genesis Workbench demo playbook (Google Doc). Use when the user says things like "update the playbook", "add to GWB demo doc", "bump playbook version", "the TOC is out of date", or when new GWB features need documentation. Knows the specific tab layout, doc ID, and tab IDs of the live GWB playbook. Defers to the global vertical-solution-playbook skill for generic patterns.
---

# Genesis Workbench Playbook Maintenance Skill

Specific to the GWB Demo Playbook Google Doc. For generic playbook-creation patterns applicable to any vertical solution, see the global `vertical-solution-playbook` skill at `~/.claude/skills/vertical-solution-playbook/SKILL.md`.

## Playbook doc

- **URL:** https://docs.google.com/document/d/19pTNC5ok6V1DwA7_hz4Pbi8h8P1ZdPlWUtf58z50SvU/edit
- **Doc ID:** `19pTNC5ok6V1DwA7_hz4Pbi8h8P1ZdPlWUtf58z50SvU`
- **Title:** "GWB Demo Playbook"
- **Maintainer:** May Merkle-Tan (may.merkletan@databricks.com)
- **Audience:** Internal Databricks SAs; pharma / drug-discovery customers; partners evaluating GWB as a reference architecture
- **Source of truth for code grounding:** `version_pinning` branch + local `mmt/ver_pin_sandbox_setup` branch in this repo

## Tab layout (as of 2026-04-21, v0.1.1)

23 content-populated tabs + 5 empty reserve tabs. Order matches Google Doc UI display order.

| # | Tab title | Tab ID | H1 | Purpose |
|---|---|---|---|---|
| 1 | Contents | `t.0` | Contents | TOC + framing |
| 2 | Pre-demo setup | `t.rd60sara42kn` | Pre-demo setup | Profile setup + endpoint warm-up + checklist |
| 3 | Basic app v0 | `t.2vssmoj3i53o` | Basic app | gwb-mmt-app walkthrough (smaller surface) |
| 4 | Home & AI Assistant | `t.r062dcvvatr5` | Home & AI Assistant | Home tab features |
| 5 | Protein — Sequence Search | `t.no4iyhh0pxev` | Protein — Sequence Search | NEW workflow |
| 6 | Protein — Structure | `t.uy50run19f8j` | Protein — Structure | ESMFold / AlphaFold2 / Boltz |
| 7 | Protein — Design + Inverse Folding | `t.xt4nslf6da6c` | Protein — Design + Inverse Folding | RFDiffusion + ProteinMPNN |
| 8 | Single Cell — Raw Processing | `t.x0v35rxip19z` | Single Cell — Raw Processing | Scanpy / Rapids pipeline |
| 9 | Single Cell — Annotation | `t.q6ieezrpjztc` | Single Cell — Annotation | SCimilarity + NB 06b |
| 10 | Single Cell — Perturbation | `t.h7xldefhmsv6` | Single Cell — Perturbation | NEW scGPT workflow |
| 11 | Small Molecule — Docking | `t.oo73i4fsbbgu` | Small Molecule — Docking | NEW DiffDock |
| 12 | Small Molecule — Binder Design | `t.51rn1d11y8t0` | Small Molecule — Binder Design | Proteina-Complexa family |
| 13 | Small Molecule — ADMET | `t.jyrwbmbwl54z` | Small Molecule — ADMET | NEW Chemprop flow |
| 14 | Disease Biology — GWAS | `t.23m4ovasjvaq` | Disease Biology — GWAS | NEW Glow pipeline |
| 15 | Disease Biology — Variant Calling | `t.l5pdqsdh4s3p` | Disease Biology — Variant Calling | Parabricks |
| 16 | Disease Biology — VCF + Annotation | `t.4yqr1b7xk68y` | Disease Biology — VCF + Annotation | NEW Glow + ClinVar |
| 17 | NVIDIA — BioNeMo | `t.6pz3h0vmwi0g` | NVIDIA — BioNeMo | ESM2 finetune + inference |
| 18 | NVIDIA — Parabricks | `t.laagocon1t44` | NVIDIA — Parabricks | GPU genomics |
| 19 | Platform Pattern (Blueprint) | `t.36q45ptuj0lg` | Platform Pattern | Reusable reference architecture |
| 20 | Setup & Caveats | `t.ghpm3jrikxfj` | Setup & Caveats | Install reference + UX gaps subset |
| 21 | FAQ — know before you deploy | `t.2f9qced8aatt` | FAQ — know before you deploy | Pre-deploy user FAQ |
| 22 | screenshots list | `t.nadlyjun15x3` | Good screenshots to grab... | Hero screenshot inventory |
| 23 | Changelog | `t.e16fjl754bzq` | Changelog | Version history |

Tabs 24-28 (`t.ewtgc7ypwqyc`, `t.416pj86d0l3h`, `t.qjn9lfa5mdkl`, `t.uyc844yvf4rv`, `t.if0es5zqiq1`) are empty reserves.

## Maintenance operations

### Editing a feature tab

1. Follow the per-tab content pattern (see global skill): H1 → intro → H2 sections (What it does / Inputs / Behind the scenes / Output / Demo tips / Known gotchas / Code / Pairs with).
2. Ground new narrative in code — read the relevant app view file + register notebook before writing.
3. Image placeholders in italic gray: `[IMAGE: <description>]`.
4. Cross-references to other tabs via anchor URLs: `https://docs.google.com/document/d/19pTNC5ok6V1DwA7_hz4Pbi8h8P1ZdPlWUtf58z50SvU/edit?tab=<tabId>`.
5. After any bulk edit, run the style-leak scanner to catch HEADING_2 cascade leaks.

### After rearranging tabs (UI action)

The display order changed but tab IDs are stable. Rebuild Tab 1's TOC:

1. Fetch current tab order with `?includeTabsContent=true`
2. Delete the existing TOC table in Tab 1 via `deleteContentRange`
3. Insert fresh table sized `(num_content_tabs + 1) × 3`
4. Populate: header row + one data row per tab in display order
5. Re-apply hyperlinks on the `Tab` column using each tab's `tabId`

### Adding a new feature tab

1. User adds the empty tab in the Google Docs UI (or via API if reliable)
2. Populate using the per-tab pattern
3. Add a row to the Tab 1 TOC in the correct display position
4. Append an entry to the Changelog table (Version | Date | Maintainer | Summary | Kind=structure+content)
5. Cross-reference from related tabs' "Pairs with" footer

### Bumping the changelog

Append a new row at the top of the Changelog table in Tab 23 (`t.e16fjl754bzq`). Format:

```
<version> | <YYYY-MM-DD> | <maintainer email/name> | <1-2 sentence summary of changes> | content / structure / asset / correction / gap
```

Also update "Current version" block at top of the tab (Version / Last updated / Status).

## Style-leak scanner

After any Google Docs API operation that modifies paragraph styles, run:

```python
# Pseudo-code; full implementation in scripts/ or the global skill
for each tab:
    expected_h1 = {tab_title}  # only the H1 title should be HEADING_1
    expected_h2 = set(...section_titles_per_this_tab...)  # from known mapping
    for each paragraph in tab.body:
        if paragraph.style == HEADING_1 and text not in expected_h1: flag
        if paragraph.style == HEADING_2 and text not in expected_h2: flag
    reset flagged paragraphs to NORMAL_TEXT
```

Maintain the `expected_h2` mapping per tab in this skill (update when adding new H2 sections). See GWB-specific `LEGIT` dict in the session history for current per-tab expected headings.

## Screenshots workflow

- **Local working folder:** `gwb_app_imgs/` (relative to May's working directory — not checked into git). Screenshots land here first as `tab<N>_<descriptor>.png`.
- **Then:** copied to a Google Drive folder alongside the playbook doc, so shared access works without re-uploading every image.
- **Then:** pasted into the matching tab's `[IMAGE: <description>]` placeholder in the Google Doc. Google Docs embeds are a COPY of the file — editing the Drive original doesn't update the inline; re-paste if the screenshot changes.
- **Tab 22 (screenshots list)** in the playbook tracks which shots have been captured + where each one lands.

## Related tooling in this repo

- `scripts/watch_gwb_jobs.py` — job monitor; useful while playbook walks the app during sandbox deploys
- `claude_skills/SKILL_GENESIS_WORKBENCH_DEPLOY_MONITOR.md` — companion skill for runtime observability
- `docs/deployments/fevm-mmt-aws-usw2/UX-GAPS.md` — running list of gaps; feeds FAQ + Setup & Caveats content
- `docs/deployments/fevm-mmt-aws-usw2/SESSION-NOTES.md` — deploy runbook
- `docs/deployments/docker-secrets-convention.md` — cross-referenced in Setup & Caveats

## Pairs with

- Global skill: `vertical-solution-playbook` (generic patterns applicable here + other repos)
- `SKILL_GENESIS_WORKBENCH_DEPLOY_WIZARD` — gets the stack deployed so there's something to walk
- `SKILL_GENESIS_WORKBENCH_DEPLOY_MONITOR` — observes the stack during/after deploy
- `SKILL_GENESIS_WORKBENCH_WORKFLOWS` — canonical per-workflow UI description (primary source for feature tab content)
- `SKILL_GENESIS_WORKBENCH_TROUBLESHOOTING` — static recipes for known failures
- `fe-google-tools:google-docs` — Google Docs API primitives

## Instructions

1. When user asks to update the GWB playbook, reference this skill for the specific tab layout + conventions.
2. Defer to the global `vertical-solution-playbook` skill for generic patterns (API gotchas, TOC rebuild, style-leak scanner).
3. On every non-trivial edit, bump the Changelog in Tab 23.
4. On any bulk style operation, run the style-leak scanner before reporting completion.
5. Image placeholder convention: italic gray `[IMAGE: description]`.
6. Cross-tab links: `?tab=<tabId>` URL anchors (doc id + tab id both needed).
