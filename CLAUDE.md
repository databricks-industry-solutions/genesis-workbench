# Genesis Workbench — Claude Code Entry Point

When a user in this repo asks for help with deployment, installation, development, troubleshooting, or workflow usage, load the relevant skill below **before** answering. These files contain authoritative step-by-step guidance, error auto-handlers, and validation commands that supersede general knowledge.

## Skill index

| User intent | Skill file |
|---|---|
| "deploy Genesis Workbench", "install GWB", "set up on new workspace", "run deploy.sh" | [`claude_skills/SKILL_GENESIS_WORKBENCH_DEPLOY_WIZARD.md`](claude_skills/SKILL_GENESIS_WORKBENCH_DEPLOY_WIZARD.md) |
| Reference docs for installation / config / deployment mechanics | [`claude_skills/SKILL_GENESIS_WORKBENCH_INSTALLATION.md`](claude_skills/SKILL_GENESIS_WORKBENCH_INSTALLATION.md) |
| Overview of GWB — modules, capabilities, architecture | [`claude_skills/SKILL_GENESIS_WORKBENCH.md`](claude_skills/SKILL_GENESIS_WORKBENCH.md) |
| Deploy failures, registration errors, endpoint issues, runtime errors | [`claude_skills/SKILL_GENESIS_WORKBENCH_TROUBLESHOOTING.md`](claude_skills/SKILL_GENESIS_WORKBENCH_TROUBLESHOOTING.md) |
| Adding new models / workflows / UI tabs | [`claude_skills/SKILL_GENESIS_WORKBENCH_DEVELOPMENT.md`](claude_skills/SKILL_GENESIS_WORKBENCH_DEVELOPMENT.md) |
| Using the GWB UI — how each tab works, inputs/outputs | [`claude_skills/SKILL_GENESIS_WORKBENCH_WORKFLOWS.md`](claude_skills/SKILL_GENESIS_WORKBENCH_WORKFLOWS.md) |

## Quick start — deploy

From this repo root:
```bash
./deploy.sh core <aws|azure|gcp>     # must be first; creates modules/core/.deployed
./deploy.sh <module> <cloud>          # each additional module, one at a time
```

Deploy modules **one at a time** and wait for each module's first post-deploy job to reach `RUNNING` before launching the next — this serializes GPU cluster-create and surfaces quota issues one module at a time. Full rationale + polling commands in the deploy wizard skill above.
