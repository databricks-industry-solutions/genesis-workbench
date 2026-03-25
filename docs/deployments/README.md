# Deployment Logs

Workspace-specific deployment documentation. Each subfolder contains debug logs, status snapshots, and operational notes for a specific workspace deployment.

## Convention

Deployment logs are **gitignored by default** — they contain workspace-specific job IDs, run URLs, and point-in-time debugging artifacts that aren't useful outside the original deployment context.

Reusable deployment knowledge (fixes, workarounds, configuration guides) should be consolidated into the root-level `CHANGELOG.md` so it's available to anyone deploying to a new workspace.

### What stays local (gitignored)

| File | Purpose |
|------|---------|
| `deploy-log.md` | Raw chronological log of commands and issues |
| `genesis-workbench-redeploy.md` | Master task tracker with workspace-specific job IDs |
| `redeploy-failed-jobs.md` | Targeted redeploy guide with specific run URLs |
| `status-YYYY-MM-DD.md` | Point-in-time status snapshots |
| `*-debug-summary.md` | Issue-specific debug writeups |
| `verify_deployment.py` | Verification script with workspace-specific IDs |
| `CHANGES.md` | Per-deployment change summary (consolidated into root CHANGELOG.md) |

### What goes in root CHANGELOG.md (tracked in git)

- Dependency fixes and version pins
- Code changes and bug fixes
- Configuration patterns (AI Gateway, workload sizing, etc.)
- Dataset compatibility notes
- Architecture decisions and rationale

## Deploying to a new workspace

1. Create a folder: `docs/deployments/<workspace-name>/`
2. Add it to `.gitignore` in this directory if you want to keep logs local
3. Use the root `CHANGELOG.md` as your reference for known issues and fixes
4. Create a workspace-specific `verify_deployment.py` with your job IDs and endpoint names
