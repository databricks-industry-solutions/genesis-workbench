#!/bin/bash
# ai_runtime_task command_path for bionemo_esm_finetune. Runs the fine-tune entry via the
# shared setup (deps + finetune_esm2 patch + job-param mapping). Staged to /Workspace by deploy.sh.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/command_common.sh" run_finetune.py
