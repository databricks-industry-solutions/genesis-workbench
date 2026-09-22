#!/bin/bash
# ai_runtime_task command_path for bionemo_esm_inference. Runs the inference entry via the
# shared setup. Staged to /Workspace by deploy.sh.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/command_common.sh" run_inference.py
