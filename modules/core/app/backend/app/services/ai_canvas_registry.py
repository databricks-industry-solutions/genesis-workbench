"""Vortex (ai_canvas) node-type registry — RE-EXPORT shim.

The node catalog now lives in the shared core: the model in
`genesis_workbench.node_catalog` and the hand-authored nodes in
`genesis_workbench.builtin_nodes`. This module re-exports them so existing
`app.services.ai_canvas_registry` importers keep working unchanged.

Authoring happens in the wheel (builtin_nodes); a deploy notebook publishes it to
the node_catalog table, which the app / executor / MCP all read.
"""
from __future__ import annotations

from genesis_workbench.builtin_nodes import (  # noqa: F401
    CURATED_BY_ENDPOINT,
    CURATED_BY_JOB,
    CURATED_BY_TYPE,
    CURATED_NODES,
)
from genesis_workbench.node_catalog import (  # noqa: F401
    NodeCategory,
    NodeType,
    ParamField,
    Port,
    PortType,
)
