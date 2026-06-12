"""Package smoke test — the wheel imports and the built-in node catalog loads.

(Replaces a stale placeholder that imported a non-existent Workbench class.)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_builtin_catalog_loads():
    # stdlib-only import path (no Databricks deps needed)
    from genesis_workbench.builtin_nodes import CURATED_NODES

    assert CURATED_NODES, "built-in node catalog is empty"
    types = [n.type for n in CURATED_NODES]
    assert len(types) == len(set(types)), "duplicate node types in the catalog"
    assert all(n.category for n in CURATED_NODES), "every node must declare a category"


def test_node_dict_round_trip():
    from genesis_workbench.builtin_nodes import CURATED_NODES
    from genesis_workbench.node_catalog import node_from_dict, node_to_dict

    assert all(node_from_dict(node_to_dict(n)) == n for n in CURATED_NODES)
