#!/usr/bin/env python3
"""Test harness for the `mcp-genesis-workbench` MCP server.

Connects to the deployed MCP server over streamable HTTP, lists its tools, and
invokes each `endpoint_*` tool with a dtype-appropriate sample input — reporting
per-tool OK / PERMISSION / UNAVAILABLE / ERROR so you can see, in one table, which
endpoints the app SP can actually reach. Optionally dispatches the `workflow_*`
tools too.

WHY THIS EXISTS
  The MCP server runs as the MCP app's service principal. If that SP lacks
  CAN_QUERY on an endpoint (or CAN_MANAGE_RUN on a job) every call fails with a
  permission error — exactly what this harness surfaces, tool by tool.

AUTH
  Databricks Apps sit behind an OAuth proxy, so you need a *workspace OAuth access
  token* (a PAT is typically rejected by the apps gateway with 401/403). Get one:
      databricks auth login  --host <workspace-url>      # one-time U2M login
      databricks auth token  --host <workspace-url>      # prints {access_token: ...}
  …or copy the `Authorization: Bearer …` token the AI Playground/browser sends to
  the app (DevTools → Network → the /mcp request). Pass it via --token or
  $DATABRICKS_TOKEN. If neither is given, the harness tries `databricks auth token`.

USAGE
  python test_mcp_server.py --url https://mcp-genesis-workbench-<id>.aws.databricksapps.com/mcp \
      [--token "$TOKEN"] [--list-only] [--only endpoint_chemprop] [--dispatch-workflows] \
      [--timeout 120]

Requires: pip install "mcp>=1.27"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

try:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
except ImportError:
    sys.exit("Missing dependency: pip install 'mcp>=1.27'")

# A small, valid multi-residue PDB (3 ALA) for pdb-typed inputs. Good enough to get
# past auth/availability to a real model response (a model-level "too short" error
# still proves the endpoint is reachable AND the SP is authorized).
_SAMPLE_PDB = """ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N
ATOM      2  CA  ALA A   1      11.639   6.071  -5.147  1.00  0.00           C
ATOM      3  C   ALA A   1      13.149   5.853  -5.175  1.00  0.00           C
ATOM      4  O   ALA A   1      13.674   5.080  -5.972  1.00  0.00           O
ATOM      5  CB  ALA A   1      11.295   7.357  -4.404  1.00  0.00           C
ATOM      6  N   ALA A   2      13.850   6.533  -4.274  1.00  0.00           N
ATOM      7  CA  ALA A   2      15.305   6.421  -4.187  1.00  0.00           C
ATOM      8  C   ALA A   2      15.703   6.035  -2.764  1.00  0.00           C
ATOM      9  O   ALA A   2      15.012   6.346  -1.789  1.00  0.00           O
ATOM     10  CB  ALA A   2      15.971   7.737  -4.572  1.00  0.00           C
ATOM     11  N   ALA A   3      16.840   5.355  -2.664  1.00  0.00           N
ATOM     12  CA  ALA A   3      17.342   4.924  -1.364  1.00  0.00           C
ATOM     13  C   ALA A   3      18.853   4.740  -1.422  1.00  0.00           C
ATOM     14  O   ALA A   3      19.376   4.213  -2.405  1.00  0.00           O
ATOM     15  CB  ALA A   3      16.682   3.611  -0.954  1.00  0.00           C
TER      16      ALA A   3
END
"""

_SEQ = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"

# Sample value per input-port NAME (preferred) then per dtype (fallback). Endpoint
# tools expose their inputs as named args (see mcp_server._tool_for).
_SAMPLE_BY_NAME = {
    "smiles": "CC(=O)Oc1ccccc1C(=O)O",          # aspirin
    "ligand_smiles": "CC(=O)Oc1ccccc1C(=O)O",
    "sequence": _SEQ,
    "sequences": _SEQ,
    "protein_sequence": _SEQ,
    "target_sequence": _SEQ,
    "fasta": f">q\n{_SEQ}\n",
    "pdb": _SAMPLE_PDB,
    "pdb_string": _SAMPLE_PDB,
    "structure": _SAMPLE_PDB,
}
_SAMPLE_BY_DTYPE = {
    "smiles": "CC(=O)Oc1ccccc1C(=O)O",
    "sequence": _SEQ,
    "sequences": _SEQ,
    "pdb": _SAMPLE_PDB,
}


def _resolve_token(token: str | None) -> str | None:
    if token:
        return token
    if os.environ.get("DATABRICKS_TOKEN"):
        return os.environ["DATABRICKS_TOKEN"]
    try:  # best-effort: an OAuth (U2M) profile yields a usable access token
        out = subprocess.run(["databricks", "auth", "token"], capture_output=True, text=True, timeout=30)
        if out.returncode == 0:
            return json.loads(out.stdout).get("access_token")
    except Exception:
        pass
    return None


def _sample_args(tool) -> dict:
    """Build sample args from the tool's JSON inputSchema (required props only)."""
    schema = getattr(tool, "inputSchema", None) or {}
    props = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])
    args: dict = {}
    for name, spec in props.items():
        if required and name not in required:
            continue  # fill only required args; let optional params default
        if name in _SAMPLE_BY_NAME:
            args[name] = _SAMPLE_BY_NAME[name]
        else:
            # crude dtype guess from the property's type/description
            desc = (str(spec.get("description", "")) + " " + str(spec.get("type", ""))).lower()
            val = next((v for k, v in _SAMPLE_BY_DTYPE.items() if k in desc), None)
            args[name] = val if val is not None else _SAMPLE_BY_NAME["smiles"]
    return args


def _classify(is_error: bool, text: str) -> str:
    t = text.lower()
    if is_error:
        if any(s in t for s in ("permission", "does not have", "can_query", "can_manage_run",
                                "403", "forbidden", "not authorized", "no permission")):
            return "PERMISSION"
        if any(s in t for s in ("not ready", "not found", "no endpoint", "does not exist",
                                "unavailable", "timed out", "timeout", "502", "503", "scale")):
            return "UNAVAILABLE"
        return "ERROR"
    return "OK"


def _content_text(result) -> str:
    parts = []
    for c in getattr(result, "content", []) or []:
        parts.append(getattr(c, "text", "") or "")
    sc = getattr(result, "structuredContent", None)
    if sc:
        parts.append(json.dumps(sc))
    return " ".join(p for p in parts if p)[:500]


async def _run(url: str, token: str | None, only: str | None, list_only: bool,
               dispatch_workflows: bool, timeout: float) -> int:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    print(f"→ connecting to {url}" + (" (with bearer token)" if token else " (NO token — likely 401)"))
    try:
        return await _session(url, headers, token, only, list_only, dispatch_workflows, timeout)
    except Exception as eg:  # anyio surfaces a connect failure as an exception group
        excs = getattr(eg, "exceptions", None) or [eg]
        msgs = "; ".join(str(e) for e in excs)[:400]
        low = msgs.lower()
        if any(s in low for s in ("401", "403", "unauthorized", "forbidden", "invalid_token", "redirect")):
            print(f"✗ auth failed connecting to the MCP gateway: {msgs}")
            print("  → Databricks Apps require a WORKSPACE OAUTH token; a PAT is rejected by the gateway.")
            print("    Get one: `databricks auth login --host <ws>` then `databricks auth token` (access_token),")
            print("    or copy the Bearer token the AI Playground sends to /mcp. Pass via --token / $DATABRICKS_TOKEN.")
        else:
            print(f"✗ could not connect to the MCP server: {msgs}")
        return 2


async def _session(url, headers, token, only, list_only, dispatch_workflows, timeout) -> int:
    async with streamablehttp_client(url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            endpoints = [t for t in tools if t.name.startswith("endpoint_")]
            workflows = [t for t in tools if t.name.startswith("workflow_")]
            other = [t for t in tools if not t.name.startswith(("endpoint_", "workflow_"))]
            print(f"✓ connected. {len(tools)} tools: {len(endpoints)} endpoint, "
                  f"{len(workflows)} workflow, {len(other)} utility ({', '.join(t.name for t in other)})")
            if list_only:
                for t in tools:
                    print(f"   - {t.name}")
                return 0

            targets = endpoints + (workflows if dispatch_workflows else [])
            if only:
                targets = [t for t in targets if only in t.name]
            print(f"\nTesting {len(targets)} tool(s)"
                  + (f" matching {only!r}" if only else "")
                  + (" (incl. workflow dispatch)" if dispatch_workflows else " (endpoints only; --dispatch-workflows for jobs)") + ":\n")

            counts = {"OK": 0, "PERMISSION": 0, "UNAVAILABLE": 0, "ERROR": 0}
            rows = []
            for t in targets:
                args = _sample_args(t)
                t0 = time.time()
                try:
                    res = await asyncio.wait_for(session.call_tool(t.name, args), timeout=timeout)
                    text = _content_text(res)
                    status = _classify(bool(getattr(res, "isError", False)), text)
                except asyncio.TimeoutError:
                    status, text = "UNAVAILABLE", f"timed out after {timeout:.0f}s (cold start?)"
                except Exception as e:  # noqa: BLE001
                    status, text = _classify(True, str(e)), str(e)[:500]
                dt = time.time() - t0
                counts[status] = counts.get(status, 0) + 1
                icon = {"OK": "✅", "PERMISSION": "🔒", "UNAVAILABLE": "🌙", "ERROR": "❌"}[status]
                print(f"  {icon} {status:11s} {t.name:40s} {dt:5.1f}s  {text[:90]}")
                rows.append((t.name, status, round(dt, 1), text[:200]))

            print(f"\n=== summary: {counts['OK']} OK · {counts['PERMISSION']} permission · "
                  f"{counts['UNAVAILABLE']} unavailable/cold · {counts['ERROR']} error ===")
            if counts["PERMISSION"]:
                print("🔒 PERMISSION failures → the MCP app SP lacks CAN_QUERY/CAN_MANAGE_RUN. "
                      "Re-run the grant job with both app names.")
            return 1 if counts["PERMISSION"] or counts["ERROR"] else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Test the mcp-genesis-workbench MCP server tools.")
    ap.add_argument("--url", default=os.environ.get("MCP_URL", ""),
                    help="MCP endpoint URL (…databricksapps.com/mcp). Or $MCP_URL.")
    ap.add_argument("--token", default=None, help="Bearer token (else $DATABRICKS_TOKEN / `databricks auth token`).")
    ap.add_argument("--only", default=None, help="Only test tools whose name contains this substring.")
    ap.add_argument("--list-only", action="store_true", help="Just list tools (verifies connectivity/auth).")
    ap.add_argument("--dispatch-workflows", action="store_true", help="Also dispatch workflow_* tools (triggers jobs).")
    ap.add_argument("--timeout", type=float, default=120.0, help="Per-tool timeout seconds (default 120).")
    a = ap.parse_args()
    if not a.url:
        return ap.error("--url (or $MCP_URL) is required, e.g. https://mcp-genesis-workbench-<id>.aws.databricksapps.com/mcp")
    url = a.url if a.url.rstrip("/").endswith("/mcp") else a.url.rstrip("/") + "/mcp"
    token = _resolve_token(a.token)
    try:
        return asyncio.run(_run(url, token, a.only, a.list_only, a.dispatch_workflows, a.timeout))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
