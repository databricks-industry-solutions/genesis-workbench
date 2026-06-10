"""Shared capability executor — the single place that *runs* a capability,
called by all three pathways (UI, Vortex orchestrator, MCP server).

`execute_capability(cap, inputs, params)` dispatches by kind:
  - endpoint        : query a serving endpoint (synchronous)
  - databricks_job  : dispatch a Jobs run (async) -> run id; poll via run_status
  - endpoint_chain  : orchestrate a few endpoint calls (synchronous)
  - transform       : deterministic data reshape

Dependency-free beyond the Databricks SDK + pandas (already lib deps): no rdkit /
parasail / viewer code — those presentation concerns stay in the UI adapter.
"""
from __future__ import annotations

import io
import json
import logging
import os
import tempfile

from databricks.sdk import WorkspaceClient

from .capabilities import CHAIN, ENDPOINT, JOB, TRANSFORM, Capability
from .models import get_endpoint_name_for_uc_model

logger = logging.getLogger(__name__)


def _w(workspace_client: WorkspaceClient | None = None) -> WorkspaceClient:
    return workspace_client or WorkspaceClient()


# ─── endpoint ────────────────────────────────────────────────────────────────


def _query_endpoint(w: WorkspaceClient, cap: Capability, inputs: dict, params: dict):
    if cap.invoke_style == "records":
        record = {**(inputs or {}), **(params or {})}
        resp = w.serving_endpoints.query(name=cap.endpoint_name, dataframe_records=[record])
    else:
        vals = list((inputs or {}).values())
        primary = vals[0] if vals else None
        payload = primary if isinstance(primary, list) else [primary]
        resp = w.serving_endpoints.query(name=cap.endpoint_name, inputs=payload)
    return resp.predictions


# ─── job ─────────────────────────────────────────────────────────────────────


def _job_id_for(w: WorkspaceClient, job_name: str) -> int | None:
    for j in w.jobs.list(name=job_name):
        return int(j.job_id)
    return None


def _dispatch_job(w: WorkspaceClient, cap: Capability, params: dict) -> dict:
    job_id = _job_id_for(w, cap.job_name)
    if job_id is None:
        raise RuntimeError(f"Job '{cap.job_name}' not found / not deployed.")
    run = w.jobs.run_now(job_id=job_id, job_parameters={k: str(v) for k, v in (params or {}).items()})
    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    return {
        "run_id": str(run.run_id),
        "job_id": str(job_id),
        "run_url": f"{host}/jobs/{job_id}/runs/{run.run_id}" if host else "",
    }


def run_status(run_id: str, workspace_client: WorkspaceClient | None = None) -> dict:
    """Status of a dispatched job run (life-cycle + result + link)."""
    w = _w(workspace_client)
    run = w.jobs.get_run(run_id=int(run_id))
    st = run.state
    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    return {
        "run_id": str(run_id),
        "life_cycle_state": st.life_cycle_state.value if st and st.life_cycle_state else None,
        "result_state": st.result_state.value if st and st.result_state else None,
        "run_url": f"{host}/jobs/{run.job_id}/runs/{run_id}" if host and run.job_id else "",
    }


# ─── endpoint-chains ─────────────────────────────────────────────────────────


def _as_list(v) -> list:
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _chain_admet_screen(w: WorkspaceClient, inputs: dict, params: dict) -> dict:
    """Run the deployed ADMET / toxicity predictors over a SMILES set and combine."""
    smiles = _as_list((inputs or {}).get("smiles"))
    out: dict = {"smiles": smiles}
    for short, key in (("chemprop_admet", "admet"), ("chemprop_bbbp", "bbbp"),
                       ("chemprop_clintox", "clintox"), ("kermt_admet", "kermt")):
        try:
            ep = get_endpoint_name_for_uc_model(short)
            out[key] = w.serving_endpoints.query(name=ep, inputs=smiles).predictions
        except Exception as e:  # noqa: BLE001 — include only deployed predictors
            logger.info("admet_screen: %s unavailable: %s", short, e)
    return out


def _reindex_chain_pdb(pdb_text: str, chain_id: str = "A") -> str:
    """Reindex chain residues contiguously (RFDiffusion inpainting needs it).
    Uses BioPython — imported lazily so the core stays import-light; callers that
    run this chain must have `biopython` installed."""
    import os
    import tempfile

    from Bio import PDB
    from Bio.PDB import PDBParser

    with tempfile.NamedTemporaryFile(suffix=".pdb") as f:
        with open(f.name, "w") as fw:
            fw.write(pdb_text)
        structure = PDBParser().get_structure("s", f.name)
    chain = structure[0][chain_id]
    new_struct = PDB.Structure.Structure("new")
    new_model = PDB.Model.Model(0)
    new_chain = PDB.Chain.Chain(chain_id)
    for i, residue in enumerate((r for r in chain if r.id[0] == " "), start=1):
        residue.id = (" ", i, " ")
        new_chain.add(residue)
    new_model.add(new_chain)
    new_struct.add(new_model)
    io_ = PDB.PDBIO()
    io_.set_structure(new_struct)
    with tempfile.NamedTemporaryFile(suffix=".pdb") as f:
        io_.save(f.name)
        with open(f.name) as fh:
            return fh.read()


def _chain_protein_design(w: WorkspaceClient, inputs: dict, params: dict) -> dict:
    """ESMFold -> reindex -> RFDiffusion x N -> ProteinMPNN -> ESMFold each design.
    Returns the initial + designed structures (lean data; the UI adds the viewer +
    alignment on top). Endpoint payloads mirror services/protein.py."""
    seq_in = str((inputs or {}).get("sequence", ""))
    start_idx, end_idx = seq_in.find("["), seq_in.find("]")
    raw = seq_in.replace("[", "").replace("]", "")
    n = int((params or {}).get("n_rfdiffusion_hits", 1) or 1)

    esmfold = get_endpoint_name_for_uc_model("esmfold")
    rfdiffusion = get_endpoint_name_for_uc_model("rfdiffusion_inpainting")
    proteinmpnn = get_endpoint_name_for_uc_model("proteinmpnn")

    def _fold(seq: str) -> str:
        return w.serving_endpoints.query(name=esmfold, inputs=[seq]).predictions[0]

    initial = _fold(raw)
    modified = _reindex_chain_pdb(initial, "A")

    designed_pdbs = [
        w.serving_endpoints.query(
            name=rfdiffusion,
            inputs=[{"pdb": modified, "start_idx": start_idx, "end_idx": end_idx}],
        ).predictions[0]
        for _ in range(n)
    ]
    seqs: list[str] = []
    for pdb in designed_pdbs:
        r = w.serving_endpoints.query(
            name=proteinmpnn, dataframe_records=[{"pdb": pdb, "fixed_positions": ""}]
        ).predictions
        seqs.extend(r if isinstance(r, list) else [r])
    return {"initial": initial, "sequences": seqs, "designs": [_fold(s) for s in seqs]}


_CHAINS = {"admet_screen": _chain_admet_screen, "protein_design": _chain_protein_design}

# Both chains execute now (protein_design needs biopython where it runs).
RUNNABLE_CHAINS = {"admet_screen", "protein_design"}


# ─── transforms ──────────────────────────────────────────────────────────────


def _read_volume_text(w: WorkspaceClient, path: str) -> str:
    resp = w.files.download(path)
    data = resp.contents.read() if hasattr(resp, "contents") else resp.read()
    return data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else str(data)


def _dig(obj, dotted: str):
    cur = obj
    for part in (p for p in (dotted or "").split(".") if p != ""):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _run_transform(w: WorkspaceClient, cap: Capability, inputs: dict, params: dict):
    op = cap.op
    inputs, params = inputs or {}, params or {}
    if op == "read_text_file":
        return {"text": _read_volume_text(w, inputs.get("file"))}
    if op == "parse_fasta":
        text = _read_volume_text(w, inputs.get("file"))
        seqs, cur = [], []
        for line in text.splitlines():
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
            elif line.strip():
                cur.append(line.strip())
        if cur:
            seqs.append("".join(cur))
        return {"sequences": seqs}
    if op == "csv_column":
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, "in.csv")
            with open(local, "w") as f:
                f.write(_read_volume_text(w, inputs.get("table")))
            df = pd.read_csv(local)
        col = params.get("column")
        return {"values": df[col].tolist() if col in df.columns else []}
    if op == "extract_field":
        return {"value": _dig(inputs.get("data"), params.get("path", ""))}
    if op == "field_mapper":
        mapping = params.get("mappings") or "{}"
        mapping = json.loads(mapping) if isinstance(mapping, str) else mapping
        data = inputs.get("data")
        return {"mapped": {tgt: _dig(data, src) for tgt, src in mapping.items()}}
    if op == "select_top_k":
        items = _as_list(inputs.get("items"))
        by, k = params.get("by"), int(params.get("k", 5) or 5)
        rev = (params.get("order", "desc") != "asc")
        if by:
            items = sorted(items, key=lambda x: (x or {}).get(by, 0) if isinstance(x, dict) else 0, reverse=rev)
        return {"top": items[:k]}
    raise RuntimeError(f"Unknown transform op '{op}'")


# ─── dispatch ────────────────────────────────────────────────────────────────


def execute_capability(
    cap: Capability,
    inputs: dict | None = None,
    params: dict | None = None,
    workspace_client: WorkspaceClient | None = None,
):
    """Run a capability. `inputs` maps input-port name -> value; `params` maps
    param name -> value. Endpoints/chains/transforms return results synchronously;
    jobs return {run_id, run_url} (poll with `run_status`)."""
    w = _w(workspace_client)
    if cap.kind == ENDPOINT:
        return _query_endpoint(w, cap, inputs or {}, params or {})
    if cap.kind == JOB:
        return _dispatch_job(w, cap, params or {})
    if cap.kind == CHAIN:
        fn = _CHAINS.get(cap.chain_id or "")
        if fn is None:
            raise RuntimeError(f"No executor for chain '{cap.chain_id}'.")
        return fn(w, inputs or {}, params or {})
    if cap.kind == TRANSFORM:
        return _run_transform(w, cap, inputs or {}, params or {})
    raise RuntimeError(f"Unknown capability kind '{cap.kind}'.")
