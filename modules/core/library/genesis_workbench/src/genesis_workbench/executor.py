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
from databricks.sdk.service.serving import DataframeSplitInput

from .capabilities import CHAIN, ENDPOINT, JOB, TRANSFORM, Capability
from .models import get_endpoint_name_for_uc_model


# ─── shared endpoint-call helpers (dep-free) ─────────────────────────────────


def _df_split(w: WorkspaceClient, short: str, columns: list[str], row: list) -> list[dict]:
    """Query a `dataframe_split` endpoint (Proteina-Complexa variants, DiffDock)
    by UC short name; returns the predictions as a list of dicts."""
    ep = get_endpoint_name_for_uc_model(short)
    resp = w.serving_endpoints.query(
        name=ep, dataframe_split=DataframeSplitInput(columns=columns, data=[row])
    )
    preds = resp.predictions
    if isinstance(preds, dict) and "predictions" in preds:
        preds = preds["predictions"]
    return [p for p in (preds or []) if isinstance(p, dict)]


def _esmfold_pdb(w: WorkspaceClient, seq: str) -> str:
    ep = get_endpoint_name_for_uc_model("esmfold")
    return w.serving_endpoints.query(name=ep, inputs=[seq]).predictions[0]


def _proteinmpnn_seqs(w: WorkspaceClient, pdb: str) -> list[str]:
    ep = get_endpoint_name_for_uc_model("proteinmpnn")
    preds = w.serving_endpoints.query(
        name=ep, dataframe_records=[{"pdb": pdb, "fixed_positions": ""}]
    ).predictions
    if isinstance(preds, dict) and "predictions" in preds:
        preds = preds["predictions"]
    return [str(s) for s in (preds or [])]


def _diffdock_best(w: WorkspaceClient, protein_pdb: str, ligand_smiles: str, n: int = 5):
    """ESM-2 embed -> DiffDock; return (best_sdf, confidence) or (None, None)."""
    emb = _df_split(w, "diffdock_esm_embeddings", ["protein_pdb"], [protein_pdb])
    b64 = emb[0].get("embeddings_b64", "{}") if emb else "{}"
    poses = _df_split(
        w, "diffdock",
        ["protein_pdb", "ligand_smiles", "samples_per_complex", "esm_embeddings_b64"],
        [protein_pdb, ligand_smiles, n, b64],
    )
    best = None
    for p in poses:
        sdf = str(p.get("ligand_sdf", ""))
        if sdf.startswith("ERROR"):
            continue
        c = float(p.get("confidence", -1e9) or -1e9)
        if best is None or c > best[0]:
            best = (c, sdf)
    return (best[1], best[0]) if best else (None, None)

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


def _as_bool(v) -> bool:
    """Canvas/MCP params arrive as a real bool or a string ('true'/'false')."""
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes")


def _emit(progress, pct: int, msg: str) -> None:
    if progress:
        try:
            progress(pct, msg)
        except Exception:  # noqa: BLE001 — a bad callback must not break execution
            pass


def _chain_admet_screen(w: WorkspaceClient, inputs: dict, params: dict, progress=None) -> dict:
    """Run the selected ADMET / toxicity predictors over a SMILES set and combine.
    Per-predictor toggles (run_admet/run_bbbp/run_clintox/run_kermt) mirror the UI;
    default ON except KERMT."""
    p = params or {}
    smiles = _as_list((inputs or {}).get("smiles"))
    out: dict = {"smiles": smiles}
    # (short, out-key, label, run-toggle-param, default-on)
    candidates = [("chemprop_admet", "admet", "ADMET (multi-task)", "run_admet", True),
                  ("chemprop_bbbp", "bbbp", "BBB penetration", "run_bbbp", True),
                  ("chemprop_clintox", "clintox", "clinical toxicity", "run_clintox", True),
                  ("kermt_admet", "kermt", "KERMT", "run_kermt", False)]
    preds = [(s, k, lbl) for (s, k, lbl, tog, dflt) in candidates if _as_bool(p.get(tog, dflt))]
    for i, (short, key, label) in enumerate(preds):
        _emit(progress, 10 + int(i / len(preds) * 85), f"Predicting {label}")
        try:
            ep = get_endpoint_name_for_uc_model(short)
            out[key] = w.serving_endpoints.query(name=ep, inputs=smiles).predictions
        except Exception as e:  # noqa: BLE001 — include only deployed predictors
            logger.info("admet_screen: %s unavailable: %s", short, e)
    _emit(progress, 100, "ADMET screen complete")
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


def _chain_protein_design(w: WorkspaceClient, inputs: dict, params: dict, progress=None) -> dict:
    """ESMFold -> reindex -> RFDiffusion x N -> ProteinMPNN -> ESMFold each design.
    Returns the initial + designed structures (lean data; the caller adds the
    viewer + alignment + MLflow on top). Endpoint payloads mirror services/protein.py.
    `progress(pct, msg)` fires at each stage if provided."""
    seq_in = str((inputs or {}).get("sequence", ""))
    start_idx, end_idx = seq_in.find("["), seq_in.find("]")
    # The region to redesign MUST be marked with square brackets, e.g.
    # MKT[AYIAK]QRQ… — RFDiffusion inpaints that span. Without it the contig
    # indices are empty and RFDiffusion fails with a cryptic
    # "invalid literal for int() with base 10: ''". Fail clearly instead.
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx + 1:
        raise RuntimeError(
            "Protein Design needs a region to redesign: mark it in the sequence with "
            "square brackets, e.g. MKT[AYIAK]QRQ. The provided sequence has no [ ] region."
        )
    raw = seq_in.replace("[", "").replace("]", "")
    n = int((params or {}).get("n_rfdiffusion_hits", 1) or 1)

    esmfold = get_endpoint_name_for_uc_model("esmfold")
    rfdiffusion = get_endpoint_name_for_uc_model("rfdiffusion_inpainting")
    proteinmpnn = get_endpoint_name_for_uc_model("proteinmpnn")

    def _fold(seq: str) -> str:
        return w.serving_endpoints.query(name=esmfold, inputs=[seq]).predictions[0]

    _emit(progress, 5, "Folding original sequence (ESMFold)")
    initial = _fold(raw)
    modified = _reindex_chain_pdb(initial, "A")

    designed_pdbs = []
    for i in range(n):
        _emit(progress, 10 + int(i / max(n, 1) * 40), f"RFDiffusion scaffold {i + 1}/{n}")
        designed_pdbs.append(
            w.serving_endpoints.query(
                name=rfdiffusion,
                inputs=[{"pdb": modified, "start_idx": start_idx, "end_idx": end_idx}],
            ).predictions[0]
        )
    seqs: list[str] = []
    for i, pdb in enumerate(designed_pdbs):
        _emit(progress, 50 + int(i / max(len(designed_pdbs), 1) * 20),
              f"ProteinMPNN sequence design {i + 1}/{len(designed_pdbs)}")
        r = w.serving_endpoints.query(
            name=proteinmpnn, dataframe_records=[{"pdb": pdb, "fixed_positions": ""}]
        ).predictions
        seqs.extend(r if isinstance(r, list) else [r])
    designs = []
    for i, s in enumerate(seqs):
        _emit(progress, 70 + int(i / max(len(seqs), 1) * 25), f"Folding designed sequence {i + 1}/{len(seqs)}")
        designs.append(_fold(s))
    _emit(progress, 100, "Protein design complete")
    return {"initial": initial, "sequences": seqs, "designs": designs}


_PC_COLS = ["target_pdb", "binder_length_min", "binder_length_max",
            "num_samples", "hotspot_residues", "target_chain"]


def _chain_protein_binder_design(w, inputs, params, progress=None) -> dict:
    """Proteina-Complexa binder design for a target protein (+ optional ESMFold
    validation). Target comes in as PDB, or as a sequence we fold first."""
    p, ins = params or {}, inputs or {}
    target_pdb = ins.get("target_pdb")
    if not target_pdb:
        seq = str(ins.get("target_sequence", "")).strip()
        if not seq:
            raise RuntimeError("protein_binder_design: target_pdb or target_sequence required")
        _emit(progress, 5, "Folding target sequence (ESMFold)")
        target_pdb = _esmfold_pdb(w, seq)
    n = int(p.get("num_samples", 2) or 2)
    _emit(progress, 20, f"Generating {n} binder design(s) (Proteina-Complexa)")
    rows = _df_split(w, "proteina_complexa", _PC_COLS,
                     [target_pdb, int(p.get("binder_length_min", 50) or 50),
                      int(p.get("binder_length_max", 80) or 80), n,
                      str(p.get("hotspot_residues", "") or ""), str(p.get("target_chain", "A") or "A")])
    validate = bool(p.get("validate_esmfold", False))
    designs = []
    for i, r in enumerate(rows):
        binder_pdb = r.get("pdb_output")
        validated = False
        if validate:
            _emit(progress, 50 + int((i + 1) / max(len(rows), 1) * 45),
                  f"Validating design {i + 1}/{len(rows)} (ESMFold)")
            try:
                folded = _esmfold_pdb(w, str(r.get("sequence", "")))
                if folded:
                    binder_pdb, validated = folded, True
            except Exception as e:  # noqa: BLE001
                logger.info("protein_binder esmfold validate failed: %s", e)
        designs.append({"sample_id": str(r.get("sample_id", "")), "sequence": str(r.get("sequence", "")),
                        "rewards": float(r.get("rewards", 0) or 0), "binder_pdb": binder_pdb,
                        "esmfold_validated": validated})
    _emit(progress, 100, "Binder design complete")
    return {"designs": designs, "target_pdb": target_pdb}


def _chain_ligand_binder_design(w, inputs, params, progress=None) -> dict:
    """Proteina-Complexa-Ligand binder design for a ligand (PDB in — SMILES->PDB
    stays in the caller). Optional ESMFold + DiffDock validation per design."""
    p, ins = params or {}, inputs or {}
    ligand_pdb = ins.get("ligand_pdb")
    if not ligand_pdb:
        raise RuntimeError("ligand_binder_design: ligand_pdb required (convert SMILES upstream)")
    n = int(p.get("num_samples", 2) or 2)
    _emit(progress, 15, f"Generating {n} protein binder(s) for the ligand")
    rows = _df_split(w, "proteina_complexa_ligand", _PC_COLS,
                     [ligand_pdb, int(p.get("binder_length_min", 50) or 50),
                      int(p.get("binder_length_max", 80) or 80), n, "", "A"])
    validate_fold = bool(p.get("validate_esmfold", False))
    smiles = str(p.get("ligand_smiles", "") or "")
    validate_dock = bool(p.get("validate_diffdock", False)) and bool(smiles)
    designs = []
    for i, r in enumerate(rows):
        esmfold_pdb = None
        if validate_fold:
            _emit(progress, 35 + int((i + 1) / max(len(rows), 1) * 25), f"Folding design {i + 1}/{len(rows)}")
            try:
                esmfold_pdb = _esmfold_pdb(w, str(r.get("sequence", "")))
            except Exception as e:  # noqa: BLE001
                logger.info("ligand_binder esmfold failed: %s", e)
        dock_sdf = dock_conf = None
        if validate_dock:
            _emit(progress, 60 + int((i + 1) / max(len(rows), 1) * 30), f"Docking design {i + 1}/{len(rows)}")
            try:
                dock_sdf, dock_conf = _diffdock_best(w, esmfold_pdb or r.get("pdb_output"), smiles, 5)
            except Exception as e:  # noqa: BLE001
                logger.info("ligand_binder diffdock failed: %s", e)
        designs.append({"sample_id": str(r.get("sample_id", "")), "sequence": str(r.get("sequence", "")),
                        "rewards": float(r.get("rewards", 0) or 0), "pdb_output": str(r.get("pdb_output", "")),
                        "esmfold_pdb": esmfold_pdb, "best_dock_sdf": dock_sdf, "dock_confidence": dock_conf})
    _emit(progress, 100, "Ligand binder design complete")
    return {"designs": designs, "ligand_pdb": ligand_pdb}


def _chain_motif_scaffolding(w, inputs, params, progress=None) -> dict:
    """Proteina-Complexa-AME scaffold generation (+ optional ProteinMPNN
    optimisation + ESMFold validation) preserving a functional motif."""
    p, ins = params or {}, inputs or {}
    motif_pdb = ins.get("motif_pdb")
    if not motif_pdb:
        raise RuntimeError("motif_scaffolding: motif_pdb required")
    n = int(p.get("num_samples", 2) or 2)
    _emit(progress, 10, f"Generating {n} scaffold(s) (Proteina-Complexa-AME)")
    rows = _df_split(w, "proteina_complexa_ame", _PC_COLS,
                     [motif_pdb, int(p.get("scaffold_length_min", 50) or 50),
                      int(p.get("scaffold_length_max", 80) or 80), n, "", str(p.get("target_chain", "B") or "B")])
    optimize = bool(p.get("optimize_mpnn", False))
    validate = bool(p.get("validate_esmfold", False))
    scaffolds = []
    for i, r in enumerate(rows):
        seq = str(r.get("sequence", ""))
        mpnn_seq = None
        if optimize:
            _emit(progress, 35 + int((i + 1) / max(len(rows), 1) * 25), f"Optimising sequence {i + 1}/{len(rows)} (ProteinMPNN)")
            try:
                seqs = _proteinmpnn_seqs(w, str(r.get("pdb_output", "")))
                mpnn_seq = seqs[0] if seqs else None
            except Exception as e:  # noqa: BLE001
                logger.info("motif proteinmpnn failed: %s", e)
        esmfold_pdb = None
        validated = False
        if validate:
            _emit(progress, 60 + int((i + 1) / max(len(rows), 1) * 30), f"Folding scaffold {i + 1}/{len(rows)}")
            try:
                esmfold_pdb = _esmfold_pdb(w, mpnn_seq or seq)
                validated = esmfold_pdb is not None
            except Exception as e:  # noqa: BLE001
                logger.info("motif esmfold failed: %s", e)
        scaffolds.append({"sample_id": str(r.get("sample_id", "")), "sequence": seq, "mpnn_sequence": mpnn_seq,
                          "rewards": float(r.get("rewards", 0) or 0), "pdb_output": str(r.get("pdb_output", "")),
                          "esmfold_pdb": esmfold_pdb, "esmfold_validated": validated})
    _emit(progress, 100, "Motif scaffolding complete")
    return {"scaffolds": scaffolds, "motif_pdb": motif_pdb}


_CHAINS = {
    "admet_screen": _chain_admet_screen,
    "protein_design": _chain_protein_design,
    "protein_binder_design": _chain_protein_binder_design,
    "ligand_binder_design": _chain_ligand_binder_design,
    "motif_scaffolding": _chain_motif_scaffolding,
}

# All chains execute (protein_design + the *_design chains touch ESMFold/Proteina;
# none needs a heavy dep in the core — SMILES->PDB stays in the UI caller).
RUNNABLE_CHAINS = set(_CHAINS)


# ─── batch-job adapters: dispatch a predefined job, wait, read its output back ──
# A batch node used to return only {job_run_id}, so a node wired downstream of it
# got the id, not the result. These adapters close that gap: each pre-creates a
# per-node child MLflow run (so artifacts don't collide across nodes/runs),
# dispatches the predefined job *into that run*, waits for completion, then reads
# the job's declared output back out — letting a batch job feed the next node on
# the canvas. The job itself is unchanged: it already logs into the run id it's
# handed (the same pre-created-run handoff the UI services use).

_ENZYME_DEFAULT_WEIGHTS = {
    "motif_rmsd": 1.0, "plddt": 1.3, "boltz": 0.5, "solubility": 1.0,
    "half_life": 2.6, "thermostab": 1.0, "immuno": 1.5,
}


def _job_id_by_name(w: WorkspaceClient, name: str) -> int:
    matches = list(w.jobs.list(name=name))
    if not matches:
        raise RuntimeError(f"Job '{name}' not found in this workspace.")
    return int(matches[0].job_id)


def _create_child_run(experiment_id, parent_run_id, node_id, run_name, user_email, feature):
    """A per-node MLflow run the dispatched job logs into — nested under the
    Vortex run and tagged so Past Runs can surface it."""
    from mlflow.tracking import MlflowClient
    r = MlflowClient().create_run(experiment_id=str(experiment_id), tags={
        "origin": "genesis_workbench", "feature": feature, "created_by": user_email or "",
        "mlflow.parentRunId": parent_run_id or "", "vortex_parent_run": parent_run_id or "",
        "vortex_node": node_id or "", "job_status": "submitted",
        "mlflow.runName": run_name or node_id or "node",
    })
    return r.info.run_id


def _download_pdb_dir(run_id: str, artifact_path: str) -> dict:
    """{name: pdb_string} for every *.pdb under an artifact dir; {} if absent."""
    from mlflow.tracking import MlflowClient
    out: dict = {}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            local = MlflowClient().download_artifacts(run_id, artifact_path, dst_path=tmp)
            for fn in sorted(os.listdir(local)):
                if fn.endswith(".pdb"):
                    with open(os.path.join(local, fn)) as f:
                        out[fn[:-4]] = f.read()
    except Exception as e:  # noqa: BLE001
        logger.info("artifacts %s not available for run %s: %s", artifact_path, run_id, e)
    return out


def _job_enzyme_optimization(w, inputs, params, ctx, progress=None) -> dict:
    """Guided Enzyme Optimization (reward-weighted GenMol loop). Mirrors the UI's
    start_enzyme_optimization_job dispatch, but waits + reads the top-K candidate
    PDBs back so a downstream node can consume them."""
    import json as _json
    import uuid as _uuid
    from mlflow.tracking import MlflowClient

    ins, p = inputs or {}, params or {}
    motif_pdb = ins.get("motif_pdb")
    if not motif_pdb:
        raise RuntimeError("enzyme_optimization: motif_pdb input required")
    substrate = str(ins.get("substrate_smiles") or p.get("substrate_smiles") or "")
    cat, sch = ctx["catalog"], ctx["schema"]

    _emit(progress, 5, "Uploading motif PDB to volume")
    motif_path = f"/Volumes/{cat}/{sch}/enzyme_optimization/{_uuid.uuid4().hex[:12]}/motif.pdb"
    w.files.upload(file_path=motif_path,
                   contents=io.BytesIO(str(motif_pdb).encode("utf-8")), overwrite=True)

    run_name = f"{ctx.get('run_name') or 'vortex'}::{ctx.get('node_id') or 'enzyme'}"
    child = _create_child_run(ctx["experiment_id"], ctx.get("parent_run_id"),
                              ctx.get("node_id"), run_name, ctx.get("user_email"),
                              "enzyme_optimization")

    num_samples = int(p.get("num_samples", 8) or 8)
    num_iter = int(p.get("num_iterations", 10) or 10)
    # Reward-axis weights: per-axis `weight_<axis>` params override the defaults.
    weights = {ax: float(p.get(f"weight_{ax}", dflt) if p.get(f"weight_{ax}") not in (None, "") else dflt)
               for ax, dflt in _ENZYME_DEFAULT_WEIGHTS.items()}
    accurate = _as_bool(p.get("use_inprocess_ame", False))
    job_name = "run_enzyme_optimization_gwb_inprocess_ame" if accurate else "run_enzyme_optimization_gwb"
    _emit(progress, 20, f"Dispatching enzyme optimization ({num_iter} iters × {num_samples} samples)")
    job_id = _job_id_by_name(w, job_name)
    run = w.jobs.run_now(job_id=job_id, job_parameters={
        "catalog": cat, "schema": sch, "cache_dir": "enzyme_optimization",
        "sql_warehouse_id": ctx.get("sql_warehouse", ""), "user_email": ctx.get("user_email", ""),
        "mlflow_experiment": ctx.get("experiment_name", ""), "mlflow_run_name": run_name,
        "mlflow_run_id": child, "motif_pdb_path": motif_path,
        "motif_residues_csv": str(p.get("motif_residues_csv", "") or ""),
        "target_chain": str(p.get("target_chain", "B") or "B"),
        "scaffold_length_min": str(int(p.get("scaffold_length_min", 80) or 80)),
        "scaffold_length_max": str(int(p.get("scaffold_length_max", 120) or 120)),
        "num_samples": str(num_samples), "num_iterations": str(num_iter),
        "substrate_smiles": substrate, "references_json": "[]",
        "half_life_margin": str(float(p.get("half_life_margin", 0.05) or 0.05)),
        "weights_json": _json.dumps(weights),
        "resampling_temperature": str(float(p.get("resampling_temperature", 0.1) or 0.1)),
        "strategy": str(p.get("strategy", "resample") or "resample"),
        "run_proteinmpnn": str(_as_bool(p.get("run_proteinmpnn", True))).lower(),
        "dev_user_prefix": ctx.get("dev_user_prefix", "") or "",
        "convergence_threshold": "0.01", "convergence_window": "2",
        "target_reward": "", "best_k_target": "", "best_k_threshold": "",
        "use_inprocess_ame": "true" if accurate else "false",
    })
    MlflowClient().set_tag(child, "job_run_id", str(run.run_id))

    _emit(progress, 30, "Enzyme optimization running — this can take a while")
    from .workbench import wait_for_job_run_completion
    wait_for_job_run_completion(int(run.run_id), timeout=21600, poll_interval=30)

    _emit(progress, 90, "Reading top candidates")
    candidates = _download_pdb_dir(child, "results/topK_pdbs")
    _emit(progress, 100, f"Enzyme optimization complete — {len(candidates)} candidate(s)")
    return {"candidates": candidates, "child_run_id": child, "job_run_id": str(run.run_id)}


# ─── shared dispatch + read-back helpers for the simpler adapters ─────────────


def _child_for(ctx, feature):
    """Per-node child MLflow run the dispatched job logs into."""
    return _create_child_run(
        ctx["experiment_id"], ctx.get("parent_run_id"), ctx.get("node_id"),
        f"{ctx.get('run_name') or 'vortex'}::{ctx.get('node_id') or feature}",
        ctx.get("user_email"), feature,
    )


def _dispatch_wait(w, job_name, job_params, child_run_id=None,
                   run_id_key="mlflow_run_id", progress=None) -> str:
    """Dispatch a predefined job (job_parameters string-coerced), tag the child
    run with the job_run_id, and block until the JOB RUN reaches a terminal
    state (so a job that crashes without updating its MLflow status still
    resolves). Returns the job_run_id."""
    jp = dict(job_params)
    if child_run_id and run_id_key:
        jp[run_id_key] = child_run_id
    jp = {k: ("" if v is None else str(v)) for k, v in jp.items()}
    run = w.jobs.run_now(job_id=_job_id_by_name(w, job_name), job_parameters=jp)
    if child_run_id:
        from mlflow.tracking import MlflowClient
        try:
            MlflowClient().set_tag(child_run_id, "job_run_id", str(run.run_id))
        except Exception as e:  # noqa: BLE001
            logger.info("could not tag job_run_id on %s: %s", child_run_id, e)
    _emit(progress, 35, "Job running — this can take a while")
    from .workbench import wait_for_job_run_completion
    wait_for_job_run_completion(int(run.run_id), timeout=21600, poll_interval=30)
    return str(run.run_id)


def _run_tag(run_id, key, default=""):
    from mlflow.tracking import MlflowClient
    try:
        return MlflowClient().get_run(run_id).data.tags.get(key, default)
    except Exception as e:  # noqa: BLE001
        logger.info("tag %s unavailable on run %s: %s", key, run_id, e)
        return default


def _run_param(run_id, key, default=""):
    from mlflow.tracking import MlflowClient
    try:
        return MlflowClient().get_run(run_id).data.params.get(key, default)
    except Exception as e:  # noqa: BLE001
        logger.info("param %s unavailable on run %s: %s", key, run_id, e)
        return default


def _run_artifact_json(run_id, artifact_path):
    from mlflow.tracking import MlflowClient
    with tempfile.TemporaryDirectory() as tmp:
        local = MlflowClient().download_artifacts(run_id, artifact_path, dst_path=tmp)
        with open(local) as f:
            return json.load(f)


# ─── the simpler job adapters (genomics / structure / opt / fine-tune) ────────
# Each job already logs into the mlflow run id it's handed (verified per job), so
# we pre-create a child run, dispatch into it, wait, and read the declared output
# (artifact / tag / param / volume / table) back out.


def _job_variant_calling(w, inputs, params, ctx, progress=None) -> dict:
    ins, p = inputs or {}, params or {}
    child = _child_for(ctx, "gwas_alignment")
    _emit(progress, 15, "Dispatching variant calling (Parabricks)")
    jr = _dispatch_wait(w, "gwas_parabricks_alignment", {
        "catalog": ctx["catalog"], "schema": ctx["schema"],
        "fastq_r1": ins.get("fastq_r1", ""), "fastq_r2": ins.get("fastq_r2", ""),
        "reference_genome_path": p.get("reference_genome_path", ""),
        "output_volume_path": p.get("output_volume_path", ""),
        "user_email": ctx.get("user_email", ""),
    }, child_run_id=child, progress=progress)
    _emit(progress, 100, "Variant calling complete")
    return {"vcf": _run_param(child, "output_vcf"), "child_run_id": child, "job_run_id": jr}


def _job_vcf_ingestion(w, inputs, params, ctx, progress=None) -> dict:
    ins, p = inputs or {}, params or {}
    child = _child_for(ctx, "vcf_ingestion")
    _emit(progress, 15, "Dispatching VCF ingestion")
    jr = _dispatch_wait(w, "vcf_ingestion_glow", {
        "catalog": ctx["catalog"], "schema": ctx["schema"],
        "sql_warehouse_id": ctx.get("sql_warehouse", ""),
        "vcf_path": ins.get("vcf", ""), "output_table_name": p.get("output_table_name", ""),
        "user_email": ctx.get("user_email", ""),
    }, child_run_id=child, progress=progress)
    _emit(progress, 100, "VCF ingestion complete")
    return {"table": _run_tag(child, "output_table"), "child_run_id": child, "job_run_id": jr}


def _job_variant_annotation(w, inputs, params, ctx, progress=None) -> dict:
    ins, p = inputs or {}, params or {}
    child = _child_for(ctx, "variant_annotation")
    _emit(progress, 15, "Dispatching variant annotation (ClinVar)")
    jr = _dispatch_wait(w, "variant_annotation_clinical", {
        "catalog": ctx["catalog"], "schema": ctx["schema"],
        "sql_warehouse_id": ctx.get("sql_warehouse", ""),
        "variants_table": ins.get("table", ""), "gene_regions": p.get("gene_regions", ""),
        "gene_panel_mode": p.get("gene_panel_mode", "custom"),
        "pathogenic_vcf_path": p.get("pathogenic_vcf_path", ""),
        "run_name": ctx.get("run_name", ""), "user_email": ctx.get("user_email", ""),
    }, child_run_id=child, progress=progress)
    _emit(progress, 100, "Variant annotation complete")
    return {"annotations": _run_tag(child, "pathogenic_table"), "child_run_id": child, "job_run_id": jr}


def _job_gwas(w, inputs, params, ctx, progress=None) -> dict:
    ins, p = inputs or {}, params or {}
    child = _child_for(ctx, "gwas")
    _emit(progress, 15, "Dispatching GWAS")
    jr = _dispatch_wait(w, "gwas_glow_analysis", {
        "catalog": ctx["catalog"], "schema": ctx["schema"],
        "vcf_path": ins.get("vcf", ""), "phenotype_path": ins.get("phenotype", ""),
        "phenotype_column": p.get("phenotype_column", "phenotype"),
        "contigs": p.get("contigs", ""), "hwe_cutoff": p.get("hwe_cutoff", "1e-6"),
        "pvalue_threshold": p.get("pvalue_threshold", "5e-8"),
        "user_email": ctx.get("user_email", ""),
    }, child_run_id=child, progress=progress)
    # Results land in a per-run Delta table keyed off the MLflow run id.
    table = f"{ctx['catalog']}.{ctx['schema']}.gwas_results_{child.replace('-', '_')}"
    _emit(progress, 100, "GWAS complete")
    return {"results": table, "child_run_id": child, "job_run_id": jr}


def _job_alphafold(w, inputs, params, ctx, progress=None) -> dict:
    ins = inputs or {}
    seq = str(ins.get("sequence", "")).strip()
    if not seq:
        raise RuntimeError("alphafold2: sequence input required")
    child = _child_for(ctx, "alphafold")
    _emit(progress, 15, "Dispatching AlphaFold")
    # NB: this job reads the run id under "run_id" (not "mlflow_run_id").
    jr = _dispatch_wait(w, "run_alphafold", {
        "catalog": ctx["catalog"], "schema": ctx["schema"],
        "protein_sequence": seq, "user_email": ctx.get("user_email", ""),
    }, child_run_id=child, run_id_key="run_id", progress=progress)
    _emit(progress, 90, "Reading folded structure")
    cat, sch = ctx["catalog"], ctx["schema"]
    # The run_id appears twice: OUTDIR results/{run_id} + AlphaFold's per-fasta subdir.
    pdb_path = f"/Volumes/{cat}/{sch}/alphafold/results/{child}/{child}/ranked_0.pdb"
    pdb = ""
    try:
        pdb = _read_volume_text(w, pdb_path)
    except Exception as e:  # noqa: BLE001
        logger.info("alphafold pdb not found at %s: %s", pdb_path, e)
    _emit(progress, 100, "AlphaFold complete")
    return {"pdb": pdb, "pdb_path": pdb_path, "child_run_id": child, "job_run_id": jr}


def _job_molecule_optimization(w, inputs, params, ctx, progress=None) -> dict:
    ins, p = inputs or {}, params or {}
    seeds = [str(s) for s in _as_list(ins.get("seed_smiles")) if str(s).strip()]
    if not seeds:
        raise RuntimeError("molecule_optimization: seed_smiles input required")
    child = _child_for(ctx, "molecule_optimization")
    _emit(progress, 15, "Dispatching molecule optimization")
    jr = _dispatch_wait(w, "run_molecule_optimization_gwb", {
        "catalog": ctx["catalog"], "schema": ctx["schema"],
        "sql_warehouse_id": ctx.get("sql_warehouse", ""), "user_email": ctx.get("user_email", ""),
        "mlflow_experiment": ctx.get("experiment_name", ""), "mlflow_run_name": ctx.get("run_name", ""),
        "seed_smiles_csv": ",".join(seeds),
        "num_samples": int(p.get("num_samples", 24) or 24),
        "num_iterations": int(p.get("num_iterations", 5) or 5),
        "select_top": int(p.get("select_top", 3) or 3),
        "dock_top_k": int(p.get("dock_top_k", 5) or 5),
        "weights_json": json.dumps({"qed_min": float(p.get("qed_min", 0.5) or 0.5),
                                    "tox_max": float(p.get("tox_max", 0.3) or 0.3)}),
        "temperature": float(p.get("temperature", 1.2) or 1.2),
        "randomness": float(p.get("randomness", 2.0) or 2.0),
        "target_sequence": str(ins.get("target_sequence", "") or ""),
        "dock_per_iter": int(p.get("dock_per_iter", 8) or 8),
        "dock_samples": int(p.get("dock_samples", 3) or 3),
    }, child_run_id=child, progress=progress)
    _emit(progress, 90, "Reading top candidates")
    top = {}
    try:
        data = _run_artifact_json(child, "top_k.json")
        top = data.get("top_k", data) if isinstance(data, dict) else data
    except Exception as e:  # noqa: BLE001
        logger.info("molecule_optimization top_k.json unavailable: %s", e)
    _emit(progress, 100, "Molecule optimization complete")
    return {"top_k": top, "child_run_id": child, "job_run_id": jr}


def _job_esm2_finetune(w, inputs, params, ctx, progress=None) -> dict:
    ins, p = inputs or {}, params or {}
    child = _child_for(ctx, "bionemo_esm_finetune")
    _emit(progress, 15, "Dispatching ESM2 fine-tune")
    jr = _dispatch_wait(w, "bionemo_esm_finetune_job", {
        "user_email": ctx.get("user_email", ""), "esm_variant": p.get("esm_variant", "650M"),
        "train_data_location": ins.get("train_data", ""),
        "validation_data_location": ins.get("evaluation_data", ""),
        "should_use_lora": "true" if _as_bool(p.get("should_use_lora", False)) else "false",
        "finetune_label": p.get("finetune_label", ""), "task_type": p.get("task_type", "regression"),
        "mlp_ft_dropout": float(p.get("mlp_ft_dropout", 0.25) or 0.25),
        "mlp_hidden_size": int(p.get("mlp_hidden_size", 256) or 256),
        "mlp_target_size": int(p.get("mlp_target_size", 1) or 1),
        "experiment_name": ctx.get("experiment_name", ""),
        "num_steps": int(p.get("num_steps", 50) or 50),
        "lr": float(p.get("mlp_lr", 5e-3) or 5e-3),
        "lr_multiplier": float(p.get("mlp_lr_multiplier", 1e2) or 1e2),
        "micro_batch_size": int(p.get("micro_batch_size", 2) or 2),
        "precision": p.get("precision", "bf16-mixed"),
    }, child_run_id=child, progress=progress)
    _emit(progress, 100, "ESM2 fine-tune complete")
    return {"weights": _run_tag(child, "result_location"), "child_run_id": child, "job_run_id": jr}


def _job_kermt_finetune(w, inputs, params, ctx, progress=None) -> dict:
    ins, p = inputs or {}, params or {}
    child = _child_for(ctx, "kermt_finetune")
    _emit(progress, 15, "Dispatching KERMT fine-tune")
    jr = _dispatch_wait(w, "kermt_finetune_job", {
        "catalog": ctx["catalog"], "schema": ctx["schema"], "user_email": ctx.get("user_email", ""),
        "train_data_location": ins.get("train_data", ""),
        "validation_data_location": ins.get("validation_data", ""),
        "test_data_location": ins.get("test_data", ""),
        "target_names": p.get("target_names", "toxicity"),
        "dataset_type": p.get("dataset_type", "classification"),
        "finetune_label": p.get("finetune_label", ""),
        "epochs": int(p.get("epochs", 20) or 20), "batch_size": int(p.get("batch_size", 16) or 16),
        "ffn_hidden_size": int(p.get("ffn_hidden_size", 700) or 700),
        "experiment_name": ctx.get("experiment_name", ""), "mlflow_run_name": ctx.get("run_name", ""),
    }, child_run_id=child, progress=progress)
    _emit(progress, 100, "KERMT fine-tune complete")
    return {"ft_id": _run_tag(child, "ft_id"), "child_run_id": child, "job_run_id": jr}


def _job_kermt_deploy(w, inputs, params, ctx, progress=None) -> dict:
    """One-off deploy action — no MLflow run handoff; just dispatch + wait on the
    job-run state, then return the model/endpoint name."""
    ins, p = inputs or {}, params or {}
    ft_id = ins.get("ft_id")
    if ft_id in (None, ""):
        raise RuntimeError("kermt_deploy: ft_id input required (from a KERMT fine-tune)")
    model_name = str(p.get("model_name", "kermt_admet") or "kermt_admet")
    jp = {"catalog": ctx["catalog"], "schema": ctx["schema"],
          "user_email": ctx.get("user_email", ""),
          "ft_id": ft_id if not isinstance(ft_id, dict) else ft_id.get("ft_id", ""),
          "model_name": model_name}
    if p.get("workload_type"):
        jp["workload_type"] = p.get("workload_type")
    _emit(progress, 15, "Dispatching KERMT deploy")
    jr = _dispatch_wait(w, "kermt_deploy_job", jp, child_run_id=None, run_id_key=None, progress=progress)
    _emit(progress, 100, "KERMT endpoint deployed")
    return {"endpoint": model_name, "job_run_id": jr}


_JOB_RUNNERS = {
    "run_enzyme_optimization_gwb": _job_enzyme_optimization,
    "enzyme_optimization": _job_enzyme_optimization,
    "gwas_parabricks_alignment": _job_variant_calling,
    "variant_calling": _job_variant_calling,
    "vcf_ingestion_glow": _job_vcf_ingestion,
    "vcf_ingestion": _job_vcf_ingestion,
    "variant_annotation_clinical": _job_variant_annotation,
    "variant_annotation": _job_variant_annotation,
    "gwas_glow_analysis": _job_gwas,
    "gwas": _job_gwas,
    "run_alphafold": _job_alphafold,
    "alphafold2": _job_alphafold,
    "run_molecule_optimization_gwb": _job_molecule_optimization,
    "molecule_optimization": _job_molecule_optimization,
    "bionemo_esm_finetune_job": _job_esm2_finetune,
    "esm2_finetune": _job_esm2_finetune,
    "kermt_finetune_job": _job_kermt_finetune,
    "kermt_finetune": _job_kermt_finetune,
    "kermt_deploy_job": _job_kermt_deploy,
    "kermt_deploy": _job_kermt_deploy,
}
# Jobs with an output-collecting adapter (vs. plain trigger-and-wait). A node
# wired downstream of one of these receives the job's real output.
RUNNABLE_JOBS = set(_JOB_RUNNERS)


def run_workflow_job(job_name, inputs=None, params=None, workspace_client=None,
                     *, ctx=None, progress=None) -> dict:
    """Run a predefined batch job via its adapter (dispatch → wait → read output).
    Raises if `job_name` has no adapter — callers should gate on RUNNABLE_JOBS and
    fall back to plain trigger-and-wait. `ctx` carries the cluster context the
    adapter needs (catalog/schema/sql_warehouse/user_email/experiment_id/
    experiment_name/parent_run_id/node_id/run_name/dev_user_prefix)."""
    fn = _JOB_RUNNERS.get(job_name or "")
    if fn is None:
        raise RuntimeError(f"No output-collecting adapter for job '{job_name}'.")
    return fn(_w(workspace_client), inputs or {}, params or {}, ctx or {}, progress)


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


def _run_transform(w: WorkspaceClient, op: str, inputs: dict, params: dict):
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
    if op == "smiles_to_pdb":
        # RDKit ETKDGv3 -> MMFF94, same as the UI's ligand_binder_design.smiles_to_pdb.
        # Lazy import keeps the core dep-free; rdkit is installed on the orchestrator
        # cluster (and present in the UI app). Mirrors the BioPython lazy import.
        from rdkit import Chem
        from rdkit.Chem import AllChem
        smiles = str(inputs.get("smiles") or params.get("smiles") or "").strip()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise RuntimeError(f"Invalid SMILES: {smiles!r}")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)
        return {"pdb": Chem.MolToPDBBlock(mol)}
    raise RuntimeError(f"Unknown transform op '{op}'")


# ─── dispatch ────────────────────────────────────────────────────────────────


def execute_capability(
    cap: Capability,
    inputs: dict | None = None,
    params: dict | None = None,
    workspace_client: WorkspaceClient | None = None,
    progress=None,
):
    """Run a capability. `inputs` maps input-port name -> value; `params` maps
    param name -> value. Endpoints/chains/transforms return results synchronously;
    jobs return {run_id, run_url} (poll with `run_status`). `progress(pct, msg)`,
    if given, fires at each chain stage — the caller decides what to do with it
    (SSE, per-node status, MLflow), keeping this core presentation-agnostic."""
    w = _w(workspace_client)
    if cap.kind == ENDPOINT:
        return _query_endpoint(w, cap, inputs or {}, params or {})
    if cap.kind == JOB:
        return _dispatch_job(w, cap, params or {})
    if cap.kind == CHAIN:
        fn = _CHAINS.get(cap.chain_id or "")
        if fn is None:
            raise RuntimeError(f"No executor for chain '{cap.chain_id}'.")
        return fn(w, inputs or {}, params or {}, progress)
    if cap.kind == TRANSFORM:
        return _run_transform(w, cap.op, inputs or {}, params or {})
    raise RuntimeError(f"Unknown capability kind '{cap.kind}'.")


# ─── direct entry points for the Vortex orchestrator (by id, no Capability) ──


def run_chain(chain_id: str, inputs: dict | None = None, params: dict | None = None,
              workspace_client: WorkspaceClient | None = None, progress=None):
    """Run an endpoint-chain by id. Used by the canvas orchestrator + UI services,
    which have the node's exec descriptor (not a full Capability). `progress(pct,
    msg)` fires at each stage if given."""
    fn = _CHAINS.get(chain_id or "")
    if fn is None:
        raise RuntimeError(f"No executor for chain '{chain_id}'.")
    return fn(_w(workspace_client), inputs or {}, params or {}, progress)


def run_transform(op: str, inputs: dict | None = None, params: dict | None = None,
                  workspace_client: WorkspaceClient | None = None) -> dict:
    """Run a transform op by name. Returns {output_port: value}."""
    return _run_transform(_w(workspace_client), op, inputs or {}, params or {})
