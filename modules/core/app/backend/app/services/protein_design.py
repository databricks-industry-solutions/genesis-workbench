"""Protein design pipeline ported from
modules/core/app/utils/protein_design.py.

ESMFold -> mask region -> RFDiffusion -> ProteinMPNN -> ESMFold each design,
logging artefacts to MLflow along the way. Synchronous; expect ~30s+ per
design when endpoints are warm, and a hard 504 from the Apps proxy if any
of the four endpoints cold-starts (tracked follow-up: move to async-job
pattern alongside Boltz)."""
from __future__ import annotations

import json
import logging
import os
import tempfile

import mlflow
from Bio import PDB
from Bio.PDB import PDBParser
from databricks.sdk import WorkspaceClient
from genesis_workbench.models import set_mlflow_experiment
from genesis_workbench.workbench import UserInfo

from app.services.protein import hit_esmfold, hit_proteinmpnn, hit_rfdiffusion
from app.services.structure_utils import select_and_align

logger = logging.getLogger(__name__)


def parse_sequence(sequence: str) -> dict:
    start_idx = sequence.find("[")
    end_idx = sequence.find("]")
    raw = sequence.replace("[", "").replace("]", "")
    return {"sequence": raw, "start_idx": start_idx, "end_idx": end_idx}


def extract_chain_reindex(structure, chain_id: str = "A") -> str:
    chain = structure[0][chain_id]

    new_structure = PDB.Structure.Structure("new_structure")
    new_model = PDB.Model.Model(0)
    new_chain = PDB.Chain.Chain(chain_id)

    for i, residue in enumerate(chain, start=1):
        if residue.id[0] == " ":
            residue.id = (" ", i, " ")
            new_chain.add(residue)

    new_model.add(new_chain)
    new_structure.add(new_model)

    io = PDB.PDBIO()
    io.set_structure(new_structure)
    with tempfile.NamedTemporaryFile(suffix=".pdb") as f:
        io.save(f.name)
        with open(f.name, "r") as fh:
            return fh.read()


def align_designed_pdbs(designed_pdbs: dict) -> list[str]:
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(len(designed_pdbs["designed"])):
            with open(os.path.join(tmpdir, f"d_{i}_structure.pdb"), "w") as f:
                f.write(designed_pdbs["designed"][i])
        with open(os.path.join(tmpdir, "init_structure.pdb"), "w") as f:
            f.write(designed_pdbs["initial"])

        init_structure = PDBParser().get_structure(
            "esmfold_initial", os.path.join(tmpdir, "init_structure.pdb")
        )
        unaligned_structures = [
            PDBParser().get_structure("designed", os.path.join(tmpdir, f"d_{i}_structure.pdb"))
            for i in range(len(designed_pdbs["designed"]))
        ]

    aligned: list[str] = []
    for i, ua in enumerate(unaligned_structures):
        init_str, designed_str = select_and_align(init_structure, ua)
        if i == 0:
            aligned.append(init_str)
        aligned.append(designed_str)
    return aligned


def make_designs(
    sequence: str,
    mlflow_experiment_name: str,
    mlflow_run_name: str,
    user_info: UserInfo,
    n_rfdiffusion_hits: int = 1,
    progress_callback=None,
) -> dict:
    """`progress_callback(pct, msg)` fires at every phase boundary. Stage
    budget (matches the frontend's `RealtimeProgress` stages):
        0 → 10  ESMFold initial
       10 → 50  RFDiffusion x N  (per-scaffold tick)
       50 → 70  ProteinMPNN x N  (per-scaffold tick)
       70 → 95  ESMFold each design  (per-design tick)
       95 →100  Aligning designs"""
    def _p(pct: int, msg: str) -> None:
        if progress_callback:
            progress_callback(pct, msg)

    w = WorkspaceClient(http_timeout_seconds=600)  # app SP; long timeout for heavy serving calls

    _p(2, "Setting up MLflow experiment")
    experiment = set_mlflow_experiment(
        experiment_tag=mlflow_experiment_name,
        user_email=user_info.user_email,
        host=None,
        token=None,
    )

    with mlflow.start_run(
        run_name=mlflow_run_name, experiment_id=experiment.experiment_id
    ) as run:
        mlflow_run_id = run.info.run_id
        mlflow.log_param("sequence", sequence)
        mlflow.log_param("n_rfdiffusion_hits", n_rfdiffusion_hits)

        # Delegate the actual ESMFold -> reindex -> RFDiffusion -> ProteinMPNN ->
        # ESMFold chain to the shared executor (same code Vortex/MCP run). The UI
        # keeps the presentation: MLflow run + artifacts, SSE progress via `_p`.
        from genesis_workbench.executor import run_chain

        result = run_chain(
            "protein_design",
            {"sequence": sequence},
            {"n_rfdiffusion_hits": n_rfdiffusion_hits},
            w,
            progress=_p,
        )
        mlflow.log_dict({"predictions": result["initial"]}, "esmfold_initial_predictions.json")
        mlflow.log_dict({"protein_mpnn_seqs": result["sequences"]}, "protein_mpnn_seqs.json")
        mlflow.log_dict({"all_pdb_results": result["designs"]}, "all_pdb_results.json")

        return {
            "initial": result["initial"],
            "designed": result["designs"],
            "experiment_id": experiment.experiment_id,
            "run_id": mlflow_run_id,
        }
