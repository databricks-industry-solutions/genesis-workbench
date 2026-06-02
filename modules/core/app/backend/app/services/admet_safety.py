"""ADMET & Safety property prediction.

Three Chemprop D-MPNN endpoints — BBB Penetration, Clinical Toxicity, and a
multi-task ADMET regression head. Each takes an `inputs=[smiles, …]` list
payload and returns per-molecule scores. The three predictors are
independent — any subset can run.

Ported from modules/core/app/views/small_molecule_workflows/admet_safety.py."""
from __future__ import annotations

import logging
from typing import Callable

from databricks.sdk import WorkspaceClient

from app.services.endpoints import get_endpoint_name

logger = logging.getLogger(__name__)


def _query_chemprop(endpoint_name: str, smiles_list: list[str]):
    """Chemprop endpoints accept a flat `inputs=[smi, …]` list. Returns
    `response.predictions` as-is — BBB/ClinTox give floats, ADMET gives a
    list of dicts (one row per molecule, columns per task)."""
    w = WorkspaceClient()
    response = w.serving_endpoints.query(name=endpoint_name, inputs=smiles_list)
    return response.predictions


def predict_bbbp(smiles_list: list[str]) -> list[float | None]:
    raw = _query_chemprop(get_endpoint_name("Chemprop BBBP"), smiles_list)
    return [None if v is None else float(v) for v in (raw or [])]


def predict_clintox(smiles_list: list[str]) -> list[float | None]:
    raw = _query_chemprop(get_endpoint_name("Chemprop ClinTox"), smiles_list)
    return [None if v is None else float(v) for v in (raw or [])]


def predict_admet(smiles_list: list[str]) -> list[dict]:
    """Returns one dict per molecule keyed by ADMET task name. The endpoint
    response is already in that shape (list of dicts) on the v2 PyFunc."""
    raw = _query_chemprop(get_endpoint_name("Chemprop ADMET"), smiles_list)
    if not raw:
        return []
    # Endpoints occasionally tunnel through `{"predictions": [...]}`. Be
    # permissive about both shapes since callers across this repo see both.
    if isinstance(raw, dict) and "predictions" in raw:
        raw = raw["predictions"]
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return raw
    # Last resort: wrap whatever non-dict thing we got into a single-row
    # dict so the frontend still has something to render.
    return [{"prediction": v} for v in raw]


def run_admet_profiling(
    smiles_list: list[str],
    run_bbbp: bool,
    run_clintox: bool,
    run_admet: bool,
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict:
    """Run any subset of the three predictors. Returns `{bbbp?, clintox?,
    admet?, warnings}`. Per-endpoint failures degrade gracefully and end up
    as warnings in the response."""
    def _p(pct: int, msg: str) -> None:
        if progress_callback:
            progress_callback(pct, msg)

    enabled = [run_bbbp, run_clintox, run_admet]
    total_steps = sum(enabled) or 1
    step = 0
    warnings: list[str] = []
    out: dict = {"smiles": list(smiles_list)}

    def _step_pct() -> int:
        # Span 10 → 95% across however many predictors are enabled.
        return 10 + int(((step + 1) / total_steps) * 85)

    if run_bbbp:
        _p(10 + int(step / total_steps * 85), "Predicting blood-brain barrier penetration")
        try:
            out["bbbp"] = predict_bbbp(smiles_list)
        except Exception as e:
            logger.warning("Chemprop BBBP failed: %s", e)
            out["bbbp"] = [None] * len(smiles_list)
            warnings.append(f"BBB Penetration prediction failed: {e}")
        step += 1
        _p(_step_pct(), "BBB prediction complete")

    if run_clintox:
        _p(10 + int(step / total_steps * 85), "Predicting clinical-trial toxicity")
        try:
            out["clintox"] = predict_clintox(smiles_list)
        except Exception as e:
            logger.warning("Chemprop ClinTox failed: %s", e)
            out["clintox"] = [None] * len(smiles_list)
            warnings.append(f"Clinical Toxicity prediction failed: {e}")
        step += 1
        _p(_step_pct(), "Clinical toxicity complete")

    if run_admet:
        _p(10 + int(step / total_steps * 85), "Predicting ADMET properties (multi-task)")
        try:
            out["admet"] = predict_admet(smiles_list)
        except Exception as e:
            logger.warning("Chemprop ADMET failed: %s", e)
            out["admet"] = []
            warnings.append(f"ADMET property prediction failed: {e}")
        step += 1
        _p(_step_pct(), "ADMET multi-task complete")

    out["warnings"] = warnings
    return out
