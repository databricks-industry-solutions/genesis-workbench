"""Workflow 4: Guided Enzyme Optimization.

Wraps Proteina-Complexa-AME + ProteinMPNN + ESMFold in a reward-weighted
resampling loop, scoring each candidate on physical fidelity (motif RMSD,
pLDDT, optional Boltz substrate confidence) and developability axes
(solubility, half-life, thermostability, immunogenicity). The loop runs as
a Databricks job so the Streamlit page stays responsive.

Form → `start_enzyme_optimization_job` → MLflow run → polling results view.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from utils.streamlit_helper import get_user_info, open_mlflow_experiment_window
from utils.small_molecule_tools import molstar_html_multi_pdb
from utils.enzyme_optimization_tools import (
    DEFAULT_AXIS_WEIGHTS,
    T4_LYSOZYME_SEQUENCE,
    get_run_status,
    load_optimization_trajectory,
    load_top_k_pdbs,
    predict_enzyme_properties,
    start_enzyme_optimization_job,
)
from views.small_molecule_workflows.motif_scaffolding import EXAMPLE_MOTIF_PDB


WORKFLOW_DESCRIPTION = (
    "Generate enzyme scaffolds with Proteina-Complexa-AME, then iterate: ProteinMPNN "
    "redesign → ESMFold → score each candidate on physical fidelity *and* developability "
    "(solubility, anchor-relative half-life, melting temperature, immunogenic burden). "
    "Each iteration's composite reward biases the next round's sampling."
)

_AXIS_LABELS = [
    ("motif_rmsd", "Motif backbone RMSD",                "Lower is better — penalizes catalytic-site drift after redesign."),
    ("plddt",      "ESMFold pLDDT",                      "Higher is better — global fold confidence."),
    ("boltz",      "Boltz substrate complex confidence", "Higher is better — only contributes if substrate SMILES is supplied."),
    ("solubility", "NetSolP solubility",                 "Higher is better — predicted E. coli solubility prob in [0, 1]."),
    ("half_life",  "PLTNUM half-life (anchor-relative)", "Higher is better — sigmoid against your reference enzyme(s) + margin. Set to 0 to drop."),
    ("thermostab", "DeepSTABp Tm (°C)",                  "Higher is better — predicted melting temperature."),
    ("immuno",     "MHCflurry immunogenic burden",       "Lower is better — strong-presenter density across the default HLA panel."),
]


# ---------------------------------------------------------------------------
# Form helpers
# ---------------------------------------------------------------------------

# T4 lysozyme is a small (164 aa) well-characterized enzyme with a stable
# fold; in mammalian cell culture (NIH3T3, the cell line PLTNUM was trained
# on) reported half-life is ~24 h. We use it as a "stable" reference.
#
# The N-end-rule variant is the same protein with the second residue mutated
# (Met-Asn → Met-Arg). After initiator-Met cleavage, an N-terminal Arg makes
# the protein a substrate for the UBR family of E3 ubiquitin ligases, dropping
# half-life to ~30 min in mammalian cells (Bachmair, Finley, Varshavsky 1986).
# Provides a "destabilized" reference so the anchor threshold sits below the
# stable form, giving the loop room to clear it.
_T4_LYSOZYME_NEND_DESTABILIZED = (
    T4_LYSOZYME_SEQUENCE[0] + "R" + T4_LYSOZYME_SEQUENCE[2:]
)


def _default_references_df() -> pd.DataFrame:
    """Pre-filled with two complementary references for the example motif input:
    a stable enzyme (T4 lysozyme, ~24 h) and an N-end-rule-destabilized variant
    of the same protein (~30 min). The user can edit, replace, or add rows."""
    return pd.DataFrame([
        {
            "sequence": T4_LYSOZYME_SEQUENCE,
            "half_life_hours": 24.0,
            "cell_system": "NIH3T3",
        },
        {
            "sequence": _T4_LYSOZYME_NEND_DESTABILIZED,
            "half_life_hours": 0.5,
            "cell_system": "NIH3T3",
        },
    ])


def _references_from_editor(df: pd.DataFrame) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        seq = str(r.get("sequence", "")).strip()
        if not seq:
            continue
        out.append({
            "sequence": seq,
            "half_life_hours": float(r.get("half_life_hours") or 0.0),
            "cell_system": str(r.get("cell_system") or "HEK293"),
        })
    return out


def _parse_residues_csv(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _live_progress_chart(status: Dict[str, Any]):
    """Lightweight chart of the iter_max_reward / iter_mean_reward history."""
    rows = []
    for step, val in status.get("iter_max_reward_history", []):
        rows.append({"iteration": step, "metric": "max",  "reward": val})
    for step, val in status.get("iter_mean_reward_history", []):
        rows.append({"iteration": step, "metric": "mean", "reward": val})
    if not rows:
        return
    df = pd.DataFrame(rows).pivot(index="iteration", columns="metric", values="reward")
    st.line_chart(df, height=200, y_label="composite reward")


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render():
    st.markdown("### Guided Enzyme Optimization")
    st.markdown(WORKFLOW_DESCRIPTION)

    # ─── FORM (two columns: inputs left, guidance right) ─────────────────────
    with st.form("enzyme_opt_form", enter_to_submit=False):
        input_col, guidance_col = st.columns([1, 1])

        with input_col:
            st.markdown("**Motif input**")
            motif_pdb = st.text_area(
                "Motif + Ligand (PDB):", value=EXAMPLE_MOTIF_PDB, height=200,
                help="PDB containing the catalytic motif residues (ATOM) and optional ligand (HETATM)",
            )
            c1, c2 = st.columns(2)
            with c1:
                target_chain = st.text_input("Motif Chain:", value="B")
            with c2:
                motif_residues_csv = st.text_input(
                    "Motif residues (CSV):", value="1,2,3",
                    help="Residue numbers (within the motif chain) used for the post-redesign motif backbone RMSD.",
                )

            st.markdown("**Loop parameters**")
            generation_mode = st.radio(
                "Generation mode:",
                options=["Fast", "Accurate"],
                index=0,
                horizontal=True,
                help=(
                    "**Fast** (~30 min, no GPU cost) — AME generates K candidates per "
                    "iteration via the existing serving endpoint, the loop scores them, "
                    "then resamples parents by reward for the next iteration. "
                    "Reward signal applies *between* iterations only.\n\n"
                    "**Accurate** (~6 hours, ~$22 GPU cost) — AME loads on an A10 GPU "
                    "cluster and uses Feynman-Kac steering to bias diffusion toward "
                    "developability axes *during* sampling: at intermediate denoising "
                    "steps, partial structures are scored and trajectories are "
                    "importance-sampled so losing branches get pruned early. Better "
                    "top-K candidates per dollar in theory; not yet validated on this "
                    "specific reward stack."
                ),
            )
            use_inprocess_ame = (generation_mode == "Accurate")

            l1, l2 = st.columns(2)
            with l1:
                scaffold_len_min = st.number_input("Scaffold length min:", value=80, min_value=20, max_value=400)
                num_samples = st.number_input("K (candidates / iteration):", value=8, min_value=2, max_value=32)
            with l2:
                scaffold_len_max = st.number_input("Scaffold length max:", value=120, min_value=20, max_value=400)
                num_iterations = st.number_input("N (iterations):", value=10, min_value=1, max_value=30,
                                                 help="Iteration ceiling. The convergence stop usually exits earlier (see Stopping criteria).")

            run_proteinmpnn = st.checkbox("Redesign each scaffold with ProteinMPNN", value=True)
            substrate_smiles = st.text_input(
                "Substrate SMILES (optional, gates Boltz axis):", value="",
                help="If supplied, Boltz scores each candidate's complex with this substrate.",
            )

            st.markdown("**MLflow tracking**")
            mlflow_experiment = st.text_input(
                "MLflow Experiment:", value="gwb_enzyme_optimization",
                key="enzyme_opt_mlflow_exp",
            )
            mlflow_run_name = st.text_input(
                "Run Name:",
                value=f"enzyme_opt_{datetime.now().strftime('%Y%m%d_%H%M')}",
                key="enzyme_opt_mlflow_run",
            )

        with guidance_col:
            st.markdown("**Guidance parameters**")

            with st.expander("Per-axis reward weights", expanded=True):
                st.caption("Weight 0 disables an axis. Each axis is z-score-then-min-max normalized within the iteration's batch before weighted sum (except half-life which is pre-normalized via the anchor sigmoid).")
                weights: Dict[str, float] = {}
                for key, label, helptext in _AXIS_LABELS:
                    weights[key] = st.slider(
                        label, min_value=0.0, max_value=5.0,
                        value=float(DEFAULT_AXIS_WEIGHTS.get(key, 1.0)),
                        step=0.1, help=helptext, key=f"enzyme_opt_w_{key}",
                    )

            with st.expander("Half-life anchor (reference enzymes)", expanded=False):
                st.caption(
                    "Provide \u2265 1 known enzyme with a measured half-life. The loop scores "
                    "candidates' PLTNUM relative-stability against these references; "
                    "candidates above `min(reference) + margin` get positive reward."
                )
                refs_df = st.data_editor(
                    _default_references_df(),
                    num_rows="dynamic",
                    use_container_width=True,
                    key="enzyme_opt_refs",
                    column_config={
                        "sequence":         st.column_config.TextColumn("Reference sequence", width="large"),
                        "half_life_hours":  st.column_config.NumberColumn("Half-life (h)", min_value=0.0, max_value=10000.0),
                        "cell_system":      st.column_config.TextColumn("Cell system",  width="small"),
                    },
                )
                half_life_margin = st.slider("Anchor margin (\u03b2):", 0.01, 0.50, 0.05, 0.01)

            with st.expander("Advanced", expanded=False):
                strategy = st.radio(
                    "Strategy:", options=["resample", "noop"], index=0,
                    horizontal=True,
                    help=(
                        "`resample` \u2014 softmax-weighted parent resampling each iteration "
                        "(Phase 1 default). `noop` \u2014 verification mode: skips re-generation "
                        "after the first iteration so the strategy hook can be smoke-tested."
                    ),
                )
                resampling_temperature = st.slider(
                    "Resampling temperature:", 0.01, 1.00, 0.10, 0.01,
                    help="Lower = greedier toward high-reward parents.",
                )

            with st.expander("Stopping criteria", expanded=False):
                st.caption(
                    "N (iterations) is the hard ceiling. The loop exits early when any "
                    "of these conditions is met. Convergence is on by default; the other "
                    "two are opt-in."
                )

                conv_enabled = st.checkbox(
                    "Convergence stop", value=True, key="enzyme_opt_conv_enabled",
                    help="Exit when iter_max_reward improvement falls below the threshold "
                         "over a window of iterations.",
                )
                conv_c1, conv_c2 = st.columns(2)
                with conv_c1:
                    convergence_threshold_ui = st.number_input(
                        "Min improvement", min_value=0.0, max_value=1.0,
                        value=0.01, step=0.01, format="%.3f",
                        key="enzyme_opt_conv_threshold",
                        help="Required reward gain over the window.",
                    )
                with conv_c2:
                    convergence_window_ui = st.number_input(
                        "Window (iters)", min_value=1, max_value=10, value=2, step=1,
                        key="enzyme_opt_conv_window",
                    )

                target_enabled = st.checkbox(
                    "Reward-threshold stop", value=False, key="enzyme_opt_target_enabled",
                    help="Exit when any candidate's composite reward reaches the target.",
                )
                target_reward_ui = st.number_input(
                    "Target composite reward", min_value=0.0, max_value=1.0,
                    value=0.90, step=0.05, key="enzyme_opt_target_value",
                )

                bestk_enabled = st.checkbox(
                    "Best-K cap stop", value=False, key="enzyme_opt_bestk_enabled",
                    help="Exit when this many candidates have reached the threshold across all iterations.",
                )
                bestk_c1, bestk_c2 = st.columns(2)
                with bestk_c1:
                    best_k_target_ui = st.number_input(
                        "K above threshold", min_value=1, max_value=200, value=10, step=1,
                        key="enzyme_opt_bestk_target",
                    )
                with bestk_c2:
                    best_k_threshold_ui = st.number_input(
                        "Threshold", min_value=0.0, max_value=1.0,
                        value=0.80, step=0.05, key="enzyme_opt_bestk_threshold",
                    )

        run_btn = st.form_submit_button("Launch optimization job", type="primary", use_container_width=True)

    # ─── Smoke test (outside form, full width) ───────────────────────────────
    with st.expander("Test predictors on T4 lysozyme", expanded=False):
        st.caption("One round-trip to each developability endpoint to confirm they're healthy.")
        if st.button("Run smoke test", key="enzyme_opt_smoke_test"):
            with st.spinner("Calling NetSolP, PLTNUM, DeepSTABp, MHCflurry..."):
                try:
                    scores = predict_enzyme_properties(T4_LYSOZYME_SEQUENCE)
                    st.json(scores)
                except Exception as e:
                    st.error(f"Smoke test failed: {e}")

    # ─── DISPATCH ────────────────────────────────────────────────────────────
    if run_btn:
        if not motif_pdb.strip():
            st.error("Motif PDB is required.")
            return
        try:
            motif_residues = _parse_residues_csv(motif_residues_csv)
        except ValueError:
            st.error(f"Could not parse motif residue list '{motif_residues_csv}' as comma-separated ints.")
            return
        if scaffold_len_max < scaffold_len_min:
            st.error("Scaffold length max must be >= min.")
            return
        if not mlflow_run_name.strip():
            st.error("MLflow Run Name is required.")
            return

        references = _references_from_editor(refs_df)
        if weights.get("half_life", 0) > 0 and not references:
            st.warning("Half-life axis is enabled but no reference enzyme was provided \u2014 half-life will fall back to a neutral 0.5 contribution. Add a reference for a real signal.")

        # Stopping criteria \u2014 convert UI controls into the dispatcher's kwargs.
        # Convergence is on-by-default; a disabled checkbox sends a negative
        # threshold so the orchestrator's check `convergence_threshold >= 0` skips it.
        convergence_threshold_kw = float(convergence_threshold_ui) if conv_enabled else -1.0
        convergence_window_kw = int(convergence_window_ui)
        target_reward_kw = float(target_reward_ui) if target_enabled else None
        best_k_target_kw = int(best_k_target_ui) if bestk_enabled else None
        best_k_threshold_kw = float(best_k_threshold_ui) if bestk_enabled else None

        user_info = get_user_info()
        with st.spinner("Dispatching enzyme optimization job..."):
            try:
                job_id, run_id = start_enzyme_optimization_job(
                    motif_pdb_str=motif_pdb,
                    motif_residues=motif_residues,
                    target_chain=target_chain,
                    scaffold_length_min=int(scaffold_len_min),
                    scaffold_length_max=int(scaffold_len_max),
                    num_samples=int(num_samples),
                    num_iterations=int(num_iterations),
                    weights=weights,
                    user_info=user_info,
                    mlflow_experiment=mlflow_experiment,
                    mlflow_run_name=mlflow_run_name,
                    substrate_smiles=substrate_smiles,
                    references=references,
                    half_life_margin=float(half_life_margin),
                    resampling_temperature=float(resampling_temperature),
                    strategy=strategy,
                    run_proteinmpnn=run_proteinmpnn,
                    convergence_threshold=convergence_threshold_kw,
                    convergence_window=convergence_window_kw,
                    target_reward=target_reward_kw,
                    best_k_target=best_k_target_kw,
                    best_k_threshold=best_k_threshold_kw,
                    use_inprocess_ame=use_inprocess_ame,
                )
            except Exception as e:
                st.error(f"Failed to dispatch job: {e}")
                return

        st.session_state["enzyme_opt_active_run"] = {
            "job_id": job_id,
            "run_id": run_id,
            "motif_pdb": motif_pdb,
        }
        st.rerun()

    # ─── RESULTS / POLLING VIEW (full width, below the form) ─────────────────
    st.divider()
    st.markdown("### Results")

    active = st.session_state.get("enzyme_opt_active_run")
    if not active:
        st.info("Submit the form above to launch a guided enzyme optimization job. Results will appear here.")
        return

    run_id = active["run_id"]
    job_id = active["job_id"]

    try:
        status = get_run_status(run_id)
    except Exception as e:
        st.warning(f"Run status not yet available (the orchestrator may still be initializing): {e}")
        status = {"status": "PENDING", "iter_max_reward_history": [],
                  "iter_mean_reward_history": [], "experiment_id": None}

    c_status, c_btn = st.columns([3, 1])
    with c_status:
        st.markdown(f"**Job {job_id}** \u2022 Run **{run_id[:12]}\u2026** \u2022 Status: `{status.get('status', 'UNKNOWN')}`")
    with c_btn:
        if status.get("experiment_id"):
            if st.button("View in MLflow", key="enzyme_opt_mlflow_btn"):
                open_mlflow_experiment_window(status["experiment_id"])

    _live_progress_chart(status)

    traj_df = load_optimization_trajectory(run_id)
    if not traj_df.empty:
        st.markdown("**Top candidates (sorted by composite reward)**")
        shown_cols = [c for c in (
            "candidate_id", "iteration", "composite_reward",
            "motif_rmsd", "plddt", "boltz",
            "solubility", "half_life", "thermostab", "immuno",
            "designed_sequence",
        ) if c in traj_df.columns]
        st.dataframe(
            traj_df[shown_cols].head(25),
            use_container_width=True, hide_index=True,
        )

        top_pdbs = load_top_k_pdbs(run_id)
        if top_pdbs:
            cand_id = st.selectbox(
                "Inspect candidate:", options=list(top_pdbs.keys()),
                key="enzyme_opt_pdb_selector",
            )
            html = molstar_html_multi_pdb([active["motif_pdb"], top_pdbs[cand_id]])
            components.html(html, height=520)
            st.caption(f"Showing input motif + designed scaffold {cand_id}")

            st.download_button(
                "Download candidate PDB",
                data=top_pdbs[cand_id],
                file_name=f"{cand_id}.pdb",
                mime="chemical/x-pdb",
            )

    if status.get("status") in (None, "PENDING", "RUNNING", "SCHEDULED", "UNKNOWN"):
        time.sleep(8)
        st.rerun()
