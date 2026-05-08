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
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from utils.streamlit_helper import (
    get_user_info,
    open_mlflow_experiment_window,
    open_run_window,
)
from utils.small_molecule_tools import molstar_html_multi_pdb
from utils.enzyme_optimization_tools import (
    DEFAULT_AXIS_WEIGHTS,
    T4_LYSOZYME_SEQUENCE,
    get_run_status,
    load_optimization_trajectory,
    load_top_k_pdbs,
    predict_enzyme_properties,
    search_enzyme_optimization_runs_by_experiment_name,
    search_enzyme_optimization_runs_by_run_name,
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
# Search-past-runs helpers (module level so the dialog decorator works)
# ---------------------------------------------------------------------------

# Stages at which the dialog has something useful to show. `started`,
# `warming_up`, `loading_ame`, and `iter_<N>_(generating|redesigning|scoring)`
# are excluded — at those stages no candidate has been logged yet so the
# trajectory + top-K artifacts don't exist.
def _is_viewable_status(status: str) -> bool:
    if not status:
        return False
    if status in ("complete", "failed"):
        return True
    return status.startswith("iter_") and status.endswith("_complete")


def _set_selected_enzyme_opt_run_status():
    selection = st.session_state["enzyme_opt_search_display_df"].selection
    if len(selection["rows"]) > 0:
        idx = selection["rows"][0]
        df = st.session_state["enzyme_opt_search_result_df"]
        st.session_state["selected_enzyme_opt_run_status"] = (
            df.iloc[idx].get("job_status", "") if "job_status" in df.columns else ""
        )
    else:
        st.session_state.pop("selected_enzyme_opt_run_status", None)


@st.dialog("Optimization Results", width="large")
def _display_enzyme_opt_result_dialog(selected_row_for_view):
    from utils.enzyme_optimization_tools import (
        get_run_status,
        load_optimization_trajectory,
        load_top_k_pdbs,
    )

    idx = selected_row_for_view[0]
    row = st.session_state["enzyme_opt_search_result_df"].iloc[idx]
    run_id = row["run_id"]
    run_name = row.get("run_name", run_id[:12])
    st.markdown(f"##### Run Name: {run_name}")

    with st.spinner("Loading run results..."):
        try:
            status = get_run_status(run_id)
        except Exception as e:
            st.error(f"Could not load run status: {e}")
            return

        st.markdown(
            f"**Status:** `{status.get('status', 'UNKNOWN')}` "
            f"• stage: `{row.get('job_status', 'unknown')}`"
        )

        if status.get("experiment_id"):
            if st.button("View in MLflow", key=f"enzyme_opt_dialog_mlflow_{run_id}"):
                open_mlflow_experiment_window(status["experiment_id"])

        _live_progress_chart(status)

        try:
            traj_df = load_optimization_trajectory(run_id)
        except Exception as e:
            st.warning(f"Trajectory not yet available: {e}")
            traj_df = pd.DataFrame()

        if not traj_df.empty:
            shown = [c for c in (
                "candidate_id", "iteration", "composite_reward",
                "motif_rmsd", "plddt", "boltz",
                "solubility", "half_life", "thermostab", "immuno",
                "designed_sequence",
            ) if c in traj_df.columns]
            with st.expander("Top candidates (sorted by composite reward)", expanded=False):
                st.dataframe(traj_df[shown].head(25), use_container_width=True, hide_index=True)

            try:
                top_pdbs = load_top_k_pdbs(run_id)
            except Exception:
                top_pdbs = {}
            if top_pdbs:
                cand_id = st.selectbox(
                    "Inspect candidate:", options=list(top_pdbs.keys()),
                    key=f"enzyme_opt_dialog_pdb_{run_id}",
                )

                # Metrics for the selected candidate, displayed above the
                # Mol* viewer. Pulled from the same trajectory dataframe.
                cand_rows = traj_df[traj_df["candidate_id"] == cand_id] \
                    if "candidate_id" in traj_df.columns else pd.DataFrame()
                if not cand_rows.empty:
                    cand_row = cand_rows.iloc[0]

                    def _fmt(val, fmt="{:.3f}"):
                        if val is None or pd.isna(val):
                            return "—"
                        try:
                            return fmt.format(float(val))
                        except (ValueError, TypeError):
                            return str(val)

                    metrics_row1 = [
                        ("Composite Reward", _fmt(cand_row.get("composite_reward"))),
                        ("Motif RMSD (Å)",   _fmt(cand_row.get("motif_rmsd"))),
                        ("pLDDT",            _fmt(cand_row.get("plddt"), "{:.1f}")),
                        ("Boltz",            _fmt(cand_row.get("boltz"))),
                    ]
                    metrics_row2 = [
                        ("Solubility",       _fmt(cand_row.get("solubility"))),
                        ("Half-Life",        _fmt(cand_row.get("half_life"))),
                        ("Thermostab (°C)",  _fmt(cand_row.get("thermostab"), "{:.1f}")),
                        ("Immunogenicity",   _fmt(cand_row.get("immuno"))),
                    ]
                    for row in (metrics_row1, metrics_row2):
                        cols = st.columns(len(row))
                        for col, (label, val) in zip(cols, row):
                            col.metric(label, val)

                html = molstar_html_multi_pdb([top_pdbs[cand_id]])
                components.html(html, height=480)
                st.download_button(
                    "Download candidate PDB",
                    data=top_pdbs[cand_id],
                    file_name=f"{cand_id}.pdb",
                    mime="chemical/x-pdb",
                    key=f"enzyme_opt_dialog_dl_{run_id}",
                )
        else:
            st.info("No completed candidates were logged for this run.")


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
                job_id, job_run_id = start_enzyme_optimization_job(
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

        st.success(f"Job started with run id: {job_run_id}.")
        st.button(
            "View Run",
            on_click=lambda: open_run_window(job_id, job_run_id),
            key="enzyme_opt_view_run_btn",
        )

    # ─── SEARCH PAST RUNS ────────────────────────────────────────────────────
    st.divider()
    st.markdown("###### Search Past Runs:")
    user_info = get_user_info()
    s_c1, s_c2, s_c3 = st.columns([1, 1, 1], vertical_alignment="bottom")
    with s_c1:
        search_mode = st.pills(
            "Search By:", ["Experiment Name", "Run Name"],
            selection_mode="single", default="Run Name",
            key="enzyme_opt_search_mode",
        )
    with s_c2:
        search_text = st.text_input(
            f"{search_mode} contains:", value="enzyme_opt",
            key="enzyme_opt_search_text",
        )
    with s_c3:
        search_btn = st.button("Search", key="enzyme_opt_search_btn")

    if search_btn:
        with st.spinner("Searching"):
            st.session_state.pop("enzyme_opt_search_result_df", None)
            st.session_state.pop("selected_enzyme_opt_run_status", None)

            if search_text.strip() != "":
                if search_mode == "Experiment Name":
                    df = search_enzyme_optimization_runs_by_experiment_name(
                        user_email=user_info.user_email,
                        experiment_name=search_text,
                    )
                else:
                    df = search_enzyme_optimization_runs_by_run_name(
                        user_email=user_info.user_email,
                        run_name=search_text,
                    )
                if not df.empty:
                    st.session_state["enzyme_opt_search_result_df"] = df
                else:
                    st.error("No results found")
            else:
                st.error("Provide a search text")

    if "enzyme_opt_search_result_df" in st.session_state:
        view_enabled = _is_viewable_status(
            st.session_state.get("selected_enzyme_opt_run_status", "")
        )
        v_c1, v_c2, v_c3 = st.columns([1, 1, 1], vertical_alignment="bottom")
        with v_c1:
            view_btn = st.button(
                "View", disabled=not view_enabled,
                key="enzyme_opt_search_view_btn",
                help=("Select a row first. View is enabled only for runs with at "
                      "least one completed iteration (iter_<N>_complete, complete, "
                      "or failed)."),
            )

        selected_event = st.dataframe(
            st.session_state["enzyme_opt_search_result_df"],
            column_config={
                "run_id": None,
                "run_name":              st.column_config.TextColumn("Run Name"),
                "experiment_name":       st.column_config.TextColumn("Experiment"),
                "generation_mode":       st.column_config.TextColumn("Mode"),
                "iter_max_reward":       st.column_config.NumberColumn(
                    "Max Reward", format="%.3f"),
                "iterations_completed":  st.column_config.NumberColumn(
                    "Iterations", format="%d",
                ),
                "start_time":            st.column_config.DatetimeColumn("Started"),
                "job_status":            st.column_config.TextColumn("Stage"),
                "progress":              st.column_config.TextColumn("Progress"),
            },
            use_container_width=True,
            hide_index=True,
            on_select=_set_selected_enzyme_opt_run_status,
            selection_mode="single-row",
            key="enzyme_opt_search_display_df",
        )

        selected_rows = selected_event.selection.rows
        if len(selected_rows) > 0 and view_btn:
            _display_enzyme_opt_result_dialog(selected_rows)
