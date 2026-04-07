"""Workflow 1: Protein Binder Design using Proteina-Complexa + ESMFold validation."""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import mlflow
from genesis_workbench.models import set_mlflow_experiment
from utils.streamlit_helper import get_user_info, open_mlflow_experiment_window
from utils.small_molecule_tools import (
    hit_proteina_complexa, hit_esmfold, sequence_to_pdb,
    molstar_html_pdb, molstar_html_multi_pdb,
    EXAMPLE_PDB,
)

WORKFLOW_DESCRIPTION = (
    "Design novel protein binders for a target protein using "
    "[Proteina-Complexa](https://github.com/NVIDIA-Digital-Bio/Proteina-Complexa). "
    "Optionally validate that designed binders fold correctly using ESMFold."
)


def render():
    st.markdown("### Protein Binder Design")
    st.markdown(WORKFLOW_DESCRIPTION)

    input_col, viewer_col = st.columns([1, 2])

    with input_col:
        input_mode = st.radio("Input type:", ["Protein Sequence", "PDB"], horizontal=True, key="binder_input_mode")

        with st.form("binder_design_form", enter_to_submit=False):
            if input_mode == "PDB":
                target_pdb = st.text_area("Target Protein (PDB):", value=EXAMPLE_PDB, height=250,
                                          help="PDB content of the target protein you want to design a binder for")
                target_sequence = None
            else:
                target_sequence = st.text_area("Protein Sequence:",
                                               value="MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDAATKTFTVTE",
                                               height=100,
                                               help="Amino acid sequence — will be folded using ESMFold to generate the PDB structure")
                target_pdb = None

            target_chain = st.text_input("Target Chain:", value="A",
                                         help="Chain ID in the target PDB to bind")
            hotspot_residues = st.text_input("Hotspot Residues (optional):", value="",
                                             help="Comma-separated residue indices to target (e.g., 10,20,30)")

            c1, c2 = st.columns(2)
            with c1:
                binder_len_min = st.number_input("Min binder length:", value=50, min_value=20, max_value=200)
            with c2:
                binder_len_max = st.number_input("Max binder length:", value=80, min_value=20, max_value=300)

            num_samples = st.slider("Number of designs:", min_value=1, max_value=10, value=2)
            validate_esmfold = st.checkbox("Validate with ESMFold", value=True,
                                           help="Run ESMFold on designed sequences to verify they fold correctly")

            st.markdown("**MLflow Tracking:**")
            mlflow_experiment = st.text_input("MLflow Experiment:", value="gwb_binder_design", key="binder_mlflow_exp")
            mlflow_run_name = st.text_input("Run Name:", value="binder_design_run", key="binder_mlflow_run")
            run_btn = st.form_submit_button("Design Binders", type="primary")

    with viewer_col:
        status_container = st.container()

        if "binder_results" in st.session_state and st.session_state["binder_results"] is not None:
            results_df = st.session_state["binder_results"]

            for w in st.session_state.get("binder_warnings", []):
                st.warning(w)

            selected_idx = st.selectbox(
                "Select design:",
                options=results_df.index,
                key="binder_design_selector",
                format_func=lambda i: f"Design {results_df.loc[i, 'sample_id']} — "
                                      f"Reward: {results_df.loc[i, 'rewards']:.4f}" if results_df.loc[i, 'rewards'] else
                                      f"Design {results_df.loc[i, 'sample_id']}"
            )

            row = results_df.loc[selected_idx]

            display_pdb = None
            caption_suffix = ""
            if "esmfold_pdb" in results_df.columns and row.get("esmfold_pdb"):
                display_pdb = row["esmfold_pdb"]
                caption_suffix = "ESMFold-validated binder"
            elif row.get("pdb_output"):
                display_pdb = row["pdb_output"]
                caption_suffix = "designed binder (CA-only backbone)"

            target_pdb_stored = st.session_state.get("binder_target_pdb", "")
            binder_pdb = display_pdb

            view_mode = st.radio("View:", ["Binder + Target", "Binder Only"], index=0, horizontal=True, key="binder_view_mode")

            # Viewer inline — below selections, above MLflow button
            if binder_pdb:
                if view_mode == "Binder + Target" and target_pdb_stored:
                    html = molstar_html_multi_pdb([target_pdb_stored, binder_pdb])
                    components.html(html, height=540)
                    st.caption(f"Showing: target + {caption_suffix}")
                else:
                    html = molstar_html_multi_pdb([binder_pdb])
                    components.html(html, height=540)
                    st.caption(f"Showing: {caption_suffix}")
            elif target_pdb_stored:
                html = molstar_html_multi_pdb([target_pdb_stored])
                components.html(html, height=540)
                st.caption("Showing: target protein (no binder structure available — try enabling ESMFold validation)")

            st.markdown(f"**Designed Sequence:** `{row['sequence']}`")

            if "binder_experiment_id" in st.session_state:
                st.button("View MLflow Experiment", key="binder_mlflow_btn",
                          on_click=lambda: open_mlflow_experiment_window(st.session_state["binder_experiment_id"]))

            with st.expander("All designs"):
                display_cols = ["sample_id", "sequence", "rewards"]
                if "esmfold_pdb" in results_df.columns:
                    display_cols.append("esmfold_validated")
                st.dataframe(
                    results_df[[c for c in display_cols if c in results_df.columns]],
                    use_container_width=True, hide_index=True,
                )

    if run_btn:
        if not mlflow_run_name or not mlflow_run_name.strip():
            st.error("MLflow Run Name is required.")
            return

        # Validate input
        if input_mode == "Protein Sequence":
            if not target_sequence or not target_sequence.strip():
                st.error("Protein sequence is required.")
                return
        elif not target_pdb or not target_pdb.strip():
            st.error("Target protein PDB is required.")
            return

        user_info = get_user_info()
        experiment = set_mlflow_experiment(experiment_tag=mlflow_experiment, user_email=user_info.user_email)

        with status_container:
            progress = st.progress(0, text="Preparing..." if input_mode == "Protein Sequence" else "Generating binder designs with Proteina-Complexa...")
            spinner = st.empty()
        with spinner, st.spinner("Running.."):
          with mlflow.start_run(run_name=mlflow_run_name, experiment_id=experiment.experiment_id) as run:
            # Resolve input to PDB if sequence mode
            if input_mode == "Protein Sequence":
                progress.progress(0, text="Folding sequence with ESMFold...")
                try:
                    target_pdb = sequence_to_pdb(target_sequence.strip())
                    mlflow.log_param("input_sequence", target_sequence.strip())
                except Exception as e:
                    st.error(f"ESMFold failed: {e}")
                    return
                progress.progress(20, text="Sequence folded. Generating binder designs with Proteina-Complexa...")

            warnings = []
            mlflow.log_params({"target_chain": target_chain, "hotspot_residues": hotspot_residues,
                               "binder_len_min": binder_len_min, "binder_len_max": binder_len_max,
                               "num_samples": num_samples, "validate_esmfold": validate_esmfold})

            try:
                results_df = hit_proteina_complexa(
                    target_pdb=target_pdb, target_chain=target_chain,
                    hotspot_residues=hotspot_residues,
                    binder_length_min=binder_len_min, binder_length_max=binder_len_max,
                    num_samples=num_samples,
                )
            except Exception as e:
                st.error(f"Binder design failed: {e}")
                return

            if len(results_df) == 0:
                st.error("No designs returned.")
                return

            mlflow.log_dict(results_df.to_dict(), "proteina_complexa_results.json")
            design_done_pct = 50 if input_mode == "PDB" else 60
            progress.progress(design_done_pct, text="Binder designs generated")

            if validate_esmfold:
                esmfold_pdbs = []
                validated = []
                total = len(results_df)
                for idx, (_, row) in enumerate(results_df.iterrows()):
                    pct = design_done_pct + int((idx + 1) / total * (95 - design_done_pct))
                    progress.progress(pct, text=f"Validating design {idx+1}/{total} with ESMFold...")
                    try:
                        pdb = hit_esmfold(row["sequence"])
                        esmfold_pdbs.append(pdb)
                        validated.append(True)
                    except Exception:
                        esmfold_pdbs.append(None)
                        validated.append(False)
                results_df["esmfold_pdb"] = esmfold_pdbs
                results_df["esmfold_validated"] = validated
                esmfold_failed = sum(1 for v in validated if not v)
                mlflow.log_dict({"esmfold_validated": validated}, "esmfold_validation.json")
                if esmfold_failed > 0:
                    warnings.append(f"ESMFold validation failed for {esmfold_failed}/{total} design(s).")

            progress.progress(100, text="Complete")
            st.session_state["binder_results"] = results_df
            st.session_state["binder_target_pdb"] = target_pdb
            st.session_state["binder_experiment_id"] = experiment.experiment_id
            st.session_state["binder_warnings"] = warnings
            mlflow.end_run(status="FINISHED")
        st.rerun()
