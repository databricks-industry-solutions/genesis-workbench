"""Workflow 1: Protein Binder Design using Proteina-Complexa + ESMFold validation."""

import streamlit as st
import streamlit.components.v1 as components
import mlflow
from genesis_workbench.models import set_mlflow_experiment
from utils.streamlit_helper import get_user_info, open_mlflow_experiment_window
from utils.small_molecule_tools import (
    hit_proteina_complexa, hit_esmfold,
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
        with st.form("binder_design_form", enter_to_submit=False):
            target_pdb = st.text_area("Target Protein (PDB):", value=EXAMPLE_PDB, height=250,
                                      help="PDB content of the target protein you want to design a binder for")
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
            mlflow_run_name = st.text_input("Run Name:", key="binder_mlflow_run")
            run_btn = st.form_submit_button("Design Binders", type="primary")

    with viewer_col:
        status_container = st.container()
        viewer_placeholder = st.empty()

        if "binder_results" in st.session_state and st.session_state["binder_results"] is not None:
            results_df = st.session_state["binder_results"]

            selected_idx = st.selectbox(
                "Select design:",
                options=results_df.index,
                key="binder_design_selector",
                format_func=lambda i: f"Design {results_df.loc[i, 'sample_id']} — "
                                      f"Reward: {results_df.loc[i, 'rewards']:.4f}" if results_df.loc[i, 'rewards'] else
                                      f"Design {results_df.loc[i, 'sample_id']}"
            )

            row = results_df.loc[selected_idx]

            # Show the designed structure
            if "esmfold_pdb" in results_df.columns and row.get("esmfold_pdb"):
                with viewer_placeholder:
                    html = molstar_html_multi_pdb([
                        st.session_state.get("binder_target_pdb", ""),
                        row["esmfold_pdb"]
                    ])
                    components.html(html, height=540)
                st.caption("Showing: target (original) + ESMFold-validated binder")
            elif row.get("pdb_output"):
                with viewer_placeholder:
                    html = molstar_html_multi_pdb([
                        st.session_state.get("binder_target_pdb", ""),
                        row["pdb_output"]
                    ])
                    components.html(html, height=540)
                st.caption("Showing: target + designed binder (CA-only backbone)")

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
        if not target_pdb.strip():
            st.error("Target protein PDB is required.")
            return

        user_info = get_user_info()
        experiment = set_mlflow_experiment(experiment_tag=mlflow_experiment, user_email=user_info.user_email,
                                           host=None, token=None, shared=True)

        with status_container:
            progress = st.progress(0, text="Generating binder designs with Proteina-Complexa...")
            spinner = st.empty()
        with spinner, st.spinner("Running.."):
          with mlflow.start_run(run_name=mlflow_run_name, experiment_id=experiment.experiment_id) as run:
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
            progress.progress(50, text="Binder designs generated")

            if validate_esmfold:
                esmfold_pdbs = []
                validated = []
                total = len(results_df)
                for idx, (_, row) in enumerate(results_df.iterrows()):
                    pct = 50 + int((idx + 1) / total * 45)
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
                mlflow.log_dict({"esmfold_validated": validated}, "esmfold_validation.json")

            progress.progress(100, text="Complete")
            st.session_state["binder_results"] = results_df
            st.session_state["binder_target_pdb"] = target_pdb
            st.session_state["binder_experiment_id"] = experiment.experiment_id
            st.rerun()
