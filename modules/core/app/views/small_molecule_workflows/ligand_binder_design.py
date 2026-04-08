"""Workflow 2: Small-Molecule Ligand Binder Design using Proteina-Complexa-Ligand + DiffDock validation."""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import mlflow
from genesis_workbench.models import set_mlflow_experiment
from utils.streamlit_helper import get_user_info, open_mlflow_experiment_window
from utils.small_molecule_tools import (
    hit_proteina_complexa_ligand, hit_diffdock, hit_esmfold, smiles_to_pdb,
    molstar_html_pdb, molstar_html_multi_pdb, molstar_html_protein_and_sdf,
    EXAMPLE_SMILES,
)

# A small ligand PDB (HETATM records) — 4-methoxybenzonitrile positioned near origin
EXAMPLE_LIGAND_PDB = """HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00           C
HETATM    2  C2  LIG A   1       1.394   0.000   0.000  1.00  0.00           C
HETATM    3  C3  LIG A   1       2.091   1.209   0.000  1.00  0.00           C
HETATM    4  C4  LIG A   1       1.394   2.418   0.000  1.00  0.00           C
HETATM    5  C5  LIG A   1       0.000   2.418   0.000  1.00  0.00           C
HETATM    6  C6  LIG A   1      -0.697   1.209   0.000  1.00  0.00           C
HETATM    7  O1  LIG A   1       3.461   1.209   0.000  1.00  0.00           O
HETATM    8  C7  LIG A   1       4.158   2.418   0.000  1.00  0.00           C
HETATM    9  C8  LIG A   1      -2.115   1.209   0.000  1.00  0.00           C
HETATM   10  N1  LIG A   1      -3.277   1.209   0.000  1.00  0.00           N
END
"""

WORKFLOW_DESCRIPTION = (
    "Design a protein that binds a specific small molecule using Proteina-Complexa-Ligand. "
    "Validate the designed protein-ligand interaction using DiffDock molecular docking."
)


def render():
    st.markdown("### Ligand Binder Design")
    st.markdown(WORKFLOW_DESCRIPTION)

    input_col, viewer_col = st.columns([1, 2])

    with input_col:
        input_mode = st.radio("Input type:", ["SMILES", "PDB"], horizontal=True, key="ligand_binder_input_mode")

        with st.form("ligand_binder_form", enter_to_submit=False):
            if input_mode == "SMILES":
                ligand_smiles = st.text_input("Ligand SMILES:", value=EXAMPLE_SMILES,
                                              help="SMILES string for the small molecule — will be converted to PDB with 3D coordinates via Open Babel")
                ligand_pdb = None
            else:
                ligand_pdb = st.text_area("Ligand (PDB with HETATM records):", value=EXAMPLE_LIGAND_PDB, height=250,
                                          help="PDB content containing the small-molecule ligand as HETATM records")
                ligand_smiles = st.text_input("Ligand SMILES (for DiffDock validation):", value=EXAMPLE_SMILES,
                                              help="SMILES string of the same ligand, used for DiffDock docking validation")

            c1, c2 = st.columns(2)
            with c1:
                binder_len_min = st.number_input("Min protein length:", value=50, min_value=20, max_value=200)
            with c2:
                binder_len_max = st.number_input("Max protein length:", value=80, min_value=20, max_value=300)

            num_samples = st.slider("Number of designs:", min_value=1, max_value=10, value=2)
            validate_esmfold = st.checkbox("Validate with ESMFold", value=True,
                                          help="Fold designed sequences with ESMFold to get full-atom structures")
            validate_diffdock = st.checkbox("Validate with DiffDock", value=True,
                                            help="Run DiffDock on designed proteins to verify ligand binding")

            st.markdown("**MLflow Tracking:**")
            mlflow_experiment = st.text_input("MLflow Experiment:", value="gwb_ligand_binder_design", key="ligand_mlflow_exp")
            mlflow_run_name = st.text_input("Run Name:", value="ligand_binder_run", key="ligand_mlflow_run")
            run_btn = st.form_submit_button("Design Ligand Binders", type="primary")

    with viewer_col:
        status_container = st.container()

        if "ligand_binder_results" in st.session_state and st.session_state["ligand_binder_results"] is not None:
            results_df = st.session_state["ligand_binder_results"]

            for w in st.session_state.get("ligand_binder_warnings", []):
                st.warning(w)

            selected_idx = st.selectbox(
                "Select design:",
                options=results_df.index,
                key="ligand_binder_design_selector",
                format_func=lambda i: f"Design {results_df.loc[i, 'sample_id']} — "
                                      f"Reward: {results_df.loc[i, 'rewards']:.4f}" if results_df.loc[i, 'rewards'] else
                                      f"Design {results_df.loc[i, 'sample_id']}"
            )

            row = results_df.loc[selected_idx]

            best_sdf = row.get("best_dock_sdf") if "best_dock_sdf" in results_df.columns else None
            pdb_out = row.get("pdb_output")
            esmfold_pdb = row.get("esmfold_pdb") if "esmfold_pdb" in results_df.columns else None
            dock_conf = row.get("dock_confidence") if "dock_confidence" in results_df.columns else None

            # Structure view selector
            view_options = ["CA Backbone"]
            if esmfold_pdb:
                view_options.append("Full Protein (ESMFold)")
            if best_sdf and pdb_out:
                view_options.append("Protein + Docked Ligand")
            if esmfold_pdb and best_sdf:
                view_options.append("Full Protein + Docked Ligand")

            view_choice = st.radio("View:", view_options, horizontal=True, key="ligand_binder_view_choice")

            # Viewer inline
            if view_choice == "Full Protein + Docked Ligand" and esmfold_pdb and best_sdf:
                html = molstar_html_protein_and_sdf(esmfold_pdb, best_sdf)
                components.html(html, height=540)
                st.caption(f"Showing: ESMFold structure + DiffDock pose (confidence: {dock_conf:.4f})" if dock_conf else "Showing: ESMFold structure + DiffDock pose")
            elif view_choice == "Protein + Docked Ligand" and best_sdf and pdb_out:
                html = molstar_html_protein_and_sdf(pdb_out, best_sdf)
                components.html(html, height=540)
                st.caption(f"Showing: CA backbone + DiffDock pose (confidence: {dock_conf:.4f})" if dock_conf else "Showing: CA backbone + DiffDock pose")
            elif view_choice == "Full Protein (ESMFold)" and esmfold_pdb:
                html = molstar_html_multi_pdb([esmfold_pdb])
                components.html(html, height=540)
                st.caption("Showing: ESMFold-validated full protein structure")
            elif pdb_out:
                html = molstar_html_multi_pdb([pdb_out])
                components.html(html, height=540)
                st.caption("Showing: designed protein (CA-only backbone)")

            st.markdown(f"**Designed Sequence:** `{row['sequence']}`")

            if "ligand_binder_experiment_id" in st.session_state:
                st.button("View MLflow Experiment", key="ligand_binder_mlflow_btn",
                          on_click=lambda: open_mlflow_experiment_window(st.session_state["ligand_binder_experiment_id"]))

            with st.expander("All designs"):
                display_cols = ["sample_id", "sequence", "rewards"]
                if "dock_confidence" in results_df.columns:
                    display_cols.append("dock_confidence")
                st.dataframe(
                    results_df[[c for c in display_cols if c in results_df.columns]],
                    use_container_width=True, hide_index=True,
                )

    if run_btn:
        if not mlflow_run_name or not mlflow_run_name.strip():
            st.error("MLflow Run Name is required.")
            return

        # Validate input
        if input_mode == "SMILES":
            if not ligand_smiles or not ligand_smiles.strip():
                st.error("Ligand SMILES is required.")
                return
        elif not ligand_pdb or not ligand_pdb.strip():
            st.error("Ligand PDB (HETATM records) is required.")
            return

        user_info = get_user_info()
        experiment = set_mlflow_experiment(experiment_tag=mlflow_experiment, user_email=user_info.user_email)

        with status_container:
            progress = st.progress(0, text="Preparing..." if input_mode == "SMILES" else "Designing ligand binders with Proteina-Complexa-Ligand...")
            spinner = st.empty()
        with spinner, st.spinner("Running.."):
          with mlflow.start_run(run_name=mlflow_run_name, experiment_id=experiment.experiment_id) as run:
            # Resolve SMILES to PDB if needed
            if input_mode == "SMILES":
                progress.progress(0, text="Converting SMILES to PDB via Open Babel...")
                try:
                    ligand_pdb = smiles_to_pdb(ligand_smiles.strip())
                    mlflow.log_param("input_smiles", ligand_smiles.strip())
                except Exception as e:
                    st.error(f"Open Babel conversion failed: {e}")
                    return
                progress.progress(10, text="SMILES converted. Designing ligand binders with Proteina-Complexa-Ligand...")

            mlflow.log_params({"ligand_smiles": ligand_smiles if ligand_smiles else "", "binder_len_min": binder_len_min,
                               "binder_len_max": binder_len_max, "num_samples": num_samples,
                               "validate_esmfold": validate_esmfold, "validate_diffdock": validate_diffdock})

            # Calculate progress segments
            total_steps = 1 + (1 if validate_esmfold else 0) + (1 if validate_diffdock else 0)
            step = 0
            warnings = []

            try:
                results_df = hit_proteina_complexa_ligand(
                    target_pdb=ligand_pdb,
                    binder_length_min=binder_len_min, binder_length_max=binder_len_max,
                    num_samples=num_samples,
                )
            except Exception as e:
                st.error(f"Ligand binder design failed: {e}")
                return

            if len(results_df) == 0:
                st.error("No designs returned.")
                return

            mlflow.log_dict(results_df.to_dict(), "proteina_complexa_ligand_results.json")
            step += 1
            progress.progress(int(step / total_steps * 100), text="Ligand binder designs generated")

            if validate_esmfold:
                esmfold_pdbs = []
                total = len(results_df)
                for idx, (_, row) in enumerate(results_df.iterrows()):
                    base_pct = int(step / total_steps * 100)
                    next_pct = int((step + 1) / total_steps * 100)
                    pct = base_pct + int((idx + 1) / total * (next_pct - base_pct))
                    progress.progress(pct, text=f"Folding design {idx+1}/{total} with ESMFold...")
                    try:
                        esmfold_pdbs.append(hit_esmfold(row["sequence"]))
                    except Exception:
                        esmfold_pdbs.append(None)
                results_df["esmfold_pdb"] = esmfold_pdbs
                esmfold_failed = sum(1 for p in esmfold_pdbs if p is None)
                mlflow.log_dict({"esmfold_success": total - esmfold_failed, "esmfold_failed": esmfold_failed}, "esmfold_validation.json")
                if esmfold_failed > 0:
                    warnings.append(f"ESMFold validation failed for {esmfold_failed}/{total} design(s). These will use CA backbone instead.")
                step += 1

            if validate_diffdock and ligand_smiles and ligand_smiles.strip():
                dock_sdfs = []
                dock_scores = []
                total = len(results_df)
                for idx, (_, row) in enumerate(results_df.iterrows()):
                    base_pct = int(step / total_steps * 100)
                    next_pct = int((step + 1) / total_steps * 100)
                    pct = base_pct + int((idx + 1) / total * (next_pct - base_pct))
                    progress.progress(pct, text=f"Docking design {idx+1}/{total} with DiffDock...")
                    try:
                        dock_pdb = row.get("esmfold_pdb") or row["pdb_output"]
                        dock_df = hit_diffdock(dock_pdb, ligand_smiles, samples_per_complex=5)
                        best = dock_df.sort_values("confidence", ascending=False).iloc[0]
                        dock_sdfs.append(best["ligand_sdf"])
                        dock_scores.append(best["confidence"])
                    except Exception:
                        dock_sdfs.append(None)
                        dock_scores.append(None)
                results_df["best_dock_sdf"] = dock_sdfs
                results_df["dock_confidence"] = dock_scores
                dock_failed = sum(1 for s in dock_sdfs if s is None)
                if dock_failed > 0:
                    warnings.append(f"DiffDock validation failed for {dock_failed}/{total} design(s). Docking results will be unavailable for these.")
                mlflow.log_dict({"dock_confidence": dock_scores}, "diffdock_validation.json")

            progress.progress(100, text="Complete")
            st.session_state["ligand_binder_results"] = results_df
            st.session_state["ligand_binder_experiment_id"] = experiment.experiment_id
            st.session_state["ligand_binder_warnings"] = warnings
            mlflow.end_run(status="FINISHED")
        st.rerun()
