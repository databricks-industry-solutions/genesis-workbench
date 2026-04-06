"""Workflow 2: Small-Molecule Ligand Binder Design using Proteina-Complexa-Ligand + DiffDock validation."""

import streamlit as st
import streamlit.components.v1 as components
import mlflow
from genesis_workbench.models import set_mlflow_experiment
from utils.streamlit_helper import get_user_info, open_mlflow_experiment_window
from utils.small_molecule_tools import (
    hit_proteina_complexa_ligand, hit_diffdock,
    molstar_html_pdb, molstar_html_protein_and_sdf,
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
    "Design a protein that binds a specific small molecule using "
    "[Proteina-Complexa-Ligand](https://github.com/NVIDIA-Digital-Bio/Proteina-Complexa). "
    "Validate the designed protein-ligand interaction using DiffDock molecular docking."
)


def render():
    st.markdown("### Ligand Binder Design")
    st.markdown(WORKFLOW_DESCRIPTION)

    input_col, viewer_col = st.columns([1, 2])

    with input_col:
        with st.form("ligand_binder_form", enter_to_submit=False):
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
            validate_diffdock = st.checkbox("Validate with DiffDock", value=True,
                                            help="Run DiffDock on designed proteins to verify ligand binding")

            st.markdown("**MLflow Tracking:**")
            mlflow_experiment = st.text_input("MLflow Experiment:", value="gwb_ligand_binder_design", key="ligand_mlflow_exp")
            mlflow_run_name = st.text_input("Run Name:", key="ligand_mlflow_run")
            run_btn = st.form_submit_button("Design Ligand Binders", type="primary")

    with viewer_col:
        status_container = st.container()
        viewer_placeholder = st.empty()

        if "ligand_binder_results" in st.session_state and st.session_state["ligand_binder_results"] is not None:
            results_df = st.session_state["ligand_binder_results"]

            selected_idx = st.selectbox(
                "Select design:",
                options=results_df.index,
                key="ligand_binder_design_selector",
                format_func=lambda i: f"Design {results_df.loc[i, 'sample_id']} — "
                                      f"Reward: {results_df.loc[i, 'rewards']:.4f}" if results_df.loc[i, 'rewards'] else
                                      f"Design {results_df.loc[i, 'sample_id']}"
            )

            row = results_df.loc[selected_idx]

            # If DiffDock validation was done, show protein + docked ligand
            if "best_dock_sdf" in results_df.columns and row.get("best_dock_sdf"):
                with viewer_placeholder:
                    html = molstar_html_protein_and_sdf(row["pdb_output"], row["best_dock_sdf"])
                    components.html(html, height=540)
                st.caption(f"Showing: designed protein + DiffDock best pose (confidence: {row.get('dock_confidence', 'N/A'):.4f})")
            elif row.get("pdb_output"):
                with viewer_placeholder:
                    html = molstar_html_pdb(row["pdb_output"])
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
        if not ligand_pdb.strip():
            st.error("Ligand PDB (HETATM records) is required.")
            return

        user_info = get_user_info()
        experiment = set_mlflow_experiment(experiment_tag=mlflow_experiment, user_email=user_info.user_email,
                                           host=None, token=None, shared=True)

        with status_container:
            progress = st.progress(0, text="Designing ligand binders with Proteina-Complexa-Ligand...")
            spinner = st.empty()
        with spinner, st.spinner("Running.."):
          with mlflow.start_run(run_name=mlflow_run_name, experiment_id=experiment.experiment_id) as run:
            mlflow.log_params({"ligand_smiles": ligand_smiles, "binder_len_min": binder_len_min,
                               "binder_len_max": binder_len_max, "num_samples": num_samples,
                               "validate_diffdock": validate_diffdock})

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
            progress.progress(50, text="Ligand binder designs generated")

            if validate_diffdock and ligand_smiles.strip():
                dock_sdfs = []
                dock_scores = []
                total = len(results_df)
                for idx, (_, row) in enumerate(results_df.iterrows()):
                    pct = 50 + int((idx + 1) / total * 45)
                    progress.progress(pct, text=f"Validating design {idx+1}/{total} with DiffDock docking...")
                    try:
                        dock_df = hit_diffdock(row["pdb_output"], ligand_smiles, samples_per_complex=5)
                        best = dock_df.sort_values("confidence", ascending=False).iloc[0]
                        dock_sdfs.append(best["ligand_sdf"])
                        dock_scores.append(best["confidence"])
                    except Exception:
                        dock_sdfs.append(None)
                        dock_scores.append(None)
                results_df["best_dock_sdf"] = dock_sdfs
                results_df["dock_confidence"] = dock_scores
                mlflow.log_dict({"dock_confidence": dock_scores}, "diffdock_validation.json")

            progress.progress(100, text="Complete")
            st.session_state["ligand_binder_results"] = results_df
            st.session_state["ligand_binder_experiment_id"] = experiment.experiment_id
            st.rerun()
